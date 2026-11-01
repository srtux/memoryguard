"""Gemini client using google-genai SDK, .env credentials, and SQLite cache.

This module wraps the official `google-genai` SDK so the rest of MemoryGuard can:
  1. Load API keys or Google Cloud project settings automatically from a `.env` file.
  2. Ignore dummy template placeholders (like `"your_gemini_api_key_here"` in `.env.example`).
  3. Automatically retry temporary network/rate-limit errors (`429`, `503`) with exponential backoff.
  4. Cache LLM responses in a local SQLite database (`.cache/llm_cache.sqlite`) so
     re-running the benchmark is instant, deterministic, and works offline.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any

from google import genai
from google.genai import errors as genai_errors, types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

# Default model used across all 5 benchmark architectures for controlled comparison.
DEFAULT_MODEL = "gemini-3.5-flash-lite"

# Known dummy values from `.env.example`. If a user copies `.env.example` to `.env`
# without replacing these strings, `_clean()` treats them as empty rather than sending
# an invalid key to the API.
_PLACEHOLDERS = frozenset({
    "your_gemini_api_key_here", "your_api_key_here", "your-api-key", "your-gcp-project-id", "change_me", "changeme",
})


def load_dotenv_file(dotenv_path: str | os.PathLike[str] | None = None, *, override: bool = False) -> dict[str, str]:
    """Load KEY=VALUE pairs from .env into os.environ without external deps."""
    # Step 1: Decide which `.env` file path to read.
    if dotenv_path is not None:
        p = Path(dotenv_path)
    else:
        # Allow tests to disable `.env` loading by setting `MEMORY_GUARD_DOTENV_PATH=""`.
        env_override = os.environ.get("MEMORY_GUARD_DOTENV_PATH")
        if env_override == "":
            return {}
        # Default to `<repo_root>/.env` (one folder above the `memoryguard/` package).
        p = Path(env_override) if env_override is not None else Path(__file__).resolve().parent.parent / ".env"

    # If no `.env` file exists on disk, return an empty dict without crashing.
    if not p.is_file():
        return {}

    # Step 2: Parse each `KEY=VALUE` line in the `.env` file.
    loaded: dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        # Skip blank lines, `#` comments, and lines without an `=` sign.
        if not line or line.startswith("#") or "=" not in line:
            continue
        # Support shell-style `export KEY=VALUE` lines by stripping the `export ` prefix.
        if line.startswith("export "):
            line = line[7:].strip()
        # Split on the first `=` into key `k` and value `v`.
        k, v = [x.strip() for x in line.split("=", 1)]
        # Strip matching single or double quotes around the value (e.g., `"my-key"` -> `my-key`).
        v = v[1:-1] if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"') else v
        if k:
            loaded[k] = v
            # Only set `os.environ[k]` if `override=True` or if the variable is not already set
            # in the user's shell environment.
            if override or k not in os.environ:
                os.environ[k] = v
    return loaded


def _clean(val: str | None) -> str:
    """Normalize environment string, returning empty if matching dummy placeholder templates."""
    cleaned = (val or "").strip()
    # Placeholder filtering: treat unedited example values as unset to avoid invalid API requests.
    return "" if cleaned.lower() in _PLACEHOLDERS else cleaned


def resolve_model(model: str | None = None) -> str:
    """Resolve model ID from parameter, GEMINI_MODEL env var, or default model."""
    load_dotenv_file()
    # Priority order: explicit `model` argument -> `GEMINI_MODEL` in `.env`/shell -> `DEFAULT_MODEL`.
    return _clean(model) or _clean(os.environ.get("GEMINI_MODEL")) or DEFAULT_MODEL


def load_api_key(required: bool = True) -> str:
    """Retrieve Gemini developer API key or verify Google Cloud project credentials."""
    load_dotenv_file()
    # Check both `GEMINI_API_KEY` and `GOOGLE_API_KEY` environment variables.
    k = _clean(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    # If `required=True` and neither an API key nor a Vertex AI project ID is configured,
    # raise a clear error message telling the user how to configure `.env`.
    if required and not k and not _clean(os.environ.get("GOOGLE_CLOUD_PROJECT")):
        raise RuntimeError("Missing Gemini credentials: set GEMINI_API_KEY in .env or GOOGLE_CLOUD_PROJECT.")
    return k


def has_live_credentials() -> bool:
    """Check whether valid non-placeholder API key or Vertex AI project credentials exist."""
    load_dotenv_file()
    # Returns True if the user has configured either a Gemini API key or a Google Cloud Project.
    return bool(load_api_key(required=False) or _clean(os.environ.get("GOOGLE_CLOUD_PROJECT")))


def parse_json_payload(raw: str) -> Any:
    """Parse JSON payload from LLM text, extracting code fences or bracketed objects."""
    # Step 1: Strip Markdown code fences like ```json ... ``` if the model wrapped its output.
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip(), flags=re.DOTALL).strip()
    candidates = [cleaned]

    # Step 2: Also extract the first `{...}` JSON object or `[...]` JSON array substring
    # in case the model added conversational text before or after the JSON.
    for pat in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pat, cleaned, re.DOTALL)
        if m:
            candidates.append(m.group(0))

    # Step 3: Try parsing each candidate string with `json.loads` and return the first valid JSON.
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return {}


def _is_transient(exc: BaseException) -> bool:
    """Check whether an exception represents a retryable rate-limit or network timeout."""
    # HTTP 429 = Rate Limit Exceeded; HTTP 500/502/503/504 = Temporary Server Error.
    if isinstance(exc, genai_errors.APIError):
        return exc.code in (429, 500, 502, 503, 504)
    # Also retry on standard Python network timeout and connection errors.
    return isinstance(exc, (TimeoutError, ConnectionError, OSError))


class GeminiClient:
    """google-genai client wrapper with SQLite prompt cache."""

    def __init__(self, model: str | None = None, cache_path: str | None = None, use_cache: bool = True, api_key: str | None = None) -> None:
        """Initialize client with API credentials and persistent SQLite cache storage."""
        load_dotenv_file()
        # If `use_cache=True`, we don't require an API key upfront because cached responses
        # can be served directly from SQLite without touching the network.
        self.api_key = _clean(api_key) or load_api_key(required=not use_cache)
        self.project = _clean(os.environ.get("GOOGLE_CLOUD_PROJECT"))
        self.location = _clean(os.environ.get("GOOGLE_CLOUD_LOCATION")) or "global"
        self.model = resolve_model(model)
        self.use_cache = use_cache
        self._sdk_client: genai.Client | None = None
        # Counters tracking how many prompts hit the live API vs. the local SQLite cache.
        self.live_calls, self.cache_hits = 0, 0

        # Default cache file location: `<repo_root>/.cache/llm_cache.sqlite`.
        repo_dir = Path(__file__).resolve().parent.parent
        self.cache_path = cache_path or str(repo_dir / ".cache" / "llm_cache.sqlite")
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        # Create the SQLite `cache` table and enable WAL (Write-Ahead Logging) mode
        # so concurrent readers/writers never lock the database file.
        with sqlite3.connect(self.cache_path, timeout=30.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    def _get_sdk_client(self) -> genai.Client:
        """Lazily initialize and return the underlying google-genai SDK Client."""
        if self._sdk_client is None:
            if self.api_key:
                # Option A: Authenticate using a Gemini Developer API key (`GEMINI_API_KEY`).
                self._sdk_client = genai.Client(api_key=self.api_key)
            elif self.project:
                # Option B: Authenticate using Google Cloud Vertex AI (`GOOGLE_CLOUD_PROJECT`).
                self._sdk_client = genai.Client(vertexai=True, project=self.project, location=self.location)
            else:
                raise RuntimeError("Missing Gemini credentials: set GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT.")
        return self._sdk_client

    def _make_cache_key(self, target_model: str, system_instruction: str, prompt: str, json_mode: bool, temperature: float, max_output_tokens: int, cache_salt: str) -> str:
        """Compute SHA-256 digest over normalized request parameters for cache indexing."""
        # Pack every parameter that affects the LLM's output into a dictionary and hash it
        # with SHA-256 (`sort_keys=True` ensures the JSON key order is always identical).
        envelope = {"v": "memory_guard_v1", "m": target_model, "s": system_instruction, "p": prompt, "j": json_mode, "t": temperature, "k": max_output_tokens, "x": cache_salt}
        return hashlib.sha256(json.dumps(envelope, sort_keys=True).encode("utf-8")).hexdigest()

    # `@retry` automatically retries `_call_model` up to 5 times with exponential backoff
    # (waiting 1.5s, 2.25s, 3.37s... up to 30s) whenever `_is_transient` returns True.
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1.5, min=1.5, max=30.0), retry=retry_if_exception(_is_transient), reraise=True)
    def _call_model(self, target_model: str, prompt: str, config: types.GenerateContentConfig) -> types.GenerateContentResponse:
        """Send generate_content request to Gemini API with exponential backoff on transient errors."""
        return self._get_sdk_client().models.generate_content(model=target_model, contents=prompt, config=config)

    def generate(self, prompt: str, *, system_instruction: str = "", json_mode: bool = False, temperature: float = 0.0, max_output_tokens: int = 1024, model: str | None = None, cache_salt: str = "", use_cache: bool | None = None) -> dict[str, Any]:
        """Generate model completion, checking and populating local SQLite cache."""
        target_model = model or self.model
        allow_cache = self.use_cache if use_cache is None else use_cache
        key = self._make_cache_key(target_model, system_instruction, prompt, json_mode, temperature, max_output_tokens, cache_salt)

        # Step 1: Check if we already have a cached response for this exact prompt in SQLite.
        if allow_cache:
            with sqlite3.connect(self.cache_path, timeout=30.0) as conn:
                row = conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
                if row is not None:
                    self.cache_hits += 1
                    return json.loads(row[0])

        # Step 2: Cache miss (or `use_cache=False`) -> call the live Gemini model.
        config = types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_output_tokens, system_instruction=system_instruction or None, response_mime_type="application/json" if json_mode else None)
        response = self._call_model(target_model, prompt, config)
        usage = response.usage_metadata
        result = {"text": (response.text or "").strip(), "prompt_tokens": int((usage.prompt_token_count if usage else 0) or 0), "output_tokens": int((usage.candidates_token_count if usage else 0) or 0)}
        finish = str(response.candidates[0].finish_reason) if response.candidates and response.candidates[0].finish_reason else ""
        self.live_calls += 1

        # Step 3: Save the response into the SQLite cache (skipping truncated `MAX_TOKENS` outputs).
        if allow_cache and result["text"] and "MAX_TOKENS" not in finish:
            with sqlite3.connect(self.cache_path, timeout=30.0) as conn:
                conn.execute("INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, json.dumps(result, sort_keys=True)))
        return result
