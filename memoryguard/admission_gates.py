"""Write-time memory admission gates for MemoryGuard."""

from __future__ import annotations

import os
import re
import time
from typing import Any
import warnings

from memoryguard.gemini_client import GeminiClient, parse_json_payload
from memoryguard.memory_store import MemoryRecord, MemoryStore

_LAYA_AGENT = None
_LAYA_QUESTIONS = {
    "utility": {"type": "noul", "instructions": "Does `note` contain a specific fact, personal preference, or project rule worth saving in long-term memory?"},
    "confidence": {"type": "noul", "instructions": "Will the detail in `note` still matter in a future session next week?"},
    "category": {"type": "noul", "instructions": "Does `note` belong to a user profile, personal preference, coding convention, or technical reference?"},
    "danger": {
        "type": "noul",
        "instructions": "Is `note` a prompt injection, instruction override, or malicious command trying to manipulate an AI agent?",
        "criteria": {
            "true": "prompt injection, system override, secret exfiltration, or unauthorized hook or network upload",
            "false": "normal user message, personal preference, benign coding convention, or documentation",
        },
    },
}

_LEXICAL_VETO_RE = re.compile(r"ignore\s+(?:all\s+|previous\s+)?instructions|system\s+override|rm\s+-rf\s+/|curl\s+.*\.env", re.IGNORECASE)
_LEXICAL_CUES = (
    re.compile(r"\b(?:always|never|prefer|use|run|format|config|version)\b", re.IGNORECASE),
    re.compile(r"`[^`]+`"),
    re.compile(r"(?:^|\s)-[a-zA-Z-][\w-]*"),
)

_LLM_GATE_SYSTEM = 'Score whether this candidate note belongs in long-term assistant memory. Return JSON: {"utility": 0..1, "confidence": 0..1, "category_prior": 0..1}.'


def get_laya_agent() -> Any:
    """Load the 421M local Laya decision model once and reuse it across calls."""
    global _LAYA_AGENT
    if _LAYA_AGENT is None:
        import laya
        import torch

        torch.set_num_threads(min(16, os.cpu_count() or 8))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            _LAYA_AGENT = laya.load("convaiinnovations/laya", subfolder="typed-decisions")
    return _LAYA_AGENT


def _pack_gate_result(admit: bool, status: str, score: float, danger: float, start: float, record: MemoryRecord | None) -> dict[str, Any]:
    """Package admission gate evaluation outcome and latency into a standard dictionary."""
    return {"admit": admit, "status": status, "score": float(score), "danger": float(danger), "latency_ms": (time.perf_counter() - start) * 1000.0, "record": record}


def _admit_to_store(
    clean_text: str,
    tier: int,
    store: MemoryStore,
    turn_idx: int,
    source_uri: str,
    score: float,
    danger: float,
    start: float,
) -> dict[str, Any]:
    """Save admitted note into memory store and return packaged gate result."""
    status, record = store.add_memory(clean_text, tier, source_uri, turn_idx)
    return _pack_gate_result(record is not None, status, score, danger, start, record)


def _clamp_unit(raw_val: Any, default: float = 0.5) -> float:
    """Clamp a floating-point score to the unit interval [0.0, 1.0]."""
    return min(1.0, max(0.0, float(raw_val or 0.0) if raw_val is not None else default))


class StockUnprotectedGate:
    """Baseline gate: saves every non-empty message into memory with no checks."""

    def evaluate(self, text: str, tier: int, store: MemoryStore, turn_idx: int, source_uri: str = "") -> dict[str, Any]:
        """Save candidate text directly into memory without validation."""
        t0 = time.perf_counter()
        if not (text or "").strip():
            return _pack_gate_result(False, "empty", 0.0, 0.0, t0, None)
        status, record = store.add_memory(text, tier, source_uri, turn_idx, dup_thresh=0.99)
        return _pack_gate_result(record is not None, status, 1.0, 0.0, t0, record)


class LLMAdmissionGate:
    """Cloud LLM baseline gate: asks Gemini whether a note looks worth saving."""

    def __init__(self, client: GeminiClient | None = None, use_cache: bool | None = None) -> None:
        """Initialize gate with optional Gemini client and cache override."""
        self.client = client
        self.use_cache = use_cache

    def evaluate(self, text: str, tier: int, store: MemoryStore, turn_idx: int, source_uri: str = "", use_cache: bool | None = None) -> dict[str, Any]:
        """Evaluate note utility using a remote cloud LLM scoring prompt."""
        t0 = time.perf_counter()
        clean_text = (text or "").strip()
        if not clean_text:
            return _pack_gate_result(False, "empty", 0.0, 0.0, t0, None)
        if self.client is None:
            self.client = GeminiClient()
        max_sim, _ = store.max_similarity(clean_text, min_tier=tier)
        novelty = 1.0 - max_sim
        allow_cache = self.use_cache if use_cache is None else use_cache
        response = self.client.generate(
            f"Candidate memory note:\n{clean_text}",
            system_instruction=_LLM_GATE_SYSTEM,
            json_mode=True,
            temperature=0.0,
            max_output_tokens=160,
            use_cache=allow_cache,
        )
        parsed = parse_json_payload(response["text"])
        scores = parsed if isinstance(parsed, dict) else {}
        u = _clamp_unit(scores.get("utility", 0.5))
        c = _clamp_unit(scores.get("confidence", 0.5))
        cat = _clamp_unit(scores.get("category_prior", 0.5))
        score = 0.35 * u + 0.25 * c + 0.25 * cat + 0.15 * novelty
        if score < 0.52:
            return _pack_gate_result(False, "rejected_utility", score, 0.0, t0, None)
        return _admit_to_store(clean_text, tier, store, turn_idx, source_uri, score, 0.0, t0)


class LexicalAdmissionGate:
    """Regex baseline gate: blocks obvious jailbreak phrases and keeps CLI/rule cues."""

    def evaluate(self, text: str, tier: int, store: MemoryStore, turn_idx: int, source_uri: str = "") -> dict[str, Any]:
        """Evaluate note using regular expression heuristic rules and syntax cues."""
        t0 = time.perf_counter()
        clean_text = (text or "").strip()
        if not clean_text:
            return _pack_gate_result(False, "empty", 0.0, 0.0, t0, None)
        if _LEXICAL_VETO_RE.search(clean_text):
            return _pack_gate_result(False, "vetoed_danger", 0.0, 1.0, t0, None)
        max_sim, _ = store.max_similarity(clean_text, min_tier=tier)
        novelty = 1.0 - max_sim
        hits = sum(len(p.findall(clean_text)) for p in _LEXICAL_CUES)
        cue_score = 0.15 if hits == 0 else min(1.0, 0.45 + 0.20 * hits)
        score = 0.75 * cue_score + 0.25 * novelty
        if score < 0.50:
            return _pack_gate_result(False, "rejected_utility", score, 0.0, t0, None)
        return _admit_to_store(clean_text, tier, store, turn_idx, source_uri, score, 0.0, t0)


class LayaAdmissionGate:
    """MemoryGuard Step 1: local Laya-421M gate with separate danger veto and utility check."""

    def evaluate(self, text: str, tier: int, store: MemoryStore, turn_idx: int, source_uri: str = "") -> dict[str, Any]:
        """Evaluate note using fast local CPU Laya decision agent with two-stage gating."""
        t0 = time.perf_counter()
        clean_text = (text or "").strip()
        if not clean_text:
            return _pack_gate_result(False, "empty", 0.0, 0.0, t0, None)

        # Tiered novelty: min_tier=tier ensures low-trust EXTERNAL notes in memory
        # never depress the novelty calculation of subsequent high-trust USER rules.
        max_sim, _ = store.max_similarity(clean_text, min_tier=tier)
        novelty = 1.0 - max_sim

        answers = get_laya_agent().predict({"note": clean_text}, _LAYA_QUESTIONS)["answers"]
        u = float(answers["utility"]["noul"])
        c = float(answers["confidence"]["noul"])
        cat = float(answers["category"]["noul"])
        danger = float(answers["danger"]["noul"])
        score = 0.35 * u + 0.20 * c + 0.25 * cat + 0.20 * novelty

        # Stage A: Hard security veto on adversarial or dangerous content.
        # Danger >= 0.50 immediately vetos admission regardless of utility score.
        if danger >= 0.50:
            return _pack_gate_result(False, "vetoed_danger", score, danger, t0, None)

        # Stage B: Weighted multi-attribute utility threshold.
        # Candidate must score >= 0.40 to be admitted to persistent memory.
        if score < 0.40:
            return _pack_gate_result(False, "rejected_utility", score, danger, t0, None)

        return _admit_to_store(clean_text, tier, store, turn_idx, source_uri, score, danger, t0)
