"""MemoryGuard: Write-time admission gating and action-time Clean-Room guard for AI agents."""

from __future__ import annotations

from memoryguard.adk_middleware import (
    DEMO_MODES,
    DemoRuntime,
    build_demo_agent,
    create_adk_model,
)
from memoryguard.admission_gates import (
    LayaAdmissionGate,
    LexicalAdmissionGate,
    LLMAdmissionGate,
    StockUnprotectedGate,
    get_laya_agent,
)
from memoryguard.gemini_client import (
    DEFAULT_MODEL,
    GeminiClient,
    has_live_credentials,
    load_api_key,
    load_dotenv_file,
    parse_json_payload,
    resolve_model,
)
from memoryguard.memory_store import (
    TIER_EXTERNAL,
    TIER_NAMES,
    TIER_REPO,
    TIER_USER,
    MemoryRecord,
    MemoryStore,
    note_similarity,
)
from memoryguard.tool_interceptor import (
    ToolInterceptor,
    calls_are_equivalent,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEMO_MODES",
    "DemoRuntime",
    "GeminiClient",
    "LayaAdmissionGate",
    "LexicalAdmissionGate",
    "LLMAdmissionGate",
    "MemoryRecord",
    "MemoryStore",
    "StockUnprotectedGate",
    "TIER_EXTERNAL",
    "TIER_NAMES",
    "TIER_REPO",
    "TIER_USER",
    "ToolInterceptor",
    "build_demo_agent",
    "calls_are_equivalent",
    "create_adk_model",
    "get_laya_agent",
    "has_live_credentials",
    "load_api_key",
    "load_dotenv_file",
    "note_similarity",
    "parse_json_payload",
    "resolve_model",
]
