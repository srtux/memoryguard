"""MemoryGuard: Write-time memory gating and action-time Clean-Room guard for AI agents.

This file (__init__.py) marks the memoryguard folder as a Python package and
re-exports the main classes, constants, and helper functions from each submodule.
That way, other scripts can simply write:
    from memoryguard import MemoryStore, LocalCPUGate, ToolInterceptor
instead of importing from each internal file separately.
"""

from __future__ import annotations

# 1. Google ADK (Agent Development Kit) middleware for live multi-session agent demos.
from memoryguard.adk_middleware import (
    DEMO_MODES,
    DemoRuntime,
    build_demo_agent,
    create_adk_model,
)

# 2. Write-time admission gates: decide which notes are safe and useful to save.
from memoryguard.admission_gates import (
    CloudAIGate,
    KeywordRuleGate,
    LocalCPUGate,
    SaveEverythingGate,
    get_local_cpu_model,
)

# 3. Gemini AI client wrapper, .env configuration loader, and SQLite response cache.
from memoryguard.gemini_client import (
    DEFAULT_MODEL,
    GeminiClient,
    has_live_credentials,
    load_api_key,
    load_dotenv_file,
    parse_json_payload,
    resolve_model,
)

# 4. Source-tiered SQLite and FTS5 memory store with trust levels.
from memoryguard.memory_store import (
    TIER_EXTERNAL,
    TIER_INTERNAL,
    TIER_NAMES,
    TIER_USER,
    MemoryRecord,
    MemoryStore,
    note_similarity,
)

# 5. Action-time Clean-Room tool interceptor: verifies tool calls before execution.
from memoryguard.tool_interceptor import (
    READ_ONLY_TOOLS,
    SENSITIVE_TOOLS,
    ToolInterceptor,
    calls_are_equivalent,
)

# __all__ explicitly lists the public symbols exported when importing memoryguard.
__all__ = [
    "CloudAIGate",
    "DEFAULT_MODEL",
    "DEMO_MODES",
    "DemoRuntime",
    "GeminiClient",
    "KeywordRuleGate",
    "LocalCPUGate",
    "MemoryRecord",
    "MemoryStore",
    "READ_ONLY_TOOLS",
    "SENSITIVE_TOOLS",
    "SaveEverythingGate",
    "TIER_EXTERNAL",
    "TIER_INTERNAL",
    "TIER_NAMES",
    "TIER_USER",
    "ToolInterceptor",
    "build_demo_agent",
    "calls_are_equivalent",
    "create_adk_model",
    "get_local_cpu_model",
    "has_live_credentials",
    "load_api_key",
    "load_dotenv_file",
    "note_similarity",
    "parse_json_payload",
    "resolve_model",
]
