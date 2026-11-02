"""Unprotected Memory ADK agent definition for comparative evaluation.

This agent runs in "unprotected_memory" mode using SaveEverythingGate() and
ToolInterceptor(mode="none"). It saves every user turn and external GitHub issue
into a persistent SQLite file (.cache/adk_unprotected.sqlite) without write-time
filtering or action-time Clean-Room verification.
"""

from __future__ import annotations

from pathlib import Path

from memoryguard.adk_middleware import build_demo_agent
from memoryguard.memory_store import MemoryStore

# Store persistent memories on disk under `<repo_root>/.cache/adk_unprotected.sqlite`
# so notes survive across multiple sessions in the `adk web agents` UI.
_db_path = Path(__file__).resolve().parent.parent.parent / ".cache" / "adk_unprotected.sqlite"
_db_path.parent.mkdir(parents=True, exist_ok=True)
_store = MemoryStore(db_path=str(_db_path))

# Build the unprotected baseline ADK agent (`root_agent`) and its `runtime` state.
root_agent, runtime = build_demo_agent("unprotected_memory", store=_store)
