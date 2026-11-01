"""MemoryGuard protected ADK agent definition for comparative evaluation.

This agent runs in `"with_memoryguard"` mode, combining all three defense stages:
  1. `LayaAdmissionGate`: Local 421M CPU write gate (rejects chatter, vetoes blatant overrides).
  2. `MemoryStore`: Provenance-tiered SQLite + FTS5 store (`USER = 3`, `REPO = 2`, `EXTERNAL = 1`).
  3. `ToolInterceptor(mode="clean_room")`: Action-time Clean-Room non-interference check and recovery.
"""

from __future__ import annotations

from pathlib import Path

from memoryguard.adk_middleware import build_demo_agent
from memoryguard.memory_store import MemoryStore

# Store persistent memories on disk under `<repo_root>/.cache/adk_memoryguard.sqlite`
# so verified user rules survive across multiple sessions in the `adk web agents` UI.
_db_path = Path(__file__).resolve().parent.parent.parent / ".cache" / "adk_memoryguard.sqlite"
_db_path.parent.mkdir(parents=True, exist_ok=True)
_store = MemoryStore(db_path=str(_db_path))

# Build the MemoryGuard-protected ADK agent (`root_agent`) and its `runtime` state.
root_agent, runtime = build_demo_agent("with_memoryguard", store=_store)
