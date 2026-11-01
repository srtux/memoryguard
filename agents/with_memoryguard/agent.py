"""MemoryGuard protected ADK agent definition for comparative evaluation."""

from __future__ import annotations

from pathlib import Path

from memoryguard.adk import build_demo_agent
from memoryguard.store import MemoryStore

_db_path = Path(__file__).resolve().parent.parent.parent / ".cache" / "adk_memoryguard.sqlite"
_db_path.parent.mkdir(parents=True, exist_ok=True)
_store = MemoryStore(db_path=str(_db_path))

root_agent, runtime = build_demo_agent("with_memoryguard", store=_store)
