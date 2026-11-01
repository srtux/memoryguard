"""Unprotected Memory ADK agent definition for comparative evaluation."""

from __future__ import annotations

from pathlib import Path

from memoryguard.adk_middleware import build_demo_agent
from memoryguard.memory_store import MemoryStore

_db_path = Path(__file__).resolve().parent.parent.parent / ".cache" / "adk_unprotected.sqlite"
_db_path.parent.mkdir(parents=True, exist_ok=True)
_store = MemoryStore(db_path=str(_db_path))

root_agent, runtime = build_demo_agent("unprotected_memory", store=_store)
