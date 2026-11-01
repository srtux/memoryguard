"""ADK agent package running with unprotected baseline memory.

When you run `adk web agents`, Google ADK discovers `root_agent` from this package
so you can interactively test how an unprotected persistent-memory agent behaves.
"""

from __future__ import annotations

# Re-export `root_agent` (for Google ADK) and `runtime` (for inspecting logs/state).
from .agent import root_agent, runtime

__all__ = ["root_agent", "runtime"]
