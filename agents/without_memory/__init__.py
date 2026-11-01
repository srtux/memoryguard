"""ADK agent package running without persistent memory.

When you run `adk web agents`, Google ADK scans each subfolder inside `agents/`
and imports `root_agent` from its `__init__.py` file to display it in the web UI.
"""

from __future__ import annotations

# Re-export `root_agent` (for Google ADK) and `runtime` (for inspecting logs/state).
from .agent import root_agent, runtime

__all__ = ["root_agent", "runtime"]
