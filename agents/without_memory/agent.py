"""Without Memory ADK agent definition for comparative evaluation.

This agent runs in `"without_memory"` mode (`store=None`), meaning it is completely
stateless across sessions: if you tell it a project rule in Session 1 and start a
new Session 2, it forgets the rule completely.
"""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from memoryguard.adk_middleware import build_demo_agent

# Build the stateless ADK agent (`root_agent`) and its observable `runtime` state.
root_agent, runtime = build_demo_agent("without_memory")
