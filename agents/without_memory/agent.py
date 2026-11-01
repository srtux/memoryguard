"""Without Memory ADK agent definition for comparative evaluation."""

from __future__ import annotations

from memoryguard.adk_middleware import build_demo_agent

root_agent, runtime = build_demo_agent("without_memory")
