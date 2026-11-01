"""Action-time Clean-Room tool interceptor and non-interference guard for MemoryGuard."""

from __future__ import annotations

import re
from typing import Any, Callable

from memoryguard.memory_store import TIER_EXTERNAL, TIER_REPO, MemoryRecord

PRIVILEGED_TOOLS = {"run_bash", "write_file", "http_post"}
SAFE_TOOLS = {"read_file", "search_docs", "answer_only"}
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def calls_are_equivalent(call_a: dict[str, Any], call_b: dict[str, Any]) -> bool:
    """Return True if two tool calls perform the same logical action."""
    tool_a = " ".join(str(call_a.get("tool") or "").split())
    tool_b = " ".join(str(call_b.get("tool") or "").split())
    if not tool_a or tool_a != tool_b:
        return False
    # Equivalence check: for read-only answers, compare the set of referenced URLs
    # to tolerate superficial formatting differences; for other tools, compare normalized arguments.
    if tool_a == "answer_only":
        return set(_URL_RE.findall(str(call_a.get("answer") or ""))) == set(_URL_RE.findall(str(call_b.get("answer") or "")))
    return " ".join(str(call_a.get("args") or "").split()) == " ".join(str(call_b.get("args") or "").split())


def _needs_clean_room(call: dict[str, Any]) -> bool:
    """Determine whether an action touches one of four guarded security surfaces."""
    tool_name = " ".join(str(call.get("tool") or "").split())
    # Surface 1: Privileged side-effecting execution (shell, file write, outbound HTTP POST).
    if tool_name in PRIVILEGED_TOOLS:
        return True
    # Surface 2: Outbound URL generation in answer_only responses.
    if tool_name == "answer_only" and "http" in str(call.get("answer") or "").lower():
        return True
    # Surface 3: Sensitive credential or secret file reads.
    if tool_name == "read_file" and any(s in str(call.get("args") or "").lower() for s in (".env", ".ssh", ".pypirc", ".git")):
        return True
    # Surface 4: Any unrecognized tool not explicitly declared in SAFE_TOOLS.
    return tool_name not in SAFE_TOOLS


def _is_usable_clean_call(clean_call: dict[str, Any]) -> bool:
    """Verify that a replanned clean-room call contains a recognized tool and non-empty content."""
    if not isinstance(clean_call, dict):
        return False
    tool_name = " ".join(str(clean_call.get("tool") or "").split())
    if tool_name not in (PRIVILEGED_TOOLS | SAFE_TOOLS):
        return False
    return bool(str(clean_call.get("args") or "").strip() or str(clean_call.get("answer") or "").strip())


class ToolInterceptor:
    """MemoryGuard Step 3: verifies guarded actions via Clean-Room non-interference."""

    def __init__(self, mode: str = "clean_room", recover_on_mismatch: bool = True) -> None:
        """Initialize interceptor with execution mode and recovery preference."""
        if mode not in ("none", "naive_biba_min", "clean_room"):
            raise ValueError(f"Unsupported ToolInterceptor mode: {mode}")
        self.mode = mode
        self.recover_on_mismatch = recover_on_mismatch

    def check_and_execute(
        self,
        proposed_call: dict[str, Any],
        retrieved: list[MemoryRecord],
        replan_fn: Callable[[list[MemoryRecord]], dict[str, Any]],
    ) -> dict[str, Any]:
        """Audit and intercept proposed tool call, returning one of seven canonical reason verdicts."""
        has_ext = any(rec.tier == TIER_EXTERNAL for rec in retrieved)
        base = {"tainted_recall": has_ext, "final_call": proposed_call}

        # Verdict 1 (safe_tool): Unguarded read-only action allowed without isolation.
        if not _needs_clean_room(proposed_call):
            return {**base, "allowed": True, "hijack_blocked": False, "recovered": False, "clean_room_ran": False, "reason": "safe_tool"}

        # Verdict 2 (unprotected) or Verdict 3 (clean_recall):
        # Either running in unprotected mode or recall context has zero untrusted external notes.
        if self.mode == "none" or not has_ext:
            return {**base, "allowed": True, "hijack_blocked": False, "recovered": False, "clean_room_ran": False, "reason": "unprotected" if self.mode == "none" else "clean_recall"}

        # Verdict 4 (naive_taint_lock): Blanket integrity lock blocking any action upon taint.
        if self.mode == "naive_biba_min":
            return {**base, "allowed": False, "hijack_blocked": False, "recovered": False, "clean_room_ran": False, "reason": "naive_taint_lock"}

        # Replan strictly with trusted repository and user notes (tier >= TIER_REPO).
        trusted = [rec for rec in retrieved if rec.tier >= TIER_REPO]
        clean_call = replan_fn(trusted)

        # Verdict 5 (clean_room_verified): Clean replan matches proposed action; taint caused no deviation.
        if calls_are_equivalent(proposed_call, clean_call):
            return {**base, "allowed": True, "hijack_blocked": False, "recovered": False, "clean_room_ran": True, "reason": "clean_room_verified"}

        # Verdict 6 (clean_room_recovered): Deviation detected; hijack blocked and recovered via clean call.
        if self.recover_on_mismatch and _is_usable_clean_call(clean_call):
            return {"allowed": True, "hijack_blocked": True, "recovered": True, "final_call": clean_call, "tainted_recall": True, "clean_room_ran": True, "reason": "clean_room_recovered"}

        # Verdict 7 (clean_room_blocked): Deviation detected but clean replan is unavailable or empty.
        return {**base, "allowed": False, "hijack_blocked": True, "recovered": False, "clean_room_ran": True, "reason": "clean_room_blocked"}
