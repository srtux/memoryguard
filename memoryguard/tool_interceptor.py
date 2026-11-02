"""Action-time Clean-Room tool interceptor for MemoryGuard.

Why do we need a command check before acting in addition to a write-time memory gate?
Because sneaky attacks (like a fake "default pytest flag" hidden inside a GitHub issue)
sound calm and helpful at write time, so write-time filters can let them into memory.

When the agent later proposes a sensitive action (like running a bash command) while an
untrusted web note (TIER_EXTERNAL) is in its retrieved context, ToolInterceptor performs
a source double-check before running commands:
  1. It strips out all untrusted web notes (TIER_EXTERNAL), keeping only trusted
     USER and INTERNAL memories.
  2. It asks the model to re-plan the tool call inside this isolated Clean-Room.
  3. If the Clean-Room tool call matches the original proposed call, the external note
     had zero effect on the command (clean_room_verified), so it runs safely.
  4. If the two calls differ, the untrusted note tried to hijack the command.
     Instead of crashing or locking out the user, MemoryGuard discards the tampered
     command and runs the clean call (clean_room_recovered).
"""

from __future__ import annotations

import re
from typing import Any, Callable

from memoryguard.memory_store import TIER_EXTERNAL, TIER_INTERNAL, MemoryRecord

# Sensitive tools that can modify the computer or send data over the network.
SENSITIVE_TOOLS = {"run_bash", "write_file", "http_post"}

# Harmless read-only tools that do not modify state (unless reading secret files or outputting URLs).
READ_ONLY_TOOLS = {"read_file", "search_docs", "answer_only"}

# Regular expression to extract http:// or https:// links from text responses.
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def calls_are_equivalent(call_a: dict[str, Any], call_b: dict[str, Any]) -> bool:
    """Return True if two tool calls perform the same logical action."""
    # Normalize the tool name (e.g., "run_bash") by stripping extra spaces.
    tool_a = " ".join(str(call_a.get("tool") or "").split())
    tool_b = " ".join(str(call_b.get("tool") or "").split())

    # If either tool name is missing or the two calls invoke different tools, they are not equivalent.
    if not tool_a or tool_a != tool_b:
        return False

    # Case 1: Read-only text answers (answer_only).
    # Two text answers might phrase sentences slightly differently while still being safe,
    # unless an attack injected an external link. So we compare the set of URLs in both.
    if tool_a == "answer_only":
        return set(_URL_RE.findall(str(call_a.get("answer") or ""))) == set(_URL_RE.findall(str(call_b.get("answer") or "")))

    # Case 2: Command or file tools (run_bash, write_file, read_file, http_post).
    # Collapse consecutive spaces into a single space and require exact argument equality
    # (for example, `pytest tests/test_runner.py -q` must match flag-for-flag).
    return " ".join(str(call_a.get("args") or "").split()) == " ".join(str(call_b.get("args") or "").split())


def _needs_clean_room(call: dict[str, Any]) -> bool:
    """Determine whether an action touches one of four guarded security surfaces."""
    tool_name = " ".join(str(call.get("tool") or "").split())

    # Surface 1: Sensitive execution (shell commands, file writes, outbound HTTP POST).
    if tool_name in SENSITIVE_TOOLS:
        return True

    # Surface 2: Outbound URL generation in answer_only responses (guards against unauthorized links).
    if tool_name == "answer_only" and "http" in str(call.get("answer") or "").lower():
        return True

    # Surface 3: Sensitive credential or secret file reads (blocks untrusted notes from reading .env or .ssh).
    if tool_name == "read_file" and any(s in str(call.get("args") or "").lower() for s in (".env", ".ssh", ".pypirc", ".git")):
        return True

    # Surface 4: Any unrecognized tool not explicitly listed in READ_ONLY_TOOLS is guarded by default.
    return tool_name not in READ_ONLY_TOOLS


def _is_usable_clean_call(clean_call: dict[str, Any]) -> bool:
    """Verify that a replanned clean-room call contains a recognized tool and non-empty content."""
    if not isinstance(clean_call, dict):
        return False
    tool_name = " ".join(str(clean_call.get("tool") or "").split())
    # Must be a known tool in either SENSITIVE_TOOLS or READ_ONLY_TOOLS.
    if tool_name not in (SENSITIVE_TOOLS | READ_ONLY_TOOLS):
        return False
    # Must have non-empty args (for commands/files) or non-empty answer (for answer_only).
    return bool(str(clean_call.get("args") or "").strip() or str(clean_call.get("answer") or "").strip())


class ToolInterceptor:
    """Verifies sensitive actions via source double-check and Clean-Room recovery."""

    def __init__(self, mode: str = "clean_room", recover_on_mismatch: bool = True) -> None:
        """Initialize interceptor with execution mode and recovery preference."""
        # Supported modes:
        #   - "none": No command check before acting (used by Approaches 1, 2, and 3).
        #   - "block_web_notes": Blocks any sensitive command whenever a web note was retrieved.
        #   - "clean_room": Strips web notes, replans with trusted notes, and recovers clean call.
        if mode not in ("none", "block_web_notes", "clean_room"):
            raise ValueError(f"Unsupported ToolInterceptor mode: {mode}")
        self.mode = mode
        self.recover_on_mismatch = recover_on_mismatch

    def check_and_execute(
        self,
        proposed_call: dict[str, Any],
        retrieved: list[MemoryRecord],
        replan_fn: Callable[[list[MemoryRecord]], dict[str, Any]],
    ) -> dict[str, Any]:
        """Audit and intercept proposed tool call, returning a verdict dictionary."""
        # Check whether any retrieved memory note came from an untrusted external source.
        has_ext = any(rec.tier == TIER_EXTERNAL for rec in retrieved)
        base = {"has_web_notes": has_ext, "final_call": proposed_call}

        # Verdict 1 (read_only_tool): The proposed action is a harmless read-only call (like reading
        # pyproject.toml), so we allow it immediately without running the Clean-Room.
        if not _needs_clean_room(proposed_call):
            return {**base, "allowed": True, "hijack_blocked": False, "recovered": False, "clean_room_ran": False, "reason": "read_only_tool"}

        # Verdict 2 (unprotected) or Verdict 3 (trusted_notes_only):
        # Either the interceptor is turned off (mode == "none") or the retrieved memories
        # contain zero external notes (not has_ext). In both cases, allow immediately.
        if self.mode == "none" or not has_ext:
            return {**base, "allowed": True, "hijack_blocked": False, "recovered": False, "clean_room_ran": False, "reason": "unprotected" if self.mode == "none" else "trusted_notes_only"}

        # Verdict 4 (blocked_web_note): Blocks any sensitive command whenever a web note was retrieved.
        if self.mode == "block_web_notes":
            return {**base, "allowed": False, "hijack_blocked": False, "recovered": False, "clean_room_ran": False, "reason": "blocked_web_note"}

        # -------------------------------------------------------------------
        # MemoryGuard Clean-Room Verification (mode == "clean_room")
        # -------------------------------------------------------------------
        # Filter retrieved memories to keep only trusted notes (tier >= TIER_INTERNAL),
        # stripping out untrusted web notes (TIER_EXTERNAL).
        trusted = [rec for rec in retrieved if rec.tier >= TIER_INTERNAL]

        # Ask the agent (replan_fn) what tool call it would propose seeing only trusted notes.
        clean_call = replan_fn(trusted)

        # Verdict 5 (clean_room_verified): The Clean-Room call matches proposed_call.
        # This confirms the external note did not alter the command, allowing execution.
        if calls_are_equivalent(proposed_call, clean_call):
            return {**base, "allowed": True, "hijack_blocked": False, "recovered": False, "clean_room_ran": True, "reason": "clean_room_verified"}

        # Verdict 6 (clean_room_recovered): The proposed call differs from the Clean-Room call.
        # The untrusted note tampered with the command. Block the tampered call and run clean_call.
        if self.recover_on_mismatch and _is_usable_clean_call(clean_call):
            return {"allowed": True, "hijack_blocked": True, "recovered": True, "final_call": clean_call, "has_web_notes": True, "clean_room_ran": True, "reason": "clean_room_recovered"}

        # Verdict 7 (clean_room_blocked): A tamper was detected, but clean_call was unusable or recovery disabled.
        return {**base, "allowed": False, "hijack_blocked": True, "recovered": False, "clean_room_ran": True, "reason": "clean_room_blocked"}
