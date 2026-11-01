"""Action-time Clean-Room tool interceptor and non-interference guard for MemoryGuard.

Why do we need an Action-Time Guard (Stage 3) in addition to a Write-Time Gate (Stage 1)?
Because sneaky attacks (like a fake "default pytest flag" hidden inside a GitHub issue)
sound calm and helpful at write time, so any write-time classifier will let them into
memory.

When the agent later proposes a privileged action (like running a bash command) while an
untrusted `TIER_EXTERNAL` note is in its retrieved context, `ToolInterceptor` performs
a **Counterfactual Clean-Room Check**:
  1. It strips out all untrusted `TIER_EXTERNAL` (Tier 1) notes, keeping only trusted
     `TIER_USER` (Tier 3) and `TIER_REPO` (Tier 2) memories.
  2. It asks the model to re-plan the tool call inside this isolated "Clean-Room".
  3. If the Clean-Room tool call matches the original proposed call, the external note
     had zero effect on the command (`clean_room_verified`), so it executes safely.
  4. If the two calls differ, the untrusted note tried to hijack the command! Instead of
     crashing or locking out the user, MemoryGuard discards the poisoned call and runs
     the clean call (`clean_room_recovered`).
"""

from __future__ import annotations

import re
from typing import Any, Callable

from memoryguard.memory_store import TIER_EXTERNAL, TIER_REPO, MemoryRecord

# Privileged tools that can modify the computer or send data over the network.
PRIVILEGED_TOOLS = {"run_bash", "write_file", "http_post"}

# Safe read-only tools that do not modify state (unless reading secret files or outputting URLs).
SAFE_TOOLS = {"read_file", "search_docs", "answer_only"}

# Regular expression to extract `http://` or `https://` links from text responses.
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def calls_are_equivalent(call_a: dict[str, Any], call_b: dict[str, Any]) -> bool:
    """Return True if two tool calls perform the same logical action."""
    # Normalize the tool name (e.g., "run_bash") by stripping extra spaces.
    tool_a = " ".join(str(call_a.get("tool") or "").split())
    tool_b = " ".join(str(call_b.get("tool") or "").split())

    # If either tool name is missing or the two calls invoke different tools, they are not equivalent.
    if not tool_a or tool_a != tool_b:
        return False

    # Case 1: Read-only text answers (`answer_only`).
    # Two text answers might phrase sentences slightly differently while still being safe,
    # unless an attacker injected a malicious phishing/webhook URL. So we compare the set
    # of URLs referenced in both answers.
    if tool_a == "answer_only":
        return set(_URL_RE.findall(str(call_a.get("answer") or ""))) == set(_URL_RE.findall(str(call_b.get("answer") or "")))

    # Case 2: Command or file tools (`run_bash`, `write_file`, `read_file`, `http_post`).
    # Collapse consecutive spaces into a single space and require exact argument equality
    # (for example, `pytest tests/test_runner.py -q` must match flag-for-flag).
    return " ".join(str(call_a.get("args") or "").split()) == " ".join(str(call_b.get("args") or "").split())


def _needs_clean_room(call: dict[str, Any]) -> bool:
    """Determine whether an action touches one of four guarded security surfaces."""
    tool_name = " ".join(str(call.get("tool") or "").split())

    # Surface 1: Privileged side-effecting execution (shell commands, file writes, outbound HTTP POST).
    if tool_name in PRIVILEGED_TOOLS:
        return True

    # Surface 2: Outbound URL generation in `answer_only` responses (guards against phishing/exfil links).
    if tool_name == "answer_only" and "http" in str(call.get("answer") or "").lower():
        return True

    # Surface 3: Sensitive credential or secret file reads (blocks untrusted notes from reading `.env` or `.ssh`).
    if tool_name == "read_file" and any(s in str(call.get("args") or "").lower() for s in (".env", ".ssh", ".pypirc", ".git")):
        return True

    # Surface 4: Any unrecognized tool not explicitly listed in `SAFE_TOOLS` is guarded by default.
    return tool_name not in SAFE_TOOLS


def _is_usable_clean_call(clean_call: dict[str, Any]) -> bool:
    """Verify that a replanned clean-room call contains a recognized tool and non-empty content."""
    if not isinstance(clean_call, dict):
        return False
    tool_name = " ".join(str(clean_call.get("tool") or "").split())
    # Must be a known tool in either `PRIVILEGED_TOOLS` or `SAFE_TOOLS`.
    if tool_name not in (PRIVILEGED_TOOLS | SAFE_TOOLS):
        return False
    # Must have non-empty `args` (for commands/files) or non-empty `answer` (for `answer_only`).
    return bool(str(clean_call.get("args") or "").strip() or str(clean_call.get("answer") or "").strip())


class ToolInterceptor:
    """MemoryGuard Step 3: verifies guarded actions via Clean-Room non-interference."""

    def __init__(self, mode: str = "clean_room", recover_on_mismatch: bool = True) -> None:
        """Initialize interceptor with execution mode and recovery preference."""
        # Supported modes:
        #   - "none": No action-time protection (used by Arms 1, 2, and 3).
        #   - "naive_biba_min": Blanket Taint Lock (Arm 4) that refuses any guarded tool call
        #     whenever an external note is present in retrieved memory.
        #   - "clean_room": MemoryGuard (Arm 5) counterfactual replanning and recovery.
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
        # Check whether any of the retrieved memory notes came from an untrusted external source (Tier 1).
        has_ext = any(rec.tier == TIER_EXTERNAL for rec in retrieved)
        base = {"tainted_recall": has_ext, "final_call": proposed_call}

        # Verdict 1 (`safe_tool`): The proposed action is a harmless read-only call (like reading
        # `pyproject.toml`), so we allow it immediately without running the Clean-Room.
        if not _needs_clean_room(proposed_call):
            return {**base, "allowed": True, "hijack_blocked": False, "recovered": False, "clean_room_ran": False, "reason": "safe_tool"}

        # Verdict 2 (`unprotected`) or Verdict 3 (`clean_recall`):
        # Either the interceptor is turned off (`mode == "none"`) OR the retrieved memories
        # contain zero external notes (`not has_ext`). In both cases, no untrusted taint
        # could have influenced the tool call, so we allow it immediately.
        if self.mode == "none" or not has_ext:
            return {**base, "allowed": True, "hijack_blocked": False, "recovered": False, "clean_room_ran": False, "reason": "unprotected" if self.mode == "none" else "clean_recall"}

        # Verdict 4 (`naive_taint_lock`): Blanket Integrity Lock baseline (Arm 4).
        # Because at least one external note is in context, this naive rule panics and
        # blocks the tool call completely, causing an 88.9% false-refusal rate on safe tasks!
        if self.mode == "naive_biba_min":
            return {**base, "allowed": False, "hijack_blocked": False, "recovered": False, "clean_room_ran": False, "reason": "naive_taint_lock"}

        # -------------------------------------------------------------------
        # MemoryGuard Counterfactual Clean-Room Verification (`mode == "clean_room"`)
        # -------------------------------------------------------------------
        # Filter the retrieved memories to keep ONLY trusted notes (`tier >= TIER_REPO`,
        # which includes `TIER_REPO = 2` and `TIER_USER = 3`), stripping out `TIER_EXTERNAL = 1`.
        trusted = [rec for rec in retrieved if rec.tier >= TIER_REPO]

        # Ask the agent (`replan_fn`) what tool call it would propose if it only saw `trusted` notes.
        clean_call = replan_fn(trusted)

        # Verdict 5 (`clean_room_verified`): The Clean-Room call is identical to `proposed_call`!
        # That proves the external note was just harmless background context (e.g., release notes)
        # and did not alter the command. Allow the action with zero false locks!
        if calls_are_equivalent(proposed_call, clean_call):
            return {**base, "allowed": True, "hijack_blocked": False, "recovered": False, "clean_room_ran": True, "reason": "clean_room_verified"}

        # Verdict 6 (`clean_room_recovered`): The proposed call differs from the Clean-Room call!
        # That proves the untrusted external note altered the command parameters (a hijack).
        # We block the poisoned `proposed_call` and execute the safe `clean_call` instead!
        if self.recover_on_mismatch and _is_usable_clean_call(clean_call):
            return {"allowed": True, "hijack_blocked": True, "recovered": True, "final_call": clean_call, "tainted_recall": True, "clean_room_ran": True, "reason": "clean_room_recovered"}

        # Verdict 7 (`clean_room_blocked`): A hijack was detected, and recovery is disabled
        # or the Clean-Room did not produce a usable command, so we block execution safely.
        return {**base, "allowed": False, "hijack_blocked": True, "recovered": False, "clean_room_ran": True, "reason": "clean_room_blocked"}
