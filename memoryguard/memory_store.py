"""SQLite + FTS5 memory store with trust tiers and duplicate prevention.

This module implements Stage 2 of MemoryGuard: a persistent database where the AI
agent saves notes across sessions. Every saved note is tagged with a Biba Integrity
Trust Tier (`USER = 3`, `REPO = 2`, `EXTERNAL = 1`) so the agent always knows
whether a memory came from the human developer, the code repository, or an
untrusted external source (like a web page or GitHub issue).
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
import sqlite3
import uuid

# ---------------------------------------------------------------------------
# 1. Three-Tier Biba Integrity Hierarchy
# ---------------------------------------------------------------------------
# Integer provenance hierarchy: USER (3) > REPO (2) > EXTERNAL (1).
# Higher integers strictly dominate lower integers during near-duplicate resolution.
# By assigning integers, we can compare trust levels with a simple `>=` or `>` check:
#   - Tier 3 (USER): Direct statements from the authenticated developer in chat.
#   - Tier 2 (REPO): Vetted configuration files checked into the git repository.
#   - Tier 1 (EXTERNAL): Untrusted text from GitHub issues, web pages, or 3rd-party docs.
TIER_EXTERNAL = 1
TIER_REPO = 2
TIER_USER = 3

# Human-readable labels for each numeric tier, used in logs and inspection output.
TIER_NAMES = {
    TIER_EXTERNAL: "EXTERNAL",
    TIER_REPO: "REPO",
    TIER_USER: "USER",
}

# Regular expression that extracts lowercase alphanumeric words (letters and digits)
# from a search query so we can safely look them up in the full-text search index.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def note_similarity(text_a: str, text_b: str) -> float:
    """Compare two notes (0.0 to 1.0) by normalized character sequence similarity."""
    # Convert both strings to lowercase and collapse extra spaces/newlines into single spaces
    # so minor formatting differences do not trick the similarity check.
    norm_a = " ".join((text_a or "").lower().split())
    norm_b = " ".join((text_b or "").lower().split())

    # If either note is empty after stripping spaces, they have 0.0 similarity.
    if not norm_a or not norm_b:
        return 0.0

    # `SequenceMatcher.ratio()` returns a float between 0.0 (completely different)
    # and 1.0 (identical character sequence).
    return SequenceMatcher(None, norm_a, norm_b).ratio()


@dataclass
class MemoryRecord:
    """One stored memory record with provenance metadata."""

    # `@dataclass` automatically generates `__init__`, `__repr__`, and `__eq__` methods
    # for this class based on the five fields declared below:
    mem_id: str      # Unique identifier for this memory row (e.g., "m_a1b2c3d4e5f6").
    text: str        # The actual note text saved in memory.
    tier: int        # Trust tier integer: 3 (USER), 2 (REPO), or 1 (EXTERNAL).
    source_uri: str  # Where the note came from (e.g., "chat://user" or "github://issues/224").
    turn_idx: int    # Chronological turn counter (1, 2, 3...) indicating when it was saved.


class MemoryStore:
    """SQLite + FTS5 memory store with trust tiers and duplicate protection."""

    def __init__(self, db_path: str = ":memory:") -> None:
        """Initialize SQLite tables for structured memory rows and full-text search."""
        self.db_path = db_path
        # Connect to SQLite. Using ":memory:" keeps the database in RAM for fast tests,
        # while passing a file path saves memories to disk across separate program runs.
        # `check_same_thread=False` allows async web servers (like Google ADK) to read/write
        # across event-loop threads.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)

        # Table 1 (`memories`): stores the full structured columns (ID, text, trust tier,
        # source URI, and chronological turn number).
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS memories ("
            "mem_id TEXT PRIMARY KEY, text TEXT NOT NULL, tier INTEGER NOT NULL, "
            "source_uri TEXT NOT NULL, turn_idx INTEGER NOT NULL)"
        )

        # Table 2 (`memories_fts`): a special SQLite FTS5 (Full-Text Search) virtual table.
        # It indexes every word using the Porter stemmer (`tokenize='porter unicode61'`),
        # so searching for "testing" or "tests" automatically matches a note containing "test".
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5("
            "mem_id UNINDEXED, text, tokenize='porter unicode61')"
        )
        # Save (commit) the newly created tables in SQLite.
        self._conn.commit()

    def _all_records(self, min_tier: int = TIER_EXTERNAL) -> list[MemoryRecord]:
        """Fetch all memory records meeting the minimum trust tier, sorted by turn index."""
        # Query all stored memories whose trust tier is at least `min_tier`.
        # Passing `min_tier=TIER_REPO` (2) automatically filters out untrusted `TIER_EXTERNAL` (1) notes!
        rows = self._conn.execute(
            "SELECT mem_id, text, tier, source_uri, turn_idx FROM memories WHERE tier >= ? ORDER BY turn_idx ASC",
            (min_tier,),
        ).fetchall()
        # Convert each raw database tuple `(mem_id, text, tier, source_uri, turn_idx)` into a `MemoryRecord`.
        return [MemoryRecord(r[0], r[1], int(r[2]), r[3], int(r[4])) for r in rows]

    def max_similarity(self, text: str, min_tier: int = TIER_EXTERNAL) -> tuple[float, MemoryRecord | None]:
        """Return the highest similarity score and matching record among stored memories."""
        # Load existing memories at or above `min_tier`.
        records = self._all_records(min_tier=min_tier)
        if not records:
            # If the memory store is empty, similarity is 0.0 (meaning the note is 100% novel).
            return 0.0, None

        # Compare `text` against every stored record and pick the pair `(similarity, record)`
        # with the highest similarity score (breaking ties in favor of higher trust tier).
        best_sim, best_rec = max(((note_similarity(text, r.text), r) for r in records), key=lambda p: (p[0], p[1].tier))
        return float(best_sim), (best_rec if best_sim > 0.0 else None)

    def add_memory(
        self, text: str, tier: int, source_uri: str = "", turn_idx: int = 0, dup_thresh: float = 0.88
    ) -> tuple[str, MemoryRecord | None]:
        """Insert or replace a memory note according to authority dominance invariants."""
        # Step 1: Trim leading/trailing spaces and reject empty notes.
        clean_text = (text or "").strip()
        if not clean_text:
            return "empty", None

        # Step 2: Build a default URI (like "tier://user") if the caller did not pass one.
        resolved_uri = source_uri or f"tier://{TIER_NAMES.get(tier, 'external').lower()}"

        # Step 3: Find any existing memories that are near-duplicates (>= 88% character similarity).
        near_dups = [r for r in self._all_records() if note_similarity(clean_text, r.text) >= dup_thresh]

        # Step 4: Authority dominance (Shadowing).
        # If any existing near-duplicate has a strictly higher trust tier (e.g., USER tier 3)
        # than the incoming note (e.g., EXTERNAL tier 1), reject the incoming note so an
        # untrusted web page or issue can never overwrite or downgrade a user's rule.
        if near_dups and any(d.tier > tier for d in near_dups):
            return "shadowed", None

        # Step 5: Superseding older duplicates of equal or lower tier.
        # If the user updates their own rule (or upgrades an external note to a user rule),
        # remove the older duplicate rows from both the main table and the FTS5 search index.
        for d in near_dups:
            self._conn.execute("DELETE FROM memories WHERE mem_id = ?", (d.mem_id,))
            self._conn.execute("DELETE FROM memories_fts WHERE mem_id = ?", (d.mem_id,))

        # Step 6: Generate a short, unique 12-character hex ID and insert the new note into both tables.
        mem_id = f"m_{uuid.uuid4().hex[:12]}"
        self._conn.execute("INSERT INTO memories VALUES (?, ?, ?, ?, ?)", (mem_id, clean_text, tier, resolved_uri, turn_idx))
        self._conn.execute("INSERT INTO memories_fts (mem_id, text) VALUES (?, ?)", (mem_id, clean_text))
        self._conn.commit()

        # Return `"superseded"` if we replaced an older duplicate, or `"inserted"` for a brand-new note.
        return ("superseded" if near_dups else "inserted"), MemoryRecord(mem_id, clean_text, tier, resolved_uri, turn_idx)

    def retrieve(self, query: str, top_k: int = 5, min_tier: int = TIER_EXTERNAL) -> list[MemoryRecord]:
        """Retrieve up to top_k memories ranked by BM25 keyword matching with recency backfill."""
        if top_k <= 0:
            return []

        # Build a lookup dictionary `{mem_id: MemoryRecord}` containing only records that meet `min_tier`.
        records_by_id = {r.mem_id: r for r in self._all_records(min_tier=min_tier)}

        # Extract unique lowercase words of length >= 2 from the user's query while preserving order.
        tokens = list(dict.fromkeys(t for t in _TOKEN_RE.findall((query or "").lower()) if len(t) >= 2))
        ranked: list[MemoryRecord] = []
        seen: set[str] = set()

        # Phase 1: BM25 Full-Text Keyword Search.
        # Double-quoted FTS5 tokens prevent syntax crashes on boolean keywords (AND, OR, NOT).
        if tokens:
            # Join search words with `OR` so a note matching any query keyword is scored by BM25.
            fts_query = " OR ".join(f'"{t}"' for t in tokens)
            # In SQLite FTS5, `bm25(memories_fts)` returns negative numbers where more negative
            # means a stronger keyword match, so `ORDER BY bm25(memories_fts) ASC` puts the best match first.
            for (mid,) in self._conn.execute(
                "SELECT mem_id FROM memories_fts WHERE memories_fts MATCH ? ORDER BY bm25(memories_fts) ASC",
                (fts_query,),
            ).fetchall():
                if mid in records_by_id and mid not in seen:
                    ranked.append(records_by_id[mid])
                    seen.add(mid)
                    # Stop early as soon as we have collected `top_k` results.
                    if len(ranked) >= top_k:
                        return ranked

        # Phase 2: Recency Backfill.
        # If keyword search returned fewer than `top_k` notes, fill the remaining slots
        # with the most recently saved memories (`turn_idx` descending) so general project
        # conventions are still included in the agent's context.
        backfill = sorted((r for mid, r in records_by_id.items() if mid not in seen), key=lambda r: r.turn_idx, reverse=True)
        ranked.extend(backfill[: top_k - len(ranked)])
        return ranked

    def active_tokens(self) -> int:
        """Count total whitespace-separated words across all stored memories."""
        # Used by the benchmark and demo to measure how many words are stored in memory
        # and how much conversational bloat the write gate pruned.
        return sum(len(r.text.split()) for r in self._all_records())
