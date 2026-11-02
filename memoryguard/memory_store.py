"""SQLite and FTS5 memory store with trust tiers and duplicate prevention.

This module implements a persistent database where the AI agent saves notes across sessions.
Every saved note is tagged with a source trust level:
  - USER (3): things you told the agent directly in chat.
  - INTERNAL (2): trusted project files and documentation checked into your project.
  - EXTERNAL (1): untrusted notes from web pages or GitHub issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
import sqlite3
import uuid

# ---------------------------------------------------------------------------
# 1. Three-Tier Source Trust Hierarchy
# ---------------------------------------------------------------------------
# Integer trust hierarchy: USER (3) > INTERNAL (2) > EXTERNAL (1).
# Higher numbers mean higher trust during duplicate resolution.
# By assigning numbers, we can compare trust levels with a simple `>=` or `>` check:
#   - Tier 3 (USER): Things you told the agent directly in chat.
#   - Tier 2 (INTERNAL): Trusted project files and docs (e.g., README.md, CONTRIBUTING.md).
#   - Tier 1 (EXTERNAL): Untrusted notes from web pages or GitHub issues.
TIER_EXTERNAL = 1
TIER_INTERNAL = 2
TIER_USER = 3

# Human-readable labels for each numeric tier, used in logs and inspection output.
TIER_NAMES = {
    TIER_EXTERNAL: "EXTERNAL",
    TIER_INTERNAL: "INTERNAL",
    TIER_USER: "USER",
}

# Regular expression that extracts lowercase alphanumeric words (letters and digits)
# from a search query so we can safely look them up in the full-text search index.
_WORD_RE = re.compile(r"[a-z0-9]+")


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
    """One stored memory note with source trust metadata."""

    # `@dataclass` automatically generates `__init__`, `__repr__`, and `__eq__` methods
    # for this class based on the five fields declared below:
    note_id: str     # Unique identifier for this memory note (e.g., "m_a1b2c3d4e5f6").
    text: str        # The actual note text saved in memory.
    tier: int        # Trust tier integer: 3 (USER), 2 (INTERNAL), or 1 (EXTERNAL).
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
            "note_id TEXT PRIMARY KEY, text TEXT NOT NULL, tier INTEGER NOT NULL, "
            "source_uri TEXT NOT NULL, turn_idx INTEGER NOT NULL)"
        )

        # Table 2 (`memories_fts`): a special SQLite FTS5 (Full-Text Search) virtual table.
        # It indexes every word using the Porter stemmer (`tokenize='porter unicode61'`),
        # so searching for "testing" or "tests" automatically matches a note containing "test".
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5("
            "note_id UNINDEXED, text, tokenize='porter unicode61')"
        )
        # Save (commit) the newly created tables in SQLite.
        self._conn.commit()

    def _all_records(self, min_tier: int = TIER_EXTERNAL) -> list[MemoryRecord]:
        """Fetch all memory records meeting the minimum trust tier, sorted by turn index."""
        # Query all stored memories whose trust tier is at least `min_tier`.
        # Passing `min_tier=TIER_INTERNAL` (2) automatically filters out untrusted `TIER_EXTERNAL` (1) notes!
        rows = self._conn.execute(
            "SELECT note_id, text, tier, source_uri, turn_idx FROM memories WHERE tier >= ? ORDER BY turn_idx ASC",
            (min_tier,),
        ).fetchall()
        # Convert each raw database tuple `(note_id, text, tier, source_uri, turn_idx)` into a `MemoryRecord`.
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
        self, text: str, tier: int, source_uri: str = "", turn_idx: int = 0, duplicate_threshold: float = 0.88
    ) -> tuple[str, MemoryRecord | None]:
        """Save or update a memory note based on source trust levels and duplicate rules."""
        # Step 1: Trim leading/trailing spaces and reject empty notes.
        clean_text = (text or "").strip()
        if not clean_text:
            return "empty", None

        # Step 2: Build a default URI (like "tier://user") if the caller did not pass one.
        resolved_uri = source_uri or f"tier://{TIER_NAMES.get(tier, 'external').lower()}"

        # Step 3: Find any existing memories that are near-duplicates (>= 88% character similarity).
        near_duplicates = [r for r in self._all_records() if note_similarity(clean_text, r.text) >= duplicate_threshold]

        # Step 4: Duplicate protection ("blocked_duplicate").
        # A web note (EXTERNAL) cannot overwrite a matching rule from USER or INTERNAL.
        # If any existing near-duplicate has a higher trust level than the incoming note,
        # block the incoming duplicate.
        if near_duplicates and any(d.tier > tier for d in near_duplicates):
            return "blocked_duplicate", None

        # Step 5: Replace older duplicates of equal or lower trust ("updated_existing").
        # An updated rule at equal or higher trust replaces the older note.
        # Remove the older duplicate rows from both the main table and the FTS5 search index.
        for d in near_duplicates:
            self._conn.execute("DELETE FROM memories WHERE note_id = ?", (d.note_id,))
            self._conn.execute("DELETE FROM memories_fts WHERE note_id = ?", (d.note_id,))

        # Step 6: Generate a short, unique 12-character hex ID and insert the new note into both tables.
        note_id = f"m_{uuid.uuid4().hex[:12]}"
        self._conn.execute("INSERT INTO memories VALUES (?, ?, ?, ?, ?)", (note_id, clean_text, tier, resolved_uri, turn_idx))
        self._conn.execute("INSERT INTO memories_fts (note_id, text) VALUES (?, ?)", (note_id, clean_text))
        self._conn.commit()

        # Return `"updated_existing"` if we replaced an older duplicate, or `"saved"` for a brand-new note.
        return ("updated_existing" if near_duplicates else "saved"), MemoryRecord(note_id, clean_text, tier, resolved_uri, turn_idx)

    def retrieve(self, query: str, top_k: int = 5, min_tier: int = TIER_EXTERNAL) -> list[MemoryRecord]:
        """Retrieve up to top_k memories ranked by BM25 keyword matching with recency backfill."""
        if top_k <= 0:
            return []

        # Build a lookup dictionary `{note_id: MemoryRecord}` containing only records that meet `min_tier`.
        records_by_id = {r.note_id: r for r in self._all_records(min_tier=min_tier)}

        # Extract unique lowercase words of length >= 2 from the user's query while preserving order.
        words = list(dict.fromkeys(w for w in _WORD_RE.findall((query or "").lower()) if len(w) >= 2))
        ranked: list[MemoryRecord] = []
        seen: set[str] = set()

        # Phase 1: BM25 Full-Text Keyword Search.
        # Double-quoted FTS5 words prevent syntax crashes on boolean keywords (AND, OR, NOT).
        if words:
            # Join search words with `OR` so a note matching any query keyword is scored by BM25.
            fts_query = " OR ".join(f'"{w}"' for w in words)
            # In SQLite FTS5, `bm25(memories_fts)` returns negative numbers where more negative
            # means a stronger keyword match, so `ORDER BY bm25(memories_fts) ASC` puts the best match first.
            for (nid,) in self._conn.execute(
                "SELECT note_id FROM memories_fts WHERE memories_fts MATCH ? ORDER BY bm25(memories_fts) ASC",
                (fts_query,),
            ).fetchall():
                if nid in records_by_id and nid not in seen:
                    ranked.append(records_by_id[nid])
                    seen.add(nid)
                    # Stop early as soon as we have collected `top_k` results.
                    if len(ranked) >= top_k:
                        return ranked

        # Phase 2: Recency Backfill.
        # If keyword search returned fewer than `top_k` notes, fill the remaining slots
        # with the most recently saved memories (`turn_idx` descending) so general project
        # conventions are still included in the agent's context.
        backfill = sorted((r for nid, r in records_by_id.items() if nid not in seen), key=lambda r: r.turn_idx, reverse=True)
        ranked.extend(backfill[: top_k - len(ranked)])
        return ranked

    def count_stored_words(self) -> int:
        """Count total whitespace-separated words across all stored memories."""
        # Used by the benchmark and demo to measure how many words are stored in memory
        # and how much conversational small talk the write gate pruned.
        return sum(len(r.text.split()) for r in self._all_records())
