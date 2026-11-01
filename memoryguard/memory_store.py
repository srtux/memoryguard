"""SQLite + FTS5 memory store with trust tiers and duplicate prevention."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
import sqlite3
import uuid

# Integer provenance hierarchy: USER (3) > REPO (2) > EXTERNAL (1).
# Higher integers strictly dominate lower integers during near-duplicate resolution.
TIER_EXTERNAL = 1
TIER_REPO = 2
TIER_USER = 3

TIER_NAMES = {
    TIER_EXTERNAL: "EXTERNAL",
    TIER_REPO: "REPO",
    TIER_USER: "USER",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def note_similarity(text_a: str, text_b: str) -> float:
    """Compare two notes (0.0 to 1.0) by normalized character sequence similarity."""
    norm_a = " ".join((text_a or "").lower().split())
    norm_b = " ".join((text_b or "").lower().split())
    if not norm_a or not norm_b:
        return 0.0
    return SequenceMatcher(None, norm_a, norm_b).ratio()


@dataclass
class MemoryRecord:
    """One stored memory record with provenance metadata."""

    mem_id: str
    text: str
    tier: int
    source_uri: str
    turn_idx: int


class MemoryStore:
    """SQLite + FTS5 memory store with trust tiers and duplicate protection."""

    def __init__(self, db_path: str = ":memory:") -> None:
        """Initialize SQLite tables for structured memory rows and full-text search."""
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS memories ("
            "mem_id TEXT PRIMARY KEY, text TEXT NOT NULL, tier INTEGER NOT NULL, "
            "source_uri TEXT NOT NULL, turn_idx INTEGER NOT NULL)"
        )
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5("
            "mem_id UNINDEXED, text, tokenize='porter unicode61')"
        )
        self._conn.commit()

    def _all_records(self, min_tier: int = TIER_EXTERNAL) -> list[MemoryRecord]:
        """Fetch all memory records meeting the minimum trust tier, sorted by turn index."""
        rows = self._conn.execute(
            "SELECT mem_id, text, tier, source_uri, turn_idx FROM memories WHERE tier >= ? ORDER BY turn_idx ASC",
            (min_tier,),
        ).fetchall()
        return [MemoryRecord(r[0], r[1], int(r[2]), r[3], int(r[4])) for r in rows]

    def max_similarity(self, text: str, min_tier: int = TIER_EXTERNAL) -> tuple[float, MemoryRecord | None]:
        """Return the highest similarity score and matching record among stored memories."""
        records = self._all_records(min_tier=min_tier)
        if not records:
            return 0.0, None
        best_sim, best_rec = max(((note_similarity(text, r.text), r) for r in records), key=lambda p: (p[0], p[1].tier))
        return float(best_sim), (best_rec if best_sim > 0.0 else None)

    def add_memory(
        self, text: str, tier: int, source_uri: str = "", turn_idx: int = 0, dup_thresh: float = 0.88
    ) -> tuple[str, MemoryRecord | None]:
        """Insert or replace a memory note according to authority dominance invariants."""
        clean_text = (text or "").strip()
        if not clean_text:
            return "empty", None
        resolved_uri = source_uri or f"tier://{TIER_NAMES.get(tier, 'external').lower()}"
        near_dups = [r for r in self._all_records() if note_similarity(clean_text, r.text) >= dup_thresh]

        # Authority dominance: if any existing near-duplicate has a strictly higher tier,
        # the incoming lower-tier candidate is shadowed and rejected without modification.
        if near_dups and any(d.tier > tier for d in near_dups):
            return "shadowed", None

        # Otherwise, replace any equal or lower tier duplicates with the fresh note.
        for d in near_dups:
            self._conn.execute("DELETE FROM memories WHERE mem_id = ?", (d.mem_id,))
            self._conn.execute("DELETE FROM memories_fts WHERE mem_id = ?", (d.mem_id,))
        mem_id = f"m_{uuid.uuid4().hex[:12]}"
        self._conn.execute("INSERT INTO memories VALUES (?, ?, ?, ?, ?)", (mem_id, clean_text, tier, resolved_uri, turn_idx))
        self._conn.execute("INSERT INTO memories_fts (mem_id, text) VALUES (?, ?)", (mem_id, clean_text))
        self._conn.commit()
        return ("superseded" if near_dups else "inserted"), MemoryRecord(mem_id, clean_text, tier, resolved_uri, turn_idx)

    def retrieve(self, query: str, top_k: int = 5, min_tier: int = TIER_EXTERNAL) -> list[MemoryRecord]:
        """Retrieve up to top_k memories ranked by BM25 keyword matching with recency backfill."""
        if top_k <= 0:
            return []
        records_by_id = {r.mem_id: r for r in self._all_records(min_tier=min_tier)}
        tokens = list(dict.fromkeys(t for t in _TOKEN_RE.findall((query or "").lower()) if len(t) >= 2))
        ranked: list[MemoryRecord] = []
        seen: set[str] = set()

        # Double-quoted FTS5 tokens prevent syntax crashes on boolean keywords (AND, OR, NOT).
        if tokens:
            fts_query = " OR ".join(f'"{t}"' for t in tokens)
            for (mid,) in self._conn.execute(
                "SELECT mem_id FROM memories_fts WHERE memories_fts MATCH ? ORDER BY bm25(memories_fts) ASC",
                (fts_query,),
            ).fetchall():
                if mid in records_by_id and mid not in seen:
                    ranked.append(records_by_id[mid])
                    seen.add(mid)
                    if len(ranked) >= top_k:
                        return ranked

        # Recency backfill: fill remaining slots up to top_k using newest turn_idx.
        backfill = sorted((r for mid, r in records_by_id.items() if mid not in seen), key=lambda r: r.turn_idx, reverse=True)
        ranked.extend(backfill[: top_k - len(ranked)])
        return ranked

    def active_tokens(self) -> int:
        """Count total whitespace-separated words across all stored memories."""
        return sum(len(r.text.split()) for r in self._all_records())
