"""SQLite access layer for Axiom.

Thin helpers around the shared schema (db/schema.sql). Keeps all SQL in one
place so the data contract is enforced from a single module.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Iterable, Sequence

from axiom import config


def connect(db_path=None) -> sqlite3.Connection:
    """Open a connection with foreign keys on and row access by column name."""
    path = str(db_path or config.DB_PATH)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables/indexes from the shared schema (idempotent)."""
    sql = config.SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(sql)
    conn.commit()


def reset_corpus(conn: sqlite3.Connection) -> None:
    """Clear all corpus data tables in FK-safe order."""
    _CORPUS_TABLES = (
        "review_queue",
        "paper_summaries",
        "reading_list",
        "concept_canonical",
        "concepts",
        "citation_edges",
        "paper_provenance",
        "papers",
    )
    for table in _CORPUS_TABLES:
        conn.execute(f"DELETE FROM {table}")
    conn.commit()


def text_hash(text: str | None) -> str:
    """Stable hash of abstract text for dedup / change detection."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def insert_paper(
    conn: sqlite3.Connection,
    *,
    openalex_id: str,
    title: str | None,
    abstract: str | None,
    publication_year: int | None,
    venue_id: str | None,
    venue: str | None,
    cited_by_count: int = 0,
    doi: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO papers
            (openalex_id, title, abstract, publication_year,
             venue_id, venue, cited_by_count, doi)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (openalex_id, title, abstract, publication_year,
         venue_id, venue, cited_by_count, doi),
    )


def insert_concepts(
    conn: sqlite3.Connection,
    paper_id: str,
    concepts: Iterable[tuple[str, int]],
) -> None:
    """concepts: iterable of (concept_name, level)."""
    conn.executemany(
        "INSERT INTO concepts (paper_id, concept, level) VALUES (?, ?, ?)",
        [(paper_id, c, lvl) for c, lvl in concepts],
    )


def insert_citation_edges(
    conn: sqlite3.Connection,
    edges: Iterable[tuple[str, str, int | None]],
) -> None:
    """edges: iterable of (src_id, dst_id, year)."""
    conn.executemany(
        "INSERT INTO citation_edges (src_id, dst_id, year) VALUES (?, ?, ?)",
        list(edges),
    )


def insert_provenance(
    conn: sqlite3.Connection,
    *,
    paper_id: str,
    source: str,
    abstract: str | None,
    has_fulltext: bool = False,
    fetched_at: datetime | None = None,
) -> None:
    ts = (fetched_at or datetime.now(timezone.utc)).isoformat()
    conn.execute(
        """
        INSERT OR REPLACE INTO paper_provenance
            (paper_id, source, fetched_at, text_hash, has_fulltext)
        VALUES (?, ?, ?, ?, ?)
        """,
        (paper_id, source, ts, text_hash(abstract), has_fulltext),
    )


def distinct_venues(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT venue FROM papers WHERE venue IS NOT NULL ORDER BY venue"
    ).fetchall()
    return [r["venue"] for r in rows]


def upsert_canonical(conn: sqlite3.Connection, concept: str, canonical: str,
                      source: str = "auto") -> None:
    conn.execute(
        "INSERT OR REPLACE INTO concept_canonical (concept, canonical, source) "
        "VALUES (?, ?, ?)",
        (concept, canonical, source),
    )


def manual_canonical_concepts(conn: sqlite3.Connection) -> set[str]:
    return {r["concept"] for r in
            conn.execute("SELECT concept FROM concept_canonical WHERE source = 'manual'").fetchall()}


def canonical_map(conn: sqlite3.Connection) -> dict[str, str]:
    """concept -> canonical. Concepts with no row map to themselves (identity)."""
    return {r["concept"]: r["canonical"] for r in
            conn.execute("SELECT concept, canonical FROM concept_canonical").fetchall()}


def save_summary(conn: sqlite3.Connection, paper_id: str, bullets: list[str],
                  model: str, created_at: datetime | None = None) -> None:
    ts = (created_at or datetime.now(timezone.utc)).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO paper_summaries (paper_id, bullets_json, model, created_at) "
        "VALUES (?, ?, ?, ?)",
        (paper_id, json.dumps(bullets), model, ts),
    )
    conn.commit()


def get_summary(conn: sqlite3.Connection, paper_id: str) -> list[str] | None:
    row = conn.execute(
        "SELECT bullets_json FROM paper_summaries WHERE paper_id = ?", (paper_id,)
    ).fetchone()
    return json.loads(row["bullets_json"]) if row else None


def add_to_review_queue(conn: sqlite3.Connection, *, gap_a_label: str, gap_b_label: str,
                         title: str, claim: str, method_sketch: str,
                         datasets: list[str], supporting_paper_ids: list[str],
                         created_at: datetime | None = None) -> int:
    ts = (created_at or datetime.now(timezone.utc)).isoformat()
    cur = conn.execute(
        "INSERT INTO review_queue (gap_a_label, gap_b_label, title, claim, "
        "method_sketch, datasets_json, supporting_ids_json, status, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
        (gap_a_label, gap_b_label, title, claim, method_sketch,
         json.dumps(datasets), json.dumps(supporting_paper_ids), ts),
    )
    conn.commit()
    return cur.lastrowid


def list_review_queue(conn: sqlite3.Connection, status: str | None = None) -> list[sqlite3.Row]:
    if status:
        return conn.execute(
            "SELECT * FROM review_queue WHERE status = ? ORDER BY created_at DESC", (status,)
        ).fetchall()
    return conn.execute("SELECT * FROM review_queue ORDER BY created_at DESC").fetchall()


def set_review_status(conn: sqlite3.Connection, item_id: int, status: str) -> None:
    assert status in ("pending", "approved", "rejected")
    conn.execute("UPDATE review_queue SET status = ? WHERE id = ?", (status, item_id))
    conn.commit()


def add_bookmark(conn: sqlite3.Connection, paper_id: str,
                  added_at: datetime | None = None) -> None:
    ts = (added_at or datetime.now(timezone.utc)).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO reading_list (paper_id, added_at) VALUES (?, ?)",
        (paper_id, ts),
    )
    conn.commit()


def remove_bookmark(conn: sqlite3.Connection, paper_id: str) -> None:
    conn.execute("DELETE FROM reading_list WHERE paper_id = ?", (paper_id,))
    conn.commit()


def list_bookmarks(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Bookmarked papers, most-recently-added first, joined with paper metadata."""
    return conn.execute(
        """
        SELECT r.paper_id, r.added_at, p.title, p.abstract, p.publication_year,
               p.venue, p.cited_by_count, p.doi
        FROM reading_list r
        JOIN papers p ON p.openalex_id = r.paper_id
        ORDER BY r.added_at DESC
        """
    ).fetchall()


def year_bounds(conn: sqlite3.Connection) -> tuple[int, int] | None:
    row = conn.execute(
        "SELECT MIN(publication_year) AS lo, MAX(publication_year) AS hi FROM papers"
    ).fetchone()
    if row is None or row["lo"] is None:
        return None
    return int(row["lo"]), int(row["hi"])


def iter_papers(conn: sqlite3.Connection) -> Sequence[sqlite3.Row]:
    """All papers, for embedding/upsert during bootstrap."""
    return conn.execute(
        "SELECT openalex_id, title, abstract, publication_year, "
        "venue, cited_by_count FROM papers"
    ).fetchall()


def papers_by_ids(
    conn: sqlite3.Connection, ids: Sequence[str]
) -> dict[str, sqlite3.Row]:
    """Fetch full paper rows (incl. abstract, doi) keyed by openalex_id.

    Used by the UI to enrich lean Qdrant hits with abstract/DOI at render time.
    """
    ids = list(ids)
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT openalex_id, title, abstract, publication_year, venue, "
        f"cited_by_count, doi FROM papers WHERE openalex_id IN ({placeholders})",
        ids,
    ).fetchall()
    return {r["openalex_id"]: r for r in rows}



