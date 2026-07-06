"""Tests for axiom.db — schema init, round-trips, FK-safe reset."""
from __future__ import annotations

import pytest

from axiom import db


def test_init_creates_all_tables(empty_conn):
    names = {r["name"] for r in empty_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"papers", "concepts", "citation_edges", "paper_provenance",
            "concept_canonical", "paper_summaries", "reading_list",
            "review_queue"} <= names


def test_insert_paper_idempotent(empty_conn):
    for _ in range(2):
        db.insert_paper(empty_conn, openalex_id="W1", title="t", abstract="a",
                        publication_year=2020, venue_id=None, venue="ACL",
                        cited_by_count=3, doi=None)
    empty_conn.commit()
    n = empty_conn.execute("SELECT COUNT(*) c FROM papers").fetchone()["c"]
    assert n == 1                              # INSERT OR REPLACE, no duplicate


def test_canonical_map_and_manual(empty_conn):
    db.upsert_canonical(empty_conn, "LoRA", "Low-Rank Adaptation", source="auto")
    db.upsert_canonical(empty_conn, "MT", "Machine Translation", source="manual")
    empty_conn.commit()
    assert db.canonical_map(empty_conn)["LoRA"] == "Low-Rank Adaptation"
    assert db.manual_canonical_concepts(empty_conn) == {"MT"}


def test_review_queue_roundtrip(empty_conn):
    rid = db.add_to_review_queue(
        empty_conn, gap_a_label="A", gap_b_label="B", title="t", claim="c",
        method_sketch="m", datasets=["d1"], supporting_paper_ids=["A1", "B1"])
    assert isinstance(rid, int)
    rows = db.list_review_queue(empty_conn)
    assert len(rows) == 1 and rows[0]["status"] == "pending"

    db.set_review_status(empty_conn, rid, "approved")
    assert db.list_review_queue(empty_conn, status="approved")[0]["id"] == rid
    assert db.list_review_queue(empty_conn, status="pending") == []


def test_set_review_status_rejects_invalid(empty_conn):
    rid = db.add_to_review_queue(
        empty_conn, gap_a_label="A", gap_b_label="B", title="t", claim="c",
        method_sketch="m", datasets=[], supporting_paper_ids=[])
    with pytest.raises(AssertionError):
        db.set_review_status(empty_conn, rid, "bogus")


def test_bookmarks_join_and_order(empty_conn):
    for pid, cites in (("W1", 5), ("W2", 9)):
        db.insert_paper(empty_conn, openalex_id=pid, title=pid, abstract="a",
                        publication_year=2020, venue_id=None, venue="ACL",
                        cited_by_count=cites, doi=None)
    empty_conn.commit()
    db.add_bookmark(empty_conn, "W1", added_at=__import__("datetime").datetime(2020, 1, 1))
    db.add_bookmark(empty_conn, "W2", added_at=__import__("datetime").datetime(2021, 1, 1))
    rows = db.list_bookmarks(empty_conn)
    assert [r["paper_id"] for r in rows] == ["W2", "W1"]   # most-recent first
    assert rows[0]["cited_by_count"] == 9                  # joined paper metadata

    db.remove_bookmark(empty_conn, "W2")
    assert [r["paper_id"] for r in db.list_bookmarks(empty_conn)] == ["W1"]


def test_summary_roundtrip(empty_conn):
    db.insert_paper(empty_conn, openalex_id="W1", title="t", abstract="a",
                    publication_year=2020, venue_id=None, venue="ACL",
                    cited_by_count=0, doi=None)
    empty_conn.commit()
    assert db.get_summary(empty_conn, "W1") is None
    db.save_summary(empty_conn, "W1", ["b1", "b2", "b3"], model="qwen")
    assert db.get_summary(empty_conn, "W1") == ["b1", "b2", "b3"]


def test_year_bounds(empty_conn):
    assert db.year_bounds(empty_conn) is None          # empty corpus
    for pid, yr in (("W1", 2019), ("W2", 2024)):
        db.insert_paper(empty_conn, openalex_id=pid, title=pid, abstract="a",
                        publication_year=yr, venue_id=None, venue="ACL",
                        cited_by_count=0, doi=None)
    empty_conn.commit()
    assert db.year_bounds(empty_conn) == (2019, 2024)


def test_papers_by_ids_empty(empty_conn):
    assert db.papers_by_ids(empty_conn, []) == {}


def test_reset_corpus_fk_safe(seeded_conn):
    db.reset_corpus(seeded_conn)
    for table in ("papers", "concepts", "citation_edges", "review_queue"):
        n = seeded_conn.execute(f"SELECT COUNT(*) c FROM {table}").fetchone()["c"]
        assert n == 0
