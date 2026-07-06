"""Tests for axiom.ingest pure helpers — no network."""
from __future__ import annotations

import pytest

from axiom import ingest


def test_short_id_strips_prefix_idempotent():
    assert ingest.short_id("https://openalex.org/W123") == "W123"
    assert ingest.short_id("W123") == "W123"       # idempotent
    assert ingest.short_id(None) is None


def test_short_id_accepts_source_ids():
    # parse_work reuses short_id for the venue's source id (S-prefixed).
    assert ingest.short_id("https://openalex.org/S999") == "S999"


def test_short_id_rejects_malformed():
    with pytest.raises(ValueError):
        ingest.short_id("https://openalex.org/not-an-id")


def test_reconstruct_abstract_orders_by_position():
    inv = {"generation": [2], "Retrieval": [0], "augmented": [1]}
    assert ingest.reconstruct_abstract(inv) == "Retrieval augmented generation"


def test_reconstruct_abstract_none_and_empty():
    assert ingest.reconstruct_abstract(None) is None
    assert ingest.reconstruct_abstract({}) is None


def test_parse_work_normalizes_fields():
    work = {
        "id": "https://openalex.org/W1",
        "title": "A paper",
        "abstract_inverted_index": {"Hello": [0], "world": [1]},
        "publication_year": 2022,
        "primary_location": {"source": {"id": "https://openalex.org/S5", "display_name": "ACL"}},
        "cited_by_count": 7,
        "doi": "10.0/x",
        "concepts": [{"display_name": "NLP", "level": 1}, {"display_name": None, "level": 2}],
        "referenced_works": ["https://openalex.org/W9", "https://openalex.org/W8"],
    }
    parsed = ingest.parse_work(work)
    assert parsed.openalex_id == "W1"
    assert parsed.abstract == "Hello world"
    assert parsed.venue == "ACL" and parsed.venue_id == "S5"
    assert parsed.cited_by_count == 7
    assert parsed.concepts == [("NLP", 1)]          # None display_name dropped
    assert parsed.references == ["W9", "W8"]


def test_parse_work_missing_fields_default():
    parsed = ingest.parse_work({"id": "https://openalex.org/W2"})
    assert parsed.cited_by_count == 0
    assert parsed.concepts == []
    assert parsed.references == []
    assert parsed.venue is None
