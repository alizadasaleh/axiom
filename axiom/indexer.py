"""Build the Qdrant index (dense SPECTER2 + sparse BM25) from SQLite.

Single source of truth for the embed/upsert path, shared by the synthetic
bootstrap and the OpenAlex ingestion CLI so the payload + vector logic lives in
exactly one place.

The heavy ML imports (torch/transformers via axiom.embed) are LAZY — done inside
the function — so importing this module is cheap and the pure-Python fetch path
(axiom.ingest) never drags in the ML stack. Calling reindex_qdrant() itself does
require torch, i.e. Python 3.10-3.12 (the 3.14 dev env can't install the pins).
"""
from __future__ import annotations

import sqlite3

from axiom import config, db


def reindex_qdrant(conn: sqlite3.Connection | None = None, *,
                    collection: str | None = None, log=print) -> int:
    """Encode every paper in SQLite and (re)load the Qdrant collection.

    `collection` overrides `config.COLLECTION_NAME` (used by the eval-corpus
    ingest, OD11, to target a separate collection without touching axiom_v1).
    Returns the number of points in the collection afterwards. Recreates the
    collection (idempotent wipe) so dense + sparse vectors always agree with the
    current SQLite corpus.
    """
    # Lazy: only pay the torch import cost when we actually index.
    from axiom.embed import Specter2Encoder
    from axiom.qdrant_client import AxiomQdrant
    from axiom.sparse import BM25SparseEncoder

    own_conn = conn is None
    if own_conn:
        conn = db.connect()
    try:
        rows = db.iter_papers(conn)
        concept_rows = conn.execute(
            "SELECT paper_id, concept FROM concepts"
        ).fetchall()
    finally:
        if own_conn:
            conn.close()

    concepts_by_paper: dict[str, list[str]] = {}
    for r in concept_rows:
        concepts_by_paper.setdefault(r["paper_id"], []).append(r["concept"])

    log(f"[embed] loading {config.MODEL_ID} (first run downloads weights)...")
    encoder = Specter2Encoder()
    titles = [r["title"] for r in rows]
    abstracts = [r["abstract"] for r in rows]
    log(f"[embed] encoding {len(rows)} abstracts on device={encoder.device}...")
    vectors = encoder.encode(titles, abstracts)

    # Sparse BM25-style vectors over title + abstract + concepts so exact terms
    # and acronyms (LoRA, RAG) match in the hybrid arm.
    sparse_docs = [
        " ".join([
            r["title"] or "",
            r["abstract"] or "",
            " ".join(concepts_by_paper.get(r["openalex_id"], [])),
        ])
        for r in rows
    ]
    bm25 = BM25SparseEncoder(sparse_docs)
    sparse_vectors = [bm25.encode_document(doc) for doc in sparse_docs]
    log(f"[sparse] built BM25 vectors (avgdl={bm25.avgdl:.1f}, "
        f"vocab={len(bm25.idf)} terms)")

    payloads = [
        {
            "paper_id": r["openalex_id"],
            "title": r["title"],
            "year": r["publication_year"],
            "venue": r["venue"],
            "cited_by_count": r["cited_by_count"],
            "concepts": concepts_by_paper.get(r["openalex_id"], []),
        }
        for r in rows
    ]

    store = AxiomQdrant(collection=collection)
    log(f"[qdrant] recreating collection '{store.collection}'...")
    store.recreate_collection()  # idempotent wipe + recreate
    store.upsert_papers(payloads, vectors, sparse_vectors=sparse_vectors)
    count = store.count()
    log(f"[qdrant] upserted {count} points into '{store.collection}' (dense + sparse)")
    return count
