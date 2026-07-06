"""Tests for axiom.sparse — tokenization, stable hashing, BM25 dot-product property."""
from __future__ import annotations

import math

from axiom import sparse


def test_tokenize_drops_stopwords_and_short():
    toks = sparse.tokenize("The LoRA a of RAG")
    assert "the" not in toks and "a" not in toks and "of" not in toks
    assert "lora" in toks and "rag" in toks


def test_term_index_stable_and_in_range():
    i1 = sparse.term_index("retrieval")
    i2 = sparse.term_index("retrieval")
    assert i1 == i2                            # process-independent
    assert 0 <= i1 < sparse.config.SPARSE_VOCAB_SIZE


def test_query_dot_document_equals_bm25():
    docs = ["retrieval augmented generation", "neural machine translation",
            "low rank adaptation retrieval"]
    enc = sparse.BM25SparseEncoder(docs)
    d_idx, d_val = enc.encode_document(docs[0])
    q_idx, q_val = sparse.encode_query_sparse("retrieval generation")
    doc_map = dict(zip(d_idx, d_val))
    dot = sum(v * doc_map.get(i, 0.0) for i, v in zip(q_idx, q_val))

    # Reference BM25 over the same query terms.
    expected = 0.0
    for term in {"retrieval", "generation"}:
        idf = enc.idf.get(term, 0.0)
        tf = docs[0].split().count(term)
        dl = len(sparse.tokenize(docs[0]))
        denom = tf + enc.k1 * (1 - enc.b + enc.b * dl / enc.avgdl)
        if idf > 0 and tf:
            expected += idf * (tf * (enc.k1 + 1)) / denom
    assert math.isclose(dot, expected, rel_tol=1e-9)


def test_empty_document():
    enc = sparse.BM25SparseEncoder(["only stopwords the a of"])
    assert enc.encode_document("the a of") == ([], [])
