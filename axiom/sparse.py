"""BM25-style sparse encoder for hybrid retrieval.

Pure-Python, zero new dependencies (respects the pinned environment). Produces
Qdrant sparse vectors (`indices`, `values`) so exact terms and acronyms the
dense SPECTER2 vector misses (LoRA, RAG, dataset names) still match.

Design:
- Tokens are mapped to indices with the *hashing trick* (a stable BLAKE2 hash),
  so no vocabulary file has to be persisted or shared between bootstrap and app.
- Document side carries the full BM25 weight (IDF * tf-saturation * length norm).
- Query side carries presence weights (1.0). Because IDF is baked into the doc
  weights, the dot product query·doc == the BM25 score of the document.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Iterable

from axiom import config

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str | None) -> list[str]:
    """Lowercase word/number tokens, dropping 1-char tokens and stopwords.

    Keeps acronyms and model names as single tokens: 'LoRA' -> 'lora',
    'RAG' -> 'rag', 'BM25' -> 'bm25'.
    """
    toks = _TOKEN_RE.findall((text or "").lower())
    return [t for t in toks if len(t) > 1 and t not in config.SPARSE_STOPWORDS]


def term_index(term: str) -> int:
    """Stable, process-independent hash of a term into the sparse index space.

    (Python's builtin hash() is salted per process, so it cannot be used here —
    bootstrap and the app must agree on indices.)
    """
    digest = hashlib.blake2b(term.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % config.SPARSE_VOCAB_SIZE


class BM25SparseEncoder:
    """Fits IDF + average-length over a corpus, then encodes documents."""

    def __init__(self, docs: Iterable[str]):
        self.k1 = config.BM25_K1
        self.b = config.BM25_B
        tokenized = [tokenize(d) for d in docs]
        self.n_docs = len(tokenized)
        self.avgdl = (sum(len(t) for t in tokenized) / self.n_docs) if self.n_docs else 0.0
        df: Counter[str] = Counter()
        for toks in tokenized:
            df.update(set(toks))
        # BM25 idf with +1 inside the log to keep weights non-negative.
        self.idf = {
            term: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))
            for term, n in df.items()
        }

    def encode_document(self, text: str) -> tuple[list[int], list[float]]:
        toks = tokenize(text)
        if not toks:
            return [], []
        tf = Counter(toks)
        dl = len(toks)
        idx_to_val: dict[int, float] = {}
        for term, freq in tf.items():
            idf = self.idf.get(term, 0.0)
            if idf <= 0.0:
                continue
            denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            weight = idf * (freq * (self.k1 + 1)) / denom
            if weight <= 0.0:
                continue
            idx = term_index(term)
            idx_to_val[idx] = idx_to_val.get(idx, 0.0) + weight
        return list(idx_to_val.keys()), list(idx_to_val.values())


def encode_query_sparse(text: str) -> tuple[list[int], list[float]]:
    """Query side: unique terms with presence weight 1.0 (no corpus needed)."""
    idx_to_val: dict[int, float] = {}
    for term in set(tokenize(text)):
        idx_to_val[term_index(term)] = 1.0
    return list(idx_to_val.keys()), list(idx_to_val.values())
