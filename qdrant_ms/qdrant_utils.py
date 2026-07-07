"""Shared Qdrant helpers for the qdrant_ms Streamlit prototype.

Targets the live `academic_papers` collection as populated by the Axiom
platform pipeline: NAMED vectors (`dense` 768-d SPECTER2 + `sparse`) and the
payload schema {paper_id, title, year, venue, cited_by_count, concepts}.

The pages only ever *read* stored vectors (they never embed free text), so no
embedding model is loaded here. Queries use `client.search` with a NamedVector
because the pinned qdrant-client (1.9.0) has no Query API (`query_points`).
"""
import numpy as np
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import NamedVector

COLLECTION_NAME = "academic_papers"
DENSE_VECTOR_NAME = "dense"


@st.cache_resource
def init_connections():
    """Cache and return the Qdrant client (no embedding model needed)."""
    return QdrantClient(host="localhost", port=6333)


def extract_vector(record):
    """Return the dense embedding as a plain float list from any Qdrant object.

    Points here carry named vectors, so `.vector` is a dict like
    {"dense": [...], "sparse": SparseVector}. Falls back gracefully to a bare
    list (unnamed) and returns None when no dense vector is present.
    """
    if record is None:
        return None
    vec = getattr(record, "vector", None)
    if isinstance(vec, dict):
        dense = vec.get(DENSE_VECTOR_NAME)
        if dense is None:  # unexpected naming — take the first list-like value
            dense = next((v for v in vec.values() if isinstance(v, (list, tuple, np.ndarray))), None)
        return list(dense) if dense is not None else None
    if isinstance(vec, (list, tuple, np.ndarray)):
        return list(vec)
    return None


def concepts_list(payload):
    """Normalise the `concepts` payload field to a clean list of strings."""
    c = payload.get("concepts", []) if payload else []
    if isinstance(c, list):
        return [str(x).strip() for x in c if str(x).strip()]
    if isinstance(c, str):
        return [x.strip() for x in c.split(",") if x.strip()]
    return []


@st.cache_data(ttl=300)
def get_all_paper_titles(_client):
    """Scroll payloads to pull all titles for the searchable dropdown."""
    try:
        records, _ = _client.scroll(
            collection_name=COLLECTION_NAME,
            limit=5000,
            with_payload=["title"],
            with_vectors=False,
        )
        return sorted({r.payload["title"] for r in records if r.payload and r.payload.get("title")})
    except Exception as e:
        st.error(f"Error fetching titles: {e}")
        return []


def fetch_paper_by_title(_client, title_string):
    """Retrieve the full record (dense vector + payload) for an exact title."""
    records, _ = _client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter={"must": [{"key": "title", "match": {"value": title_string}}]},
        limit=1,
        with_vectors=True,
        with_payload=True,
    )
    return records[0] if records else None


@st.cache_data(ttl=600, show_spinner="Loading corpus vectors from Qdrant…")
def load_corpus(_client, max_points=20000):
    """Scroll the whole collection once; return dense vectors + payloads.

    Cached so the clustering page doesn't re-download every rerun. Returns:
        ids, vectors (np.ndarray N×768), titles, payloads.
    """
    ids, vectors, titles, payloads = [], [], [], []
    offset = None
    while len(ids) < max_points:
        records, offset = _client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        if not records:
            break
        for r in records:
            vec = extract_vector(r)
            if vec is None:
                continue
            ids.append(r.id)
            vectors.append(vec)
            payload = r.payload or {}
            titles.append(payload.get("title", "Untitled"))
            payloads.append(payload)
        if offset is None:
            break
    return {
        "ids": ids,
        "vectors": np.asarray(vectors, dtype=np.float32),
        "titles": titles,
        "payloads": payloads,
    }


def nearest_to_vector(client, vector, limit=5, exclude_titles=None, with_vectors=True):
    """Dense nearest neighbours of an arbitrary point in vector space.

    Uses `client.search` with a NamedVector (dense). Results carry `.score`
    (cosine similarity), `.payload`, and — when `with_vectors` — `.vector`.
    Titles in `exclude_titles` are skipped (drops the anchor papers themselves).
    """
    exclude_titles = set(exclude_titles or [])
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=NamedVector(name=DENSE_VECTOR_NAME, vector=vector),
        limit=limit + len(exclude_titles),
        with_payload=True,
        with_vectors=with_vectors,
    )
    out = []
    for h in hits:
        if h.payload and h.payload.get("title") in exclude_titles:
            continue
        out.append(h)
        if len(out) >= limit:
            break
    return out


def cosine_similarity(v1, v2):
    """Cosine similarity between two vectors (lists or arrays)."""
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / denom) if denom else 0.0
