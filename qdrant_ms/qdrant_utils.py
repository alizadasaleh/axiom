import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "academic_papers"

@st.cache_resource
def init_connections():
    """Cache the Qdrant client connection and local embedding model."""
    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return client, model

@st.cache_data(ttl=300)
def get_all_paper_titles(_client):
    """Scroll Qdrant payloads to pull all unique titles for the searchable dropdown options."""
    try:
        # Pulling a healthy limit of papers for interactive dropdown selections
        records, _ = _client.scroll(
            collection_name=COLLECTION_NAME,
            limit=5000,
            with_payload=["title"],
            with_vectors=False
        )
        return sorted(list(set([r.payload["title"] for r in records if r.payload and "title" in r.payload])))
    except Exception as e:
        st.error(f"Error fetching directory titles: {e}")
        return []

def fetch_paper_by_title(_client, title_string):
    """Retrieve full vector and payload metadata matching the chosen dropdown title string."""
    records, _ = _client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter={
            "must": [{"key": "title", "match": {"text": title_string}}]
        },
        limit=1,
        with_vectors=True,
        with_payload=True
    )
    return records[0] if records else None