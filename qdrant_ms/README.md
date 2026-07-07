# qdrant_ms — Lightweight Qdrant Prototype

> **Integration status:** the pages in this directory are now available as the
> **🔗 Vector tools** tab inside the main UI (`streamlit run app/streamlit_app.py`).
> The standalone entry point below is retained for development and debugging.
> Both the standalone app and the main UI read the same `axiom_v1` collection.

A focused Streamlit + Qdrant app for exploring the paper corpus by semantic
proximity. It reads the **live `axiom_v1` collection** as populated by the
Axiom platform pipeline, so it works directly on the real data:

| Aspect       | Value                                                         |
|--------------|---------------------------------------------------------------|
| Collection   | `axiom_v1` (unified with main platform; was `academic_papers`) |
| Vectors      | **named**: `dense` (768-d SPECTER2) + `sparse` (BM25-style)   |
| Payload      | `paper_id, title, year, venue, cited_by_count, concepts`      |
| This app uses | the `dense` vector only (semantic proximity)                 |

The pages only ever **read stored vectors** — they never embed free text — so no
embedding model is loaded in `qdrant_utils.py`. Queries use `client.search` with
a `NamedVector` (the pinned qdrant-client 1.9.0 has no Query API).

> **Legacy note:** `scripts/ingest_papers.py` / `search_papers.py` / `nebula.py`
> predate this and assume an *unnamed* 384-d MiniLM collection built from raw
> `paperlists/*.json`. They do **not** match the current `academic_papers`
> schema and are kept only for reference. The Streamlit pages and
> `qdrant_utils.py` are the maintained path.

## Layout

```
qdrant_ms/
├── Home.py                    # Streamlit entrypoint (landing page)
├── qdrant_utils.py            # shared Qdrant client + embedding model helpers
├── pages/                     # Streamlit multipage nav (auto-discovered)
│   ├── 1_Proximity_Analyzer.py
│   └── 2_Gap_Discovery_Feed.py
└── scripts/                   # standalone CLI utilities (not Streamlit pages)
    ├── ingest_papers.py       # build the `academic_papers` collection
    ├── search_papers.py       # quick semantic query from the terminal
    ├── compare_papers.py      # cosine similarity between two titles
    └── nebula.py              # t-SNE + HDBSCAN "nebula" map → nebula_map.html
```

> `qdrant_utils.py` lives at the app root (not in `pages/`) on purpose: Streamlit
> turns *every* file in `pages/` into a navigation entry, so shared helpers must
> stay outside it.

## Run

From the **repo root**, with the `axiom_v1` collection already populated:

```bash
# 1. Start Qdrant (shared with the main platform)
docker compose up -d qdrant

# 2. Launch the Streamlit app
streamlit run qdrant_ms/Home.py
```

## Features

- **Paper Similarity Analyzer** — pick two papers and get a cosine-similarity
  score with a plain-language verdict, a **neighbourhood map** (each paper's
  nearest work projected together so the score has context), the **bridge papers**
  that currently sit at their midpoint, and shared concepts.
- **Gap Discovery** — a corpus-wide, deterministic scan. It clusters every paper
  into topics (KMeans, labelled by their top `concepts`), then for each
  *bridgeable* pair of topics checks whether any real paper is close to **both**
  topics at once. If the best candidate barely beats the trivial "sits on one
  topic's edge" baseline, the two topics are effectively unconnected — a gap.
  Gaps are ranked and shown on a landscape map.

  > Why not random paper pairs? The midpoint of two *unrelated* papers isn't a
  > meaningful concept, so random sampling surfaces noise and never covers the
  > whole set. Clustering first makes gaps reproducible and explainable ("nothing
  > bridges Topic A and Topic B").
