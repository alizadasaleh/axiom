# qdrant_ms — Lightweight Qdrant Prototype

A **standalone** Streamlit + Qdrant prototype for exploring the paper corpus by
raw semantic proximity. It is intentionally separate from the main Axiom
platform (`app/streamlit_app.py`):

|                | `qdrant_ms/` (this app)              | `app/streamlit_app.py` (platform) |
|----------------|--------------------------------------|-----------------------------------|
| Embeddings     | `all-MiniLM-L6-v2` (384-d)           | SPECTER2 (`axiom/embed.py`)       |
| Collection     | `academic_papers`                    | axiom platform collection         |
| Data source    | raw `paperlists/*.json`              | OpenAlex → SQLite → Qdrant        |
| Purpose        | quick proximity/gap prototype        | full trends + gaps product        |

It shares only the running Qdrant container (`docker-compose.yml`, port 6333) —
no Python code is shared, so the two apps can run independently.

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

From the **repo root**:

```bash
# 1. Start Qdrant (shared with the main platform)
docker compose up -d qdrant

# 2. Ingest the raw paperlists into the `academic_papers` collection
python qdrant_ms/scripts/ingest_papers.py --root_dir ./paperlists

# 3. Launch the Streamlit prototype
streamlit run qdrant_ms/Home.py
```

CLI utilities (after ingestion):

```bash
python qdrant_ms/scripts/search_papers.py
python qdrant_ms/scripts/compare_papers.py --title1 "..." --title2 "..."
python qdrant_ms/scripts/nebula.py
```

## Features

- **Proximity Analyzer** — pick two papers, see cosine similarity/distance and a
  2-D PCA projection of their vectors.
- **Gap Discovery Feed** — samples paper pairs, computes their vector midpoint,
  and flags "voids" where no existing paper sits near that midpoint.
