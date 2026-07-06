"""Ingest a real corpus from OpenAlex into SQLite (+ optionally Qdrant).

Snowball / 2-hop expansion from a seed topic (decision OD7), writing through the
frozen db.insert_* helpers — no schema change. Then re-embeds into Qdrant via the
shared indexer.

    # full run (fetch + embed), needs Python 3.10-3.12 for torch:
    python scripts/ingest_openalex.py --topic "retrieval-augmented generation"

    # fetch only (runs on any Python, e.g. the 3.14 dev env):
    python scripts/ingest_openalex.py --no-index

The fetch step is pure-Python (httpx). Only the --index step (default on) needs
the ML stack; if torch isn't importable we skip it with a clear message instead
of crashing, so the SQLite corpus is still produced.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axiom import config, db, ingest


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest OpenAlex -> SQLite (+Qdrant).")
    parser.add_argument("--topic", default=config.SEED_QUERY,
                        help="seed search query (default: %(default)r)")
    parser.add_argument("--target", type=int, default=config.CORPUS_TARGET,
                        help="approx. corpus size to snowball to (default: %(default)s)")
    parser.add_argument("--seeds", type=int, default=config.SEED_COUNT,
                        help="number of most-cited seed papers (default: %(default)s)")
    parser.add_argument("--no-index", action="store_true",
                        help="fetch + write SQLite only; skip embedding into Qdrant")
    parser.add_argument("--all-venues", action="store_true",
                        help="disable the NLP/ML venue allow-list "
                             "(config.RELEVANT_VENUE_PATTERNS); admit every venue")
    args = parser.parse_args()

    keep = None if args.all_venues else ingest.make_venue_filter(
        config.RELEVANT_VENUE_PATTERNS)
    if keep is not None:
        print(f"[ingest] venue filter ON — keeping only "
              f"{config.RELEVANT_VENUE_PATTERNS}")

    with ingest.OpenAlexClient() as client:
        works = ingest.snowball(
            client,
            seed_query=args.topic,
            target=args.target,
            seed_count=args.seeds,
            keep=keep,
        )

    conn = db.connect()
    try:
        n = ingest.ingest_to_sqlite(conn, works.values())
    finally:
        conn.close()

    if args.no_index:
        print(f"\nFetched {n} papers into SQLite (no indexing). "
              f"Run without --no-index on Python 3.10-3.12 to embed into Qdrant.")
        return

    # torch/transformers are imported lazily inside reindex_qdrant, so an
    # unavailable ML stack (e.g. the 3.14 dev env) surfaces here as ImportError.
    try:
        from axiom.indexer import reindex_qdrant
        reindex_qdrant()
    except ImportError as exc:
        print(f"\n[skip] Qdrant indexing unavailable ({exc}). "
              f"SQLite corpus is ready; re-run on Python 3.10-3.12 (torch) to embed.")
        return

    print("\nIngestion complete. Run:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
