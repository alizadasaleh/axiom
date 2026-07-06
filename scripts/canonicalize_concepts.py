"""Run LLM keyword canonicalization (Task 3.2, OD14) over the corpus in
data/axiom.db, persisting concept -> canonical mappings so the OD10 velocity
engine merges synonyms (e.g. "LoRA" / "Low-Rank Adaptation") into one trend.

Usage (from repo root, needs `ollama serve` running locally):
    python scripts/canonicalize_concepts.py
    python scripts/canonicalize_concepts.py --model llama3.1:latest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axiom import canonicalize, config, db


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.OLLAMA_MODEL)
    ap.add_argument("--batch-size", type=int, default=50)
    args = ap.parse_args()

    conn = db.connect()
    try:
        canonicalize.run(conn, batch_size=args.batch_size, model=args.model)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
