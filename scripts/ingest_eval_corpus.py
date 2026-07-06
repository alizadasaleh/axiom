"""Ingest a stratified sample of the real ACL/EMNLP/COLING/NAACL corpus into a
SEPARATE eval SQLite db + Qdrant collection (OD11) — used only for PBI 8's
nDCG@10 retrieval eval (Task 8.2), never mixed into the gaps/velocity corpus
(data/axiom.db, collection axiom_v1).

Source: data/FULL_DATASET.jsonl, pulled from the team's `main` branch
(23,002 real papers, 2020-2025). It has no citation edges and no
concept/keyword tags, so it can't feed OD9 gap detection or the OD10 velocity
engine — only the title/abstract/venue/year/citation-count fields retrieval
eval actually needs.

Usage (from repo root):
    python scripts/ingest_eval_corpus.py --sample-size 1500
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axiom import config, db
from axiom.indexer import reindex_qdrant


def load_records(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def stratified_sample(records: list[dict], target: int, seed: int) -> list[dict]:
    """Even sample across (venue, year) buckets so no single slice dominates."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        buckets[(r.get("venue"), r.get("year"))].append(r)

    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    per_bucket = max(1, target // len(buckets)) if buckets else 0
    sample: list[dict] = []
    for bucket in buckets.values():
        sample.extend(bucket[:per_bucket])
    if len(sample) < target:
        leftovers = [r for bucket in buckets.values() for r in bucket[per_bucket:]]
        rng.shuffle(leftovers)
        sample.extend(leftovers[: target - len(sample)])
    rng.shuffle(sample)
    return sample[:target]


def load_sqlite(records: list[dict]) -> None:
    conn = db.connect(config.EVAL_DB_PATH)
    try:
        db.init_db(conn)
        db.reset_corpus(conn)

        for r in records:
            pid = r["id"]
            db.insert_paper(
                conn,
                openalex_id=pid,
                title=r.get("title"),
                abstract=r.get("abstract"),
                publication_year=r.get("year"),
                venue_id=None,
                venue=r.get("venue"),
                cited_by_count=r.get("gs_citation") or 0,
                doi=None,
            )
            db.insert_provenance(
                conn, paper_id=pid, source="acl_anthology_full_dataset",
                abstract=r.get("abstract"), has_fulltext=False,
            )
        conn.commit()
        print(f"[sqlite] wrote {len(records)} papers to {config.EVAL_DB_PATH}")
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=config.EVAL_SAMPLE_SIZE)
    ap.add_argument("--seed", type=int, default=42,
                    help="fixed seed so the sample (and downstream judgments) reproduce")
    args = ap.parse_args()

    if not config.EVAL_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{config.EVAL_DATASET_PATH} not found — pull it from origin/main:\n"
            f"  git show origin/main:FULL_DATASET.jsonl > {config.EVAL_DATASET_PATH}"
        )

    print(f"[read] loading {config.EVAL_DATASET_PATH}...")
    records = [r for r in load_records(config.EVAL_DATASET_PATH)
               if r.get("title") and r.get("abstract")]
    print(f"[read] {len(records)} records with title+abstract")

    sample = stratified_sample(records, args.sample_size, args.seed)
    print(f"[sample] {len(sample)} papers, stratified across venue/year "
          f"(seed={args.seed})")

    load_sqlite(sample)
    conn = db.connect(config.EVAL_DB_PATH)
    count = reindex_qdrant(conn, collection=config.EVAL_COLLECTION_NAME)
    conn.close()
    print(f"\nEval corpus ready: {count} points in '{config.EVAL_COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
