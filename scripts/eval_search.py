"""Search-quality eval harness for Axiom P1.

Runs a fixed query set and prints the top-5 (score, title, year, venue) per
query, for one or more retrieval MODES so improvements are demonstrable rather
than eyeballed.

Modes:
    raw        Dense search with the query embedded as a *document* (the
               pre-upgrade behavior — terse queries land in a mushy region).
    expansion  Dense search with encode_query()'s abstract-shaped wrapping
               (Step 2). For "adapter" query paths this is the adapter vector.
    hybrid     Dense + sparse (BM25-style) fused with RRF (Step 3). Falls back
               with a clear message if the hybrid index isn't built yet.

Usage (from repo root):
    python scripts/eval_search.py                       # raw vs expansion vs hybrid
    python scripts/eval_search.py --modes raw expansion
    python scripts/eval_search.py --check               # assert acceptance criteria
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axiom import config, db
from axiom.embed import Specter2Encoder
from axiom.qdrant_client import AxiomQdrant

# 5 full-phrased demo queries + 5 terse/awkward ones that currently fail.
QUERIES = [
    "reducing hallucination in retrieval-augmented generation",
    "parameter efficient fine-tuning with low-rank adapters",
    "low-resource machine translation with limited parallel data",
    "evaluating language models beyond accuracy",
    "chain-of-thought reasoning and self-consistency",
    "TOOLS for AI",
    "LoRA",
    "RAG",
    "how to make models hallucinate less",
    "cheap finetuning",
]

ALL_MODES = ["raw", "expansion", "hybrid"]

# Acceptance criteria: (query, expected substring in the top-1 title). These are
# asserted via `--check` with the DEFAULT full-span year filter active — the
# exact scenario where zero-score sparse hits used to dilute RRF and drop the
# right paper. Guards that regression.
CHECKS = [
    ("LoRA", "LoRA: Low-Rank"),
    ("RAG", "Retrieval-Augmented Generation for"),
    ("reducing hallucination in retrieval-augmented generation", "Surveying Hallucination"),
    ("low-resource machine translation with limited parallel data", "Back-Translation"),
    ("evaluating language models beyond accuracy", "Beyond Accuracy"),
    ("chain-of-thought reasoning and self-consistency", "Self-Consistency"),
]


def run_mode(mode: str, query: str, encoder: Specter2Encoder,
             store: AxiomQdrant, top_k: int = 5):
    if mode == "raw":
        # Embed the query as if it were a document (no expansion template).
        vec = encoder.encode([query], abstracts=None)[0]
        return store.search(vec, top_k=top_k)
    if mode == "expansion":
        vec = encoder.encode_query(query)
        return store.search(vec, top_k=top_k)
    if mode == "hybrid":
        vec = encoder.encode_query(query)
        return store.search_hybrid(query_vector=vec, query_text=query, top_k=top_k)
    raise ValueError(f"unknown mode: {mode}")


def fmt_hits(hits) -> str:
    if not hits:
        return "      (no hits)"
    lines = []
    for h in hits:
        lines.append(f"      {h.score:6.3f}  [{h.year} {h.venue:<7}] {h.title}")
    spread = max(h.score for h in hits) - min(h.score for h in hits)
    lines.append(f"      score spread (top-{len(hits)}): {spread:.3f}")
    return "\n".join(lines)


def run_checks(encoder: Specter2Encoder, store: AxiomQdrant) -> int:
    """Assert acceptance criteria for hybrid search WITH the default year filter.

    Returns a process exit code (0 = all passed). Suitable for CI.
    """
    if not store.supports_hybrid():
        print("[check] FAIL: hybrid index not built — run scripts/bootstrap_synthetic.py")
        return 1
    # Replicate the app's default filter: full year span, all venues.
    conn = db.connect()
    try:
        year_range = db.year_bounds(conn)
    finally:
        conn.close()

    failures = 0
    print(f"[check] hybrid + year filter {year_range} (the app's default)\n")
    for query, expected in CHECKS:
        vec = encoder.encode_query(query)
        hits = store.search_hybrid(
            query_vector=vec, query_text=query, top_k=5, year_range=year_range
        )
        top = hits[0].title if hits else "(no hits)"
        ok = bool(hits) and expected in top
        print(f"  [{'PASS' if ok else 'FAIL'}] {query!r}\n"
              f"         want top-1 ~ {expected!r}; got {top!r}")
        failures += 0 if ok else 1

    print(f"\n[check] {len(CHECKS) - failures}/{len(CHECKS)} passed")
    return 1 if failures else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=ALL_MODES,
                    choices=ALL_MODES, help="which retrieval modes to compare")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--check", action="store_true",
                    help="assert acceptance criteria (exit 1 on failure) and quit")
    args = ap.parse_args()

    encoder = Specter2Encoder()
    store = AxiomQdrant()

    if args.check:
        sys.exit(run_checks(encoder, store))

    # Drop hybrid gracefully if the index has no sparse vectors yet.
    modes = list(args.modes)
    if "hybrid" in modes and not store.supports_hybrid():
        print("[eval] hybrid index not built (no sparse vectors) — skipping "
              "'hybrid'. Re-run scripts/bootstrap_synthetic.py after Step 3.\n")
        modes = [m for m in modes if m != "hybrid"]

    print(f"[eval] collection={config.COLLECTION_NAME} points={store.count()} "
          f"query_path={encoder.query_path} modes={modes}\n")

    for q in QUERIES:
        print(f"Q: {q!r}")
        for mode in modes:
            print(f"   [{mode}]")
            print(fmt_hits(run_mode(mode, q, encoder, store, args.top_k)))
        print()


if __name__ == "__main__":
    main()
