"""nDCG@10 retrieval-quality eval against the real ACL/EMNLP/COLING/NAACL eval
corpus (Backlog PBI 8 / Task 8.2), run against `data/eval.db` + Qdrant
collection `axiom_eval_v1` (OD11) -- built by `scripts/ingest_eval_corpus.py`.

Relevance judgments in eval/ndcg_queries.json are SINGLE-PASS, AI-DRAFTED
(see that file's "_provenance" field and docs/DECISIONS.md OD11) -- the
backlog's acceptance criterion ("2 annotators") is not yet met, so the numbers
here are a working demo signal, not a validated quality claim.

Usage (from repo root):
    python scripts/eval_ndcg.py                 # hybrid retrieval (default)
    python scripts/eval_ndcg.py --mode dense
    python scripts/eval_ndcg.py --report eval/report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from math import log2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axiom import config
from axiom.embed import Specter2Encoder
from axiom.qdrant_client import AxiomQdrant, SearchHit

QUERIES_PATH = Path(__file__).resolve().parent.parent / "eval" / "ndcg_queries.json"


def dcg(relevances: list[int]) -> float:
    return sum(rel / log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(ranked_relevances: list[int], all_judged_relevances: list[int], k: int = 10) -> float:
    """ranked_relevances: judged relevance (0 if unjudged) of the top-k retrieved hits.

    all_judged_relevances: every judgment for this query, used to build the
    ideal ranking (ties broken by insertion order, ideal DCG uses top-k of the
    sorted-descending judgments same as standard nDCG@k).
    """
    actual = dcg(ranked_relevances[:k])
    ideal = dcg(sorted(all_judged_relevances, reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def run_query(mode: str, query: str, encoder: Specter2Encoder,
              store: AxiomQdrant, top_k: int) -> list[SearchHit]:
    if mode == "dense":
        vec = encoder.encode_query(query)
        return store.search(vec, top_k=top_k)
    if mode == "hybrid":
        vec = encoder.encode_query(query)
        return store.search_hybrid(query_vector=vec, query_text=query, top_k=top_k)
    raise ValueError(f"unknown mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dense", "hybrid"], default="hybrid")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--report", type=str, default=None,
                    help="write a markdown report to this path (e.g. eval/report.md)")
    args = ap.parse_args()

    data = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    queries = data["queries"]

    encoder = Specter2Encoder()
    store = AxiomQdrant(collection=config.EVAL_COLLECTION_NAME)
    if store.count() == 0:
        raise SystemExit(
            f"'{config.EVAL_COLLECTION_NAME}' is empty -- run "
            f"scripts/ingest_eval_corpus.py first."
        )

    rows = []
    for q in queries:
        judgments: dict[str, int] = q["judgments"]
        hits = run_query(args.mode, q["query"], encoder, store, args.k)
        ranked_rel = [judgments.get(h.paper_id, 0) for h in hits]
        score = ndcg_at_k(ranked_rel, list(judgments.values()), k=args.k)
        rows.append((q["query"], score, hits))
        print(f"nDCG@{args.k} = {score:.3f}   {q['query']!r}")

    mean_ndcg = sum(r[1] for r in rows) / len(rows) if rows else 0.0
    print(f"\nMean nDCG@{args.k} over {len(rows)} queries "
          f"(mode={args.mode}, collection={config.EVAL_COLLECTION_NAME}): {mean_ndcg:.3f}")

    if args.report:
        report_path = Path(args.report)
        lines = [
            "# nDCG@10 retrieval eval report",
            "",
            f"Generated {datetime.now(timezone.utc).isoformat()} · mode=`{args.mode}` · "
            f"collection=`{config.EVAL_COLLECTION_NAME}` · "
            f"corpus={store.count()} papers · {len(rows)} queries",
            "",
            "> **Caveat:** relevance judgments in `eval/ndcg_queries.json` are "
            "single-pass, AI-drafted (see its `_provenance` field and "
            "`docs/DECISIONS.md` OD11) -- not the backlog's '2 annotators' "
            "acceptance criterion. Treat this as a working demo signal, not a "
            "validated quality claim. A low score can also mean the judgment "
            "set didn't happen to cover a paper the retriever correctly "
            "surfaced (judgments were built via keyword-pattern search over "
            "the sample, not an exhaustive relevance pass) -- inspect low "
            "scorers before treating them as retrieval failures.",
            "",
            f"**Mean nDCG@{args.k}: {mean_ndcg:.3f}**",
            "",
            "| Query | nDCG@10 |",
            "|---|---|",
        ]
        for query, score, _ in rows:
            lines.append(f"| {query} | {score:.3f} |")
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n[report] wrote {report_path}")


if __name__ == "__main__":
    main()
