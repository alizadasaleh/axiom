"""Export OD9/OD17 gap candidates to a CSV for human labeling (Backlog Task 8.1).

Step 1 of the calibration scaffold: runs the real gap analysis over the live
corpus (data/axiom.db + Qdrant axiom_v1) and writes one row per gap candidate
with its component scores and an EMPTY `label` column for a human to fill:

    label = 1  → a genuine, thesis-worthy gap
    label = 0  → not a real/interesting gap

Then a human fills `label`, and scripts/calibrate_gap_thresholds.py turns the
labeled file into eval/calibration.json (τ + weights) that gaps.py consumes.

Requires the ML stack + a running Qdrant (uses SPECTER2 vectors), so run it in
the project .venv (Python 3.10-3.12), not the bare test env.

Usage (from repo root):
    python scripts/export_gap_labels.py
    python scripts/export_gap_labels.py --out eval/labels/gap_candidates.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axiom import config, db
from axiom import gaps as gapsmod
from axiom import graph as graphmod
from axiom.qdrant_client import AxiomQdrant

_HEADER_COMMENT = (
    "# Gap candidates for τ-calibration (OD17, Task 8.1).\n"
    "# Fill the `label` column: 1 = genuine thesis-worthy gap, 0 = not.\n"
    "# S/D/V/A are the normalized G-score components; g_score is their weighted blend.\n"
    "# Then: python scripts/calibrate_gap_thresholds.py --labels <this file>\n"
)

_FIELDS = ["candidate_id", "a_label", "b_label", "a_size", "b_size",
           "semantic_similarity", "inter_citations", "S", "D", "V", "A",
           "g_score", "gap_score", "label"]


def candidate_id(a_label: str, b_label: str) -> str:
    """Stable id for a gap pair, order-independent (sha1 of the sorted labels)."""
    key = "||".join(sorted((a_label, b_label)))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def build_rows(analysis: gapsmod.GapAnalysis) -> list[dict]:
    rows = []
    for c in analysis.gaps:
        comp = c.components
        rows.append({
            "candidate_id": candidate_id(c.a.label, c.b.label),
            "a_label": c.a.label, "b_label": c.b.label,
            "a_size": c.a.size, "b_size": c.b.size,
            "semantic_similarity": round(c.semantic_similarity, 4),
            "inter_citations": c.inter_citations,
            "S": round(comp.get("similarity", 0.0), 4),
            "D": round(comp.get("disconnection", 0.0), 4),
            "V": round(comp.get("velocity", 0.0), 4),
            "A": round(comp.get("authority", 0.0), 4),
            "g_score": round(c.g_score, 4),
            "gap_score": round(c.gap_score, 4),
            "label": "",                       # human fills this
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write(_HEADER_COMMENT)
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("eval/labels/gap_candidates.csv"))
    ap.add_argument("--top-k", type=int, default=50,
                    help="max gap candidates to export for labeling")
    args = ap.parse_args(argv)

    conn = db.connect()
    try:
        g = graphmod.load_graph(conn)
        vectors = AxiomQdrant().fetch_dense_vectors()
        analysis = gapsmod.analyze(g, conn, vectors, top_k=args.top_k)
    finally:
        conn.close()

    rows = build_rows(analysis)
    write_csv(args.out, rows)
    print(f"[export] wrote {len(rows)} gap candidates to {args.out} — "
          f"fill the `label` column, then run scripts/calibrate_gap_thresholds.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
