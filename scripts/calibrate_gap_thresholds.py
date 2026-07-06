"""Calibrate the OD17 gap G-score threshold τ (and optionally the weights) from
human labels (Backlog PBI 8 / Task 8.1).

This is the consumer end of the calibration scaffold:

    scripts/export_gap_labels.py   →  eval/labels/gap_candidates.csv   (empty `label` column)
    <a human rates each row 1/0>    →  eval/labels/gap_labels.csv
    scripts/calibrate_gap_thresholds.py  →  eval/calibration.json + eval/calibration_report.md

`eval/calibration.json` is then read automatically by `axiom/config.load_calibration()`
(consumed in `axiom/gaps.analyze`), overriding the uncalibrated dev defaults.

A SCRIPT, not a notebook: this repo keeps its eval tooling as scripts that emit
markdown reports (see scripts/eval_ndcg.py), and this one is pure stdlib so it
runs anywhere (no torch/qdrant/pandas), including the Python 3.14 dev box.

Usage (from repo root):
    python scripts/calibrate_gap_thresholds.py --labels eval/labels/gap_labels.csv
    python scripts/calibrate_gap_thresholds.py --labels <csv> --min-precision 0.8
    python scripts/calibrate_gap_thresholds.py --labels <csv> --sweep-weights
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axiom import config

_COMPONENTS = ("similarity", "disconnection", "velocity", "authority")
# CSV column names carrying each normalized component (see export_gap_labels.py).
_COMPONENT_COLS = {"similarity": "S", "disconnection": "D",
                   "velocity": "V", "authority": "A"}


# --- Pure functions (unit-tested) --------------------------------------------
def recompute_g(row: dict, weights: dict) -> float:
    """G-score for one labeled row from its stored S/D/V/A columns + weights.

    Recomputing from components (not reusing the exported g_score) lets a weight
    sweep re-score every row without touching the graph/corpus again.
    """
    return sum(weights[c] * float(row[_COMPONENT_COLS[c]]) for c in _COMPONENTS)


def precision_recall(rows: list[dict], tau: float, weights: dict) -> dict:
    """Precision/recall/F1 treating G >= τ as the positive prediction.

    τ is rounded to 4dp up front so the threshold that's compared here is exactly
    the one reported/persisted — otherwise a sweep value like 0.30000000000000004
    would silently behave as a threshold just above the 0.3 it's reported as.
    """
    tau = round(tau, 4)
    tp = fp = fn = tn = 0
    for r in rows:
        predicted = recompute_g(r, weights) >= tau
        actual = int(r["label"]) == 1
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"tau": round(tau, 4), "precision": precision, "recall": recall,
            "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def sweep_tau(rows: list[dict], weights: dict, step: float = 0.01) -> list[dict]:
    """Precision/recall/F1 across τ ∈ [0, 1] at `step` resolution."""
    points = []
    steps = int(round(1.0 / step))
    for i in range(steps + 1):
        points.append(precision_recall(rows, i * step, weights))
    return points


def best_operating_point(points: list[dict], min_precision: float | None = None) -> dict:
    """Pick the max-F1 point, optionally among those meeting a precision floor."""
    pool = [p for p in points if min_precision is None or p["precision"] >= min_precision]
    if not pool:                               # floor unmet everywhere → ignore it
        pool = points
    # Tie-break toward higher recall, then lower τ, for a stable choice.
    return max(pool, key=lambda p: (p["f1"], p["recall"], -p["tau"]))


def weight_grid(step: float = 0.1) -> list[dict]:
    """All 4-way weight simplexes summing to 1 at `step` resolution."""
    n = int(round(1.0 / step))
    grid = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            for k in range(n + 1 - i - j):
                l = n - i - j - k
                grid.append({"similarity": i / n, "disconnection": j / n,
                             "velocity": k / n, "authority": l / n})
    return grid


def load_labeled_csv(path: Path) -> list[dict]:
    """Rows from the human-labeled CSV, keeping only those with a 0/1 `label`."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(row for row in fh if not row.startswith("#"))
        rows = []
        for r in reader:
            label = (r.get("label") or "").strip()
            if label in ("0", "1"):
                rows.append(r)
    return rows


# --- IO ----------------------------------------------------------------------
def write_calibration(path: Path, *, tau: float, weights: dict, stats: dict,
                      source_csv: Path, labeled_n: int, positives: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tau": round(tau, 4),
        "weights": {k: round(v, 4) for k, v in weights.items()},
        "labeled_n": labeled_n,
        "positives": positives,
        "precision": round(stats["precision"], 4),
        "recall": round(stats["recall"], 4),
        "f1": round(stats["f1"], 4),
        "source_csv": str(source_csv),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, *, points: list[dict], chosen: dict, weights: dict,
                 labeled_n: int, positives: int, source_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Gap G-score threshold calibration (OD17, Task 8.1)",
        "",
        f"- Source labels: `{source_csv}` ({labeled_n} labeled, {positives} positive)",
        f"- Weights: {', '.join(f'{k}={v:.2f}' for k, v in weights.items())}",
        f"- **Chosen τ = {chosen['tau']:.2f}** "
        f"(precision {chosen['precision']:.3f}, recall {chosen['recall']:.3f}, "
        f"F1 {chosen['f1']:.3f})",
        "",
        "| τ | precision | recall | F1 | tp | fp | fn |",
        "|---|---|---|---|---|---|---|",
    ]
    # Report every 0.05 to keep the table readable.
    for p in points:
        if abs((p["tau"] / 0.05) - round(p["tau"] / 0.05)) < 1e-9:
            mark = " ⬅" if abs(p["tau"] - chosen["tau"]) < 1e-9 else ""
            lines.append(f"| {p['tau']:.2f}{mark} | {p['precision']:.3f} | "
                         f"{p['recall']:.3f} | {p['f1']:.3f} | {p['tp']} | "
                         f"{p['fp']} | {p['fn']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=Path("eval/labels/gap_labels.csv"),
                    help="human-labeled CSV (export_gap_labels.py output with `label` filled)")
    ap.add_argument("--out-json", type=Path, default=config.CALIBRATION_PATH)
    ap.add_argument("--out-report", type=Path, default=Path("eval/calibration_report.md"))
    ap.add_argument("--min-precision", type=float, default=None,
                    help="only consider τ with precision >= this")
    ap.add_argument("--sweep-weights", action="store_true",
                    help="also grid-search weights (0.1 simplex) for best F1")
    args = ap.parse_args(argv)

    if not args.labels.exists():
        ap.error(f"labels file not found: {args.labels} "
                 f"(run scripts/export_gap_labels.py, then fill the `label` column)")
    rows = load_labeled_csv(args.labels)
    if not rows:
        ap.error(f"no labeled rows (0/1 in `label`) in {args.labels}")
    positives = sum(1 for r in rows if int(r["label"]) == 1)

    weights = dict(config.G_SCORE_WEIGHTS)
    if args.sweep_weights:
        best = None
        for w in weight_grid():
            pt = best_operating_point(sweep_tau(rows, w), args.min_precision)
            if best is None or pt["f1"] > best[1]["f1"]:
                best = (w, pt)
        weights, chosen = best
        points = sweep_tau(rows, weights)
    else:
        points = sweep_tau(rows, weights)
        chosen = best_operating_point(points, args.min_precision)

    write_calibration(args.out_json, tau=chosen["tau"], weights=weights, stats=chosen,
                      source_csv=args.labels, labeled_n=len(rows), positives=positives)
    write_report(args.out_report, points=points, chosen=chosen, weights=weights,
                 labeled_n=len(rows), positives=positives, source_csv=args.labels)
    print(f"[calibrate] {len(rows)} labeled ({positives} positive) → "
          f"τ={chosen['tau']:.2f} (F1={chosen['f1']:.3f}); wrote {args.out_json} and {args.out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
