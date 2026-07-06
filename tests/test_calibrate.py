"""Tests for config.load_calibration + scripts/calibrate_gap_thresholds functions."""
from __future__ import annotations

import importlib
import json
import math

from axiom import config

calibrate = importlib.import_module("scripts.calibrate_gap_thresholds")


# --- config.load_calibration -------------------------------------------------
def test_load_calibration_defaults_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CALIBRATION_PATH", tmp_path / "nope.json")
    calib = config.load_calibration()
    assert calib["tau"] == config.GAP_TAU_DEFAULT
    assert calib["weights"] == config.G_SCORE_WEIGHTS


def test_load_calibration_override(tmp_path, monkeypatch):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"tau": 0.42, "weights": {"similarity": 0.7}}))
    monkeypatch.setattr(config, "CALIBRATION_PATH", path)
    calib = config.load_calibration()
    assert calib["tau"] == 0.42
    assert calib["weights"]["similarity"] == 0.7
    # unspecified weights keep their defaults
    assert calib["weights"]["authority"] == config.G_SCORE_WEIGHTS["authority"]


def test_load_calibration_ignores_malformed(tmp_path, monkeypatch):
    path = tmp_path / "calibration.json"
    path.write_text("{not valid json")
    monkeypatch.setattr(config, "CALIBRATION_PATH", path)
    assert config.load_calibration()["tau"] == config.GAP_TAU_DEFAULT


# --- calibration math --------------------------------------------------------
_W = {"similarity": 1.0, "disconnection": 0.0, "velocity": 0.0, "authority": 0.0}


def _row(s, label):
    # Only S matters under _W, so G == s.
    return {"S": str(s), "D": "0", "V": "0", "A": "0", "label": str(label)}


def test_recompute_g_uses_weights():
    row = {"S": "0.8", "D": "0.4", "V": "0.5", "A": "0.2"}
    w = {"similarity": 0.5, "disconnection": 0.5, "velocity": 0.0, "authority": 0.0}
    assert math.isclose(calibrate.recompute_g(row, w), 0.5 * 0.8 + 0.5 * 0.4)


def test_precision_recall_counts():
    # Positives at S>=0.6; τ=0.6 separates them cleanly.
    rows = [_row(0.9, 1), _row(0.7, 1), _row(0.3, 0), _row(0.1, 0)]
    pr = calibrate.precision_recall(rows, 0.6, _W)
    assert pr["tp"] == 2 and pr["fp"] == 0 and pr["fn"] == 0 and pr["tn"] == 2
    assert pr["precision"] == 1.0 and pr["recall"] == 1.0 and pr["f1"] == 1.0


def test_sweep_and_best_operating_point():
    rows = [_row(0.9, 1), _row(0.7, 1), _row(0.3, 0), _row(0.1, 0)]
    points = calibrate.sweep_tau(rows, _W, step=0.1)
    best = calibrate.best_operating_point(points)
    assert best["f1"] == 1.0
    # A τ in (0.3, 0.7] perfectly separates; chosen τ must sit there.
    assert 0.3 < best["tau"] <= 0.7


def test_min_precision_floor():
    # One positive, one negative sharing the same score → precision capped at 0.5.
    rows = [_row(0.5, 1), _row(0.5, 0), _row(0.1, 0)]
    points = calibrate.sweep_tau(rows, _W, step=0.1)
    best = calibrate.best_operating_point(points, min_precision=0.9)
    # No τ reaches precision 0.9, so the floor is ignored (pool falls back to all).
    assert best is not None


def test_load_labeled_csv_skips_unlabeled(tmp_path):
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "# a comment line\n"
        "candidate_id,S,D,V,A,label\n"
        "c1,0.9,0,0,0,1\n"
        "c2,0.2,0,0,0,\n"          # unlabeled → skipped
        "c3,0.1,0,0,0,0\n"
    )
    rows = calibrate.load_labeled_csv(csv_path)
    assert [r["candidate_id"] for r in rows] == ["c1", "c3"]


def test_end_to_end_writes_json_and_report(tmp_path):
    csv_path = tmp_path / "gap_labels.csv"
    csv_path.write_text(
        "candidate_id,S,D,V,A,label\n"
        "c1,0.9,0,0,0,1\n"
        "c2,0.7,0,0,0,1\n"
        "c3,0.3,0,0,0,0\n"
        "c4,0.1,0,0,0,0\n"
    )
    out_json = tmp_path / "calibration.json"
    out_report = tmp_path / "report.md"
    rc = calibrate.main(["--labels", str(csv_path), "--out-json", str(out_json),
                         "--out-report", str(out_report), "--sweep-weights"])
    assert rc == 0
    payload = json.loads(out_json.read_text())
    assert payload["labeled_n"] == 4 and payload["positives"] == 2
    assert 0.0 <= payload["tau"] <= 1.0
    assert math.isclose(sum(payload["weights"].values()), 1.0, abs_tol=1e-6)
    assert out_report.exists() and "Chosen τ" in out_report.read_text()
