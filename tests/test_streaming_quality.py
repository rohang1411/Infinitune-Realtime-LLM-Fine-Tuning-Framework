"""
Unit tests for WP-20: Streaming quality suite and offline parity bounds verification.
"""

from pathlib import Path
import json
import pytest

from benchmarks.streaming_quality import (
    wilson_score_interval,
    percentile_bootstrap_ci,
    compute_streaming_metrics,
    check_parity_bounds,
    run_streaming_quality_suite,
    RUN10_ACCURACY_BOOTSTRAP_CI_95,
    RUN10_FINAL_ACCURACY,
)

ROOT = Path(__file__).resolve().parents[1]


def test_wilson_score_interval():
    low, high = wilson_score_interval(3306, 4000)
    assert 0.81 <= low <= 0.82
    assert 0.83 <= high <= 0.84

    # Empty total
    assert wilson_score_interval(0, 0) == (0.0, 0.0)


def test_compute_streaming_metrics():
    # 80 positive correct, 80 negative correct, 20 positive wrong, 20 'other' out of format
    preds = []
    for i in range(80):
        preds.append({"id": f"p{i}", "gold": "positive", "pred": "positive", "raw": "positive", "correct": True})
    for i in range(80):
        preds.append({"id": f"n{i}", "gold": "negative", "pred": "negative", "raw": "negative", "correct": True})
    for i in range(20):
        preds.append({"id": f"pw{i}", "gold": "positive", "pred": "negative", "raw": "negative", "correct": False})
    for i in range(20):
        preds.append({"id": f"po{i}", "gold": "negative", "pred": "other", "raw": "unrelated", "correct": False})

    m = compute_streaming_metrics(preds)
    assert m["total"] == 200
    assert m["correct"] == 160
    assert m["accuracy"] == 0.8000
    assert m["other_rate"] == 0.1000
    # Target-class macro F1 must be higher than diluted f1_with_other
    assert m["f1_macro"] > m["f1_with_other"]
    assert m["dilution_pct"] > 0.0
    assert m["mcc"] > 0.5


def test_parity_bounds_check():
    assert check_parity_bounds(RUN10_FINAL_ACCURACY) is True
    assert check_parity_bounds(0.8200) is True
    assert check_parity_bounds(0.7500) is False
    assert check_parity_bounds(0.9000) is False


def test_streaming_quality_suite_execution(tmp_path):
    class MockArgs:
        profile = "smoke"
        live_config = str(ROOT / "configs" / "imdb_quantitative.yaml")

    class MockRun:
        directory = tmp_path

    summary = run_streaming_quality_suite(MockRun(), MockArgs())
    assert summary["status"] == "complete"
    assert "final_accuracy" in summary
    assert "aauc_trapezoidal" in summary
    assert "aauc_uniform_mean" in summary
    assert (tmp_path / "streaming_quality" / "summary.json").exists()
    assert (tmp_path / "streaming_quality" / "curve.csv").exists()
