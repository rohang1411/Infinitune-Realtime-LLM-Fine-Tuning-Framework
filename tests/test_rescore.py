import torch
import json
from pathlib import Path
import pytest
from benchmarks.rescore_run import (
    wilson_score_interval,
    percentile_bootstrap_ci,
    compute_step_mcc,
    compute_f1_pair,
    rescore_run,
    run_rescore_suite,
)


def test_wilson_score_interval_properties():
    # Symmetric 50%
    low, high = wilson_score_interval(50, 100)
    assert 0.40 <= low < 0.50
    assert 0.50 < high <= 0.60
    assert round((low + high) / 2.0, 2) == 0.50

    # Boundary 0% and 100%
    low_0, high_0 = wilson_score_interval(0, 100)
    assert low_0 == 0.0
    assert high_0 > 0.0

    low_1, high_1 = wilson_score_interval(100, 100)
    assert low_1 < 1.0
    assert high_1 == 1.0


def test_percentile_bootstrap_ci_properties():
    y_true = ["positive"] * 50 + ["negative"] * 50
    # 80% accuracy
    y_pred = ["positive"] * 40 + ["negative"] * 10 + ["negative"] * 40 + ["positive"] * 10
    
    low, high = percentile_bootstrap_ci(
        y_true, y_pred,
        lambda yt, yp: sum(1 for a, b in zip(yt, yp) if a == b) / float(len(yt)),
        draws=500, seed=42,
    )
    assert 0.70 <= low <= 0.80
    assert 0.80 <= high <= 0.90


def test_dilution_and_mcc_properties():
    y_true = ["positive", "positive", "negative", "negative"]
    y_pred = ["positive", "other", "negative", "negative"]

    mcc = compute_step_mcc(y_true, y_pred)
    assert -1.0 <= mcc <= 1.0

    f1_macro, f1_with_other, dilution = compute_f1_pair(y_true, y_pred)
    assert f1_macro > 0
    assert f1_with_other < f1_macro  # 'other' dilutes F1
    assert dilution > 0


def test_rescore_run_smoke(tmp_path):
    out_dir = tmp_path / "rescore_smoke"
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    summary = rescore_run(
        checkpoint_root=str(ckpt_dir),
        out_dir=str(out_dir),
        include_base=True,
        enforce_gate=False,
        max_samples=5,
    )

    assert (out_dir / "summary.json").exists()
    assert (out_dir / "curve.csv").exists()
    assert (out_dir / "pool_ids.json").exists()

    assert summary["steps_scored"] >= 1
    assert "aauc_trapezoidal" in summary
    assert "aauc_uniform_mean" in summary
    assert summary["pool_count"] == 5
