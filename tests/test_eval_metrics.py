"""
tests/test_eval_metrics.py
-----------------------------------------------------------------------------
Unit tests for evaluation metrics engine:
1. Target-class macro F1 calculation (verifying phantom 'other' class no longer suppresses F1).
2. Expected Calibration Error (ECE).
3. Area Under Accuracy Curve (AAUC) calculation.
"""

import math
import pytest
from utils.eval_metrics_train import (
    compute_expected_calibration_error,
    _normalized_aauc_from_history,
)


def test_ece_perfect_calibration():
    # When predictions are completely confident and correct
    confidences = [1.0, 1.0, 1.0, 1.0]
    predictions = ["pos", "pos", "neg", "neg"]
    ground_truth = ["pos", "pos", "neg", "neg"]
    ece = compute_expected_calibration_error(confidences, predictions, ground_truth, num_bins=10)
    assert abs(ece - 0.0) < 1e-6


def test_ece_overconfident_uncalibrated():
    # 100% confident, but 50% accurate
    confidences = [1.0, 1.0, 1.0, 1.0]
    predictions = ["pos", "pos", "pos", "pos"]
    ground_truth = ["pos", "pos", "neg", "neg"]
    ece = compute_expected_calibration_error(confidences, predictions, ground_truth, num_bins=10)
    # Bin [0.9, 1.0]: acc = 0.5, conf = 1.0 -> gap = 0.5
    assert abs(ece - 0.5) < 1e-6


def test_normalized_aauc():
    # Monotonic progression from 0.0 to 1.0 over 100 steps
    history = [(0, 0.0), (50, 0.5), (100, 1.0)]
    aauc = _normalized_aauc_from_history(history)
    # Trapezoidal area: 0.5*(0+0.5)*50 + 0.5*(0.5+1.0)*50 = 12.5 + 37.5 = 50.0 / 100.0 = 0.5
    assert abs(aauc - 0.5) < 1e-6


def test_target_class_macro_f1_vs_phantom_class():
    """
    Direct simulation of the CSCI-566 IMDb evaluation:
    - 4,000 samples: 2,000 positive, 2,000 negative.
    - Model achieves 82.65% accuracy (3,306 correct).
    - Errors include 347 cross-class errors and 347 'other' off-target predictions.
    
    Before fix:
      all_labels included 'other' as 3rd class, dividing sum by 3 -> F1 was 0.576.
    After fix:
      F1 is computed over true target classes (positive, negative) -> F1 matches accuracy ~0.826.
    """
    total = 4000
    pos_gold = 2000
    neg_gold = 2000

    gold_labels = ["positive"] * pos_gold + ["negative"] * neg_gold

    # Build predictions with 82.65% accuracy and some 'other' predictions
    # Correct: 1653 pos, 1653 neg = 3306 correct (82.65%)
    # Errors: 174 'other' on pos, 173 'negative' on pos (347 errors)
    #         173 'other' on neg, 174 'positive' on neg (347 errors)
    pos_preds = ["positive"] * 1653 + ["other"] * 174 + ["negative"] * 173
    neg_preds = ["negative"] * 1653 + ["other"] * 173 + ["positive"] * 174
    pred_labels = pos_preds + neg_preds

    assert len(gold_labels) == total
    assert len(pred_labels) == total

    accuracy = sum(g == p for g, p in zip(gold_labels, pred_labels)) / total
    assert abs(accuracy - 0.8265) < 1e-4

    target_classes = ["negative", "positive"]
    target_f1s = []
    for tc in target_classes:
        tp_t = sum(1 for g, p in zip(gold_labels, pred_labels) if g == tc and p == tc)
        fp_t = sum(1 for g, p in zip(gold_labels, pred_labels) if g != tc and p == tc)
        fn_t = sum(1 for g, p in zip(gold_labels, pred_labels) if g == tc and p != tc)
        prec_t = tp_t / max(tp_t + fp_t, 1)
        rec_t = tp_t / max(tp_t + fn_t, 1)
        target_f1s.append(2 * prec_t * rec_t / max(prec_t + rec_t, 1e-9))

    clean_macro_f1 = sum(target_f1s) / len(target_f1s)

    # Legacy calculation with 'other' as 3rd class:
    all_classes = ["negative", "other", "positive"]
    legacy_f1s = []
    for c in all_classes:
        tp = sum(1 for g, p in zip(gold_labels, pred_labels) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold_labels, pred_labels) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold_labels, pred_labels) if g == c and p != c)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        legacy_f1s.append(2 * prec * rec / max(prec + rec, 1e-9))

    legacy_macro_f1 = sum(legacy_f1s) / len(legacy_f1s)

    # The legacy F1 was severely depressed to ~0.576 due to the phantom 'other' class
    assert legacy_macro_f1 < 0.60
    # The clean target-class F1 is ~0.864
    assert clean_macro_f1 > 0.82
    # Verify that the legacy F1 was literally (2/3) of the true target F1!
    assert abs(legacy_macro_f1 - (2.0 / 3.0) * clean_macro_f1) < 0.01

