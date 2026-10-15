"""
tests/test_cold_start.py
-----------------------------------------------------------------------------
Unit tests for WP-14 cold-start benchmark suite and checkpoint recovery metrics.
"""

import pytest

from benchmarks.run import (
    summarize_cold_start_results,
    SUITES,
)


def test_cold_start_suite_registered():
    assert "cold_start" in SUITES
    desc = SUITES["cold_start"].lower()
    assert "checkpoint recovery" in desc or "cold" in desc


def test_summarize_cold_start_results():
    arm_a = [
        {"arm": "arm_a_recovery_on", "ready_s": 2.1, "time_to_trained_weights_s": 0.05, "listener_ready": True},
        {"arm": "arm_a_recovery_on", "ready_s": 2.3, "time_to_trained_weights_s": 0.04, "listener_ready": True},
        {"arm": "arm_a_recovery_on", "ready_s": 2.0, "time_to_trained_weights_s": 0.06, "listener_ready": True},
    ]
    arm_b = [
        {"arm": "arm_b_recovery_off", "ready_s": 2.0, "time_to_trained_weights_s": 60.0},
        {"arm": "arm_b_recovery_off", "ready_s": 2.1, "time_to_trained_weights_s": 60.0},
        {"arm": "arm_b_recovery_off", "ready_s": 2.2, "time_to_trained_weights_s": 60.0},
    ]

    summary = summarize_cold_start_results(arm_a, arm_b)

    assert summary["n_a"] == 3
    assert summary["n_b"] == 3

    a_stats = summary["arm_a_recovery_on"]
    assert a_stats["ready_s"]["p50"] == 2.1
    assert a_stats["time_to_trained_weights_s"]["p50"] == 0.05
    assert a_stats["listener_ready_all"] is True

    b_stats = summary["arm_b_recovery_off"]
    assert b_stats["ready_s"]["p50"] == 2.1
    assert b_stats["time_to_trained_weights_s"]["p50"] == 60.0

    # Delta proves arm A served trained weights ~60s faster than waiting for push
    assert pytest.approx(summary["delta_time_to_trained_p50_s"], rel=1e-3) == 59.95

    assert "Model load time is not removed" in summary["boundary"]
    assert "answers with untrained base weights" in summary["boundary"]


def test_summarize_cold_start_empty_guard():
    summary = summarize_cold_start_results([], [])
    assert summary["n_a"] == 0
    assert summary["n_b"] == 0
    assert summary["delta_time_to_trained_p50_s"] == 0.0
