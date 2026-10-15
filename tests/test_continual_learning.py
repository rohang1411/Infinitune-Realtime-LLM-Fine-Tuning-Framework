"""
tests/test_continual_learning.py
-----------------------------------------------------------------------------
Unit tests for ReservoirReplayBuffer, GoldenCanaryEvaluator, and ADWINDriftDetector.
"""

import pytest
from utils.replay_buffer import ReservoirReplayBuffer
from utils.continual_engine import GoldenCanaryEvaluator, ADWINDriftDetector


def test_reservoir_buffer_capacity():
    buf = ReservoirReplayBuffer(capacity=50, seed=42)
    # Stream 500 items into buffer of capacity 50
    for i in range(500):
        buf.add({"id": i, "text": f"sample_{i}"})

    assert len(buf) == 50
    assert buf.total_seen == 500

    # Sampling 10 items returns 10 unique items
    samples = buf.sample(10)
    assert len(samples) == 10


def test_reservoir_buffer_collate_with_stream():
    buf = ReservoirReplayBuffer(capacity=20, seed=42)
    for i in range(20):
        buf.add({"id": f"history_{i}"})

    stream_batch = [{"id": f"stream_{i}"} for i in range(10)]
    # 20% replay on batch of 10 should include 2 history samples
    mixed = buf.collate_with_stream(stream_batch, replay_ratio=0.20)

    assert len(mixed) == 10
    history_count = sum(1 for item in mixed if "history_" in item["id"])
    stream_count = sum(1 for item in mixed if "stream_" in item["id"])

    assert history_count == 2
    assert stream_count == 8


def test_golden_canary_gate_approval_and_rejection():
    evaluator = GoldenCanaryEvaluator(
        canary_data=[{"text": "canary"}],
        max_regression_ratio=0.10,  # allow up to +10% regression
    )

    # 1. Establish baseline
    appr, metrics = evaluator.evaluate_gate(candidate_loss=1.00)
    assert appr is True
    assert metrics["baseline_loss"] == 1.00

    # 2. Candidate with +5% regression (loss=1.05) should be approved
    appr, metrics = evaluator.evaluate_gate(candidate_loss=1.05)
    assert appr is True
    assert metrics["approved"] is True

    # 3. Candidate with +25% regression (loss=1.25) should be rejected
    appr, metrics = evaluator.evaluate_gate(candidate_loss=1.25)
    assert appr is False
    assert metrics["approved"] is False
    assert metrics["regression_ratio"] == 0.25


def test_adwin_drift_detector():
    detector = ADWINDriftDetector(delta=0.01, min_window_size=10, max_window_size=100)

    # Stationary sequence of 0.1
    for _ in range(50):
        drift = detector.update(0.1)
        assert drift is False

    # Sudden distribution shift to 5.0
    drift_seen = False
    for _ in range(30):
        if detector.update(5.0):
            drift_seen = True
            break

    assert drift_seen is True
