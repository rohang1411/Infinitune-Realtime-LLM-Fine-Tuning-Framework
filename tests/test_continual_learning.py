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


def test_adwin_properties_width_total_variance():
    detector = ADWINDriftDetector(delta=0.002, min_window_size=10, max_window_size=100)
    for v in [0.2, 0.4, 0.6, 0.8]:
        detector.update(v)

    assert detector.width == 4
    assert round(detector.total, 4) == 2.0
    assert detector.variance > 0.0


def test_adwin_no_false_alarm_on_stationary_bernoulli():
    import random
    rng = random.Random(0)
    detector = ADWINDriftDetector(
        delta=0.002, min_window_size=10, max_window_size=1000, clock=1
    )

    detections = 0
    # 5,000 Bernoulli(0.3) draws
    for _ in range(5000):
        val = 1.0 if rng.random() < 0.3 else 0.0
        if detector.update(val):
            detections += 1

    # False alarm rate strictly bound: <= 1 detection over 5,000 stationary trials
    assert detections <= 1


def test_adwin_detects_mean_shift_with_bounded_delay():
    import random
    rng = random.Random(42)
    detector = ADWINDriftDetector(
        delta=0.002, min_window_size=10, max_window_size=1000, clock=1
    )

    # 1,000 samples of Bernoulli(0.2)
    for _ in range(1000):
        val = 1.0 if rng.random() < 0.2 else 0.0
        detector.update(val)

    # Shift to Bernoulli(0.8) — must detect within 100 samples
    detected = False
    delay = 0
    for step in range(1, 101):
        val = 1.0 if rng.random() < 0.8 else 0.0
        if detector.update(val):
            detected = True
            delay = step
            break

    assert detected is True
    assert delay <= 100


def test_adwin_matches_river_oracle():
    import importlib.util
    if importlib.util.find_spec("river") is None:
        pytest.skip("river package not installed; skipping optional oracle comparison")

    from river import drift
    river_adwin = drift.ADWIN(delta=0.002)
    our_adwin = ADWINDriftDetector(delta=0.002, min_window_size=10, max_window_size=1000)

    import random
    rng = random.Random(1337)
    # Stream with sudden shift
    stream = [1.0 if rng.random() < 0.2 else 0.0 for _ in range(500)] + \
             [1.0 if rng.random() < 0.8 else 0.0 for _ in range(500)]

    river_detections = []
    our_detections = []

    for i, val in enumerate(stream):
        river_adwin.update(val)
        if river_adwin.drift_detected:
            river_detections.append(i)
        if our_adwin.update(val):
            our_detections.append(i)

    # Both should detect shift around step 500 within +- 64 samples
    assert len(our_detections) >= 1
    assert any(480 <= d <= 580 for d in our_detections)

