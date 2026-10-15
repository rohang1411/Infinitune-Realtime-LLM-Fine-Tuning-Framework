"""
tests/test_propagation.py
-----------------------------------------------------------------------------
Unit tests for WP-13 propagation suite: timestamp extraction, log parsing,
join math, and /adapter/history endpoint.
"""

from pathlib import Path
import pytest
import time
from unittest.mock import MagicMock

from benchmarks.run import (
    parse_trainer_publish_log,
    join_propagation_timestamps,
    SUITES,
)
import inference


def test_propagation_suite_registered():
    assert "propagation" in SUITES
    assert "publish to apply" in SUITES["propagation"].lower() or "propagation" in SUITES["propagation"].lower()


def test_parse_trainer_publish_log(tmp_path: Path):
    log_file = tmp_path / "trainer.log"
    content = (
        "[2026-10-15 01:00:00][TRAINER] TrainingArguments: ...\n"
        "[2026-10-15 01:00:05][TRAINER] Published adapter step=step_1 seq=1 run_id=run_20261015_001 "
        "t_publish_start=1700000000.100000 t_flushed=1700000000.350000 publish_s=0.250000\n"
        "[2026-10-15 01:00:10][TRAINER] Published adapter step=step_2 seq=2 run_id=run_20261015_001 "
        "t_publish_start=1700000005.200000 t_flushed=1700000005.450000 publish_s=0.250000\n"
    )
    log_file.write_text(content, encoding="utf-8")

    records = parse_trainer_publish_log(log_file)
    assert len(records) == 2
    key1 = ("run_20261015_001", 1)
    key2 = ("run_20261015_001", 2)
    assert key1 in records
    assert key2 in records
    assert records[key1]["t_publish_start"] == 1700000000.1
    assert records[key1]["t_flushed"] == 1700000000.35
    assert pytest.approx(records[key1]["publish_s"], rel=1e-5) == 0.25


def test_join_propagation_timestamps():
    trainer_records = {
        ("run_A", 1): {
            "step": "step_1",
            "seq": 1,
            "run_id": "run_A",
            "t_publish_start": 100.0,
            "t_flushed": 100.2,
            "publish_s": 0.2,
        },
        ("run_A", 2): {
            "step": "step_2",
            "seq": 2,
            "run_id": "run_A",
            "t_publish_start": 105.0,
            "t_flushed": 105.3,
            "publish_s": 0.3,
        },
    }

    inference_history = [
        {
            "version": "step_1",
            "run_id": "run_A",
            "seq": 1,
            "t_publish_start": 100.0,
            "t_assembled": 100.5,
            "t_applied": 100.7,
        },
        {
            "version": "step_2",
            "run_id": "run_A",
            "seq": 2,
            "t_publish_start": 105.0,
            "t_assembled": 105.6,
            "t_applied": 105.9,
        },
    ]

    joined = join_propagation_timestamps(trainer_records, inference_history)
    assert len(joined) == 2

    first = joined[0]
    assert first["version"] == "step_1"
    assert first["run_id"] == "run_A"
    assert first["seq"] == 1
    assert pytest.approx(first["publish_s"]) == 0.2
    assert pytest.approx(first["transport_assemble_s"]) == 0.3
    assert pytest.approx(first["apply_s"]) == 0.2
    assert pytest.approx(first["propagation_s"]) == 0.7

    # Invariant: propagation_s == publish_s + transport_assemble_s + apply_s
    for item in joined:
        total = item["publish_s"] + item["transport_assemble_s"] + item["apply_s"]
        assert pytest.approx(item["propagation_s"], rel=1e-5) == total


def test_adapter_history_endpoint():
    inference.adapter_history_global.clear()
    inference.adapter_history_global.append({
        "version": "step_10",
        "run_id": "run_test_hist",
        "seq": 10,
        "t_publish_start": 1000.0,
        "t_assembled": 1000.4,
        "t_applied": 1000.6,
    })

    client = inference.app.test_client()
    resp = client.get("/adapter/history")
    assert resp.status_code == 200
    data = resp.get_json()

    assert "history" in data
    assert len(data["history"]) == 1
    assert data["history"][0]["version"] == "step_10"
    assert data["history"][0]["run_id"] == "run_test_hist"
    assert data["history"][0]["seq"] == 10
    assert data["history"][0]["t_publish_start"] == 1000.0
    assert data["history"][0]["t_assembled"] == 1000.4
    assert data["history"][0]["t_applied"] == 1000.6

    # Dict access compatibility
    assert "step_10" in data["versions"]
    assert data["step_10"]["seq"] == 10
