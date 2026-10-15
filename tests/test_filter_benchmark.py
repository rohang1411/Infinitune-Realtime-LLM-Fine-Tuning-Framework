import json
from pathlib import Path
import pytest
from benchmarks.filter_benchmark import (
    build_filter_benchmark_config,
    generate_synthetic_spam_records,
    load_or_generate_alpaca_records,
    run_filter_benchmark_part_a,
    run_filter_benchmark_part_b_functional,
    run_filter_benchmark_suite,
)
from utils.stream_filter import (
    StreamFilter,
    STAGE_LENGTH,
    STAGE_CONTENT,
    STAGE_REPETITION,
    STAGE_CHAT,
    STAGE_REGEX,
)


def test_5_stages_active_in_benchmark_config():
    cfg = build_filter_benchmark_config()
    sf = StreamFilter(cfg)
    stages = sf.active_stages()
    assert set(stages) == {
        STAGE_LENGTH,
        STAGE_CONTENT,
        STAGE_REPETITION,
        STAGE_CHAT,
        STAGE_REGEX,
    }


def test_synthetic_spam_recall_and_detection():
    cfg = build_filter_benchmark_config()
    alpaca = load_or_generate_alpaca_records(max_records=100, offline=True)
    spam = generate_synthetic_spam_records(seed=42, count_per_type=50)

    summary, dropped_samples = run_filter_benchmark_part_a(cfg, alpaca, spam, seed=42)

    assert summary["target_recall_met"] is True
    assert summary["recall_by_type"]["repeated_phrase"] >= 0.99
    assert summary["recall_by_type"]["symbol_noise"] >= 0.99
    assert summary["recall_by_type"]["url_spam"] >= 0.99
    assert summary["recall_by_type"]["length_bounds"] >= 0.99
    assert summary["overall_spam_recall"] >= 0.99
    assert summary["throughput_records_per_second"] > 0
    assert "dropped_by_stage" in summary


def test_no_stall_functional_proof():
    result = run_filter_benchmark_part_b_functional()
    assert result["no_stall_proven"] is True
    assert result["total_malformed_dropped"] == 50
    assert result["valid_records_passed"] == 50
    assert result["exceptions_raised"] == 0
    assert set(result["reasons_recorded"].keys()) == {
        "empty",
        "not_utf8",
        "not_json",
        "not_object",
    }


def test_suite_filter_benchmark_execution(tmp_path):
    class MockArgs:
        profile = "smoke"
        offline = True
        seed = 42

    class MockRun:
        directory = tmp_path

    res = run_filter_benchmark_suite(MockRun(), MockArgs())

    summary_file = tmp_path / "filter_benchmark" / "summary.json"
    csv_file = tmp_path / "filter_benchmark" / "dropped_sample.csv"

    assert summary_file.exists()
    assert csv_file.exists()

    saved_summary = json.loads(summary_file.read_text(encoding="utf-8"))
    assert saved_summary["part_a"]["target_recall_met"] is True
    assert saved_summary["part_b_no_stall"]["no_stall_proven"] is True
