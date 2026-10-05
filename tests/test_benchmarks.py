"""
tests/test_benchmarks.py
-----------------------------------------------------------------------------
Unit tests for benchmark harnesses: load testing, canary freshness, and CL runner.
"""

import pytest
from benchmarks.run_streaming_cl import (
    compute_continual_learning_metrics,
    generate_markdown_report,
)


def test_cl_metrics_calculation():
    # Matrix R[i][j]: after training task i, eval on task j
    # Phase 1: 80% on Task 1, 50% on Task 2
    # Phase 2: 70% on Task 1 (-10% forgetting), 85% on Task 2
    matrix = [
        [0.80, 0.50],
        [0.70, 0.85],
    ]
    tasks = ["Task1", "Task2"]

    metrics = compute_continual_learning_metrics(matrix, tasks)

    # BWT on Task 1 = R[1][0] - R[0][0] = 0.70 - 0.80 = -0.10
    assert abs(metrics["backward_transfer"] - (-0.10)) < 1e-4
    assert metrics["forgetting_per_task"]["Task1"] == 0.10
    assert metrics["forgetting_per_task"]["Task2"] == 0.0
    assert abs(metrics["final_average_accuracy"] - 0.775) < 1e-4


def test_cl_markdown_report_formatting():
    matrix = [
        [0.80, 0.50],
        [0.70, 0.85],
    ]
    tasks = ["Task1", "Task2"]
    report = generate_markdown_report(matrix, tasks, method_name="Test Model")

    assert "### Continual Learning Benchmark: Test Model" in report
    assert "| After Phase 1 (Task1) |" in report
    assert "**Backward Transfer (BWT):** -0.1000" in report
