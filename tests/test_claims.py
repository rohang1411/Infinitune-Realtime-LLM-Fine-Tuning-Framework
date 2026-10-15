"""
Unit tests for WP-29: Master evidence ledger and claim audit compilation.
"""

from pathlib import Path
import json
import pytest

from benchmarks.claims import (
    evaluate_threshold,
    build_evidence_ledger,
    generate_claims_markdown,
    CLAIMS,
)

ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_threshold():
    assert evaluate_threshold(5.0, "<", 8.0) is True
    assert evaluate_threshold(8.0, "<", 8.0) is False
    assert evaluate_threshold(140.0, "<=", 140.0) is True
    assert evaluate_threshold(140.1, "<=", 140.0) is False
    assert evaluate_threshold(0.85, ">=", 0.85) is True
    assert evaluate_threshold(0, "==", 0) is True
    assert evaluate_threshold(1, "==", 0) is False
    assert evaluate_threshold(None, "==", 0) is False


def test_build_evidence_ledger_synthetic(tmp_path):
    # Create fake summary JSONs
    run_dir = tmp_path / "run_synthetic"
    run_dir.mkdir()

    prop_dir = run_dir / "propagation"
    prop_dir.mkdir()
    (prop_dir / "summary.json").write_text(json.dumps({
        "propagation_s_p95": 6.42,
        "matched_count": 25,
    }), encoding="utf-8")

    swap_dir = run_dir / "swap_stress"
    swap_dir.mkdir()
    (swap_dir / "summary.json").write_text(json.dumps({
        "swap_lock_held_ms_p50": 94.2,
        "requests": 500,
    }), encoding="utf-8")

    fault_dir = run_dir / "fault_injection"
    fault_dir.mkdir()
    (fault_dir / "summary.json").write_text(json.dumps({
        "corrupted_states": 0,
        "request_failures": 0,
    }), encoding="utf-8")

    ledger = build_evidence_ledger([run_dir])
    by_claim = {e["claim_id"]: e for e in ledger}

    assert by_claim["SDE-1"]["status"] == "PASS"
    assert by_claim["SDE-1"]["measured_value"] == 6.42

    assert by_claim["SDE-2_swap"]["status"] == "PASS"
    assert by_claim["SDE-2_swap"]["measured_value"] == 94.2

    assert by_claim["SDE-3"]["status"] == "PASS"
    assert by_claim["SDE-3"]["measured_value"] == 0

    md = generate_claims_markdown(ledger)
    assert "# InfiniTune — Master Evidence & Resume Claims Ledger" in md
    assert "SDE-1" in md
    assert "**PASS**" in md
