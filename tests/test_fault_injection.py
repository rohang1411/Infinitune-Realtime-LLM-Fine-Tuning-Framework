"""
Unit tests for Fault Injection Suite (WP-16).

Tests complete fault catalogue F01 through F16 in-memory without requiring
a live Kafka broker or live network, adhering strictly to Rule R7.
"""

import os
import json
import pickle
import pytest
import torch
import safetensors.torch

from utils.stream_protocol import (
    SnapshotAssembler,
    canonical_json,
    manifest_digest,
    compute_manifest_hmac,
    classify_rejection_reason,
)
from benchmarks.fault_injection import (
    build_snapshot_frames,
    run_fault_simulation_on_assembler,
    serialize_tensor_frame,
)
from inference import process_adapter_record


@pytest.fixture
def base_tensors():
    torch.manual_seed(42)
    return {
        "base_model.model.transformer.h.0.attn.c_attn.lora_A.default.weight": torch.randn(4, 8),
        "base_model.model.transformer.h.0.attn.c_attn.lora_B.default.weight": torch.randn(8, 4),
    }


def test_classify_rejection_reason():
    assert classify_rejection_reason(ValueError("sha256_mismatch for tensor x")) == "sha256_mismatch"
    assert classify_rejection_reason(ValueError("Rejected incomplete adapter step 5")) == "incomplete_snapshot"
    assert classify_rejection_reason(ValueError("Duplicate adapter tensor foo")) == "duplicate_tensor"
    assert classify_rejection_reason(ValueError("Undeclared adapter tensor bar")) == "undeclared_tensor"
    assert classify_rejection_reason(ValueError("shape_or_dtype_mismatch for tensor")) == "shape_mismatch"
    assert classify_rejection_reason(ValueError("Rejected commit for unknown or expired step")) == "unknown_step"
    assert classify_rejection_reason(ValueError("Commit manifest_sha256 mismatch: expected x got y")) == "manifest_digest_mismatch"
    assert classify_rejection_reason(ValueError("Malformed manifest JSON: invalid syntax")) == "bad_manifest"
    assert classify_rejection_reason(ValueError("stale_version: seq 1 <= active seq 5")) == "stale_version"
    assert classify_rejection_reason(ValueError("Invalid manifest HMAC signature")) == "hmac_mismatch"
    assert classify_rejection_reason(ValueError("Rejected v1 manifest: protocol_version 2 required")) == "protocol_version"


def test_f01_truncated_tensor(base_tensors):
    assembler = SnapshotAssembler()
    res = run_fault_simulation_on_assembler(assembler, "F01", base_tensors)
    assert res["passed"]
    assert "sha256_mismatch" in res["observed_reasons"]


def test_f02_bit_flip(base_tensors):
    assembler = SnapshotAssembler()
    res = run_fault_simulation_on_assembler(assembler, "F02", base_tensors)
    assert res["passed"]
    assert "sha256_mismatch" in res["observed_reasons"]


def test_f03_incomplete_snapshot(base_tensors):
    assembler = SnapshotAssembler()
    res = run_fault_simulation_on_assembler(assembler, "F03", base_tensors)
    assert res["passed"]
    assert "incomplete_snapshot" in res["observed_reasons"]


def test_f04_duplicate_tensor(base_tensors):
    assembler = SnapshotAssembler()
    res = run_fault_simulation_on_assembler(assembler, "F04", base_tensors)
    assert res["passed"]
    assert "duplicate_tensor" in res["observed_reasons"]


def test_f05_undeclared_tensor(base_tensors):
    assembler = SnapshotAssembler()
    res = run_fault_simulation_on_assembler(assembler, "F05", base_tensors)
    assert res["passed"]
    assert "undeclared_tensor" in res["observed_reasons"]


def test_f06_shape_mismatch(base_tensors):
    assembler = SnapshotAssembler()
    res = run_fault_simulation_on_assembler(assembler, "F06", base_tensors)
    assert res["passed"]
    assert "shape_mismatch" in res["observed_reasons"]


def test_f07_pickle_payload(base_tensors):
    assembler = SnapshotAssembler()
    res = run_fault_simulation_on_assembler(assembler, "F07", base_tensors)
    assert res["passed"]
    assert any(r in ("sha256_mismatch", "ValueError") for r in res["observed_reasons"])


def test_f08_commit_without_manifest(base_tensors):
    assembler = SnapshotAssembler()
    res = run_fault_simulation_on_assembler(assembler, "F08", base_tensors)
    assert res["passed"]
    assert "unknown_step" in res["observed_reasons"]


def test_f09_wrong_manifest_sha256(base_tensors):
    assembler = SnapshotAssembler()
    res = run_fault_simulation_on_assembler(assembler, "F09", base_tensors)
    assert res["passed"]
    assert "manifest_digest_mismatch" in res["observed_reasons"]


def test_f10_bad_manifest(base_tensors):
    assembler = SnapshotAssembler()
    res = run_fault_simulation_on_assembler(assembler, "F10", base_tensors)
    assert res["passed"]
    assert "bad_manifest" in res["observed_reasons"]


def test_f11_stale_version_replay(base_tensors):
    assembler = SnapshotAssembler()
    res = run_fault_simulation_on_assembler(assembler, "F11", base_tensors)
    assert res["passed"]
    assert "stale_version" in res["observed_reasons"]


def test_f12_hmac_mismatch(base_tensors):
    assembler = SnapshotAssembler()
    res = run_fault_simulation_on_assembler(assembler, "F12", base_tensors)
    assert res["passed"]
    assert "hmac_mismatch" in res["observed_reasons"]


def test_f13_protocol_v1_unsigned(base_tensors):
    assembler = SnapshotAssembler(integrity="sha256")
    res = run_fault_simulation_on_assembler(assembler, "F13", base_tensors)
    assert res["passed"]
    assert "protocol_version" in res["observed_reasons"]


def test_f14_interleaved_snapshots(base_tensors):
    assembler = SnapshotAssembler()
    res = run_fault_simulation_on_assembler(assembler, "F14", base_tensors)
    assert res["passed"]
    assert res["committed_step"].endswith("_b")


def test_f15_producer_oversized_frame(base_tensors):
    assembler = SnapshotAssembler()
    res = run_fault_simulation_on_assembler(assembler, "F15", base_tensors)
    assert res["passed"]


def test_f16_burst_bad_frames(base_tensors):
    assembler = SnapshotAssembler()
    res = run_fault_simulation_on_assembler(assembler, "F16", base_tensors)
    assert res["passed"]
    assert len(res["observed_reasons"]) == 20


def test_recovery_probe_after_faults(base_tensors):
    """Verify that after multiple rejected faults, a valid snapshot commits cleanly."""
    assembler = SnapshotAssembler()
    counters = {"rejected_frames": 0, "rejected_by_reason": {}}

    # Inject several bad frames via process_adapter_record
    process_adapter_record(assembler, "__manifest__:bad1", b"NOT_JSON", counters=counters)
    process_adapter_record(assembler, "bad1:tensor", b"corrupted_bytes", counters=counters)
    process_adapter_record(assembler, "__commit__:bad1", b"{}", counters=counters)

    assert counters["rejected_frames"] == 3
    assert "bad_manifest" in counters["rejected_by_reason"]
    assert "undeclared_tensor" in counters["rejected_by_reason"]

    # Now send valid recovery probe
    probe_step = "probe_recovery_1"
    m_frame, t_frames, c_frame, _ = build_snapshot_frames(
        probe_step, "run_recovery", 100, base_tensors
    )

    res_m = process_adapter_record(assembler, *m_frame, counters=counters)
    assert res_m is None

    for tf in t_frames:
        res_t = process_adapter_record(assembler, *tf, counters=counters)
        assert res_t is None

    res_c = process_adapter_record(assembler, *c_frame, counters=counters)
    assert res_c is not None
    committed_step, committed_batch = res_c
    assert committed_step == probe_step
    assert len(committed_batch) == len(base_tensors)
