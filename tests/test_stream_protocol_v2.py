"""tests/test_stream_protocol_v2.py
-----------------------------------------------------------------------------
Comprehensive unit tests for Streaming Integrity Protocol v2:
- Per-tensor SHA-256 verification before deserialization
- Canonical JSON manifest digest validation on commit
- Optional HMAC-SHA256 signature verification over manifest
- Monotonic version guard preventing replay and stale rollbacks
- Frame size guard preventing partial sends exceeding max_request_size
- Protocol v1 compatibility mode
"""

import hashlib
import json
import pytest
import torch
import safetensors.torch
from unittest.mock import MagicMock

from utils.stream_protocol import (
    SnapshotAssembler,
    canonical_json,
    manifest_digest,
    compute_manifest_hmac,
)
from trainer import LoRAProducer


def _build_v2_fixtures(
    step_id="100",
    run_id="run_20261015-101500_ab12",
    seq=7,
    created_at=1760520000.123,
    integrity="sha256",
    hmac_key=None,
):
    t1 = torch.randn(2, 4, dtype=torch.float32)
    b1 = safetensors.torch.save({"weight": t1})
    h1 = hashlib.sha256(b1).hexdigest()

    tensor_names = ["layer.weight"]
    tensors_meta = {
        "layer.weight": {
            "sha256": h1,
            "nbytes": len(b1),
            "dtype": "float32",
            "shape": [2, 4],
        }
    }

    manifest = {
        "protocol_version": 2,
        "run_id": run_id,
        "step_id": step_id,
        "seq": seq,
        "created_at": created_at,
        "num_tensors": 1,
        "tensor_names": tensor_names,
        "tensors": tensors_meta,
        "integrity": integrity,
        "hmac_sha256": None,
    }
    if integrity == "hmac" and hmac_key:
        manifest["hmac_sha256"] = compute_manifest_hmac(manifest, hmac_key)

    m_digest = manifest_digest(manifest)
    commit = {
        "step_id": step_id,
        "status": "committed",
        "manifest_sha256": m_digest,
        "timestamp": created_at + 0.1,
    }
    return manifest, {"layer.weight": b1}, commit, {"layer.weight": t1}


def test_v2_roundtrip_commits():
    manifest, frames, commit, tensors = _build_v2_fixtures()
    assembler = SnapshotAssembler()

    step_id = manifest["step_id"]
    assert assembler.add(f"__manifest__:{step_id}", canonical_json(manifest)) is None
    assert assembler.add(f"{step_id}:layer.weight", frames["layer.weight"]) is None
    res = assembler.add(f"__commit__:{step_id}", canonical_json(commit))

    assert res is not None
    committed_step, batch = res
    assert committed_step == step_id
    assert "layer.weight" in batch
    assert torch.allclose(batch["layer.weight"], tensors["layer.weight"])
    assert assembler.active == (manifest["run_id"], manifest["seq"], manifest["created_at"])


def test_v2_rejects_bitflip_in_tensor_frame():
    manifest, frames, commit, _ = _build_v2_fixtures()
    assembler = SnapshotAssembler()
    step_id = manifest["step_id"]

    assert assembler.add(f"__manifest__:{step_id}", canonical_json(manifest)) is None

    # Corrupt one byte of the serialized frame
    corrupted_bytes = bytearray(frames["layer.weight"])
    corrupted_bytes[10] ^= 0xFF
    corrupted_bytes = bytes(corrupted_bytes)

    with pytest.raises(ValueError, match="sha256_mismatch"):
        assembler.add(f"{step_id}:layer.weight", corrupted_bytes)


def test_v2_rejects_truncated_frame():
    manifest, frames, commit, _ = _build_v2_fixtures()
    assembler = SnapshotAssembler()
    step_id = manifest["step_id"]

    assert assembler.add(f"__manifest__:{step_id}", canonical_json(manifest)) is None

    # Truncate serialized payload
    truncated_bytes = frames["layer.weight"][:20]

    with pytest.raises(ValueError):
        assembler.add(f"{step_id}:layer.weight", truncated_bytes)


def test_v2_rejects_shape_or_dtype_mismatch_with_correct_hash():
    t = torch.randn(2, 4, dtype=torch.float32)
    b = safetensors.torch.save({"weight": t})
    h = hashlib.sha256(b).hexdigest()

    manifest = {
        "protocol_version": 2,
        "run_id": "run_test_shape",
        "step_id": "50",
        "seq": 1,
        "created_at": 1000.0,
        "num_tensors": 1,
        "tensor_names": ["layer.weight"],
        "tensors": {
            "layer.weight": {
                "sha256": h,
                "nbytes": len(b),
                "dtype": "float32",
                "shape": [4, 2],  # Declared mismatch: [4, 2] vs actual [2, 4]
            }
        },
        "integrity": "sha256",
        "hmac_sha256": None,
    }

    assembler = SnapshotAssembler()
    assert assembler.add("__manifest__:50", canonical_json(manifest)) is None

    with pytest.raises(ValueError, match="shape_or_dtype_mismatch"):
        assembler.add("50:layer.weight", b)


def test_v2_rejects_commit_with_wrong_manifest_digest():
    manifest, frames, commit, _ = _build_v2_fixtures()
    assembler = SnapshotAssembler()
    step_id = manifest["step_id"]

    assert assembler.add(f"__manifest__:{step_id}", canonical_json(manifest)) is None
    assert assembler.add(f"{step_id}:layer.weight", frames["layer.weight"]) is None

    # Tamper commit manifest digest
    commit["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Commit manifest_sha256 mismatch"):
        assembler.add(f"__commit__:{step_id}", canonical_json(commit))


def test_v2_hmac_accepts_correct_key_and_rejects_wrong_key():
    secret_key = "infinitune-secret-key-123"
    assembler = SnapshotAssembler(integrity="hmac", hmac_key=secret_key)

    # 1. Valid signature with secret_key
    manifest_good, frames, commit, _ = _build_v2_fixtures(integrity="hmac", hmac_key=secret_key)
    assert assembler.add(f"__manifest__:{manifest_good['step_id']}", canonical_json(manifest_good)) is None

    # 2. Signed with wrong key
    manifest_bad, _, _, _ = _build_v2_fixtures(step_id="101", seq=8, integrity="hmac", hmac_key="wrong-key")
    with pytest.raises(ValueError, match="HMAC"):
        assembler.add(f"__manifest__:{manifest_bad['step_id']}", canonical_json(manifest_bad))


def test_v2_receiver_in_hmac_mode_rejects_unsigned_manifest():
    assembler = SnapshotAssembler(integrity="hmac", hmac_key="secret-key")
    # Manifest without HMAC signature
    manifest, _, _, _ = _build_v2_fixtures(integrity="sha256", hmac_key=None)

    with pytest.raises(ValueError, match="hmac_sha256"):
        assembler.add(f"__manifest__:{manifest['step_id']}", canonical_json(manifest))


def test_v1_manifest_rejected_by_default_and_accepted_in_compat_mode():
    v1_manifest = {
        "step_id": "1",
        "num_tensors": 1,
        "tensor_names": ["layer.weight"],
    }

    # Default assembler rejects v1
    assembler_default = SnapshotAssembler()
    with pytest.raises(ValueError, match="protocol_version 2 required"):
        assembler_default.add("__manifest__:1", v1_manifest)

    # Compat assembler accepts v1
    assembler_compat = SnapshotAssembler(integrity="none_v1_compat")
    assert assembler_compat.add("__manifest__:1", v1_manifest) is None


def test_monotonic_guard_rejects_replayed_older_seq():
    assembler = SnapshotAssembler(initial_active=("run_A", 10, 1000.0))

    # Older seq from same run_A
    m_older, _, _, _ = _build_v2_fixtures(step_id="5", run_id="run_A", seq=5, created_at=1050.0)
    with pytest.raises(ValueError, match="stale_version"):
        assembler.add("__manifest__:5", canonical_json(m_older))

    # Same seq from same run_A
    m_same, _, _, _ = _build_v2_fixtures(step_id="10", run_id="run_A", seq=10, created_at=1050.0)
    with pytest.raises(ValueError, match="stale_version"):
        assembler.add("__manifest__:10", canonical_json(m_same))


def test_monotonic_guard_accepts_newer_run():
    assembler = SnapshotAssembler(initial_active=("run_A", 10, 1000.0))

    # Newer run with created_at > 1000.0
    m_new, frames, commit, _ = _build_v2_fixtures(step_id="1", run_id="run_B", seq=1, created_at=2000.0)
    assert assembler.add("__manifest__:1", canonical_json(m_new)) is None
    assert assembler.add("1:layer.weight", frames["layer.weight"]) is None
    res = assembler.add("__commit__:1", canonical_json(commit))
    assert res is not None
    assert assembler.active == ("run_B", 1, 2000.0)

    # Stale run with created_at <= 2000.0
    m_stale, _, _, _ = _build_v2_fixtures(step_id="50", run_id="run_C", seq=50, created_at=1500.0)
    with pytest.raises(ValueError, match="stale_version"):
        assembler.add("__manifest__:50", canonical_json(m_stale))


def test_size_guard_raises_before_any_send(monkeypatch):
    mock_producer = MagicMock()
    monkeypatch.setattr("trainer.KafkaProducer", MagicMock(return_value=mock_producer))
    config = {
        "kafka": {
            "lora_updates_topic": "test_topic",
            "bootstrap_servers": ["localhost:9092"],
            "max_request_size": 100,  # Strict 100-byte max size
            "integrity": "sha256",
        }
    }
    producer = LoRAProducer(config)

    large_tensor = torch.randn(50, 50)  # > 5000 bytes safetensors
    weights = {"big.layer": large_tensor}

    with pytest.raises(ValueError, match="max_request_size"):
        producer.send_weights(weights, step_id=1)

    # Size guard must fail closed before sending ANY message to Kafka
    assert mock_producer.send.call_count == 0
