import io
import json
import pickle
import pytest
import torch
import safetensors.torch
from unittest.mock import MagicMock

from inference import process_adapter_record, stream_state_global, app
from utils.stream_protocol import SnapshotAssembler


def test_process_adapter_record_truncated_safetensors():
    assembler = SnapshotAssembler()
    counters = {"rejected_frames": 0, "rejected_by_reason": {}}
    
    # Create valid safetensors bytes and truncate them
    tensor = torch.randn(4, 4)
    full_bytes = safetensors.torch.save({"weight": tensor})
    truncated_bytes = full_bytes[:15]
    
    res = process_adapter_record(assembler, "1:layer.weight", truncated_bytes, counters=counters)
    assert res is None
    assert counters["rejected_frames"] == 1
    assert "ValueError" in counters["rejected_by_reason"]
    assert counters["rejected_by_reason"]["ValueError"] == 1


def test_process_adapter_record_random_bytes():
    assembler = SnapshotAssembler()
    counters = {"rejected_frames": 0, "rejected_by_reason": {}}
    
    random_bytes = b"\xde\xad\xbe\xef\x01\x02\x03\x04\xff\xfe\x00\x00corruptdata"
    res = process_adapter_record(assembler, "1:layer.weight", random_bytes, counters=counters)
    assert res is None
    assert counters["rejected_frames"] == 1
    assert "ValueError" in counters["rejected_by_reason"]
    assert counters["rejected_by_reason"]["ValueError"] == 1


def test_process_adapter_record_pickle_blob():
    assembler = SnapshotAssembler()
    counters = {"rejected_frames": 0, "rejected_by_reason": {}}
    
    pickle_blob = pickle.dumps({"malicious": "payload", "a": 1})
    res = process_adapter_record(assembler, "1:layer.weight", pickle_blob, counters=counters)
    assert res is None
    assert counters["rejected_frames"] == 1
    assert "ValueError" in counters["rejected_by_reason"]
    assert counters["rejected_by_reason"]["ValueError"] == 1


def test_process_adapter_record_valid_sequence():
    assembler = SnapshotAssembler()
    counters = {"rejected_frames": 0, "rejected_by_reason": {}}
    step_id = "42"
    tensor_name = "model.layers.0.self_attn.q_proj.weight"
    
    # 1. Manifest
    manifest_payload = {
        "step_id": step_id,
        "num_tensors": 1,
        "tensor_names": [tensor_name],
    }
    manifest_bytes = json.dumps(manifest_payload).encode("utf-8")
    res_manifest = process_adapter_record(
        assembler, f"__manifest__:{step_id}", manifest_bytes, counters=counters
    )
    assert res_manifest is None
    assert counters["rejected_frames"] == 0
    
    # 2. Tensor
    t = torch.randn(8, 8)
    tensor_bytes = safetensors.torch.save({"weight": t})
    res_tensor = process_adapter_record(
        assembler, f"{step_id}:{tensor_name}", tensor_bytes, counters=counters
    )
    assert res_tensor is None
    assert counters["rejected_frames"] == 0
    
    # 3. Commit
    commit_payload = {
        "step_id": step_id,
        "status": "committed",
    }
    commit_bytes = json.dumps(commit_payload).encode("utf-8")
    res_commit = process_adapter_record(
        assembler, f"__commit__:{step_id}", commit_bytes, counters=counters
    )
    assert res_commit is not None
    committed_step, committed_batch = res_commit
    assert committed_step == step_id
    assert tensor_name in committed_batch
    assert torch.allclose(committed_batch[tensor_name], t)
    assert counters["rejected_frames"] == 0


def test_process_adapter_record_fake_assembler_exception():
    fake_assembler = MagicMock()
    fake_assembler.add.side_effect = RuntimeError("Custom assembler boom")
    counters = {"rejected_frames": 0, "rejected_by_reason": {}}
    
    payload = json.dumps({"step_id": "1"}).encode("utf-8")
    res = process_adapter_record(fake_assembler, "__manifest__:1", payload, counters=counters)
    assert res is None
    assert counters["rejected_frames"] == 1
    assert counters["rejected_by_reason"]["RuntimeError"] == 1


def test_process_adapter_record_none_counters():
    assembler = SnapshotAssembler()
    random_bytes = b"garbage_data"
    # Should not raise even when counters is None
    res = process_adapter_record(assembler, "1:layer.weight", random_bytes, counters=None)
    assert res is None


def test_process_adapter_record_missing_key_or_value():
    assembler = SnapshotAssembler()
    counters = {"rejected_frames": 0, "rejected_by_reason": {}}
    
    res1 = process_adapter_record(assembler, None, b"valid_looking_bytes", counters=counters)
    assert res1 is None
    assert counters["rejected_frames"] == 1
    
    res2 = process_adapter_record(assembler, "1:layer.weight", None, counters=counters)
    assert res2 is None
    assert counters["rejected_frames"] == 2


def test_metrics_endpoint_exposes_stream_counters():
    with app.test_client() as client:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "stream" in data
        assert "counters" in data["stream"]
        assert "rejected_frames" in data["stream"]["counters"]
        assert "rejected_by_reason" in data["stream"]["counters"]
