"""
Unit tests for WP-28: Split deployment Mode B (clock offset merge) and Mode C (checkpoint watcher).
"""

import json
import time
from pathlib import Path
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

from utils.serving_router import DoubleBufferedAdapterRouter
from utils.adapter_manifest import create_and_save_adapter_artifact
from inference import checkpoint_watcher_thread
from benchmarks.merge_split_propagation import (
    merge_split_propagation_records,
    query_ntp_offset,
)


@pytest.fixture
def mock_tiny_model():
    cfg = GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=50)
    base = GPT2LMHeadModel(cfg)
    l_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"])
    model = get_peft_model(base, l_cfg)
    return model


def test_watcher_ignores_dir_without_committed_marker(tmp_path, mock_tiny_model):
    router = DoubleBufferedAdapterRouter(mock_tiny_model, None, "cpu")
    initial_version = router.active_version

    sub = tmp_path / "step_000100"
    sub.mkdir()
    (sub / "adapter_model.safetensors").write_bytes(b"dummy")

    import threading
    stop_event = threading.Event()
    stop_event.set()  # Run 1 iteration then stop immediately

    checkpoint_watcher_thread(str(tmp_path), router, poll_interval=0.01, stop_event=stop_event)
    assert router.active_version == initial_version, "Must ignore directory without COMMITTED marker"


def test_watcher_rejects_checksum_mismatch(tmp_path, mock_tiny_model):
    router = DoubleBufferedAdapterRouter(mock_tiny_model, None, "cpu")
    initial_version = router.active_version

    sub = tmp_path / "step_000200"
    sub.mkdir()
    (sub / "adapter_model.safetensors").write_bytes(b"corrupted bytes")
    manifest = {
        "adapter_version": "step_000200",
        "base_model": "test_base",
        "step": 200,
        "artifact_path": str(sub / "adapter_model.safetensors"),
        "checksum_sha256": "0" * 64,  # Intentionally mismatched
        "status": "APPROVED",
    }
    (sub / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (sub / "COMMITTED").write_text(json.dumps({"step": 200}), encoding="utf-8")

    import threading
    stop_event = threading.Event()
    stop_event.set()

    checkpoint_watcher_thread(str(tmp_path), router, poll_interval=0.01, stop_event=stop_event)
    assert router.active_version == initial_version, "Must reject corrupted checkpoint with checksum mismatch"


def test_watcher_applies_new_committed_checkpoint_once(tmp_path, mock_tiny_model):
    router = DoubleBufferedAdapterRouter(mock_tiny_model, None, "cpu")
    weights = get_peft_model_state_dict(mock_tiny_model)

    sub = tmp_path / "step_000300"
    create_and_save_adapter_artifact(
        state_dict=weights,
        output_dir=str(sub),
        adapter_version="step_000300",
        base_model="gpt2",
        step=300,
        use_safetensors=True,
    )

    assert (sub / "COMMITTED").exists(), "Artifact creation must write COMMITTED marker"

    import threading
    stop_event = threading.Event()
    stop_event.set()

    checkpoint_watcher_thread(str(tmp_path), router, poll_interval=0.01, stop_event=stop_event)
    assert router.active_version == "step_000300", "Router should have applied committed checkpoint"
    assert router.swap_count == 1

    # Run again: should not re-apply the same directory
    stop_event.set()
    checkpoint_watcher_thread(str(tmp_path), router, poll_interval=0.01, stop_event=stop_event)
    assert router.swap_count == 1, "Must not re-apply already applied directory"


def test_watcher_retries_transient_checksum_mismatch(tmp_path, mock_tiny_model):
    router = DoubleBufferedAdapterRouter(mock_tiny_model, None, "cpu")
    initial_version = router.active_version

    weights = get_peft_model_state_dict(mock_tiny_model)
    sub = tmp_path / "step_000400"

    # Save artifact with correct manifest and COMMITTED
    manifest, _ = create_and_save_adapter_artifact(
        state_dict=weights,
        output_dir=str(sub),
        adapter_version="step_000400",
        base_model="gpt2",
        step=400,
        use_safetensors=True,
    )

    # Now intentionally corrupt the weights file to simulate partial cloud sync download
    weights_file = sub / "adapter_model.safetensors"
    valid_bytes = weights_file.read_bytes()
    weights_file.write_bytes(b"partial_sync_bytes")

    import threading
    stop_event = threading.Event()

    # Run watcher in background thread
    t = threading.Thread(target=checkpoint_watcher_thread, args=(str(tmp_path), router, 0.05, stop_event))
    t.start()

    time.sleep(0.1)
    # Check that watcher did NOT apply corrupted file
    assert router.active_version == initial_version

    # Now restore the valid bytes (simulating Google Drive sync completing download)
    weights_file.write_bytes(valid_bytes)
    time.sleep(0.2)

    stop_event.set()
    t.join(timeout=2.0)

    # The watcher should have retried and successfully applied the valid weights!
    assert router.active_version == "step_000400"
    assert router.swap_count == 1


def test_merge_split_propagation_offset_correction():
    trainer_recs = [
        {"run_id": "split_run", "seq": 1, "t_publish_start": 100.0},
        {"run_id": "split_run", "seq": 2, "t_publish_start": 200.0},
    ]
    # Server clock is running 2.0s fast (local = server + 2.0 -> offset = -2.0)
    server_recs = [
        {"run_id": "split_run", "seq": 1, "t_applied": 103.5},
        {"run_id": "split_run", "seq": 2, "t_applied": 204.0},
    ]

    # Trainer offset = 0.5s, Server offset = -1.5s
    # Skew correction: t_applied_corrected = t_applied - server_offset + trainer_offset
    # For seq 1: 103.5 - (-1.5) + (0.5) = 105.5
    # Propagation: 105.5 - 100.0 = 5.5s
    summary = merge_split_propagation_records(
        trainer_records=trainer_recs,
        server_records=server_recs,
        trainer_offset_start=0.5,
        trainer_offset_end=0.5,
        server_offset_start=-1.5,
        server_offset_end=-1.5,
    )

    assert summary["status"] == "complete"
    assert summary["matched_count"] == 2
    assert summary["drift_warning_flag"] is False
    assert abs(summary["items"][0]["propagation_s"] - 5.5) < 1e-4
    assert abs(summary["items"][1]["propagation_s"] - 6.0) < 1e-4
    assert abs(summary["propagation_s_p50"] - 5.5) < 1e-4


def test_merge_split_propagation_drift_flag():
    trainer_recs = [{"run_id": "r", "seq": 1, "t_publish_start": 10.0}]
    server_recs = [{"run_id": "r", "seq": 1, "t_applied": 12.0}]

    # Drift of 60ms between start and end (>50ms threshold)
    summary = merge_split_propagation_records(
        trainer_records=trainer_recs,
        server_records=server_recs,
        trainer_offset_start=0.100,
        trainer_offset_end=0.160,
        server_offset_start=0.0,
        server_offset_end=0.0,
    )
    assert summary["drift_warning_flag"] is True
    assert summary["trainer_clock_drift_s"] == 0.06
