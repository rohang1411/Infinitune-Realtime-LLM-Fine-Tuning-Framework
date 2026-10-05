"""
tests/test_adapter_manifest.py
-----------------------------------------------------------------------------
Unit tests for production adapter manifest & safetensors serialization protocol.
"""

import os
import tempfile
import pytest
import torch

from utils.adapter_manifest import (
    save_adapter_weights,
    load_adapter_weights,
    compute_file_sha256,
    AdapterManifest,
    ChecksumMismatchError,
    InvalidManifestError,
    create_and_save_adapter_artifact,
)


@pytest.fixture
def dummy_weights():
    return {
        "lora_A.weight": torch.randn(4, 16),
        "lora_B.weight": torch.randn(16, 4),
    }


def test_save_and_load_weights_safetensors(dummy_weights):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "adapter.safetensors")
        saved_path = save_adapter_weights(dummy_weights, path, use_safetensors=True)
        assert os.path.exists(saved_path)

        loaded = load_adapter_weights(saved_path)
        assert set(loaded.keys()) == set(dummy_weights.keys())
        for k in dummy_weights:
            assert torch.allclose(dummy_weights[k], loaded[k])


def test_checksum_computation_and_verification(dummy_weights):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "adapter.safetensors")
        saved_path = save_adapter_weights(dummy_weights, path, use_safetensors=True)
        checksum = compute_file_sha256(saved_path)
        assert len(checksum) == 64  # SHA-256 hex string

        # Load with correct checksum passes
        loaded = load_adapter_weights(saved_path, expected_checksum=checksum)
        assert len(loaded) == 2

        # Load with incorrect checksum raises ChecksumMismatchError
        with pytest.raises(ChecksumMismatchError):
            load_adapter_weights(saved_path, expected_checksum="0" * 64)


def test_tamper_detection(dummy_weights):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "adapter.safetensors")
        saved_path = save_adapter_weights(dummy_weights, path, use_safetensors=True)
        checksum = compute_file_sha256(saved_path)

        # Tamper with the saved file by corrupting bytes at the end
        with open(saved_path, "ab") as f:
            f.write(b"CORRUPTED_BYTES")

        with pytest.raises(ChecksumMismatchError):
            load_adapter_weights(saved_path, expected_checksum=checksum)


def test_manifest_serialization_and_validation(dummy_weights):
    with tempfile.TemporaryDirectory() as tmpdir:
        weight_path = os.path.join(tmpdir, "adapter.safetensors")
        saved_path = save_adapter_weights(dummy_weights, weight_path, use_safetensors=True)
        checksum = compute_file_sha256(saved_path)

        manifest = AdapterManifest(
            adapter_version="v1.0",
            base_model="Qwen/Qwen2.5-1.5B",
            step=100,
            artifact_path=saved_path,
            checksum_sha256=checksum,
            canary_metrics={"eval_loss": 0.12},
            status="APPROVED",
        )

        assert manifest.verify_artifact() is True

        # Test JSON roundtrip
        json_str = manifest.to_json()
        restored = AdapterManifest.from_json(json_str)
        assert restored.adapter_version == "v1.0"
        assert restored.step == 100
        assert restored.checksum_sha256 == checksum
        assert restored.canary_metrics["eval_loss"] == 0.12

        # Test loading weights through manifest
        weights = restored.load_weights()
        assert "lora_A.weight" in weights


def test_invalid_manifest_missing_keys():
    incomplete_data = {
        "manifest_version": "1.0",
        "adapter_version": "v1.0",
        # Missing step, base_model, artifact_path, checksum_sha256, status
    }
    with pytest.raises(InvalidManifestError):
        AdapterManifest.from_dict(incomplete_data)


def test_create_and_save_adapter_artifact_e2e(dummy_weights):
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest, manifest_path = create_and_save_adapter_artifact(
            state_dict=dummy_weights,
            output_dir=tmpdir,
            adapter_version="step_500",
            base_model="distilgpt2",
            step=500,
            canary_metrics={"canary_loss": 0.25},
        )

        assert os.path.exists(manifest_path)
        assert manifest.verify_artifact() is True
        loaded_manifest = AdapterManifest.from_file(manifest_path)
        assert loaded_manifest.step == 500
        assert loaded_manifest.base_model == "distilgpt2"
        weights = loaded_manifest.load_weights()
        assert len(weights) == 2
