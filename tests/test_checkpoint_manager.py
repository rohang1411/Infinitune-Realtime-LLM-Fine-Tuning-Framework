"""Unit tests for CheckpointManager sorting, listing, and latest resolution.

Verifies WP-02 requirements:
- Checkpoints are sorted by run name first, then by step (numbered ascending, final last).
- latest_checkpoint() resolves the newest run's final if present, otherwise highest step.
- resolve_checkpoint_path() returns the newest run match for equal step numbers.
- Ordering is independent of filesystem directory creation order.
- Legacy flat checkpoint layout remains supported.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from utils.checkpoint_manager import CheckpointManager


def _create_fake_checkpoint(dir_path: Path, step_name: str, timestamp: str | None = None) -> Path:
    step_dir = dir_path / step_name
    step_dir.mkdir(parents=True, exist_ok=True)
    # _load_step_dir requires adapter_config.json
    (step_dir / "adapter_config.json").write_text(json.dumps({"r": 8}), encoding="utf-8")
    meta = {"timestamp": timestamp or "2026-04-10T12:00:00"}
    (step_dir / "checkpoint_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return step_dir


@pytest.fixture
def mock_ckpt_mgr(tmp_path: Path) -> CheckpointManager:
    config = {
        "project": {"output_dir": str(tmp_path)},
        "model": {"name": "distilgpt2"},
        "dataset": {"name": "imdb"},
        "lora": {"r": 8},
    }
    return CheckpointManager(config=config)


def test_latest_prefers_newest_run_even_if_older_run_has_higher_step(mock_ckpt_mgr: CheckpointManager):
    root = Path(mock_ckpt_mgr.checkpoint_root)
    run_a = root / "run_20260410-100000_1111"
    run_b = root / "run_20260411-100000_2222"

    _create_fake_checkpoint(run_a, "step_000500")
    _create_fake_checkpoint(run_b, "step_000050")

    latest = mock_ckpt_mgr.latest_checkpoint()
    assert latest is not None
    assert latest["run_name"] == "run_20260411-100000_2222"
    assert latest["step"] == 50
    assert "step_000050" in latest["path"]


def test_latest_prefers_final_within_newest_run(mock_ckpt_mgr: CheckpointManager):
    root = Path(mock_ckpt_mgr.checkpoint_root)
    run_a = root / "run_20260410-100000_1111"
    run_b = root / "run_20260411-100000_2222"

    _create_fake_checkpoint(run_a, "step_000500")
    _create_fake_checkpoint(run_b, "step_000050")
    _create_fake_checkpoint(run_b, "step_000100")
    _create_fake_checkpoint(run_b, "final")

    latest = mock_ckpt_mgr.latest_checkpoint()
    assert latest is not None
    assert latest["run_name"] == "run_20260411-100000_2222"
    assert latest["step"] == "final"
    assert latest["name"] == "final"


def test_resolve_step_returns_newest_run_match(mock_ckpt_mgr: CheckpointManager):
    root = Path(mock_ckpt_mgr.checkpoint_root)
    run_a = root / "run_20260410-100000_1111"
    run_b = root / "run_20260411-100000_2222"

    _create_fake_checkpoint(run_a, "step_000100")
    _create_fake_checkpoint(run_b, "step_000100")

    resolved = mock_ckpt_mgr.resolve_checkpoint_path(100)
    assert resolved is not None
    assert "run_20260411-100000_2222" in resolved


def test_order_is_independent_of_directory_creation_order(mock_ckpt_mgr: CheckpointManager):
    root = Path(mock_ckpt_mgr.checkpoint_root)
    # Create run_b first on disk, then run_a
    run_b = root / "run_20260411-100000_2222"
    _create_fake_checkpoint(run_b, "step_000050")

    run_a = root / "run_20260410-100000_1111"
    _create_fake_checkpoint(run_a, "step_000050")

    ckpts = mock_ckpt_mgr.list_checkpoints()
    assert len(ckpts) == 2
    assert ckpts[0]["run_name"] == "run_20260410-100000_1111"
    assert ckpts[1]["run_name"] == "run_20260411-100000_2222"

    runs = mock_ckpt_mgr.list_runs()
    assert runs == ["run_20260410-100000_1111", "run_20260411-100000_2222"]


def test_legacy_flat_layout_compatibility(mock_ckpt_mgr: CheckpointManager):
    root = Path(mock_ckpt_mgr.checkpoint_root)
    # Step dirs directly under root (no run_ prefix)
    _create_fake_checkpoint(root, "step_000050")
    _create_fake_checkpoint(root, "final")

    # Plus a newer hierarchical run
    run_b = root / "run_20260411-100000_2222"
    _create_fake_checkpoint(run_b, "step_000020")

    ckpts = mock_ckpt_mgr.list_checkpoints()
    assert len(ckpts) == 3
    # Flat checkpoints sort before run_ dirs because run_name is None ("")
    assert ckpts[0]["run_name"] is None
    assert ckpts[0]["step"] == 50
    assert ckpts[1]["run_name"] is None
    assert ckpts[1]["step"] == "final"
    assert ckpts[2]["run_name"] == "run_20260411-100000_2222"
    assert ckpts[2]["step"] == 20

    # latest_checkpoint resolves the newest run (run_b)
    latest = mock_ckpt_mgr.latest_checkpoint()
    assert latest is not None
    assert latest["run_name"] == "run_20260411-100000_2222"
    assert latest["step"] == 20


def test_load_respects_precision_and_revision(tmp_path: Path, monkeypatch):
    import torch
    from unittest.mock import MagicMock

    config = {
        "project": {"output_dir": str(tmp_path)},
        "model": {"name": "distilgpt2", "precision": "bf16", "revision": "test_sha_123"},
        "dataset": {"name": "imdb"},
        "lora": {"r": 8},
    }
    ckpt_mgr = CheckpointManager(config=config)
    ckpt_dir = _create_fake_checkpoint(tmp_path, "step_000100")

    captured_kwargs = {}

    mock_base = MagicMock()
    mock_base.to.return_value = mock_base
    mock_peft = MagicMock()

    def fake_from_pretrained(model_name, **kwargs):
        captured_kwargs.update(kwargs)
        return mock_base

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", fake_from_pretrained)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: MagicMock())
    monkeypatch.setattr("peft.PeftModel.from_pretrained", lambda b, c: mock_peft)

    model, tokenizer = ckpt_mgr.load(str(ckpt_dir), device="cpu")

    assert captured_kwargs.get("torch_dtype") == torch.bfloat16
    assert captured_kwargs.get("revision") == "test_sha_123"
    mock_base.to.assert_called_with("cpu")
    assert model == mock_peft


def test_load_defaults_fp32(tmp_path: Path, monkeypatch):
    import torch
    from unittest.mock import MagicMock

    config = {
        "project": {"output_dir": str(tmp_path)},
        "model": {"name": "distilgpt2"},
        "dataset": {"name": "imdb"},
        "lora": {"r": 8},
    }
    ckpt_mgr = CheckpointManager(config=config)
    ckpt_dir = _create_fake_checkpoint(tmp_path, "step_000100")

    captured_kwargs = {}

    mock_base = MagicMock()
    mock_base.to.return_value = mock_base
    mock_peft = MagicMock()

    def fake_from_pretrained(model_name, **kwargs):
        captured_kwargs.update(kwargs)
        return mock_base

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", fake_from_pretrained)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: MagicMock())
    monkeypatch.setattr("peft.PeftModel.from_pretrained", lambda b, c: mock_peft)

    ckpt_mgr.load(str(ckpt_dir), device="cpu")

    assert captured_kwargs.get("torch_dtype") == torch.float32
    assert "revision" not in captured_kwargs

