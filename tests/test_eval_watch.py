import json
import os
import sys
import pytest
import yaml
import evaluate
from utils.checkpoint_manager import CheckpointManager


@pytest.fixture
def make_config_file(tmp_path):
    def _make(config_dict):
        cfg_file = tmp_path / "test_config.yaml"
        with open(cfg_file, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f)
        return str(cfg_file)
    return _make


def _create_checkpoint(ckpt_dir, step=10, complete=True):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (ckpt_dir / "adapter_model.safetensors").write_bytes(b"dummy_weights")
    if complete:
        meta = {
            "step": step,
            "loss": 0.42,
            "timestamp": "2026-04-14T10:00:00",
        }
        (ckpt_dir / "checkpoint_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_watch_evaluates_each_checkpoint_once(monkeypatch, tmp_path, make_config_file):
    run_dir = tmp_path / "checkpoints" / "test_model__test_ds" / "run_watch"
    ckpt1 = run_dir / "step_000010"
    ckpt2 = run_dir / "step_000020"
    _create_checkpoint(ckpt1, step=10, complete=True)
    _create_checkpoint(ckpt2, step=20, complete=True)

    dummy_config = {
        "project": {"name": "test_proj", "output_dir": str(tmp_path)},
        "model": {"name": "test_model"},
        "dataset": {"name": "test_ds"},
        "training": {"max_steps": 200},
        "evaluation": {"enabled": True},
    }
    cfg_path = make_config_file(dummy_config)

    evaluated_paths = []
    def fake_evaluate_checkpoint(ckpt_path, *args, **kwargs):
        evaluated_paths.append(str(ckpt_path))
        return {"loss": 0.35, "eval_accuracy": 0.88}

    monkeypatch.setattr(evaluate, "evaluate_checkpoint", fake_evaluate_checkpoint)
    monkeypatch.setattr(evaluate, "_generate_plots", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--config",
            cfg_path,
            "--run",
            "run_watch",
            "--watch",
            "--poll-seconds",
            "0.01",
            "--max-iterations",
            "2",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        evaluate.main()
    assert excinfo.value.code == 0

    assert len(evaluated_paths) == 2
    assert any("step_000010" in p for p in evaluated_paths)
    assert any("step_000020" in p for p in evaluated_paths)

    state_file = tmp_path / "eval_watch_state.json"
    metrics_file = tmp_path / "watch_metrics.json"

    assert state_file.exists()
    assert metrics_file.exists()

    state = json.loads(state_file.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))

    assert len(metrics["checkpoints"]) == 2
    assert "queue_delay_p50" in metrics["summary"]
    assert "eval_latency_p50" in metrics["summary"]
    assert metrics["summary"]["evaluated_count"] == 2


def test_watch_skips_incomplete_checkpoint_until_meta_exists(monkeypatch, tmp_path, make_config_file):
    run_dir = tmp_path / "checkpoints" / "test_model__test_ds" / "run_watch"
    ckpt1 = run_dir / "step_000010"
    _create_checkpoint(ckpt1, step=10, complete=False)

    dummy_config = {
        "project": {"name": "test_proj", "output_dir": str(tmp_path)},
        "model": {"name": "test_model"},
        "dataset": {"name": "test_ds"},
        "training": {"max_steps": 200},
        "evaluation": {"enabled": True},
    }
    cfg_path = make_config_file(dummy_config)

    evaluated_paths = []
    def fake_evaluate_checkpoint(ckpt_path, *args, **kwargs):
        evaluated_paths.append(str(ckpt_path))
        return {"loss": 0.40}

    monkeypatch.setattr(evaluate, "evaluate_checkpoint", fake_evaluate_checkpoint)
    monkeypatch.setattr(evaluate, "_generate_plots", lambda *args, **kwargs: None)

    # Iteration 1: Incomplete checkpoint (meta missing) -> should skip
    config = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    summary1 = evaluate.run_watch_mode(
        config,
        cfg_path,
        run_name="run_watch",
        poll_seconds=0.01,
        max_iterations=1,
    )
    assert summary1["evaluated_count"] == 0
    assert len(evaluated_paths) == 0

    # Iteration 2: Write meta to make checkpoint complete -> should evaluate
    meta = {"step": 10, "loss": 0.42, "timestamp": "2026-04-14T10:00:00"}
    (ckpt1 / "checkpoint_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    summary2 = evaluate.run_watch_mode(
        config,
        cfg_path,
        run_name="run_watch",
        poll_seconds=0.01,
        max_iterations=1,
    )
    assert summary2["evaluated_count"] == 1
    assert len(evaluated_paths) == 1
    assert "step_000010" in evaluated_paths[0]


def test_watch_resumes_from_state_file(monkeypatch, tmp_path, make_config_file):
    run_dir = tmp_path / "checkpoints" / "test_model__test_ds" / "run_watch"
    ckpt1 = run_dir / "step_000010"
    ckpt2 = run_dir / "step_000020"
    _create_checkpoint(ckpt1, step=10, complete=True)
    _create_checkpoint(ckpt2, step=20, complete=True)

    dummy_config = {
        "project": {"name": "test_proj", "output_dir": str(tmp_path)},
        "model": {"name": "test_model"},
        "dataset": {"name": "test_ds"},
        "training": {"max_steps": 200},
        "evaluation": {"enabled": True},
    }
    cfg_path = make_config_file(dummy_config)

    # Pre-populate state file marking step_000010 as already evaluated
    state_file = tmp_path / "eval_watch_state.json"
    state_file.write_text(
        json.dumps({
            "done_checkpoints": ["step_000010", str(ckpt1.resolve())],
            "runs": {"run_watch": ["step_000010"]},
        }),
        encoding="utf-8",
    )

    evaluated_paths = []
    def fake_evaluate_checkpoint(ckpt_path, *args, **kwargs):
        evaluated_paths.append(str(ckpt_path))
        return {"loss": 0.30}

    monkeypatch.setattr(evaluate, "evaluate_checkpoint", fake_evaluate_checkpoint)
    monkeypatch.setattr(evaluate, "_generate_plots", lambda *args, **kwargs: None)

    config = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    summary = evaluate.run_watch_mode(
        config,
        cfg_path,
        run_name="run_watch",
        poll_seconds=0.01,
        max_iterations=1,
    )

    assert summary["evaluated_count"] == 1
    assert len(evaluated_paths) == 1
    assert "step_000020" in evaluated_paths[0]
