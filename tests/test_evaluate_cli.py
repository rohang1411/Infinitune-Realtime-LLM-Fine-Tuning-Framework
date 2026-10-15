import os
import sys
import yaml
import pytest
import torch
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


def test_step_flag_resolves_previous_run(monkeypatch, tmp_path, make_config_file):
    ckpt_dir = tmp_path / "checkpoints" / "run_20260413_test" / "step_000050"
    ckpt_dir.mkdir(parents=True)

    dummy_config = {
        "project": {"name": "test_proj", "output_dir": str(tmp_path)},
        "model": {"name": "test_model"},
        "dataset": {"name": "test_ds"},
        "training": {"max_steps": 200},
        "evaluation": {"enabled": True},
    }
    cfg_path = make_config_file(dummy_config)

    recorded_resolves = []
    def fake_resolve(self, target, run_name=None):
        recorded_resolves.append((target, run_name))
        return str(ckpt_dir)

    monkeypatch.setattr(CheckpointManager, "resolve_checkpoint_path", fake_resolve)
    monkeypatch.setattr(CheckpointManager, "list_runs", lambda self: ["run_20260414_new", "run_20260413_test"])
    monkeypatch.setattr(evaluate, "evaluate_checkpoint", lambda *args, **kwargs: {"loss": 0.42})

    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--config", cfg_path, "--step", "50"])

    with pytest.raises(SystemExit) as excinfo:
        evaluate.main()
    assert excinfo.value.code == 0
    assert len(recorded_resolves) >= 1
    assert recorded_resolves[0][0] == 50


def test_failure_gives_exit_code_2(monkeypatch, tmp_path, make_config_file):
    ckpt_dir = tmp_path / "checkpoints" / "step_000050"
    ckpt_dir.mkdir(parents=True)

    dummy_config = {
        "project": {"name": "test_proj", "output_dir": str(tmp_path)},
        "model": {"name": "test_model"},
        "dataset": {"name": "test_ds"},
        "training": {"max_steps": 200},
        "evaluation": {"enabled": True},
    }
    cfg_path = make_config_file(dummy_config)

    monkeypatch.setattr(CheckpointManager, "resolve_checkpoint_path", lambda self, target, run_name=None: str(ckpt_dir))
    monkeypatch.setattr(CheckpointManager, "list_runs", lambda self: [])

    def fake_eval_boom(*args, **kwargs):
        raise RuntimeError("Simulated evaluator explosion")

    monkeypatch.setattr(evaluate, "evaluate_checkpoint", fake_eval_boom)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--config", cfg_path, "--step", "50"])

    with pytest.raises(SystemExit) as excinfo:
        evaluate.main()
    assert excinfo.value.code == 2


def test_steps_passed_to_evaluators_are_real(monkeypatch, tmp_path, make_config_file):
    dummy_config = {
        "project": {"name": "test_proj", "output_dir": str(tmp_path)},
        "model": {"name": "test_model"},
        "dataset": {"name": "test_ds"},
        "training": {"max_steps": 1000},
        "evaluation": {"enabled": True, "eval_interval": 50},
    }
    cfg_path = make_config_file(dummy_config)

    fake_ckpts = [
        {"name": "step_000050", "step": 50, "path": str(tmp_path / "s50"), "timestamp": None},
        {"name": "step_000100", "step": 100, "path": str(tmp_path / "s100"), "timestamp": None},
        {"name": "final", "step": "final", "path": str(tmp_path / "final"), "timestamp": None},
    ]

    monkeypatch.setattr(CheckpointManager, "list_runs", lambda self: ["run1"])
    monkeypatch.setattr(CheckpointManager, "list_checkpoints", lambda self, run_name=None: fake_ckpts)

    recorded_steps = []
    def fake_eval(ckpt_path, config, config_file, evaluator_bundle=None,
                  run_quantitative=True, run_qualitative=True, step=0, device=None):
        recorded_steps.append(step)
        return {"eval_loss": 0.5}

    monkeypatch.setattr(evaluate, "evaluate_checkpoint", fake_eval)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--config", cfg_path, "--all-checkpoints"])

    with pytest.raises(SystemExit) as excinfo:
        evaluate.main()
    assert excinfo.value.code == 0
    # Base model gets step 0, then step 50, step 100, and final gets max_steps (1000)
    assert recorded_steps == [0, 50, 100, 1000]


def test_step_flag_accepts_final(monkeypatch, tmp_path, make_config_file):
    ckpt_dir = tmp_path / "checkpoints" / "final"
    ckpt_dir.mkdir(parents=True)

    dummy_config = {
        "project": {"name": "test_proj", "output_dir": str(tmp_path)},
        "model": {"name": "test_model"},
        "dataset": {"name": "test_ds"},
        "training": {"max_steps": 1563},
        "evaluation": {"enabled": True},
    }
    cfg_path = make_config_file(dummy_config)

    monkeypatch.setattr(CheckpointManager, "resolve_checkpoint_path", lambda self, target, run_name=None: str(ckpt_dir))
    monkeypatch.setattr(CheckpointManager, "list_runs", lambda self: ["run10"])

    recorded_steps = []
    def fake_eval(ckpt_path, config, config_file, evaluator_bundle=None,
                  run_quantitative=True, run_qualitative=True, step=0, device=None):
        recorded_steps.append(step)
        return {"eval_loss": 0.3}

    monkeypatch.setattr(evaluate, "evaluate_checkpoint", fake_eval)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--config", cfg_path, "--step", "final"])

    with pytest.raises(SystemExit) as excinfo:
        evaluate.main()
    assert excinfo.value.code == 0
    assert recorded_steps == [1563]
