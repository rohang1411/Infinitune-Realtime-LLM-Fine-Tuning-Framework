"""Failure-path tests for benchmark provenance and adapter integrity."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmarks.run import Run
from benchmarks.repro_eval import score_predictions, paired_bootstrap
from inference import app, deserialize_tensor, stream_state_global
from trainer import finish_optimizer_step
from utils.stream_protocol import SnapshotAssembler
from utils.serving_router import DoubleBufferedAdapterRouter


def test_snapshot_rejects_incomplete_commit_and_undeclared_tensor():
    assembler = SnapshotAssembler(integrity="none_v1_compat")
    manifest = {"step_id": "7", "num_tensors": 2,
                "tensor_names": ["a", "b"]}
    assert assembler.add("__manifest__:7", manifest) is None
    assert assembler.add("7:a", torch.ones(1)) is None
    with pytest.raises(ValueError, match="incomplete"):
        assembler.add("__commit__:7", {"step_id": "7", "status": "committed"})
    with pytest.raises(ValueError, match="Undeclared"):
        assembler.add("7:b", torch.ones(1))


def test_snapshot_complete_only_after_commit():
    assembler = SnapshotAssembler(integrity="none_v1_compat")
    assert assembler.add("__manifest__:8", {"step_id": "8", "num_tensors": 1,
                                            "tensor_names": ["x"]}) is None
    assert assembler.add("8:x", torch.ones(2)) is None
    step, weights = assembler.add("__commit__:8", {"step_id": "8", "status": "committed"})
    assert step == "8" and list(weights) == ["x"]


def test_untrusted_pickle_bytes_are_rejected():
    import io
    buffer = io.BytesIO()
    torch.save(torch.ones(1), buffer)
    with pytest.raises(ValueError, match="neither"):
        deserialize_tensor(buffer.getvalue())


def test_optimizer_step_respects_accumulation_and_scheduler():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    before = model.weight.detach().clone()
    model.weight.grad = torch.ones_like(model.weight) * 2
    did_step, norm = finish_optimizer_step(model, optimizer, scheduler,
                                           SimpleNamespace(scaler=None),
                                           gradient_rescale=.5)
    assert did_step and norm == pytest.approx(1.0)
    assert not torch.equal(before, model.weight)
    assert scheduler.last_epoch == 1
    assert model.weight.grad is None


def test_run_manifest_records_failure_then_resumes(tmp_path):
    run = Run(tmp_path / "one", {"suite": "failure"})
    assert run.stage("stage", lambda: 1 / 0) is None
    entry = run.manifest["stages"]["stage"]
    assert entry["status"] == "failed"
    assert "ZeroDivisionError" in Path(entry["traceback_path"]).read_text()
    assert run.stage("stage", lambda: {"score": .5}) == {"score": .5}
    reloaded = Run(tmp_path / "one", {"suite": "failure"})
    assert reloaded.stage("stage", lambda: 1 / 0) == {"score": .5}


def test_resume_recomputes_missing_or_changed_raw_artifact(tmp_path):
    from benchmarks.repro_eval import file_sha256
    run = Run(tmp_path / "raw", {"suite": "raw"})
    raw = run.directory / "predictions.jsonl"

    def operation():
        raw.write_text('{"id":"1"}\n', encoding="utf-8")
        return {"predictions_path": str(raw),
                "predictions_sha256": file_sha256(raw)}

    run.stage("prediction", operation)
    raw.write_text("tampered\n", encoding="utf-8")
    assert run.stage("prediction", operation)["predictions_sha256"] == file_sha256(raw)
    assert raw.read_text(encoding="utf-8") == '{"id":"1"}\n'


def test_code_change_archives_old_scores(tmp_path, monkeypatch):
    import benchmarks.run as runner
    monkeypatch.setattr(runner, "code_fingerprint", lambda: "old")
    directory = tmp_path / "run"
    run = Run(directory, {"suite": "audit"})
    run.stage("metric", lambda: {"accuracy": .9})
    (directory / "predictions").mkdir()
    (directory / "predictions" / "raw.jsonl").write_text("old raw output")
    monkeypatch.setattr(runner, "code_fingerprint", lambda: "new")
    new_run = Run(directory, {"suite": "audit"})
    assert new_run.manifest["stages"] == {}
    archives = list((directory / "attempt_archive").iterdir())
    assert len(archives) == 1
    assert (archives[0] / "predictions" / "raw.jsonl").read_text() == "old raw output"
    assert new_run.stage("metric", lambda: {"accuracy": .5})["accuracy"] == .5


def test_dependency_change_archives_old_scores(tmp_path, monkeypatch):
    import benchmarks.run as runner
    monkeypatch.setattr(runner, "installed_packages", lambda: {"torch": "old"})
    directory = tmp_path / "dependency_run"
    Run(directory, {"suite": "audit"}).stage("metric", lambda: {"accuracy": .9})
    monkeypatch.setattr(runner, "installed_packages", lambda: {"torch": "new"})
    resumed = Run(directory, {"suite": "audit"})
    assert resumed.manifest["stages"] == {}
    assert resumed.manifest["environment"]["packages"] == {"torch": "new"}
    assert list((directory / "attempt_archive").iterdir())


def test_external_checkpoint_change_archives_old_scores(tmp_path):
    directory = tmp_path / "checkpoint_run"
    Run(directory, {"suite": "audit"}, {"adapter": {"weights_sha256": "old"}}).stage(
        "metric", lambda: {"accuracy": .9})
    resumed = Run(directory, {"suite": "audit"},
                  {"adapter": {"weights_sha256": "new"}})
    assert resumed.manifest["stages"] == {}
    assert resumed.manifest["input_artifacts"]["adapter"]["weights_sha256"] == "new"


def test_uncaught_suite_failure_is_persisted_and_nonzero(tmp_path, monkeypatch):
    import benchmarks.run as runner
    import sys

    def explode(*_args):
        raise RuntimeError("suite setup broke")

    monkeypatch.setattr(runner, "suite_integrity", explode)
    run_dir = tmp_path / "suite_failure"
    monkeypatch.setattr(sys, "argv", ["benchmarks.run", "run", "--suites",
                                  "integrity", "--run-dir", str(run_dir)])
    assert runner.main() == 1
    failure = json.loads((run_dir / "run_manifest.json").read_text())["suite_failures"]["integrity"]
    assert failure["error"] == "suite setup broke"
    assert "RuntimeError" in Path(failure["traceback_path"]).read_text()


def test_stream_worker_failure_changes_health():
    old = dict(stream_state_global)
    try:
        stream_state_global.update(enabled=True, consumer_ready=False,
                                   consumer_error="NoBrokersAvailable", applier_error=None)
        response = app.test_client().get("/health")
        assert response.status_code == 503
        assert response.json["stream"]["consumer_error"] == "NoBrokersAvailable"
    finally:
        stream_state_global.clear()
        stream_state_global.update(old)


def test_target_f1_and_paired_ids():
    rows_a = [
        {"id": "1", "label": 0, "prediction": "negative"},
        {"id": "2", "label": 1, "prediction": "other"},
        {"id": "3", "label": 1, "prediction": "negative"},
        {"id": "4", "label": 0, "prediction": "negative"},
    ]
    rows_b = [dict(r, prediction="positive" if r["label"] else "negative")
              for r in rows_a]
    score = score_predictions(rows_a)
    assert score["accuracy"] == .5
    assert score["other_rate"] == .25
    assert score["f1_macro_target_classes"] < 1
    paired = paired_bootstrap(rows_a, rows_b, draws=100, seed=2)
    assert paired["accuracy_delta"] == .5
    with pytest.raises(ValueError, match="identical"):
        paired_bootstrap(rows_a, list(reversed(rows_b)), draws=10)


def test_real_peft_router_rejects_partial_and_applies_complete_snapshot():
    from transformers import GPT2Config, GPT2LMHeadModel
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

    base = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16,
                                      vocab_size=32, n_positions=32))
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4,
        target_modules=["c_attn", "c_proj"], task_type="CAUSAL_LM"))
    router = DoubleBufferedAdapterRouter(model, tokenizer=None)
    weights = get_peft_model_state_dict(model)
    with pytest.raises(ValueError, match="Adapter key mismatch"):
        router.swap_weights(dict(list(weights.items())[:-1]), "partial")
    assert router.active_version == "v0_base"
    changed = {key: value.clone() for key, value in weights.items()}
    first = next(iter(changed))
    changed[first] += 1
    router.swap_weights(changed, "complete")
    assert router.active_version == "complete"
    assert torch.equal(get_peft_model_state_dict(model)[first], changed[first])
