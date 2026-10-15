"""Meaningful dashboard checks: invalid settings, evidence access, launch contract."""

import json

import pytest

from benchmarks.evaluation_dashboard import create_app, runner_command, validate_settings


def test_settings_reject_nonsensical_combinations(tmp_path):
    with pytest.raises(ValueError, match="distinct"):
        validate_settings({"suites": ["integrity"], "seeds": [42, 42]}, preflight=False)
    with pytest.raises(ValueError, match="Soak duration"):
        validate_settings({"suites": ["integrity"], "soak_seconds": 10}, preflight=False)
    with pytest.raises(ValueError, match="IMDb checkpoint"):
        validate_settings({"suites": ["classification"],
                           "checkpoint": str(tmp_path)}, preflight=False)
    with pytest.raises(ValueError, match="Select one"):
        validate_settings({"suites": ["made_up"]}, preflight=False)
    with pytest.raises(ValueError, match="Training seeds apply"):
        validate_settings({"suites": ["integrity"], "seeds": [3]}, preflight=False)
    with pytest.raises(ValueError, match="three samples"):
        validate_settings({"suites": ["soak"], "soak_seconds": 3,
                           "soak_rate": 0.01}, preflight=False)


def test_settings_make_exact_runner_argv(tmp_path):
    settings = validate_settings({"suites": ["integrity", "baseline"],
                                  "profile": "standard", "seeds": [3, 5, 7],
                                  "max_seq_length": 192, "offline": True}, preflight=False)
    command = runner_command(settings, tmp_path)
    assert command[2:6] == ["-m", "benchmarks.run", "run", "--suites"]
    assert command[6:8] == ["integrity", "baseline"]
    assert command[command.index("--seeds")+1:command.index("--max-seq-length")] == ["3", "5", "7"]
    assert "--offline" in command


def test_existing_run_evidence_and_mutation_token(tmp_path):
    run = tmp_path / "old_smoke"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"created_utc": "2026-10-06T00:00:00Z",
        "spec": {"profile": "smoke", "suites": ["classification"]},
        "stages": {"classification/base": {"status": "complete", "result": {"accuracy": 0.5}},
                   "classification/checkpoint": {"status": "complete", "result": {"accuracy": 0.75,
                       "correct": 12, "n": 16}}}}))
    (run / "events.jsonl").write_text('{"kind":"stage_complete"}\n')
    app = create_app(run_root=tmp_path, open_terminal=False)
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/static/evaluation_dashboard.js").status_code == 200
    assert client.get("/api/runs").json[0]["id"] == "old_smoke"
    assert client.get("/api/runs").json[0]["headlines"] == ["IMDb 75% (12/16)"]
    detail = client.get("/api/runs/old_smoke").json
    assert detail["manifest"]["stages"]["classification/base"]["result"]["accuracy"] == 0.5
    assert client.get("/api/runs/old_smoke/log").json["available"] is False
    assert client.get("/api/runs/old_smoke/file/events.jsonl").status_code == 200
    assert client.get("/api/runs/old_smoke/file/../run_manifest.json").status_code in (404, 308)
    assert client.post("/api/runs", json={"suites": ["integrity"]}).status_code == 403


def test_run_launch_records_settings_without_shell(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("benchmarks.evaluation_dashboard.subprocess.Popen",
                        lambda argv, **kwargs: calls.append((argv, kwargs)))
    app = create_app(run_root=tmp_path, open_terminal=False)
    client = app.test_client()
    response = client.post("/api/runs", json={"suites": ["integrity"]},
                           headers={"X-Evaluation-Token": app.config["TOKEN"]})
    assert response.status_code == 201
    run = tmp_path / response.json["run_id"]
    assert json.loads((run / "dashboard_settings.json").read_text())["suites"] == ["integrity"]
    assert calls[0][0][3] == "benchmarks.evaluation_job"
    assert calls[0][1].get("shell") is not True


def test_resume_uses_saved_settings_and_rejects_untracked_cli_run(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("benchmarks.evaluation_dashboard.subprocess.Popen",
                        lambda argv, **kwargs: calls.append(argv))
    app = create_app(run_root=tmp_path, open_terminal=False)
    client = app.test_client()
    headers = {"X-Evaluation-Token": app.config["TOKEN"]}
    created = client.post("/api/runs", json={"suites": ["integrity"]}, headers=headers).json
    run = tmp_path / created["run_id"]
    (run / "dashboard_state.json").write_text(json.dumps({"status": "failed", "exit_code": 1}))
    response = client.post(f"/api/runs/{run.name}/resume", headers=headers)
    assert response.status_code == 200
    assert calls[-1][-1] == str(run / "dashboard_settings.json")
    old = tmp_path / "old_cli"
    old.mkdir()
    assert client.post("/api/runs/old_cli/resume", headers=headers).status_code == 400
