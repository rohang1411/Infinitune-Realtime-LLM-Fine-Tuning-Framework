"""Local-only control panel for reproducible benchmark runs and their evidence.

The browser never constructs a shell command. Every form value is checked here,
then a separate job process writes a durable console log and exit status.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import webbrowser

from flask import Flask, abort, jsonify, render_template, request, send_file
import psutil

from benchmarks.run import PROFILES, ROOT, SUITES, code_fingerprint, e2e_checkpoint_path

RUNS = ROOT / "reports" / "benchmark_runs"
DEFAULT_ADAPTER = ROOT / "output/imdb/checkpoints/distilgpt2__imdb/final"
ADAPTER_SUITES = {"classification", "calibration", "serving", "freshness", "soak", "swap_stress"}
KAFKA_SUITES = {"freshness", "pipeline"}
SERVER_SUITES = {"serving", "freshness", "pipeline", "soak"}
DS_SUITES = {"classification", "calibration", "e2e", "baseline", "continual"}
SEED_DEFAULTS = {"smoke": [42], "standard": [42, 1337, 2026],
                 "full": [42, 1337, 2026, 7, 11]}
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
ALLOWED_FIELDS = {"suites", "profile", "seed", "seeds", "max_seq_length", "concurrency",
                  "port", "kafka_bootstrap", "pipeline_timeout", "soak_seconds", "soak_rate",
                  "checkpoint", "e2e_checkpoint", "offline", "compile"}


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}.")
    return value


def _number(value, name, low, high):
    if type(value) not in (int, float) or not low <= value <= high:
        raise ValueError(f"{name} must be a number from {low} to {high}.")
    return float(value)


def _adapter(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be an existing adapter directory.")
    directory = Path(value).expanduser().resolve()
    missing = [filename for filename in ("adapter_model.safetensors", "adapter_config.json")
               if not (directory / filename).is_file()]
    if missing:
        raise ValueError(f"{name}: {directory} is missing {', '.join(missing)}.")
    return str(directory)


def _host_port(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9._-]+:[0-9]{1,5}", value):
        raise ValueError("Kafka address must be a hostname and port, for example localhost:9092.")
    host, port_text = value.rsplit(":", 1)
    port = _integer(int(port_text), "Kafka port", 1, 65535)
    return host, port


def _reachable(host, port):
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def validate_settings(raw, *, preflight=True):
    """Return a canonical runner specification; reject incompatible settings early."""
    if not isinstance(raw, dict):
        raise ValueError("The request must be a JSON object.")
    unexpected = set(raw) - ALLOWED_FIELDS
    if unexpected:
        raise ValueError(f"Unknown settings: {', '.join(sorted(unexpected))}.")
    suites = raw.get("suites", ["integrity"])
    if (not isinstance(suites, list) or not suites or
            any(not isinstance(item, str) or item not in SUITES for item in suites) or
            len(suites) != len(set(suites))):
        raise ValueError("Select one or more distinct listed suites.")
    # Preserve the runner's canonical suite order for stable resume specs.
    suites = [name for name in SUITES if name in suites]
    profile = raw.get("profile", "smoke")
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ValueError("Profile must be smoke, standard, or full.")
    seed = _integer(raw.get("seed", 42), "Sampling seed", 0, 2**32 - 1)
    seeds = raw.get("seeds", SEED_DEFAULTS[profile])
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 5 or
            any(type(item) is not int or not 0 <= item <= 2**32 - 1 for item in seeds) or
            len(seeds) != len(set(seeds))):
        raise ValueError("Training seeds must be 1–5 distinct nonnegative integers.")
    if not set(suites) & {"baseline", "continual"} and seeds != SEED_DEFAULTS[profile]:
        raise ValueError("Training seeds apply only to baseline or continual suites.")
    sequence = _integer(raw.get("max_seq_length", {"smoke": 128, "standard": 256,
                                                   "full": 512}[profile]),
                        "Sequence length", 32, 512)
    concurrency = _integer(raw.get("concurrency", 2), "Concurrency", 1, 64)
    port = _integer(raw.get("port", 8000), "Inference port", 1024, 65535)
    timeout = _integer(raw.get("pipeline_timeout", 600), "Pipeline timeout", 30, 7200)
    soak = raw.get("soak_seconds")
    if soak is not None:
        soak = _integer(soak, "Soak seconds", 3, 86400)
    rate = _number(raw.get("soak_rate", 1.0), "Soak requests per second", 0.01, 100)
    for name in ("offline", "compile"):
        if type(raw.get(name, False)) is not bool:
            raise ValueError(f"{name} must be true or false.")
    kafka = raw.get("kafka_bootstrap", "localhost:9092")
    host, kafka_port = _host_port(kafka)
    if "soak" not in suites and soak is not None:
        raise ValueError("Soak duration applies only when the soak suite is selected.")
    if "soak" not in suites and rate != 1.0:
        raise ValueError("Soak rate applies only when the soak suite is selected.")
    if "soak" in suites:
        duration = soak or {"smoke": 30, "standard": 600, "full": 86400}[profile]
        if duration * rate < 3:
            raise ValueError("Soak duration and request rate must allow at least three samples.")
    if not set(suites) & KAFKA_SUITES and kafka != "localhost:9092":
        raise ValueError("Kafka address applies only to freshness or pipeline.")
    if "pipeline" not in suites and timeout != 600:
        raise ValueError("Pipeline timeout applies only to the pipeline suite.")
    if not set(suites) & SERVER_SUITES and port != 8000:
        raise ValueError("Inference port applies only to a live server suite.")
    if not set(suites) & {"serving", "swap_stress"} and concurrency != 2:
        raise ValueError("Concurrency applies only to serving or swap stress.")
    if "precision" not in suites and raw.get("compile", False):
        raise ValueError("Compile applies only to precision.")
    checkpoint = raw.get("checkpoint") or str(DEFAULT_ADAPTER)
    if set(suites) & ADAPTER_SUITES:
        checkpoint = _adapter(checkpoint, "IMDb checkpoint")
    elif raw.get("checkpoint"):
        raise ValueError("IMDb checkpoint applies only to saved-adapter suites.")
    else:
        checkpoint = None
    e2e_checkpoint = raw.get("e2e_checkpoint")
    if "e2e" in suites:
        if not e2e_checkpoint:
            # The runner uses the newest local final checkpoint by default.
            from types import SimpleNamespace
            found = e2e_checkpoint_path(SimpleNamespace(e2e_checkpoint=None))
            e2e_checkpoint = str(found) if found else None
        e2e_checkpoint = _adapter(e2e_checkpoint, "E2E checkpoint")
    elif e2e_checkpoint:
        raise ValueError("E2E checkpoint applies only to the e2e suite.")
    if preflight and set(suites) & KAFKA_SUITES and not _reachable(host, kafka_port):
        raise ValueError(f"Kafka broker is not reachable at {kafka}. Start it, then retry.")
    if preflight and set(suites) & SERVER_SUITES:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise ValueError(f"Inference port {port} is already in use. Choose a free port.")
    return {"suites": suites, "profile": profile, "seed": seed, "seeds": seeds,
            "max_seq_length": sequence, "concurrency": concurrency, "port": port,
            "kafka_bootstrap": kafka, "pipeline_timeout": timeout,
            "soak_seconds": soak, "soak_rate": rate, "checkpoint": checkpoint,
            "e2e_checkpoint": e2e_checkpoint, "offline": raw.get("offline", False),
            "compile": raw.get("compile", False)}


def runner_command(settings, run_dir):
    """An argument vector avoids shell quoting and command injection."""
    cmd = [sys.executable, "-u", "-m", "benchmarks.run", "run", "--suites",
           *settings["suites"], "--profile", settings["profile"],
           "--run-dir", str(run_dir), "--seed", str(settings["seed"]),
           "--seeds", *(str(seed) for seed in settings["seeds"]),
           "--max-seq-length", str(settings["max_seq_length"]),
           "--concurrency", str(settings["concurrency"]), "--port", str(settings["port"]),
           "--kafka-bootstrap", settings["kafka_bootstrap"],
           "--pipeline-timeout", str(settings["pipeline_timeout"]),
           "--soak-rate", str(settings["soak_rate"])]
    for key, flag in (("checkpoint", "--checkpoint"),
                      ("e2e_checkpoint", "--e2e-checkpoint"),
                      ("soak_seconds", "--soak-seconds")):
        if settings[key] is not None:
            cmd.extend([flag, str(settings[key])])
    for key in ("offline", "compile"):
        if settings[key]:
            cmd.append("--" + key)
    return cmd


def _run_dir(run_id):
    if not RUN_ID.fullmatch(run_id):
        abort(404)
    directory = (RUNS / run_id).resolve()
    if directory.parent != RUNS.resolve() or not directory.is_dir():
        abort(404)
    return directory


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _headlines(stages):
    """Short labels copied from measured stage results, never inferred scores."""
    def result(name):
        stage = stages.get(name, {})
        return stage.get("result", {}) if stage.get("status") == "complete" else {}

    lines = []
    classification = result("classification/checkpoint")
    if isinstance(classification.get("accuracy"), (int, float)) and classification.get("n"):
        lines.append(f"IMDb {classification['accuracy']:.0%} ({classification.get('correct', '?')}/{classification['n']})")
    e2e = result("e2e/generation")
    if isinstance(e2e.get("bleu"), (int, float)) and e2e.get("n_unique_mrs"):
        lines.append(f"E2E BLEU {e2e['bleu']:.2f} ({e2e['n_unique_mrs']} MRs)")
    pipeline = result("pipeline/record_to_version")
    if isinstance(pipeline.get("latency_s"), (int, float)):
        lines.append(f"Record to version {pipeline['latency_s']:.2f} s")
    serving = result("serving/paired_summary")
    pairs = serving.get("pairs", [])
    if isinstance(pairs, list) and pairs and isinstance(pairs[0], dict):
        pair = pairs[0]
        if isinstance(pair.get("router_qps"), (int, float)) and isinstance(pair.get("legacy_qps"), (int, float)):
            lines.append(f"Serving {pair['router_qps']:.2f} / {pair['legacy_qps']:.2f} QPS")
    if not lines and result("integrity/pytest").get("passed") is True:
        lines.append("Integrity tests passed")
    return lines[:3]


def run_summary(directory, current_fingerprint=None):
    manifest = _read_json(directory / "run_manifest.json") or {}
    state = _read_json(directory / "dashboard_state.json") or {}
    stages = manifest.get("stages", {})
    counts = {name: sum(stage.get("status") == name for stage in stages.values())
              for name in ("complete", "running", "failed", "interrupted")}
    if state.get("status") in ("running", "launching"):
        status = state["status"] if _job_active(state) else "interrupted: launcher stopped"
    elif state.get("status") in ("complete", "failed"):
        status = state["status"]
    elif not stages:
        status = "no stages recorded"
    elif counts["failed"] or counts["interrupted"] or manifest.get("suite_failures"):
        status = "failed"
    elif counts["running"]:
        status = "interrupted or running outside dashboard"
    else:
        status = "stages complete; external CLI exit not recorded"
    return {"id": directory.name, "status": status, "counts": counts,
            "profile": manifest.get("spec", {}).get("profile"),
            "suites": manifest.get("spec", {}).get("suites", []),
            "created_utc": manifest.get("created_utc") or state.get("started_utc"),
            "has_console_log": (directory / "console.log").is_file(),
            "has_manifest": bool(manifest), "exit_code": state.get("exit_code"),
            "headlines": _headlines(stages),
            "ds_stages": sum(stage.get("status") == "complete" and name.split("/")[0] in DS_SUITES
                             for name, stage in stages.items()),
            "code_matches_current": (manifest.get("code_fingerprint") == current_fingerprint)
            if manifest.get("code_fingerprint") and current_fingerprint else None}


def _job_active(state):
    if state.get("status") == "launching":
        try:
            started = datetime.fromisoformat(state["started_utc"])
            return (datetime.now(timezone.utc) - started).total_seconds() < 30
        except (KeyError, ValueError):
            return False
    if state.get("status") != "running":
        return False
    # If the wrapper crashes, the runner can remain alive. Check both PIDs so
    # a resume cannot start a second process against the same run directory.
    for key in ("launcher_pid", "runner_pid"):
        try:
            process = psutil.Process(state[key])
            if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                return True
        except (KeyError, TypeError, ValueError, psutil.Error):
            continue
    return False


def _launch_terminal(directory):
    """Open a visible watcher; the browser also streams the same saved log."""
    log = directory / "console.log"
    state = directory / "dashboard_state.json"
    command = [sys.executable, "-u", "-m", "benchmarks.log_watch", str(log), str(state)]
    if sys.platform == "win32":
        subprocess.Popen(command, cwd=ROOT, creationflags=subprocess.CREATE_NEW_CONSOLE)
    elif sys.platform == "darwin":
        import shlex
        script = directory / "watch.command"
        script.write_text("#!/bin/sh\ncd " + shlex.quote(str(ROOT)) + "\n" +
                          " ".join(shlex.quote(part) for part in command) + "\n", encoding="utf-8")
        script.chmod(0o700)
        subprocess.Popen(["open", "-a", "Terminal", str(script)], cwd=ROOT)
    elif shutil.which("x-terminal-emulator"):
        subprocess.Popen(["x-terminal-emulator", "-e", *command], cwd=ROOT)


def create_app(*, run_root=RUNS, open_terminal=True):
    app = Flask(__name__, template_folder="templates")
    app.config["RUN_ROOT"] = Path(run_root).resolve()
    app.config["TOKEN"] = secrets.token_urlsafe(32)
    app.config["OPEN_TERMINAL"] = open_terminal

    def directory(run_id):
        if not RUN_ID.fullmatch(run_id):
            abort(404)
        path = (app.config["RUN_ROOT"] / run_id).resolve()
        if path.parent != app.config["RUN_ROOT"] or not path.is_dir():
            abort(404)
        return path

    def authorize():
        if request.headers.get("X-Evaluation-Token") != app.config["TOKEN"]:
            abort(403)

    @app.get("/")
    def index():
        return render_template("evaluation_dashboard.html", token=app.config["TOKEN"],
                               suites=SUITES, profiles=PROFILES, ds_suites=DS_SUITES,
                               adapter_exists=all((DEFAULT_ADAPTER / name).is_file()
                                                  for name in ("adapter_model.safetensors", "adapter_config.json")))

    @app.get("/api/runs")
    def list_runs():
        root = app.config["RUN_ROOT"]
        root.mkdir(parents=True, exist_ok=True)
        fingerprint = code_fingerprint()
        runs = [run_summary(path, fingerprint) for path in root.iterdir()
                if path.is_dir() and RUN_ID.fullmatch(path.name)]
        return jsonify(sorted(runs, key=lambda item: item["created_utc"] or "", reverse=True))

    @app.get("/api/runs/<run_id>")
    def get_run(run_id):
        path = directory(run_id)
        return jsonify({"summary": run_summary(path, code_fingerprint()),
                        "manifest": _read_json(path / "run_manifest.json"),
                        "state": _read_json(path / "dashboard_state.json")})

    @app.get("/api/runs/<run_id>/log")
    def get_log(run_id):
        path = directory(run_id) / "console.log"
        if not path.is_file():
            return jsonify({"text": "", "size": 0, "available": False})
        size = path.stat().st_size
        # Byte offset polling bounds browser payload while retaining the complete file.
        offset = request.args.get("offset", "0")
        if not offset.isdecimal():
            abort(400)
        offset = min(int(offset), size)
        with path.open("rb") as stream:
            stream.seek(offset)
            chunk = stream.read(131072)
        return jsonify({"text": chunk.decode("utf-8", errors="replace").replace("\r", "\n"),
                        "size": offset + len(chunk), "available": True})

    @app.get("/api/runs/<run_id>/files")
    def get_files(run_id):
        path = directory(run_id)
        files = [{"path": item.relative_to(path).as_posix(), "bytes": item.stat().st_size}
                 for item in path.rglob("*") if item.is_file() and
                 "checkpoints" not in item.relative_to(path).parts]
        return jsonify(sorted(files, key=lambda item: item["path"]))

    @app.get("/api/runs/<run_id>/file/<path:relative>")
    def get_file(run_id, relative):
        root = directory(run_id)
        file = (root / relative).resolve()
        if root not in file.parents or not file.is_file() or "checkpoints" in file.relative_to(root).parts:
            abort(404)
        return send_file(file, as_attachment=True)

    @app.post("/api/runs")
    def start_run():
        authorize()
        try:
            settings = validate_settings(request.get_json(silent=True))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        root = app.config["RUN_ROOT"]
        root.mkdir(parents=True, exist_ok=True)
        for path in root.iterdir():
            if path.is_dir() and _job_active(_read_json(path / "dashboard_state.json") or {}):
                return jsonify({"error": f"Run {path.name} is already active. Wait for it to finish."}), 409
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "_" + secrets.token_hex(3)
        path = root / run_id
        path.mkdir(exist_ok=False)
        (path / "dashboard_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        (path / "dashboard_state.json").write_text(json.dumps({"status": "launching",
            "started_utc": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")
        try:
            subprocess.Popen([sys.executable, "-u", "-m", "benchmarks.evaluation_job",
                              str(path / "dashboard_settings.json")], cwd=ROOT,
                             creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                             start_new_session=os.name != "nt")
            if app.config["OPEN_TERMINAL"]:
                try:
                    _launch_terminal(path)
                except OSError as exc:
                    (path / "terminal_open_error.txt").write_text(str(exc), encoding="utf-8")
        except OSError as exc:
            (path / "launch_error.txt").write_text(str(exc), encoding="utf-8")
            (path / "dashboard_state.json").write_text(json.dumps({"status": "failed",
                "exit_code": None, "error": str(exc)}), encoding="utf-8")
            return jsonify({"error": f"Could not launch evaluation: {exc}", "run_id": run_id}), 500
        return jsonify({"run_id": run_id, "path": str(path)}), 201

    @app.post("/api/runs/<run_id>/resume")
    def resume_run(run_id):
        authorize()
        path = directory(run_id)
        if _job_active(_read_json(path / "dashboard_state.json") or {}):
            return jsonify({"error": "This run is still active."}), 409
        settings = _read_json(path / "dashboard_settings.json")
        if settings is None:
            return jsonify({"error": "This run predates the dashboard. Resume it with the CLI and its manifest spec."}), 400
        try:
            validate_settings(settings)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        (path / "dashboard_state.json").write_text(json.dumps({"status": "launching",
            "started_utc": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")
        try:
            subprocess.Popen([sys.executable, "-u", "-m", "benchmarks.evaluation_job",
                              str(path / "dashboard_settings.json")], cwd=ROOT,
                             creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                             start_new_session=os.name != "nt")
        except OSError as exc:
            (path / "launch_error.txt").write_text(str(exc), encoding="utf-8")
            (path / "dashboard_state.json").write_text(json.dumps({"status": "failed",
                "exit_code": None, "error": str(exc)}), encoding="utf-8")
            return jsonify({"error": f"Could not resume evaluation: {exc}"}), 500
        if app.config["OPEN_TERMINAL"]:
            try:
                _launch_terminal(path)
            except OSError as exc:
                (path / "terminal_open_error.txt").write_text(str(exc), encoding="utf-8")
        return jsonify({"run_id": run_id, "path": str(path)})

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("dashboard port must be between 1024 and 65535")
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Evaluation dashboard: {url}", flush=True)
    print(f"Saved runs: {RUNS}", flush=True)
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    # Frequent browser polling should not bury actionable server errors.
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    create_app().run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
