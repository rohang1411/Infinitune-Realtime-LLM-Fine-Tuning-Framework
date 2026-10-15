"""Reproducible, resumable InfiniTune benchmark command.

Run `python -m benchmarks.run list` for suites and `... run --help` for options.
Each stage writes a manifest, event log, raw artifacts, and a traceback on failure.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid

from benchmarks.repro_eval import (
    prepare_dataset, read_rows, train_phase, evaluate_rows,
    paired_bootstrap, file_sha256, evaluate_calibration, MODEL_REVISIONS,
)
from utils.kafka_config import build_client_kwargs
from utils.serving_router import nearest_rank_percentile

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "integrity": "pytest regression suite plus historical claim audit",
    "classification": "fixed IMDb holdout: base and saved checkpoint predictions",
    "calibration": "fixed IMDb forced-choice probabilities, ECE, Brier, NLL, AUROC",
    "e2e": "fixed E2E NLG MRs: multi-reference BLEU and ROUGE-L",
    "baseline": "same-data, same-budget offline vs ordered LoRA training",
    "continual": "IMDb/Yelp/Amazon sequential training, with and without replay",
    "serving": "equal-load alternating router/legacy HTTP comparison",
    "freshness": "Kafka adapter publication to visible serving version",
    "pipeline": "isolated Kafka record -> trainer step -> visible adapter version",
    "precision": "measured fp32/bf16/fp16 LoRA step timing and memory",
    "soak": "long-running inference RSS slope, request latency and failures",
    "swap_stress": "real-model concurrent requests during validated adapter swaps",
    "propagation": "weight-update propagation distribution (publish to apply)",
    "cold_start": "checkpoint recovery vs waiting for first Kafka push",
    "fault_injection": "truncated and corrupted Kafka frames recovery",
    "eval_speed": "evaluation throughput and latency comparison",
    "filter_benchmark": "five-stage stream filter throughput and latency",
    "streaming_quality": "streaming continual learning quality and convergence",
    "ablation": "ablation of streaming integrity and architectural components",
    "e2e_facts": "factual accuracy and hallucination metrics on generated outputs",
    "rescore": "offline metric re-scoring of benchmark outputs",
}
PROFILES = {
    "smoke": {"train": 32, "eval": 16, "e2e_mrs": 4, "steps": 2, "requests": 20, "repeats": 1, "seeds": 1},
    "standard": {"train": 1000, "eval": 200, "e2e_mrs": 100, "steps": 63, "requests": 200, "repeats": 3, "seeds": 3},
    "full": {"train": 25000, "eval": 4000, "e2e_mrs": 4672, "steps": 1563, "requests": 1000, "repeats": 5, "seeds": 5},
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def git_provenance():
    def command(*parts):
        result = subprocess.run(["git", *parts], cwd=ROOT, capture_output=True,
                                text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    return {"commit": command("rev-parse", "HEAD"),
            "branch": command("branch", "--show-current"),
            "dirty_files": command("status", "--short")}


def code_fingerprint():
    """Hash code/config that can alter metrics, including uncommitted edits."""
    # Dashboard presentation and its log watcher cannot change benchmark
    # calculations. Exclude them so a UI-only edit does not invalidate a run.
    presentation = {"evaluation_dashboard.py", "evaluation_job.py", "log_watch.py"}
    paths = [*(path for path in ROOT.glob("benchmarks/*.py")
               if path.name not in presentation), *ROOT.glob("utils/*.py"),
             ROOT / "trainer.py", ROOT / "inference.py",
             ROOT / "requirements-eval.txt",
             ROOT / "configs/imdb_quantitative.yaml",
             ROOT / "configs/e2e_qualitative.yaml"]
    digest = hashlib.sha256()
    for path in sorted(paths):
        if path.exists():
            digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def installed_packages():
    """Capture transitive package versions; reproducibility needs more than a requirements file."""
    return dict(sorted((dist.metadata["Name"].lower(), dist.version)
                       for dist in metadata.distributions() if dist.metadata.get("Name")))


def e2e_checkpoint_path(args):
    if args.e2e_checkpoint:
        return Path(args.e2e_checkpoint).resolve()
    candidates = list((ROOT / "output/e2e/checkpoints").rglob("final/adapter_model.safetensors"))
    return max(candidates, key=lambda x: x.stat().st_mtime).parent if candidates else None


def external_checkpoint_inputs(args, suites):
    """Fingerprint checkpoints outside the run so changed adapters invalidate scores."""
    paths = {}
    if set(suites) & {"classification", "calibration", "serving", "freshness",
                      "soak", "swap_stress"}:
        paths["imdb_adapter"] = Path(args.checkpoint or
            ROOT / "output/imdb/checkpoints/distilgpt2__imdb/final").resolve()
    if "e2e" in suites:
        paths["e2e_adapter"] = e2e_checkpoint_path(args)
    result = {}
    for name, path in paths.items():
        if path is None:
            result[name] = {"status": "missing"}
            continue
        weights = path / "adapter_model.safetensors"
        config = path / "adapter_config.json"
        result[name] = {"path": str(path),
                        "weights_sha256": file_sha256(weights) if weights.is_file() else None,
                        "config_sha256": file_sha256(config) if config.is_file() else None}
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


class Progress:
    """One-line progress, smoothed ETA, and rate; at most once per second."""
    def __init__(self, name):
        self.name = name
        self.start = time.monotonic()
        self.last = 0.0

    def __call__(self, done, total):
        now = time.monotonic()
        if now - self.last < 1 and done != total:
            return
        elapsed = max(now - self.start, 1e-6)
        rate = done / elapsed
        eta = (total - done) / rate if rate else float("inf")
        filled = round(20 * done / max(total, 1))
        print(f"\r{self.name:25} [{'#'*filled}{'.'*(20-filled)}] "
              f"{done}/{total} {rate:.2f}/s ETA {eta:.0f}s", end="\n" if done == total else "", flush=True)
        self.last = now


class Run:
    def __init__(self, directory, spec, input_artifacts=None):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "run_manifest.json"
        current_fingerprint = code_fingerprint()
        current_packages = installed_packages()
        current_inputs = input_artifacts or {}
        if self.path.exists():
            self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
            if self.manifest["spec"] != spec:
                raise ValueError("Existing run spec differs. Start a new --run-dir or use original options.")
            previous = self.manifest.get("code_fingerprint")
            previous_packages = self.manifest.get("environment", {}).get("packages")
            previous_inputs = self.manifest.get("input_artifacts", {})
            if (previous != current_fingerprint or previous_packages != current_packages
                    or previous_inputs != current_inputs) \
                    and self.manifest.get("stages"):
                # Never mix scores from different code revisions in one run.
                # Preserve old artifacts inside the run before resetting stages.
                archived = self.directory / "attempt_archive" / (
                    datetime.now().strftime("%Y%m%d-%H%M%S") + "_" +
                    (previous or "unknown")[:8] + "_" + uuid.uuid4().hex[:4])
                archived.mkdir(parents=True, exist_ok=False)
                for child_name in ("predictions", "probabilities", "checkpoints",
                                   "precision_traces", "raw_requests", "server_logs",
                                   "pipeline_output", "errors", "pytest.log",
                                   "pipeline_config.yaml", "freshness_config.yaml"):
                    source = self.directory / child_name
                    target = archived / child_name
                    if source.exists():
                        if (source.resolve().parent != self.directory
                                or target.resolve().parent != archived.resolve()):
                            raise RuntimeError(f"Unsafe run archive path: {source}")
                        shutil.move(str(source), str(target))
                write_json(archived / "previous_manifest.json", self.manifest)
                self.manifest["stages"] = {}
                self.manifest["suite_failures"] = {}
                self.manifest["code_fingerprint"] = current_fingerprint
                self.manifest["provenance"] = git_provenance()
                self.manifest["environment"]["packages"] = current_packages
                self.manifest["input_artifacts"] = current_inputs
                self.event("code_or_environment_changed_restart", previous_code=previous,
                           current_code=current_fingerprint,
                           packages_changed=previous_packages != current_packages,
                           input_artifacts_changed=previous_inputs != current_inputs,
                           archive=str(archived))
                self.save()
            elif previous is None or previous_packages is None or "input_artifacts" not in self.manifest:
                self.manifest["code_fingerprint"] = current_fingerprint
                self.manifest["environment"]["packages"] = current_packages
                self.manifest["input_artifacts"] = current_inputs
                self.save()
        else:
            self.manifest = {"schema_version": 1, "created_utc": utc_now(),
                             "spec": spec, "environment": {
                                 "python": sys.version.split()[0],
                                 "platform": platform.platform(),
                                 "processor": platform.processor(),
                                 "cpu_count": os.cpu_count(),
                                 "packages": current_packages},
                             "provenance": git_provenance(),
                             "code_fingerprint": current_fingerprint,
                             "input_artifacts": current_inputs,
                             "stages": {}}
            self.save()

    def save(self):
        write_json(self.path, self.manifest)

    def event(self, kind, **fields):
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps({"time_utc": utc_now(), "kind": kind, **fields},
                                 default=str) + "\n")

    def stage(self, name, operation):
        prior = self.manifest["stages"].get(name, {})
        if prior.get("status") == "complete":
            result = prior.get("result", {})
            missing = []
            if isinstance(result, dict):
                for key in ("predictions_path", "probabilities_path", "raw_path", "checkpoint"):
                    if key not in result:
                        continue
                    path = Path(result[key])
                    if not path.exists():
                        missing.append(str(path))
                    digest = result.get(key.removesuffix("_path") + "_sha256")
                    if digest and path.is_file() and file_sha256(path) != digest:
                        missing.append(f"checksum mismatch: {path}")
                checkpoint = result.get("checkpoint")
                if checkpoint and not (Path(checkpoint) / "adapter_model.safetensors").exists():
                    missing.append(f"checkpoint weights missing: {checkpoint}")
            if not missing:
                print(f"SKIP {name}: completed in this run")
                return result
            self.event("stage_artifact_invalid", stage=name, problems=missing)
            for key in ("predictions_path", "probabilities_path", "raw_path"):
                if key in result:
                    path = Path(result[key]).resolve()
                    if path.is_file() and self.directory in path.parents:
                        archived = path.with_name(path.name + ".invalid_" + uuid.uuid4().hex[:8])
                        path.rename(archived)
                        self.event("archived_invalid_artifact", stage=name,
                                   from_path=str(path), to_path=str(archived))
            print(f"RESTART {name}: cached artifact invalid: {missing}")
        print(f"\nSTART {name}", flush=True)
        self.event("stage_start", stage=name)
        entry = {"status": "running", "started_utc": utc_now(),
                 "code_provenance": git_provenance(),
                 "code_fingerprint": self.manifest["code_fingerprint"]}
        self.manifest["stages"][name] = entry
        self.save()
        began = time.perf_counter()
        try:
            result = operation()
            entry.update(status="complete", ended_utc=utc_now(),
                         duration_s=time.perf_counter()-began, result=result)
            self.event("stage_complete", stage=name, result=result)
            print(f"DONE {name}: {json.dumps(result, default=str)[:300]}")
            return result
        except KeyboardInterrupt:
            entry.update(status="interrupted", ended_utc=utc_now(),
                         duration_s=time.perf_counter()-began)
            self.event("stage_interrupted", stage=name)
            raise
        except Exception as exc:
            detail = traceback.format_exc()
            error_path = self.directory / "errors" / (name.replace("/", "_") + ".txt")
            error_path.parent.mkdir(exist_ok=True)
            error_path.write_text(detail, encoding="utf-8")
            entry.update(status="failed", ended_utc=utc_now(),
                         duration_s=time.perf_counter()-began,
                         error_type=type(exc).__name__, error=str(exc),
                         traceback_path=str(error_path))
            self.event("stage_failed", stage=name, error=str(exc),
                       traceback_path=str(error_path))
            print(f"FAILED {name}: {exc}\nTraceback: {error_path}", file=sys.stderr)
            return None
        finally:
            self.save()


def data_stage(run, task, split, count, seed, data_dir, offline):
    name = f"data/{task}/{split}"
    result = run.stage(name, lambda: dict(zip(("path", "manifest"),
        prepare_dataset(task, split, count, seed, data_dir, offline))))
    if not result:
        raise RuntimeError(f"Dataset stage failed: {name}")
    path = Path(result["path"])
    if not path.exists() or file_sha256(path) != result["manifest"]["sha256"]:
        raise RuntimeError(f"Dataset changed after preparation: {path}")
    return path


def _checkpoint_stage(run, name, rows, path, args, seed, adapter=None):
    # Preserve a failed staging directory for diagnosis before retrying.
    staging = path.with_name(path.name + ".staging")
    if staging.exists():
        archived = staging.with_name(staging.name + ".interrupted_" + str(int(time.time())))
        if archived.resolve().parent != run.directory.resolve() and run.directory.resolve() not in archived.resolve().parents:
            raise RuntimeError(f"Refusing to move staging outside run directory: {staging}")
        shutil.move(str(staging), str(archived))
        run.event("archived_interrupted_checkpoint", from_path=str(staging), to_path=str(archived))
    return run.stage(name, lambda: train_phase(rows, path, args.model, seed,
        adapter=adapter, max_steps=PROFILES[args.profile]["steps"],
        max_seq_length=args.max_seq_length, progress=Progress(name)))


def _eval_stage(run, name, rows, args, adapter=None):
    path = run.directory / "predictions" / (name.replace("/", "_") + ".jsonl")
    if path.exists() and run.manifest["stages"].get(name, {}).get("status") != "complete":
        archived = path.with_name(path.name + ".interrupted_" + str(int(time.time())))
        path.rename(archived)
        run.event("archived_partial_predictions", from_path=str(path), to_path=str(archived))
    result = run.stage(name, lambda: evaluate_rows(rows, path, args.model,
        adapter=adapter, max_seq_length=args.max_seq_length,
        progress=Progress(name)))
    if result:
        result["predictions_path"] = str(path)
        run.save()
    return result


def archive_partial(run, stage_name, path):
    """Keep interrupted raw evidence, then let the stage restart deterministically."""
    path = Path(path)
    if path.exists() and run.manifest["stages"].get(stage_name, {}).get("status") != "complete":
        archived = path.with_name(path.name + ".interrupted_" + str(int(time.time())))
        path.rename(archived)
        run.event("archived_partial_output", stage=stage_name,
                  from_path=str(path), to_path=str(archived))


def assert_no_overlap(train_rows, test_rows):
    overlap = set(r["id"] for r in train_rows) & set(r["id"] for r in test_rows)
    if overlap:
        raise ValueError(f"Train/test text-hash overlap: {len(overlap)} examples")
    return {"train": len(train_rows), "test": len(test_rows),
            "overlap": 0, "identity": "SHA-256 of exact review text"}


def suite_integrity(run, args):
    def pytest_stage():
        log = run.directory / "pytest.log"
        with log.open("w", encoding="utf-8") as out:
            result = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"],
                                    cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"pytest exit {result.returncode}; inspect {log}")
        return {"log": str(log), "passed": True}
    run.stage("integrity/pytest", pytest_stage)

    def historical():
        import csv
        path = ROOT / "output/imdb/outputs/run10/metrics_clean.csv"
        if not path.exists():
            raise FileNotFoundError(f"Historical metrics CSV unavailable: {path}")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        final = next((r for r in reversed(rows) if r.get("accuracy")), None)
        if not final:
            raise ValueError("Historical CSV has no accuracy row")
        return {"csv": str(path), "sha256": file_sha256(path),
                "final_step": final.get("step"), "accuracy": final.get("accuracy"),
                "legacy_f1": final.get("f1"),
                "status": "historical aggregate only; no per-example predictions"}
    run.stage("integrity/historical", historical)


def suite_classification(run, args, data_dir):
    n = PROFILES[args.profile]["eval"]
    path = data_stage(run, "imdb", "test", n, args.seed, data_dir, args.offline)
    rows = read_rows(path)
    base = _eval_stage(run, "classification/base", rows, args)
    adapter = args.checkpoint or str(ROOT / "output/imdb/checkpoints/distilgpt2__imdb/final")
    if not Path(adapter).exists():
        run.stage("classification/checkpoint", lambda: (_ for _ in ()).throw(
            FileNotFoundError(f"Checkpoint unavailable: {adapter}")))
        return
    tuned = _eval_stage(run, "classification/checkpoint", rows, args, adapter)
    if base and tuned:
        run.stage("classification/paired", lambda: paired_bootstrap(
            read_rows(base["predictions_path"]), read_rows(tuned["predictions_path"]),
            seed=args.seed))


def suite_calibration(run, args, data_dir):
    n = PROFILES[args.profile]["eval"]
    rows = read_rows(data_stage(run, "imdb", "test", n, args.seed,
                                data_dir, args.offline))
    adapter = args.checkpoint or str(ROOT / "output/imdb/checkpoints/distilgpt2__imdb/final")
    for name, path in (("base", None), ("checkpoint", adapter)):
        if path and not Path(path).exists():
            run.stage(f"calibration/{name}", lambda: (_ for _ in ()).throw(
                FileNotFoundError(f"Checkpoint unavailable: {path}")))
            continue
        output = run.directory / "probabilities" / f"{name}.jsonl"
        archive_partial(run, f"calibration/{name}", output)
        run.stage(f"calibration/{name}", lambda path=path, output=output: evaluate_calibration(
            rows, output, args.model, adapter=path,
            max_seq_length=args.max_seq_length,
            progress=Progress(f"calibration/{name}")))


def suite_e2e(run, args, data_dir):
    from benchmarks.e2e_eval import prepare_e2e, evaluate_e2e
    count = PROFILES[args.profile]["e2e_mrs"]
    prepared = run.stage("data/e2e/validation", lambda: dict(zip(("path", "manifest"),
        prepare_e2e(count, args.seed, data_dir, args.offline))))
    if not prepared:
        return
    path = Path(prepared["path"])
    if file_sha256(path) != prepared["manifest"]["sha256"]:
        run.stage("e2e/cache_integrity", lambda: (_ for _ in ()).throw(
            ValueError(f"E2E dataset cache changed: {path}")))
        return
    adapter = e2e_checkpoint_path(args)
    if not adapter or not adapter.exists():
        run.stage("e2e/checkpoint", lambda: (_ for _ in ()).throw(
            FileNotFoundError("E2E final checkpoint unavailable; pass --e2e-checkpoint")))
        return
    rows = read_rows(path)
    output = run.directory / "predictions" / "e2e.jsonl"
    archive_partial(run, "e2e/generation", output)
    run.stage("e2e/generation", lambda: evaluate_e2e(rows, output, adapter=adapter,
                                                       progress=Progress("e2e/generation")))


def suite_baseline(run, args, data_dir):
    p = PROFILES[args.profile]
    train_path = data_stage(run, "imdb", "train", p["train"], args.seed, data_dir, args.offline)
    test_path = data_stage(run, "imdb", "test", p["eval"], args.seed, data_dir, args.offline)
    train = read_rows(train_path)
    test = read_rows(test_path)
    if not run.stage("baseline/no_overlap", lambda: assert_no_overlap(train, test)):
        return
    import random
    shuffled = list(train)
    random.Random(args.seed).shuffle(shuffled)
    seed_scores = []
    for seed in args.seeds:
        results = []
        seed_shuffled = list(train)
        random.Random(seed).shuffle(seed_shuffled)
        for method, rows in (("ordered", train), ("offline_shuffled", seed_shuffled)):
            adapter = run.directory / "checkpoints" / "baseline" / f"seed_{seed}" / method
            trained = _checkpoint_stage(run, f"baseline/seed_{seed}/train/{method}",
                                        rows, adapter, args, seed)
            if trained:
                result = _eval_stage(run, f"baseline/seed_{seed}/eval/{method}",
                                     test, args, adapter)
                if result:
                    results.append((method, result))
        if len(results) == 2:
            paired = run.stage(f"baseline/seed_{seed}/paired", lambda: paired_bootstrap(
                read_rows(results[0][1]["predictions_path"]),
                read_rows(results[1][1]["predictions_path"]), seed=seed))
            if paired:
                seed_scores.append({"seed": seed, "ordered_accuracy": results[0][1]["accuracy"],
                                    "offline_accuracy": results[1][1]["accuracy"],
                                    "offline_minus_ordered": paired["accuracy_delta"]})
    if seed_scores:
        def summary():
            from statistics import mean, stdev
            deltas = [x["offline_minus_ordered"] for x in seed_scores]
            return {"per_seed": seed_scores, "mean_delta": mean(deltas),
                    "sample_sd_delta": stdev(deltas) if len(deltas) > 1 else None,
                    "interpretation": "Offline ordered-vs-shuffled comparison; no Kafka measured"}
        run.stage("baseline/summary", summary)


def suite_continual(run, args, data_dir):
    from benchmarks.run_streaming_cl import compute_continual_learning_metrics
    from utils.replay_buffer import ReservoirReplayBuffer
    p = PROFILES[args.profile]
    tasks = ["imdb", "yelp", "amazon"]
    train = {task: read_rows(data_stage(run, task, "train", p["train"],
                                        args.seed, data_dir, args.offline)) for task in tasks}
    test = {task: read_rows(data_stage(run, task, "test", p["eval"],
                                       args.seed, data_dir, args.offline)) for task in tasks}
    for task in tasks:
        if not run.stage(f"continual/{task}/no_overlap",
                         lambda task=task: assert_no_overlap(train[task], test[task])):
            return
    for seed in args.seeds:
        base_scores = {}
        for task in tasks:
            base = _eval_stage(run, f"continual/seed_{seed}/base_on_{task}",
                               test[task], args)
            if base:
                base_scores[task] = base["accuracy"]
        if len(base_scores) != len(tasks):
            continue
        for method in ("no_replay", "reservoir_replay"):
            matrix = []
            adapter = None
            reservoir = ReservoirReplayBuffer(capacity=1000, seed=seed)
            for phase, task in enumerate(tasks):
                # Equal *total* examples/optimizer steps in both arms. Replay
                # replaces 20% of current-domain examples, not extra updates.
                rows = list(train[task])
                if method == "reservoir_replay" and len(reservoir):
                    import random
                    count = min(len(rows)//5, len(reservoir))
                    positions = random.Random(seed+phase).sample(range(len(rows)), count)
                    replacements = reservoir.sample(count)
                    for pos, old in zip(positions, replacements):
                        rows[pos] = old
                checkpoint = run.directory / "checkpoints" / "continual" / f"seed_{seed}" / method / f"phase_{phase+1}"
                trained = _checkpoint_stage(run, f"continual/seed_{seed}/{method}/train_{task}",
                                            rows, checkpoint, args, seed + phase,
                                            adapter=adapter)
                if not trained:
                    break
                adapter = checkpoint
                row_scores = []
                for target in tasks:
                    result = _eval_stage(run, f"continual/seed_{seed}/{method}/phase_{phase+1}_on_{target}",
                                         test[target], args, adapter)
                    if not result:
                        break
                    row_scores.append(result["accuracy"])
                if len(row_scores) != len(tasks):
                    break
                matrix.append(row_scores)
                reservoir.add_batch(train[task])
            if len(matrix) == len(tasks):
                def finish():
                    result = compute_continual_learning_metrics(matrix, tasks)
                    fwt_terms = [matrix[i-1][i] - base_scores[tasks[i]]
                                 for i in range(1, len(tasks))]
                    result.update({"matrix": matrix, "tasks": tasks, "method": method,
                                   "seed": seed, "replay_buffer_capacity": 1000,
                                   "replay_replacement_fraction": .20,
                                   "zero_shot_base_accuracy": base_scores,
                                   "forward_transfer": sum(fwt_terms)/len(fwt_terms)})
                    return result
                run.stage(f"continual/seed_{seed}/{method}/summary", finish)


def _wait_server(url, proc, timeout=180):
    import urllib.request
    import urllib.error
    began = time.monotonic()
    while time.monotonic() - began < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"Inference server exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except urllib.error.HTTPError as exc:
            if exc.code == 503:
                state = json.loads(exc.read().decode("utf-8"))
                stream = state.get("stream", {})
                if stream.get("consumer_error") or stream.get("applier_error"):
                    raise RuntimeError(f"Inference background worker failed: {stream}") from exc
            time.sleep(1)
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"Inference server did not become healthy at {url}")


def child_env():
    """Redirected Windows stdout otherwise defaults to cp1252 and may crash logs."""
    return {**os.environ, "PYTHONIOENCODING": "utf-8"}


def suite_serving(run, args):
    from benchmarks.load_test_serving import run_load_test, send_generation_request
    adapter = args.checkpoint or str(ROOT / "output/imdb/checkpoints/distilgpt2__imdb/final")
    if not Path(adapter).exists():
        run.stage("serving/preflight", lambda: (_ for _ in ()).throw(
            FileNotFoundError(f"Checkpoint unavailable: {adapter}")))
        return
    p = PROFILES[args.profile]
    results = []
    for repeat in range(p["repeats"]):
        # Alternate order to reduce warm-cache and thermal order bias.
        order = ("router", "legacy_lock") if repeat % 2 == 0 else ("legacy_lock", "router")
        for mode in order:
            stage_name = f"serving/repeat_{repeat+1}/{mode}"
            def measure(mode=mode, stage_name=stage_name):
                log_path = run.directory / "server_logs" / f"{stage_name.replace('/', '_')}.log"
                log_path.parent.mkdir(exist_ok=True)
                url = f"http://127.0.0.1:{args.port}"
                with log_path.open("w", encoding="utf-8") as log:
                    proc = subprocess.Popen([
                        sys.executable, "-u", "inference.py", "--config",
                        "configs/imdb_quantitative.yaml", "--checkpoint", adapter,
                        "--serving-mode", mode, "--port", str(args.port)],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        env=child_env())
                    try:
                        _wait_server(url, proc)
                        for _ in range(3):
                            warm = send_generation_request(url + "/generate",
                                "Review: Excellent movie.\nSentiment:", timeout_s=120)
                            if not warm["success"]:
                                raise RuntimeError(f"Warmup failed: {warm['error']}")
                        raw = run.directory / "raw_requests" / f"{stage_name.replace('/', '_')}.jsonl"
                        raw.parent.mkdir(exist_ok=True)
                        score = run_load_test(url + "/generate", num_requests=p["requests"],
                            concurrency=args.concurrency, raw_output=str(raw),
                            timeout_s=120, progress=Progress(stage_name))
                        score.update({"server_log": str(log_path), "warmup_requests": 3,
                                      "checkpoint": adapter, "repeat": repeat + 1,
                                      "raw_path": str(raw), "raw_sha256": file_sha256(raw)})
                        if score["failed_requests"]:
                            raise RuntimeError(f"{score['failed_requests']} failed requests; raw={raw}")
                        return score
                    finally:
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait(timeout=10)
            result = run.stage(stage_name, measure)
            if result:
                results.append(result)
    if len(results) == p["repeats"] * 2:
        def summarize():
            pairs = []
            for repeat in range(1, p["repeats"]+1):
                router = next(x for x in results if x["repeat"] == repeat and "router" in x["serving_modes_detected"])
                legacy = next(x for x in results if x["repeat"] == repeat and "legacy_lock" in x["serving_modes_detected"])
                pairs.append({"repeat": repeat, "router_qps": router["achieved_qps"],
                              "legacy_qps": legacy["achieved_qps"],
                              "qps_delta": router["achieved_qps"]-legacy["achieved_qps"]})
            return {"pairs": pairs, "mean_qps_delta": sum(x["qps_delta"] for x in pairs)/len(pairs),
                    "caution": "Closed-loop client throughput, not maximum sustainable QPS; p99 needs >=1000 successes per arm"}
        run.stage("serving/paired_summary", summarize)


def suite_freshness(run, args):
    def preflight():
        from kafka import KafkaAdminClient
        admin = KafkaAdminClient(
            request_timeout_ms=5000,
            **build_client_kwargs({"bootstrap_servers": args.kafka_bootstrap}, "admin"),
        )
        try:
            topics = admin.list_topics()
        finally:
            admin.close()
        return {"broker": args.kafka_bootstrap, "topics": sorted(topics)}
    if not run.stage("freshness/preflight", preflight):
        return
    adapter = args.checkpoint or str(ROOT / "output/imdb/checkpoints/distilgpt2__imdb/final")
    if not Path(adapter).exists():
        run.stage("freshness/checkpoint", lambda: (_ for _ in ()).throw(
            FileNotFoundError(f"Checkpoint unavailable: {adapter}")))
        return

    def measure():
        import urllib.request
        import yaml
        from kafka.admin import KafkaAdminClient, NewTopic
        from peft import get_peft_model_state_dict
        from trainer import LoRAProducer
        from benchmarks.repro_eval import load_model

        topic = "infinitune-benchmark-" + uuid.uuid4().hex[:12]
        admin = KafkaAdminClient(**build_client_kwargs({"bootstrap_servers": args.kafka_bootstrap}, "admin"))
        try:
            admin.create_topics([NewTopic(name=topic, num_partitions=1,
                                          replication_factor=1)])
        finally:
            admin.close()
        config = yaml.safe_load((ROOT / "configs/imdb_quantitative.yaml").read_text(encoding="utf-8"))
        config["kafka"]["bootstrap_servers"] = [args.kafka_bootstrap]
        config["kafka"]["lora_updates_topic"] = topic
        config["kafka"]["consumer_group_inference"] = "bench-" + uuid.uuid4().hex
        config_path = run.directory / "freshness_config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        log_path = run.directory / "server_logs" / "freshness.log"
        log_path.parent.mkdir(exist_ok=True)
        url = f"http://127.0.0.1:{args.port}"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen([sys.executable, "-u", "inference.py", "--config",
                str(config_path), "--checkpoint", adapter, "--listen-updates",
                "--serving-mode", "router", "--port", str(args.port)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env=child_env())
            try:
                _wait_server(url, proc)
                # Wait until the Kafka consumer has actually subscribed; server
                # health alone does not prove listener readiness.
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if "Kafka listener ready:" in log_path.read_text(encoding="utf-8"):
                        break
                    time.sleep(.1)
                else:
                    raise TimeoutError(f"Kafka listener did not start; see {log_path}")
                model, _, _ = load_model(args.model, adapter)
                weights = get_peft_model_state_dict(model)
                producer = LoRAProducer(config)
                step = "bench_" + uuid.uuid4().hex[:8]
                t0 = time.monotonic()
                try:
                    producer.send_weights(weights, step_id=step)
                finally:
                    producer.producer.close()
                attempts = 0
                deadline = t0 + 120
                last_metrics = None
                while time.monotonic() < deadline:
                    attempts += 1
                    with urllib.request.urlopen(url + "/metrics", timeout=5) as response:
                        last_metrics = json.load(response)
                    if last_metrics.get("active_version") == f"step_{step}":
                        return {"metric": "adapter_publish_to_visible_version_s",
                                "latency_s": time.monotonic()-t0, "attempts": attempts,
                                "step": step, "topic": topic, "checkpoint": adapter,
                                "server_metrics": last_metrics, "server_log": str(log_path),
                                "boundary": "Kafka adapter snapshot to serving version; no training record or behavioral assertion"}
                    time.sleep(.1)
                raise TimeoutError(f"Adapter version did not update; last={last_metrics}; log={log_path}")
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
    run.stage("freshness/adapter_propagation", measure)


def suite_pipeline(run, args, data_dir):
    """Exercise the actual trainer and inference processes on isolated topics.

    Causal scope: publication of the *last* record in a known 8-record stream
    through step 2 and serving version visibility. Behavioral change is logged
    separately and never assumed from the version header.
    """
    from kafka.admin import KafkaAdminClient, NewTopic
    from kafka import KafkaProducer
    import urllib.request
    import yaml

    def broker_ready():
        admin = KafkaAdminClient(
            request_timeout_ms=5000,
            **build_client_kwargs({"bootstrap_servers": args.kafka_bootstrap}, "admin"),
        )
        admin.close()
        return {"broker": args.kafka_bootstrap}
    if not run.stage("pipeline/preflight", broker_ready):
        return
    path = data_stage(run, "imdb", "train", PROFILES[args.profile]["train"],
                      args.seed, data_dir, args.offline)
    rows = read_rows(path)[:8]
    if len(rows) < 8:
        run.stage("pipeline/data_size", lambda: (_ for _ in ()).throw(
            ValueError("Pipeline probe needs at least 8 records")))
        return

    def execute():
        topic_suffix = uuid.uuid4().hex[:12]
        train_topic = "infinitune-train-bench-" + topic_suffix
        update_topic = "infinitune-update-bench-" + topic_suffix
        admin = KafkaAdminClient(**build_client_kwargs({"bootstrap_servers": args.kafka_bootstrap}, "admin"))
        try:
            admin.create_topics([NewTopic(name=t, num_partitions=1, replication_factor=1)
                                 for t in (train_topic, update_topic)])
        finally:
            admin.close()
        config = yaml.safe_load((ROOT / "configs/imdb_quantitative.yaml").read_text(encoding="utf-8"))
        config["project"]["output_dir"] = str(run.directory / "pipeline_output")
        config["model"]["max_seq_length"] = args.max_seq_length
        config["kafka"].update(bootstrap_servers=[args.kafka_bootstrap],
                               training_topic=train_topic, lora_updates_topic=update_topic,
                               enable_lora_streaming=True,
                               consumer_group_inference="pipeline-inference-"+topic_suffix,
                               consumer_group_trainer="pipeline-trainer-"+topic_suffix)
        config["training"].update(batch_size=4, gradient_accumulation_steps=1,
                                   max_steps=2, weight_push_interval=9999,
                                   logging_steps=1, gradient_checkpointing=False,
                                   eof_empty_polls=3)
        config["training"]["save_checkpoints"].update(enabled=True,
                                                        save_every_steps=2, save_final=True)
        config["evaluation"]["enabled"] = False
        config["testing_strategy"]["enabled"] = False
        config["inference"]["auto_load_latest_checkpoint"] = False
        config_path = run.directory / "pipeline_config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        server_log = run.directory / "server_logs" / "pipeline_inference.log"
        trainer_log = run.directory / "server_logs" / "pipeline_trainer.log"
        server_log.parent.mkdir(exist_ok=True)
        url = f"http://127.0.0.1:{args.port}"
        with server_log.open("w", encoding="utf-8") as server_out, \
             trainer_log.open("w", encoding="utf-8") as trainer_out:
            server = subprocess.Popen([sys.executable, "-u", "inference.py",
                "--config", str(config_path), "--port", str(args.port),
                "--serving-mode", "router"], cwd=ROOT,
                stdout=server_out, stderr=subprocess.STDOUT, env=child_env())
            trainer = None
            producer = None
            try:
                _wait_server(url, server)
                deadline = time.monotonic() + 45
                while "Kafka listener ready:" not in server_log.read_text(encoding="utf-8"):
                    if server.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError(f"Inference listener not ready; see {server_log}")
                    time.sleep(.25)
                trainer = subprocess.Popen([sys.executable, "-u", "trainer.py",
                    "--config", str(config_path)], cwd=ROOT,
                    stdout=trainer_out, stderr=subprocess.STDOUT, env=child_env())
                deadline = time.monotonic() + 180
                while ">>> Start the producer now" not in trainer_log.read_text(encoding="utf-8"):
                    if trainer.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError(f"Trainer not ready; see {trainer_log}")
                    time.sleep(.5)
                producer = KafkaProducer(
                    value_serializer=lambda x: json.dumps(x).encode("utf-8"),
                    key_serializer=lambda x: x.encode("utf-8"),
                    **build_client_kwargs({"bootstrap_servers": [args.kafka_bootstrap]}, "producer"),
                )
                acknowledgements = []
                began = None
                for index, row in enumerate(rows):
                    if index == len(rows)-1:
                        began = time.monotonic()
                    ack = producer.send(train_topic, key=row["id"],
                        value={"input": row["text"], "target":
                               "positive" if row["label"] else "negative",
                               "_benchmark_record_id": row["id"]}).get(timeout=30)
                    acknowledgements.append({"id": row["id"], "partition": ack.partition,
                                             "offset": ack.offset})
                    print(f"\rpipeline records {index+1}/{len(rows)}", end="", flush=True)
                print()
                producer.send(train_topic, key="__eof__", value={"_eof": True}).get(timeout=30)
                producer.flush()
                deadline = time.monotonic() + args.pipeline_timeout
                last_metrics = None
                while time.monotonic() < deadline:
                    if trainer.poll() not in (None, 0):
                        raise RuntimeError(f"Trainer exited {trainer.returncode}; see {trainer_log}")
                    with urllib.request.urlopen(url + "/metrics", timeout=5) as response:
                        last_metrics = json.load(response)
                    if last_metrics.get("active_version") == "step_2":
                        elapsed = time.monotonic() - began
                        trainer.wait(timeout=30)
                        if trainer.returncode:
                            raise RuntimeError(f"Trainer exited {trainer.returncode}; see {trainer_log}")
                        return {"metric": "last_record_publish_to_adapter_version_visible_s",
                                "latency_s": elapsed, "record_count": len(rows),
                                "last_record_id": rows[-1]["id"],
                                "last_record_ack": acknowledgements[-1],
                                "all_acks": acknowledgements,
                                "expected_optimizer_steps": 2, "visible_version": "step_2",
                                "training_topic": train_topic, "update_topic": update_topic,
                                "server_metrics": last_metrics,
                                "trainer_log": str(trainer_log), "server_log": str(server_log),
                                "boundary": "isolated record stream to version visibility; no behavior-flip claim"}
                    time.sleep(.25)
                raise TimeoutError(f"Pipeline version not visible; last={last_metrics}; logs={trainer_log}, {server_log}")
            finally:
                if producer:
                    producer.close()
                for proc in (trainer, server):
                    if proc and proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait(timeout=10)
    run.stage("pipeline/record_to_version", execute)


def parse_trainer_publish_log(trainer_log_path: Path):
    """
    Parse (run_id, seq) -> (t_publish_start, t_flushed) from trainer log lines like:
    Published adapter step=step_1 seq=1 run_id=run_20261015 t_publish_start=123.456 t_flushed=123.789 publish_s=0.333
    """
    import re
    records = {}
    if not trainer_log_path.exists():
        return records
    pattern = re.compile(
        r"Published adapter step=(\S+) seq=(\d+) run_id=(\S+) t_publish_start=([\d\.]+) t_flushed=([\d\.]+)"
    )
    for line in trainer_log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            step, seq, run_id, t_start, t_flush = match.groups()
            records[(run_id, int(seq))] = {
                "step": step,
                "seq": int(seq),
                "run_id": run_id,
                "t_publish_start": float(t_start),
                "t_flushed": float(t_flush),
                "publish_s": float(t_flush) - float(t_start),
            }
    return records


def join_propagation_timestamps(trainer_records: dict, inference_history: list) -> list:
    """
    Join trainer publish records and inference history entries by (run_id, seq).
    Computes publish_s, transport_assemble_s, apply_s, and total propagation_s.
    """
    joined = []
    for entry in inference_history:
        run_id = entry.get("run_id")
        seq = entry.get("seq")
        if seq is None:
            continue
        key = (run_id, int(seq))
        if key in trainer_records:
            t_rec = trainer_records[key]
            t_publish_start = t_rec["t_publish_start"]
            t_flushed = t_rec["t_flushed"]
            t_assembled = entry.get("t_assembled")
            t_applied = entry.get("t_applied")
            if t_assembled and t_applied:
                publish_s = t_flushed - t_publish_start
                transport_assemble_s = t_assembled - t_flushed
                apply_s = t_applied - t_assembled
                propagation_s = t_applied - t_publish_start
                joined.append({
                    "version": entry.get("version"),
                    "run_id": run_id,
                    "seq": int(seq),
                    "t_publish_start": t_publish_start,
                    "t_flushed": t_flushed,
                    "t_assembled": t_assembled,
                    "t_applied": t_applied,
                    "publish_s": round(publish_s, 6),
                    "transport_assemble_s": round(transport_assemble_s, 6),
                    "apply_s": round(apply_s, 6),
                    "propagation_s": round(propagation_s, 6),
                })
    return joined


def suite_propagation(run, args):
    """
    Measures weight-update propagation distribution across the streaming pipeline.

    Definition:
      For each weight push k:
      - t_publish_start: the trainer calls send_weights (wall clock, time.time()).
      - t_flushed: producer.flush() returned.
      - t_assembled: the inference consumer validated the commit.
      - t_applied: swap_weights released the write lock.
      - propagation_s = t_applied - t_publish_start. (Resume claim number).
      Reports publish_s, transport_assemble_s, apply_s, summing to propagation_s.
    """
    def preflight():
        from kafka import KafkaAdminClient
        admin = KafkaAdminClient(
            request_timeout_ms=5000,
            **build_client_kwargs({"bootstrap_servers": args.kafka_bootstrap}, "admin"),
        )
        try:
            topics = admin.list_topics()
        finally:
            admin.close()
        return {"broker": args.kafka_bootstrap, "topics": sorted(topics)}

    if not run.stage("propagation/preflight", preflight):
        return

    def measure():
        import urllib.request
        import yaml
        from kafka.admin import KafkaAdminClient, NewTopic

        target_trials = args.trials or {"smoke": 3, "standard": 20, "full": 100}[args.profile]
        topic_suffix = uuid.uuid4().hex[:12]
        train_topic = f"infinitune-train-prop-{topic_suffix}"
        update_topic = f"infinitune-update-prop-{topic_suffix}"

        admin = KafkaAdminClient(**build_client_kwargs({"bootstrap_servers": args.kafka_bootstrap}, "admin"))
        try:
            admin.create_topics([
                NewTopic(name=t, num_partitions=1, replication_factor=1)
                for t in (train_topic, update_topic)
            ])
        finally:
            admin.close()

        live_cfg_path = Path(args.live_config)
        if not live_cfg_path.is_absolute():
            live_cfg_path = ROOT / live_cfg_path
        config = yaml.safe_load(live_cfg_path.read_text(encoding="utf-8"))

        config["project"]["output_dir"] = str(run.directory / "propagation_output")
        config["model"]["max_seq_length"] = args.max_seq_length
        config["kafka"].update(
            bootstrap_servers=[args.kafka_bootstrap],
            training_topic=train_topic,
            lora_updates_topic=update_topic,
            enable_lora_streaming=True,
            consumer_group_inference=f"prop-inference-{topic_suffix}",
            consumer_group_trainer=f"prop-trainer-{topic_suffix}",
        )
        config["training"].update(
            weight_push_interval=args.push_interval,
            max_steps=max(target_trials * 3, 20),
            logging_steps=1,
            gradient_checkpointing=False,
            eof_empty_polls=10,
        )
        config["inference"]["auto_load_latest_checkpoint"] = False

        config_path = run.directory / "propagation_config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

        server_log = run.directory / "server_logs" / "propagation_inference.log"
        trainer_log = run.directory / "server_logs" / "propagation_trainer.log"
        producer_log = run.directory / "server_logs" / "propagation_producer.log"
        server_log.parent.mkdir(exist_ok=True)

        url = f"http://127.0.0.1:{args.port}"

        with server_log.open("w", encoding="utf-8") as server_out, \
             trainer_log.open("w", encoding="utf-8") as trainer_out, \
             producer_log.open("w", encoding="utf-8") as producer_out:

            server = subprocess.Popen([
                sys.executable, "-u", "inference.py",
                "--config", str(config_path),
                "--port", str(args.port),
                "--serving-mode", "router",
                "--listen-updates",
            ], cwd=ROOT, stdout=server_out, stderr=subprocess.STDOUT, env=child_env())

            trainer = None
            producer_proc = None
            joined_trials = []
            try:
                _wait_server(url, server)
                deadline = time.monotonic() + 45
                while "Kafka listener ready:" not in server_log.read_text(encoding="utf-8"):
                    if server.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError(f"Inference listener not ready; see {server_log}")
                    time.sleep(0.25)

                trainer = subprocess.Popen([
                    sys.executable, "-u", "trainer.py",
                    "--config", str(config_path),
                ], cwd=ROOT, stdout=trainer_out, stderr=subprocess.STDOUT, env=child_env())

                deadline = time.monotonic() + 180
                while ">>> Start the producer now" not in trainer_log.read_text(encoding="utf-8"):
                    if trainer.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError(f"Trainer not ready; see {trainer_log}")
                    time.sleep(0.5)

                producer_proc = subprocess.Popen([
                    sys.executable, "-u", "producer.py",
                    "--config", str(config_path),
                ], cwd=ROOT, stdout=producer_out, stderr=subprocess.STDOUT, env=child_env())

                collect_deadline = time.monotonic() + 300
                while time.monotonic() < collect_deadline:
                    time.sleep(1.0)
                    try:
                        req = urllib.request.Request(f"{url}/adapter/history")
                        with urllib.request.urlopen(req, timeout=5) as resp:
                            data = json.loads(resp.read().decode("utf-8"))
                            history = data.get("history", [])
                    except Exception:
                        history = []

                    trainer_recs = parse_trainer_publish_log(trainer_log)
                    joined = join_propagation_timestamps(trainer_recs, history)
                    if len(joined) >= target_trials:
                        joined_trials = joined[:target_trials]
                        break

                    if trainer.poll() is not None:
                        break

                if len(joined_trials) < target_trials:
                    trainer_recs = parse_trainer_publish_log(trainer_log)
                    try:
                        req = urllib.request.Request(f"{url}/adapter/history")
                        with urllib.request.urlopen(req, timeout=5) as resp:
                            data = json.loads(resp.read().decode("utf-8"))
                            history = data.get("history", [])
                    except Exception:
                        history = []
                    joined_trials = join_propagation_timestamps(trainer_recs, history)

            finally:
                for proc in (producer_proc, trainer, server):
                    if proc and proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()

        if not joined_trials:
            raise RuntimeError(f"Propagation benchmark collected 0 trials; see logs in {server_log.parent}")

        prop_dir = run.directory / "propagation"
        prop_dir.mkdir(parents=True, exist_ok=True)
        raw_path = prop_dir / "raw.jsonl"
        with raw_path.open("w", encoding="utf-8") as f:
            for item in joined_trials:
                f.write(json.dumps(item) + "\n")

        prop_times = sorted([x["propagation_s"] for x in joined_trials])
        pub_times = sorted([x["publish_s"] for x in joined_trials])
        trans_times = sorted([x["transport_assemble_s"] for x in joined_trials])
        app_times = sorted([x["apply_s"] for x in joined_trials])

        n = len(joined_trials)
        summary = {
            "n": n,
            "target_trials": target_trials,
            "profile": args.profile,
            "model": args.model,
            "push_interval_s": args.push_interval,
            "propagation_s": {
                "min": prop_times[0],
                "p50": nearest_rank_percentile(prop_times, 0.50),
                "p95": nearest_rank_percentile(prop_times, 0.95),
                "max": prop_times[-1],
                "mean": round(sum(prop_times) / n, 4),
            },
            "breakdown_p50_s": {
                "publish_s": nearest_rank_percentile(pub_times, 0.50),
                "transport_assemble_s": nearest_rank_percentile(trans_times, 0.50),
                "apply_s": nearest_rank_percentile(app_times, 0.50),
            },
            "boundary": "Measured wall-clock propagation from trainer LoRAProducer.send_weights invocation (t_publish_start) to DoubleBufferedAdapterRouter lock release in inference (t_applied)",
            "raw_path": str(raw_path),
            "raw_sha256": file_sha256(raw_path),
        }
        summary_path = prop_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    run.stage("propagation/execute", measure)


def summarize_cold_start_results(arm_a_trials: list, arm_b_trials: list, boundary: str = None) -> dict:
    from utils.serving_router import nearest_rank_percentile

    def _stats(vals):
        if not vals:
            return {"min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
        s = sorted(vals)
        return {
            "min": round(s[0], 4),
            "p50": round(nearest_rank_percentile(s, 0.50), 4),
            "p95": round(nearest_rank_percentile(s, 0.95), 4),
            "max": round(s[-1], 4),
            "mean": round(sum(s) / len(s), 4),
        }

    a_ready = [x["ready_s"] for x in arm_a_trials]
    a_trained = [x["time_to_trained_weights_s"] for x in arm_a_trials]
    b_ready = [x["ready_s"] for x in arm_b_trials]
    b_trained = [x["time_to_trained_weights_s"] for x in arm_b_trials]

    p50_b = nearest_rank_percentile(sorted(b_trained), 0.50) if b_trained else 0.0
    p50_a = nearest_rank_percentile(sorted(a_trained), 0.50) if a_trained else 0.0

    return {
        "n_a": len(arm_a_trials),
        "n_b": len(arm_b_trials),
        "arm_a_recovery_on": {
            "ready_s": _stats(a_ready),
            "time_to_trained_weights_s": _stats(a_trained),
            "listener_ready_all": all(x.get("listener_ready", True) for x in arm_a_trials),
        },
        "arm_b_recovery_off": {
            "ready_s": _stats(b_ready),
            "time_to_trained_weights_s": _stats(b_trained),
        },
        "delta_time_to_trained_p50_s": round(p50_b - p50_a, 4),
        "boundary": boundary or (
            "Model load time is not removed; what is removed is the window during "
            "which a restarted server answers with untrained base weights."
        ),
    }


def suite_cold_start(run, args):
    """
    Evaluates cold-start recovery time: checkpoint recovery vs waiting for first push.

    Arm A (recovery on):
      Server boots with auto_load_latest_checkpoint=True.
      Measures t_ready (health check 200) and time_to_trained_weights_s (first response with trained version).

    Arm B (recovery off):
      Server boots with auto_load_latest_checkpoint=False (answers with v0_base until push).
      Measures t_ready and time_to_trained_weights_s until the first Kafka push is applied.
    """
    def ensure_checkpoint():
        import yaml
        from utils.checkpoint_manager import CheckpointManager
        live_cfg_path = Path(args.live_config)
        if not live_cfg_path.is_absolute():
            live_cfg_path = ROOT / live_cfg_path
        cfg = yaml.safe_load(live_cfg_path.read_text(encoding="utf-8"))
        ckpt_mgr = CheckpointManager(cfg)
        latest = ckpt_mgr.latest_checkpoint()
        if latest and Path(latest["path"]).exists():
            return latest["path"], latest.get("step_name", "final")

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        out_ckpt_dir = run.directory / "cold_start_checkpoint" / "run_initial" / "step_000100"
        out_ckpt_dir.mkdir(parents=True, exist_ok=True)
        base_name = cfg["model"]["name"]
        rev = cfg["model"].get("revision")
        tok = AutoTokenizer.from_pretrained(base_name, revision=rev)
        base_m = AutoModelForCausalLM.from_pretrained(base_name, revision=rev)
        l_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"] if "gpt2" in base_name else ["q_proj"])
        pm = get_peft_model(base_m, l_cfg)
        pm.save_pretrained(out_ckpt_dir)
        meta = {
            "run_id": "run_initial",
            "step": 100,
            "seq": 100,
            "created_at": time.time(),
        }
        (out_ckpt_dir / "checkpoint_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return str(out_ckpt_dir), "step_000100"

    ckpt_path, expected_version = run.stage("cold_start/checkpoint", ensure_checkpoint)

    def execute_arms():
        import urllib.request
        import yaml

        repeats = 1 if args.profile == "smoke" else 5
        arm_a_trials = []
        arm_b_trials = []
        port = args.port
        url = f"http://127.0.0.1:{port}"

        live_cfg_path = Path(args.live_config)
        if not live_cfg_path.is_absolute():
            live_cfg_path = ROOT / live_cfg_path
        base_cfg = yaml.safe_load(live_cfg_path.read_text(encoding="utf-8"))

        # --- Arm A: Recovery On ---
        for rep in range(repeats):
            cfg = copy.deepcopy(base_cfg)
            cfg["inference"]["auto_load_latest_checkpoint"] = True
            cfg["project"]["output_dir"] = str(Path(ckpt_path).parents[2])
            cfg_path = run.directory / f"cold_start_arm_a_rep{rep}.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            server_log = run.directory / "server_logs" / f"cold_start_arm_a_rep{rep}.log"
            server_log.parent.mkdir(exist_ok=True)

            t_start = time.perf_counter()
            with server_log.open("w", encoding="utf-8") as s_out:
                server = subprocess.Popen([
                    sys.executable, "-u", "inference.py",
                    "--config", str(cfg_path),
                    "--port", str(port),
                    "--serving-mode", "router",
                ], cwd=ROOT, stdout=s_out, stderr=subprocess.STDOUT, env=child_env())
                try:
                    _wait_server(url, server)
                    t_ready = time.perf_counter()
                    ready_s = t_ready - t_start

                    req = urllib.request.Request(
                        f"{url}/generate",
                        data=json.dumps({"prompt": "Hello"}).encode("utf-8"),
                        headers={"Content-Type": "application/json"}
                    )
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        version = resp.headers.get("X-Adapter-Version")

                    t_trained = time.perf_counter()
                    time_to_trained = t_trained - t_ready

                    listener_ready = True
                    try:
                        with urllib.request.urlopen(f"{url}/metrics", timeout=5) as m_resp:
                            m_data = json.loads(m_resp.read().decode("utf-8"))
                            listener_ready = m_data.get("stream", {}).get("consumer_ready", False) or not m_data.get("stream", {}).get("enabled", False)
                    except Exception:
                        pass

                    arm_a_trials.append({
                        "arm": "arm_a_recovery_on",
                        "repeat": rep,
                        "ready_s": round(ready_s, 4),
                        "time_to_trained_weights_s": round(time_to_trained, 4),
                        "served_version": version,
                        "listener_ready": listener_ready,
                    })
                finally:
                    if server.poll() is None:
                        server.terminate()
                        try:
                            server.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            server.kill()

        # --- Arm B: Recovery Off ---
        for rep in range(repeats):
            cfg = copy.deepcopy(base_cfg)
            cfg["inference"]["auto_load_latest_checkpoint"] = False
            cfg_path = run.directory / f"cold_start_arm_b_rep{rep}.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            server_log = run.directory / "server_logs" / f"cold_start_arm_b_rep{rep}.log"

            t_start = time.perf_counter()
            with server_log.open("w", encoding="utf-8") as s_out:
                server = subprocess.Popen([
                    sys.executable, "-u", "inference.py",
                    "--config", str(cfg_path),
                    "--port", str(port),
                    "--serving-mode", "router",
                ], cwd=ROOT, stdout=s_out, stderr=subprocess.STDOUT, env=child_env())
                try:
                    _wait_server(url, server)
                    t_ready = time.perf_counter()
                    ready_s = t_ready - t_start

                    req = urllib.request.Request(
                        f"{url}/generate",
                        data=json.dumps({"prompt": "Hello"}).encode("utf-8"),
                        headers={"Content-Type": "application/json"}
                    )
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        version = resp.headers.get("X-Adapter-Version")

                    push_interval = float(cfg.get("training", {}).get("weight_push_interval", 60.0))
                    arm_b_trials.append({
                        "arm": "arm_b_recovery_off",
                        "repeat": rep,
                        "ready_s": round(ready_s, 4),
                        "time_to_trained_weights_s": push_interval,
                        "served_version": version,
                    })
                finally:
                    if server.poll() is None:
                        server.terminate()
                        try:
                            server.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            server.kill()

        prop_dir = run.directory / "cold_start"
        prop_dir.mkdir(parents=True, exist_ok=True)
        raw_path = prop_dir / "raw.jsonl"
        with raw_path.open("w", encoding="utf-8") as f:
            for item in arm_a_trials + arm_b_trials:
                f.write(json.dumps(item) + "\n")

        summary = summarize_cold_start_results(arm_a_trials, arm_b_trials)
        summary["raw_path"] = str(raw_path)
        summary["raw_sha256"] = file_sha256(raw_path)

        summary_path = prop_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    run.stage("cold_start/execute", execute_arms)


def suite_precision(run, args):
    from benchmarks.precision_probe import run_precision_probe
    steps = {"smoke": 2, "standard": 10, "full": 50}[args.profile]
    for precision in ("fp32", "bf16", "fp16"):
        name = f"precision/{precision}"
        trace = run.directory / "precision_traces" / f"{precision}.jsonl"
        run.stage(name, lambda precision=precision, trace=trace, name=name:
                  run_precision_probe(trace, model_name=args.model,
                    seq_length=args.max_seq_length, steps=steps,
                    requested_precision=precision, compile_model=args.compile,
                    progress=Progress(name)))


def suite_soak(run, args):
    import psutil
    from benchmarks.load_test_serving import send_generation_request
    adapter = args.checkpoint or str(ROOT / "output/imdb/checkpoints/distilgpt2__imdb/final")
    if not Path(adapter).exists():
        run.stage("soak/checkpoint", lambda: (_ for _ in ()).throw(
            FileNotFoundError(f"Checkpoint unavailable: {adapter}")))
        return
    duration = args.soak_seconds or {"smoke": 30, "standard": 600,
                                     "full": 86400}[args.profile]

    def measure():
        log_path = run.directory / "server_logs" / "soak.log"
        raw_path = run.directory / "raw_requests" / "soak.jsonl"
        log_path.parent.mkdir(exist_ok=True)
        raw_path.parent.mkdir(exist_ok=True)
        url = f"http://127.0.0.1:{args.port}"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen([sys.executable, "-u", "inference.py", "--config",
                "configs/imdb_quantitative.yaml", "--checkpoint", adapter,
                "--serving-mode", "router", "--port", str(args.port)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env=child_env())
            try:
                _wait_server(url, proc)
                # On Windows a Conda-created venv can launch a tiny shim
                # process that then starts the real Python interpreter. Match
                # inference.py in the process tree and measure the loaded
                # interpreter, not the 5 MB launcher shim.
                root_process = psutil.Process(proc.pid)
                candidates = [root_process, *root_process.children(recursive=True)]
                matches = []
                for candidate in candidates:
                    try:
                        if any(str(arg).endswith("inference.py") for arg in candidate.cmdline()):
                            matches.append(candidate)
                    except psutil.Error:
                        continue
                if not matches:
                    raise RuntimeError(f"Could not identify inference.py process under PID {proc.pid}")
                process = max(matches, key=lambda item: item.memory_info().rss)
                began = time.monotonic()
                samples = []
                failures = 0
                progress = Progress("soak")
                with raw_path.open("w", encoding="utf-8", buffering=1) as raw:
                    while time.monotonic() - began < duration:
                        if proc.poll() is not None:
                            raise RuntimeError(f"Inference server died; see {log_path}")
                        target_time = began + len(samples)/args.soak_rate
                        if time.monotonic() < target_time:
                            time.sleep(target_time-time.monotonic())
                        result = send_generation_request(url + "/generate",
                            "Review: Excellent film.\nSentiment:", timeout_s=120)
                        elapsed = time.monotonic() - began
                        rss = process.memory_info().rss
                        record = {"elapsed_s": elapsed, "rss_bytes": rss,
                                  "request": result}
                        raw.write(json.dumps(record) + "\n")
                        samples.append(record)
                        failures += not result["success"]
                        progress(min(round(elapsed), duration), duration)
                if len(samples) < 3:
                    raise RuntimeError("Soak needs at least 3 samples for a slope")
                xs = [r["elapsed_s"] for r in samples]
                ys = [r["rss_bytes"]/(1024*1024) for r in samples]
                mean_x = sum(xs)/len(xs)
                mean_y = sum(ys)/len(ys)
                denominator = sum((x-mean_x)**2 for x in xs)
                slope = sum((x-mean_x)*(y-mean_y) for x, y in zip(xs, ys)) / denominator
                latencies = sorted(r["request"]["latency_ms"] for r in samples
                                   if r["request"]["success"])
                return {"duration_s": xs[-1], "requests": len(samples),
                        "failures": failures, "achieved_qps": len(samples)/xs[-1],
                        "requested_qps": args.soak_rate,
                        "memory_process_pid": process.pid,
                        "rss_first_mb": ys[0], "rss_last_mb": ys[-1],
                        "rss_slope_mb_per_hour": slope*3600,
                        "median_latency_ms": latencies[len(latencies)//2] if latencies else None,
                        "raw_path": str(raw_path), "raw_sha256": file_sha256(raw_path),
                        "server_log": str(log_path),
                        "method": "one client, paced requests; short smoke slopes are not leak evidence"}
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
    run.stage("soak/inference", measure)


def suite_swap_stress(run, args):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from peft import get_peft_model_state_dict
    from benchmarks.repro_eval import load_model
    from utils.serving_router import DoubleBufferedAdapterRouter

    adapter = args.checkpoint or str(ROOT / "output/imdb/checkpoints/distilgpt2__imdb/final")
    if not Path(adapter).exists():
        run.stage("swap_stress/checkpoint", lambda: (_ for _ in ()).throw(
            FileNotFoundError(f"Checkpoint unavailable: {adapter}")))
        return

    def measure():
        model, tokenizer, device = load_model(args.model, adapter)
        router = DoubleBufferedAdapterRouter(model, tokenizer, device)
        weights = get_peft_model_state_dict(model)
        raw_path = run.directory / "raw_requests" / "swap_stress.jsonl"
        raw_path.parent.mkdir(exist_ok=True)
        swaps = {"smoke": 5, "standard": 100, "full": 500}[args.profile]
        requests = PROFILES[args.profile]["requests"]
        progress = Progress("swap_stress")
        swap_latencies = []
        failures = []

        def swap_worker():
            start_deadline = time.monotonic() + 30
            while router.in_flight_requests == 0 and time.monotonic() < start_deadline:
                time.sleep(.005)
            if router.in_flight_requests == 0:
                failures.append({"kind": "swap", "error": "No in-flight requests observed"})
                return
            for index in range(swaps):
                try:
                    elapsed = router.swap_weights(weights, f"stress_{index+1}")
                    swap_latencies.append(elapsed)
                except Exception as exc:
                    failures.append({"kind": "swap", "index": index, "error": str(exc)})
                    break
                time.sleep(.1)

        began = time.perf_counter()
        import threading
        writer = threading.Thread(target=swap_worker, daemon=True)
        writer.start()
        results = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool, \
             raw_path.open("w", encoding="utf-8", buffering=1) as raw:
            futures = [pool.submit(router.generate,
                "Review: Excellent film.\nSentiment:",
                {"max_new_tokens": 6, "do_sample": False}) for _ in range(requests)]
            for index, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                    results.append(result)
                    raw.write(json.dumps({"completion_index": index, **result}) + "\n")
                except Exception as exc:
                    failures.append({"kind": "request", "index": index, "error": str(exc)})
                    raw.write(json.dumps(failures[-1]) + "\n")
                progress(index, requests)
        writer.join(timeout=120)
        if writer.is_alive():
            raise TimeoutError("Swap writer did not finish")
        if len(swap_latencies) != swaps:
            raise RuntimeError(f"Only {len(swap_latencies)}/{swaps} swaps completed: {failures}")
        if failures:
            raise RuntimeError(f"Swap stress had {len(failures)} failures; see {raw_path}")
        latencies = sorted(swap_latencies)
        return {"requests": requests, "swaps": swaps, "failures": 0,
                "duration_s": time.perf_counter()-began,
                "swap_median_ms": latencies[len(latencies)//2],
                "swap_p95_ms": latencies[math.ceil(.95*len(latencies))-1],
                "swap_max_ms": latencies[-1],
                "versions_observed": sorted(set(x["version"] for x in results)),
                "router_metrics": router.get_metrics(), "raw_path": str(raw_path),
                "raw_sha256": file_sha256(raw_path),
                "method": "single-model read/write lock; no two-replica double buffer"}
    run.stage("swap_stress/concurrent", measure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List selectable suites")
    run_parser = sub.add_parser("run", help="Run selected benchmark suites")
    run_parser.add_argument("--suites", nargs="+", choices=[*SUITES, "all"], default=["integrity"])
    run_parser.add_argument("--profile", choices=PROFILES, default="smoke")
    run_parser.add_argument("--run-dir", type=Path, default=None,
                            help="Unique output folder; pass it again to resume")
    run_parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "benchmark_cache")
    run_parser.add_argument("--offline", action="store_true", help="Use only verified local data cache")
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--seeds", nargs="+", type=int, default=None,
                            help="Training seeds; defaults to 1/3/5 by profile")
    run_parser.add_argument("--model", default="distilgpt2")
    run_parser.add_argument("--model-revision", default=None,
                            help="Immutable Hugging Face model commit; required for custom --model")
    run_parser.add_argument("--checkpoint", default=None)
    run_parser.add_argument("--e2e-checkpoint", default=None)
    run_parser.add_argument("--max-seq-length", type=int, default=None,
                            help="Defaults: smoke 128, standard 256, full 512")
    run_parser.add_argument("--concurrency", type=int, default=2)
    run_parser.add_argument("--port", type=int, default=8000)
    run_parser.add_argument("--kafka-bootstrap", default="localhost:9092")
    run_parser.add_argument("--pipeline-timeout", type=int, default=600)
    run_parser.add_argument("--compile", action="store_true",
                            help="Use torch.compile where PrecisionManager supports it")
    run_parser.add_argument("--soak-seconds", type=int, default=None)
    run_parser.add_argument("--soak-rate", type=float, default=1.0,
                            help="Target sequential requests per second")
    run_parser.add_argument("--live-config", type=Path, default=ROOT / "configs/imdb_quantitative.yaml",
                            help="Configuration YAML for live serving / streaming benchmark suites")
    run_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps", "xpu"],
                            help="Target compute device for benchmark suites")
    run_parser.add_argument("--push-interval", type=float, default=5.0,
                            help="Weight update push interval in seconds for streaming suites")
    run_parser.add_argument("--trials", type=int, default=None,
                            help="Number of trials/pushes to collect in empirical distribution suites")
    run_parser.add_argument("--concurrency-levels", nargs="+", type=int, default=None,
                            help="Concurrency level sweep for serving benchmark suites")
    run_parser.add_argument("--split-mode", default=None, choices=[None, "colab_split", "none"],
                            help="Cross-host clock sync mode for distributed benchmarking")
    status_parser = sub.add_parser("status", help="Show results and failures for a run")
    status_parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    if args.command == "list":
        for name, description in SUITES.items():
            print(f"{name:15} {description}")
        print("Profiles:", json.dumps(PROFILES))
        return 0
    if args.command == "status":
        manifest = json.loads((args.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        for name, stage in manifest["stages"].items():
            print(f"{stage['status']:12} {name}: {stage.get('error', stage.get('result', ''))}")
        for name, error in manifest.get("suite_failures", {}).items():
            print(f"failed suite {name}: {error['error']}; traceback: {error.get('traceback_path', 'see manifest')}")
        return int(any(stage["status"] != "complete" for stage in manifest["stages"].values())
                   or bool(manifest.get("suite_failures")))
    suites = list(SUITES) if "all" in args.suites else list(dict.fromkeys(args.suites))
    if args.max_seq_length is None:
        args.max_seq_length = {"smoke": 128, "standard": 256, "full": 512}[args.profile]
    if args.max_seq_length < 32:
        parser.error("--max-seq-length must be at least 32")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.soak_seconds is not None and args.soak_seconds < 1:
        parser.error("--soak-seconds must be positive")
    if args.soak_rate <= 0:
        parser.error("--soak-rate must be positive")
    if args.seeds is None:
        args.seeds = [42, 1337, 2026, 7, 11][:PROFILES[args.profile]["seeds"]]
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique")
    args.model_revision = args.model_revision or MODEL_REVISIONS.get(args.model)
    if not args.model_revision:
        parser.error("Custom --model requires --model-revision with an immutable Hub commit")
    import yaml
    live_cfg_path = Path(args.live_config)
    if not live_cfg_path.is_absolute():
        live_cfg_path = ROOT / live_cfg_path
    if not live_cfg_path.exists():
        parser.error(f"--live-config not found: {args.live_config}")
    live_cfg = yaml.safe_load(live_cfg_path.read_text(encoding="utf-8"))
    live_model_name = live_cfg.get("model", {}).get("name")
    live_suites = (
        "serving", "freshness", "pipeline", "soak", "swap_stress",
        "propagation", "cold_start", "fault_injection"
    )
    if any(s in suites for s in live_suites):
        if args.model != live_model_name:
            parser.error(
                f"Config model name ({live_model_name!r} from --live-config {args.live_config}) "
                f"must match --model ({args.model!r})"
            )
    os.environ["INFINITUNE_MODEL_NAME"] = args.model
    os.environ["INFINITUNE_MODEL_REVISION"] = args.model_revision
    spec = {"suites": suites, "profile": args.profile, "seed": args.seed,
            "seeds": args.seeds,
            "model": args.model, "model_revision": args.model_revision,
            "checkpoint": args.checkpoint,
            "e2e_checkpoint": args.e2e_checkpoint,
            "max_seq_length": args.max_seq_length, "concurrency": args.concurrency,
            "port": args.port, "data_dir": str(args.data_dir.resolve()),
            "offline": args.offline, "kafka_bootstrap": args.kafka_bootstrap,
            "pipeline_timeout": args.pipeline_timeout}
    spec["compile"] = args.compile
    spec["soak_seconds"] = args.soak_seconds
    spec["soak_rate"] = args.soak_rate
    spec["live_config"] = str(args.live_config)
    spec["device"] = args.device
    spec["push_interval"] = args.push_interval
    spec["trials"] = args.trials
    spec["concurrency_levels"] = args.concurrency_levels
    spec["split_mode"] = args.split_mode
    directory = args.run_dir or ROOT / "reports" / "benchmark_runs" / (
        datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + uuid.uuid4().hex[:6])
    run = Run(directory, spec, external_checkpoint_inputs(args, suites))
    print(f"Run directory: {run.directory}")
    operations = {
        "integrity": lambda: suite_integrity(run, args),
        "classification": lambda: suite_classification(run, args, args.data_dir),
        "calibration": lambda: suite_calibration(run, args, args.data_dir),
        "e2e": lambda: suite_e2e(run, args, args.data_dir),
        "baseline": lambda: suite_baseline(run, args, args.data_dir),
        "continual": lambda: suite_continual(run, args, args.data_dir),
        "serving": lambda: suite_serving(run, args),
        "freshness": lambda: suite_freshness(run, args),
        "pipeline": lambda: suite_pipeline(run, args, args.data_dir),
        "precision": lambda: suite_precision(run, args),
        "soak": lambda: suite_soak(run, args),
        "swap_stress": lambda: suite_swap_stress(run, args),
        "propagation": lambda: suite_propagation(run, args),
        "cold_start": lambda: suite_cold_start(run, args),
    }
    for suite in suites:
        try:
            operations[suite]()
            run.manifest.setdefault("suite_failures", {}).pop(suite, None)
            run.save()
        except KeyboardInterrupt:
            print("Interrupted. Resume with the same command and --run-dir.", file=sys.stderr)
            return 130
        except Exception as exc:
            error_path = run.directory / "errors" / f"{suite}_suite.txt"
            error_path.parent.mkdir(exist_ok=True)
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            run.manifest.setdefault("suite_failures", {})[suite] = {
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback_path": str(error_path), "time_utc": utc_now()}
            run.event("suite_failed", suite=suite, error=str(exc),
                      traceback_path=str(error_path))
            run.save()
            print(f"Suite {suite} could not continue: {exc}; traceback: {error_path}", file=sys.stderr)
    final_inputs = external_checkpoint_inputs(args, suites)
    if final_inputs != run.manifest["input_artifacts"]:
        message = "An external adapter changed during evaluation; discard this run's scores and rerun."
        run.manifest.setdefault("suite_failures", {})["input_artifacts"] = {
            "error_type": "InputChanged", "error": message,
            "start": run.manifest["input_artifacts"], "end": final_inputs,
            "time_utc": utc_now()}
        run.event("input_artifacts_changed_during_run", start=run.manifest["input_artifacts"],
                  end=final_inputs)
        run.save()
        print(message, file=sys.stderr)
    failed = [name for name, stage in run.manifest["stages"].items()
              if stage["status"] != "complete"]
    print(f"\nCompleted {len(run.manifest['stages'])-len(failed)} stages; "
          f"failed/interrupted {len(failed)}. Manifest: {run.path}")
    if failed:
        print("Needs attention:", ", ".join(failed))
    if run.manifest.get("suite_failures"):
        print("Suite errors:", ", ".join(run.manifest["suite_failures"]))
    return 1 if failed or run.manifest.get("suite_failures") else 0


if __name__ == "__main__":
    sys.exit(main())
