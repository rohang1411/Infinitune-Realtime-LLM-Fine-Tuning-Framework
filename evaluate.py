"""
evaluate.py
─────────────────────────────────────────────────────────────────────────────
Standalone decoupled evaluation script for InfiniTune.

Loads a saved LoRA adapter checkpoint (produced by trainer.py) and runs the
full evaluation suite (quantitative + qualitative) as configured in the YAML
config.  Results are saved to a versioned directory that never overwrites
previous evaluation runs.

Usage
-----
  # Evaluate the 'final' checkpoint (default)
  python evaluate.py --config configs/imdb_quantitative.yaml

  # Evaluate a specific step
  python evaluate.py --config configs/imdb_quantitative.yaml --step 200

  # Evaluate a specific checkpoint directory directly
  python evaluate.py --config configs/imdb_quantitative.yaml \\
      --checkpoint-dir output/imdb/checkpoints/distilgpt2__stanfordnlp_imdb/step_000200

  # Evaluate ALL checkpoints for a config (sequential)
  python evaluate.py --config configs/imdb_quantitative.yaml --all-checkpoints

  # List available checkpoints without running evaluation
  python evaluate.py --config configs/imdb_quantitative.yaml --list

Output directory layout
-----------------------
  <output_dir>/eval_results/<model>__<dataset>/
      step_000200/
          eval_20260412-150230_a3f2/       ← timestamp + uuid suffix (never overwrites)
              eval_results.json            ← full metrics dict
              eval_config.json             ← config snapshot used for this eval
              plots/
                  *.png
"""

import argparse
import csv
import json
import os
import sys
import time
import uuid
import signal
import statistics
from pathlib import Path
from typing import Optional

import torch
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers (consistent with rest of codebase)
# ─────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}][EVALUATE] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Device detection
# ─────────────────────────────────────────────────────────────────────────────

def _get_device(device_str: str = "auto") -> torch.device:
    if device_str and device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Results persistence
# ─────────────────────────────────────────────────────────────────────────────

def _make_eval_output_dir(config: dict, checkpoint_name: str) -> str:
    """
    Create a versioned, never-overwriting output directory for this eval run.

    Layout:  <output_dir>/eval_results/<model>__<dataset>/<checkpoint>/eval_<ts>_<uid>/
    """
    from utils.checkpoint_manager import _slugify

    project_cfg  = config.get("project", {})
    model_cfg    = config.get("model", {})
    dataset_cfg  = config.get("dataset", {})

    output_dir   = project_cfg.get("output_dir", "./output")
    model_slug   = _slugify(os.path.basename(model_cfg.get("name", "model")))
    dataset_slug = _slugify(dataset_cfg.get("name", "dataset"))
    scope        = f"{model_slug}__{dataset_slug}"

    ts     = time.strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:4]
    run    = f"eval_{ts}_{suffix}"

    out_dir = os.path.join(output_dir, "eval_results", scope, checkpoint_name, run)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _save_results(out_dir: str, metrics: dict, config: dict, checkpoint_path: str) -> None:
    """Write eval_results.json, eval_config.json, and config_snapshot.yaml to *out_dir*."""
    results_path = os.path.join(out_dir, "eval_results.json")
    config_json_path = os.path.join(out_dir, "eval_config.json")
    config_yaml_path = os.path.join(out_dir, "config_snapshot.yaml")

    payload = {
        "checkpoint_path": checkpoint_path,
        "eval_timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metrics":         metrics,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    # Also save a clean YAML snapshot for human readability
    try:
        import yaml
        with open(config_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    except Exception:
        pass  # yaml optional — JSON is always written

    _log(f"Results saved to: {results_path}")


def _generate_plots(metrics_over_time: list, out_dir: str, config: dict = None) -> None:
    """
    Generate the shared evaluation artifact bundle from a list of
    {metric_name: value} dicts.

    For a single checkpoint, metrics_over_time has one entry.
    For --all-checkpoints, it has one entry per checkpoint.
    """
    # Write a temporary CSV that generate_plots() can read
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Collect all keys
    all_keys = set()
    for row in metrics_over_time:
        all_keys.update(row.keys())
    all_keys.discard("checkpoint")
    # Ensure step is first, then sorted rest
    fieldnames = ["step"] + sorted(k for k in all_keys if k != "step")

    csv_path = os.path.join(plots_dir, "eval_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in metrics_over_time:
            # Normalise step: map "final" numerically for plotting
            clean_row = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(clean_row)

    try:
        from utils.evaluation_artifacts import generate_evaluation_artifacts
        context = "all_checkpoints_eval" if len(metrics_over_time) > 1 else "decoupled_eval"
        manifest = generate_evaluation_artifacts(
            metrics_csv_path=csv_path,
            run_root=out_dir,
            config=config,
            context=context,
        )
        bundle = manifest.get("artifact_bundle")
        if bundle:
            _log(f"Evaluation artifacts generated in: {bundle}")
    except Exception as exc:
        _log(f"Warning: plot generation failed: {exc}")
        import traceback; traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Full-pool evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    model,
    tokenizer,
    config: dict,
    device: torch.device,
    evaluator_bundle: dict | None = None,
    run_quantitative: bool = True,
    run_qualitative: bool = True,
    step: int = 0,
) -> dict:
    """
    Run the enabled evaluation suite for one checkpoint and return a flat dict
    of all metric values.

    Unlike inline eval, this runs the entire configured evaluation pool for
    the evaluators selected for this checkpoint.
    """
    from utils.eval_metrics_train import Evaluator
    from utils.eval_qualitative import QualitativeEvaluator
    from trainer import tokenize_with_label_masking, pad_batch

    combined_metrics: dict = {}

    eval_cfg = config.get("evaluation", {})
    if eval_cfg.get("enabled", False):
        if run_quantitative:
            _log("Running quantitative evaluation (full pool)...")
            evaluator = (evaluator_bundle or {}).get("quantitative")
            if evaluator is None:
                evaluator = Evaluator(
                    config, tokenizer, device,
                    tokenize_with_label_masking, pad_batch,
                )
                if evaluator_bundle is not None:
                    evaluator_bundle["quantitative"] = evaluator
            else:
                evaluator.tokenizer = tokenizer
                evaluator.device = device
            if evaluator.enabled and evaluator.eval_data:
                pool_size = len(evaluator.eval_data)
                evaluator.eval_batch_size = pool_size
                quant_metrics = evaluator.evaluate(model, step=step) or {}
                combined_metrics.update(quant_metrics)
                _log(f"Quantitative metrics: {list(quant_metrics.keys())}")
            else:
                _log("Quantitative evaluator is disabled or has no data - skipping.")
        else:
            _log(
                "Skipping quantitative evaluation at this checkpoint because "
                f"evaluation.eval_interval={eval_cfg.get('eval_interval', 1)} does not select this step."
            )
    else:
        _log("Quantitative evaluation disabled in config (evaluation.enabled: false).")

    ts_cfg = config.get("testing_strategy", {})
    if ts_cfg.get("enabled", False):
        if run_qualitative:
            _log("Running qualitative evaluation (full pool)...")
            qual_eval = (evaluator_bundle or {}).get("qualitative")
            if qual_eval is None:
                qual_eval = QualitativeEvaluator(config, tokenizer, device)
                if evaluator_bundle is not None:
                    evaluator_bundle["qualitative"] = qual_eval
            else:
                qual_eval.tokenizer = tokenizer
                qual_eval.device = device
            if qual_eval.enabled and qual_eval.eval_data:
                qual_eval._eval_samples = len(qual_eval.eval_data)
                qual_metrics = qual_eval.run(model, step=step) or {}
                combined_metrics.update(qual_metrics)
                _log(f"Qualitative metrics: {list(qual_metrics.keys())}")
            else:
                _log("Qualitative evaluator is disabled or has no data - skipping.")
        else:
            _log(
                "Skipping qualitative evaluation at this checkpoint because "
                f"testing_strategy.eval_interval={ts_cfg.get('eval_interval', 1)} does not select this step."
            )
    else:
        _log("Qualitative evaluation disabled in config (testing_strategy.enabled: false).")

    return combined_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation flow
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_checkpoint(
    ckpt_path: str,
    config: dict,
    config_file: str,
    evaluator_bundle: dict | None = None,
    run_quantitative: bool = True,
    run_qualitative: bool = True,
    step: int = 0,
    device: torch.device | None = None,
) -> dict:
    """Load a checkpoint and run evaluation. Returns the combined metrics dict."""
    from utils.checkpoint_manager import CheckpointManager

    if device is None:
        device = _get_device()
    _log(f"Device: {device}")

    ckpt_manager = CheckpointManager(config, config_path=config_file)

    if ckpt_path == "base_model":
        _log("Loading base model (no adapter)...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = config.get("model", {}).get("name")
        prec = config.get("model", {}).get("precision", "fp32")
        dtype = torch.float16 if prec == 'fp16' else (torch.bfloat16 if prec == 'bf16' else torch.float32)
        
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        _log(f"Loading checkpoint: {ckpt_path}")
        model, tokenizer = ckpt_manager.load(ckpt_path, device=str(device))

    metrics = run_evaluation(
        model,
        tokenizer,
        config,
        device,
        evaluator_bundle=evaluator_bundle,
        run_quantitative=run_quantitative,
        run_qualitative=run_qualitative,
        step=step,
    )

    # Free GPU/MPS memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Offline Checkpoint Auditing Watch Service
# ─────────────────────────────────────────────────────────────────────────────

def is_checkpoint_complete(ckpt_dir: Path | str) -> bool:
    """Check if checkpoint directory has valid checkpoint_meta.json and adapter weights file."""
    p = Path(ckpt_dir)
    if not p.is_dir():
        return False
    meta_path = p / "checkpoint_meta.json"
    if not meta_path.is_file():
        return False
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            return False
    except Exception:
        return False

    weights_exist = (
        (p / "adapter_model.safetensors").is_file() or
        (p / "adapter_model.bin").is_file()
    )
    return weights_exist


def compute_watch_summary(eval_records: list) -> dict:
    """Compute summary statistics (p50, p95, mean) for watch evaluation metrics."""
    if not eval_records:
        return {
            "queue_delay_p50": 0.0,
            "queue_delay_p95": 0.0,
            "queue_delay_mean": 0.0,
            "eval_latency_p50": 0.0,
            "eval_latency_p95": 0.0,
            "eval_latency_mean": 0.0,
            "evaluated_count": 0,
        }
    q_delays = sorted(r["queue_delay_s"] for r in eval_records)
    latencies = sorted(r["eval_latency_s"] for r in eval_records)

    def _perc(vals, q):
        if not vals:
            return 0.0
        from utils.serving_router import nearest_rank_percentile
        return nearest_rank_percentile(vals, q)

    return {
        "queue_delay_p50": round(_perc(q_delays, 0.50), 4),
        "queue_delay_p95": round(_perc(q_delays, 0.95), 4),
        "queue_delay_mean": round(statistics.mean(q_delays), 4),
        "eval_latency_p50": round(_perc(latencies, 0.50), 4),
        "eval_latency_p95": round(_perc(latencies, 0.95), 4),
        "eval_latency_mean": round(statistics.mean(latencies), 4),
        "evaluated_count": len(eval_records),
    }


def run_watch_mode(
    config: dict,
    config_path: str,
    run_name: Optional[str] = None,
    poll_seconds: float = 15.0,
    max_iterations: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Offline checkpoint auditing service (evaluate.py --watch).
    Polls checkpoint root and evaluates each new complete checkpoint once.
    Persists evaluation progress in eval_watch_state.json and metrics in watch_metrics.json.
    """
    from utils.checkpoint_manager import CheckpointManager

    output_dir = Path(config.get("project", {}).get("output_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    state_file = output_dir / "eval_watch_state.json"
    metrics_file = output_dir / "watch_metrics.json"

    done_checkpoints = set()
    runs_state = {}
    if state_file.exists():
        try:
            sdata = json.loads(state_file.read_text(encoding="utf-8"))
            done_checkpoints = set(sdata.get("done_checkpoints", []))
            runs_state = sdata.get("runs", {})
            _log(f"[WATCH] Resumed state from {state_file}: {len(done_checkpoints)} checkpoint(s) already done.")
        except Exception as exc:
            _log(f"[WATCH] Warning: failed reading state file {state_file}: {exc}")

    eval_records = []
    if metrics_file.exists():
        try:
            mdata = json.loads(metrics_file.read_text(encoding="utf-8"))
            eval_records = mdata.get("checkpoints", [])
        except Exception:
            pass

    stop_requested = False
    original_sigint = None

    def sigint_handler(signum, frame):
        nonlocal stop_requested
        _log("[WATCH] SIGINT received: finishing current checkpoint and stopping cleanly...")
        stop_requested = True

    try:
        original_sigint = signal.signal(signal.SIGINT, sigint_handler)
    except (ValueError, AttributeError):
        pass

    iteration = 0
    ckpt_manager = CheckpointManager(config, config_path=config_path)
    evaluator_bundle = {}
    max_steps = int(config.get("training", {}).get("max_steps", 1000))
    target_device = device or _get_device("auto")

    try:
        while not stop_requested:
            iteration += 1
            runs = ckpt_manager.list_runs()
            if run_name and run_name != "latest":
                target_run = run_name
            else:
                target_run = runs[0] if runs else None

            if target_run:
                available = ckpt_manager.list_checkpoints(run_name=target_run)
                for ckpt in available:
                    if stop_requested:
                        break

                    ckpt_path = Path(ckpt["path"])
                    ckpt_key = str(ckpt_path.resolve())
                    if ckpt_key in done_checkpoints or ckpt["name"] in done_checkpoints:
                        continue

                    if not is_checkpoint_complete(ckpt_path):
                        _log(f"[WATCH] Skipping incomplete checkpoint '{ckpt['name']}' until metadata and weights exist.")
                        continue

                    meta_file = ckpt_path / "checkpoint_meta.json"
                    mtime = meta_file.stat().st_mtime
                    eval_start = time.time()
                    queue_delay_s = max(0.0, eval_start - mtime)

                    raw_step = ckpt.get("step")
                    if raw_step == "final" or str(raw_step).lower() == "final":
                        eval_step = max_steps
                    elif isinstance(raw_step, int):
                        eval_step = raw_step
                    elif isinstance(raw_step, str) and raw_step.isdigit():
                        eval_step = int(raw_step)
                    elif isinstance(raw_step, str) and raw_step.startswith("step_") and raw_step[5:].isdigit():
                        eval_step = int(raw_step[5:])
                    else:
                        eval_step = 0

                    _log(f"[WATCH] Evaluating new complete checkpoint: {ckpt['name']} at {ckpt_path} (step {eval_step})")
                    try:
                        metrics = evaluate_checkpoint(
                            str(ckpt_path),
                            config,
                            config_path,
                            evaluator_bundle=evaluator_bundle,
                            step=eval_step,
                            device=target_device,
                        )
                        eval_end = time.time()
                        eval_latency_s = eval_end - eval_start

                        out_dir = _make_eval_output_dir(config, ckpt["name"])
                        _save_results(out_dir, metrics, config, str(ckpt_path))
                        _generate_plots([{"step": eval_step, **metrics}], out_dir, config=config)

                        rec = {
                            "checkpoint": ckpt["name"],
                            "path": ckpt_key,
                            "checkpoint_mtime": mtime,
                            "eval_start": eval_start,
                            "eval_end": eval_end,
                            "queue_delay_s": round(queue_delay_s, 4),
                            "eval_latency_s": round(eval_latency_s, 4),
                        }
                        eval_records.append(rec)
                        done_checkpoints.add(ckpt_key)
                        done_checkpoints.add(ckpt["name"])
                        runs_state.setdefault(target_run, []).append(ckpt["name"])

                        state_payload = {
                            "done_checkpoints": sorted(list(done_checkpoints)),
                            "runs": runs_state,
                            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        }
                        state_file.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")

                        summary_stats = compute_watch_summary(eval_records)
                        metrics_payload = {
                            "checkpoints": eval_records,
                            "summary": summary_stats,
                        }
                        metrics_file.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
                        _log(f"[WATCH] Completed evaluation of '{ckpt['name']}' in {eval_latency_s:.2f}s (queue delay: {queue_delay_s:.2f}s)")

                    except Exception as exc:
                        _log(f"[WATCH] ERROR during evaluation of '{ckpt['name']}': {exc}")
                        import traceback; traceback.print_exc()

            if max_iterations and iteration >= max_iterations:
                _log(f"[WATCH] Reached max_iterations={max_iterations}, exiting watch loop.")
                break

            poll_deadline = time.monotonic() + poll_seconds
            while time.monotonic() < poll_deadline and not stop_requested:
                time.sleep(0.25)

    finally:
        if original_sigint is not None:
            try:
                signal.signal(signal.SIGINT, original_sigint)
            except Exception:
                pass

    return compute_watch_summary(eval_records)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="InfiniTune — Decoupled Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the YAML config file used during training.",
    )

    # Checkpoint selection (mutually exclusive)
    ckpt_group = parser.add_mutually_exclusive_group()
    ckpt_group.add_argument(
        "--step", default=None,
        help="Evaluate the checkpoint saved at this optimizer step.",
    )
    ckpt_group.add_argument(
        "--checkpoint-dir", default=None,
        help="Path to a specific checkpoint directory (overrides --step).",
    )
    ckpt_group.add_argument(
        "--all-checkpoints", action="store_true",
        help="Sequentially evaluate ALL checkpoints for this config.",
    )
    ckpt_group.add_argument(
        "--list", action="store_true",
        help="List available checkpoints and exit without evaluating.",
    )
    ckpt_group.add_argument(
        "--watch", action="store_true",
        help="Poll the checkpoint root and evaluate each new complete checkpoint once.",
    )
    parser.add_argument(
        "--poll-seconds", type=float, default=15.0,
        help="Polling interval in seconds for --watch mode (default: 15.0).",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=None,
        help="Maximum polling iterations for --watch mode (testing only; default: unlimited).",
    )
    parser.add_argument(
        "--run", default=None,
        help="Run directory name to evaluate (default: newest run from list_runs()).",
    )
    parser.add_argument(
        "--all-runs", action="store_true",
        help="When used with --all-checkpoints, evaluate checkpoints across ALL runs rather than just the selected run.",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps", "xpu"], default="auto",
        help="Device to use for evaluation (default: auto).",
    )

    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    if not os.path.isfile(args.config):
        _log(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    _log(f"Loaded config: {args.config}")

    from utils.checkpoint_manager import (
        CheckpointManager,
        get_evaluation_schedule,
        select_evaluation_checkpoints,
    )
    ckpt_manager = CheckpointManager(config, config_path=args.config)
    runs = ckpt_manager.list_runs()
    selected_run = args.run if args.run else (runs[0] if runs else None)
    device = _get_device(args.device)

    # ── --watch ──────────────────────────────────────────────────────────────
    if args.watch:
        _log("Starting offline checkpoint auditing service (--watch)...")
        summary = run_watch_mode(
            config,
            args.config,
            run_name=args.run,
            poll_seconds=args.poll_seconds,
            max_iterations=args.max_iterations,
            device=device,
        )
        _log(f"Watch mode finished. Evaluated {summary.get('evaluated_count', 0)} checkpoints.")
        sys.exit(0)

    # ── --list ───────────────────────────────────────────────────────────────
    if args.list:
        run_filter = None if args.all_runs else (args.run if args.run else selected_run)
        checkpoints = ckpt_manager.list_checkpoints(run_name=run_filter)
        if not checkpoints:
            _log(f"No checkpoints found in: {ckpt_manager.checkpoint_root}")
            sys.exit(0)
        print(f"\nAvailable checkpoints in: {ckpt_manager.checkpoint_root}")
        print(f"{'Step':<12} {'Timestamp':<26} {'Path'}")
        print("─" * 80)
        for c in checkpoints:
            step_label = str(c["step"])
            ts         = c["timestamp"] or "(no metadata)"
            print(f"{step_label:<12} {ts:<26} {c['path']}")
        print()
        sys.exit(0)

    # ── Resolve which checkpoints to evaluate ────────────────────────────────
    if args.all_checkpoints:
        run_filter = None if args.all_runs else selected_run
        available_checkpoints = ckpt_manager.list_checkpoints(run_name=run_filter)
        if not available_checkpoints:
            _log(f"No checkpoints found in: {ckpt_manager.checkpoint_root}")
            sys.exit(1)

        checkpoints = select_evaluation_checkpoints(available_checkpoints, config)
        if not checkpoints:
            _log(
                "No checkpoints matched the configured decoupled evaluation cadence. "
                "Check evaluation.eval_interval / testing_strategy.eval_interval."
            )
            sys.exit(1)

        schedule = get_evaluation_schedule(config)
        _log(
            "Checkpoint selection for decoupled evaluation: "
            f"{len(checkpoints)} of {len(available_checkpoints)} saved checkpoints selected "
            f"(quantitative every {schedule['quantitative_interval']} step(s) if enabled, "
            f"qualitative every {schedule['qualitative_interval']} step(s) if enabled, final always included)."
        )

        checkpoints.insert(0, {
            "name": "base_model",
            "step": 0,
            "path": "base_model",
            "timestamp": None,
            "run_quantitative": schedule["quantitative_enabled"],
            "run_qualitative": schedule["qualitative_enabled"],
            "evaluation_targets": [
                target for target, enabled in (
                    ("quantitative", schedule["quantitative_enabled"]),
                    ("qualitative", schedule["qualitative_enabled"]),
                ) if enabled
            ],
        })
        _log(f"Added base_model for baseline comparison. Evaluating {len(checkpoints)} checkpoint target(s) total.")
    elif args.checkpoint_dir:
        if not os.path.isdir(args.checkpoint_dir):
            _log(f"ERROR: Checkpoint directory not found: {args.checkpoint_dir}")
            sys.exit(1)
        checkpoints = [{
            "name": os.path.basename(args.checkpoint_dir),
            "step": os.path.basename(args.checkpoint_dir),
            "path": args.checkpoint_dir,
            "timestamp": None,
        }]
    elif args.step is not None:
        target_step = int(args.step) if str(args.step).isdigit() else args.step
        ckpt_path = ckpt_manager.resolve_checkpoint_path(target_step, run_name=selected_run)
        if not ckpt_path and not args.run:
            ckpt_path = ckpt_manager.resolve_checkpoint_path(target_step, run_name=None)
        if not ckpt_path or not os.path.isdir(ckpt_path):
            _log(f"ERROR: No checkpoint found for step {args.step} at: {ckpt_path}")
            _log("Run with --list to see available checkpoints.")
            sys.exit(1)
        checkpoints = [{
            "name": os.path.basename(ckpt_path),
            "step": args.step,
            "path": ckpt_path,
            "timestamp": None,
        }]
    else:
        # Default: evaluate "final"
        ckpt_path = ckpt_manager.resolve_checkpoint_path("final", run_name=selected_run)
        if not ckpt_path and not args.run:
            ckpt_path = ckpt_manager.resolve_checkpoint_path("final", run_name=None)
        if not ckpt_path or not os.path.isdir(ckpt_path):
            _log(
                f"No 'final' checkpoint found at: {ckpt_path}\n"
                "Train with save_checkpoints.save_final: true (the default) or "
                "pass --step N or --checkpoint-dir to specify a checkpoint."
            )
            sys.exit(1)
        checkpoints = [{
            "name": "final",
            "step": "final",
            "path": ckpt_path,
            "timestamp": None,
        }]

    # ── Run evaluation for each checkpoint ───────────────────────────────────
    all_results = []  # list of {step, checkpoint, **metrics} for multi-ckpt plots
    evaluator_bundle: dict = {}
    failed = []
    evaluated_count = 0
    max_steps = int(config.get("training", {}).get("max_steps", 1000))

    for ckpt in checkpoints:
        ckpt_path = ckpt["path"]
        ckpt_name = ckpt["name"]
        raw_step = ckpt.get("step")
        if raw_step == "final" or str(raw_step).lower() == "final":
            eval_step = max_steps
        elif isinstance(raw_step, int):
            eval_step = raw_step
        elif isinstance(raw_step, str) and raw_step.isdigit():
            eval_step = int(raw_step)
        elif isinstance(raw_step, str) and raw_step.startswith("step_") and raw_step[5:].isdigit():
            eval_step = int(raw_step[5:])
        else:
            try:
                eval_step = int(raw_step)
            except (ValueError, TypeError):
                eval_step = 0
        run_quantitative = ckpt.get("run_quantitative", config.get("evaluation", {}).get("enabled", False))
        run_qualitative = ckpt.get("run_qualitative", config.get("testing_strategy", {}).get("enabled", False))
        targets = ckpt.get("evaluation_targets") or ((["quantitative"] if run_quantitative else []) + (["qualitative"] if run_qualitative else []))
        _log(f"\n{'='*60}")
        _log(f"Evaluating checkpoint: {ckpt_name}  ({ckpt_path})")
        if targets:
            _log(f"Active decoupled evaluators for this checkpoint: {', '.join(targets)}")
        _log(f"{'='*60}")

        try:
            metrics = evaluate_checkpoint(
                ckpt_path,
                config,
                args.config,
                evaluator_bundle=evaluator_bundle,
                run_quantitative=run_quantitative,
                run_qualitative=run_qualitative,
                step=eval_step,
                device=device,
            )
            evaluated_count += 1
        except Exception as exc:
            _log(f"ERROR: Evaluation failed for '{ckpt_name}': {exc}")
            import traceback; traceback.print_exc()
            failed.append(ckpt_name)
            continue

        if not metrics:
            _log(f"Warning: No metrics produced for checkpoint '{ckpt_name}'.")
            continue

        # ── Save results for this checkpoint ─────────────────────────────────
        out_dir = _make_eval_output_dir(config, ckpt_name)
        _save_results(out_dir, metrics, config, ckpt_path)
        _generate_plots([{"step": eval_step, **metrics}], out_dir, config=config)

        # Print summary
        _log("── Metrics summary ──")
        for k, v in metrics.items():
            if isinstance(v, float):
                _log(f"  {k}: {v:.4f}")
            else:
                _log(f"  {k}: {v}")

        all_results.append({"step": eval_step, "checkpoint": ckpt_name, **metrics})

    # ── If multiple checkpoints, also produce combined plots ─────────────────
    if len(all_results) > 1:
        combined_out = _make_eval_output_dir(config, "all_checkpoints")
        _generate_plots(all_results, combined_out, config=config)

        # Write a combined CSV for easy analysis
        csv_path = os.path.join(combined_out, "all_checkpoints_results.csv")
        all_keys = set()
        for r in all_results:
            all_keys.update(r.keys())
        all_keys = sorted(all_keys)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)
        _log(f"Combined results CSV: {csv_path}")
        _log(f"Combined plots: {os.path.join(combined_out, 'plots')}")

    _log(f"\nEvaluated {evaluated_count} checkpoints, {len(failed)} failed: {failed}")
    if failed:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
