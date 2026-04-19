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

def _get_device() -> torch.device:
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
                quant_metrics = evaluator.evaluate(model, step=0) or {}
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
                qual_metrics = qual_eval.run(model, step=0) or {}
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
) -> dict:
    """Load a checkpoint and run evaluation. Returns the combined metrics dict."""
    from utils.checkpoint_manager import CheckpointManager

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
    )

    # Free GPU/MPS memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


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
        "--step", type=int, default=None,
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

    # ── --list ───────────────────────────────────────────────────────────────
    if args.list:
        checkpoints = ckpt_manager.list_checkpoints()
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
        available_checkpoints = ckpt_manager.list_checkpoints()
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
        ckpt_path = ckpt_manager.get_checkpoint_path(args.step)
        if not os.path.isdir(ckpt_path):
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
        ckpt_path = ckpt_manager.get_checkpoint_path("final")
        if not os.path.isdir(ckpt_path):
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

    for ckpt in checkpoints:
        ckpt_path = ckpt["path"]
        ckpt_name = ckpt["name"]
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
            )
        except Exception as exc:
            _log(f"ERROR: Evaluation failed for '{ckpt_name}': {exc}")
            import traceback; traceback.print_exc()
            continue

        if not metrics:
            _log(f"Warning: No metrics produced for checkpoint '{ckpt_name}'.")
            continue

        # ── Save results for this checkpoint ─────────────────────────────────
        out_dir = _make_eval_output_dir(config, ckpt_name)
        _save_results(out_dir, metrics, config, ckpt_path)
        _generate_plots([{"step": ckpt["step"], **metrics}], out_dir, config=config)

        # Print summary
        _log("── Metrics summary ──")
        for k, v in metrics.items():
            if isinstance(v, float):
                _log(f"  {k}: {v:.4f}")
            else:
                _log(f"  {k}: {v}")

        all_results.append({"step": ckpt["step"], "checkpoint": ckpt_name, **metrics})

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

    _log("\nDecoupled evaluation complete.")


if __name__ == "__main__":
    main()
