"""
benchmarks/eval_speed.py
─────────────────────────────────────────────────────────────────────────────
Qualitative Evaluation Speed Benchmark (WP-17 / Claim SDE-4).

Compares sequential vs batched response generation throughput on E2E NLG
meaning representations (MRs) with consistency sampling runs.

Measures:
  - sequential_s (wall-clock time)
  - batched_s (wall-clock time)
  - reduction_pct (1 - batched/sequential)
  - speedup_ratio (sequential/batched)
  - identical_fraction (greedy equivalence on fp32 base model)
"""

import os
import sys
import time
import json
import statistics
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import yaml
from jinja2 import Template

from benchmarks.repro_eval import (
    MODEL_REVISIONS,
    file_sha256,
    load_model,
    device_name,
)
from benchmarks.e2e_eval import prepare_e2e, read_rows
from utils.eval_qualitative import QualitativeEvaluator, _compute_universal_qualitative_metrics

ROOT = Path(__file__).resolve().parent.parent


def sync_device():
    """Synchronize compute device before reading high-precision timers."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def evaluate_greedy_equivalence(
    evaluator: QualitativeEvaluator,
    model,
    test_window: List[Dict[str, Any]],
    n_check: int = 50,
) -> float:
    """
    Equivalence check: with do_sample=False and num_return_sequences=1 on fp32,
    batched and sequential outputs should be identical strings for >=98% of MRs.
    """
    check_window = test_window[:min(len(test_window), n_check)]
    if not check_window:
        return 1.0

    seq_preds, _ = evaluator._generate_responses_sequential(
        model, check_window, do_sample=False, num_return_sequences=1
    )
    batch_preds, _ = evaluator._generate_responses(
        model, check_window, do_sample=False, num_return_sequences=1
    )

    identical_count = sum(
        1 for s, b in zip(seq_preds, batch_preds)
        if s.strip() == b.strip()
    )
    return identical_count / len(check_window)


def run_speed_trials(
    evaluator: QualitativeEvaluator,
    model,
    test_window: List[Dict[str, Any]],
    consistency_runs: int,
    temperature: float = 0.7,
    repeats: int = 3,
) -> List[Dict[str, Any]]:
    """
    Measures sequential vs batched generation wall time across multiple repeats.
    """
    trials = []

    # Warm-up (1 batch of 2 samples, not counted)
    warmup_window = test_window[:min(2, len(test_window))]
    if warmup_window:
        evaluator._generate_responses(
            model, warmup_window, do_sample=True, temperature=temperature, num_return_sequences=2
        )
        evaluator._generate_responses_sequential(
            model, warmup_window, do_sample=True, temperature=temperature, num_return_sequences=2
        )

    for rep in range(1, repeats + 1):
        # 1. Sequential Mode
        sync_device()
        t0 = time.perf_counter()
        evaluator._generate_responses_sequential(
            model, test_window, do_sample=True, temperature=temperature, num_return_sequences=consistency_runs
        )
        sync_device()
        seq_elapsed = time.perf_counter() - t0

        # 2. Batched Mode
        sync_device()
        t0 = time.perf_counter()
        evaluator._generate_responses(
            model, test_window, do_sample=True, temperature=temperature, num_return_sequences=consistency_runs
        )
        sync_device()
        batched_elapsed = time.perf_counter() - t0

        reduction = 1.0 - (batched_elapsed / seq_elapsed) if seq_elapsed > 0 else 0.0
        speedup = seq_elapsed / batched_elapsed if batched_elapsed > 0 else 1.0

        trials.append({
            "repeat": rep,
            "sequential_s": round(seq_elapsed, 4),
            "batched_s": round(batched_elapsed, 4),
            "reduction_pct": round(reduction * 100.0, 2),
            "speedup_ratio": round(speedup, 2),
            "total_generations": len(test_window) * consistency_runs,
        })

    return trials


def summarize_eval_speed_results(
    trials: List[Dict[str, Any]],
    identical_fraction: float,
    n_mrs: int,
    consistency_runs: int,
) -> Dict[str, Any]:
    """Summarize speed comparison metrics with quantiles and boundary."""
    seq_times = [t["sequential_s"] for t in trials]
    batch_times = [t["batched_s"] for t in trials]
    reductions = [t["reduction_pct"] for t in trials]
    speedups = [t["speedup_ratio"] for t in trials]

    return {
        "suite": "eval_speed",
        "n_mrs": n_mrs,
        "consistency_runs": consistency_runs,
        "total_generations_per_run": n_mrs * consistency_runs,
        "repeats": len(trials),
        "sequential_s": {
            "mean": round(statistics.mean(seq_times), 4),
            "min": round(min(seq_times), 4),
            "max": round(max(seq_times), 4),
        },
        "batched_s": {
            "mean": round(statistics.mean(batch_times), 4),
            "min": round(min(batch_times), 4),
            "max": round(max(batch_times), 4),
        },
        "reduction_pct_mean": round(statistics.mean(reductions), 2),
        "speedup_ratio_mean": round(statistics.mean(speedups), 2),
        "greedy_identical_fraction": round(identical_fraction, 4),
        "boundary": "Sequential (sample-by-sample) vs batched generation throughput on E2E NLG meaning representations",
        "trials": trials,
    }


def run_eval_speed_suite(run, args, data_dir) -> Dict[str, Any]:
    """Execute the full eval_speed benchmark suite."""
    p_name = args.profile
    # Profiles: smoke=8 MRs x 2 runs = 16; standard=334 MRs x 10 runs = 3340; full=334 MRs x 10 runs = 3340
    n_mrs = {"smoke": 8, "standard": 334, "full": 334}[p_name]
    consistency_runs = {"smoke": 2, "standard": 10, "full": 10}[p_name]
    repeats = 1 if p_name == "smoke" else 3

    # Load E2E validation data
    cache_path, manifest = prepare_e2e(n_mrs, args.seed, data_dir, args.offline)
    rows = read_rows(cache_path)
    eval_window = [{"input": r["mr"], "reference": r["references"][0] if r["references"] else ""} for r in rows]

    # Load gpt2-medium pinned model
    model_name = args.model if args.model in MODEL_REVISIONS else "gpt2-medium"
    model, tokenizer, device = load_model(model_name)

    # Synthetic config for QualitativeEvaluator
    cfg = {
        "model": {"name": model_name, "max_seq_length": args.max_seq_length or 128},
        "testing_strategy": {
            "enabled": True,
            "method": "structured_slot_coverage",
            "eval_interval": 1,
            "eval_samples": n_mrs,
            "max_new_tokens": 50,
            "consistency_runs": consistency_runs,
            "consistency_temperature": 0.7,
            "eval_batch_size": 16,
            "generation_mode": "batched",
        },
        "preprocessing": {
            "prompt_template": "MR: {{ input }}\nDescription:",
        },
    }

    evaluator = QualitativeEvaluator(cfg, tokenizer, device)
    evaluator.eval_data = eval_window

    # 1. Greedy Equivalence Check
    identical_frac = evaluate_greedy_equivalence(evaluator, model, eval_window, n_check=50)

    # 2. Timing trials
    trials = run_speed_trials(
        evaluator, model, eval_window, consistency_runs=consistency_runs,
        temperature=0.7, repeats=repeats,
    )

    summary = summarize_eval_speed_results(trials, identical_frac, n_mrs, consistency_runs)

    out_dir = run.directory / "eval_speed"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as f:
        for t in trials:
            f.write(json.dumps(t) + "\n")

    summary["raw_path"] = str(raw_path)
    summary["raw_sha256"] = file_sha256(raw_path)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
