"""
Standalone utility to generate training metric plots from a metrics CSV file.

Usage:
    python utils/plot_metrics.py <path_to_metrics.csv>
    python utils/plot_metrics.py <path_to_metrics.csv> --out-dir ./my_plots

If --out-dir is not specified, plots are saved alongside the CSV file.

Requires: matplotlib (pip install matplotlib)
"""

import argparse
import csv
import os
import sys


def read_metrics_csv(csv_path):
    """Read a metrics CSV file and return a list of dicts (one per row)."""
    if not os.path.isfile(csv_path):
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        print(f"Error: CSV is empty: {csv_path}")
        sys.exit(1)

    return rows


def extract_series(rows, key):
    """Extract (steps, values) for a given column, skipping blanks."""
    xs, ys = [], []
    for r in rows:
        val = r.get(key, "")
        if val in ("", None):
            continue
        try:
            xs.append(int(r.get("step", 0)))
            ys.append(float(val))
        except (ValueError, TypeError):
            continue
    return xs, ys


def generate_plots(csv_path, out_dir=None):
    """Read a metrics CSV and generate PNG plots for each metric."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required. Install it with: pip install matplotlib")
        sys.exit(1)

    rows = read_metrics_csv(csv_path)

    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(out_dir, exist_ok=True)

    plot_defs = [
        # ── Quantitative metrics ──────────────────────────────────────────────
        ("train_loss", "Training Loss", "loss"),
        ("eval_loss", "Eval Loss", "eval_loss"),
        ("perplexity", "Perplexity", "perplexity"),
        ("accuracy", "Accuracy", "accuracy"),
        ("aauc", "AAUC (normalized)", "aauc"),
        ("backward_transfer", "Backward Transfer", "backward_transfer"),
        ("f1", "Macro F1 Score", "f1"),
        ("mcc", "Matthews Correlation Coefficient", "mcc"),
        ("kappa", "Cohen's Kappa", "kappa"),
        ("exact_match", "Exact Match Rate", "exact_match"),
        ("qafacteval", "QAFactEval Score", "qafacteval"),
        ("forgetting_max", "Max Forgetting (tracked metrics)", "forgetting_max"),
        ("update_latency_s", "Update Latency (s since last eval)", "update_latency_s"),
        ("grad_norm", "Gradient Norm", "grad_norm"),
        ("tokens_per_sec", "Token Throughput (tok/s)", "tokens_per_sec"),
        # ── Qualitative metrics (populated when testing_strategy is enabled) ──
        ("qual_semantic_similarity", "Semantic Similarity (MiniLM)", "qual_semantic_similarity"),
        ("qual_keyword_density", "Domain Keyword Density", "qual_keyword_density"),
        ("qual_type_token_ratio", "Type-Token Ratio (Lexical Diversity)", "qual_type_token_ratio"),
        ("qual_hapax_ratio", "Hapax Ratio (Word Uniqueness)", "qual_hapax_ratio"),
        ("qual_cot_anchor_count", "CoT Logic Anchor Count (mean)", "qual_cot_anchor_count_mean"),
        ("qual_cot_step_length", "CoT Step Length — chars between anchors (mean)", "qual_cot_step_length_mean"),
        ("qual_cot_coverage", "CoT Coverage Rate — responses with ≥1 anchor", "qual_cot_coverage_rate"),
        ("qual_mean_response_length", "Mean Response Length (words)", "qual_mean_response_length"),
        ("qual_repetition_rate", "Bigram Repetition Rate", "qual_repetition_rate"),
        ("qual_non_empty_rate", "Non-Empty Response Rate", "qual_non_empty_rate"),
        # ── E2E NLG metrics (populated when strategy=perplexity + e2e config) ──
        ("qual_slot_coverage_mean", "Slot Coverage (mean)", "qual_slot_coverage_mean"),
        ("qual_consistency_score_mean", "Consistency Score (mean)", "qual_consistency_score_mean"),
        ("answer_overlap_f1", "Answer Overlap F1 (token-level)", "answer_overlap_f1"),
    ]

    generated = 0
    for filename, title, key in plot_defs:
        xs, ys = extract_series(rows, key)
        if not xs:
            continue

        plt.figure(figsize=(10, 6))
        plt.plot(xs, ys, marker="o", markersize=3, linewidth=1.2)
        plt.title(title, fontsize=14)
        plt.xlabel("Step", fontsize=12)
        plt.ylabel(key, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"{filename}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")
        generated += 1

    if generated == 0:
        print("No plottable data found in the CSV.")
    else:
        print(f"\nGenerated {generated} plot(s) in: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate training metric plots from a metrics CSV file."
    )
    parser.add_argument(
        "csv_path",
        help="Path to the metrics.csv file produced by the trainer",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to save plots (default: same directory as the CSV)",
    )
    args = parser.parse_args()

    print(f"Reading: {args.csv_path}")
    generate_plots(args.csv_path, args.out_dir)


if __name__ == "__main__":
    main()
