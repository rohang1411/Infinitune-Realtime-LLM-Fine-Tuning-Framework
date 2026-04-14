"""
utils/plot_metrics.py
─────────────────────────────────────────────────────────────────────────────
Metric plotting utility for InfiniTune.

Generates individual PNG plots for each metric column in a metrics CSV, PLUS
a unified dashboard (dashboard.png) grouping all metrics into labelled panels
with a config header strip showing the key hyperparameters used for this run.

Usage (standalone):
    python utils/plot_metrics.py <path_to_metrics.csv>
    python utils/plot_metrics.py <path_to_metrics.csv> --out-dir ./my_plots
    python utils/plot_metrics.py <path_to_metrics.csv> --config configs/e2e_qualitative.yaml

If --out-dir is not specified, plots are saved alongside the CSV file.

Requires: matplotlib (pip install matplotlib)
"""

import argparse
import csv
import os
import re
import sys


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_metrics_csv(csv_path: str) -> list:
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


def extract_series(rows: list, key: str) -> tuple:
    """Extract (steps, values) for a given column, skipping blanks."""
    xs, ys = [], []
    for r in rows:
        val = r.get(key, "")
        if val in ("", None):
            continue
        try:
            # step column may be "final" — map to max_step+1 for plotting
            step_raw = r.get("step", "0")
            if step_raw == "final":
                continue  # handled separately in dashboard
            xs.append(int(step_raw))
            ys.append(float(val))
        except (ValueError, TypeError):
            continue
    return xs, ys


def get_per_slot_columns(rows: list) -> list:
    """
    Auto-detect per-slot coverage columns from CSV headers.
    Returns list of (column_key, slot_label) for any column matching
    qual_slot_<name>_coverage.
    """
    if not rows:
        return []
    per_slot = []
    for key in rows[0].keys():
        m = re.match(r"qual_slot_(.+)_coverage$", key)
        if m:
            slot_label = m.group(1).replace("_", " ")
            per_slot.append((key, slot_label))
    return per_slot


# ─────────────────────────────────────────────────────────────────────────────
# Config header helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_config_header(config: dict) -> str:
    """
    Build a compact one-to-two-line config summary string for the dashboard.
    Reads model, lora, training, and testing_strategy sections.
    """
    if not config:
        return "InfiniTune — config not provided"

    project = config.get("project", {})
    model = config.get("model", {})
    lora = config.get("lora", {})
    training = config.get("training", {})
    ts = config.get("testing_strategy", {})

    run_name = project.get("name", "")
    model_name = model.get("name", "")
    precision = model.get("precision", "")

    lr = training.get("learning_rate", "")
    max_steps = training.get("max_steps", "")
    batch = training.get("batch_size", "")
    grad_acc = training.get("gradient_accumulation_steps", 1)
    eff_batch = (int(batch) * int(grad_acc)) if batch and grad_acc else ""

    lora_r = lora.get("r", "")
    lora_alpha = lora.get("alpha", "")
    target_mods = lora.get("target_modules", [])
    target_str = ", ".join(target_mods) if target_mods else ""

    method = ts.get("method", "")
    eval_samples = ts.get("eval_samples", "")
    consistency_runs = ts.get("consistency_runs", "")

    line1 = f"Run: {run_name}  |  Model: {model_name} ({precision})  |  LR: {lr}  |  Steps: {max_steps}  |  Eff. Batch: {eff_batch}"
    line2 = f"LoRA r={lora_r} α={lora_alpha}  |  Targets: {target_str}  |  Qual. method: {method}  |  Eval samples: {eval_samples}  |  Consistency runs: {consistency_runs}"
    return f"{line1}\n{line2}"


# ─────────────────────────────────────────────────────────────────────────────
# Individual plot generator (backward-compatible)
# ─────────────────────────────────────────────────────────────────────────────

# Static plot definitions: (filename, title, csv_key)
_STATIC_PLOT_DEFS = [
    # ── Quantitative metrics ──────────────────────────────────────────────────
    ("train_loss",           "Training Loss",                          "train_loss"),
    ("eval_loss",            "Eval Loss",                              "eval_loss"),
    ("perplexity",           "Perplexity",                             "perplexity"),
    ("accuracy",             "Accuracy",                               "accuracy"),
    ("aauc",                 "AAUC (normalized)",                      "aauc"),
    ("backward_transfer",    "Backward Transfer",                      "backward_transfer"),
    ("f1",                   "Macro F1 Score",                         "f1"),
    ("mcc",                  "Matthews Correlation Coefficient",       "mcc"),
    ("kappa",                "Cohen's Kappa",                          "kappa"),
    ("exact_match",          "Exact Match Rate",                       "exact_match"),
    ("qafacteval",           "QAFactEval Score",                       "qafacteval"),
    ("forgetting_max",       "Max Forgetting (tracked metrics)",       "forgetting_max"),
    ("update_latency_s",     "Update Latency (s since last eval)",     "update_latency_s"),
    ("grad_norm",            "Gradient Norm",                          "grad_norm"),
    ("tokens_per_sec",       "Token Throughput (tok/s)",               "tokens_per_sec"),
    ("answer_overlap_f1",    "Answer Overlap F1 (token-level)",        "answer_overlap_f1"),
    # ── Qualitative — universal ───────────────────────────────────────────────
    ("qual_semantic_similarity",   "Semantic Similarity (MiniLM)",           "qual_semantic_similarity"),
    ("qual_keyword_density",       "Domain Keyword Density",                  "qual_keyword_density"),
    ("qual_type_token_ratio",      "Type-Token Ratio (Lexical Diversity)",    "qual_type_token_ratio"),
    ("qual_hapax_ratio",           "Hapax Ratio (Word Uniqueness)",           "qual_hapax_ratio"),
    ("qual_cot_anchor_count",      "CoT Logic Anchor Count (mean)",           "qual_cot_anchor_count_mean"),
    ("qual_cot_step_length",       "CoT Step Length — chars between anchors", "qual_cot_step_length_mean"),
    ("qual_cot_coverage",          "CoT Coverage Rate — responses with ≥1 anchor", "qual_cot_coverage_rate"),
    ("qual_mean_response_length",  "Mean Response Length (words)",            "qual_mean_response_length"),
    ("qual_repetition_rate",       "Bigram Repetition Rate",                  "qual_repetition_rate"),
    ("qual_non_empty_rate",        "Non-Empty Response Rate",                 "qual_non_empty_rate"),
    # ── Qualitative — structured slot coverage ────────────────────────────────
    ("qual_slot_coverage_mean",        "Slot Coverage (mean)",                      "qual_slot_coverage_mean"),
    ("qual_consistency_score_mean",    "Consistency Score (mean)",                  "qual_consistency_score_mean"),
    ("qual_perfect_coverage_rate",     "Perfect Coverage Rate",                     "qual_perfect_coverage_rate"),
    ("qual_familyFriendly_inversion",  "familyFriendly Inversion Rate",             "qual_slot_familyFriendly_inversion_rate"),
    ("qual_pinned_coverage",           "Pinned Anchor Slot Coverage (mean)",        "qual_pinned_slot_coverage_mean"),
    ("qual_pinned_perfect",            "Pinned Anchor Perfect Coverage Rate",       "qual_pinned_perfect_coverage_rate"),
    ("qual_pinned_consistency",        "Pinned Anchor Consistency Score",           "qual_pinned_consistency_score"),
]


def _plot_single(ax, xs, ys, title, ylabel, color="steelblue"):
    """Plot a single metric onto a matplotlib Axes object."""
    ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.5, color=color)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("Step", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25, linestyle="--")
    if ys:
        ax.set_ylim(bottom=max(0, min(ys) * 0.9), top=max(ys) * 1.1 + 1e-9)


def generate_individual_plots(rows: list, out_dir: str, extra_cols: list = None) -> int:
    """
    Generate one PNG per metric column.

    Args:
        rows      : CSV rows (list of dicts).
        out_dir   : Output directory.
        extra_cols: Additional (filename, title, key) tuples to plot (e.g. per-slot columns).

    Returns number of plots generated.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required. Install it with: pip install matplotlib")
        sys.exit(1)

    all_defs = list(_STATIC_PLOT_DEFS)
    if extra_cols:
        all_defs.extend(extra_cols)

    generated = 0
    for filename, title, key in all_defs:
        xs, ys = extract_series(rows, key)
        if not xs:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        _plot_single(ax, xs, ys, title, key)
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"{filename}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")
        generated += 1

    return generated


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard generator
# ─────────────────────────────────────────────────────────────────────────────

# Color palette for dashboard panels
_PALETTE = {
    "quant":     "#4C72B0",   # muted blue — training health
    "coverage":  "#55A868",   # green — slot coverage summary
    "per_slot":  "#C44E52",   # red — per-slot breakdown
    "consist":   "#8172B3",   # purple — consistency / inversion
    "universal": "#CCB974",   # gold — universal quality
    "pinned":    "#64B5CD",   # teal — pinned anchors
}


def generate_dashboard(rows: list, out_dir: str, config: dict = None) -> str:
    """
    Build a unified multi-panel dashboard PNG.

    Layout (panels adapt based on which columns are actually present in the CSV):
      Row 0: Config header strip
      Row 1: Training Health — eval_loss + perplexity
      Row 2: Slot Coverage Summary — slot_coverage_mean + perfect_coverage_rate
      Row 3: Per-Slot Breakdown — one subplot per qual_slot_<name>_coverage column
      Row 4: Consistency & Inversion — consistency_score_mean + familyFriendly_inversion_rate
      Row 5: Universal Quality — mean_response_length + repetition_rate
      Row 6: Pinned Anchors — pinned_slot_coverage_mean + pinned_perfect_coverage_rate + pinned_consistency

    Panels with no data are omitted automatically.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        print("Error: matplotlib is required. Install it with: pip install matplotlib")
        return ""

    per_slot_cols = get_per_slot_columns(rows)

    # ── Determine which panel rows to include ────────────────────────────────
    panels = []  # list of (panel_id, n_subplots_in_row)

    def _has(key):
        xs, _ = extract_series(rows, key)
        return bool(xs)

    # Panel 1: Training Health
    health_cols = [
        ("eval_loss", "Eval Loss", "eval_loss", _PALETTE["quant"]),
        ("perplexity", "Perplexity", "perplexity", _PALETTE["quant"]),
    ]
    health_present = [(t, lbl, k, c) for t, lbl, k, c in health_cols if _has(k)]
    if health_present:
        panels.append(("health", health_present))

    # Panel 2: Slot Coverage Summary
    cov_cols = [
        ("qual_slot_coverage_mean",    "Slot Coverage (mean)",          _PALETTE["coverage"]),
        ("qual_perfect_coverage_rate", "Perfect Coverage Rate",          _PALETTE["coverage"]),
    ]
    cov_present = [(lbl, k, c) for lbl, k, c in cov_cols if _has(k)]
    if cov_present:
        panels.append(("coverage_summary", cov_present))

    # Panel 3: Per-slot breakdown
    per_slot_present = [(k, lbl) for k, lbl in per_slot_cols if _has(k)]
    if per_slot_present:
        panels.append(("per_slot", per_slot_present))

    # Panel 4: Consistency & Inversion
    consist_cols = [
        ("qual_consistency_score_mean",           "Consistency Score (mean)",        _PALETTE["consist"]),
        ("qual_slot_familyFriendly_inversion_rate", "familyFriendly Inversion Rate", _PALETTE["consist"]),
    ]
    consist_present = [(lbl, k, c) for lbl, k, c in consist_cols if _has(k)]
    if consist_present:
        panels.append(("consistency", consist_present))

    # Panel 5: Universal Quality
    univ_cols = [
        ("qual_mean_response_length", "Mean Response Length (words)", _PALETTE["universal"]),
        ("qual_repetition_rate",      "Bigram Repetition Rate",       _PALETTE["universal"]),
    ]
    univ_present = [(lbl, k, c) for lbl, k, c in univ_cols if _has(k)]
    if univ_present:
        panels.append(("universal", univ_present))

    # Panel 6: Pinned Anchors
    pinned_cols = [
        ("qual_pinned_slot_coverage_mean",    "Pinned Coverage (mean)",        _PALETTE["pinned"]),
        ("qual_pinned_perfect_coverage_rate", "Pinned Perfect Coverage Rate",   _PALETTE["pinned"]),
        ("qual_pinned_consistency_score",     "Pinned Consistency Score",       _PALETTE["pinned"]),
    ]
    pinned_present = [(lbl, k, c) for lbl, k, c in pinned_cols if _has(k)]
    if pinned_present:
        panels.append(("pinned", pinned_present))

    if not panels:
        print("  No plottable data found for dashboard.")
        return ""

    # ── Figure layout ─────────────────────────────────────────────────────────
    # Compute rows needed: 1 header row + 1 row per panel
    # Max subplots per panel row = max cols present in any panel
    MAX_COLS = 4

    n_data_rows = len(panels)
    fig_height = 2.0 + n_data_rows * 3.2  # header + data rows
    fig_width  = 16

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="#1a1a2e")

    # Outer gridspec: header row (1) + data rows (n_data_rows)
    outer_gs = gridspec.GridSpec(
        n_data_rows + 1, 1,
        figure=fig,
        height_ratios=[0.55] + [1.0] * n_data_rows,
        hspace=0.55,
    )

    # ── Header strip ──────────────────────────────────────────────────────────
    header_ax = fig.add_subplot(outer_gs[0])
    header_ax.set_facecolor("#16213e")
    header_ax.axis("off")

    header_text = _build_config_header(config) if config else "InfiniTune — Evaluation Dashboard"
    header_ax.text(
        0.5, 0.60,
        "InfiniTune — Evaluation Dashboard",
        transform=header_ax.transAxes,
        ha="center", va="center",
        fontsize=14, fontweight="bold", color="#e0e0e0",
    )
    header_ax.text(
        0.5, 0.18,
        header_text,
        transform=header_ax.transAxes,
        ha="center", va="center",
        fontsize=7.5, color="#a0a0c0",
        family="monospace",
    )

    # ── Data panels ───────────────────────────────────────────────────────────
    for panel_row_idx, (panel_id, panel_cols) in enumerate(panels):
        n_subplots = min(len(panel_cols), MAX_COLS)
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            1, n_subplots,
            subplot_spec=outer_gs[panel_row_idx + 1],
            wspace=0.35,
        )

        for col_idx in range(n_subplots):
            ax = fig.add_subplot(inner_gs[col_idx])
            ax.set_facecolor("#0f3460")

            # Unpack column info based on panel type
            col_info = panel_cols[col_idx]
            if panel_id == "per_slot":
                key, label = col_info
                color = _PALETTE["per_slot"]
                title = f"Slot: {label}"
                ylabel = "Coverage Rate"
            else:
                label, key, color = col_info
                title = label
                ylabel = key

            xs, ys = extract_series(rows, key)
            if xs:
                ax.plot(xs, ys, marker="o", markersize=3.5, linewidth=1.5, color=color)
                ax.fill_between(xs, ys, alpha=0.12, color=color)

            ax.set_title(title, fontsize=9, fontweight="bold", color="#e0e0e0", pad=5)
            ax.set_xlabel("Step", fontsize=7.5, color="#9090b0")
            ax.set_ylabel(ylabel, fontsize=7.5, color="#9090b0")
            ax.tick_params(colors="#8080a0", labelsize=7)
            ax.spines["bottom"].set_color("#334466")
            ax.spines["left"].set_color("#334466")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.18, linestyle="--", color="#4466aa")

    out_path = os.path.join(out_dir, "dashboard.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Dashboard saved: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_plots(csv_path: str, out_dir: str = None, config: dict = None) -> None:
    """
    Generate individual metric plots AND the unified dashboard.

    Args:
        csv_path : Path to the metrics CSV produced by the trainer / evaluator.
        out_dir  : Output directory. Defaults to the same directory as the CSV.
        config   : Optional config dict for dashboard header. When passed,
                   the header displays model, LR, LoRA, and eval settings.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("Error: matplotlib is required. Install it with: pip install matplotlib")
        sys.exit(1)

    rows = read_metrics_csv(csv_path)

    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(out_dir, exist_ok=True)

    # Auto-detect per-slot columns for individual plots
    per_slot_cols = get_per_slot_columns(rows)
    extra_plot_defs = []
    for key, slot_label in per_slot_cols:
        safe_filename = f"qual_slot_{re.sub(r'[^a-zA-Z0-9]', '_', slot_label)}_coverage"
        extra_plot_defs.append((safe_filename, f"Slot Coverage: {slot_label}", key))

    # Generate individual plots
    generated = generate_individual_plots(rows, out_dir, extra_cols=extra_plot_defs)

    if generated == 0:
        print("No plottable data found in the CSV.")
    else:
        print(f"\nGenerated {generated} individual plot(s) in: {out_dir}")

    # Generate dashboard
    generate_dashboard(rows, out_dir, config=config)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate training metric plots and dashboard from a metrics CSV file."
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
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the YAML config file — used to populate the dashboard header",
    )
    args = parser.parse_args()

    config = None
    if args.config:
        try:
            import yaml
            with open(args.config, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            print(f"Config loaded from: {args.config}")
        except Exception as exc:
            print(f"Warning: could not load config '{args.config}': {exc}")

    print(f"Reading: {args.csv_path}")
    generate_plots(args.csv_path, args.out_dir, config=config)


if __name__ == "__main__":
    main()
