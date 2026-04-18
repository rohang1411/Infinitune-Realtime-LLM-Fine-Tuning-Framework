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
import json
import math
import os
import re
import sys
import textwrap
from typing import Any, Dict, Optional

# When executed as `python utils/plot_metrics.py`, Python adds `utils/` (the
# script dir) to `sys.path`, which prevents resolving the top-level `utils`
# package. Fix by ensuring the project root is on `sys.path`.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.report_utils import (
    DARK,
    LIGHT,
    build_presentation_spec,
    detect_usecase,
    generate_insights,
    presentation_to_dict,
    select_kpis,
)

# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_metrics_csv(csv_path: str) -> list:
    """Read a metrics CSV file and return a list of dicts (one per row)."""
    if not os.path.isfile(csv_path):
        print(f"Warning: File not found (plotting skipped): {csv_path}")
        return []

    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        print(f"Warning: CSV is empty (plotting skipped): {csv_path}")
        return []

    return rows


def _extract_series_raw(rows: list, key: str) -> tuple:
    """Extract (steps, values) for a given column, skipping blanks."""
    xs, ys = [], []
    for r in rows:
        val = r.get(key, "")
        if val in ("", None):
            continue
        try:
            step_raw = r.get("step", "0")
            if step_raw == "final":
                continue  # handled separately in dashboard
            xs.append(int(step_raw))
            ys.append(float(val))
        except (ValueError, TypeError):
            continue
    return xs, ys


def _derive_continual_learning_stability(rows: list) -> tuple:
    """
    Derive stability as baseline_accuracy - current_accuracy.
    Negative values therefore indicate improvement over the initial eval.
    """
    xs, ys = _extract_series_raw(rows, "accuracy")
    if not ys:
        return [], []
    baseline = ys[0]
    return xs, [baseline - y for y in ys]


def _get_plotting_cfg(config: dict) -> dict:
    evaluation_cfg = (config or {}).get("evaluation", {}) if isinstance(config, dict) else {}
    plotting_cfg = evaluation_cfg.get("plotting") or {}

    try:
        window = int(plotting_cfg.get("rolling_average_window", 1))
    except (TypeError, ValueError):
        window = 1
    window = max(1, window)

    include = plotting_cfg.get("rolling_average_include")
    include_keys = None
    if isinstance(include, (list, tuple)):
        include_keys = {
            str(key).strip() for key in include if str(key).strip()
        } or None

    enabled = bool(plotting_cfg.get("rolling_average_enabled", False) or window > 1)
    return {
        "enabled": enabled and window > 1,
        "window": window,
        "include": include_keys,
    }


def _apply_centered_rolling_average(values: list, window: int) -> list:
    if window <= 1 or len(values) < 2:
        return list(values)

    half = window // 2
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - half)
        end = min(len(values), idx + half + 1)
        chunk = values[start:end]
        smoothed.append(sum(chunk) / max(len(chunk), 1))
    return smoothed


def _should_smooth_metric(key: str, config: dict) -> bool:
    plotting_cfg = _get_plotting_cfg(config)
    if not plotting_cfg["enabled"]:
        return False
    include = plotting_cfg["include"]
    if include is None:
        return True
    return key in include


def _rolling_suffix(keys, config: dict) -> str:
    keys = [keys] if isinstance(keys, str) else list(keys or [])
    plotting_cfg = _get_plotting_cfg(config)
    if not plotting_cfg["enabled"]:
        return ""
    if not any(_should_smooth_metric(key, config) for key in keys):
        return ""
    return f" ({plotting_cfg['window']}-pt rolling avg)"


def extract_series(rows: list, key: str, config: dict = None) -> tuple:
    """Extract (steps, values) for a metric key, including derived series."""
    if key == "continual_learning_stability":
        xs, ys = _derive_continual_learning_stability(rows)
    else:
        xs, ys = _extract_series_raw(rows, key)

    if xs and _should_smooth_metric(key, config):
        ys = _apply_centered_rolling_average(ys, _get_plotting_cfg(config)["window"])
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
    evaluation = config.get("evaluation", {})
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
    eval_pool = evaluation.get("eval_pool_size", "")
    eval_batch = evaluation.get("eval_batch_size", "")
    smoothing_suffix = _rolling_suffix(
        ["accuracy", "mcc", "f1", "kappa", "exact_match", "continual_learning_stability"],
        config,
    )

    line1 = f"Run: {run_name}  |  Model: {model_name} ({precision})  |  LR: {lr}  |  Steps: {max_steps}  |  Eff. Batch: {eff_batch}"
    line2 = (
        f"LoRA r={lora_r} alpha={lora_alpha}  |  Targets: {target_str}  |  "
        f"Eval pool/batch: {eval_pool}/{eval_batch}  |  Qual. method: {method}  |  "
        f"Eval samples: {eval_samples}  |  Consistency runs: {consistency_runs}{smoothing_suffix}"
    )
    return f"{line1}\n{line2}"


def _build_wrapped_config_header(config: dict) -> str:
    raw_header = _build_config_header(config)
    return "\n".join(textwrap.fill(line, width=118) for line in str(raw_header).splitlines() if line.strip())


# ─────────────────────────────────────────────────────────────────────────────
# Individual plot generator (backward-compatible)
# ─────────────────────────────────────────────────────────────────────────────

# Static plot definitions: (filename, title, csv_key)
_STATIC_PLOT_DEFS = [
    # ── Quantitative metrics ──────────────────────────────────────────────────
    ("train_loss",           "Training Loss",                          "loss"),
    ("learning_rate",        "Learning Rate",                          "lr"),
    ("eval_loss",            "Eval Loss",                              "eval_loss"),
    ("perplexity",           "Perplexity",                             "perplexity"),
    ("accuracy",             "Accuracy",                               "accuracy"),
    ("average_accuracy",     "Average Accuracy (over training)",       "average_accuracy"),
    (
        "continual_learning_stability",
        "Continual Learning Stability (negative = improvement vs initial eval)",
        "continual_learning_stability",
    ),
    ("f1",                   "Macro F1 Score (normalized labels)",     "f1"),
    ("mcc",                  "Matthews Correlation Coefficient",       "mcc"),
    ("kappa",                "Cohen's Kappa",                          "kappa"),
    ("exact_match",          "Exact Match Rate",                       "exact_match"),
    ("qafacteval",           "QAFactEval Score",                       "qafacteval"),
    ("forgetting_max",       "Max Forgetting (tracked metrics)",       "forgetting_max"),
    ("eval_cycle_time_s",    "Eval Cycle Time (s)",                    "eval_cycle_time_s"),
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


def _plot_single(ax, xs, ys, title, ylabel, theme: dict, color: str):
    """Plot a single metric onto a matplotlib Axes object."""
    ax.set_facecolor(theme.get("panel_bg", "#ffffff"))
    ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.5, color=color)
    ax.fill_between(xs, ys, alpha=0.08, color=color)

    ax.set_title(title, fontsize=10, fontweight="bold", pad=6, color=theme.get("text", "#111111"))
    ax.set_xlabel("Step", fontsize=8, color=theme.get("subtext", "#666666"))
    ax.set_ylabel(ylabel, fontsize=8, color=theme.get("subtext", "#666666"))
    ax.tick_params(labelsize=7, colors=theme.get("subtext", "#666666"))
    ax.grid(True, alpha=0.18, linestyle="--", color=theme.get("grid", "#cccccc"))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(theme.get("border", "#dddddd"))
    ax.spines["bottom"].set_color(theme.get("border", "#dddddd"))

    if ys:
        ymin = min(ys)
        ymax = max(ys)
        pad = (ymax - ymin) * 0.1 + 1e-9
        ax.set_ylim(ymin - pad, ymax + pad)


def generate_individual_plots(
    rows: list,
    out_dir: str,
    extra_cols: list = None,
    theme_name: str = "light",
    theme: dict = None,
    config: dict = None,
) -> int:
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

    if theme is None:
        theme = LIGHT if theme_name == "light" else DARK

    all_defs = list(_STATIC_PLOT_DEFS)
    if extra_cols:
        all_defs.extend(extra_cols)

    generated = 0
    theme_out_dir = os.path.join(out_dir, "plots", theme_name)
    os.makedirs(theme_out_dir, exist_ok=True)

    for i, (filename, title, key) in enumerate(all_defs):
        xs, ys = extract_series(rows, key, config=config)
        if not xs:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.set_facecolor(theme.get("fig_bg", "#ffffff"))
        color = theme.get("lines", ["#81b29a"])[i % len(theme.get("lines", ["#81b29a"]))]
        plot_title = f"{title}{_rolling_suffix(key, config)}"
        _plot_single(ax, xs, ys, plot_title, key, theme=theme, color=color)
        fig.tight_layout()

        out_path = os.path.join(theme_out_dir, f"{filename}.png")
        _savefig(fig, out_path, dpi=170)
        print(f"  Saved: {out_path}")
        generated += 1

    return generated


#
# ─────────────────────────────────────────────────────────────────────────────
# Dual-theme charts + adaptive dashboard
# ─────────────────────────────────────────────────────────────────────────────
#

_PERCENT_SUBSTRINGS = (
    "accuracy",
    "f1",
    "kappa",
    "exact_match",
    "coverage",
    "inversion",
    "non_empty",
    "repetition",
    "semantic_similarity",
    "keyword_density",
)


def _should_percent(key: str) -> bool:
    lk = (key or "").lower()
    return any(s in lk for s in _PERCENT_SUBSTRINGS)


def _to_percent_if_needed(key: str, ys: list) -> tuple:
    """
    If values look like fractions in [0, 1], convert them to [0, 100].
    Returns (new_ys, did_convert).
    """
    if not ys:
        return ys, False
    if not _should_percent(key):
        return ys, False
    try:
        max_abs = max(abs(float(y)) for y in ys if y is not None)
    except Exception:
        return ys, False
    if max_abs <= 1.1:
        return [float(y) * 100.0 for y in ys], True
    return ys, False


def _setup_ax(ax, theme: dict):
    ax.set_facecolor(theme.get("panel_bg", "#ffffff"))
    ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.8, color=theme.get("grid", "#cccccc"))
    ax.tick_params(colors=theme.get("subtext", "#666666"), labelsize=8.8, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(theme.get("border", "#dddddd"))
    ax.spines["bottom"].set_color(theme.get("border", "#dddddd"))
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.margins(x=0.04, y=0.14)


def _smooth_series_for_display(xs: list, ys: list, samples_per_segment: int = 18) -> tuple:
    if len(xs) < 4 or len(xs) != len(ys):
        return list(xs), list(ys)

    xsf = [float(x) for x in xs]
    ysf = [float(y) for y in ys]
    h = [xsf[i + 1] - xsf[i] for i in range(len(xsf) - 1)]
    if any(step <= 0 for step in h):
        return list(xs), list(ys)

    delta = [(ysf[i + 1] - ysf[i]) / h[i] for i in range(len(h))]
    m = [0.0] * len(xsf)
    m[0] = delta[0]
    m[-1] = delta[-1]
    for idx in range(1, len(xsf) - 1):
        left = delta[idx - 1]
        right = delta[idx]
        if left == 0.0 or right == 0.0 or left * right < 0:
            m[idx] = 0.0
            continue
        w1 = 2 * h[idx] + h[idx - 1]
        w2 = h[idx] + 2 * h[idx - 1]
        m[idx] = (w1 + w2) / ((w1 / left) + (w2 / right))

    smooth_x = [xsf[0]]
    smooth_y = [ysf[0]]
    for idx in range(len(xsf) - 1):
        x0, x1 = xsf[idx], xsf[idx + 1]
        y0, y1 = ysf[idx], ysf[idx + 1]
        width = x1 - x0
        for sample in range(1, samples_per_segment + 1):
            t = sample / samples_per_segment
            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2
            smooth_x.append(x0 + width * t)
            smooth_y.append(h00 * y0 + h10 * width * m[idx] + h01 * y1 + h11 * width * m[idx + 1])
    return smooth_x, smooth_y


def _style_legend(legend, theme: dict) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor(theme.get("panel_bg", "#ffffff"))
    frame.set_edgecolor(theme.get("border", "#dddddd"))
    frame.set_alpha(0.95)
    for text in legend.get_texts():
        text.set_color(theme.get("text", "#111111"))


def _value_str(value: float, unit: str) -> str:
    if unit == "%":
        return f"{value:.1f}%"
    if unit.strip() == "words":
        return f"{value:.1f}"
    return f"{value:.3f}" if abs(value) <= 1.0 else f"{value:.2f}"


def _delta_color(kpi, theme: dict) -> str:
    if kpi.delta is None:
        return theme.get("muted", theme.get("subtext", "#666666"))
    is_good = (kpi.direction == "higher_better" and kpi.delta >= 0) or (kpi.direction == "lower_better" and kpi.delta <= 0)
    return theme.get("positive") if is_good else theme.get("negative")


def _savefig(fig, out_path: str, dpi: int = 200):
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_group_training_health(rows: list, out_path: str, theme: dict, config: dict = None) -> bool:
    import matplotlib.pyplot as plt

    xs_loss, ys_loss = extract_series(rows, "loss", config=config)
    xs_lr, ys_lr = extract_series(rows, "lr", config=config)
    if len(xs_loss) < 2 and len(xs_lr) < 2:
        return False

    fig, ax1 = plt.subplots(figsize=(12, 5))
    fig.set_facecolor(theme.get("fig_bg", "#ffffff"))
    _setup_ax(ax1, theme)
    color_loss = theme.get("negative", "#FF7B72")
    if len(xs_loss) >= 2:
        ax1.plot(xs_loss, ys_loss, marker="o", markersize=3.5, linewidth=2, color=color_loss, label="loss")
    ax1.set_xlabel("Step", color=theme.get("subtext", "#666666"))
    ax1.set_ylabel("Loss", color=theme.get("subtext", "#666666"))

    ax2 = ax1.twinx()
    _setup_ax(ax2, theme)
    color_lr = theme.get("neutral", "#79B8FF")
    if len(xs_lr) >= 2:
        ax2.plot(xs_lr, ys_lr, marker="o", markersize=3.5, linewidth=2, color=color_lr, label="lr")
    ax2.set_ylabel("Learning Rate", color=theme.get("subtext", "#666666"))

    fig.suptitle("Training Health (loss + learning rate)", color=theme.get("text", "#111111"), fontsize=14, fontweight="bold", y=0.98)
    _savefig(fig, out_path, dpi=200)
    return True


def _plot_group_throughput(rows: list, out_path: str, theme: dict, config: dict = None) -> bool:
    import matplotlib.pyplot as plt

    xs, ys = extract_series(rows, "tokens_per_sec", config=config)
    if len(xs) < 2:
        return False
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.set_facecolor(theme.get("fig_bg", "#ffffff"))
    _setup_ax(ax, theme)
    ax.plot(xs, ys, marker="o", markersize=4, linewidth=2, color=theme.get("neutral", "#79B8FF"))
    ax.set_xlabel("Step", color=theme.get("subtext", "#666666"))
    ax.set_ylabel("Tokens/sec", color=theme.get("subtext", "#666666"))
    fig.suptitle("Throughput (tokens/sec)", color=theme.get("text", "#111111"), fontsize=14, fontweight="bold", y=0.98)
    _savefig(fig, out_path, dpi=200)
    return True


def _plot_group_line_multi(rows: list, out_path: str, theme: dict, title: str, ylabel: str, series: list, config: dict = None) -> bool:
    """
    series: List of dicts {key, label, color}
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.set_facecolor(theme.get("fig_bg", "#ffffff"))
    _setup_ax(ax, theme)

    plotted = 0
    for si, item in enumerate(series):
        key = item["key"]
        label = item.get("label", key)
        color = item.get("color", theme.get("lines", ["#79B8FF"])[si % len(theme.get("lines", ["#79B8FF"]))])
        xs, ys = extract_series(rows, key, config=config)
        if len(xs) < 2:
            continue
        ys2, converted = _to_percent_if_needed(key, ys)
        ax.plot(xs, ys2, marker="o", markersize=3.5, linewidth=2, color=color, label=label)
        ax.fill_between(xs, ys2, alpha=0.08, color=color)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return False

    ax.set_xlabel("Step", color=theme.get("subtext", "#666666"))
    ax.set_ylabel(ylabel, color=theme.get("subtext", "#666666"))
    legend = ax.legend(loc="best", frameon=True, fontsize=9)
    _style_legend(legend, theme)
    fig.suptitle(
        f"{title}{_rolling_suffix([item['key'] for item in series], config)}",
        color=theme.get("text", "#111111"),
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    _savefig(fig, out_path, dpi=200)
    return True


def _plot_group_dual_axis(rows: list, out_path: str, theme: dict, title: str, left_key: str, right_key: str, left_label: str, right_label: str, left_color: str, right_color: str, config: dict = None) -> bool:
    import matplotlib.pyplot as plt

    xl, yl = extract_series(rows, left_key, config=config)
    xr, yr = extract_series(rows, right_key, config=config)
    if len(xl) < 2 and len(xr) < 2:
        return False
    fig, ax1 = plt.subplots(figsize=(12, 5))
    fig.set_facecolor(theme.get("fig_bg", "#ffffff"))
    _setup_ax(ax1, theme)
    if len(xl) >= 2:
        yl2, left_percent = _to_percent_if_needed(left_key, yl)
        ax1.plot(xl, yl2, marker="o", markersize=3.5, linewidth=2, color=left_color, label=left_label)
        ax1.set_ylabel(f"{left_label}", color=theme.get("subtext", "#666666"))
    ax1.set_xlabel("Step", color=theme.get("subtext", "#666666"))

    ax2 = ax1.twinx()
    _setup_ax(ax2, theme)
    if len(xr) >= 2:
        yr2, right_percent = _to_percent_if_needed(right_key, yr)
        ax2.plot(xr, yr2, marker="o", markersize=3.5, linewidth=2, color=right_color, label=right_label)
        ax2.set_ylabel(f"{right_label}", color=theme.get("subtext", "#666666"))

    fig.suptitle(
        f"{title}{_rolling_suffix([left_key, right_key], config)}",
        color=theme.get("text", "#111111"),
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    _savefig(fig, out_path, dpi=200)
    return True


def _plot_group_coverage_multi(rows: list, out_path: str, theme: dict, title: str, slot_cols: list, max_lines: int = 6, config: dict = None) -> bool:
    import matplotlib.pyplot as plt

    per_slot = []
    for col_key, slot_label in slot_cols:
        xs, ys = extract_series(rows, col_key, config=config)
        if len(xs) >= 2:
            per_slot.append((col_key, slot_label, xs, ys))
    if not per_slot:
        return False

    per_slot.sort(key=lambda t: (t[1] or ""))
    per_slot = per_slot[:max_lines]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.set_facecolor(theme.get("fig_bg", "#ffffff"))
    _setup_ax(ax, theme)

    for i, (col_key, slot_label, xs, ys) in enumerate(per_slot):
        ys2, _ = _to_percent_if_needed(col_key, ys)
        color = theme.get("lines", ["#79B8FF"])[i % len(theme.get("lines", ["#79B8FF"]))]
        ax.plot(xs, ys2, marker="o", markersize=3.5, linewidth=2, color=color, label=slot_label)

    ax.set_xlabel("Step", color=theme.get("subtext", "#666666"))
    ax.set_ylabel("Coverage (%)", color=theme.get("subtext", "#666666"))
    legend = ax.legend(loc="best", frameon=True, fontsize=9)
    _style_legend(legend, theme)
    fig.suptitle(
        f"{title}{_rolling_suffix([col_key for col_key, _ in slot_cols], config)}",
        color=theme.get("text", "#111111"),
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    _savefig(fig, out_path, dpi=200)
    return True


def _get_last_valid_for_key(rows: list, key: str) -> Optional[float]:
    last = None
    for r in rows:
        v = r.get(key)
        if v in ("", None):
            continue
        try:
            last = float(v)
        except (TypeError, ValueError):
            continue
    return last


def _plot_group_per_slot_final_bar(rows: list, out_path: str, theme: dict, slot_cols: list) -> bool:
    import matplotlib.pyplot as plt

    finals = []
    for col_key, slot_label in slot_cols:
        v = _get_last_valid_for_key(rows, col_key)
        if v is None:
            continue
        finals.append((slot_label, v))
    if len(finals) < 2:
        return False

    # Convert values to percent if they look like fractions.
    values = [v for _, v in finals]
    max_abs = max(abs(float(v)) for v in values) if values else 0.0
    if max_abs <= 1.1:
        values = [float(v) * 100.0 for v in values]
    else:
        values = [float(v) for v in values]

    # Order by slot performance
    ordered = sorted(zip([s for s, _ in finals], values), key=lambda t: t[1])
    labels = [o[0] for o in ordered]
    vals = [o[1] for o in ordered]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.set_facecolor(theme.get("fig_bg", "#ffffff"))
    _setup_ax(ax, theme)

    # Color by performance bands
    bars = []
    for i, v in enumerate(vals):
        if v >= 85:
            c = theme.get("positive", "#3FB68E")
        elif v >= 70:
            c = theme.get("warning", "#E3B341")
        else:
            c = theme.get("negative", "#FF7B72")
        bars.append(c)
    ax.barh(labels, vals, color=bars, alpha=0.95)
    ax.set_xlabel("Final coverage (%)", color=theme.get("subtext", "#666666"))
    ax.set_ylabel("Slot", color=theme.get("subtext", "#666666"))
    fig.suptitle("Per-slot coverage at final checkpoint", color=theme.get("text", "#111111"), fontsize=14, fontweight="bold", y=0.98)
    _savefig(fig, out_path, dpi=200)
    return True


def generate_grouped_insight_charts(
    rows: list,
    out_dir: str,
    config: dict,
    profile,
    theme_name: str,
    theme: dict,
) -> dict:
    """Return a mapping of semantic chart ids to generated PNG paths."""
    insights_dir = os.path.join(out_dir, "insights", theme_name)
    os.makedirs(insights_dir, exist_ok=True)

    slot_cols = get_per_slot_columns(rows)
    chart_paths: dict = {}

    def maybe_chart(chart_id: str, ok: bool, filename: str) -> None:
        if ok:
            chart_paths[chart_id] = os.path.join(insights_dir, filename)

    maybe_chart(
        "training_health",
        _plot_group_training_health(rows, os.path.join(insights_dir, f"insight_training_health_{theme_name}.png"), theme, config=config),
        f"insight_training_health_{theme_name}.png",
    )
    maybe_chart(
        "throughput",
        _plot_group_throughput(rows, os.path.join(insights_dir, f"insight_throughput_{theme_name}.png"), theme, config=config),
        f"insight_throughput_{theme_name}.png",
    )

    if profile.name == "classification":
        maybe_chart(
            "quant_quality",
            _plot_group_line_multi(
                rows,
                os.path.join(insights_dir, f"insight_classification_quality_{theme_name}.png"),
                theme,
                "Classification quality over time",
                "Rate / Score",
                [
                    {"key": "accuracy", "label": "Accuracy", "color": theme.get("positive")},
                    {"key": "f1", "label": "Macro F1", "color": theme.get("neutral")},
                    {"key": "mcc", "label": "MCC", "color": theme.get("lines", ["#79B8FF"])[0]},
                    {"key": "kappa", "label": "Kappa", "color": theme.get("lines", ["#79B8FF"])[3]},
                ],
                config=config,
            ),
            f"insight_classification_quality_{theme_name}.png",
        )
        maybe_chart(
            "stability",
            _plot_group_line_multi(
                rows,
                os.path.join(insights_dir, f"insight_stability_{theme_name}.png"),
                theme,
                "Continual-learning stability",
                "Delta vs initial accuracy",
                [{"key": "continual_learning_stability", "label": "Stability", "color": theme.get("warning")}],
                config=config,
            ),
            f"insight_stability_{theme_name}.png",
        )

    if profile.name in ("math_reasoning_quant", "math_reasoning_cot"):
        maybe_chart(
            "quant_quality",
            _plot_group_line_multi(
                rows,
                os.path.join(insights_dir, f"insight_math_reasoning_quality_{theme_name}.png"),
                theme,
                "Reasoning quality over time",
                "Rate / Score",
                [
                    {"key": "accuracy", "label": "Accuracy", "color": theme.get("positive")},
                    {"key": "exact_match", "label": "Exact match", "color": theme.get("neutral")},
                    {"key": "answer_overlap_f1", "label": "Answer overlap F1", "color": theme.get("lines", ["#79B8FF"])[3]},
                ],
                config=config,
            ),
            f"insight_math_reasoning_quality_{theme_name}.png",
        )
    if profile.name == "math_reasoning_cot":
        maybe_chart(
            "qual_quality",
            _plot_group_line_multi(
                rows,
                os.path.join(insights_dir, f"insight_cot_structure_{theme_name}.png"),
                theme,
                "Reasoning structure",
                "Rate / Count",
                [
                    {"key": "qual_cot_coverage_rate", "label": "CoT coverage", "color": theme.get("positive")},
                    {"key": "qual_cot_anchor_count_mean", "label": "Anchor count", "color": theme.get("neutral")},
                    {"key": "qual_cot_step_length_mean", "label": "Step length", "color": theme.get("lines", ["#79B8FF"])[1]},
                ],
                config=config,
            ),
            f"insight_cot_structure_{theme_name}.png",
        )
        maybe_chart(
            "response_health",
            _plot_group_dual_axis(
                rows,
                os.path.join(insights_dir, f"insight_response_health_{theme_name}.png"),
                theme,
                "Response health",
                left_key="qual_non_empty_rate",
                right_key="qual_mean_response_length",
                left_label="Non-empty rate",
                right_label="Mean response length",
                left_color=theme.get("positive"),
                right_color=theme.get("lines", ["#79B8FF"])[2],
                config=config,
            ),
            f"insight_response_health_{theme_name}.png",
        )

    if profile.name == "structured_nlg":
        maybe_chart(
            "qual_quality",
            _plot_group_coverage_multi(
                rows,
                os.path.join(insights_dir, f"insight_slot_coverage_multi_{theme_name}.png"),
                theme,
                "Per-slot coverage across training",
                slot_cols,
                config=config,
            ),
            f"insight_slot_coverage_multi_{theme_name}.png",
        )
        maybe_chart(
            "coverage_consistency_arc",
            _plot_group_line_multi(
                rows,
                os.path.join(insights_dir, f"insight_coverage_consistency_arc_{theme_name}.png"),
                theme,
                "Coverage and consistency",
                "Coverage (%)",
                [
                    {"key": "qual_slot_coverage_mean", "label": "Coverage (mean)", "color": theme.get("positive")},
                    {"key": "qual_pinned_slot_coverage_mean", "label": "Pinned coverage", "color": theme.get("lines", ["#79B8FF"])[1]},
                    {"key": "qual_consistency_score_mean", "label": "Consistency", "color": theme.get("lines", ["#79B8FF"])[3]},
                    {"key": "qual_pinned_consistency_score", "label": "Pinned consistency", "color": theme.get("lines", ["#79B8FF"])[4]},
                ],
                config=config,
            ),
            f"insight_coverage_consistency_arc_{theme_name}.png",
        )
        maybe_chart(
            "per_slot_final_bar",
            _plot_group_per_slot_final_bar(rows, os.path.join(insights_dir, f"insight_per_slot_final_bar_{theme_name}.png"), theme, slot_cols),
            f"insight_per_slot_final_bar_{theme_name}.png",
        )
        maybe_chart(
            "inversion_vs_coverage",
            _plot_group_dual_axis(
                rows,
                os.path.join(insights_dir, f"insight_inversion_vs_coverage_{theme_name}.png"),
                theme,
                "Inversion rate vs slot coverage",
                left_key="qual_slot_familyFriendly_inversion_rate",
                right_key="qual_slot_coverage_mean",
                left_label="Inversion rate",
                right_label="Coverage mean",
                left_color=theme.get("negative"),
                right_color=theme.get("positive"),
                config=config,
            ),
            f"insight_inversion_vs_coverage_{theme_name}.png",
        )
        maybe_chart(
            "quant_quality",
            _plot_group_line_multi(
                rows,
                os.path.join(insights_dir, f"insight_structured_nlg_quant_{theme_name}.png"),
                theme,
                "Quantitative quality",
                "Score",
                [
                    {"key": "answer_overlap_f1", "label": "Answer overlap F1", "color": theme.get("neutral")},
                    {"key": "perplexity", "label": "Perplexity", "color": theme.get("lines", ["#79B8FF"])[2]},
                ],
                config=config,
            ),
            f"insight_structured_nlg_quant_{theme_name}.png",
        )

    if profile.name == "domain_adaptation_keyword":
        maybe_chart(
            "qual_quality",
            _plot_group_line_multi(
                rows,
                os.path.join(insights_dir, f"insight_keyword_quality_{theme_name}.png"),
                theme,
                "Domain vocabulary adoption",
                "Rate / Score",
                [
                    {"key": "qual_keyword_density", "label": "Keyword density", "color": theme.get("positive")},
                    {"key": "qual_type_token_ratio", "label": "Type-token ratio", "color": theme.get("neutral")},
                    {"key": "qual_hapax_ratio", "label": "Hapax ratio", "color": theme.get("lines", ["#79B8FF"])[3]},
                ],
                config=config,
            ),
            f"insight_keyword_quality_{theme_name}.png",
        )
        maybe_chart(
            "response_health",
            _plot_group_dual_axis(
                rows,
                os.path.join(insights_dir, f"insight_response_health_{theme_name}.png"),
                theme,
                "Response health",
                left_key="qual_non_empty_rate",
                right_key="qual_repetition_rate",
                left_label="Non-empty rate",
                right_label="Repetition rate",
                left_color=theme.get("positive"),
                right_color=theme.get("warning"),
                config=config,
            ),
            f"insight_response_health_{theme_name}.png",
        )

    if profile.name in ("instruction_following_semantic", "generic"):
        maybe_chart(
            "qual_quality",
            _plot_group_line_multi(
                rows,
                os.path.join(insights_dir, f"insight_semantic_quality_{theme_name}.png"),
                theme,
                "Qualitative learning signal",
                "Similarity / Score",
                [
                    {"key": "qual_semantic_similarity", "label": "Semantic similarity", "color": theme.get("neutral")},
                    {"key": "qual_keyword_density", "label": "Keyword density", "color": theme.get("positive")},
                ],
                config=config,
            ),
            f"insight_semantic_quality_{theme_name}.png",
        )
        maybe_chart(
            "response_health",
            _plot_group_dual_axis(
                rows,
                os.path.join(insights_dir, f"insight_response_health_{theme_name}.png"),
                theme,
                "Response health",
                left_key="qual_non_empty_rate",
                right_key="qual_mean_response_length",
                left_label="Non-empty rate",
                right_label="Mean response length",
                left_color=theme.get("positive"),
                right_color=theme.get("lines", ["#79B8FF"])[2],
                config=config,
            ),
            f"insight_response_health_{theme_name}.png",
        )

    return chart_paths


def generate_dashboard(
    rows: list,
    out_dir: str,
    config: dict = None,
    theme_name: str = "dark",
    theme: dict = None,
) -> str:
    """
    Build a dual-theme dashboard PNG with:
      - KPI cards (adaptive per usecase)
      - auto-insights text
      - a primary grouped chart + optional secondary charts

    Panels are omitted if they have no underlying data (prevents empty graphs).
    """
    if theme is None:
        theme = DARK if theme_name == "dark" else LIGHT

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        print("Error: matplotlib is required. Install it with: pip install matplotlib")
        return ""

    profile = detect_usecase(rows, config)
    kpis = select_kpis(rows, profile)
    insights = generate_insights(rows, profile)

    # Generate grouped charts (so dashboard can just lay them out)
    group_paths = generate_grouped_insight_charts(rows, out_dir, config, profile, theme_name, theme)

    header_text = _build_config_header(config) if config else "InfiniTune — Evaluation Dashboard"

    # Select which charts to show based on availability
    primary = group_paths.get("primary_quality")
    secondary_left = group_paths.get("coverage_consistency_arc") or group_paths.get("primary_quality")
    secondary_right = group_paths.get("per_slot_final_bar") or group_paths.get("inversion_vs_coverage")
    train_health = group_paths.get("training_health")
    throughput = group_paths.get("throughput")

    any_chart = bool(primary or secondary_left or secondary_right or train_health or throughput)
    if not any_chart:
        print("  No plottable data found for dashboard.")
        return ""

    # Figure layout
    fig_w = 16
    base_h = 8.5
    fig_h = base_h + (2.5 if insights else 0.0)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=theme.get("fig_bg", "#ffffff"))
    outer = gridspec.GridSpec(5, 1, figure=fig, height_ratios=[0.65, 1.2, 0.8, 3.2, 2.4], hspace=0.35)

    # Header strip
    header_ax = fig.add_subplot(outer[0])
    header_ax.set_facecolor(theme.get("panel_bg", "#ffffff"))
    header_ax.axis("off")
    header_ax.text(
        0.02, 0.60,
        "InfiniTune",
        transform=header_ax.transAxes,
        ha="left", va="center",
        fontsize=18, fontweight="bold", color=theme.get("text", "#111111"),
    )
    header_ax.text(
        0.98, 0.60,
        f"Evaluation Dashboard ({theme_name})",
        transform=header_ax.transAxes,
        ha="right", va="center",
        fontsize=10, fontweight="bold", color=theme.get("subtext", "#666666"),
    )
    header_ax.text(
        0.02, 0.18,
        header_text,
        transform=header_ax.transAxes,
        ha="left", va="bottom",
        fontsize=7.5, color=theme.get("subtext", "#666666"),
        family="monospace",
    )

    # KPI strip
    kpi_ax = fig.add_subplot(outer[1])
    kpi_ax.set_facecolor(theme.get("panel_bg", "#ffffff"))
    kpi_ax.axis("off")

    kpi_cards = kpis[:6]
    card_n = max(1, len(kpi_cards))
    card_w = 0.95 / card_n
    for idx, kpi in enumerate(kpi_cards):
        x0 = 0.02 + idx * card_w
        y0 = 0.08
        rect = FancyBboxPatch(
            (x0, y0), card_w - 0.015, 0.84,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.0,
            edgecolor=theme.get("border", "#dddddd"),
            facecolor=theme.get("neutral", "#79B8FF") if kpi.delta is None else (theme.get("positive") if ((kpi.direction == "higher_better" and (kpi.delta or 0) > 0) or (kpi.direction == "lower_better" and (kpi.delta or 0) < 0)) else theme.get("negative")),
            alpha=0.18,
            transform=kpi_ax.transAxes,
        )
        kpi_ax.add_patch(rect)
        kpi_ax.text(
            x0 + 0.012, y0 + 0.62,
            kpi.label,
            transform=kpi_ax.transAxes,
            ha="left", va="center",
            fontsize=9,
            color=theme.get("text", "#111111"),
            fontweight="bold",
        )
        val = kpi.value
        # Heuristic: show percent if unit is '%' or if value looks like a fraction for known keys.
        if kpi.unit == "%":
            kpi_val_str = f"{val:.1f}%"
        else:
            kpi_val_str = f"{val:.3f}" if abs(val) <= 1.0 else f"{val:.2f}"
        kpi_ax.text(
            x0 + 0.012, y0 + 0.36,
            kpi_val_str,
            transform=kpi_ax.transAxes,
            ha="left", va="center",
            fontsize=18,
            color=theme.get("text", "#111111"),
            fontweight="bold",
        )
        if kpi.delta is not None and kpi.delta_label != "":
            delta = kpi.delta
            sign = "+" if delta >= 0 else "−"
            delta_abs = abs(delta)
            delta_str = f"{sign}{delta_abs:.2f}"
            kpi_ax.text(
                x0 + 0.012, y0 + 0.20,
                f"{delta_str} {kpi.delta_label}",
                transform=kpi_ax.transAxes,
                ha="left", va="center",
                fontsize=8.5,
                color=theme.get("subtext", "#666666"),
            )

    # Insights
    insights_ax = fig.add_subplot(outer[2])
    insights_ax.set_facecolor(theme.get("panel_bg", "#ffffff"))
    insights_ax.axis("off")
    if insights:
        insights_ax.text(
            0.02, 0.55,
            "Key insights",
            transform=insights_ax.transAxes,
            ha="left", va="center",
            fontsize=11,
            fontweight="bold",
            color=theme.get("text", "#111111"),
        )
        y = 0.28
        for line in insights[:6]:
            insights_ax.text(
                0.02, y,
                f"• {line}",
                transform=insights_ax.transAxes,
                ha="left", va="center",
                fontsize=9,
                color=theme.get("subtext", "#666666"),
            )
            y -= 0.13

    # Primary chart area
    charts_ax = fig.add_subplot(outer[3])
    charts_ax.set_facecolor(theme.get("panel_bg", "#ffffff"))
    charts_ax.axis("off")

    # Layout inside charts: [primary] + optional [secondary_right]
    internal = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[3], wspace=0.15, hspace=0.12)
    # Primary spans full width (row 0 + 1 col 0)
    if primary:
        ax0 = fig.add_subplot(internal[0, 0])
        img0 = plt.imread(primary)
        ax0.imshow(img0)
        ax0.axis("off")
    if secondary_left:
        ax1 = fig.add_subplot(internal[1, 0])
        img1 = plt.imread(secondary_left)
        ax1.imshow(img1)
        ax1.axis("off")
    if secondary_right:
        ax2 = fig.add_subplot(internal[0, 1])
        img2 = plt.imread(secondary_right)
        ax2.imshow(img2)
        ax2.axis("off")

    # Bottom charts (training health + throughput) — dynamic to avoid empty panels.
    bottom_ax = fig.add_subplot(outer[4])
    bottom_ax.set_facecolor(theme.get("panel_bg", "#ffffff"))
    bottom_ax.axis("off")

    bottom_imgs = []
    if train_health:
        bottom_imgs.append(("training_health", train_health))
    if throughput:
        bottom_imgs.append(("throughput", throughput))

    if len(bottom_imgs) == 2:
        bottom_grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[4], wspace=0.15)
        axl = fig.add_subplot(bottom_grid[0, 0])
        axl.imshow(plt.imread(bottom_imgs[0][1]))
        axl.axis("off")

        axr = fig.add_subplot(bottom_grid[0, 1])
        axr.imshow(plt.imread(bottom_imgs[1][1]))
        axr.axis("off")
    elif len(bottom_imgs) == 1:
        bottom_grid = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[4], wspace=0.15)
        ax = fig.add_subplot(bottom_grid[0, 0])
        ax.imshow(plt.imread(bottom_imgs[0][1]))
        ax.axis("off")
    else:
        bottom_ax.text(
            0.02,
            0.50,
            "No training health / throughput data available.",
            transform=bottom_ax.transAxes,
            ha="left",
            va="center",
            fontsize=11,
            color=theme.get("subtext", "#666666"),
        )

    out_path = os.path.join(out_dir, f"dashboard_{theme_name}.png")
    _savefig(fig, out_path, dpi=200)
    print(f"  Dashboard saved: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────────────────────────────────────

def _dedupe_existing_paths(paths: list) -> list:
    seen = set()
    ordered = []
    for path in paths:
        if not path or not os.path.isfile(path) or path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def _dashboard_sections(out_dir: str, theme_name: str, profile, group_paths: dict) -> list:
    plot_theme_dir = os.path.join(out_dir, "plots", theme_name)
    plot_paths = []
    if os.path.isdir(plot_theme_dir):
        for file_name in os.listdir(plot_theme_dir):
            if file_name.endswith(".png"):
                plot_paths.append(os.path.join(plot_theme_dir, file_name))

    training = [group_paths.get("training_health"), group_paths.get("throughput")]
    training.extend(
        path for path in plot_paths
        if os.path.basename(path) in {"grad_norm.png", "eval_cycle_time_s.png", "train_loss.png", "learning_rate.png", "tokens_per_sec.png"}
    )

    quantitative = []
    if profile.name in ("classification", "math_reasoning_quant", "math_reasoning_cot", "structured_nlg"):
        quantitative.extend([group_paths.get("quant_quality"), group_paths.get("stability")])
        quantitative.extend(
            path for path in plot_paths
            if os.path.basename(path) in {"perplexity.png", "eval_loss.png", "answer_overlap_f1.png", "accuracy.png", "f1.png", "mcc.png", "kappa.png", "exact_match.png"}
        )

    qualitative = [group_paths.get("qual_quality"), group_paths.get("coverage_consistency_arc"), group_paths.get("per_slot_final_bar"), group_paths.get("inversion_vs_coverage"), group_paths.get("response_health")]
    qualitative.extend(
        path for path in plot_paths
        if os.path.basename(path) in {
            "qual_semantic_similarity.png",
            "qual_keyword_density.png",
            "qual_type_token_ratio.png",
            "qual_hapax_ratio.png",
            "qual_cot_anchor_count.png",
            "qual_cot_step_length.png",
            "qual_cot_coverage.png",
            "qual_mean_response_length.png",
            "qual_repetition_rate.png",
            "qual_non_empty_rate.png",
            "qual_slot_coverage_mean.png",
            "qual_consistency_score_mean.png",
            "qual_pinned_consistency.png",
        }
    )

    return [
        ("Training Curves", "Loss, learning-rate, throughput, and training-health signals.", _dedupe_existing_paths(training)),
        ("Quantitative Metrics", f"Hard evaluation signals that show {profile.evidence_label}.", _dedupe_existing_paths(quantitative)),
        ("Qualitative Metrics", "Usecase-aware qualitative signals that explain how the model is improving, not just whether it improved.", _dedupe_existing_paths(qualitative)),
    ]


def _theme_line_color(theme: dict, color_key: str) -> str:
    line_map = theme.get("line_map", {})
    if isinstance(line_map, dict) and color_key in line_map:
        return line_map[color_key]
    lines = theme.get("lines", {})
    if isinstance(lines, dict) and color_key in lines:
        return lines[color_key]
    return theme.get(color_key, theme.get("neutral", "#79B8FF"))


def _dashboard_surface_color(status: str, theme: dict, alpha: float = 0.16):
    from matplotlib.colors import to_rgba

    lookup = {
        "good": theme.get("positive", "#3FB68E"),
        "bad": theme.get("negative", "#FF7B72"),
        "warning": theme.get("warning", "#E3B341"),
        "neutral": theme.get("neutral", "#79B8FF"),
    }
    return to_rgba(lookup.get(status, lookup["neutral"]), alpha)


def _wrap_chip_lines(chips: list, max_chars: int = 54) -> list:
    lines = []
    current = []
    current_len = 0
    for chip in chips:
        token = str(chip).strip()
        if not token:
            continue
        extra = len(token) + (3 if current else 0)
        if current and (current_len + extra) > max_chars:
            lines.append("  •  ".join(current))
            current = [token]
            current_len = len(token)
        else:
            current.append(token)
            current_len += extra
    if current:
        lines.append("  •  ".join(current))
    return lines[:3]


def _draw_kpi_cards(ax, cards: list, theme: dict, cols: int) -> None:
    from matplotlib.patches import FancyBboxPatch

    ax.axis("off")
    if not cards:
        return
    rows_needed = max(1, math.ceil(len(cards) / cols))
    gap_x = 0.022
    gap_y = 0.15
    usable_w = 0.96
    usable_h = 0.95
    card_w = (usable_w - gap_x * (cols - 1)) / cols
    card_h = (usable_h - gap_y * (rows_needed - 1)) / rows_needed
    single_centered = len(cards) == 1 and cols == 1
    for idx, kpi in enumerate(cards):
        row = idx // cols
        col = idx % cols
        current_w = 0.64 if single_centered else card_w
        x0 = 0.18 if single_centered else 0.02 + col * (card_w + gap_x)
        y0 = 0.98 - (row + 1) * card_h - row * gap_y
        rect = FancyBboxPatch(
            (x0, y0),
            current_w,
            card_h,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            linewidth=1.0,
            edgecolor=theme.get("border", "#D0D7DE"),
            facecolor=_dashboard_surface_color(getattr(kpi, "status", "neutral"), theme, alpha=0.18),
            transform=ax.transAxes,
        )
        accent = FancyBboxPatch(
            (x0, y0 + card_h - 0.024),
            current_w,
            0.024,
            boxstyle="round,pad=0.005,rounding_size=0.02",
            linewidth=0,
            facecolor=_theme_line_color(theme, "positive" if getattr(kpi, "status", "neutral") == "good" else "warning" if getattr(kpi, "status", "neutral") == "warning" else "negative" if getattr(kpi, "status", "neutral") == "bad" else "neutral"),
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.add_patch(accent)
        ax.text(
            x0 + 0.026,
            y0 + card_h - 0.075,
            textwrap.fill(kpi.label, width=18 if cols >= 4 else 24),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.4,
            fontweight="bold",
            color=theme.get("subtext", "#667085"),
            linespacing=1.18,
        )
        ax.text(
            x0 + 0.026,
            y0 + card_h * 0.57,
            _value_str(kpi.value, kpi.unit),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=23.8 if cols >= 4 else 25.5,
            fontweight="bold",
            color=theme.get("text", "#111111"),
        )
        delta_text = kpi.delta_display or ""
        if not delta_text and kpi.delta is not None:
            arrow = "↑" if kpi.delta >= 0 else "↓"
            amount = f"{abs(kpi.delta):.1f} percentage points" if kpi.unit == "%" else f"{abs(kpi.delta):.3f}"
            suffix = f" {kpi.delta_label}".strip()
            delta_text = f"{arrow} {amount}{suffix}"
        ax.text(
            x0 + 0.026,
            y0 + card_h * 0.28,
            textwrap.fill(delta_text, width=24 if cols >= 4 else 32),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=_delta_color(kpi, theme) if getattr(kpi, "delta", None) is not None else theme.get("neutral"),
            linespacing=1.22,
        )
        supporting = getattr(kpi, "comparison_basis", "") or getattr(kpi, "supporting_caption", "") or getattr(kpi, "source_note", "")
        if supporting:
            ax.text(
                x0 + 0.026,
                y0 + 0.05,
                textwrap.shorten(supporting, width=36 if cols >= 4 else 44, placeholder="…"),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7.9,
                color=theme.get("muted", "#7E8AA5"),
            )


def _draw_takeaway_cards(ax, cards: list, theme: dict) -> None:
    from matplotlib.patches import FancyBboxPatch

    def _clamp_lines(text: str, width: int, max_lines: int = 2) -> str:
        lines = textwrap.wrap((text or "").strip(), width=width)
        if not lines:
            return ""
        if len(lines) <= max_lines:
            return "\n".join(lines)
        kept = lines[:max_lines]
        kept[-1] = textwrap.shorten(kept[-1], width=max(12, width - 2), placeholder="...")
        return "\n".join(kept)

    ax.axis("off")
    if not cards:
        return
    cards = cards[:2]
    cols = min(2, len(cards))
    gap_x = 0.024
    usable_w = 0.96
    card_w = (usable_w - gap_x * (cols - 1)) / cols
    for idx, card in enumerate(cards):
        x0 = 0.02 + idx * (card_w + gap_x)
        y0 = 0.06
        card_h = 0.84
        rect = FancyBboxPatch(
            (x0, y0),
            card_w,
            card_h,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            linewidth=1.0,
            edgecolor=theme.get("border", "#D0D7DE"),
            facecolor=_dashboard_surface_color(getattr(card, "tone", "neutral"), theme, alpha=0.11),
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        tone_key = getattr(card, "tone", "neutral")
        tone_color_key = "positive" if tone_key == "good" else "negative" if tone_key == "bad" else tone_key if tone_key in {"warning", "neutral"} else "neutral"
        inset = 0.028
        accent = FancyBboxPatch(
            (x0, y0 + card_h - 0.02),
            card_w,
            0.018,
            boxstyle="round,pad=0.0,rounding_size=0.012",
            linewidth=0,
            facecolor=_theme_line_color(theme, tone_color_key),
            transform=ax.transAxes,
        )
        ax.add_patch(accent)

        title_width = 24 if cols == 2 else 38
        body_width = 32 if cols == 2 else 52
        title_text = _clamp_lines(card.title, width=title_width, max_lines=2)
        body_text = _clamp_lines(card.body, width=body_width, max_lines=2)
        badge_text = textwrap.shorten(getattr(card, "badge", "") or "takeaway", width=22, placeholder="...")

        ax.text(
            x0 + inset,
            y0 + card_h - 0.12,
            title_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            color=theme.get("text", "#111111"),
            linespacing=1.16,
        )

        if badge_text:
            badge_w = min(0.18, max(0.09, 0.0105 * len(badge_text)))
            badge = FancyBboxPatch(
                (x0 + card_w - badge_w - inset, y0 + card_h - 0.19),
                badge_w,
                0.10,
                boxstyle="round,pad=0.012,rounding_size=0.025",
                linewidth=0,
                facecolor=_dashboard_surface_color(getattr(card, "tone", "neutral"), theme, alpha=0.24),
                transform=ax.transAxes,
            )
            ax.add_patch(badge)
            ax.text(
                x0 + card_w - badge_w - inset + 0.012,
                y0 + card_h - 0.14,
                badge_text,
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=7.1,
                fontweight="bold",
                color=_theme_line_color(theme, tone_color_key),
            )

        ax.text(
            x0 + inset,
            y0 + 0.14,
            body_text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.0,
            color=theme.get("subtext", "#667085"),
            linespacing=1.5,
        )


def _render_chart_spec(ax, chart_spec, theme: dict) -> None:
    import matplotlib.pyplot as plt

    _setup_ax(ax, theme)
    ax.set_title(chart_spec.title, loc="left", fontsize=13.4, fontweight="bold", color=theme.get("text", "#111111"), pad=16)
    if chart_spec.subtitle:
        ax.text(0.0, 1.015, chart_spec.subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.2, color=theme.get("subtext", "#667085"), linespacing=1.4)
    if chart_spec.chart_type == "barh":
        labels = []
        if chart_spec.note:
            try:
                labels = json.loads(chart_spec.note).get("labels", [])
            except Exception:
                labels = []
        trace = chart_spec.traces[0]
        values = list(trace.x)
        y_pos = list(range(len(values)))
        colors = [_theme_line_color(theme, trace.color_key)] * len(values)
        if len(values) > 1:
            min_index = values.index(min(values))
            colors[min_index] = theme.get("negative", "#FF7B72")
        ax.barh(y_pos, values, color=colors, alpha=0.95)
        ax.set_yticks(y_pos, labels if labels else [str(idx) for idx in y_pos])
        ax.set_xlabel(chart_spec.x_label or "Value", color=theme.get("subtext", "#667085"), labelpad=12)
        ax.set_ylabel("")
        ax.tick_params(axis="y", pad=8)
        ax.tick_params(axis="x", pad=6)
        ax.margins(y=0.10)
        ax.invert_yaxis()
        return

    secondary_present = any(trace.axis == "secondary" for trace in chart_spec.traces) or any(th.axis == "secondary" for th in chart_spec.thresholds)
    ax2 = ax.twinx() if secondary_present else None
    if ax2 is not None:
        ax2.set_facecolor("none")
        ax2.tick_params(colors=theme.get("subtext", "#667085"), labelsize=9, pad=7)
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["right"].set_color(theme.get("border", "#D0D7DE"))
        ax2.grid(False)

    for trace in chart_spec.traces:
        target_ax = ax2 if trace.axis == "secondary" and ax2 is not None else ax
        color = _theme_line_color(theme, trace.color_key)
        linestyle = "--" if trace.style == "dashed" else ":" if trace.style == "dotted" else "-"
        if trace.trace_type == "bar":
            target_ax.bar(trace.x, trace.y, color=color, alpha=0.7, label=trace.name)
        else:
            smooth_x, smooth_y = _smooth_series_for_display(trace.x, trace.y) if linestyle == "-" else (trace.x, trace.y)
            target_ax.plot(
                smooth_x,
                smooth_y,
                color=color,
                linewidth=3.0,
                linestyle=linestyle,
                alpha=0.98,
                solid_capstyle="round",
                solid_joinstyle="round",
                label=trace.name,
            )
            target_ax.plot(
                trace.x,
                trace.y,
                linestyle="None",
                marker="o",
                markersize=4.4,
                markerfacecolor=color,
                markeredgewidth=0,
                alpha=0.95,
            )
            if trace.fill:
                target_ax.fill_between(smooth_x, smooth_y, color=color, alpha=0.12)

    for threshold in chart_spec.thresholds:
        target_ax = ax2 if threshold.axis == "secondary" and ax2 is not None else ax
        color = _theme_line_color(theme, threshold.color_key)
        linestyle = "--" if threshold.style == "dashed" else "-"
        target_ax.axhline(threshold.value, color=color, linewidth=1.2, linestyle=linestyle, alpha=0.7, label=threshold.label)

    ax.set_xlabel(chart_spec.x_label or "Step", color=theme.get("subtext", "#667085"), labelpad=14)
    ax.set_ylabel(chart_spec.y_label or "", color=theme.get("subtext", "#667085"), labelpad=14)
    if ax2 is not None:
        ax2.set_ylabel(chart_spec.y2_label or "", color=theme.get("subtext", "#667085"), labelpad=14)

    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles.extend(handles2)
        labels.extend(labels2)
    if handles:
        legend_cols = 1 if len(handles) <= 2 else 2 if len(handles) <= 4 else min(4, math.ceil(len(handles) / 2))
        legend = ax.legend(handles, labels, loc="upper left", frameon=True, fontsize=8.7, ncol=legend_cols, borderpad=0.7, labelspacing=0.7, handlelength=2.4, columnspacing=1.0)
        _style_legend(legend, theme)


def _save_chart_spec_png(chart_spec, out_path: str, theme: dict) -> str:
    import matplotlib.pyplot as plt

    figsize = (12, 6.2) if chart_spec.preferred_aspect == "wide" else (8.5, 6.0)
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor(theme.get("fig_bg", "#FFFFFF"))
    _render_chart_spec(ax, chart_spec, theme)
    _savefig(fig, out_path, dpi=220)
    return out_path


def generate_dashboard_v2(
    rows: list,
    out_dir: str,
    config: dict = None,
    theme_name: str = "dark",
    theme: dict = None,
    presentation_spec=None,
) -> Dict[str, str]:
    if theme is None:
        theme = DARK if theme_name == "dark" else LIGHT

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("Error: matplotlib is required. Install it with: pip install matplotlib")
        return {}

    spec = presentation_spec or build_presentation_spec(rows, config)
    sections = [section for section in spec.sections if section.charts]
    primary_kpis = spec.kpi_cards[:4]
    secondary_kpis = spec.kpi_cards[4:6]
    takeaways = spec.takeaway_cards[:2]
    hero_chart = None
    for preferred_section in ("quantitative", "qualitative", "training"):
        for section in sections:
            if section.id != preferred_section:
                continue
            hero_chart = next((chart for chart in section.charts if chart.role == "hero"), None)
            if hero_chart:
                break
        if hero_chart:
            break

    hero_section_id = hero_chart.section if hero_chart else ""
    hero_support = []
    if hero_chart:
        hero_support = [
            chart
            for section in sections
            if section.id == hero_section_id
            for chart in section.charts
            if chart.id != hero_chart.id
        ]
    remaining_sections = []
    for section in sections:
        remaining = [chart for chart in section.charts if chart.id != (hero_chart.id if hero_chart else "")]
        if section.id == hero_section_id:
            remaining = [chart for chart in remaining if chart.id not in {chart.id for chart in hero_support}]
        if remaining:
            remaining_sections.append((section, remaining))

    height_ratios = [1.45, 2.15]
    if secondary_kpis:
        height_ratios.append(1.48)
    if takeaways:
        height_ratios.append(1.8)
    if hero_chart:
        height_ratios.append(3.35)
    if hero_support:
        height_ratios.append(2.65)
    for section, charts in remaining_sections:
        height_ratios.append(0.9)
        rows_needed = math.ceil(len(charts) / 2)
        height_ratios.extend([2.55] * rows_needed)

    fig = plt.figure(figsize=(16.2, max(12.8, sum(height_ratios) * 1.16)), facecolor=theme.get("fig_bg", "#FFFFFF"))
    outer = gridspec.GridSpec(len(height_ratios), 1, figure=fig, height_ratios=height_ratios, hspace=0.56)
    fig.subplots_adjust(left=0.04, right=0.985, top=0.982, bottom=0.03)
    row_idx = 0

    header_grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row_idx], width_ratios=[1.15, 1.0], wspace=0.08)
    header_ax = fig.add_subplot(header_grid[0, 0])
    meta_ax = fig.add_subplot(header_grid[0, 1])
    row_idx += 1
    header_ax.axis("off")
    meta_ax.axis("off")
    header_ax.set_facecolor(theme.get("fig_bg", "#FFFFFF"))
    meta_ax.set_facecolor(theme.get("fig_bg", "#FFFFFF"))
    header_ax.text(0.00, 1.0, spec.header.get("product", "InfiniTune"), transform=header_ax.transAxes, ha="left", va="top", fontsize=17.4, fontweight="bold", color=theme.get("text", "#111111"))
    header_ax.text(0.00, 0.63, spec.header.get("title", "Evaluation Dashboard"), transform=header_ax.transAxes, ha="left", va="top", fontsize=13.6, fontweight="bold", color=theme.get("text", "#111111"))
    header_ax.text(0.00, 0.02, textwrap.fill(spec.header.get("subtitle", ""), width=60), transform=header_ax.transAxes, ha="left", va="bottom", fontsize=9.5, color=theme.get("subtext", "#667085"), linespacing=1.52)
    chips = spec.header.get("chips", [])[:8]
    chip_lines = _wrap_chip_lines(chips, max_chars=54)
    if chip_lines:
        meta_ax.text(
            1.0,
            0.6,
            "\n".join(chip_lines),
            transform=meta_ax.transAxes,
            ha="right",
            va="center",
            fontsize=9.7,
            color=theme.get("subtext", "#667085"),
            linespacing=1.58,
        )

    kpi_ax = fig.add_subplot(outer[row_idx])
    row_idx += 1
    kpi_ax.set_facecolor(theme.get("fig_bg", "#FFFFFF"))
    _draw_kpi_cards(kpi_ax, primary_kpis, theme, cols=min(4, max(1, len(primary_kpis))))

    if secondary_kpis:
        secondary_ax = fig.add_subplot(outer[row_idx])
        row_idx += 1
        secondary_ax.set_facecolor(theme.get("fig_bg", "#FFFFFF"))
        _draw_kpi_cards(secondary_ax, secondary_kpis, theme, cols=min(2, max(1, len(secondary_kpis))))

    if takeaways:
        takeaway_ax = fig.add_subplot(outer[row_idx])
        row_idx += 1
        takeaway_ax.set_facecolor(theme.get("fig_bg", "#FFFFFF"))
        _draw_takeaway_cards(takeaway_ax, takeaways, theme)

    if hero_chart:
        hero_ax = fig.add_subplot(outer[row_idx])
        row_idx += 1
        hero_ax.set_facecolor(theme.get("panel_bg", "#FFFFFF"))
        _render_chart_spec(hero_ax, hero_chart, theme)

    if hero_support:
        cols = 2 if len(hero_support) > 1 else 1
        support_grid = gridspec.GridSpecFromSubplotSpec(1, cols, subplot_spec=outer[row_idx], wspace=0.24, hspace=0.0)
        for idx, chart in enumerate(hero_support[:2]):
            chart_ax = fig.add_subplot(support_grid[0, idx])
            chart_ax.set_facecolor(theme.get("panel_bg", "#FFFFFF"))
            _render_chart_spec(chart_ax, chart, theme)
        row_idx += 1

    for section, section_charts in remaining_sections:
        section_ax = fig.add_subplot(outer[row_idx])
        row_idx += 1
        section_ax.axis("off")
        section_ax.set_facecolor(theme.get("fig_bg", "#FFFFFF"))
        section_ax.text(0.0, 0.78, section.title, transform=section_ax.transAxes, ha="left", va="center", fontsize=14.4, fontweight="bold", color=theme.get("text", "#111111"))
        section_ax.text(0.0, 0.12, textwrap.fill(section.description, width=92), transform=section_ax.transAxes, ha="left", va="center", fontsize=9.4, color=theme.get("subtext", "#667085"), linespacing=1.48)

        for chart_start in range(0, len(section_charts), 2):
            row_charts = section_charts[chart_start:chart_start + 2]
            cols = 2 if len(row_charts) > 1 else 1
            support_grid = gridspec.GridSpecFromSubplotSpec(1, cols, subplot_spec=outer[row_idx], wspace=0.24, hspace=0.0)
            for idx, chart in enumerate(row_charts):
                chart_ax = fig.add_subplot(support_grid[0, idx])
                chart_ax.set_facecolor(theme.get("panel_bg", "#FFFFFF"))
                _render_chart_spec(chart_ax, chart, theme)
            row_idx += 1

    dashboards_dir = os.path.join(out_dir, "dashboards")
    os.makedirs(dashboards_dir, exist_ok=True)
    png_path = os.path.join(dashboards_dir, f"dashboard_{theme_name}.png")
    svg_path = os.path.join(dashboards_dir, f"dashboard_{theme_name}.svg")
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Dashboard saved: {png_path}")
    return {"png": png_path, "svg": svg_path}


def render_plot_artifacts(rows: list, artifact_root: str, config: dict = None) -> Dict[str, Any]:
    """
    Low-level renderer used by the shared artifact orchestrator.

    Returns a payload of generated files and non-fatal warnings/errors.
    """
    payload: Dict[str, Any] = {
        "dashboards": {},
        "insights": {"dark": [], "light": []},
        "plots": {"dark": [], "light": []},
        "presentation": {},
        "warnings": [],
        "errors": [],
    }

    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError as exc:
        payload["errors"].append(f"matplotlib unavailable: {exc}")
        return payload

    if not rows:
        payload["warnings"].append("No metric rows available for plot rendering.")
        return payload

    presentation_spec = build_presentation_spec(rows, config)
    for block in getattr(presentation_spec, "details_blocks", []) or []:
        items = block.get("items", []) if isinstance(block, dict) else []
        for item in items:
            payload["warnings"].append(str(item))
    per_slot_cols = get_per_slot_columns(rows)
    extra_plot_defs = []
    for key, slot_label in per_slot_cols:
        safe_filename = f"qual_slot_{re.sub(r'[^a-zA-Z0-9]', '_', slot_label)}_coverage"
        extra_plot_defs.append((safe_filename, f"Slot Coverage: {slot_label}", key))

    for theme_name, theme in (("light", LIGHT), ("dark", DARK)):
        try:
            generated = generate_individual_plots(
                rows,
                artifact_root,
                extra_cols=extra_plot_defs,
                theme_name=theme_name,
                theme=theme,
                config=config,
            )
            plot_dir = os.path.join(artifact_root, "plots", theme_name)
            if os.path.isdir(plot_dir):
                payload["plots"][theme_name] = [
                    os.path.join(plot_dir, name)
                    for name in sorted(os.listdir(plot_dir))
                    if name.endswith(".png")
                ]
            if generated == 0:
                payload["warnings"].append(f"No individual plots were generated for theme '{theme_name}'.")
        except Exception as exc:
            payload["errors"].append(f"individual_plots[{theme_name}] failed: {exc}")

        try:
            dashboard_paths = generate_dashboard_v2(rows, artifact_root, config=config, theme_name=theme_name, theme=theme, presentation_spec=presentation_spec)
            if dashboard_paths.get("png"):
                payload["dashboards"][theme_name] = dashboard_paths["png"]
            if dashboard_paths.get("svg"):
                payload["dashboards"][f"{theme_name}_svg"] = dashboard_paths["svg"]
        except Exception as exc:
            payload["errors"].append(f"dashboard[{theme_name}] failed: {exc}")

        try:
            profile = presentation_spec.profile or detect_usecase(rows, config)
            generate_grouped_insight_charts(rows, artifact_root, config, profile, theme_name, theme)
            insight_dir = os.path.join(artifact_root, "insights", theme_name)
            os.makedirs(insight_dir, exist_ok=True)
            exported = []
            for chart in presentation_spec.chart_specs:
                chart_path = os.path.join(insight_dir, f"presentation_{chart.id}_{theme_name}.png")
                try:
                    _save_chart_spec_png(chart, chart_path, theme)
                    chart.fallback_paths[theme_name] = chart_path
                    exported.append(chart_path)
                except Exception as chart_exc:
                    payload["warnings"].append(f"presentation_chart[{theme_name}:{chart.id}] skipped: {chart_exc}")
            payload["insights"][theme_name] = sorted(
                {
                    *exported,
                    *[
                        os.path.join(insight_dir, name)
                        for name in os.listdir(insight_dir)
                        if name.endswith(".png")
                    ],
                }
            )
        except Exception as exc:
            payload["errors"].append(f"insight_index[{theme_name}] failed: {exc}")

    payload["presentation"] = presentation_to_dict(presentation_spec)
    return payload


def generate_plots(csv_path: str, out_dir: str = None, config: dict = None) -> None:
    """
    Generate individual metric plots AND the unified dashboard.

    Args:
        csv_path : Path to the metrics CSV produced by the trainer / evaluator.
        out_dir  : Output directory. Defaults to the same directory as the CSV.
        config   : Optional config dict for dashboard header. When passed,
                   the header displays model, LR, LoRA, and eval settings.
    """
    from utils.evaluation_artifacts import generate_evaluation_artifacts

    run_root = out_dir or os.path.dirname(os.path.abspath(csv_path))
    manifest = generate_evaluation_artifacts(
        metrics_csv_path=csv_path,
        run_root=run_root,
        config=config,
        context="standalone",
    )
    bundle = manifest.get("artifact_bundle", "")
    if bundle:
        print(f"Artifacts generated in: {bundle}")


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
