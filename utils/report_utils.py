"""
Shared evaluation reporting helpers for dashboard and HTML report generation.

This module is intentionally presentation-oriented: it detects the evaluation
usecase, selects the most convincing KPIs for that usecase, generates short
analytical takeaways, and builds a structured presentation spec that both the
PNG dashboard and report.html can render without guessing from filenames.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


DARK: Dict[str, Any] = {
    "fig_bg": "#0B1020",
    "panel_bg": "#101826",
    "card_bg": "#141E2E",
    "text": "#F6F8FF",
    "subtext": "#A8B4CB",
    "muted": "#7E8AA5",
    "border": "#27324A",
    "grid": "#243149",
    "positive": "#78D686",
    "negative": "#FF7F79",
    "neutral": "#86B7FF",
    "warning": "#F7C97B",
    "lines": ["#4E9DFF", "#9C87E2", "#F7C97B", "#B084EB", "#33C6A6"],
    "line_map": {
        "accuracy": "#59C08A",
        "mcc": "#4E9DFF",
        "f1": "#9C87E2",
        "kappa": "#B084EB",
        "perplexity": "#F7C97B",
        "eval_loss": "#F08AA5",
        "loss": "#F08AA5",
        "forgetting": "#FF9770",
        "coverage": "#33C6A6",
        "consistency": "#A48DFF",
        "secondary": "#FF6B6B",
        "response": "#F4A261",
        "length": "#86B7FF",
        "diversity": "#6DD3CE",
        "semantic": "#6DD3CE",
        "keyword": "#78D686",
    },
}

LIGHT: Dict[str, Any] = {
    "fig_bg": "#F6F8FC",
    "panel_bg": "#FFFFFF",
    "card_bg": "#FFFFFF",
    "text": "#142033",
    "subtext": "#51607A",
    "muted": "#6D7A92",
    "border": "#DCE3EF",
    "grid": "#E7ECF5",
    "positive": "#1C8F52",
    "negative": "#D14A4A",
    "neutral": "#236CE5",
    "warning": "#B87400",
    "lines": ["#236CE5", "#7757D8", "#B87400", "#8D63D5", "#0B9E7C"],
    "line_map": {
        "accuracy": "#1C8F52",
        "mcc": "#236CE5",
        "f1": "#7757D8",
        "kappa": "#8D63D5",
        "perplexity": "#B87400",
        "eval_loss": "#CA4F7B",
        "loss": "#CA4F7B",
        "forgetting": "#D86A34",
        "coverage": "#0B9E7C",
        "consistency": "#6B57D8",
        "secondary": "#D14A4A",
        "response": "#C27400",
        "length": "#236CE5",
        "diversity": "#16858A",
        "semantic": "#16858A",
        "keyword": "#1C8F52",
    },
}


@dataclass
class KPI:
    label: str
    value: float
    unit: str = ""
    delta: Optional[float] = None
    delta_label: str = ""
    direction: str = "higher_better"
    comparison_basis: str = ""
    comparison_value: Optional[float] = None
    delta_display: str = ""
    status: str = "neutral"
    supporting_caption: str = ""
    metric_key: str = ""
    source_kind: str = ""
    source_step: Optional[int] = None
    source_note: str = ""


@dataclass
class TakeawayCard:
    title: str
    badge: str
    body: str
    tone: str = "neutral"


@dataclass
class ChartTrace:
    name: str
    x: List[float]
    y: List[float]
    axis: str = "primary"
    trace_type: str = "line"
    style: str = "solid"
    color_key: str = "neutral"
    fill: bool = False


@dataclass
class ThresholdLine:
    value: float
    label: str
    axis: str = "primary"
    style: str = "dashed"
    color_key: str = "muted"


@dataclass
class ChartSpec:
    id: str
    section: str
    role: str
    title: str
    subtitle: str
    chart_type: str
    preferred_aspect: str = "wide"
    traces: List[ChartTrace] = field(default_factory=list)
    thresholds: List[ThresholdLine] = field(default_factory=list)
    y_label: str = ""
    y2_label: str = ""
    x_label: str = "Step"
    fallback_paths: Dict[str, str] = field(default_factory=dict)
    note: str = ""


@dataclass
class SectionSpec:
    id: str
    title: str
    description: str
    charts: List[ChartSpec] = field(default_factory=list)


@dataclass
class UsecaseProfile:
    slug: str
    label: str
    strategy: str
    method: str
    evidence_label: str

    @property
    def name(self) -> str:
        return self.slug


@dataclass
class PresentationSpec:
    header: Dict[str, Any]
    kpi_cards: List[KPI]
    takeaway_cards: List[TakeawayCard]
    sections: List[SectionSpec]
    chart_specs: List[ChartSpec]
    empty_states: List[Dict[str, str]] = field(default_factory=list)
    details_blocks: List[Dict[str, str]] = field(default_factory=list)
    profile: Optional[UsecaseProfile] = None


def _safe_float(value: Any) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_step(value: Any) -> Optional[int]:
    if value in ("", None, "final"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _direct_row_plot_step(row: Dict[str, Any]) -> Optional[int]:
    eval_step = _parse_step(row.get("eval_step"))
    if eval_step is not None:
        return eval_step
    return _parse_step(row.get("step"))


def _infer_final_plot_step(rows: Sequence[Dict[str, Any]]) -> Optional[int]:
    numeric_steps = [_direct_row_plot_step(row) for row in rows]
    numeric_steps = [step for step in numeric_steps if step is not None]
    if not numeric_steps:
        return None
    if len(numeric_steps) == 1:
        return numeric_steps[-1] + 1
    deltas = [
        curr - prev
        for prev, curr in zip(numeric_steps, numeric_steps[1:])
        if curr > prev
    ]
    step_size = deltas[-1] if deltas else 1
    return numeric_steps[-1] + max(1, step_size)


def _row_plot_step(row: Dict[str, Any], rows: Optional[Sequence[Dict[str, Any]]] = None) -> Optional[int]:
    """Return the numeric x-axis step for a row, preferring eval_step when present."""
    direct = _direct_row_plot_step(row)
    if direct is not None:
        return direct
    if rows is not None and str(row.get("step", "")).strip().lower() == "final":
        return _infer_final_plot_step(rows)
    return None


def _numeric_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if _row_plot_step(row, rows) is not None]


def _summary_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if str(row.get("step", "")).strip().lower() == "final"]


def _initial_numeric_value(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    for row in _numeric_rows(rows):
        value = _safe_float(row.get(key))
        if value is not None:
            return value
    return _initial_value(rows, key)


def _final_checkpoint_row(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    summaries = _summary_rows(rows)
    if summaries:
        return summaries[-1]
    numeric = _numeric_rows(rows)
    return numeric[-1] if numeric else (rows[-1] if rows else None)


def _final_checkpoint_step(rows: Sequence[Dict[str, Any]]) -> Optional[int]:
    row = _final_checkpoint_row(rows)
    return _row_plot_step(row or {}, rows)


def _final_checkpoint_value(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    row = _final_checkpoint_row(rows)
    value = _safe_float((row or {}).get(key))
    if value is not None:
        return value
    return _final_value(rows, key)


def _summary_value(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    for row in reversed(_summary_rows(rows)):
        value = _safe_float(row.get(key))
        if value is not None:
            return value
    return None


def _metric_series(rows: Sequence[Dict[str, Any]], key: str) -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    for row in rows:
        step = _row_plot_step(row, rows)
        value = _safe_float(row.get(key))
        if step is None or value is None:
            continue
        xs.append(step)
        ys.append(value)
    return xs, ys


def _final_value(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    for row in reversed(rows):
        value = _safe_float(row.get(key))
        if value is not None:
            return value
    return None


def _initial_value(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    for row in rows:
        value = _safe_float(row.get(key))
        if value is not None:
            return value
    return None


def _peak_value(rows: Sequence[Dict[str, Any]], key: str, higher_better: bool = True) -> Tuple[Optional[float], Optional[int]]:
    xs, ys = _metric_series(rows, key)
    if not ys:
        return None, None
    best_index = max(range(len(ys)), key=lambda idx: ys[idx]) if higher_better else min(range(len(ys)), key=lambda idx: ys[idx])
    return ys[best_index], xs[best_index]


def _step_when(rows: Sequence[Dict[str, Any]], key: str, predicate) -> Optional[int]:
    xs, ys = _metric_series(rows, key)
    for step, value in zip(xs, ys):
        if predicate(value):
            return step
    return None


def _tail_window(values: Sequence[float], frac: float = 0.25) -> Sequence[float]:
    if not values:
        return []
    window = max(3, int(math.ceil(len(values) * frac)))
    return values[-window:]


def _tail_plateau(values: Sequence[float], tolerance: float) -> bool:
    tail = list(_tail_window(values))
    if len(tail) < 3:
        return False
    return (max(tail) - min(tail)) <= tolerance


def _pp(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return value * 100.0 if abs(value) <= 1.0 else value


def _status_from_delta(delta: Optional[float], direction: str) -> str:
    if delta is None:
        return "neutral"
    if direction == "lower_better":
        if delta < 0:
            return "good"
        if delta > 0:
            return "bad"
        return "neutral"
    if delta > 0:
        return "good"
    if delta < 0:
        return "bad"
    return "neutral"


def _fmt_value(value: Optional[float], unit: str = "") -> str:
    if value is None:
        return ""
    if abs(value) < (0.05 if unit == "%" else 0.0005):
        value = 0.0
    if unit == "%":
        return f"{value:.1f}%"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _fmt_plain_value(value: Any) -> str:
    if value in (None, "", []):
        return ""
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _fmt_lr(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric == 0:
        return "0"
    if abs(numeric) < 0.001 or abs(numeric) >= 1000:
        return f"{numeric:.1e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"{numeric:g}"


def _join_non_empty(values: Sequence[str], sep: str = ", ") -> str:
    return sep.join(value for value in values if value)


def _delta_display(delta: Optional[float], unit: str = "%", direction: str = "higher_better", suffix: str = "vs step 0 baseline") -> str:
    if delta is None:
        return ""
    arrow = "↑" if delta >= 0 else "↓"
    magnitude = abs(delta)
    amount = f"{magnitude:.1f} percentage points" if unit == "%" else _fmt_value(magnitude, unit)
    return f"{arrow} {amount} {suffix}".strip()


def _find_per_slot_columns(rows: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
    if not rows:
        return []
    columns = []
    for key in rows[0].keys():
        if key.startswith("qual_slot_") and key.endswith("_coverage"):
            if key in {"qual_slot_coverage_mean", "qual_pinned_slot_coverage_mean"}:
                continue
            slot = key[len("qual_slot_"):-len("_coverage")].replace("_", " ")
            columns.append((key, slot))
    return columns


def _header_from_config(config: Optional[Dict[str, Any]], profile: UsecaseProfile) -> Dict[str, Any]:
    config = config or {}
    project = config.get("project") or {}
    model = config.get("model") or {}
    lora = config.get("lora") or {}
    training = config.get("training") or {}
    evaluation = config.get("evaluation") or {}
    testing = config.get("testing_strategy") or {}
    dataset = config.get("dataset") or {}

    batch = training.get("batch_size")
    grad_acc = training.get("gradient_accumulation_steps")
    effective_batch = None
    try:
        if batch not in (None, "") and grad_acc not in (None, ""):
            effective_batch = int(batch) * int(grad_acc)
    except (TypeError, ValueError):
        effective_batch = None

    target_modules = lora.get("target_modules") or []
    if isinstance(target_modules, (list, tuple)):
        targets_value = ", ".join(str(item) for item in target_modules[:4])
        if len(target_modules) > 4:
            targets_value += f" +{len(target_modules) - 4}"
    else:
        targets_value = str(target_modules) if target_modules else ""

    chips = []
    if project.get("name"):
        chips.append(project["name"])
    if dataset.get("name"):
        chips.append(str(dataset["name"]))
    if model.get("name"):
        chips.append(model["name"])
    if model.get("precision"):
        chips.append(str(model["precision"]).upper())
    if training.get("learning_rate") not in (None, ""):
        chips.append(f"LR {_fmt_lr(training['learning_rate'])}")
    if training.get("max_steps") not in (None, ""):
        chips.append(f"{training['max_steps']} steps")
    if lora.get("r") not in (None, ""):
        chips.append(f"LoRA r={lora['r']}")
    if evaluation.get("strategy"):
        chips.append(f"Eval {evaluation['strategy']}")
    if testing.get("method"):
        chips.append(f"Qual {testing['method']}")

    overview_items = [
        {"label": "Usecase", "value": profile.label},
        {"label": "Dataset", "value": _fmt_plain_value(dataset.get("name")) or _fmt_plain_value(project.get("name"))},
        {"label": "Model", "value": _fmt_plain_value(model.get("name"))},
        {"label": "Precision", "value": str(model.get("precision", "")).upper() if model.get("precision") else ""},
        {"label": "Max Sequence Length", "value": _fmt_plain_value(model.get("max_seq_length"))},
    ]
    training_items = [
        {"label": "Learning Rate", "value": _fmt_lr(training.get("learning_rate"))},
        {"label": "Optimizer Steps", "value": _fmt_plain_value(training.get("max_steps"))},
        {"label": "Batch Size", "value": _fmt_plain_value(training.get("batch_size"))},
        {"label": "Effective Batch", "value": _fmt_plain_value(effective_batch)},
        {"label": "Scheduler", "value": _fmt_plain_value((training.get("lr_scheduler") or {}).get("type"))},
        {"label": "Warmup Steps", "value": _fmt_plain_value((training.get("lr_scheduler") or {}).get("warmup_steps"))},
    ]
    adaptation_items = [
        {"label": "LoRA Rank", "value": _fmt_plain_value(lora.get("r"))},
        {"label": "LoRA Alpha", "value": _fmt_plain_value(lora.get("alpha"))},
        {"label": "LoRA Dropout", "value": _fmt_plain_value(lora.get("dropout"))},
        {"label": "Target Modules", "value": targets_value},
    ]
    evaluation_items = [
        {"label": "Primary Eval Strategy", "value": _fmt_plain_value(evaluation.get("strategy"))},
        {"label": "Qualitative Method", "value": _fmt_plain_value(testing.get("method"))},
        {"label": "Eval Interval", "value": _fmt_plain_value(evaluation.get("eval_interval") or testing.get("eval_interval"))},
        {"label": "Eval Pool Size", "value": _fmt_plain_value(evaluation.get("eval_pool_size") or testing.get("eval_pool_size"))},
        {"label": "Eval Samples", "value": _fmt_plain_value(testing.get("eval_samples"))},
        {"label": "Consistency Runs", "value": _fmt_plain_value(testing.get("consistency_runs"))},
    ]

    metadata_cards = [
        {"title": "Run Overview", "description": "What this run trained and what the report is evaluating.", "items": [item for item in overview_items if item["value"]]},
        {"title": "Training Setup", "description": "The main optimization settings used during fine-tuning.", "items": [item for item in training_items if item["value"]]},
        {"title": "Adapter Setup", "description": "LoRA settings that control how the base model was adapted.", "items": [item for item in adaptation_items if item["value"]]},
        {"title": "Evaluation Setup", "description": "How checkpoints were measured and what signals were collected.", "items": [item for item in evaluation_items if item["value"]]},
    ]

    return {
        "product": "InfiniTune",
        "title": f"{profile.label} Evaluation",
        "subtitle": f"Usecase-aware evaluation story focused on {profile.evidence_label}.",
        "chips": chips,
        "cards": [card for card in metadata_cards if card["items"]],
        "meta": {
            "run_name": project.get("name", ""),
            "model": model.get("name", ""),
            "strategy": profile.strategy,
            "method": profile.method,
        },
    }


def detect_usecase(rows: Sequence[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> UsecaseProfile:
    config = config or {}
    evaluation = config.get("evaluation") or {}
    testing = config.get("testing_strategy") or {}
    dataset = config.get("dataset") or {}

    strategy = str(evaluation.get("strategy") or "").strip().lower()
    method = str(testing.get("method") or "").strip().lower()
    dataset_name = str(dataset.get("name") or "").strip().lower()
    columns = set(rows[0].keys()) if rows else set()

    if method == "structured_slot_coverage" or "qual_slot_coverage_mean" in columns or any(col.startswith("qual_slot_") for col in columns):
        return UsecaseProfile("structured_nlg", "Structured NLG", strategy or "perplexity", method or "structured_slot_coverage", "slot coverage and consistency learning")
    if method == "structural_cot" or any(col.startswith("qual_cot_") for col in columns):
        return UsecaseProfile("math_reasoning_cot", "Math Reasoning (CoT)", strategy or "regex_extract", method or "structural_cot", "reasoning structure adoption")
    if method == "semantic_similarity" or "qual_semantic_similarity" in columns:
        return UsecaseProfile("instruction_following_semantic", "Instruction Following", strategy or "perplexity", method or "semantic_similarity", "semantic alignment and response quality")
    if method == "keyword_density" or "qual_keyword_density" in columns:
        return UsecaseProfile("domain_adaptation_keyword", "Domain Adaptation", strategy or "perplexity", method or "keyword_density", "domain keyword adoption")
    if strategy == "regex_extract" or dataset_name == "gsm8k" or "answer_overlap_f1" in columns:
        return UsecaseProfile("math_reasoning_quant", "Math Reasoning", strategy or "regex_extract", method or "quantitative", "correct final-answer learning")
    if strategy == "class_match" or any(col in columns for col in ("accuracy", "mcc", "kappa")):
        return UsecaseProfile("classification", "Classification", strategy or "class_match", method or "quantitative", "classification accuracy and class balance")
    return UsecaseProfile("generic", "General Evaluation", strategy or "generic", method or "generic", "learning progress")


def _classification_kpis(rows: Sequence[Dict[str, Any]]) -> List[KPI]:
    return _classification_kpis_v2(rows)

    peak_acc, peak_acc_step = _peak_value(rows, "accuracy", higher_better=True)
    peak_mcc, peak_mcc_step = _peak_value(rows, "mcc", higher_better=True)
    final_kappa = _pp(_final_value(rows, "kappa"))
    base_acc = _pp(_initial_value(rows, "accuracy"))
    peak_acc_pp = _pp(peak_acc)
    peak_mcc_base = _initial_value(rows, "mcc") or 0.0
    forget_peak, forget_peak_step = _peak_value(rows, "forgetting_accuracy", higher_better=True)
    if forget_peak is None:
        forget_peak, forget_peak_step = _peak_value(rows, "forgetting_max", higher_better=True)

    acc70_step = _step_when(rows, "accuracy", lambda value: value >= 0.70)
    final_acc = _pp(_final_value(rows, "accuracy"))

    kpis: List[KPI] = []
    if peak_acc_pp is not None:
        delta = None if base_acc is None else peak_acc_pp - base_acc
        kpis.append(
            KPI(
                label="Peak Accuracy",
                value=peak_acc_pp,
                unit="%",
                delta=delta,
                delta_label="vs step 0 baseline",
                direction="higher_better",
                comparison_basis="step 0 baseline",
                comparison_value=base_acc,
                delta_display=f"{'↑' if (delta or 0) >= 0 else '↓'} {abs(delta or 0):.1f} percentage points vs step 0 baseline" if delta is not None else "",
                status=_status_from_delta(delta, "higher_better"),
                supporting_caption=f"peaks at step {peak_acc_step}" if peak_acc_step is not None else "",
            )
        )
    if peak_mcc is not None:
        delta = peak_mcc - peak_mcc_base
        kpis.append(
            KPI(
                label="Peak MCC",
                value=peak_mcc,
                unit="",
                delta=delta,
                delta_label="vs step 0 baseline",
                direction="higher_better",
                comparison_basis="step 0 baseline",
                comparison_value=peak_mcc_base,
                delta_display=f"{'↑' if delta >= 0 else '↓'} {abs(delta):.3f} vs step 0 baseline",
                status=_status_from_delta(delta, "higher_better"),
                supporting_caption=f"crosses strongest class-balance point at step {peak_mcc_step}" if peak_mcc_step is not None else "",
            )
        )
    if acc70_step is not None:
        kpis.append(
            KPI(
                label="Sample Efficiency",
                value=float(acc70_step),
                unit="",
                delta=-float(acc70_step),
                delta_label="to reach 70% accuracy",
                direction="lower_better",
                comparison_basis="optimization steps",
                comparison_value=0.0,
                delta_display=f"reaches 70% at step {acc70_step}",
                status="good",
                supporting_caption="speed-to-learning milestone",
            )
        )
    if forget_peak is not None:
        forget_pp = _pp(forget_peak)
        kpis.append(
            KPI(
                label="Forgetting",
                value=forget_pp or 0.0,
                unit="%",
                delta=forget_pp,
                delta_label="peak late-stage drift",
                direction="lower_better",
                comparison_basis="zero drift",
                comparison_value=0.0,
                delta_display=f"{abs(forget_pp or 0):.1f} percentage-point max drift",
                status="good" if (forget_pp or 0) <= 7 else "warning" if (forget_pp or 0) <= 12 else "bad",
                supporting_caption=f"worst forgetting appears near step {forget_peak_step}" if forget_peak_step is not None else "retention stability proxy",
            )
        )
    if final_acc is not None:
        delta = None if base_acc is None else final_acc - base_acc
        kpis.append(
            KPI(
                label="Final Accuracy",
                value=final_acc,
                unit="%",
                delta=delta,
                delta_label="vs step 0 baseline",
                comparison_basis="step 0 baseline",
                comparison_value=base_acc,
                delta_display=f"{'↑' if (delta or 0) >= 0 else '↓'} {abs(delta or 0):.1f} percentage points to final",
                status=_status_from_delta(delta, "higher_better"),
                supporting_caption="final checkpoint performance",
            )
        )
    if final_kappa is not None:
        kpis.append(
            KPI(
                label="Final Kappa",
                value=final_kappa,
                unit="%",
                direction="higher_better",
                comparison_basis="agreement score",
                delta_display="agreement with labels at final checkpoint",
                status="good" if final_kappa >= 60 else "neutral",
                supporting_caption="class-agreement stability",
            )
        )
    return kpis


def _classification_kpis_v2(rows: Sequence[Dict[str, Any]]) -> List[KPI]:
    peak_acc, peak_acc_step = _peak_value(rows, "accuracy", higher_better=True)
    peak_mcc, peak_mcc_step = _peak_value(rows, "mcc", higher_better=True)
    final_step = _final_checkpoint_step(rows)
    final_kappa = _pp(_final_checkpoint_value(rows, "kappa"))
    base_acc = _pp(_initial_numeric_value(rows, "accuracy"))
    base_kappa = _pp(_initial_numeric_value(rows, "kappa"))
    peak_acc_pp = _pp(peak_acc)
    peak_mcc_base = _initial_numeric_value(rows, "mcc") or 0.0
    final_acc = _pp(_final_checkpoint_value(rows, "accuracy"))
    final_kappa_raw = _final_checkpoint_value(rows, "kappa")
    forget_peak, forget_peak_step = _peak_value(rows, "forgetting_accuracy", higher_better=True)
    forget_metric_key = "forgetting_accuracy"
    if forget_peak is None:
        forget_peak, forget_peak_step = _peak_value(rows, "forgetting_max", higher_better=True)
        forget_metric_key = "forgetting_max"

    kpis: List[KPI] = []
    if peak_acc_pp is not None:
        delta = None if base_acc is None else peak_acc_pp - base_acc
        kpis.append(
            KPI(
                label="Peak Accuracy",
                value=peak_acc_pp,
                unit="%",
                delta=delta,
                delta_label="vs step 0 baseline",
                direction="higher_better",
                comparison_basis="step 0 baseline",
                comparison_value=base_acc,
                delta_display=_delta_display(delta) if delta is not None else "",
                status=_status_from_delta(delta, "higher_better"),
                supporting_caption=f"peaks at step {peak_acc_step}" if peak_acc_step is not None else "",
                metric_key="accuracy",
                source_kind="peak",
                source_step=peak_acc_step,
            )
        )
    if peak_mcc is not None:
        delta = peak_mcc - peak_mcc_base
        kpis.append(
            KPI(
                label="Peak MCC",
                value=peak_mcc,
                unit="",
                delta=delta,
                delta_label="vs step 0 baseline",
                direction="higher_better",
                comparison_basis="step 0 baseline",
                comparison_value=peak_mcc_base,
                delta_display=f"{'↑' if delta >= 0 else '↓'} {abs(delta):.3f} vs step 0 baseline",
                status=_status_from_delta(delta, "higher_better"),
                supporting_caption=f"crosses strongest class-balance point at step {peak_mcc_step}" if peak_mcc_step is not None else "",
                metric_key="mcc",
                source_kind="peak",
                source_step=peak_mcc_step,
            )
        )
    if final_acc is not None:
        delta = None if base_acc is None else final_acc - base_acc
        final_basis = f"final checkpoint row (eval step {final_step})" if final_step is not None else "final checkpoint row"
        kpis.append(
            KPI(
                label="Final Checkpoint Accuracy",
                value=final_acc,
                unit="%",
                delta=delta,
                delta_label="vs step 0 baseline",
                comparison_basis=f"step 0 baseline = {_fmt_value(base_acc, '%')}" if base_acc is not None else "baseline unavailable",
                comparison_value=base_acc,
                delta_display=_delta_display(delta) if delta is not None else "accuracy at the final checkpoint",
                status=_status_from_delta(delta, "higher_better"),
                supporting_caption=f"value taken directly from the {final_basis}",
                metric_key="accuracy",
                source_kind="final_checkpoint",
                source_step=final_step,
                source_note=final_basis,
            )
        )
    if final_kappa is not None:
        final_basis = f"final checkpoint row (eval step {final_step})" if final_step is not None else "final checkpoint row"
        delta = None if base_kappa is None or final_kappa_raw is None else final_kappa - base_kappa
        kpis.append(
            KPI(
                label="Final Checkpoint Kappa",
                value=final_kappa,
                unit="%",
                delta=delta,
                delta_label="vs step 0 baseline",
                direction="higher_better",
                comparison_basis=f"step 0 baseline = {_fmt_value(base_kappa, '%')}" if base_kappa is not None else "baseline unavailable",
                comparison_value=base_kappa,
                delta_display=_delta_display(delta) if delta is not None else "agreement score at the final checkpoint",
                status="good" if final_kappa >= 60 else "neutral",
                supporting_caption=f"value taken directly from the {final_basis}",
                metric_key="kappa",
                source_kind="final_checkpoint",
                source_step=final_step,
                source_note=final_basis,
            )
        )
    if forget_peak is not None:
        forget_pp = _pp(forget_peak)
        kpis.append(
            KPI(
                label="Forgetting",
                value=forget_pp or 0.0,
                unit="%",
                delta=forget_pp,
                delta_label="peak late-stage drift",
                direction="lower_better",
                comparison_basis="zero drift",
                comparison_value=0.0,
                delta_display=f"{abs(forget_pp or 0):.1f} percentage-point max drift",
                status="good" if (forget_pp or 0) <= 7 else "warning" if (forget_pp or 0) <= 12 else "bad",
                supporting_caption=f"worst forgetting appears near step {forget_peak_step}" if forget_peak_step is not None else "retention stability proxy",
                metric_key=forget_metric_key,
                source_kind="max",
                source_step=forget_peak_step,
            )
        )
    return kpis


def _math_quant_kpis(rows: Sequence[Dict[str, Any]]) -> List[KPI]:
    peak_em, peak_em_step = _peak_value(rows, "exact_match", higher_better=True)
    peak_acc, peak_acc_step = _peak_value(rows, "accuracy", higher_better=True)
    base_em = _pp(_initial_numeric_value(rows, "exact_match") or _initial_numeric_value(rows, "accuracy"))
    final_ppl = _final_checkpoint_value(rows, "perplexity")
    start_ppl = _initial_numeric_value(rows, "perplexity")
    first_correct_step = _step_when(rows, "exact_match", lambda value: value > 0) or _step_when(rows, "accuracy", lambda value: value > 0)
    answer_overlap = _pp(_final_checkpoint_value(rows, "answer_overlap_f1"))

    kpis: List[KPI] = []
    best_score = peak_em if peak_em is not None else peak_acc
    best_step = peak_em_step if peak_em is not None else peak_acc_step
    if best_score is not None:
        best_pp = _pp(best_score)
        delta = None if base_em is None else best_pp - base_em
        kpis.append(
            KPI(
                label="Peak Exact Match" if peak_em is not None else "Peak Accuracy",
                value=best_pp or 0.0,
                unit="%",
                delta=delta,
                delta_label="vs step 0 baseline",
                comparison_basis="step 0 baseline",
                comparison_value=base_em,
                delta_display=f"{'↑' if (delta or 0) >= 0 else '↓'} {abs(delta or 0):.1f} percentage points vs step 0 baseline",
                status=_status_from_delta(delta, "higher_better"),
                supporting_caption=f"peaks at step {best_step}" if best_step is not None else "",
            )
        )
    if first_correct_step is not None:
        kpis.append(
            KPI(
                label="First Correct Step",
                value=float(first_correct_step),
                unit="",
                delta=-float(first_correct_step),
                delta_label="to first non-zero correctness",
                direction="lower_better",
                comparison_basis="optimization steps",
                comparison_value=0.0,
                delta_display=f"first non-zero correctness at step {first_correct_step}",
                status="good",
                supporting_caption="answer-format adoption milestone",
            )
        )
    if start_ppl is not None and final_ppl is not None:
        delta = final_ppl - start_ppl
        kpis.append(
            KPI(
                label="Final Perplexity",
                value=final_ppl,
                unit="",
                delta=delta,
                delta_label="vs step 0 baseline",
                direction="lower_better",
                comparison_basis="step 0 baseline",
                comparison_value=start_ppl,
                delta_display=f"{'↓' if delta <= 0 else '↑'} {abs(delta):.2f} vs step 0 baseline",
                status=_status_from_delta(delta, "lower_better"),
                supporting_caption="language-model confidence on eval set",
            )
        )
    if answer_overlap is not None:
        kpis.append(
            KPI(
                label="Final Answer Overlap",
                value=answer_overlap,
                unit="%",
                direction="higher_better",
                comparison_basis="token overlap",
                delta_display="final token-level answer similarity",
                status="good" if answer_overlap >= 50 else "neutral",
                supporting_caption="useful when exact match is sparse",
            )
        )
    return kpis


def _math_cot_kpis(rows: Sequence[Dict[str, Any]]) -> List[KPI]:
    anchor = _pp(_final_checkpoint_value(rows, "qual_cot_coverage") or _final_checkpoint_value(rows, "qual_cot_anchor_coverage"))
    anchor_count = _final_checkpoint_value(rows, "qual_cot_anchor_count")
    step_length = _final_checkpoint_value(rows, "qual_cot_step_length")
    exact_match = _pp(_final_checkpoint_value(rows, "exact_match"))
    non_empty = _pp(_final_checkpoint_value(rows, "qual_non_empty_rate"))
    return [
        KPI("CoT Coverage", anchor or 0.0, "%", comparison_basis="anchor coverage", delta_display="final reasoning-anchor coverage", status="good" if (anchor or 0) >= 60 else "neutral", supporting_caption="structured reasoning adoption"),
        KPI("Anchor Count", anchor_count or 0.0, "", comparison_basis="mean anchors", delta_display="mean reasoning anchors per response", status="good" if (anchor_count or 0) >= 2 else "neutral", supporting_caption="not just short answers"),
        KPI("Step Length", step_length or 0.0, "", comparison_basis="tokens per step", delta_display="intermediate reasoning depth", status="good" if (step_length or 0) >= 6 else "neutral", supporting_caption="reasoning content density"),
        KPI("Final Exact Match", exact_match or 0.0, "%", comparison_basis="answer correctness", delta_display="final numeric correctness", status="good" if (exact_match or 0) > 0 else "neutral", supporting_caption="links structure to correctness"),
        KPI("Non-empty Rate", non_empty or 0.0, "%", comparison_basis="response health", delta_display="responses produced at evaluation time", status="good" if (non_empty or 0) >= 95 else "warning", supporting_caption="guards against silent failures"),
    ]


def _structured_nlg_kpis(rows: Sequence[Dict[str, Any]]) -> List[KPI]:
    coverage = _pp(_final_checkpoint_value(rows, "qual_slot_coverage_mean"))
    coverage_base = _pp(_initial_numeric_value(rows, "qual_slot_coverage_mean"))
    pinned = _structured_pinned_value(rows, final=True)
    pinned_base = _structured_pinned_value(rows, final=False)
    consistency = _pp(_final_checkpoint_value(rows, "qual_consistency_score_mean"))
    consistency_base = _pp(_initial_numeric_value(rows, "qual_consistency_score_mean"))
    inversion = _structured_inversion_value(rows, final=True)
    inversion_base = _structured_inversion_value(rows, final=False)
    per_slot = [(label, _pp(_final_checkpoint_value(rows, key))) for key, label in _find_per_slot_columns(rows)]
    weakest_slot = min((item for item in per_slot if item[1] is not None), key=lambda item: item[1], default=None)
    sorted_slots = sorted((item for item in per_slot if item[1] is not None), key=lambda item: item[1], reverse=True)
    next_lowest_gap = None
    if weakest_slot and len(sorted_slots) > 1:
        next_lowest_gap = sorted_slots[-2][1] - weakest_slot[1]

    kpis: List[KPI] = []
    if coverage is not None:
        delta = None if coverage_base is None else coverage - coverage_base
        kpis.append(KPI("Mean Slot Coverage", coverage, "%", delta=delta, comparison_basis=f"step 0 baseline = {_fmt_value(coverage_base, '%')}" if coverage_base is not None else "baseline unavailable", delta_display=_delta_display(delta) or "final slot coverage across tracked fields", status="good" if coverage >= 85 else "neutral", supporting_caption="final slot coverage across tracked fields"))
    if pinned is not None:
        delta = None if pinned_base is None else pinned - pinned_base
        kpis.append(KPI("Pinned Consistency", pinned, "%", delta=delta, comparison_basis=f"step 0 baseline = {_fmt_value(pinned_base, '%')}" if pinned_base is not None else "baseline unavailable", delta_display=_delta_display(delta) or "repeated-anchor consistency at final checkpoint", status="good" if pinned >= 75 else "neutral", supporting_caption="same-prompt consistency across pinned anchors"))
    if inversion is not None:
        delta = None if inversion_base is None else inversion - inversion_base
        kpis.append(KPI("Inversion Rate", inversion, "%", delta=delta, direction="lower_better", comparison_basis=f"step 0 baseline = {_fmt_value(inversion_base, '%')}" if inversion_base is not None else "baseline unavailable", delta_display=_delta_display(delta, direction="lower_better") or "lower is better", status="good" if inversion <= 3 else "warning" if inversion <= 10 else "bad", supporting_caption="lower means the slot meaning is verbalized correctly"))
    if weakest_slot and weakest_slot[1] is not None:
        delta_display = "lowest final slot coverage"
        comparison_basis = "final per-slot comparison"
        if next_lowest_gap is not None:
            delta_display = f"↓ {next_lowest_gap:.1f} percentage points below next-lowest slot"
            comparison_basis = "compared with the next-weakest tracked slot"
        kpis.append(KPI(f"Weakest Slot ({weakest_slot[0]})", weakest_slot[1], "%", delta=(-(next_lowest_gap or 0.0)) if next_lowest_gap is not None else None, comparison_basis=comparison_basis, delta_display=delta_display, status="warning" if weakest_slot[1] < 80 else "neutral", supporting_caption="structural outlier for targeted follow-up"))
    if consistency is not None:
        delta = None if consistency_base is None else consistency - consistency_base
        kpis.append(KPI("Pool Consistency", consistency, "%", delta=delta, comparison_basis=f"step 0 baseline = {_fmt_value(consistency_base, '%')}" if consistency_base is not None else "baseline unavailable", delta_display=_delta_display(delta) or "full-pool consistency", status="good" if consistency >= 70 else "neutral", supporting_caption="consistency across the broader evaluation pool"))
    return kpis


def _semantic_kpis(rows: Sequence[Dict[str, Any]]) -> List[KPI]:
    similarity = _pp(_final_checkpoint_value(rows, "qual_semantic_similarity"))
    non_empty = _pp(_final_checkpoint_value(rows, "qual_non_empty_rate"))
    response_len = _final_checkpoint_value(rows, "qual_mean_response_length")
    repetition = _pp(_final_checkpoint_value(rows, "qual_repetition_rate"))
    perplexity = _final_checkpoint_value(rows, "perplexity")
    return [
        KPI("Semantic Similarity", similarity or 0.0, "%", comparison_basis="semantic similarity", delta_display="final semantic match to references", status="good" if (similarity or 0) >= 70 else "neutral", supporting_caption="primary task-fit signal"),
        KPI("Non-empty Rate", non_empty or 0.0, "%", comparison_basis="response health", delta_display="responses produced", status="good" if (non_empty or 0) >= 95 else "warning", supporting_caption="guards against degeneration"),
        KPI("Response Length", response_len or 0.0, "", comparison_basis="mean tokens", delta_display="mean response length", status="neutral", supporting_caption="helps interpret similarity"),
        KPI("Repetition Rate", repetition or 0.0, "%", direction="lower_better", comparison_basis="repetition", delta_display="lower is better", status="good" if (repetition or 0) <= 10 else "warning", supporting_caption="checks verbosity quality"),
        KPI("Perplexity", perplexity or 0.0, "", direction="lower_better", comparison_basis="eval perplexity", delta_display="final perplexity", status="good" if perplexity and perplexity <= 20 else "neutral", supporting_caption="language-model confidence"),
    ]


def _keyword_kpis(rows: Sequence[Dict[str, Any]]) -> List[KPI]:
    density = _pp(_final_checkpoint_value(rows, "qual_keyword_density"))
    ttr = _pp(_final_checkpoint_value(rows, "qual_type_token_ratio"))
    hapax = _pp(_final_checkpoint_value(rows, "qual_hapax_ratio"))
    non_empty = _pp(_final_checkpoint_value(rows, "qual_non_empty_rate"))
    repetition = _pp(_final_checkpoint_value(rows, "qual_repetition_rate"))
    return [
        KPI("Keyword Density", density or 0.0, "%", comparison_basis="domain keyword density", delta_display="final in-domain keyword adoption", status="good" if (density or 0) >= 10 else "neutral", supporting_caption="specialization signal"),
        KPI("Type/Token Ratio", ttr or 0.0, "%", comparison_basis="lexical diversity", delta_display="final diversity", status="good" if (ttr or 0) >= 25 else "neutral", supporting_caption="guards against rote templates"),
        KPI("Hapax Ratio", hapax or 0.0, "%", comparison_basis="rare-token ratio", delta_display="final long-tail lexical signal", status="neutral", supporting_caption="variation beyond repeated boilerplate"),
        KPI("Non-empty Rate", non_empty or 0.0, "%", comparison_basis="response health", delta_display="responses produced", status="good" if (non_empty or 0) >= 95 else "warning", supporting_caption="availability sanity check"),
        KPI("Repetition Rate", repetition or 0.0, "%", direction="lower_better", comparison_basis="repetition", delta_display="lower is better", status="good" if (repetition or 0) <= 10 else "warning", supporting_caption="checks degeneration"),
    ]


def _structured_pinned_value(rows: Sequence[Dict[str, Any]], final: bool = True) -> Optional[float]:
    resolver = _final_checkpoint_value if final else _initial_numeric_value
    primary = resolver(rows, "qual_pinned_consistency")
    fallback = resolver(rows, "qual_pinned_consistency_score")
    value = primary if primary is not None else fallback
    return _pp(value)


def _structured_inversion_value(rows: Sequence[Dict[str, Any]], final: bool = True) -> Optional[float]:
    resolver = _final_checkpoint_value if final else _initial_numeric_value
    for key in ("qual_familyFriendly_inversion", "qual_slot_familyFriendly_inversion_rate", "qual_slot_inversion_rate"):
        value = resolver(rows, key)
        if value is not None:
            return _pp(value)
    return None


def select_kpis(rows: Sequence[Dict[str, Any]], profile: UsecaseProfile) -> List[KPI]:
    if profile.slug == "classification":
        return _classification_kpis_v2(rows)
    if profile.slug == "math_reasoning_quant":
        return _math_quant_kpis(rows)
    if profile.slug == "math_reasoning_cot":
        return _math_cot_kpis(rows)
    if profile.slug == "structured_nlg":
        return _structured_nlg_kpis(rows)
    if profile.slug == "instruction_following_semantic":
        return _semantic_kpis(rows)
    if profile.slug == "domain_adaptation_keyword":
        return _keyword_kpis(rows)
    return _classification_kpis_v2(rows) if any(col in (rows[0].keys() if rows else []) for col in ("accuracy", "mcc")) else _semantic_kpis(rows)


def _classification_takeaways(rows: Sequence[Dict[str, Any]]) -> List[TakeawayCard]:
    _, acc = _metric_series(rows, "accuracy")
    _, mcc = _metric_series(rows, "mcc")
    _, ppl = _metric_series(rows, "perplexity")
    forgetting = _pp(_peak_value(rows, "forgetting_accuracy", higher_better=True)[0] or _peak_value(rows, "forgetting_max", higher_better=True)[0] or 0.0)
    acc_peak, acc_peak_step = _peak_value(rows, "accuracy", True)
    plateau = _tail_plateau(acc, 0.03)
    cards = []
    if acc_peak is not None:
        cards.append(TakeawayCard("Classification learning arc", "clean monotonic story" if plateau else "continued headroom", f"Accuracy peaks at {_fmt_value(_pp(acc_peak), '%')} around step {acc_peak_step or 0}, and the late-stage curve {'stabilizes cleanly' if plateau else 'still moves enough to justify more training'}.", "good" if plateau else "neutral"))
    if mcc:
        swing = max(mcc) - min(mcc)
        cards.append(TakeawayCard("Class balance signal", "balanced learning" if max(mcc) >= 0.5 else "weak calibration", f"MCC moves by {swing:.3f} over training, which is the best compact proof that the model is learning the label boundary rather than only the majority class.", "good" if max(mcc) >= 0.5 else "warning"))
    if ppl:
        cards.append(TakeawayCard("Task format acquisition", "fast adaptation" if ppl[0] > ppl[-1] else "watch divergence", f"Perplexity changes from {_fmt_value(ppl[0])} to {_fmt_value(ppl[-1])}, showing how quickly the model adapts to the evaluation prompt/label format.", "good" if ppl[0] > ppl[-1] else "warning"))
    if forgetting:
        cards.append(TakeawayCard("Retention stability", "retention healthy" if forgetting <= 7 else "retention risk", f"Maximum forgetting stays around {_fmt_value(forgetting, '%')}, which {'supports a stable learning story' if forgetting <= 7 else 'suggests a late-stage drift worth watching'}.", "good" if forgetting <= 7 else "warning"))
    return cards[:3]


def _math_quant_takeaways(rows: Sequence[Dict[str, Any]]) -> List[TakeawayCard]:
    peak_em, peak_step = _peak_value(rows, "exact_match", True)
    peak_acc, peak_acc_step = _peak_value(rows, "accuracy", True)
    ppl = _metric_series(rows, "perplexity")[1]
    first_non_zero = _step_when(rows, "exact_match", lambda value: value > 0) or _step_when(rows, "accuracy", lambda value: value > 0)
    cards: List[TakeawayCard] = []
    signal = peak_em if peak_em is not None else peak_acc
    signal_step = peak_step if peak_em is not None else peak_acc_step
    if signal is not None:
        cards.append(TakeawayCard("Correct-answer acquisition", "first real correctness", f"The model reaches its best hard-answer score of {_fmt_value(_pp(signal), '%')} around step {signal_step or 0}, which is the clearest proof that learning is translating into final answers.", "good"))
    if first_non_zero is not None:
        cards.append(TakeawayCard("Format breakthrough", "answer format learned", f"Non-zero correctness first appears at step {first_non_zero}, marking the point where the model starts producing evaluable answers instead of only plausible text.", "good"))
    if ppl:
        cards.append(TakeawayCard("Language-model confidence", "expected divergence" if len(ppl) > 1 and ppl[-1] > ppl[0] else "fluency improving", f"Perplexity {'rises' if len(ppl) > 1 and ppl[-1] > ppl[0] else 'falls'} from {_fmt_value(ppl[0])} to {_fmt_value(ppl[-1])}; interpret this alongside exact match rather than in isolation.", "neutral"))
    return cards[:3]


def _math_cot_takeaways(rows: Sequence[Dict[str, Any]]) -> List[TakeawayCard]:
    coverage = _pp(_final_checkpoint_value(rows, "qual_cot_coverage") or _final_checkpoint_value(rows, "qual_cot_anchor_coverage"))
    anchors = _final_checkpoint_value(rows, "qual_cot_anchor_count")
    step_length = _final_checkpoint_value(rows, "qual_cot_step_length")
    exact_match = _pp(_final_checkpoint_value(rows, "exact_match"))
    return [
        TakeawayCard("Reasoning structure adoption", "structure learned" if (coverage or 0) >= 50 else "partial structure", f"Final CoT coverage reaches {_fmt_value(coverage, '%')}, showing whether the model has actually adopted multi-step reasoning markers.", "good" if (coverage or 0) >= 50 else "neutral"),
        TakeawayCard("Reasoning density", "not just prefixes" if (anchors or 0) >= 2 and (step_length or 0) >= 6 else "shallow chains", f"Anchor count {_fmt_value(anchors)} with step length {_fmt_value(step_length)} helps separate genuine chain-of-thought behavior from short templated prefixes.", "good" if (anchors or 0) >= 2 and (step_length or 0) >= 6 else "warning"),
        TakeawayCard("Structure vs correctness", "linked signals" if (exact_match or 0) > 0 else "structure ahead of accuracy", f"Exact match at {_fmt_value(exact_match, '%')} should be read beside CoT coverage to show whether structured reasoning is already turning into correct answers.", "neutral"),
    ]


def _structured_nlg_takeaways(rows: Sequence[Dict[str, Any]]) -> List[TakeawayCard]:
    coverage = _pp(_final_checkpoint_value(rows, "qual_slot_coverage_mean"))
    coverage_base = _pp(_initial_numeric_value(rows, "qual_slot_coverage_mean"))
    pinned = _structured_pinned_value(rows, final=True)
    pinned_base = _structured_pinned_value(rows, final=False)
    inversion = _structured_inversion_value(rows, final=True)
    inversion_base = _structured_inversion_value(rows, final=False)
    weakest = min(((slot, _pp(_final_checkpoint_value(rows, key))) for key, slot in _find_per_slot_columns(rows) if _final_checkpoint_value(rows, key) is not None), key=lambda item: item[1], default=None)
    coverage_gain = None if coverage is None or coverage_base is None else coverage - coverage_base
    pinned_gain = None if pinned is None or pinned_base is None else pinned - pinned_base
    inversion_change = None if inversion is None or inversion_base is None else inversion - inversion_base
    story_bits = []
    if coverage is not None:
        if coverage_gain is not None:
            story_bits.append(f"mean slot coverage climbs from {_fmt_value(coverage_base, '%')} to {_fmt_value(coverage, '%')}")
        else:
            story_bits.append(f"mean slot coverage finishes at {_fmt_value(coverage, '%')}")
    if pinned is not None:
        if pinned_gain is not None:
            story_bits.append(f"pinned consistency improves from {_fmt_value(pinned_base, '%')} to {_fmt_value(pinned, '%')}")
        else:
            story_bits.append(f"pinned consistency finishes at {_fmt_value(pinned, '%')}")
    cards = [
        TakeawayCard(
            "Coverage + consistency arc",
            "clean monotonic story" if (coverage or 0) >= 85 and (pinned or 0) >= 75 else "still uneven",
            (_join_non_empty(story_bits, "; ").capitalize() + ". This tells us whether the model is reliably turning the same meaning representation into the right set of slot mentions.") if story_bits else "Structured coverage and consistency signals were not available for this run.",
            "good" if (coverage or 0) >= 85 else "neutral",
        ),
    ]
    if inversion is not None:
        if inversion_change is not None and inversion_change <= 0:
            body = f"Inversion falls from {_fmt_value(inversion_base, '%')} to {_fmt_value(inversion, '%')}. In plain English, this means the model is getting better at saying the slot with the correct meaning, not just mentioning the slot name."
        elif inversion_change is not None:
            body = f"Inversion rises from {_fmt_value(inversion_base, '%')} to {_fmt_value(inversion, '%')}, so this slot deserves monitoring. The absolute rate is still low, which means polarity mistakes are rare rather than dominant."
        else:
            body = f"Inversion ends at {_fmt_value(inversion, '%')}. Lower values mean the model is saying the slot with the correct meaning, not flipping the polarity after mentioning it."
        cards.append(TakeawayCard("Semantic integrity", "semantics learned" if inversion <= 5 else "semantic mismatch", body, "good" if inversion <= 5 else "warning"))
    if weakest:
        cards.append(TakeawayCard("Structural outlier", f"{weakest[0]} needs work", f"The weakest tracked slot is {weakest[0]} at {_fmt_value(weakest[1], '%')}. This is the clearest place to focus if we want the next round of improvements to be visible in client-facing outputs.", "warning" if (weakest[1] or 100) < 80 else "neutral"))
    return cards[:3]


def _semantic_takeaways(rows: Sequence[Dict[str, Any]]) -> List[TakeawayCard]:
    similarity = _pp(_final_checkpoint_value(rows, "qual_semantic_similarity"))
    repetition = _pp(_final_checkpoint_value(rows, "qual_repetition_rate"))
    length = _final_checkpoint_value(rows, "qual_mean_response_length")
    return [
        TakeawayCard("Task-fit signal", "semantic alignment" if (similarity or 0) >= 70 else "weak alignment", f"Semantic similarity reaches {_fmt_value(similarity, '%')}, which is the best single summary of whether the model is responding in the intended style/content space.", "good" if (similarity or 0) >= 70 else "neutral"),
        TakeawayCard("Response health", "healthy outputs" if (repetition or 100) <= 10 else "watch repetition", f"Mean response length is {_fmt_value(length)} with repetition at {_fmt_value(repetition, '%')}, helping distinguish useful adaptation from degenerate verbosity.", "good" if (repetition or 100) <= 10 else "warning"),
    ]


def _keyword_takeaways(rows: Sequence[Dict[str, Any]]) -> List[TakeawayCard]:
    density = _pp(_final_checkpoint_value(rows, "qual_keyword_density"))
    ttr = _pp(_final_checkpoint_value(rows, "qual_type_token_ratio"))
    repetition = _pp(_final_checkpoint_value(rows, "qual_repetition_rate"))
    return [
        TakeawayCard("Specialization signal", "domain vocabulary learned" if (density or 0) >= 10 else "light specialization", f"Keyword density reaches {_fmt_value(density, '%')}, which is the best direct sign that the model is adopting the target domain language.", "good" if (density or 0) >= 10 else "neutral"),
        TakeawayCard("Vocabulary quality", "diverse enough" if (ttr or 0) >= 25 else "too templated", f"Type/token ratio is {_fmt_value(ttr, '%')} with repetition at {_fmt_value(repetition, '%')}, so we can judge whether specialization still looks natural.", "good" if (ttr or 0) >= 25 and (repetition or 100) <= 10 else "warning"),
    ]


def generate_insights(rows: Sequence[Dict[str, Any]], profile: UsecaseProfile) -> List[str]:
    if profile.slug == "classification":
        cards = _classification_takeaways(rows)
    elif profile.slug == "math_reasoning_quant":
        cards = _math_quant_takeaways(rows)
    elif profile.slug == "math_reasoning_cot":
        cards = _math_cot_takeaways(rows)
    elif profile.slug == "structured_nlg":
        cards = _structured_nlg_takeaways(rows)
    elif profile.slug == "instruction_following_semantic":
        cards = _semantic_takeaways(rows)
    elif profile.slug == "domain_adaptation_keyword":
        cards = _keyword_takeaways(rows)
    else:
        cards = _semantic_takeaways(rows)
    return [f"{card.title}: {card.body}" for card in cards]


def json_safe(value: Any) -> str:
    try:
        import json
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return ""


def _make_line_chart(chart_id: str, section: str, role: str, title: str, subtitle: str, traces: List[ChartTrace], y_label: str, y2_label: str = "", thresholds: Optional[List[ThresholdLine]] = None, note: str = "", aspect: str = "wide") -> Optional[ChartSpec]:
    useful = [trace for trace in traces if trace.x and trace.y]
    if not useful:
        return None
    return ChartSpec(chart_id, section, role, title, subtitle, "line", aspect, useful, thresholds or [], y_label, y2_label, "Step", {}, note)


def _classification_sections(rows: Sequence[Dict[str, Any]]) -> List[SectionSpec]:
    acc_x, acc = _metric_series(rows, "accuracy")
    mcc_x, mcc = _metric_series(rows, "mcc")
    f1_x, f1 = _metric_series(rows, "f1")
    kappa_x, kappa = _metric_series(rows, "kappa")
    loss_x, loss = _metric_series(rows, "loss")
    eval_loss_x, eval_loss = _metric_series(rows, "eval_loss")
    ppl_x, ppl = _metric_series(rows, "perplexity")
    forget_x, forget = _metric_series(rows, "forgetting_accuracy")
    if not forget:
        forget_x, forget = _metric_series(rows, "forgetting_max")
    throughput_x, throughput = _metric_series(rows, "tokens_per_sec")
    grad_x, grad = _metric_series(rows, "grad_norm")
    lr_x, lr = _metric_series(rows, "lr")
    cycle_x, cycle = _metric_series(rows, "eval_cycle_time_s")

    quality = _make_line_chart(
        "classification_quality", "quantitative", "hero",
        "Accuracy and balance over training",
        "The core learning story: hard accuracy plus class-balance quality.",
        [
            ChartTrace("Accuracy", acc_x, [_pp(v) or 0.0 for v in acc], color_key="accuracy"),
            ChartTrace("MCC", mcc_x, [_pp(v) or 0.0 for v in mcc], color_key="mcc"),
            ChartTrace("Macro F1", f1_x, [_pp(v) or 0.0 for v in f1], color_key="f1"),
            ChartTrace("Kappa", kappa_x, [_pp(v) or 0.0 for v in kappa], color_key="kappa"),
        ],
        "Score (%)",
        thresholds=[ThresholdLine(70.0, "70% accuracy", color_key="muted")],
    )
    stability = _make_line_chart(
        "classification_stability", "quantitative", "support",
        "Forgetting and retention stability",
        "Late-stage drift should stay small if learning is stable.",
        [ChartTrace("Forgetting", forget_x, [_pp(v) or 0.0 for v in forget], color_key="forgetting", fill=True)],
        "Drift (pp)",
        thresholds=[ThresholdLine(5.0, "5pp watch line"), ThresholdLine(10.0, "10pp warning line", color_key="warning")],
        aspect="square",
    )
    loss_diag = _make_line_chart(
        "classification_loss_ppl", "quantitative", "support",
        "Perplexity and eval loss",
        "Reads task-format adaptation and late-stage evaluation confidence.",
        [
            ChartTrace("Perplexity", ppl_x, ppl, color_key="perplexity"),
            ChartTrace("Eval Loss", eval_loss_x, eval_loss, axis="secondary", color_key="eval_loss"),
        ],
        "Perplexity",
        "Eval loss",
        aspect="square",
    )
    training = _make_line_chart(
        "classification_training_health", "training", "detail",
        "Training health",
        "Optimization and throughput signals when they are available.",
        [
            ChartTrace("Train Loss", loss_x, loss, color_key="loss"),
            ChartTrace("Learning Rate", lr_x, lr, axis="secondary", color_key="neutral"),
            ChartTrace("Tokens/sec", throughput_x, throughput, axis="secondary", color_key="coverage"),
            ChartTrace("Grad Norm", grad_x, grad, axis="secondary", color_key="response"),
            ChartTrace("Eval Cycle Time", cycle_x, cycle, axis="secondary", color_key="warning"),
        ],
        "Loss",
        "Auxiliary",
    )
    return [
        SectionSpec("training", "Training Curves", "Optimization, throughput, and evaluator cadence.", [chart for chart in [training] if chart]),
        SectionSpec("quantitative", "Quantitative Metrics", "The strongest evidence that the model is learning the classification task.", [chart for chart in [quality, stability, loss_diag] if chart]),
        SectionSpec("qualitative", "Qualitative Metrics", "No qualitative metrics were logged for this run.", []),
    ]


def _math_quant_sections(rows: Sequence[Dict[str, Any]]) -> List[SectionSpec]:
    em_x, em = _metric_series(rows, "exact_match")
    acc_x, acc = _metric_series(rows, "accuracy")
    overlap_x, overlap = _metric_series(rows, "answer_overlap_f1")
    ppl_x, ppl = _metric_series(rows, "perplexity")
    eval_loss_x, eval_loss = _metric_series(rows, "eval_loss")
    hero = _make_line_chart(
        "math_quant_quality", "quantitative", "hero",
        "Correct-answer learning curve",
        "Exact match is the primary proof signal; overlap helps when exact match is sparse.",
        [
            ChartTrace("Exact Match", em_x, [_pp(v) or 0.0 for v in em], color_key="accuracy"),
            ChartTrace("Accuracy", acc_x, [_pp(v) or 0.0 for v in acc], color_key="mcc"),
            ChartTrace("Answer Overlap", overlap_x, [_pp(v) or 0.0 for v in overlap], color_key="f1"),
        ],
        "Score (%)",
    )
    support = _make_line_chart(
        "math_quant_confidence", "quantitative", "support",
        "Perplexity and eval loss",
        "Use this beside correctness to distinguish fluency from genuine answer learning.",
        [ChartTrace("Perplexity", ppl_x, ppl, color_key="perplexity"), ChartTrace("Eval Loss", eval_loss_x, eval_loss, axis="secondary", color_key="eval_loss")],
        "Perplexity",
        "Eval loss",
        aspect="square",
    )
    return [
        SectionSpec("training", "Training Curves", "Training-health metrics only appear when logged.", []),
        SectionSpec("quantitative", "Quantitative Metrics", "Hard correctness signals for math reasoning.", [chart for chart in [hero, support] if chart]),
        SectionSpec("qualitative", "Qualitative Metrics", "No qualitative metrics were logged for this run.", []),
    ]


def _math_cot_sections(rows: Sequence[Dict[str, Any]]) -> List[SectionSpec]:
    em_x, em = _metric_series(rows, "exact_match")
    coverage_x, coverage = _metric_series(rows, "qual_cot_coverage")
    if not coverage:
        coverage_x, coverage = _metric_series(rows, "qual_cot_anchor_coverage")
    anchor_x, anchor = _metric_series(rows, "qual_cot_anchor_count")
    step_x, step = _metric_series(rows, "qual_cot_step_length")
    non_empty_x, non_empty = _metric_series(rows, "qual_non_empty_rate")
    hero = _make_line_chart(
        "math_cot_structure", "qualitative", "hero",
        "Reasoning structure over training",
        "Shows whether the model is actually learning to produce chain-of-thought scaffolding.",
        [ChartTrace("CoT Coverage", coverage_x, [_pp(v) or 0.0 for v in coverage], color_key="coverage"), ChartTrace("Exact Match", em_x, [_pp(v) or 0.0 for v in em], color_key="mcc")],
        "Score (%)",
    )
    support_one = _make_line_chart(
        "math_cot_anchor_density", "qualitative", "support",
        "Anchor count and step length",
        "Separates genuine multi-step reasoning from short templated prefixes.",
        [ChartTrace("Anchor Count", anchor_x, anchor, color_key="semantic"), ChartTrace("Step Length", step_x, step, axis="secondary", color_key="response")],
        "Anchor count",
        "Step length",
        aspect="square",
    )
    support_two = _make_line_chart(
        "math_cot_response_health", "qualitative", "support",
        "Response health",
        "Confirms that structural gains are not coming from collapsing output quality.",
        [ChartTrace("Non-empty Rate", non_empty_x, [_pp(v) or 0.0 for v in non_empty], color_key="accuracy")],
        "Rate (%)",
        aspect="square",
    )
    return [
        SectionSpec("training", "Training Curves", "Training-health metrics only appear when logged.", []),
        SectionSpec("quantitative", "Quantitative Metrics", "Hard-answer signals for reasoning runs.", []),
        SectionSpec("qualitative", "Qualitative Metrics", "Structural reasoning signals explain how the model is learning.", [chart for chart in [hero, support_one, support_two] if chart]),
    ]


def _structured_nlg_sections(rows: Sequence[Dict[str, Any]]) -> List[SectionSpec]:
    coverage_x, coverage = _metric_series(rows, "qual_slot_coverage_mean")
    pool_x, pool = _metric_series(rows, "qual_consistency_score_mean")
    pinned_x, pinned = _metric_series(rows, "qual_pinned_consistency")
    if not pinned:
        pinned_x, pinned = _metric_series(rows, "qual_pinned_consistency_score")
    inversion_x, inversion = _metric_series(rows, "qual_familyFriendly_inversion")
    if not inversion:
        inversion_x, inversion = _metric_series(rows, "qual_slot_familyFriendly_inversion_rate")
    if not inversion:
        inversion_x, inversion = _metric_series(rows, "qual_slot_inversion_rate")
    ppl_x, ppl = _metric_series(rows, "perplexity")
    hero = _make_line_chart(
        "structured_arc", "qualitative", "hero",
        "Coverage and consistency arc",
        "The best single summary of whether the model is learning to verbalize structured meaning representations reliably.",
        [
            ChartTrace("Mean Slot Coverage", coverage_x, [_pp(v) or 0.0 for v in coverage], color_key="coverage"),
            ChartTrace("Pinned Consistency", pinned_x, [_pp(v) or 0.0 for v in pinned], color_key="consistency", style="dashed"),
            ChartTrace("Pool Consistency", pool_x, [_pp(v) or 0.0 for v in pool], color_key="secondary"),
        ],
        "Score (%)",
    )
    slot_pairs = [(slot, _pp(_final_checkpoint_value(rows, key))) for key, slot in _find_per_slot_columns(rows)]
    slot_pairs = [(slot, value) for slot, value in slot_pairs if value is not None]
    slot_pairs.sort(key=lambda item: item[1], reverse=True)
    slot_bar = None
    if slot_pairs:
        slot_bar = ChartSpec(
            id="structured_slot_bar",
            section="qualitative",
            role="support",
            title="Per-slot coverage at final checkpoint",
            subtitle="Spot the structural outlier rather than averaging it away.",
            chart_type="barh",
            preferred_aspect="square",
            traces=[ChartTrace("Coverage", [value for _, value in slot_pairs], list(range(len(slot_pairs))), trace_type="barh", color_key="coverage")],
            x_label="Coverage (%)",
            y_label="",
            note=json_safe({"labels": [slot for slot, _ in slot_pairs]}),
        )
    inversion_chart = _make_line_chart(
        "structured_inversion", "qualitative", "support",
        "Inversion vs coverage",
        "Low inversion matters because mentioning a slot is not enough if the polarity is wrong.",
        [ChartTrace("Coverage", coverage_x, [_pp(v) or 0.0 for v in coverage], color_key="coverage"), ChartTrace("Inversion Rate", inversion_x, [_pp(v) or 0.0 for v in inversion], axis="secondary", color_key="secondary")],
        "Coverage (%)",
        "Inversion (%)",
        aspect="square",
    )
    ppl_chart = _make_line_chart(
        "structured_ppl_tradeoff", "quantitative", "support",
        "Perplexity vs slot coverage",
        "Specialization can increase perplexity while structured coverage improves; that is often a real finding, not a bug.",
        [ChartTrace("Perplexity", ppl_x, ppl, color_key="perplexity"), ChartTrace("Coverage", coverage_x, [_pp(v) or 0.0 for v in coverage], axis="secondary", color_key="coverage")],
        "Perplexity",
        "Coverage (%)",
        aspect="square",
    )
    return [
        SectionSpec("training", "Training Curves", "Training-health metrics only appear when logged.", []),
        SectionSpec("quantitative", "Quantitative Metrics", "Perplexity and overlap-type signals that complement structural metrics.", [chart for chart in [ppl_chart] if chart]),
        SectionSpec("qualitative", "Qualitative Metrics", "Usecase-specific structural signals for structured generation.", [chart for chart in [hero, slot_bar, inversion_chart] if chart]),
    ]


def _semantic_sections(rows: Sequence[Dict[str, Any]], profile: UsecaseProfile) -> List[SectionSpec]:
    similarity_x, similarity = _metric_series(rows, "qual_semantic_similarity")
    density_x, density = _metric_series(rows, "qual_keyword_density")
    ttr_x, ttr = _metric_series(rows, "qual_type_token_ratio")
    repetition_x, repetition = _metric_series(rows, "qual_repetition_rate")
    non_empty_x, non_empty = _metric_series(rows, "qual_non_empty_rate")
    ppl_x, ppl = _metric_series(rows, "perplexity")
    response_x, response_len = _metric_series(rows, "qual_mean_response_length")
    primary_traces = []
    if similarity:
        primary_traces.append(ChartTrace("Semantic Similarity", similarity_x, [_pp(v) or 0.0 for v in similarity], color_key="semantic"))
    if density:
        primary_traces.append(ChartTrace("Keyword Density", density_x, [_pp(v) or 0.0 for v in density], color_key="keyword"))
    hero = _make_line_chart(f"{profile.slug}_primary", "qualitative", "hero", "Primary qualitative learning signal", "This chart shows the strongest direct evidence that the model is adopting the intended response style.", primary_traces, "Rate (%)")
    support_one = _make_line_chart(
        f"{profile.slug}_response_health", "qualitative", "support",
        "Response health",
        "Keeps an eye on non-empty rate, repetition, and response length so gains still look usable.",
        [ChartTrace("Non-empty Rate", non_empty_x, [_pp(v) or 0.0 for v in non_empty], color_key="accuracy"), ChartTrace("Repetition", repetition_x, [_pp(v) or 0.0 for v in repetition], axis="secondary", color_key="secondary"), ChartTrace("Response Length", response_x, response_len, axis="secondary", color_key="response")],
        "Rate (%)", "Length / repetition", aspect="square",
    )
    support_two = _make_line_chart(
        f"{profile.slug}_diversity", "qualitative", "support",
        "Diversity and perplexity",
        "Balances specialization with output variety and fluency.",
        [ChartTrace("Type/Token Ratio", ttr_x, [_pp(v) or 0.0 for v in ttr], color_key="diversity"), ChartTrace("Perplexity", ppl_x, ppl, axis="secondary", color_key="perplexity")],
        "TTR (%)", "Perplexity", aspect="square",
    )
    return [
        SectionSpec("training", "Training Curves", "Training-health metrics only appear when logged.", []),
        SectionSpec("quantitative", "Quantitative Metrics", "Loss and perplexity style metrics for qualitative runs.", []),
        SectionSpec("qualitative", "Qualitative Metrics", "Signals that explain how model behavior changes, not only whether it improved.", [chart for chart in [hero, support_one, support_two] if chart]),
    ]


def _classification_sections_v2(rows: Sequence[Dict[str, Any]]) -> List[SectionSpec]:
    acc_x, acc = _metric_series(rows, "accuracy")
    mcc_x, mcc = _metric_series(rows, "mcc")
    f1_x, f1 = _metric_series(rows, "f1")
    kappa_x, kappa = _metric_series(rows, "kappa")
    loss_x, loss = _metric_series(rows, "loss")
    eval_loss_x, eval_loss = _metric_series(rows, "eval_loss")
    ppl_x, ppl = _metric_series(rows, "perplexity")
    forget_x, forget = _metric_series(rows, "forgetting_accuracy")
    if not forget:
        forget_x, forget = _metric_series(rows, "forgetting_max")
    throughput_x, throughput = _metric_series(rows, "tokens_per_sec")
    grad_x, grad = _metric_series(rows, "grad_norm")
    lr_x, lr = _metric_series(rows, "lr")
    cycle_x, cycle = _metric_series(rows, "eval_cycle_time_s")

    quality = _make_line_chart(
        "classification_quality", "quantitative", "hero",
        "Accuracy and balance over training",
        "The core learning story: hard accuracy plus class-balance quality.",
        [
            ChartTrace("Accuracy", acc_x, [_pp(v) or 0.0 for v in acc], color_key="accuracy"),
            ChartTrace("MCC", mcc_x, [_pp(v) or 0.0 for v in mcc], color_key="mcc"),
            ChartTrace("Macro F1", f1_x, [_pp(v) or 0.0 for v in f1], color_key="f1"),
            ChartTrace("Kappa", kappa_x, [_pp(v) or 0.0 for v in kappa], color_key="kappa"),
        ],
        "Score (%)",
        thresholds=[ThresholdLine(70.0, "70% accuracy", color_key="muted")],
    )
    stability = _make_line_chart(
        "classification_stability", "quantitative", "support",
        "Forgetting and retention stability",
        "Late-stage drift should stay small if learning is stable.",
        [ChartTrace("Forgetting", forget_x, [_pp(v) or 0.0 for v in forget], color_key="forgetting", fill=True)],
        "Drift (percentage points)",
        thresholds=[
            ThresholdLine(5.0, "5 percentage-point watch line"),
            ThresholdLine(10.0, "10 percentage-point warning line", color_key="warning"),
        ],
        aspect="square",
    )
    loss_diag = _make_line_chart(
        "classification_loss_ppl", "quantitative", "support",
        "Perplexity and eval loss",
        "Reads task-format adaptation and late-stage evaluation confidence.",
        [
            ChartTrace("Perplexity", ppl_x, ppl, color_key="perplexity"),
            ChartTrace("Eval Loss", eval_loss_x, eval_loss, axis="secondary", color_key="eval_loss"),
        ],
        "Perplexity",
        "Eval loss",
        aspect="square",
    )

    training = None
    if cycle and not any([loss, lr, throughput, grad]):
        training = _make_line_chart(
            "classification_eval_cycle", "training", "detail",
            "Evaluation cycle time",
            "Time spent between consecutive evaluations; lower values mean faster feedback loops.",
            [ChartTrace("Eval Cycle Time", cycle_x, cycle, color_key="warning")],
            "Eval cycle time (seconds)",
        )
    else:
        training_traces: List[ChartTrace] = []
        y_label = "Train Loss"
        y2_label = ""
        if loss:
            training_traces.append(ChartTrace("Train Loss", loss_x, loss, color_key="loss"))
        secondary = []
        if lr:
            secondary.append(ChartTrace("Learning Rate", lr_x, lr, axis="secondary", color_key="neutral"))
        if throughput:
            secondary.append(ChartTrace("Tokens / sec", throughput_x, throughput, axis="secondary", color_key="coverage"))
        if grad:
            secondary.append(ChartTrace("Grad Norm", grad_x, grad, axis="secondary", color_key="response"))
        if cycle:
            secondary.append(ChartTrace("Eval Cycle Time", cycle_x, cycle, axis="secondary", color_key="warning"))
        if not training_traces and len(secondary) == 1:
            single = secondary[0]
            single.axis = "primary"
            training_traces = [single]
            y_label = "Eval cycle time (seconds)" if single.name == "Eval Cycle Time" else single.name
        else:
            training_traces.extend(secondary)
            if secondary:
                y2_label = "Secondary training metrics"
        if training_traces:
            training = _make_line_chart(
                "classification_training_health", "training", "detail",
                "Training and evaluation diagnostics",
                "Optimization, throughput, and evaluator cadence when these signals are available.",
                training_traces,
                y_label,
                y2_label,
            )

    return [
        SectionSpec("training", "Training Curves", "Optimization, throughput, and evaluator cadence.", [chart for chart in [training] if chart]),
        SectionSpec("quantitative", "Quantitative Metrics", "The strongest evidence that the model is learning the classification task.", [chart for chart in [quality, stability, loss_diag] if chart]),
        SectionSpec("qualitative", "Qualitative Metrics", "No qualitative metrics were logged for this run.", []),
    ]


def _structured_nlg_sections_v2(rows: Sequence[Dict[str, Any]]) -> List[SectionSpec]:
    coverage_x, coverage = _metric_series(rows, "qual_slot_coverage_mean")
    pool_x, pool = _metric_series(rows, "qual_consistency_score_mean")
    pinned_x, pinned = _metric_series(rows, "qual_pinned_consistency")
    if not pinned:
        pinned_x, pinned = _metric_series(rows, "qual_pinned_consistency_score")
    inversion_x, inversion = _metric_series(rows, "qual_familyFriendly_inversion")
    if not inversion:
        inversion_x, inversion = _metric_series(rows, "qual_slot_familyFriendly_inversion_rate")
    if not inversion:
        inversion_x, inversion = _metric_series(rows, "qual_slot_inversion_rate")
    ppl_x, ppl = _metric_series(rows, "perplexity")

    hero = _make_line_chart(
        "structured_arc", "qualitative", "hero",
        "Coverage and consistency arc",
        "The best single summary of whether the model is learning to verbalize structured meaning representations reliably.",
        [
            ChartTrace("Mean Slot Coverage", coverage_x, [_pp(v) or 0.0 for v in coverage], color_key="coverage"),
            ChartTrace("Pinned Consistency", pinned_x, [_pp(v) or 0.0 for v in pinned], color_key="consistency", style="dashed"),
            ChartTrace("Pool Consistency", pool_x, [_pp(v) or 0.0 for v in pool], color_key="secondary"),
        ],
        "Score (%)",
    )

    slot_columns = _find_per_slot_columns(rows)
    slot_palette = ["coverage", "consistency", "secondary", "semantic", "keyword", "diversity", "response", "accuracy"]
    slot_history_traces: List[ChartTrace] = []
    for idx, (key, slot) in enumerate(slot_columns):
        xs, ys = _metric_series(rows, key)
        if not ys:
            continue
        slot_history_traces.append(
            ChartTrace(
                slot,
                xs,
                [_pp(v) or 0.0 for v in ys],
                color_key=slot_palette[idx % len(slot_palette)],
            )
        )
    slot_history = _make_line_chart(
        "structured_slot_history", "qualitative", "support",
        "Per-slot coverage over time",
        "One line per slot makes structural gaps visible without scanning separate charts.",
        slot_history_traces,
        "Coverage (%)",
        aspect="square",
    )

    slot_pairs = [(slot, _pp(_final_checkpoint_value(rows, key))) for key, slot in slot_columns]
    slot_pairs = [(slot, value) for slot, value in slot_pairs if value is not None]
    slot_pairs.sort(key=lambda item: item[1], reverse=True)
    slot_bar = None
    if slot_pairs:
        slot_bar = ChartSpec(
            id="structured_slot_bar",
            section="qualitative",
            role="detail",
            title="Per-slot coverage at final checkpoint",
            subtitle="Spot the structural outlier rather than averaging it away.",
            chart_type="barh",
            preferred_aspect="square",
            traces=[ChartTrace("Coverage", [value for _, value in slot_pairs], list(range(len(slot_pairs))), trace_type="barh", color_key="coverage")],
            x_label="Coverage (%)",
            y_label="Slot",
            note=json_safe({"labels": [slot for slot, _ in slot_pairs]}),
        )

    inversion_chart = _make_line_chart(
        "structured_inversion", "qualitative", "support",
        "Inversion vs coverage",
        "Low inversion matters because mentioning a slot is not enough if the polarity is wrong.",
        [
            ChartTrace("Coverage", coverage_x, [_pp(v) or 0.0 for v in coverage], color_key="coverage"),
            ChartTrace("Inversion Rate", inversion_x, [_pp(v) or 0.0 for v in inversion], axis="secondary", color_key="secondary"),
        ],
        "Coverage (%)",
        "Inversion (%)",
        aspect="square",
    )
    ppl_chart = _make_line_chart(
        "structured_ppl_tradeoff", "quantitative", "support",
        "Perplexity vs slot coverage",
        "Specialization can increase perplexity while structured coverage improves; that is often a real finding, not a bug.",
        [ChartTrace("Perplexity", ppl_x, ppl, color_key="perplexity"), ChartTrace("Coverage", coverage_x, [_pp(v) or 0.0 for v in coverage], axis="secondary", color_key="coverage")],
        "Perplexity",
        "Coverage (%)",
        aspect="square",
    )
    return [
        SectionSpec("training", "Training Curves", "Training-health metrics only appear when logged.", []),
        SectionSpec("quantitative", "Quantitative Metrics", "Perplexity and overlap-type signals that complement structural metrics.", [chart for chart in [ppl_chart] if chart]),
        SectionSpec("qualitative", "Qualitative Metrics", "Usecase-specific structural signals for structured generation.", [chart for chart in [hero, slot_history, inversion_chart, slot_bar] if chart]),
    ]


def _validate_presentation_spec(rows: Sequence[Dict[str, Any]], spec: PresentationSpec) -> List[str]:
    warnings: List[str] = []
    final_step = _final_checkpoint_step(rows)
    summary_present = bool(_summary_rows(rows))

    for kpi in spec.kpi_cards:
        metric_key = getattr(kpi, "metric_key", "") or ""
        source_kind = getattr(kpi, "source_kind", "") or ""
        expected: Optional[float] = None
        if metric_key and source_kind == "final_checkpoint":
            expected = _final_checkpoint_value(rows, metric_key)
        elif metric_key and source_kind == "peak":
            expected, _ = _peak_value(rows, metric_key, higher_better=(kpi.direction != "lower_better"))
        elif metric_key and source_kind == "max":
            expected, _ = _peak_value(rows, metric_key, higher_better=True)
        if expected is not None and kpi.unit == "%":
            expected = _pp(expected)
        if expected is not None and abs((expected or 0.0) - (kpi.value or 0.0)) > 1e-6:
            warnings.append(f"KPI '{kpi.label}' does not match its source metric '{metric_key}'.")
        if "pp" in (kpi.delta_display or "").lower():
            warnings.append(f"KPI '{kpi.label}' still uses the ambiguous 'pp' shorthand.")
        if source_kind == "final_checkpoint" and final_step is not None:
            source_note = getattr(kpi, "source_note", "") or ""
            if str(final_step) not in source_note and "final checkpoint" not in source_note.lower():
                warnings.append(f"KPI '{kpi.label}' should state that it comes from the final checkpoint row (eval step {final_step}).")

    for chart in spec.chart_specs:
        if chart.chart_type != "barh" and not chart.x_label:
            warnings.append(f"Chart '{chart.title}' is missing an x-axis label.")
        if chart.chart_type != "barh" and not chart.y_label:
            warnings.append(f"Chart '{chart.title}' is missing a primary y-axis label.")
        if chart.y_label == "Auxiliary":
            warnings.append(f"Chart '{chart.title}' uses the ambiguous y-axis label 'Auxiliary'.")
        if chart.chart_type == "line" and chart.x_label == "Value":
            warnings.append(f"Chart '{chart.title}' uses the generic x-axis label 'Value'; it should name the actual scale.")

    series_keys = {
        "accuracy",
        "mcc",
        "kappa",
        "f1",
        "exact_match",
        "perplexity",
        "eval_loss",
        "qual_slot_coverage_mean",
        "qual_consistency_score_mean",
        "qual_pinned_consistency",
        "qual_pinned_consistency_score",
        "qual_semantic_similarity",
        "qual_keyword_density",
        "qual_cot_coverage",
        "qual_cot_anchor_coverage",
    }
    for key, _slot in _find_per_slot_columns(rows):
        series_keys.add(key)
    for key in sorted(series_keys):
        xs, ys = _metric_series(rows, key)
        final_val = _final_checkpoint_value(rows, key)
        if not ys or final_val is None:
            continue
        if abs(ys[-1] - final_val) > 1e-6:
            warnings.append(
                f"Series resolution mismatch for '{key}'; rendered plots must end at the CSV final checkpoint row exactly."
            )

    if summary_present and final_step is not None:
        for key in ("accuracy", "mcc", "kappa", "f1", "perplexity"):
            summary_val = _summary_value(rows, key)
            final_val = _final_checkpoint_value(rows, key)
            if summary_val is None or final_val is None:
                continue
            if abs(summary_val - final_val) > 1e-6:
                warnings.append(
                    f"Final checkpoint resolution mismatch for '{key}' at eval step {final_step}; rendered outputs must use the CSV final row exactly."
                )
    return warnings


def build_presentation_spec(rows: Sequence[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> PresentationSpec:
    profile = detect_usecase(rows, config)
    kpis = select_kpis(rows, profile)
    if profile.slug == "classification":
        takeaways = _classification_takeaways(rows)
        sections = _classification_sections_v2(rows)
    elif profile.slug == "math_reasoning_quant":
        takeaways = _math_quant_takeaways(rows)
        sections = _math_quant_sections(rows)
    elif profile.slug == "math_reasoning_cot":
        takeaways = _math_cot_takeaways(rows)
        sections = _math_cot_sections(rows)
    elif profile.slug == "structured_nlg":
        takeaways = _structured_nlg_takeaways(rows)
        sections = _structured_nlg_sections_v2(rows)
    elif profile.slug == "instruction_following_semantic":
        takeaways = _semantic_takeaways(rows)
        sections = _semantic_sections(rows, profile)
    elif profile.slug == "domain_adaptation_keyword":
        takeaways = _keyword_takeaways(rows)
        sections = _semantic_sections(rows, profile)
    else:
        takeaways = _semantic_takeaways(rows)
        sections = _semantic_sections(rows, profile)
    chart_specs = [chart for section in sections for chart in section.charts]
    empty_states = [{"section": section.id, "message": section.description} for section in sections if not section.charts]
    spec = PresentationSpec(
        header=_header_from_config(config, profile),
        kpi_cards=kpis,
        takeaway_cards=takeaways,
        sections=sections,
        chart_specs=chart_specs,
        empty_states=empty_states,
        details_blocks=[],
        profile=profile,
    )
    validation_warnings = _validate_presentation_spec(rows, spec)
    if validation_warnings:
        spec.details_blocks.append(
            {
                "title": "Presentation validation",
                "kind": "warning",
                "items": validation_warnings,
            }
        )
    return spec


def presentation_to_dict(spec: PresentationSpec) -> Dict[str, Any]:
    payload = asdict(spec)
    if spec.profile is not None:
        payload["profile"] = asdict(spec.profile)
    return payload
