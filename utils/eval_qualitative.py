"""
utils/eval_qualitative.py
─────────────────────────────────────────────────────────────────────────────
Qualitative Evaluation Suite for InfiniTune.

Implements three lightweight CPU-bound proxy strategies that measure
"vibes", tone, and structural improvement without any LLM API calls and
with minimal RAM overhead.  The entire module is self-contained — it has
zero dependency on eval_metrics_train.py.

Strategies
----------
- semantic_similarity  : Cosine similarity between generated and golden
                         responses using all-MiniLM-L6-v2 (CPU, ~90 MB).
- keyword_density      : Domain-specific keyword density + Type-Token Ratio
                         (TTR) to prove vocabulary adoption. Reference-free.
- structural_cot       : Counts chain-of-thought "logic anchors" (regex) and
                         measures inter-anchor step length to prove CoT
                         adoption on reasoning tasks.

All strategies also compute universal metrics that run on already-generated
text at zero additional cost:
  • qual_mean_response_length  — word count, detects collapse / verbosity
  • qual_repetition_rate       — bigram repetition, detects stuck generation
  • qual_non_empty_rate        — fraction of non-empty responses

Output
------
QualitativeEvaluator.run() returns a flat dict[str, float] whose keys are
all prefixed with "qual_" so they integrate cleanly with MetricsLogger.COLUMNS
and never collide with quantitative metric names.

Hardware Safety
---------------
- SentenceTransformers model is always loaded with device="cpu" regardless of
  GPU availability, so it never competes with the LLM for VRAM.
- MiniLM is lazy-loaded on the first compute() call to avoid download delays
  on startup when qualitative eval is disabled.
- sentence-transformers is only imported inside SemanticSimilarityMetric so
  other strategies don't require it as a dependency.
"""

import re
import time
import collections
from abc import ABC, abstractmethod
from jinja2 import Template
from datasets import load_dataset
from transformers import GenerationConfig
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers (consistent with rest of codebase)
# ─────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}][QUAL_EVAL] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Universal qualitative metrics (strategy-agnostic, zero extra compute)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_universal_qualitative_metrics(predictions: list) -> dict:
    """
    Compute universal qualitative proxy metrics from a list of generated text
    strings.  Operates purely on string data — no model calls required.

    Returns keys:
        qual_mean_response_length : mean word count across non-empty responses
        qual_repetition_rate      : mean fraction of repeated bigrams per response
        qual_non_empty_rate       : fraction of predictions that are non-empty
    """
    if not predictions:
        return {}

    lengths = []
    bigram_repetition_rates = []
    non_empty_count = 0

    for pred in predictions:
        text = pred.strip() if pred else ""
        if not text:
            continue

        non_empty_count += 1
        words = text.lower().split()
        lengths.append(len(words))

        # Bigram repetition: fraction of bigrams that are repeated
        if len(words) >= 2:
            bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
            bigram_counts = collections.Counter(bigrams)
            repeated = sum(c - 1 for c in bigram_counts.values() if c > 1)
            bigram_repetition_rates.append(repeated / len(bigrams))
        else:
            bigram_repetition_rates.append(0.0)

    total = len(predictions)
    metrics = {
        "qual_non_empty_rate": non_empty_count / total if total > 0 else 0.0,
    }
    if lengths:
        metrics["qual_mean_response_length"] = sum(lengths) / len(lengths)
    if bigram_repetition_rates:
        metrics["qual_repetition_rate"] = sum(bigram_repetition_rates) / len(bigram_repetition_rates)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Strategy Base
# ─────────────────────────────────────────────────────────────────────────────

class QualitativeMetric(ABC):
    """
    Abstract base for all qualitative proxy metric strategies.

    Subclasses implement compute() and return a flat dict of qual_* metrics.
    The references list may contain None entries for reference-free strategies
    (e.g. keyword_density does not require a golden answer).
    """

    @abstractmethod
    def compute(self, predictions: list, references: list) -> dict:
        """
        Args:
            predictions : List of model-generated text strings.
            references  : List of golden reference strings, or list of None
                          for reference-free strategies.
        Returns:
            dict[str, float] — all keys prefixed with "qual_".
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: Semantic Similarity (UltraChat / Conversational)
# ─────────────────────────────────────────────────────────────────────────────

class SemanticSimilarityMetric(QualitativeMetric):
    """
    Computes cosine similarity between generated responses and golden
    references using a sentence-embedding model (default: all-MiniLM-L6-v2).

    The model is loaded onto CPU with device="cpu" to avoid competing with
    the LLM training session for GPU VRAM.  A 16 GB RAM machine can hold
    all-MiniLM-L6-v2 (~90 MB) alongside the training LLM without pressure.

    Model is lazy-loaded on the first compute() call so startup time is not
    penalised when this strategy is disabled.

    Returns keys:
        qual_semantic_similarity : mean cosine similarity in [0, 1]
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Import guarded: only crashes if this strategy is actually used
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
            self._SentenceTransformer = SentenceTransformer
        except ImportError:
            raise ImportError(
                "The 'sentence-transformers' package is required for the "
                "semantic_similarity strategy.  Install it with:\n"
                "  pip install sentence-transformers"
            )
        self._model_name = model_name
        self._model = None  # Lazy-loaded on first compute() call

    def _ensure_model_loaded(self) -> None:
        if self._model is None:
            _log(
                f"Loading sentence embedding model '{self._model_name}' on CPU "
                f"(first use — may download ~90 MB)..."
            )
            # device="cpu" is explicit — never touches GPU VRAM
            self._model = self._SentenceTransformer(self._model_name, device="cpu")
            _log("Sentence embedding model ready.")

    def compute(self, predictions: list, references: list) -> dict:
        """
        Returns qual_semantic_similarity: mean cosine similarity across pairs.
        Pairs where either prediction or reference is empty score 0.0.
        """
        self._ensure_model_loaded()

        if not predictions:
            return {}

        similarities = []
        valid_preds = []
        valid_refs = []

        for pred, ref in zip(predictions, references):
            pred_text = (pred or "").strip()
            ref_text = (ref or "").strip()

            # Empty prediction or reference cannot be compared; score 0.0
            if not pred_text or not ref_text:
                similarities.append(0.0)
            else:
                valid_preds.append(pred_text)
                valid_refs.append(ref_text)

        # Batch-encode all valid pairs at once for efficiency
        if valid_preds:
            import torch as _torch
            pred_embeddings = self._model.encode(valid_preds, convert_to_tensor=True, device="cpu")
            ref_embeddings  = self._model.encode(valid_refs,  convert_to_tensor=True, device="cpu")

            # Cosine similarity per pair
            cos_sim = _torch.nn.functional.cosine_similarity(pred_embeddings, ref_embeddings)
            # Clamp to [0, 1] — cosine can technically be negative for dissimilar vectors
            cos_sim = cos_sim.clamp(min=0.0, max=1.0)
            similarities.extend(cos_sim.tolist())

        if not similarities:
            return {}

        return {"qual_semantic_similarity": sum(similarities) / len(similarities)}


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: Keyword Density & Lexical Profiling (IMDb / Domain Adaptation)
# ─────────────────────────────────────────────────────────────────────────────

class KeywordDensityMetric(QualitativeMetric):
    """
    Measures how much domain-specific vocabulary the model is adopting through:
      - Keyword Density  : fraction of words that are domain keywords
      - Type-Token Ratio : lexical diversity (unique words / total words)
      - Hapax Ratio      : fraction of words used exactly once per response

    This is a reference-free metric — it analyses the model's generated text
    without needing a golden answer.

    Returns keys:
        qual_keyword_density : mean keyword density across responses
        qual_type_token_ratio: mean TTR (skips responses < 5 words)
        qual_hapax_ratio     : mean hapax ratio
    """

    def __init__(self, keywords: list):
        """
        Args:
            keywords: List of domain-specific words or phrases (case-insensitive).
                      These come directly from testing_strategy.keywords in YAML.
        """
        if not keywords:
            raise ValueError("KeywordDensityMetric requires a non-empty keywords list.")
        # Lower-case all keywords at init time; avoid repeated lower() in hot path
        self._keywords = [str(k).lower().strip() for k in keywords if k]
        _log(f"KeywordDensityMetric initialized with {len(self._keywords)} keywords.")

    def compute(self, predictions: list, references: list) -> dict:
        """
        references is ignored — this is a reference-free metric.
        """
        if not predictions:
            return {}

        density_scores = []
        ttr_scores = []
        hapax_scores = []

        for pred in predictions:
            text = (pred or "").strip().lower()
            if not text:
                continue

            words = text.split()
            if not words:
                continue

            total_words = len(words)

            # --- Keyword Density ---
            keyword_hits = sum(1 for w in words if w in self._keywords)
            density_scores.append(keyword_hits / total_words)

            # --- Type-Token Ratio (skip very short responses to avoid inflation) ---
            if total_words >= 5:
                unique_words = set(words)
                ttr_scores.append(len(unique_words) / total_words)

            # --- Hapax Ratio ---
            word_counts = collections.Counter(words)
            hapax = sum(1 for count in word_counts.values() if count == 1)
            hapax_scores.append(hapax / total_words)

        metrics = {}
        if density_scores:
            metrics["qual_keyword_density"] = sum(density_scores) / len(density_scores)
        if ttr_scores:
            metrics["qual_type_token_ratio"] = sum(ttr_scores) / len(ttr_scores)
        if hapax_scores:
            metrics["qual_hapax_ratio"] = sum(hapax_scores) / len(hapax_scores)

        return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3: Structural CoT Adherence (GSM8k / Reasoning)
# ─────────────────────────────────────────────────────────────────────────────

class StructuralCoTMetric(QualitativeMetric):
    """
    Measures Chain-of-Thought structural adoption by counting "logic anchors"
    (e.g. "First,", "Therefore,", "Step \\d+:") and measuring the length of
    reasoning content between those anchors.

    A model that is learning CoT will show:
      - Increasing anchor_count over training steps
      - Increasing step_length (it's generating actual reasoning, not just
        inserting anchors at the start and stopping)

    Logic anchors are compiled as regex patterns from the YAML config.  Invalid
    patterns are logged and skipped gracefully.

    Returns keys:
        qual_cot_anchor_count_mean : mean number of anchors found per response
        qual_cot_step_length_mean  : mean character length between anchors
        qual_cot_coverage_rate     : fraction of responses with ≥1 anchor
    """

    # Fallback anchors used when none are specified in config
    DEFAULT_ANCHORS = [
        r"First[,\s]",
        r"Second[,\s]",
        r"Third[,\s]",
        r"Next[,\s]",
        r"Then[,\s]",
        r"Finally[,\s]",
        r"Therefore[,\s]",
        r"Thus[,\s]",
        r"So[,\s]",
        r"Hence[,\s]",
        r"Step\s*\d+[:\.]",
        r"Let(?:'s| us)\s",
        r"We (?:know|have|can|need)",
        r"This means",
        r"In other words",
    ]

    def __init__(self, logic_anchors: list = None):
        """
        Args:
            logic_anchors: List of regex strings from testing_strategy.logic_anchors.
                           Falls back to DEFAULT_ANCHORS if None or empty.
        """
        raw_patterns = logic_anchors if logic_anchors else self.DEFAULT_ANCHORS
        self._patterns = []
        for pattern_str in raw_patterns:
            try:
                # YAML strings like "Step \\d+:" arrive in Python as "Step \\d+:"
                # re.compile handles the double-backslash correctly
                self._patterns.append(re.compile(str(pattern_str), re.IGNORECASE))
            except re.error as exc:
                _log(f"Warning: invalid logic_anchor regex '{pattern_str}': {exc} — skipping.")

        _log(f"StructuralCoTMetric initialized with {len(self._patterns)} anchor patterns.")

    def _count_anchors_and_step_length(self, text: str):
        """
        Returns (anchor_count: int, mean_step_length: float | None).

        mean_step_length is the mean character count of segments between anchors.
        Returns None if no anchors are found.
        """
        if not text.strip():
            return 0, None

        # Find all anchor match positions
        positions = []
        for pat in self._patterns:
            for m in pat.finditer(text):
                positions.append(m.start())

        if not positions:
            return 0, None

        positions.sort()
        anchor_count = len(positions)

        # Measure the length of each "step" (segment between consecutive anchors)
        step_lengths = []
        boundaries = [0] + positions + [len(text)]
        for i in range(len(positions)):
            seg_start = positions[i]
            seg_end = boundaries[i + 2]
            step_lengths.append(seg_end - seg_start)

        mean_step_length = sum(step_lengths) / len(step_lengths) if step_lengths else None
        return anchor_count, mean_step_length

    def compute(self, predictions: list, references: list) -> dict:
        """
        references is used only to log comparison info; the structural metrics
        are computed solely on the generated text.
        """
        if not predictions:
            return {}

        anchor_counts = []
        step_lengths = []
        responses_with_anchors = 0

        for pred in predictions:
            text = (pred or "").strip()
            count, mean_len = self._count_anchors_and_step_length(text)
            anchor_counts.append(count)
            if count > 0:
                responses_with_anchors += 1
            if mean_len is not None:
                step_lengths.append(mean_len)

        total = len(predictions)
        metrics = {}

        if anchor_counts:
            metrics["qual_cot_anchor_count_mean"] = sum(anchor_counts) / len(anchor_counts)
        if step_lengths:
            metrics["qual_cot_step_length_mean"] = sum(step_lengths) / len(step_lengths)
        if total > 0:
            metrics["qual_cot_coverage_rate"] = responses_with_anchors / total

        return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Factory Helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_metric(ts_cfg: dict) -> QualitativeMetric:
    """
    Instantiate the correct QualitativeMetric subclass from the
    testing_strategy config block.

    Args:
        ts_cfg: The testing_strategy dict from the YAML config.

    Returns:
        A QualitativeMetric instance.

    Raises:
        ValueError: If method is unrecognised.
    """
    method = (ts_cfg.get("method") or "").strip().lower()

    if method == "semantic_similarity":
        model_name = ts_cfg.get("sentence_model") or "sentence-transformers/all-MiniLM-L6-v2"
        return SemanticSimilarityMetric(model_name=model_name)

    if method == "keyword_density":
        raw_keywords = ts_cfg.get("keywords") or []
        return KeywordDensityMetric(keywords=raw_keywords)

    if method == "structural_cot":
        raw_anchors = ts_cfg.get("logic_anchors")  # None → use defaults
        return StructuralCoTMetric(logic_anchors=raw_anchors)

    raise ValueError(
        f"Unknown testing_strategy.method: '{method}'. "
        f"Supported values: semantic_similarity | keyword_density | structural_cot"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator / Public API
# ─────────────────────────────────────────────────────────────────────────────

class QualitativeEvaluator:
    """
    Orchestrates qualitative evaluation during training.

    Reads the testing_strategy block from the YAML config, instantiates the
    correct QualitativeMetric via the factory, loads an eval data pool,
    generates model responses, and returns a flat metrics dict.

    If the config has no testing_strategy block, or if
    testing_strategy.enabled is false, self.enabled is False and run()
    is an immediate no-op — existing configs are not affected.

    Constructor Args
    ----------------
    config    : Full YAML config dict.
    tokenizer : HuggingFace tokenizer (shared with trainer).
    device    : torch.device the model lives on (cuda / cpu / mps).
                Used to move tokenized inputs to the model's device for
                model.generate(). The SentenceTransformers model always
                stays on CPU internally regardless of this value.
    """

    def __init__(self, config: dict, tokenizer, device):
        self.config = config
        self.tokenizer = tokenizer
        self.device = device

        ts_cfg = config.get("testing_strategy") or {}
        self.enabled = bool(ts_cfg.get("enabled", False))

        if not self.enabled:
            _log("QualitativeEvaluator disabled (no testing_strategy block or enabled: false).")
            self._metric = None
            self.eval_data = []
            return

        # --- Build the metric strategy ---
        self._method = (ts_cfg.get("method") or "").strip().lower()
        self._metric: QualitativeMetric = _build_metric(ts_cfg)

        # --- Evaluation cadence ---
        self.eval_interval: int = int(ts_cfg.get("eval_interval", 50))
        self._eval_samples: int = int(ts_cfg.get("eval_samples", 20))
        self._max_new_tokens: int = int(ts_cfg.get("max_new_tokens", 150))

        # --- Jinja2 templates (for prompt construction during generation) ---
        preproc = config.get("preprocessing", {})
        self._prompt_template = Template(preproc.get("prompt_template", "{{ input }}"))

        # --- Load eval data pool ---
        self.eval_data: list = []
        self._eval_cursor: int = 0
        self._load_eval_data(ts_cfg)

        if not self.eval_data:
            _log("Warning: QualitativeEvaluator could not load eval data — disabling.")
            self.enabled = False
            return

        _log(
            f"QualitativeEvaluator ready: method={self._method}, "
            f"eval_interval={self.eval_interval}, eval_samples={self._eval_samples}, "
            f"eval_pool_size={len(self.eval_data)}, max_new_tokens={self._max_new_tokens}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Data Loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_eval_data(self, ts_cfg: dict) -> None:
        """
        Load evaluation data pool from the dataset configured in
        config.dataset.  The data schema varies by strategy:

        - semantic_similarity  : needs {input, reference} pairs
        - keyword_density      : only needs {input} prompts (reference-free)
        - structural_cot       : needs {input} prompts; references used
                                 only for logging
        """
        dataset_cfg = self.config.get("dataset", {})
        col_map = dataset_cfg.get("column_mapping", {})
        eval_split = dataset_cfg.get("eval_split", "test")
        label_map = dataset_cfg.get("label_map")

        input_col = col_map.get("input_col", "input")
        target_col = col_map.get("target_col", "target")

        ds_kwargs = {"path": dataset_cfg.get("name", ""), "split": eval_split}
        if dataset_cfg.get("config_name"):
            ds_kwargs["name"] = dataset_cfg["config_name"]

        _log(f"Loading qualitative eval pool from dataset '{ds_kwargs['path']}' split='{eval_split}'...")

        # Load more samples than eval_samples so the sliding window
        # actually rotates through different data each eval step.
        # Default pool multiplier: 5× eval_samples.
        pool_size = int(ts_cfg.get("eval_pool_size", self._eval_samples * 5))
        pool_size = max(pool_size, self._eval_samples)  # never smaller than window

        try:
            dataset = load_dataset(**ds_kwargs)
            dataset = dataset.shuffle(seed=42)

            for i, example in enumerate(dataset):
                if i >= pool_size:
                    break

                input_val = str(example.get(input_col, ""))

                # Reference handling: extract and apply label_map if present
                raw_target = example.get(target_col)
                if raw_target is None:
                    target_val = None
                else:
                    if label_map is not None:
                        if raw_target in label_map:
                            raw_target = label_map[raw_target]
                        elif str(raw_target) in label_map:
                            raw_target = label_map[str(raw_target)]
                    target_val = str(raw_target)

                # Handle datasets where target_col is a list of message dicts
                # (e.g. conversational datasets with {role, content} structure).
                # Extract the last assistant message as the golden reference.
                if isinstance(raw_target, list):
                    assistant_msgs = [
                        m.get("content", "")
                        for m in raw_target
                        if isinstance(m, dict) and m.get("role") == "assistant"
                    ]
                    target_val = assistant_msgs[-1].strip() if assistant_msgs else None

                self.eval_data.append({"input": input_val, "reference": target_val})

            _log(f"Loaded {len(self.eval_data)} qualitative eval samples.")

        except Exception as exc:
            _log(f"Warning: Failed to load qualitative eval dataset: {exc}")
            self.eval_data = []

    # ──────────────────────────────────────────────────────────────────────────
    # Sliding Window
    # ──────────────────────────────────────────────────────────────────────────

    def _get_eval_window(self) -> list:
        """Return the next sliding window of eval samples and advance cursor."""
        pool = self.eval_data
        pool_size = len(pool)
        start = self._eval_cursor
        end = start + self._eval_samples

        if end <= pool_size:
            window = pool[start:end]
        else:
            window = pool[start:] + pool[: end - pool_size]

        self._eval_cursor = end % pool_size
        return window

    # ──────────────────────────────────────────────────────────────────────────
    # Generation
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_responses(self, model, window: list) -> tuple:
        """
        Generate model responses for the eval window.

        Returns (predictions, references) as parallel lists.
        model is temporarily set to .eval() and restored to .train() after.
        """
        model.eval()
        predictions = []
        references = []

        gen_config = GenerationConfig(
            max_new_tokens=self._max_new_tokens,
            do_sample=False,  # Deterministic for reproducible evals
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        max_seq_length = self.config.get("model", {}).get("max_seq_length", 512)

        for i, sample in enumerate(window):
            try:
                prompt_text = self._prompt_template.render(**sample)
                inputs = self.tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                ).to(self.device)

                prompt_token_len = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    output_ids = model.generate(**inputs, generation_config=gen_config)

                generated_ids = output_ids[0, prompt_token_len:]
                response = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                ).strip()

                predictions.append(response)
                references.append(sample.get("reference"))

            except Exception as exc:
                _log(f"  Warning: generation failed for eval sample {i}: {exc}")
                predictions.append("")
                references.append(sample.get("reference"))

        model.train()
        return predictions, references

    # ──────────────────────────────────────────────────────────────────────────
    # Public Entry Point
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, model, step: int) -> dict:
        """
        Run qualitative evaluation and return a flat dict of qual_* metrics.

        Args:
            model : The PEFT model (in training mode — temporarily switched
                    to eval mode for generation, then restored).
            step  : Current optimizer step (for logging only).

        Returns:
            dict[str, float] with all keys prefixed "qual_".
            Returns {} if disabled, no data, or all generations failed.
        """
        if not self.enabled or not self.eval_data:
            return {}

        _log(f"--- Qualitative Eval @ Step {step} ({self._method}) ---")
        t_start = time.monotonic()

        window = self._get_eval_window()
        predictions, references = self._generate_responses(model, window)

        if not any(p.strip() for p in predictions):
            _log("  Warning: all qualitative eval generations were empty — skipping metrics.")
            return {}

        # Run the selected strategy metric
        strategy_metrics = {}
        try:
            strategy_metrics = self._metric.compute(predictions, references)
        except Exception as exc:
            _log(f"  Warning: strategy metric ({self._method}) failed: {exc}")

        # Run universal metrics (always, zero extra compute)
        universal_metrics = _compute_universal_qualitative_metrics(predictions)

        combined = {**strategy_metrics, **universal_metrics}

        elapsed = time.monotonic() - t_start
        _log(f"  Qualitative eval completed in {elapsed:.2f}s")
        for k, v in combined.items():
            if isinstance(v, float):
                _log(f"  {k}: {v:.4f}")
            else:
                _log(f"  {k}: {v}")
        _log("--- End Qualitative Eval ---")

        return combined
