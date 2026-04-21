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
- structured_slot_coverage : Evaluates whether the model successfully
                         verbalizes all attributes from a structured Meaning
                         Representation (MR). Supports per-slot tracking,
                         negation-aware boolean checkers, pinned anchor
                         evaluation, and perfect-coverage-rate reporting.
                         Dataset-specific options are driven from the YAML
                         config under testing_strategy.e2e_nlg_options.

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
    def compute(self, predictions: list, references: list, inputs: list = None) -> dict:
        """
        Args:
            predictions : List of model-generated text strings.
            references  : List of golden reference strings, or list of None
                          for reference-free strategies.
            inputs      : List of input strings used for generation.
        Returns:
            dict[str, float] — all keys prefixed with "qual_".
        """
        ...
        
    def compute_consistency(self, predictions_matrix: list, references: list, inputs: list = None) -> dict:
        """
        Base default: Returns an empty dict.
        Subclasses can override this to evaluate consistency over multiple generation runs for the same input.
        
        Args:
            predictions_matrix: List of Lists of strings. Shape: [len(window), consistency_runs].
            references: List of reference strings.
            inputs: List of input strings.
        Returns:
            dict[str, float] with consistency metrics.
        """
        return {}

    def compute_pinned(self, predictions_matrix: list, inputs: list) -> dict:
        """
        Base default: Returns an empty dict.
        Subclasses override this to compute metrics on the fixed pinned anchor set.

        Args:
            predictions_matrix: List of Lists of strings. Shape: [n_anchors, consistency_runs].
            inputs: List of pinned MR input strings.
        Returns:
            dict[str, float] with pinned anchor metrics, all prefixed "qual_pinned_".
        """
        return {}


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
            self._available = True
        except ImportError:
            self._available = False
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            if not self._available:
                raise ImportError(
                    "sentence-transformers is required for semantic_similarity. "
                    "Install it: pip install sentence-transformers"
                )
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name, device="cpu")
        return self._model

    def compute(self, predictions: list, references: list, inputs: list = None) -> dict:
        if not predictions or not references:
            return {}

        valid_pairs = [
            (p, r) for p, r in zip(predictions, references)
            if p and r and p.strip() and r.strip()
        ]
        if not valid_pairs:
            return {}

        model = self._get_model()
        preds_clean, refs_clean = zip(*valid_pairs)

        import torch
        pred_embs = model.encode(list(preds_clean), convert_to_tensor=True, device="cpu")
        ref_embs  = model.encode(list(refs_clean),  convert_to_tensor=True, device="cpu")

        cos_sims = torch.nn.functional.cosine_similarity(pred_embs, ref_embs, dim=1)
        return {"qual_semantic_similarity": float(cos_sims.mean().item())}


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: Keyword Density (Domain Vocabulary Adoption)
# ─────────────────────────────────────────────────────────────────────────────

class KeywordDensityMetric(QualitativeMetric):
    """
    Measures how densely the model uses a provided list of domain keywords,
    plus Type-Token Ratio (TTR) and Hapax Ratio (word uniqueness).

    Reference-free: does not need golden answers.

    Returns keys:
        qual_keyword_density : fraction of keywords present in mean response
        qual_type_token_ratio : unique tokens / total tokens (lexical diversity)
        qual_hapax_ratio      : fraction of words appearing exactly once
    """

    def __init__(self, keywords: list = None):
        self._keywords = [k.lower() for k in (keywords or [])]

    def compute(self, predictions: list, references: list, inputs: list = None) -> dict:
        if not predictions:
            return {}

        keyword_hits = []
        ttr_scores = []
        hapax_scores = []

        for pred in predictions:
            text = (pred or "").strip().lower()
            words = text.split()
            if not words:
                continue

            # Keyword density
            if self._keywords:
                hits = sum(1 for kw in self._keywords if kw in text)
                keyword_hits.append(hits / len(self._keywords))

            # Type-Token Ratio
            word_counts = collections.Counter(words)
            ttr_scores.append(len(word_counts) / len(words))

            # Hapax ratio
            hapax = sum(1 for c in word_counts.values() if c == 1)
            hapax_scores.append(hapax / len(words))

        metrics = {}
        if keyword_hits:
            metrics["qual_keyword_density"] = sum(keyword_hits) / len(keyword_hits)
        if ttr_scores:
            metrics["qual_type_token_ratio"] = sum(ttr_scores) / len(ttr_scores)
        if hapax_scores:
            metrics["qual_hapax_ratio"] = sum(hapax_scores) / len(hapax_scores)

        return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3: Structural CoT (Chain-of-Thought Logic Anchor Counting)
# ─────────────────────────────────────────────────────────────────────────────

class StructuralCoTMetric(QualitativeMetric):
    """
    Counts occurrences of "logic anchor" phrases (regex patterns) in generated
    text and measures the mean character length of reasoning steps between them.

    Default anchors detect common English chain-of-thought discourse markers
    (\"first\", \"therefore\", \"step N:\", \"because\", etc.).

    Returns keys:
        qual_cot_anchor_count_mean : mean anchor count per response
        qual_cot_step_length_mean  : mean chars between consecutive anchors
        qual_cot_coverage_rate     : fraction of responses with ≥1 anchor
    """

    _DEFAULT_ANCHORS = [
        r"\bfirst\b",
        r"\bsecond\b",
        r"\bthird\b",
        r"\btherefore\b",
        r"\bthus\b",
        r"\bbecause\b",
        r"\bsince\b",
        r"\bhence\b",
        r"\bin conclusion\b",
        r"\bfinally\b",
        r"\bstep\s+\d+",
        r"\b\d+\.\s",
    ]

    def __init__(self, logic_anchors: list = None):
        raw_patterns = logic_anchors if logic_anchors is not None else self._DEFAULT_ANCHORS
        self._patterns = []
        for pattern_str in raw_patterns:
            try:
                self._patterns.append(re.compile(pattern_str, re.IGNORECASE))
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

    def compute(self, predictions: list, references: list, inputs: list = None) -> dict:
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
# Strategy 4: Structured Slot Coverage (E2E NLG / Consistency Validation)
# ─────────────────────────────────────────────────────────────────────────────

class StructuredSlotCoverageMetric(QualitativeMetric):
    """
    Evaluates whether the model successfully verbalizes all attributes specified
    in a structured Meaning Representation (MR).

    Parses slots shaped exactly like: `name[The Punter] | food[Indian]`
    and checks if the target value string is included in the generation.

    Enhanced features (all config-driven, all optional):
    ─────────────────────────────────────────────────────
    • Per-slot coverage tracking  : reports qual_slot_<name>_coverage for each
                                    slot in e2e_nlg_options.track_per_slot
    • Negation-aware checkers     : maps slot_name → checker_type via
                                    e2e_nlg_options.slot_checkers
                                    "boolean_negation" detects positive/negative
                                    phrasing rather than substring match
    • Perfect coverage rate       : qual_perfect_coverage_rate = fraction of
                                    samples achieving 100% slot coverage
    • familyFriendly inversion    : qual_slot_familyFriendly_inversion_rate =
                                    fraction where model semantics contradict MR
    • Pinned anchor evaluation    : compute_pinned() evaluates a fixed anchor
                                    set for cross-checkpoint comparison
    """

    def __init__(self, slot_keywords: dict = None, e2e_nlg_options: dict = None):
        self._pattern = re.compile(r"(\w[\w ]*)\[([^\]]+)\]")
        opts = e2e_nlg_options or {}
        self._track_per_slot: list = opts.get("track_per_slot") or []
        self._slot_checkers: dict = opts.get("slot_checkers") or {}

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _parse_mr(self, mr_string: str) -> dict:
        """Parse 'name[The Punter], food[Chinese]' → {'name': 'The Punter', 'food': 'Chinese'}."""
        slots = {}
        for match in self._pattern.finditer(mr_string):
            attr = match.group(1).strip()
            val  = match.group(2).strip()
            # For duplicate slots (e.g. two customer rating values), keep first
            if attr not in slots:
                slots[attr] = val
        return slots

    def _check_slot(self, slot_name: str, slot_value: str, text: str) -> bool:
        """
        Check whether a single slot is covered in the generated text.
        Dispatches to the appropriate checker based on slot_checkers config.
        """
        checker = self._slot_checkers.get(slot_name, "substring")
        if checker == "boolean_negation":
            return self._check_boolean_negation(slot_value, text)

        # Fuzzy Price Matching to combat stringent substring dropout
        if slot_name.lower() == "pricerange":
            PRICE_ALIASES = {
                "£20-25": ["£20", "20-25", "twenty"],
                "less than £20": ["under £20", "below £20", "less than 20"],
                "more than £30": ["over £30", "above £30", "more than 30"],
            }
            # Check exact match first
            if slot_value.lower() in text:
                return True
            # Fallback to aliases
            if slot_value in PRICE_ALIASES:
                if any(alias in text for alias in PRICE_ALIASES[slot_value]):
                    return True
                
        # Default: simple substring match on the slot value
        return slot_value.lower() in text

    def _check_boolean_negation(self, slot_value: str, text: str) -> bool:
        """
        Negation-aware checker for boolean yes/no slots (e.g. familyFriendly).

        For slot_value='yes': output must contain a positive phrase and NOT a
        negative phrase.
        For slot_value='no': output must contain a negative phrase, OR must
        contain NO positive phrase (absence of positive = implicit negative).

        This fixes the systematic under-counting caused by naive string
        matching on 'familyFriendly', which would score "not family friendly"
        and "family friendly" identically.
        """
        positive_phrases = [
            "family friendly", "family-friendly",
            "kid friendly", "kid-friendly",
            "child friendly", "child-friendly",
            "children friendly", "children-friendly",
            "family-orientated", "family orientated",
        ]
        negative_phrases = [
            "not family", "not child", "not kid",
            "not family-friendly", "not family friendly",
            "non family", "non-family",
            "adults only", "adult only",
            "no children", "not child-friendly",
        ]
        has_positive = any(p in text for p in positive_phrases)
        has_negative = any(n in text for n in negative_phrases)

        if slot_value.lower() == "yes":
            return has_positive and not has_negative
        else:  # 'no'
            return has_negative or not has_positive

    def _is_valid_restaurant_description(self, name_value: str, text: str) -> bool:
        """Fluency gate to combat false positives in early hallucinations."""
        if name_value and name_value.lower() in text:
            return True
            
        domain_keywords = [
            'coffee shop', 'restaurant', 'pub', 'food', 'menu', 
            'eat', 'serve', 'taste', 'price', 'bar', 'cafe', 'café'
        ]
        if any(keyword in text for keyword in domain_keywords):
            return True
            
        return False

    def _score_sample(self, mr_string: str, text: str) -> tuple:
        """
        Score a single (MR, generated text) pair.

        Returns:
            (coverage: float, per_slot: dict[str, bool | None])
            per_slot maps each slot_name → True/False/None (None if not in MR).
        """
        text_lower = text.lower()
        slots = self._parse_mr(mr_string)
        if not slots:
            return 0.0, {}

        # Fluency Gate: Kill false positive metrics if the text is pure conversational hallucination
        if not self._is_valid_restaurant_description(slots.get("name", ""), text_lower):
            return 0.0, {slot_name: False for slot_name in slots.keys()}

        hits = 0
        per_slot: dict = {}
        for slot_name, slot_val in slots.items():
            covered = self._check_slot(slot_name, slot_val, text_lower)
            per_slot[slot_name] = covered
            if covered:
                hits += 1

        return hits / len(slots), per_slot

    # ── Public API ────────────────────────────────────────────────────────────

    def compute(self, predictions: list, references: list, inputs: list = None) -> dict:
        """Compute mean slot coverage + per-slot coverage over a flat predictions list."""
        if not predictions or not inputs:
            return {}

        coverage_scores = []
        per_slot_hits: dict = collections.defaultdict(list)

        for pred, inp in zip(predictions, inputs):
            text = (pred or "").strip().lower()
            coverage, per_slot = self._score_sample(inp or "", text)
            coverage_scores.append(coverage)
            for slot_name, covered in per_slot.items():
                per_slot_hits[slot_name].append(float(covered))

        metrics = {}
        if coverage_scores:
            metrics["qual_slot_coverage_mean"] = sum(coverage_scores) / len(coverage_scores)
            metrics["qual_perfect_coverage_rate"] = sum(
                1 for c in coverage_scores if c >= 1.0
            ) / len(coverage_scores)

        # Per-slot metrics for tracked slots
        for slot_name in self._track_per_slot:
            hits = per_slot_hits.get(slot_name)
            if hits:
                safe_key = re.sub(r"[^a-zA-Z0-9]", "_", slot_name).rstrip("_")
                metrics[f"qual_slot_{safe_key}_coverage"] = sum(hits) / len(hits)

        return metrics

    def compute_consistency(self, predictions_matrix: list, references: list, inputs: list = None) -> dict:
        """
        Evaluate consistency across multiple generation runs per sample.

        Computes:
          qual_consistency_score_mean         : fraction of runs with ≥85% coverage per sample, averaged
          qual_perfect_coverage_rate          : fraction of samples where mean coverage = 100%
          qual_slot_<name>_coverage           : per-slot coverage averaged over all runs
          qual_slot_familyFriendly_inversion_rate : fraction of familyFriendly samples where
                                                    model semantics contradict the MR
        Logs a markdown table of MR → first-run output → coverage.
        """
        if not predictions_matrix or not inputs:
            return {}

        n_runs = len(predictions_matrix[0]) if predictions_matrix else 1

        # Log markdown table header
        _log("")
        _log("| MR Input | Generated Output (Sample Run) | Coverage | Perfect? | Diagnostics |")
        _log("|---|---|---|---|---|")

        full_coverage_counts = []
        mean_coverage_per_sample = []
        per_slot_hits_all: dict = collections.defaultdict(list)
        # familyFriendly inversion tracking
        ff_inversion_count = 0
        ff_total_count = 0

        for pred_runs, inp in zip(predictions_matrix, inputs):
            mr_string = (inp or "")
            slots = self._parse_mr(mr_string)
            if not slots:
                full_coverage_counts.append(0.0)
                mean_coverage_per_sample.append(0.0)
                continue

            runs_with_full_coverage = 0
            sample_run_text = ""
            sample_coverage = 0.0
            sample_per_slot = {}
            run_coverages = []
            per_slot_run_hits: dict = collections.defaultdict(list)

            # familyFriendly inversion: check if the model inverts the semantics
            ff_slot_val = slots.get("familyFriendly")
            if ff_slot_val and self._slot_checkers.get("familyFriendly") == "boolean_negation":
                ff_total_count += 1
                # Count inversion: positive phrasing when MR says 'no', or negative when MR says 'yes'
                positive_phrases = [
                    "family friendly", "family-friendly", "kid friendly",
                    "kid-friendly", "child friendly", "child-friendly",
                    "children friendly", "children-friendly",
                ]
                negative_phrases = [
                    "not family", "not child", "not kid",
                    "not family-friendly", "non family", "non-family",
                    "adults only",
                ]
                inverted_runs = 0
                for text in pred_runs:
                    text_lower = (text or "").strip().lower()
                    has_pos = any(p in text_lower for p in positive_phrases)
                    has_neg = any(n in text_lower for n in negative_phrases)
                    if ff_slot_val.lower() == "yes":
                        inverted = has_neg and not has_pos
                    else:
                        inverted = has_pos and not has_neg
                    if inverted:
                        inverted_runs += 1
                if inverted_runs > (n_runs / 2):  # majority of runs inverted = inverted sample
                    ff_inversion_count += 1

            for run_idx, text in enumerate(pred_runs):
                clean_text = (text or "").strip().lower()
                coverage, per_slot = self._score_sample(mr_string, clean_text)
                run_coverages.append(coverage)

                for slot_name, covered in per_slot.items():
                    per_slot_run_hits[slot_name].append(float(covered))

                if coverage >= 0.85:
                    runs_with_full_coverage += 1

                if not sample_run_text:
                    sample_run_text = text.strip().replace("\n", " ")
                    sample_coverage = coverage
                    sample_per_slot = per_slot

            full_coverage_counts.append(runs_with_full_coverage / n_runs)
            mean_cov = sum(run_coverages) / len(run_coverages) if run_coverages else 0.0
            mean_coverage_per_sample.append(mean_cov)

            # Aggregate per-slot hits (mean across runs for this sample)
            for slot_name, hits in per_slot_run_hits.items():
                per_slot_hits_all[slot_name].append(sum(hits) / len(hits))

            # Diagnostic string generation
            if sample_coverage >= 1.0:
                perfect_str = "✅"
                missing_str = "-"
            elif sample_coverage == 0.0 and sample_per_slot and all(v is False for v in sample_per_slot.values()):
                perfect_str = "❌"
                missing_str = "**[HALLUCINATION IGNORED]**"
            else:
                perfect_str = "❌"
                missed = [k for k, v in sample_per_slot.items() if not v]
                missing_str = f"Missing: [{', '.join(missed)}]"

            # Log table row
            _log(f"| {mr_string} | {sample_run_text} | {sample_coverage*100:.0f}% | {perfect_str} | {missing_str} |")

        _log("")

        metrics = {}
        if full_coverage_counts:
            metrics["qual_consistency_score_mean"] = (
                sum(full_coverage_counts) / len(full_coverage_counts)
            )
        if mean_coverage_per_sample:
            metrics["qual_perfect_coverage_rate"] = sum(
                1 for c in mean_coverage_per_sample if c >= 1.0
            ) / len(mean_coverage_per_sample)

        # Per-slot metrics
        for slot_name in self._track_per_slot:
            hits = per_slot_hits_all.get(slot_name)
            if hits:
                safe_key = re.sub(r"[^a-zA-Z0-9]", "_", slot_name).rstrip("_")
                metrics[f"qual_slot_{safe_key}_coverage"] = sum(hits) / len(hits)

        # familyFriendly inversion rate
        if ff_total_count > 0:
            metrics["qual_slot_familyFriendly_inversion_rate"] = ff_inversion_count / ff_total_count

        return metrics

    def compute_pinned(self, predictions_matrix: list, inputs: list) -> dict:
        """
        Compute metrics on the fixed pinned anchor set.

        Returns:
            qual_pinned_slot_coverage_mean    : mean slot coverage across all anchors & runs
            qual_pinned_perfect_coverage_rate : fraction of anchors with mean coverage = 100%
            qual_pinned_consistency_score     : fraction of runs with ≥85% coverage, averaged

        Logs a dedicated pinned-anchor markdown table.
        """
        if not predictions_matrix or not inputs:
            return {}

        n_runs = len(predictions_matrix[0]) if predictions_matrix else 1

        _log("")
        _log("=== Pinned Anchor Evaluation ===")
        _log("| Anchor MR | Generated Output (Sample Run) | Coverage | Perfect? | Diagnostics |")
        _log("|---|---|---|---|---|")

        all_mean_coverages = []
        all_consistency_scores = []

        for pred_runs, inp in zip(predictions_matrix, inputs):
            mr_string = (inp or "")
            slots = self._parse_mr(mr_string)
            if not slots:
                continue

            run_coverages = []
            runs_with_good_cov = 0
            sample_text = ""
            sample_cov = 0.0
            sample_per_slot = {}

            for text in pred_runs:
                clean_text = (text or "").strip().lower()
                coverage, per_slot = self._score_sample(mr_string, clean_text)
                run_coverages.append(coverage)
                if coverage >= 0.85:
                    runs_with_good_cov += 1
                if not sample_text:
                    sample_text = text.strip().replace("\n", " ")
                    sample_cov = coverage
                    sample_per_slot = per_slot

            mean_cov = sum(run_coverages) / len(run_coverages) if run_coverages else 0.0
            all_mean_coverages.append(mean_cov)
            all_consistency_scores.append(runs_with_good_cov / n_runs)
            
            # Diagnostic string generation
            if sample_cov >= 1.0:
                perfect_str = "✅"
                missing_str = "-"
            elif sample_cov == 0.0 and sample_per_slot and all(v is False for v in sample_per_slot.values()):
                perfect_str = "❌"
                missing_str = "**[HALLUCINATION IGNORED]**"
            else:
                perfect_str = "❌"
                missed = [k for k, v in sample_per_slot.items() if not v]
                missing_str = f"Missing: [{', '.join(missed)}]"

            _log(f"| {mr_string} | {sample_text} | {sample_cov*100:.0f}% | {perfect_str} | {missing_str} |")

        _log("")

        metrics = {}
        if all_mean_coverages:
            metrics["qual_pinned_slot_coverage_mean"] = (
                sum(all_mean_coverages) / len(all_mean_coverages)
            )
            metrics["qual_pinned_perfect_coverage_rate"] = sum(
                1 for c in all_mean_coverages if c >= 1.0
            ) / len(all_mean_coverages)
        if all_consistency_scores:
            metrics["qual_pinned_consistency_score"] = (
                sum(all_consistency_scores) / len(all_consistency_scores)
            )

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

    if method == "structured_slot_coverage":
        slot_keywords = ts_cfg.get("slot_keywords")
        e2e_nlg_options = ts_cfg.get("e2e_nlg_options") or {}
        return StructuredSlotCoverageMetric(
            slot_keywords=slot_keywords,
            e2e_nlg_options=e2e_nlg_options,
        )

    raise ValueError(
        f"Unknown testing_strategy.method: '{method}'. "
        f"Supported values: semantic_similarity | keyword_density | structural_cot | structured_slot_coverage"
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
            self._pinned_eval_data = []
            self.eval_interval = 50 # Default safe value even if disabled
            return

        # --- Build the metric strategy ---
        self._method = (ts_cfg.get("method") or "").strip().lower()
        self._metric: QualitativeMetric = _build_metric(ts_cfg)

        # --- Evaluation cadence ---
        self.eval_interval: int = int(ts_cfg.get("eval_interval", 50))
        self._eval_samples: int = int(ts_cfg.get("eval_samples", 20))
        self._max_new_tokens: int = int(ts_cfg.get("max_new_tokens", 150))

        # --- Consistency loops ---
        self._consistency_runs: int = int(ts_cfg.get("consistency_runs", 1))
        self._consistency_temperature: float = float(ts_cfg.get("consistency_temperature", 0.7))
        self._eval_batch_size: int = int(ts_cfg.get("eval_batch_size", 16))

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

        # --- Load pinned anchor samples (structured_slot_coverage only) ---
        self._pinned_eval_data: list = []
        e2e_opts = ts_cfg.get("e2e_nlg_options") or {}
        raw_anchors = e2e_opts.get("pinned_anchors") or []
        if raw_anchors and self._method == "structured_slot_coverage":
            self._pinned_eval_data = [
                {"input": mr_string, "reference": None}
                for mr_string in raw_anchors
                if mr_string and mr_string.strip()
            ]
            _log(f"Loaded {len(self._pinned_eval_data)} pinned anchor samples for cross-checkpoint comparison.")

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
        if dataset_cfg.get("data_files"):
            ds_kwargs["data_files"] = dataset_cfg["data_files"]
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

    def reset_runtime_state(self):
        """Reset sliding-window state without reloading the dataset."""
        self._eval_cursor = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Generation
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_responses(self, model, window: list, do_sample: bool = False, temperature: float = 1.0, num_return_sequences: int = 1) -> tuple:
        """
        Generate model responses for the eval window.

        Returns (predictions, references) as parallel lists.
        Generates in high-speed batches seamlessly.
        """
        was_training = model.training
        model.eval()
        predictions = []
        references = []

        gen_config_kwargs = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_config_kwargs["temperature"] = temperature
            
        gen_config = GenerationConfig(**gen_config_kwargs)
        max_seq_length = self.config.get("model", {}).get("max_seq_length", 512)

        # Batch generation with decoder-only models natively requires left padding
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        batch_size = getattr(self, "_eval_batch_size", 16)
        
        import math
        total_batches = math.ceil(len(window) / batch_size)

        for batch_i in range(0, len(window), batch_size):
            batch_window = window[batch_i:batch_i + batch_size]
            prompt_texts = []
            
            for sample in batch_window:
                prompt_texts.append(self._prompt_template.render(**sample))
                references.append(sample.get("reference"))

            try:
                inputs = self.tokenizer(
                    prompt_texts,
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                    padding=True,
                ).to(self.device)

                prompt_token_len = inputs["input_ids"].shape[1]

                with torch.inference_mode():
                    output_ids = model.generate(**inputs, generation_config=gen_config)

                # Exactly slice off only the newly generated dimension identically for all
                generated_ids = output_ids[:, prompt_token_len:]
                responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                for response in responses:
                    predictions.append(response.strip())

                del inputs
                del output_ids
                del generated_ids
                
            except Exception as exc:
                _log(f"  Warning: batch generation failed for batch offsets {batch_i}: {exc}")
                # Fallback purely to maintain identical lengths securely
                for _ in batch_window:
                    for _ in range(num_return_sequences):
                        predictions.append("")

        self.tokenizer.padding_side = original_padding_side
        if was_training:
            model.train()
        else:
            model.eval()
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
        inputs = [sample.get("input", "") for sample in window]
        
        # Generalized Consistency Generation (GPU Parallelized)
        if self._consistency_runs > 1:
            _log(f"  Running GPU-parallel matrix generation: {self._consistency_runs} runs per sample (temp={self._consistency_temperature})")
            
            # Generate all sequences in a single, massively parallel GPU forward pass
            flat_preds, references = self._generate_responses(
                model, window, do_sample=True, temperature=self._consistency_temperature, num_return_sequences=self._consistency_runs
            )
            
            # Reshape 1D flat tensor response back into 2D (window_size x consistency_runs)
            predictions_matrix = []
            for i in range(len(window)):
                start_idx = i * self._consistency_runs
                end_idx = start_idx + self._consistency_runs
                predictions_matrix.append(flat_preds[start_idx:end_idx])
                
            predictions = [runs[0] for runs in predictions_matrix if runs]  # first run for universal metrics
        else:
            predictions, references = self._generate_responses(model, window, do_sample=False)
            predictions_matrix = None

        if not any(p.strip() for p in predictions):
            _log("  Warning: all qualitative eval generations were empty — skipping metrics.")
            return {}

        # Run the selected strategy metric
        strategy_metrics = {}
        try:
            strategy_metrics = self._metric.compute(predictions, references, inputs=inputs)
            if self._consistency_runs > 1 and predictions_matrix is not None:
                consistency_metrics = self._metric.compute_consistency(predictions_matrix, references, inputs=inputs)
                strategy_metrics.update(consistency_metrics)
        except Exception as exc:
            _log(f"  Warning: strategy metric ({self._method}) failed: {exc}")

        # Run pinned anchor evaluation (structured_slot_coverage only)
        if self._pinned_eval_data and self._consistency_runs > 1:
            try:
                _log(f"  Running pinned anchor generation: {len(self._pinned_eval_data)} anchors × {self._consistency_runs} runs")
                pinned_matrix = [[] for _ in range(len(self._pinned_eval_data))]
                for _ in range(self._consistency_runs):
                    preds, _ = self._generate_responses(
                        model, self._pinned_eval_data,
                        do_sample=True, temperature=self._consistency_temperature
                    )
                    for i, p in enumerate(preds):
                        pinned_matrix[i].append(p)

                pinned_inputs = [s["input"] for s in self._pinned_eval_data]
                pinned_metrics = self._metric.compute_pinned(pinned_matrix, pinned_inputs)
                strategy_metrics.update(pinned_metrics)

                for k, v in pinned_metrics.items():
                    if isinstance(v, float):
                        _log(f"  {k}: {v:.4f}")
            except Exception as exc:
                _log(f"  Warning: pinned anchor evaluation failed: {exc}")

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

        # Memory cleanup after generation passes
        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        return combined
