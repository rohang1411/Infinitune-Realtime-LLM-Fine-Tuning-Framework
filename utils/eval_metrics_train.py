import time
import math
import re
import string
import collections
import torch
from jinja2 import Template
from datasets import load_dataset
from transformers import GenerationConfig

def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _log(msg):
    print(f"[{_ts()}][EVAL] {msg}", flush=True)


class _QAFactEvalScorer:
    """Factual consistency scorer via extractive QA.

    For each key span in the *source* text, a QA model is asked to find that
    span inside the *generated* text.  Token-level F1 between the source span
    and the model's extracted answer gives a per-span score; the final score
    is the mean over all spans.

    Score range: 0 (completely inconsistent) → 1 (fully consistent).

    The QA pipeline is loaded lazily on the first call to score() so that
    importing this module never triggers a model download.
    """

    DEFAULT_MODEL = "deepset/minilm-uncased-squad2"

    def __init__(self, model_name: str = None):
        self._model_name = model_name or self.DEFAULT_MODEL
        self._pipeline = None  # lazy init

    def _init_pipeline(self):
        if self._pipeline is not None:
            return
        from transformers import pipeline as hf_pipeline
        _log(f"QAFactEval: loading QA model '{self._model_name}' (first use — may download ~120 MB)")
        self._pipeline = hf_pipeline(
            "question-answering",
            model=self._model_name,
            tokenizer=self._model_name,
        )
        _log("QAFactEval: QA model ready.")

    @staticmethod
    def _extract_spans(text: str, max_spans: int = 5) -> list:
        """Split text into up to *max_spans* sentence-level spans."""
        # Split on sentence-ending punctuation; fall back to the full text.
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if not sentences:
            return [text.strip()]
        # Evenly sample up to max_spans to cover the whole document.
        if len(sentences) <= max_spans:
            return sentences
        step = len(sentences) / max_spans
        return [sentences[int(i * step)] for i in range(max_spans)]

    @staticmethod
    def _token_f1(pred: str, ref: str) -> float:
        """Compute token-level F1 (same formula as SQuAD official eval)."""
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        common = collections.Counter(pred_tokens) & collections.Counter(ref_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            return 0.0
        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    def score(self, source: str, generated: str) -> float:
        """Return a QAFactEval score in [0, 1] for one (source, generated) pair."""
        self._init_pipeline()
        if not generated.strip():
            return 0.0
        spans = self._extract_spans(source)
        f1_scores = []
        for span in spans:
            question = f"What does the text say about: {span[:120]}?"
            try:
                result = self._pipeline(question=question, context=generated)
                extracted = result.get("answer", "")
                f1_scores.append(self._token_f1(extracted, span))
            except Exception:
                # If the context is too short the QA model may error — skip.
                f1_scores.append(0.0)
        return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

def _normalized_aauc_from_history(history):
    """Trapezoidal area under accuracy vs training step, divided by step span.
    history is a list of (step, accuracy) in chronological order.
    Single point: returns that accuracy. Two or more: normalized AUC.
    """
    if not history:
        return 0.0
    if len(history) == 1:
        return float(history[0][1])
    raw = 0.0
    for i in range(len(history) - 1):
        s0, a0 = history[i]
        s1, a1 = history[i + 1]
        ds = s1 - s0
        if ds > 0:
            raw += 0.5 * (float(a0) + float(a1)) * ds
    span = history[-1][0] - history[0][0]
    if span <= 0:
        return float(history[-1][1])
    return raw / float(span)

# Forgetting: higher is better vs lower is better (used only when compute_forgetting is on).
_HIGHER_IS_BETTER = frozenset(
    {"accuracy", "exact_match", "f1", "mcc", "kappa"}
)
_LOWER_IS_BETTER = frozenset({"eval_loss", "perplexity"})


def _default_metric_flags(strategy: str):
    """Strategy-appropriate defaults so generative / open-ended data does not
    run confusion-matrix metrics unless the user explicitly enables them."""
    loss = {"compute_loss": True}
    _online = {
        # Peak-vs-current drop (continual-learning style). Off by default: misleading
        # if eval window slides or data distribution shifts.
        "compute_forgetting": False,
        # Wall-clock seconds since the *previous* eval finished (training + I/O between evals).
        "compute_update_latency": False,
    }
    if strategy == "class_match":
        return {
            **loss,
            **_online,
            "compute_accuracy": True,
            "compute_backward_transfer": True,
            "compute_backward_transfer": True,
            "compute_exact_match": True,
            "compute_f1": True,
            "compute_mcc": True,
            "compute_kappa": True,
            "compute_qafacteval": False,
        }
    if strategy == "regex_extract":
        return {
            **loss,
            **_online,
            "compute_accuracy": True,
            "compute_backward_transfer": True,
            "compute_backward_transfer": True,
            "compute_exact_match": True,
            # F1 / MCC / kappa are usually misleading for free-form or
            # high-cardinality extracted answers unless labels are discrete.
            "compute_f1": False,
            "compute_mcc": False,
            "compute_kappa": False,
            "compute_qafacteval": False,
        }
    # perplexity-only or unknown strategy: forward-pass metrics only
    return {
        **loss,
        **_online,
        "compute_accuracy": False,
        "compute_backward_transfer": False,
        "compute_backward_transfer": False,
        "compute_exact_match": False,
        "compute_f1": False,
        "compute_mcc": False,
        "compute_kappa": False,
        "compute_qafacteval": False,
    }


def _merge_metric_flags(strategy: str, eval_cfg: dict) -> dict:
    """Merge user `evaluation.metrics` onto strategy defaults (user wins)."""
    base = _default_metric_flags(strategy)
    user = eval_cfg.get("metrics") or {}
    if not isinstance(user, dict):
        return base
    for key in base:
        if key in user and user[key] is not None:
            base[key] = bool(user[key])
    return base


def _forgetting_track_keys(metrics_block: dict):
    """Which scalar metrics to track for forgetting; None = auto (intersection with computed keys)."""
    raw = metrics_block.get("forgetting_track_metrics")
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        return None
    out = []
    for x in raw:
        if x is None:
            continue
        s = str(x).strip()
        if s:
            out.append(s)
    return out or None


class Evaluator:
    """Evaluates model performance using configurable strategies.
    
    Strategies:
      - perplexity   : forward-pass loss → exp(loss)  [always computed]
      - class_match  : generate and check if prediction starts with target class
      - regex_extract: generate, extract answer via regex, compare to gold
    """

    def __init__(self, config, tokenizer, device, tokenize_fn, pad_fn):
        self.config = config
        self.tokenize_fn = tokenize_fn
        self.pad_fn = pad_fn
        self.eval_cfg = config.get('evaluation', {})
        self.enabled = self.eval_cfg.get('enabled', False)
        self.strategy = self.eval_cfg.get('strategy', 'perplexity')
        self.eval_interval = self.eval_cfg.get('eval_interval', 100)
        self.answer_regex = self.eval_cfg.get('answer_regex')
        self.tokenizer = tokenizer
        self.device = device

        # New config-driven parameters (backward-compat: fall back to eval_samples)
        self.eval_pool_size = self.eval_cfg.get(
            'eval_pool_size', self.eval_cfg.get('eval_samples', 50)
        )
        self.eval_batch_size = self.eval_cfg.get('eval_batch_size', self.eval_pool_size)
        self.verbose = self.eval_cfg.get('verbose', False)

        # Which metrics to compute (config + strategy defaults; user overrides win)
        self.metric_flags = _merge_metric_flags(self.strategy, self.eval_cfg)
        # Skip F1/MCC/kappa when too many distinct labels (open-ended generations)
        metrics_block = self.eval_cfg.get("metrics") or {}
        try:
            self.max_distinct_labels = int(
                metrics_block.get("max_distinct_labels_for_structure_metrics", 64)
            )
        except (TypeError, ValueError):
            self.max_distinct_labels = 64
        self.max_distinct_labels = max(2, self.max_distinct_labels)

        # QAFactEval scorer (lazy-loaded; only instantiated when metric is enabled)
        qafacteval_model = metrics_block.get("qafacteval_model") or _QAFactEvalScorer.DEFAULT_MODEL
        self._qafacteval_scorer = (
            _QAFactEvalScorer(qafacteval_model)
            if self.metric_flags.get("compute_qafacteval", False)
            else None
        )

        # Sliding window cursor � tracks position within the eval pool
        self._eval_cursor = 0
        self.past_sample_accuracies = {}  # Tracks max accuracy per eval sample for BWT
        self.past_sample_accuracies = {}  # Tracks max accuracy per eval sample for BWT

        # Forgetting + update latency (online / continual-learning style diagnostics)
        self._best_eval = {}  # metric_name -> running best (peak or valley)
        self._last_eval_end_wall = None  # time.monotonic() after last successful eval
        self._forgetting_track = _forgetting_track_keys(metrics_block)

        # (training_step, accuracy) for normalized AAUC across eval calls
        self._aauc_history = []

        # Pre-compile Jinja2 templates
        preproc = config['preprocessing']
        self.prompt_template = Template(preproc['prompt_template'])
        self.response_template = Template(preproc.get('response_template', ' {{ target }}'))
        self.max_seq_length = config['model'].get('max_seq_length', 512)

        # Pre-load eval data
        self.eval_data = []
        self._regex_missing_warned = False
        if self.enabled:
            self._load_eval_data()
            _log(
                f"Metric flags: loss={self.metric_flags['compute_loss']}, "
                f"accuracy={self.metric_flags['compute_accuracy']}, "
                f"exact_match={self.metric_flags['compute_exact_match']}, "
                f"f1={self.metric_flags['compute_f1']}, mcc={self.metric_flags['compute_mcc']}, "
                f"kappa={self.metric_flags['compute_kappa']}, "
                f"qafacteval={self.metric_flags.get('compute_qafacteval', False)} "
                f"kappa={self.metric_flags['compute_kappa']}, "
                f"forgetting={self.metric_flags.get('compute_forgetting', False)}, "
                f"update_latency={self.metric_flags.get('compute_update_latency', False)} "
                f"(max_distinct_labels={self.max_distinct_labels})"
            )

    def _load_eval_data(self):
        """Load evaluation dataset pool from the configured eval_split."""
        dataset_cfg = self.config['dataset']
        eval_split = dataset_cfg.get('eval_split', 'test')
        col_map = dataset_cfg['column_mapping']
        label_map = dataset_cfg.get('label_map')

        ds_kwargs = {"path": dataset_cfg['name'], "split": eval_split}
        if dataset_cfg.get("data_files"):
            ds_kwargs["data_files"] = dataset_cfg["data_files"]
        if dataset_cfg.get('config_name'):
            ds_kwargs["name"] = dataset_cfg['config_name']

        try:
            dataset = load_dataset(**ds_kwargs)

            # Always shuffle eval data with a fixed seed so the sample is
            # diverse (e.g. IMDb test split is sorted by label — taking the
            # first N would give all-same-class samples and fake 100% accuracy).
            dataset = dataset.shuffle(seed=42)

            for i, example in enumerate(dataset):
                if i >= self.eval_pool_size:
                    break
                input_val = str(example[col_map['input_col']])
                target_val = example[col_map['target_col']]
                if label_map is not None:
                    if target_val in label_map:
                        target_val = label_map[target_val]
                    elif str(target_val) in label_map:
                        target_val = label_map[str(target_val)]
                target_val = str(target_val)
                self.eval_data.append({"input": input_val, "target": target_val})
            _log(f"Loaded {len(self.eval_data)} evaluation samples into pool (split='{eval_split}', batch_size={self.eval_batch_size}).")
        except Exception as e:
            _log(f"Warning: Could not load eval dataset: {e}")
            self.enabled = False

    def _get_eval_window(self):
        """Return the next sliding window of eval samples and advance cursor."""
        pool_size = len(self.eval_data)
        start = self._eval_cursor
        end = start + self.eval_batch_size

        if end <= pool_size:
            window = self.eval_data[start:end]
        else:
            # Wrap around: take the tail + the head
            window = self.eval_data[start:] + self.eval_data[:end - pool_size]

        self._eval_cursor = end % pool_size
        return window, start

    def evaluate(self, model, step):
        """Run evaluation and return a metrics dict (keys depend on config flags)."""
        if not self.enabled or not self.eval_data:
            return None

        t_wall_start = time.monotonic()
        mf = self.metric_flags

        # Select the current eval window (sliding)
        eval_window, window_start = self._get_eval_window()
        window_end = (window_start + len(eval_window) - 1) % len(self.eval_data)

        _log(f"--- Eval @ Step {step} | samples [{window_start}..{window_end}] ({len(eval_window)} samples) ---")
        model.eval()
        metrics = {}

        if mf.get("compute_update_latency", False) and self._last_eval_end_wall is not None:
            try:
                metrics["update_latency_s"] = float(t_wall_start - self._last_eval_end_wall)
            except (TypeError, ValueError):
                pass

        # --- Forward loss / perplexity (safe for any causal-LM data) ---
        if mf.get("compute_loss", True):
            total_loss = 0.0
            count = 0
            for sample in eval_window:
                try:
                    prompt_text = self.prompt_template.render(**sample)
                    response_text = self.response_template.render(**sample)
                    tok = self.tokenize_fn(
                        self.tokenizer, prompt_text, response_text, self.max_seq_length
                    )
                    batch = self.pad_fn([tok], self.tokenizer.pad_token_id, self.device)
                    with torch.no_grad():
                        outputs = model(**batch)
                        total_loss += outputs.loss.item()
                        count += 1
                except Exception as e:
                    _log(f"  Warning: skipping eval sample in loss pass: {e}")
            if count > 0:
                avg_loss = total_loss / count
                metrics["eval_loss"] = avg_loss
                metrics["perplexity"] = min(torch.exp(torch.tensor(avg_loss)).item(), 1e6)

        need_generation = self.strategy in ("class_match", "regex_extract") and any(
            mf.get(k, False)
            for k in (
                "compute_accuracy",
                "compute_exact_match",
                "compute_f1",
                "compute_mcc",
                "compute_kappa",
            )
        )

        if need_generation:
            cfg_max_new_tokens = self.config.get("inference", {}).get("max_new_tokens", 50)
            if self.strategy == "class_match":
                eval_max_new_tokens = min(cfg_max_new_tokens, 10)
            else:
                eval_max_new_tokens = cfg_max_new_tokens

            if self.strategy == "regex_extract" and not self.answer_regex:
                if not self._regex_missing_warned:
                    _log(
                        "  Warning: strategy=regex_extract but answer_regex is missing; "
                        "using raw string comparison for generation metrics (consider setting metrics flags)."
                    )
                    self._regex_missing_warned = True

            gold_labels = []
            pred_labels = []
            source_texts = []   # raw input texts for QAFactEval
            generated_responses = []  # full model responses for QAFactEval

            for i, sample in enumerate(eval_window):
                try:
                    prompt_text = self.prompt_template.render(**sample)
                    inputs = self.tokenizer(
                        prompt_text,
                        return_tensors="pt",
                        max_length=self.max_seq_length,
                        truncation=True,
                    ).to(self.device)
                    prompt_token_len = inputs["input_ids"].shape[1]

                    gen_config = GenerationConfig(
                        max_new_tokens=eval_max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                    with torch.no_grad():
                        output_ids = model.generate(**inputs, generation_config=gen_config)

                    generated_ids = output_ids[0, prompt_token_len:]
                    response = self.tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    ).strip()

                    gold = str(sample.get("target", "")).lower().strip()
                    pred = None

                    if self.strategy == "class_match":
                        target_words = gold.split()
                        response_words = response.lower().split()
                        pred_words = response_words[: len(target_words)]
                        pred = " ".join(pred_words)
                    elif self.strategy == "regex_extract" and self.answer_regex:
                        try:
                            pred_match = re.search(self.answer_regex, response)
                            gold_match = re.search(self.answer_regex, sample["target"])
                            if pred_match:
                                pred = pred_match.group(1).strip().lower()
                            if gold_match:
                                gold = gold_match.group(1).strip().lower()
                        except re.error as re_err:
                            _log(f"  Warning: invalid answer_regex: {re_err}")
                            pred = response.lower().strip()
                    else:
                        pred = response.lower().strip()

                    if pred is None:
                        pred = response.lower().strip()

                    gold_labels.append(gold)
                    pred_labels.append(pred)
                    source_texts.append(str(sample.get("input", "")))
                    generated_responses.append(response)

                    # Intelligently clean boundary matrices immediately preserving PyTorch native graphs safely intrinsically beautifully securely effectively
                    del inputs
                    del output_ids
                    del generated_ids

                    if self.verbose:
                        mark = "✓" if pred == gold else "✗"
                        _log(
                            f"  [{mark}] sample {i}: expected='{sample.get('target', '')}' "
                            f"got='{response[:80]}'"
                        )
                except Exception as e:
                    _log(f"  Warning: skipping eval sample {i} in generation pass: {e}")

            total = len(gold_labels)
            if total == 0:
                _log("  Warning: no valid generation samples; skipping generation metrics.")
            else:
                if mf.get("compute_accuracy", False):
                    correct_list = [g == p for g, p in zip(gold_labels, pred_labels)]
                    correct = sum(correct_list)
                    metrics["accuracy"] = correct / total
                    _log(f"  Correct: {correct} / {total}")

                    if mf.get("compute_backward_transfer", False):
                        bwt_sum = 0.0
                        bwt_count = 0
                        for i, is_correct in enumerate(correct_list):
                            sample_idx = (window_start + i) % len(self.eval_data)
                            curr_acc = 1.0 if is_correct else 0.0
                            
                            if sample_idx in self.past_sample_accuracies:
                                max_past = self.past_sample_accuracies[sample_idx]
                                bwt_sum += (curr_acc - max_past)
                                bwt_count += 1
                                if curr_acc > max_past:
                                    self.past_sample_accuracies[sample_idx] = curr_acc
                            else:
                                self.past_sample_accuracies[sample_idx] = curr_acc
                                
                        if bwt_count > 0:
                            metrics["backward_transfer"] = bwt_sum / bwt_count
                    self._aauc_history.append((step, metrics["accuracy"]))
                    metrics["aauc"] = _normalized_aauc_from_history(self._aauc_history)

                    if mf.get("compute_backward_transfer", False):
                        bwt_sum = 0.0
                        bwt_count = 0
                        for i, is_correct in enumerate(correct_list):
                            sample_idx = (window_start + i) % len(self.eval_data)
                            curr_acc = 1.0 if is_correct else 0.0
                            
                            if sample_idx in self.past_sample_accuracies:
                                max_past = self.past_sample_accuracies[sample_idx]
                                bwt_sum += (curr_acc - max_past)
                                bwt_count += 1
                                if curr_acc > max_past:
                                    self.past_sample_accuracies[sample_idx] = curr_acc
                            else:
                                self.past_sample_accuracies[sample_idx] = curr_acc
                                
                        if bwt_count > 0:
                            metrics["backward_transfer"] = bwt_sum / bwt_count

                if mf.get("compute_exact_match", False):
                    _punct_table = str.maketrans("", "", string.punctuation)

                    def _normalize(s):
                        return str(s).lower().translate(_punct_table).strip()

                    em_correct = sum(
                        _normalize(g) == _normalize(p)
                        for g, p in zip(gold_labels, pred_labels)
                    )
                    metrics["exact_match"] = em_correct / total

                structure_requested = any(
                    mf.get(k, False) for k in ("compute_f1", "compute_mcc", "compute_kappa")
                )
                if structure_requested:
                    all_labels = set(gold_labels) | set(pred_labels)
                    n_distinct = len(all_labels)
                    if n_distinct > self.max_distinct_labels:
                        _log(
                            f"  Skipping F1/MCC/kappa: {n_distinct} distinct labels "
                            f"> max_distinct_labels_for_structure_metrics ({self.max_distinct_labels})."
                        )
                    else:
                        try:
                            classes = sorted(all_labels)
                            class_idx = {c: i for i, c in enumerate(classes)}
                            n = len(classes)
                            cm = [[0] * n for _ in range(n)]
                            for g, p in zip(gold_labels, pred_labels):
                                if g in class_idx and p in class_idx:
                                    cm[class_idx[g]][class_idx[p]] += 1

                            if mf.get("compute_f1", False) and n > 0:
                                f1_scores = []
                                for i in range(n):
                                    tp = cm[i][i]
                                    fp = sum(cm[r][i] for r in range(n)) - tp
                                    fn = sum(cm[i][c] for c in range(n)) - tp
                                    prec = tp / max(tp + fp, 1)
                                    rec = tp / max(tp + fn, 1)
                                    f1_scores.append(
                                        2 * prec * rec / max(prec + rec, 1e-9)
                                    )
                                metrics["f1"] = sum(f1_scores) / max(len(f1_scores), 1)

                            if mf.get("compute_mcc", False):
                                t = sum(cm[i][i] for i in range(n))
                                s = total
                                p_k = [
                                    sum(cm[i][c] for c in range(n)) for i in range(n)
                                ]
                                t_k = [
                                    sum(cm[r][i] for r in range(n)) for i in range(n)
                                ]
                                cov_yy = s * s - sum(pk * pk for pk in p_k)
                                cov_xx = s * s - sum(tk * tk for tk in t_k)
                                if cov_yy > 0 and cov_xx > 0:
                                    mcc_num = t * s - sum(
                                        p_k[i] * t_k[i] for i in range(n)
                                    )
                                    metrics["mcc"] = mcc_num / math.sqrt(cov_yy * cov_xx)
                                else:
                                    metrics["mcc"] = 0.0

                            if mf.get("compute_kappa", False):
                                correct = sum(g == p for g, p in zip(gold_labels, pred_labels))
                                p_o = correct / max(total, 1)
                                p_e = sum(
                                    (
                                        sum(cm[i][c] for c in range(n)) / max(total, 1)
                                    )
                                    * (
                                        sum(cm[r][i] for r in range(n)) / max(total, 1)
                                    )
                                    for i in range(n)
                                )
                                metrics["kappa"] = (p_o - p_e) / max(1.0 - p_e, 1e-9)
                        except Exception as e:
                            _log(f"  Warning: structure metrics failed (skipped): {e}")

                if mf.get("compute_qafacteval", False) and self._qafacteval_scorer is not None:
                    try:
                        _log("  Computing QAFactEval scores (this may take a moment)...")
                        qaf_scores = []
                        for src, gen in zip(source_texts, generated_responses):
                            if src and gen:
                                qaf_scores.append(self._qafacteval_scorer.score(src, gen))
                        if qaf_scores:
                            metrics["qafacteval"] = sum(qaf_scores) / len(qaf_scores)
                    except Exception as e:
                        _log(f"  Warning: QAFactEval scoring failed (skipped): {e}")

        # --- Forgetting: drop from running peak (higher-better) or rise from running
        # best minimum (lower-better). Only for known scalar keys; skips unknown types.
        if mf.get("compute_forgetting", False):
            metrics_block = self.eval_cfg.get("metrics") or {}
            track = self._forgetting_track
            if track is None:
                track = [
                    k
                    for k in (
                        "accuracy",
                        "exact_match",
                        "f1",
                        "mcc",
                        "kappa",
                        "eval_loss",
                        "perplexity",
                    )
                    if k in metrics
                ]
            forgetting_components = []
            for key in track:
                if key not in metrics:
                    continue
                if key not in _HIGHER_IS_BETTER and key not in _LOWER_IS_BETTER:
                    _log(
                        f"  Warning: forgetting skipped for '{key}' "
                        f"(not in known higher/lower-is-better sets)."
                    )
                    continue
                try:
                    v = float(metrics[key])
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(v):
                    continue

                if key in _HIGHER_IS_BETTER:
                    prev_peak = self._best_eval.get(key)
                    if prev_peak is None:
                        self._best_eval[key] = v
                        fg = 0.0
                    else:
                        fg = max(0.0, prev_peak - v)
                        if v > prev_peak:
                            self._best_eval[key] = v
                else:
                    prev_best = self._best_eval.get(key)
                    if prev_best is None:
                        self._best_eval[key] = v
                        fg = 0.0
                    else:
                        fg = max(0.0, v - prev_best)
                        if v < prev_best:
                            self._best_eval[key] = v

                fk = f"forgetting_{key}"
                metrics[fk] = fg
                forgetting_components.append(fg)

            if forgetting_components:
                metrics["forgetting_max"] = max(forgetting_components)


        # --- Verbose Manual Verification (Sample Generation) ---
        if self.verbose and not need_generation:
            _log("  Verbose mode: generating sample records for manual verification...")
            cfg_max_new_tokens = self.config.get("inference", {}).get("max_new_tokens", 50)
            
            # Use a small subset of the current window for verbosity
            sample_subset = eval_window[:5] 
            for idx, sample in enumerate(sample_subset):
                try:
                    prompt_text = self.prompt_template.render(**sample)
                    inputs = self.tokenizer(
                        prompt_text,
                        return_tensors="pt",
                        max_length=self.max_seq_length,
                        truncation=True,
                    ).to(self.device)
                    prompt_token_len = inputs["input_ids"].shape[1]

                    gen_config = GenerationConfig(
                        max_new_tokens=cfg_max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                    with torch.no_grad():
                        output_ids = model.generate(**inputs, generation_config=gen_config)

                    response = self.tokenizer.decode(
                        output_ids[0, prompt_token_len:], skip_special_tokens=True
                    ).strip()

                    _log(f"  [SAMPLE {idx}]")
                    _log(f"    Input  : {prompt_text[:150].replace('\\n', ' ')}")
                    _log(f"    Target : {str(sample.get('target', ''))[:150].replace('\\n', ' ')}")
                    _log(f"    Model  : {response}")
                    
                    del inputs
                    del output_ids
                except Exception as e:
                    _log(f"    Warning: skip verbose sample {idx}: {e}")

        self._last_eval_end_wall = time.monotonic()

        model.train()

        for k, v in metrics.items():
            if isinstance(v, float):
                _log(f"  {k}: {v:.4f}")
            else:
                _log(f"  {k}: {v}")
        _log("--- End Eval ---")

        # Aggressive memory cleanup after generation passes
        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics if metrics else None
