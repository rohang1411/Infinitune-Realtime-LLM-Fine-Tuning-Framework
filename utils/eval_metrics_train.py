import time
import math
import re
import torch
from jinja2 import Template
from datasets import load_dataset
from transformers import GenerationConfig

def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _log(msg):
    print(f"[{_ts()}][EVAL] {msg}", flush=True)

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

        # Sliding window cursor — tracks position within the eval pool
        self._eval_cursor = 0

        # Pre-compile Jinja2 templates
        preproc = config['preprocessing']
        self.prompt_template = Template(preproc['prompt_template'])
        self.response_template = Template(preproc.get('response_template', ' {{ target }}'))
        self.max_seq_length = config['model'].get('max_seq_length', 512)

        # Pre-load eval data
        self.eval_data = []
        if self.enabled:
            self._load_eval_data()

    def _load_eval_data(self):
        """Load evaluation dataset pool from the configured eval_split."""
        dataset_cfg = self.config['dataset']
        eval_split = dataset_cfg.get('eval_split', 'test')
        col_map = dataset_cfg['column_mapping']
        label_map = dataset_cfg.get('label_map')

        ds_kwargs = {"path": dataset_cfg['name'], "split": eval_split}
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
        """Run evaluation and return a metrics dict."""
        if not self.enabled or not self.eval_data:
            return None

        # Select the current eval window (sliding)
        eval_window, window_start = self._get_eval_window()
        window_end = (window_start + len(eval_window) - 1) % len(self.eval_data)

        _log(f"--- Eval @ Step {step} | samples [{window_start}..{window_end}] ({len(eval_window)} samples) ---")
        model.eval()
        metrics = {}

        # --- Perplexity (response-only loss, matching training) ---
        total_loss = 0.0
        count = 0

        for sample in eval_window:
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

        avg_loss = total_loss / max(count, 1)
        metrics['eval_loss'] = avg_loss
        metrics['perplexity'] = torch.exp(torch.tensor(avg_loss)).item()

        # --- Strategy-specific evaluation (generation-based) ---
        if self.strategy in ('class_match', 'regex_extract'):
            cfg_max_new_tokens = self.config['inference'].get('max_new_tokens', 50)

            # For classification, the answer is a few words — cap generation to
            # avoid wasting compute and to surface repetition problems honestly.
            if self.strategy == 'class_match':
                eval_max_new_tokens = min(cfg_max_new_tokens, 10)
            else:
                eval_max_new_tokens = cfg_max_new_tokens

            # Collect (gold_label, pred_label) pairs for all samples so we can
            # compute the full confusion matrix in one pass at the end.
            gold_labels = []
            pred_labels = []

            for i, sample in enumerate(eval_window):
                prompt_text = self.prompt_template.render(**sample)
                # Truncate prompt the same way training does so behaviour
                # is consistent (long reviews won't exceed context window).
                inputs = self.tokenizer(
                    prompt_text, return_tensors="pt",
                    max_length=self.max_seq_length, truncation=True,
                ).to(self.device)
                prompt_token_len = inputs['input_ids'].shape[1]

                gen_config = GenerationConfig(
                    max_new_tokens=eval_max_new_tokens,
                    do_sample=False,  # greedy decoding for deterministic eval
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                with torch.no_grad():
                    output_ids = model.generate(**inputs, generation_config=gen_config)

                # Extract ONLY the generated tokens (after prompt) — token-based, not char-based
                generated_ids = output_ids[0, prompt_token_len:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                gold = sample['target'].lower().strip()
                pred = None

                if self.strategy == 'class_match':
                    # Compare first N words exactly (handles single-word labels
                    # like "negative" and multi-word labels like "very positive"
                    # without being fooled by repetition).
                    target_words = gold.split()
                    response_words = response.lower().split()
                    pred_words = response_words[:len(target_words)]
                    pred = " ".join(pred_words)
                elif self.strategy == 'regex_extract' and self.answer_regex:
                    pred_match = re.search(self.answer_regex, response)
                    gold_match = re.search(self.answer_regex, sample['target'])
                    if pred_match:
                        pred = pred_match.group(1).strip().lower()
                    if gold_match:
                        gold = gold_match.group(1).strip().lower()

                if pred is None:
                    pred = response.lower().strip()

                gold_labels.append(gold)
                pred_labels.append(pred)

                # Log each prediction only in verbose mode
                if self.verbose:
                    mark = "✓" if pred == gold else "✗"
                    _log(f"  [{mark}] sample {i}: expected='{sample['target']}' got='{response[:80]}'")

            # ── Derived metrics from the collected labels ──────────────────────
            total = len(gold_labels)
            correct = sum(g == p for g, p in zip(gold_labels, pred_labels))
            metrics['accuracy'] = correct / max(total, 1)
            _log(f"  Correct: {correct} / {total}")

            # Exact Match: full-string equality after stripping punctuation
            import string as _string
            _punct_table = str.maketrans('', '', _string.punctuation)
            def _normalize(s):
                return s.lower().translate(_punct_table).strip()
            em_correct = sum(
                _normalize(g) == _normalize(p)
                for g, p in zip(gold_labels, pred_labels)
            )
            metrics['exact_match'] = em_correct / max(total, 1)

            # Build confusion matrix over unique label classes
            classes = sorted(set(gold_labels))
            class_idx = {c: i for i, c in enumerate(classes)}
            n = len(classes)
            cm = [[0] * n for _ in range(n)]
            for g, p in zip(gold_labels, pred_labels):
                if g in class_idx and p in class_idx:
                    cm[class_idx[g]][class_idx[p]] += 1

            # Macro F1
            f1_scores = []
            for i in range(n):
                tp = cm[i][i]
                fp = sum(cm[r][i] for r in range(n)) - tp
                fn = sum(cm[i][c] for c in range(n)) - tp
                prec = tp / max(tp + fp, 1)
                rec  = tp / max(tp + fn, 1)
                f1_scores.append(2 * prec * rec / max(prec + rec, 1e-9))
            metrics['f1'] = sum(f1_scores) / max(len(f1_scores), 1)

            # Matthews Correlation Coefficient (multi-class generalisation)
            t = sum(cm[i][i] for i in range(n))
            s = total   # grand total
            p_k = [sum(cm[i][c] for c in range(n)) for i in range(n)]  # row sums (actual)
            t_k = [sum(cm[r][i] for r in range(n)) for i in range(n)]  # col sums (predicted)
            cov_yy = s * s - sum(pk * pk for pk in p_k)
            cov_xx = s * s - sum(tk * tk for tk in t_k)
            if cov_yy > 0 and cov_xx > 0:
                mcc_num = t * s - sum(p_k[i] * t_k[i] for i in range(n))
                metrics['mcc'] = mcc_num / math.sqrt(cov_yy * cov_xx)
            else:
                metrics['mcc'] = 0.0

            # Cohen's Kappa
            p_o = correct / max(total, 1)   # observed agreement
            p_e = sum(
                (sum(cm[i][c] for c in range(n)) / max(total, 1)) *
                (sum(cm[r][i] for r in range(n)) / max(total, 1))
                for i in range(n)
            )                               # chance agreement
            metrics['kappa'] = (p_o - p_e) / max(1.0 - p_e, 1e-9)

        model.train()

        for k, v in metrics.items():
            _log(f"  {k}: {v:.4f}")
        _log("--- End Eval ---")

        return metrics
