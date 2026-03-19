import time
import io
import re
import json
import sys
import argparse
import torch
import yaml
import os
import csv
from jinja2 import Template
from kafka import KafkaProducer, KafkaConsumer
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    GenerationConfig
)
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _log(msg):
    print(f"[{_ts()}][TRAINER] {msg}", flush=True)

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class MetricsLogger:
    """
    CSV logger for training/eval metrics + optional plotting.

    The CSV is opened, written, and closed on every call to log() so the
    file is never held locked (Windows blocks readers on open handles).
    Plotting reads from the CSV on disk, so plots can be generated even
    after a crash / Ctrl-C as long as the CSV was flushed.
    """

    COLUMNS = [
        "step", "loss", "eval_loss", "perplexity", "accuracy",
        "step_time_s", "records_used_total",
    ]

    def __init__(self, output_dir: str, run_name: str):
        ts = time.strftime("%Y%m%d-%H%M%S")
        safe_run = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in run_name)
        self.dir = os.path.join(output_dir, "logs", safe_run, ts)
        os.makedirs(self.dir, exist_ok=True)

        self.metrics_path = os.path.join(self.dir, "metrics.csv")
        self.params_path = os.path.join(self.dir, "run_params.json")
        self._header_written = False

    def write_params(self, params: dict):
        with open(self.params_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, sort_keys=True)

    def log(self, row: dict):
        mode = "a" if self._header_written else "w"
        with open(self.metrics_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS, extrasaction="ignore")
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            out = {k: row.get(k, "") for k in self.COLUMNS}
            writer.writerow(out)

    def _read_csv(self):
        rows = []
        if not os.path.exists(self.metrics_path):
            return rows
        with open(self.metrics_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        return rows

    def generate_plots(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            _log(f"Plotting skipped (matplotlib not available): {e}")
            return

        rows = self._read_csv()
        if not rows:
            _log("Plotting skipped (no metrics rows in CSV).")
            return

        def _series(key):
            xs, ys = [], []
            for r in rows:
                val = r.get(key, "")
                if val in ("", None):
                    continue
                try:
                    xs.append(int(r.get("step", 0)))
                    ys.append(float(val))
                except Exception:
                    continue
            return xs, ys

        plots = [
            ("train_loss", "Training Loss", "loss"),
            ("eval_loss", "Eval Loss", "eval_loss"),
            ("perplexity", "Perplexity", "perplexity"),
            ("accuracy", "Accuracy", "accuracy"),
        ]

        generated = 0
        for filename, title, key in plots:
            xs, ys = _series(key)
            if not xs:
                continue
            plt.figure()
            plt.plot(xs, ys, marker="o", markersize=3, linewidth=1.2)
            plt.title(title)
            plt.xlabel("Step")
            plt.ylabel(key)
            plt.grid(True, alpha=0.3)
            out_path = os.path.join(self.dir, f"{filename}.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            generated += 1

        _log(f"Generated {generated} plot(s) in: {self.dir}")

# --------------------------------------------------
# LoRAProducer: sends adapter weights via Kafka
# --------------------------------------------------
class LoRAProducer:
    def __init__(self, config):
        kafka_cfg = config['kafka']
        self.topic = kafka_cfg['lora_updates_topic']
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_cfg['bootstrap_servers'],
            value_serializer=lambda v: self.serialize_tensor(v),
            key_serializer=lambda k: k  # identity — key is pre-encoded to bytes in send_weights
        )

    def serialize_tensor(self, tensor):
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        return buffer.getvalue()
    
    def send_weights(self, adapter_weights):
        _log(f"Sending weights update... tensors={len(adapter_weights)} topic='{self.topic}'")
        for name, param in adapter_weights.items():
            self.producer.send(
                topic=self.topic,
                key=name.encode("utf-8") if isinstance(name, str) else name,
                value=param
            )
        self.producer.flush()

    def send_done_signal(self):
        """Send a sentinel message so the inference server knows training is over
        and can stop listening for LoRA updates."""
        _log(f"Sending training-done signal to inference on topic '{self.topic}'...")
        self.producer.send(
            topic=self.topic,
            key="__done__".encode("utf-8"),
            value=torch.tensor([0.0]),
        )
        self.producer.flush()

# --------------------------------------------------
# Tokenization helper: prompt/response aware
# --------------------------------------------------
def tokenize_with_label_masking(tokenizer, prompt_text, response_text, max_seq_length):
    """
    Tokenize prompt and response separately.  Truncate the PROMPT
    (not the response) so the label always fits within max_seq_length.
    Return input_ids, attention_mask, labels with -100 on prompt tokens.

    An EOS token is appended to the response so the model learns to stop
    generating after the answer.  If the response must be truncated, the
    last token is always replaced with EOS to preserve the stopping signal.
    """
    eos_id = tokenizer.eos_token_id

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)

    # Append EOS so the model learns to stop after the response
    if eos_id is not None:
        response_ids = response_ids + [eos_id]

    # Truncate prompt to leave room for the response (+ EOS already included)
    max_prompt_len = max_seq_length - len(response_ids)
    if max_prompt_len <= 0:
        # Edge case: response itself exceeds limit — truncate but keep EOS
        if eos_id is not None:
            response_ids = response_ids[:max_seq_length - 1] + [eos_id]
        else:
            response_ids = response_ids[:max_seq_length]
        prompt_ids = []
    else:
        prompt_ids = prompt_ids[:max_prompt_len]

    input_ids = prompt_ids + response_ids
    attention_mask = [1] * len(input_ids)
    # Labels: -100 for prompt tokens (ignored in loss), actual ids for response tokens
    labels = [-100] * len(prompt_ids) + response_ids

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'prompt_len': len(prompt_ids),
    }


def pad_batch(samples, pad_token_id, device):
    """Pad a list of tokenized dicts into a batch tensor dict."""
    max_len = max(len(s['input_ids']) for s in samples)
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    for s in samples:
        pad_len = max_len - len(s['input_ids'])
        padded_input_ids.append(s['input_ids'] + [pad_token_id] * pad_len)
        padded_attention_mask.append(s['attention_mask'] + [0] * pad_len)
        padded_labels.append(s['labels'] + [-100] * pad_len)

    return {
        'input_ids': torch.tensor(padded_input_ids, dtype=torch.long).to(device),
        'attention_mask': torch.tensor(padded_attention_mask, dtype=torch.long).to(device),
        'labels': torch.tensor(padded_labels, dtype=torch.long).to(device),
    }


# --------------------------------------------------
# Evaluator: Polymorphic evaluation strategies
# --------------------------------------------------
class Evaluator:
    """Evaluates model performance using configurable strategies.
    
    Strategies:
      - perplexity   : forward-pass loss → exp(loss)  [always computed]
      - class_match  : generate and check if prediction starts with target class
      - regex_extract: generate, extract answer via regex, compare to gold
    """

    def __init__(self, config, tokenizer, device):
        self.config = config
        self.eval_cfg = config.get('evaluation', {})
        self.enabled = self.eval_cfg.get('enabled', False)
        self.strategy = self.eval_cfg.get('strategy', 'perplexity')
        self.eval_interval = self.eval_cfg.get('eval_interval', 100)
        self.eval_samples = self.eval_cfg.get('eval_samples', 50)
        self.answer_regex = self.eval_cfg.get('answer_regex')
        self.tokenizer = tokenizer
        self.device = device

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
        """Load a small evaluation dataset from the configured eval_split."""
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
                if i >= self.eval_samples:
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
            _log(f"Loaded {len(self.eval_data)} evaluation samples (split='{eval_split}').")
        except Exception as e:
            _log(f"Warning: Could not load eval dataset: {e}")
            self.enabled = False

    def evaluate(self, model, step):
        """Run evaluation and return a metrics dict."""
        if not self.enabled or not self.eval_data:
            return None

        _log(f"--- Eval @ Step {step} ---")
        model.eval()
        metrics = {}

        # --- Perplexity (response-only loss, matching training) ---
        total_loss = 0.0
        count = 0

        for sample in self.eval_data:
            prompt_text = self.prompt_template.render(**sample)
            response_text = self.response_template.render(**sample)
            tok = tokenize_with_label_masking(
                self.tokenizer, prompt_text, response_text, self.max_seq_length
            )
            batch = pad_batch([tok], self.tokenizer.pad_token_id, self.device)
            with torch.no_grad():
                outputs = model(**batch)
                total_loss += outputs.loss.item()
                count += 1

        avg_loss = total_loss / max(count, 1)
        metrics['eval_loss'] = avg_loss
        metrics['perplexity'] = torch.exp(torch.tensor(avg_loss)).item()

        # --- Strategy-specific evaluation (generation-based) ---
        if self.strategy in ('class_match', 'regex_extract'):
            correct = 0
            total = 0
            cfg_max_new_tokens = self.config['inference'].get('max_new_tokens', 50)

            # For classification, the answer is a few words — cap generation to
            # avoid wasting compute and to surface repetition problems honestly.
            if self.strategy == 'class_match':
                eval_max_new_tokens = min(cfg_max_new_tokens, 10)
            else:
                eval_max_new_tokens = cfg_max_new_tokens

            # Only evaluate a small subset for generation (it's expensive)
            gen_samples = self.eval_data[:min(10, len(self.eval_data))]

            for i, sample in enumerate(gen_samples):
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

                is_correct = False
                if self.strategy == 'class_match':
                    # Compare first N words exactly (handles single-word labels
                    # like "negative" and multi-word labels like "very positive"
                    # without being fooled by repetition).
                    target_words = sample['target'].lower().split()
                    response_words = response.lower().split()
                    pred_words = response_words[:len(target_words)]
                    if pred_words == target_words:
                        is_correct = True
                elif self.strategy == 'regex_extract' and self.answer_regex:
                    pred_match = re.search(self.answer_regex, response)
                    gold_match = re.search(self.answer_regex, sample['target'])
                    if pred_match and gold_match and pred_match.group(1) == gold_match.group(1):
                        is_correct = True

                if is_correct:
                    correct += 1
                total += 1

                # Log each prediction so we can see what the model is doing
                mark = "✓" if is_correct else "✗"
                _log(f"  [{mark}] sample {i}: expected='{sample['target']}' got='{response[:80]}'")

            metrics['accuracy'] = correct / max(total, 1)

        model.train()

        for k, v in metrics.items():
            _log(f"  {k}: {v:.4f}")
        _log("--- End Eval ---")

        return metrics

# --------------------------------------------------
# Manual training loop function
# --------------------------------------------------
def train_model(config):
    _log("Starting preparation for training")

    model_cfg = config['model']
    lora_cfg = config['lora']
    training_cfg = config['training']
    kafka_cfg = config['kafka']
    preproc_cfg = config['preprocessing']

    test_mode = training_cfg.get('test_mode', False)
    if test_mode:
        _log("*** TEST MODE enabled: will train on entire dataset, stop all services on EOF. ***")

    _log(f"Config summary: model='{model_cfg.get('name')}', task_type='{model_cfg.get('task_type', 'CAUSAL_LM')}', precision='{model_cfg.get('precision', 'fp16')}'")
    _log(f"Kafka: bootstrap_servers={kafka_cfg.get('bootstrap_servers')}, training_topic='{kafka_cfg.get('training_topic')}', lora_updates_topic='{kafka_cfg.get('lora_updates_topic')}'")
    _log(f"Training: batch_size={training_cfg.get('batch_size', 4)}, grad_accum={training_cfg.get('gradient_accumulation_steps', 2)}, lr={training_cfg.get('learning_rate', 2e-4)}, max_steps={training_cfg.get('max_steps', 1000)}")

    # Metrics logging (CSV + params JSON + optional plots at end)
    run_name = config.get("project", {}).get("name", "run")
    output_dir = config.get("project", {}).get("output_dir", "./output")
    metrics_logger = MetricsLogger(output_dir=output_dir, run_name=run_name)
    metrics_logger.write_params(
        {
            "project": config.get("project", {}),
            "model": config.get("model", {}),
            "lora": config.get("lora", {}),
            "dataset": config.get("dataset", {}),
            "kafka": config.get("kafka", {}),
            "training": config.get("training", {}),
            "inference": config.get("inference", {}),
            "evaluation": config.get("evaluation", {}),
        }
    )

    # Build LoRA config from YAML
    lora_config = LoraConfig(
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['alpha'],
        target_modules=lora_cfg['target_modules'],
        lora_dropout=lora_cfg.get('dropout', 0.05),
        bias=lora_cfg.get('bias', 'none'),
        task_type=model_cfg.get('task_type', 'CAUSAL_LM'),
    )

    # Choose device. (On macs using MPS or defaulting to CPU/GPU.)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    _log(f"Selected device: {device}")
    
    # Load the model and tokenizer
    model_name = model_cfg['name']
    _log(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure there is a pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    _log("Loaded model; applying PEFT and LoRA adapter configuration...")

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Set the model to training mode.
    model.train()
    
    # Set up Kafka consumer to stream training data.
    poll_timeout_ms = int(kafka_cfg.get('poll_timeout_ms', 1000))
    consumer_group = kafka_cfg.get('consumer_group_trainer', 'trainer-group')
    _log(f"Connecting Kafka consumer: topic='{kafka_cfg['training_topic']}', consumer_group='{consumer_group}', poll_timeout_ms={poll_timeout_ms}")
    consumer = KafkaConsumer(
        kafka_cfg['training_topic'],
        bootstrap_servers=kafka_cfg['bootstrap_servers'],
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        group_id=consumer_group,
        auto_offset_reset="earliest",
    )

    # Force partition assignment and log it (otherwise first poll() does it silently)
    _log("Waiting for Kafka partition assignment...")
    consumer.poll(timeout_ms=0)  # triggers group join / rebalance
    assigned = consumer.assignment()
    _log(f"Partition assignment: {assigned if assigned else '(none yet — will assign on next poll)'}")

    if test_mode and assigned:
        # In test_mode, skip ALL stale data from previous producer runs.
        # Seek to the very end of the topic so the trainer only consumes
        # records sent AFTER this point (i.e. the current producer run).
        consumer.seek_to_end()
        for tp in assigned:
            pos = consumer.position(tp)
            _log(f"  {tp}: seeked to end, position={pos}")
        _log("Test mode: Positioned at end of topic — only NEW records from this producer run will be consumed.")
        _log(">>> Start the producer now (if not already running). <<<")
    else:
        for tp in assigned:
            pos = consumer.position(tp)
            _log(f"  {tp}: current_position={pos}")
    
    # Training hyperparameters
    training_args = TrainingArguments(
        output_dir=config['project'].get('output_dir', './output'),
        per_device_train_batch_size=training_cfg.get('batch_size', 4),
        gradient_accumulation_steps=training_cfg.get('gradient_accumulation_steps', 2),
        learning_rate=float(training_cfg.get('learning_rate', 2e-4)),
        logging_steps=training_cfg.get('logging_steps', 1),
        max_steps=training_cfg.get('max_steps', 1000),
        fp16=(model_cfg.get('precision', 'fp16') == 'fp16' and torch.cuda.is_available())
    )
    _log(f"TrainingArguments: per_device_train_batch_size={training_args.per_device_train_batch_size}, grad_accum={training_args.gradient_accumulation_steps}, fp16={training_args.fp16}")
    
    # Create an optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    
    # Instantiate our Kafka producer for LoRA updates.
    lora_producer = LoRAProducer(config)

    # Compile Jinja2 templates for text formatting
    prompt_template = Template(preproc_cfg['prompt_template'])
    response_template = Template(preproc_cfg.get('response_template', ' {{ target }}'))
    _log("Jinja2 templates compiled (prompt_template + response_template).")

    # Initialize evaluator
    evaluator = Evaluator(config, tokenizer, device)
    
    # Set up timing for sending weights
    last_send_time = time.time()
    weight_push_interval = training_cfg.get('weight_push_interval', 60)
    
    # Counters for gradient accumulation and optimization steps
    optimization_step = 0
    grad_accum_counter = 0
    accumulated_loss = 0.0
    step_start = time.time()

    max_seq_length = model_cfg.get('max_seq_length', 512)
    
    total_messages_seen = 0
    last_data_time = time.time()
    last_heartbeat_time = time.time()
    heartbeat_every_s = 5.0
    _log("Starting manual training loop...")
    _log("Label masking enabled: loss computed on response tokens only (prompt tokens masked with -100).")

    eof_received = False
    stop_requested = False

    # In test_mode we train on the ENTIRE dataset (stop on EOF, not max_steps).
    # max_steps still serves as a safety cap in normal mode.
    effective_max_steps = sys.maxsize if test_mode else training_args.max_steps
    if test_mode:
        _log(f"Test mode: will train until all data is consumed (max_steps ignored, EOF is the stop signal).")
    else:
        _log(f"Normal mode: will train up to {training_args.max_steps} steps.")

    def _log_eval(step):
        eval_metrics = evaluator.evaluate(model, step) or {}
        if eval_metrics:
            metrics_logger.log(
                {
                    "step": step,
                    "eval_loss": eval_metrics.get("eval_loss"),
                    "perplexity": eval_metrics.get("perplexity"),
                    "accuracy": eval_metrics.get("accuracy"),
                    "records_used_total": total_messages_seen,
                }
            )
        return eval_metrics

    _log(f"Metrics will be saved to: {metrics_logger.metrics_path}")
    _log(f"  (To regenerate plots later: python utils/plot_metrics.py \"{metrics_logger.metrics_path}\")")

    interrupted = False

    try:
        while optimization_step < effective_max_steps and not stop_requested:
            batch_samples = []
            # Assemble a mini-batch of raw examples.
            while len(batch_samples) < training_args.per_device_train_batch_size:
                messages = consumer.poll(timeout_ms=poll_timeout_ms)
                if not messages:
                    # Heartbeat so it never looks "stuck" while waiting for producer data.
                    now = time.time()
                    if eof_received and (now - last_data_time) >= 2.0:
                        _log("End-of-stream: no more data from producer. Exiting training loop.")
                        stop_requested = True
                        break
                    if now - last_heartbeat_time >= heartbeat_every_s:
                        _log(f"Waiting for Kafka data... batch_progress={len(batch_samples)}/{training_args.per_device_train_batch_size}, total_messages_seen={total_messages_seen}, idle_for={now - last_data_time:.1f}s")
                        last_heartbeat_time = now
                    time.sleep(0.1)
                    continue

                for tp, records in messages.items():
                    for message in records:
                        if message is None or message.value is None:
                            continue
                        sample_data = message.value

                        # Skip internal control messages from the producer
                        if isinstance(sample_data, dict) and ("_eof" in sample_data or "_verify" in sample_data):
                            if sample_data.get("_eof"):
                                eof_received = True
                                _log("Received end-of-stream marker from producer.")
                            continue

                        prompt_text = prompt_template.render(**sample_data)
                        response_text = response_template.render(**sample_data)

                        # Tokenize with prompt/response separation, truncate prompt (not response),
                        # and mask prompt tokens in labels.
                        tok = tokenize_with_label_masking(
                            tokenizer, prompt_text, response_text, max_seq_length
                        )
                        batch_samples.append(tok)
                        total_messages_seen += 1
                        last_data_time = time.time()

                        if len(batch_samples) >= training_args.per_device_train_batch_size:
                            break
                    if len(batch_samples) >= training_args.per_device_train_batch_size:
                        break

            # If we broke out with no samples at all, nothing left to train on.
            if not batch_samples:
                break

            # Pad batch and create tensors
            batch = pad_batch(batch_samples, tokenizer.pad_token_id, device)

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            accumulated_loss += loss.item()
            # Scale loss for gradient accumulation before backward
            scaled_loss = loss / training_args.gradient_accumulation_steps
            scaled_loss.backward()
            grad_accum_counter += 1

            # Once enough mini-batches are accumulated, update the optimizer.
            if grad_accum_counter % training_args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                optimization_step += 1

                avg_step_loss = accumulated_loss / training_args.gradient_accumulation_steps

                if optimization_step % training_args.logging_steps == 0:
                    elapsed = time.time() - step_start
                    _log(
                        f"Step {optimization_step}: loss = {avg_step_loss:.4f} "
                        f"(step_time={elapsed:.2f}s, records_used_total={total_messages_seen})"
                    )
                    metrics_logger.log(
                        {
                            "step": optimization_step,
                            "loss": avg_step_loss,
                            "step_time_s": elapsed,
                            "records_used_total": total_messages_seen,
                        }
                    )

                # Reset accumulators for the next optimization step
                accumulated_loss = 0.0
                step_start = time.time()

                # Periodic evaluation
                if evaluator.enabled and optimization_step % evaluator.eval_interval == 0:
                    _log_eval(optimization_step)

            # Every weight_push_interval seconds, send the current LoRA adapter weights.
            if time.time() - last_send_time >= weight_push_interval:
                _log(f"{weight_push_interval}s elapsed at optimization step {optimization_step}. Sending adapter weights...")
                lora_producer.send_weights(get_peft_model_state_dict(model))
                last_send_time = time.time()

    except KeyboardInterrupt:
        interrupted = True
        _log("KeyboardInterrupt received — skipping final eval, proceeding to save metrics and plots.")
    except Exception as e:
        _log(f"Training loop error: {e}")

    # =====================================================================
    #  CLEANUP — guaranteed to run via finally-like structure.
    #  Everything here is wrapped in its own try/except so one failure
    #  (e.g. a second Ctrl-C during eval) can never prevent plots from
    #  being generated.
    # =====================================================================

    # 1. Send final LoRA weights + done signal to inference.
    try:
        _log("Training complete. Sending final adapter weights.")
        lora_producer.send_weights(get_peft_model_state_dict(model))
        lora_producer.send_done_signal()
    except Exception as e:
        _log(f"Warning: Failed to send final weights / done signal: {e}")

    # 2. Final evaluation — skip if the user already pressed Ctrl-C (they
    #    want to exit, not wait for a slow model.generate() pass).
    if evaluator.enabled and not interrupted:
        try:
            _log_eval(optimization_step)
        except KeyboardInterrupt:
            _log("Final evaluation interrupted — skipping. Metrics already saved.")
        except Exception as e:
            _log(f"Warning: Final evaluation failed: {e}")

    # 3. Always print the CSV location and generate plots.
    _log(f"Metrics CSV saved to: {metrics_logger.metrics_path}")
    try:
        metrics_logger.generate_plots()
    except Exception as e:
        _log(f"Warning: Plot generation failed: {e}")
    _log(f"To regenerate plots later: python utils/plot_metrics.py \"{metrics_logger.metrics_path}\"")

# --------------------------------------------------
# Main entry point
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InfiniTune Trainer")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration YAML file")
    args = parser.parse_args()

    config = load_config(args.config)
    _log(f"Loaded config: {args.config}")
    train_model(config)