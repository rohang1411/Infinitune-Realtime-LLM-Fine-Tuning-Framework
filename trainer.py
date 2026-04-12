import time
import io
import re
import json
import sys
import argparse
import torch
import math
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
from utils.eval_metrics_train import Evaluator

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
        "step",
        "loss",
        "lr",
        "eval_loss",
        "perplexity",
        "accuracy",
        "aauc",
        "f1",
        "mcc",
        "kappa",
        "exact_match",
        "qafacteval",
        "forgetting_max",
        "update_latency_s",
        "backward_transfer",
        "grad_norm",
        "tokens_per_sec",
        "step_time_s",
        "records_used_total"
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
            ("aauc", "AAUC (normalized)", "aauc"),
            ("backward_transfer", "Backward Transfer", "backward_transfer"),
            ("f1", "Macro F1 Score", "f1"),
            ("mcc", "Matthews Correlation Coefficient", "mcc"),
            ("kappa", "Cohen's Kappa", "kappa"),
            ("exact_match", "Exact Match Rate", "exact_match"),
            ("forgetting_max", "Max Forgetting (tracked metrics)", "forgetting_max"),
            ("update_latency_s", "Update Latency (s since last eval)", "update_latency_s"),
            ("grad_norm", "Gradient Norm", "grad_norm"),
            ("tokens_per_sec", "Token Throughput (tok/s)", "tokens_per_sec"),
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
# Adaptive LR Scheduler
# --------------------------------------------------
def build_lr_scheduler(optimizer, config):
    """Build a LambdaLR scheduler from config.

    Supported types:
      - cosine_with_warmup : linear warmup then cosine decay to min_lr_ratio
      - linear             : linear warmup then linear decay to min_lr_ratio
      - constant           : no scheduling (lambda = 1)
    """
    sched_cfg = config.get('training', {}).get('lr_scheduler', {})
    sched_type = sched_cfg.get('type', 'constant')

    if sched_type == 'constant' or not sched_cfg:
        _log("LR scheduler: constant (no scheduling).")
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    warmup_steps = sched_cfg.get('warmup_steps', 0)
    min_lr_ratio = sched_cfg.get('min_lr_ratio', 0.0)
    T_max = sched_cfg.get('T_max', 1000)

    if sched_type == 'cosine_with_warmup':
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup: 0 → 1
                return max(current_step / max(warmup_steps, 1), 0.0)
            # Cosine decay: 1 → min_lr_ratio
            progress = (current_step - warmup_steps) / max(T_max - warmup_steps, 1)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        _log(f"LR scheduler: cosine_with_warmup (warmup={warmup_steps}, T_max={T_max}, min_lr_ratio={min_lr_ratio})")
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    elif sched_type == 'linear':
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return max(current_step / max(warmup_steps, 1), 0.0)
            progress = (current_step - warmup_steps) / max(T_max - warmup_steps, 1)
            progress = min(progress, 1.0)
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)

        _log(f"LR scheduler: linear (warmup={warmup_steps}, T_max={T_max}, min_lr_ratio={min_lr_ratio})")
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    else:
        _log(f"LR scheduler: unknown type '{sched_type}', falling back to constant.")
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)


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

    # Create LR scheduler
    scheduler = build_lr_scheduler(optimizer, config)
    
    # Instantiate our Kafka producer for LoRA updates.
    lora_producer = LoRAProducer(config)

    # Compile Jinja2 templates for text formatting
    prompt_template = Template(preproc_cfg['prompt_template'])
    response_template = Template(preproc_cfg.get('response_template', ' {{ target }}'))
    _log("Jinja2 templates compiled (prompt_template + response_template).")

    # Initialize evaluator
    evaluator = Evaluator(config, tokenizer, device, tokenize_with_label_masking, pad_batch)
    
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
                    "aauc": eval_metrics.get("aauc"),
                    "backward_transfer": eval_metrics.get("backward_transfer"),
                    "f1": eval_metrics.get("f1"),
                    "mcc": eval_metrics.get("mcc"),
                    "kappa": eval_metrics.get("kappa"),
                    "exact_match": eval_metrics.get("exact_match"),
                    "forgetting_max": eval_metrics.get("forgetting_max"),
                    "update_latency_s": eval_metrics.get("update_latency_s"),
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

            # Count response tokens in the batch for throughput calculation
            batch_response_tokens = sum(
                sum(1 for lbl in s['labels'] if lbl != -100)
                for s in batch_samples
            )

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            accumulated_loss += loss.item()
            # Scale loss for gradient accumulation before backward
            scaled_loss = loss / training_args.gradient_accumulation_steps
            scaled_loss.backward()
            grad_accum_counter += 1

            # Compute gradient norm BEFORE the optimizer clips / steps so we
            # capture the raw pre-clip magnitude for diagnostic purposes.
            step_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    step_grad_norm += p.grad.detach().norm(2).item() ** 2
            step_grad_norm = step_grad_norm ** 0.5
            accumulated_grad_norm = getattr(train_model, '_acc_grad_norm', 0.0) + step_grad_norm
            train_model._acc_grad_norm = accumulated_grad_norm

            # Accumulate response‑token count across micro-batches
            accumulated_response_tokens = getattr(train_model, '_acc_tokens', 0) + batch_response_tokens
            train_model._acc_tokens = accumulated_response_tokens

            # Once enough mini-batches are accumulated, update the optimizer.
            if grad_accum_counter % training_args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimization_step += 1

                avg_step_loss = accumulated_loss / training_args.gradient_accumulation_steps
                avg_grad_norm = train_model._acc_grad_norm / training_args.gradient_accumulation_steps
                current_lr = scheduler.get_last_lr()[0]

                if optimization_step % training_args.logging_steps == 0:
                    elapsed = time.time() - step_start
                    tokens_per_sec = train_model._acc_tokens / max(elapsed, 1e-6)
                    _log(
                        f"Step {optimization_step}: loss = {avg_step_loss:.4f}, lr = {current_lr:.2e}, "
                        f"grad_norm = {avg_grad_norm:.4f}, tok/s = {tokens_per_sec:.1f} "
                        f"(step_time={elapsed:.2f}s, records_used_total={total_messages_seen})"
                    )
                    metrics_logger.log(
                        {
                            "step": optimization_step,
                            "loss": avg_step_loss,
                            "lr": current_lr,
                            "grad_norm": avg_grad_norm,
                            "tokens_per_sec": tokens_per_sec,
                            "step_time_s": elapsed,
                            "records_used_total": total_messages_seen,
                        }
                    )

                # Reset per-step accumulators
                train_model._acc_grad_norm = 0.0
                train_model._acc_tokens = 0

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