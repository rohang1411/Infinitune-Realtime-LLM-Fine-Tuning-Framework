import time
import io
import re
import json
import sys
import uuid
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
from utils.eval_qualitative import QualitativeEvaluator
from utils.checkpoint_manager import CheckpointManager

def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _log(msg):
    print(f"[{_ts()}][TRAINER] {msg}", flush=True)

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _apply_label_map_if_configured(target_value, label_map):
    """Normalize raw dataset labels to configured textual labels when possible."""
    if label_map is None:
        return target_value
    if target_value in label_map:
        return label_map[target_value]
    target_str = str(target_value)
    if target_str in label_map:
        return label_map[target_str]
    return target_value


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
        # Backward/forward-compatible aliases:
        # - utils/eval_metrics_train.py returns `average_accuracy` (normalized AUC)
        # - utils/plot_metrics.py expects `average_accuracy` for the chart
        # - older code used `aauc`
        "average_accuracy",
        "f1",
        "mcc",
        "kappa",
        "exact_match",
        "qafacteval",
        "answer_overlap_f1",
        "forgetting_max",
        "update_latency_s",
        # utils/eval_metrics_train.py returns `eval_cycle_time_s`
        # utils/plot_metrics.py expects `eval_cycle_time_s`
        "eval_cycle_time_s",
        "backward_transfer",
        "grad_norm",
        "tokens_per_sec",
        "step_time_s",
        "records_used_total",
        # ── Qualitative metrics — universal ──────────────────────────────────
        "qual_semantic_similarity",
        "qual_keyword_density",
        "qual_type_token_ratio",
        "qual_hapax_ratio",
        "qual_cot_anchor_count_mean",
        "qual_cot_step_length_mean",
        "qual_cot_coverage_rate",
        "qual_mean_response_length",
        "qual_repetition_rate",
        "qual_non_empty_rate",
        # ── E2E NLG — slot coverage summary ──────────────────────────────────
        "qual_slot_coverage_mean",
        "qual_consistency_score_mean",
        "qual_perfect_coverage_rate",
        "qual_slot_familyFriendly_inversion_rate",
        # ── E2E NLG — pinned anchor metrics ──────────────────────────────────
        "qual_pinned_slot_coverage_mean",
        "qual_pinned_perfect_coverage_rate",
        "qual_pinned_consistency_score",
    ]

    def __init__(self, output_dir: str, run_name: str):
        ts = time.strftime("%Y%m%d-%H%M%S")
        # Append a short random suffix so rapid restarts never collide on the
        # same second, which would cause the new run to silently append to the
        # old metrics.csv instead of starting a fresh file.
        suffix = uuid.uuid4().hex[:4]
        safe_run = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in run_name)
        run_dir_name = f"{ts}_{suffix}"
        self.dir = os.path.join(output_dir, "logs", safe_run, run_dir_name)
        os.makedirs(self.dir, exist_ok=True)

        self.metrics_path = os.path.join(self.dir, "metrics.csv")
        self.params_path = os.path.join(self.dir, "run_params.json")
        self._header_written = False

    def write_params(self, params: dict):
        with open(self.params_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, sort_keys=True)

    def log(self, row: dict):
        # Dynamic columns: anything in row not in COLUMNS is appended.
        # This handles per-slot columns like qual_slot_food_coverage which
        # are dataset-specific and not known at class definition time.
        extra_keys = [k for k in row if k not in self.COLUMNS]
        fieldnames = self.COLUMNS + extra_keys

        mode = "a" if self._header_written else "w"
        with open(self.metrics_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
                self._fieldnames = fieldnames  # cache for later reads
            out = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(out)

        # If new extra keys arrived on a subsequent log() call, we need to
        # re-write the header. Instead of rewriting the whole file (expensive),
        # we mark the flag so the DictWriter picks up the expanded set next time.
        if extra_keys and not hasattr(self, '_fieldnames'):
            self._fieldnames = fieldnames

    def _read_csv(self):
        rows = []
        if not os.path.exists(self.metrics_path):
            return rows
        with open(self.metrics_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        return rows

    def finalize_csv(self):
        """Write metrics_clean.csv with only columns that have at least one
        non-empty value. The original metrics.csv is preserved unchanged."""
        rows = self._read_csv()
        if not rows:
            return
        all_keys = list(rows[0].keys())
        populated = [
            k for k in all_keys
            if any(r.get(k, "") not in ("", None) for r in rows)
        ]
        clean_path = os.path.join(self.dir, "metrics_clean.csv")
        with open(clean_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=populated, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in populated})
        _log(f"Clean metrics CSV (populated columns only) saved to: {clean_path}")

    def save_verbose_samples(self, step: int, samples: list, source: str = "quantitative"):
        """Append verbose evaluation samples as a Markdown table to verbose_samples.md.

        Parameters
        ----------
        step    : Optimization step at which eval ran.
        samples : List of dicts with keys: sample_idx, input, target,
                  prediction, correct (bool or None).
        source  : 'quantitative' or 'qualitative' — used as the section heading.
        """
        if not samples:
            return
        path = os.path.join(self.dir, "verbose_samples.md")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n## {source.title()} Eval @ Step {step}\n\n")
            f.write("| # | Input (truncated) | Target | Prediction | Match |\n")
            f.write("|---|---|---|---|---|\n")
            for s in samples:
                correct = s.get("correct")
                if correct is True:
                    mark = "\u2713"
                elif correct is False:
                    mark = "\u2717"
                else:
                    mark = "—"  # no comparison available (perplexity path)
                inp  = str(s.get("input",      "")).replace("|", "\u00a6")
                tgt  = str(s.get("target",     "")).replace("|", "\u00a6")
                pred = str(s.get("prediction", "")).replace("|", "\u00a6")
                f.write(f"| {s.get('sample_idx', '')} | {inp} | {tgt} | {pred} | {mark} |\n")
            f.write("\n---\n")

    def generate_plots(self, config: dict = None):
        """Generate the evaluation artifact bundle for this training run.

        Delegates to `utils.evaluation_artifacts.generate_evaluation_artifacts()`
        so plots, dashboards, insights, and `report.html` are produced through
        the shared non-overwriting bundle pipeline.

        Parameters
        ----------
        config : Full YAML config dict used to drive usecase-aware KPIs,
                 insights, and dashboard/report metadata.
        """
        if not os.path.exists(self.metrics_path):
            _log("Plotting skipped (no metrics CSV found).")
            return

        # Use metrics_clean.csv if present (fewer empty columns = cleaner plots)
        clean_path = os.path.join(self.dir, "metrics_clean.csv")
        csv_to_plot = clean_path if os.path.exists(clean_path) else self.metrics_path

        try:
            from utils.evaluation_artifacts import generate_evaluation_artifacts

            manifest = generate_evaluation_artifacts(
                metrics_csv_path=csv_to_plot,
                run_root=self.dir,
                config=config,
                context="inline_training",
            )
            bundle = manifest.get("artifact_bundle")
            if bundle:
                _log(f"Evaluation artifacts generated in: {bundle}")
        except Exception as exc:
            _log(f"Plotting failed: {exc}")
            import traceback; traceback.print_exc()


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

    bos_id = tokenizer.bos_token_id

    # Compute effective target length cleanly guaranteeing targets get full priority bounds mapping
    effective_response_len = len(response_ids)
    if eos_id is not None:
        effective_response_len += 1

    # Overly massive responses (rare corner case mapping limit bounds)
    if effective_response_len > max_seq_length:
        prompt_ids = []
        if eos_id is not None:
            response_ids = response_ids[:max_seq_length - 1] + [eos_id]
        else:
            response_ids = response_ids[:max_seq_length]
    else:
        # Prompt must be cleanly truncated from the LEFT to make room for targets safely natively
        max_prompt_len = max_seq_length - effective_response_len
        if len(prompt_ids) > max_prompt_len:
            if bos_id is not None and prompt_ids[0] == bos_id and max_prompt_len > 0:
                prompt_ids = [bos_id] + prompt_ids[-(max_prompt_len - 1):]
            else:
                prompt_ids = prompt_ids[-max_prompt_len:]
                
        if eos_id is not None:
            response_ids = response_ids + [eos_id]

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
def train_model(config, config_path: str = "(unknown)"):
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
    
    model_name = model_cfg['name']
    prec = model_cfg.get('precision', 'fp32')
    dtype = torch.float16 if prec == 'fp16' else (torch.bfloat16 if prec == 'bf16' else torch.float32)
    
    _log(f"Loading base model: {model_name} with dtype {dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure there is a pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    _log("Loaded model; applying PEFT and LoRA adapter configuration...")

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    
    # Enable gradient checkpointing — trades ~30% extra compute for ~60% less
    # activation memory.  Critical on memory-constrained GPUs (Colab T4/A10G).
    if training_cfg.get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        _log("Gradient checkpointing enabled (use_reentrant=False).")
    else:
        _log("Gradient checkpointing disabled.")

    # Set the model to training mode.
    model.train()
    
    # Set up Kafka consumer to stream training data.
    poll_timeout_ms = int(kafka_cfg.get('poll_timeout_ms', 1000))
    consumer_group = kafka_cfg.get('consumer_group_trainer', 'trainer-group')
    # In test_mode, use a unique consumer group per run so there are no stale
    # committed offsets from previous runs. Combined with auto_offset_reset='earliest'
    # this ensures ALL messages in the topic are consumed regardless of producer
    # start order (fixes the seek_to_end() data-loss bug).
    if test_mode:
        unique_suffix = uuid.uuid4().hex[:8]
        consumer_group = f"{consumer_group}-{unique_suffix}"
        _log(f"Test mode: unique consumer group '{consumer_group}' (no stale offset risk)")
    _log(f"Connecting Kafka consumer: topic='{kafka_cfg['training_topic']}', consumer_group='{consumer_group}', poll_timeout_ms={poll_timeout_ms}")
    consumer = KafkaConsumer(
        kafka_cfg['training_topic'],
        bootstrap_servers=kafka_cfg['bootstrap_servers'],
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        group_id=consumer_group,
        auto_offset_reset="earliest",
        # max_poll_interval_ms: how long poll() can be absent before the broker
        # evicts this consumer from the group. Training pauses during eval
        # (especially qualitative eval with consistency_runs), so we set this
        # generously. Default is 300,000 ms (5 min) which is too short.
        max_poll_interval_ms=int(kafka_cfg.get('max_poll_interval_ms', 1800000)),   # 30 min
        session_timeout_ms=int(kafka_cfg.get('session_timeout_ms', 30000)),          # 30 sec
        heartbeat_interval_ms=int(kafka_cfg.get('heartbeat_interval_ms', 10000)),    # 10 sec
    )

    # Force partition assignment and log it (otherwise first poll() does it silently)
    _log("Waiting for Kafka partition assignment...")
    consumer.poll(timeout_ms=0)  # triggers group join / rebalance
    assigned = consumer.assignment()
    _log(f"Partition assignment: {assigned if assigned else '(none yet — will assign on next poll)'}")

    if test_mode:
        # Consuming earliest offsets with a unique group — no seek needed.
        # The producer can start before or after the trainer; all data is consumed.
        for tp in assigned:
            pos = consumer.position(tp) if assigned else 0
            _log(f"  {tp}: starting at position={pos}")
        _log("Test mode: consuming from earliest offset. Producer can start in any order.")
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
    
    # Create an optimizer — only pass trainable (LoRA) parameters.
    # Passing all params works but wastes time iterating 1.5B frozen params
    # during every step/zero_grad call.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    _log(f"Optimizer will track {len(trainable_params)} trainable parameter tensors.")
    optimizer = torch.optim.AdamW(trainable_params, lr=training_args.learning_rate)

    # Create LR scheduler
    scheduler = build_lr_scheduler(optimizer, config)
    
    # Instantiate our Kafka producer for LoRA updates.
    lora_producer = LoRAProducer(config)

    # Compile Jinja2 templates for text formatting
    prompt_template = Template(preproc_cfg['prompt_template'])
    response_template = Template(preproc_cfg.get('response_template', ' {{ target }}'))
    _log("Jinja2 templates compiled (prompt_template + response_template).")

    is_decoupled = config.get("evaluation", {}).get("decoupled", False)
    if is_decoupled:
        _log("Decoupled evaluation flag is TRUE. All inline evaluation during training will be skipped.")

    # Initialize quantitative evaluator
    evaluator = Evaluator(config, tokenizer, device, tokenize_with_label_masking, pad_batch)
    if is_decoupled:
        evaluator.enabled = False

    # Initialize qualitative evaluator (no-op if testing_strategy block absent or disabled)
    qual_evaluator = QualitativeEvaluator(config, tokenizer, device)
    if is_decoupled:
        qual_evaluator.enabled = False

    # Initialize CheckpointManager — saves LoRA adapters every N steps.
    # Enabled by default (save_checkpoints.enabled: true); disable explicitly
    # in config if not needed.  Checkpoints are always useful for decoupled
    # evaluation via evaluate.py, regardless of inline eval settings.
    ckpt_cfg = training_cfg.get("save_checkpoints", {})
    ckpt_manager = None
    save_every_steps = 100
    save_final_ckpt  = True
    if ckpt_cfg.get("enabled", True):
        ckpt_manager      = CheckpointManager(config, config_path=config_path)
        save_every_steps  = int(ckpt_cfg.get("save_every_steps", 100))
        save_final_ckpt   = bool(ckpt_cfg.get("save_final", True))
        _log(
            f"Checkpoint saving enabled: every {save_every_steps} steps, "
            f"save_final={save_final_ckpt}, root={ckpt_manager.checkpoint_root}"
        )
    else:
        _log("Checkpoint saving disabled (save_checkpoints.enabled: false).")
    
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
    consecutive_empty_polls_after_eof = 0
    max_consecutive_empty_polls = 100  # 100 x 1s = ~100s of true silence after offsets caught up

    # In test_mode we train on the ENTIRE dataset (stop on EOF, not max_steps).
    # max_steps still serves as a safety cap in normal mode.
    effective_max_steps = sys.maxsize if test_mode else training_args.max_steps
    if test_mode:
        _log(f"Test mode: will train until all data is consumed (max_steps ignored, EOF is the stop signal).")
    else:
        _log(f"Normal mode: will train up to {training_args.max_steps} steps.")

    def _log_eval(step):
        """Run quantitative evaluation and write its metrics to the CSV.
        Returns the quantitative metrics dict (may be empty)."""
        eval_metrics = evaluator.evaluate(model, step) or {}
        if eval_metrics:
            metrics_logger.log(
                {
                    "step": step,
                    "eval_loss": eval_metrics.get("eval_loss"),
                    "perplexity": eval_metrics.get("perplexity"),
                    "accuracy": eval_metrics.get("accuracy"),
                    # Prefer the evaluator's native names, but fill aliases
                    # so plotting stays consistent across all call sites.
                    "aauc": eval_metrics.get("aauc", None)
                    or eval_metrics.get("average_accuracy"),
                    "average_accuracy": eval_metrics.get("average_accuracy", None)
                    or eval_metrics.get("aauc"),
                    "backward_transfer": eval_metrics.get("backward_transfer"),
                    "f1": eval_metrics.get("f1"),
                    "mcc": eval_metrics.get("mcc"),
                    "kappa": eval_metrics.get("kappa"),
                    "exact_match": eval_metrics.get("exact_match"),
                    "answer_overlap_f1": eval_metrics.get("answer_overlap_f1"),
                    "forgetting_max": eval_metrics.get("forgetting_max"),
                    "update_latency_s": eval_metrics.get("update_latency_s", None)
                    or eval_metrics.get("eval_cycle_time_s"),
                    "eval_cycle_time_s": eval_metrics.get("eval_cycle_time_s", None)
                    or eval_metrics.get("update_latency_s"),
                    "records_used_total": total_messages_seen,
                }
            )
            # Persist verbose samples to Markdown if present
            verbose = eval_metrics.get("_verbose_samples")
            if verbose:
                metrics_logger.save_verbose_samples(step, verbose, source="quantitative")
        return eval_metrics

    def _log_qual_eval(step):
        """Run qualitative evaluation and write its metrics to the CSV.
        Independent from _log_eval — qualitative eval has its own interval
        and runs even when quantitative eval is disabled.
        Returns the qualitative metrics dict (may be empty)."""
        qual_metrics = qual_evaluator.run(model, step) or {}
        if qual_metrics:
            metrics_logger.log(
                {
                    "step": step,
                    "records_used_total": total_messages_seen,
                    **qual_metrics,
                }
            )
        return qual_metrics

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
                    # After EOF is received, use Kafka offset metadata to verify
                    # the topic is truly drained before starting the countdown.
                    # This prevents false-positive termination from transient empty
                    # polls caused by consumer group rebalancing or fetch timing.
                    if eof_received:
                        all_caught_up = False
                        try:
                            end_offsets = consumer.end_offsets(list(consumer.assignment()))
                            all_caught_up = all(
                                consumer.position(tp) >= end_offsets.get(tp, 0)
                                for tp in consumer.assignment()
                            )
                        except Exception:
                            pass  # be conservative: don't count if check fails

                        if all_caught_up:
                            consecutive_empty_polls_after_eof += 1
                            if consecutive_empty_polls_after_eof >= max_consecutive_empty_polls:
                                _log(
                                    f"End-of-stream reached: all offsets consumed, "
                                    f"{consecutive_empty_polls_after_eof} consecutive empty polls. "
                                    f"Exiting training loop. (total_messages_seen={total_messages_seen})"
                                )
                                stop_requested = True
                                break
                        # else: offsets show more data exists — keep polling, don't count
                    
                    # Heartbeat so it never looks "stuck" while waiting for producer data.
                    now = time.time()
                    if now - last_heartbeat_time >= heartbeat_every_s:
                        _log(
                            f"Waiting for Kafka data... batch_progress={len(batch_samples)}/{training_args.per_device_train_batch_size}, "
                            f"total_messages_seen={total_messages_seen}, idle_for={now - last_data_time:.1f}s"
                            + (f", consecutive_empty_polls_after_eof={consecutive_empty_polls_after_eof}/{max_consecutive_empty_polls}" if eof_received else "")
                        )
                        last_heartbeat_time = now
                    time.sleep(0.1)
                    continue

                # Reset counter whenever we get data after EOF
                if eof_received:
                    consecutive_empty_polls_after_eof = 0

                for tp, records in messages.items():
                    for message in records:
                        if message is None or message.value is None:
                            continue
                        sample_data = message.value

                        # Skip internal control messages from the producer
                        if isinstance(sample_data, dict) and ("_eof" in sample_data or "_verify" in sample_data):
                            if sample_data.get("_eof"):
                                eof_received = True
                                consecutive_empty_polls_after_eof = 0  # Reset on EOF marker
                                _log("Received end-of-stream marker from producer.")
                            continue

                        if "target" in sample_data:
                            sample_data["target"] = _apply_label_map_if_configured(
                                sample_data.get("target"),
                                config.get("dataset", {}).get("label_map"),
                            )

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

            # CRITICAL MEMORY OPTIMIZATION:
            # Delete massive computational graph references immediately!
            # Since this is a real-time framework, consumer.poll() could block for seconds.
            # If we don't delete these, the Mac hoards the entire graph in Unified Memory while idling.
            del outputs
            del loss
            del scaled_loss
            del batch
            del batch_samples

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
                # Crucial step: Clip gradients to prevent fp16 explosions causing NaN loss
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimization_step += 1

                # Periodic checkpoint save
                if ckpt_manager and optimization_step % save_every_steps == 0:
                    ckpt_manager.save(
                        model, tokenizer, optimization_step,
                        loss=accumulated_loss / training_args.gradient_accumulation_steps,
                    )

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

                # Periodic quantitative evaluation
                if evaluator.enabled and optimization_step % evaluator.eval_interval == 0:
                    _log_eval(optimization_step)

                # Periodic qualitative evaluation (independent cadence)
                if qual_evaluator.enabled and optimization_step % qual_evaluator.eval_interval == 0:
                    _log_qual_eval(optimization_step)

                # Aggressive garbage collection EVERY step to prevent VRAM
                # fragmentation from accumulating across optimization steps.
                import gc
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Every weight_push_interval seconds, send the current LoRA adapter weights.
            if time.time() - last_send_time >= weight_push_interval:
                if kafka_cfg.get('enable_lora_streaming', True):
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

    # 0. Flush any pending partial gradient accumulation so the last few
    #    micro-batches at EOF don't silently lose their gradient signal.
    if not interrupted and grad_accum_counter % training_args.gradient_accumulation_steps != 0:
        try:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optimization_step += 1
            _log(f"Flushed partial gradient accumulation at step {optimization_step} "
                 f"({grad_accum_counter % training_args.gradient_accumulation_steps} pending micro-batches).")
        except Exception as e:
            _log(f"Warning: Partial gradient flush failed: {e}")

    # 1. Save final checkpoint to disk (always, regardless of inline eval settings).
    if ckpt_manager and save_final_ckpt:
        try:
            ckpt_manager.save(model, tokenizer, "final", force=True)
        except Exception as e:
            _log(f"Warning: Failed to save final checkpoint: {e}")

    # 2. Send final LoRA weights + done signal to inference server via Kafka (if enabled).
    try:
        if kafka_cfg.get('enable_lora_streaming', True):
            _log("Training complete. Sending final adapter weights.")
            lora_producer.send_weights(get_peft_model_state_dict(model))
            lora_producer.send_done_signal()
        else:
            _log("Training complete. LoRA streaming disabled, skipping final Kafka push.")
    except Exception as e:
        _log(f"Warning: Failed to send final weights / done signal: {e}")

    # 3. Final evaluation — skip if the user already pressed Ctrl-C (they
    #    want to exit, not wait for a slow model.generate() pass).
    if evaluator.enabled and not interrupted:
        try:
            _log_eval(optimization_step)
        except KeyboardInterrupt:
            _log("Final evaluation interrupted — skipping. Metrics already saved.")
        except Exception as e:
            _log(f"Warning: Final evaluation failed: {e}")

    # 3b. Final qualitative evaluation (independent of quantitative eval)
    if qual_evaluator.enabled and not interrupted:
        try:
            _log_qual_eval(optimization_step)
        except KeyboardInterrupt:
            _log("Final qualitative evaluation interrupted — skipping.")
        except Exception as e:
            _log(f"Warning: Final qualitative evaluation failed: {e}")

    # 4. Always print the CSV location and generate plots.
    _log(f"Metrics CSV saved to: {metrics_logger.metrics_path}")
    try:
        metrics_logger.finalize_csv()
    except Exception as e:
        _log(f"Warning: Clean CSV generation failed: {e}")
    try:
        metrics_logger.generate_plots(config=config)
    except Exception as e:
        _log(f"Warning: Plot generation failed: {e}")
    _log(f"To regenerate plots later: python utils/plot_metrics.py \"{metrics_logger.metrics_path}\"")
    if ckpt_manager:
        _log(f"Checkpoints saved to: {ckpt_manager.checkpoint_root} (run: {ckpt_manager._run_dir})")
        _log("To run decoupled evaluation: python evaluate.py --config <config> [--step N | --all-checkpoints]")

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
    train_model(config, config_path=args.config)
