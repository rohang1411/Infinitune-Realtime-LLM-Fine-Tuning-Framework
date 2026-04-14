# InfiniTune — Complete Project Context & Technical Reference

> **Purpose of this document:** A single-source, end-to-end reference for any person or AI agent who needs to understand the full InfiniTune codebase — its motivation, architecture, design decisions, code internals, and operational details — without having to read any other file.

---

## Table of Contents

1. [Motivation & Problem Statement](#1-motivation--problem-statement)
2. [Use Cases](#2-use-cases)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Repository Structure](#4-repository-structure)
5. [Core Concepts](#5-core-concepts)
   - [QLoRA / LoRA Adapters](#51-qlora--lora-adapters)
   - [Apache Kafka as the Data Backbone](#52-apache-kafka-as-the-data-backbone)
   - [Online (Streaming) Learning](#53-online-streaming-learning)
6. [Component Deep-Dives](#6-component-deep-dives)
   - [producer.py — Data Streamer](#61-producerpy--data-streamer)
   - [trainer.py — Streaming Trainer](#62-trainerpy--streaming-trainer)
   - [inference.py — Live Inference Server](#63-inferencepy--live-inference-server)
   - [utils/stream_filter.py — Data Quality Gate](#64-utilsstream_filterpy--data-quality-gate)
   - [utils/plot_metrics.py — Offline Plot Utility](#65-utilsplot_metricspy--offline-plot-utility)
7. [Configuration System](#7-configuration-system)
   - [Schema Reference (all keys explained)](#71-schema-reference)
   - [IMDb Config Walkthrough](#72-imdb-config-walkthrough)
   - [GSM8K Config Walkthrough](#73-gsm8k-config-walkthrough)
   - [E2E NLG Config Walkthrough](#74-e2e-nlg-config-walkthrough)
8. [Data Flow — Step-by-Step](#8-data-flow--step-by-step)
9. [Key Design Decisions & Engineering Notes](#9-key-design-decisions--engineering-notes)
10. [Training Internals](#10-training-internals)
    - [Label Masking](#101-label-masking)
    - [Gradient Accumulation](#102-gradient-accumulation)
    - [LR Scheduler](#103-lr-scheduler)
    - [Evaluation Strategies](#104-evaluation-strategies)
    - [Metrics Logging & Plots](#105-metrics-logging--plots)
11. [Inference Server Internals](#11-inference-server-internals)
    - [Hot-Swap Mechanism](#111-hot-swap-mechanism)
    - [Thread Safety](#112-thread-safety)
    - [REST API Reference](#113-rest-api-reference)
12. [Dependencies & Tech Stack](#12-dependencies--tech-stack)
13. [Setup & Running the System](#13-setup--running-the-system)
14. [Extending InfiniTune to a New Task](#14-extending-infinitune-to-a-new-task)
15. [Glossary](#15-glossary)

---

## 1. Motivation & Problem Statement

### The Traditional Fine-Tuning Bottleneck

The conventional way to fine-tune a large language model (LLM) looks like this:

```
Collect static dataset → Train offline → Save checkpoint → Deploy new model → Repeat
```

This loop has three major bottlenecks:

| Bottleneck | Description |
|---|---|
| **Static data** | The model only sees data that existed at training time. It cannot incorporate new information without a full re-run. |
| **Offline training** | Training stops production inference. You either maintain two separate model instances (expensive) or take the service offline. |
| **Manual deployment** | Every training cycle requires a human (or CI/CD pipeline) to copy weight files, restart servers, and validate. |

### What InfiniTune Does Differently

InfiniTune eliminates all three bottlenecks:

1. **Static data → Streaming data**: Training samples are published to Apache Kafka in real time as they are generated or collected. The trainer consumes them on-the-fly.
2. **Offline training → Background training**: The trainer runs as an independent background process, never blocking the inference server.
3. **Manual deployment → Automatic weight hot-swap**: After every `weight_push_interval` seconds, the trainer serializes the learned LoRA adapter weights and publishes them back to Kafka. The inference server picks them up and applies them to the live model — **no restart, no downtime**.

The key insight is that **LoRA adapters are small** (often < 1% of the total model parameters), so serializing and transferring them over Kafka is fast and cheap — even for large base models.

---

## 2. Use Cases

InfiniTune is designed for scenarios where:

### Primary Use Cases

| Use Case | Description | Example Config |
|---|---|---|
| **Continuously updating classifiers** | A sentiment/intent/toxicity classifier that must track evolving language, slang, or domain-specific terminology without ever going offline. | `imdb_config.yaml` |
| **Adaptive math / reasoning models** | A model that fine-tunes on a stream of math problems for targeted reasoning skill improvement. | `gsm8k_config.yaml` |
| **Structured data-to-text NLG** | A model that learns to convert structured meaning representations (key-value slot pairs) into fluent natural language descriptions. | `e2e_qualitative.yaml` |
| **Real-time personalization** | An LLM that adapts to a specific user's writing style, topics of interest, or interaction history as new conversations arrive. | (Custom config) |
| **Research / Experimentation** | Researchers who want to study online/continual learning dynamics without building their own training infrastructure from scratch. | Either config |

### Secondary Use Cases (Extensible)

- **Domain adaptation on live data feeds**: A news-domain model that continuously ingests headlines.
- **Active learning loops**: A pipeline where model errors on live traffic are automatically routed back as training examples.
- **Multi-task streaming**: Running multiple Kafka topics simultaneously, each training a different task-specific LoRA adapter on the same base model.

### What InfiniTune Is NOT Designed For

- Full model fine-tuning (it uses LoRA — parameter-efficient)
- Training from scratch or pre-training
- Production-grade distributed training across many GPU nodes (it is single-process, single-GPU or CPU/MPS)

---

## 3. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          InfiniTune System                               │
│                                                                          │
│  ┌──────────────┐   JSON training   ┌────────────────┐                  │
│  │  producer.py │ ─────────────────►│  Kafka Broker  │                  │
│  │              │   samples          │ localhost:9092  │                  │
│  │ HuggingFace  │                   │                │                  │
│  │ Dataset      │                   │  Topics:        │                  │
│  │ + Filtering  │                   │  training-data  │                  │
│  └──────────────┘                   │  lora-updates   │                  │
│                                     └──────┬──┬───────┘                  │
│                                            │  │                          │
│                              training-data │  │ lora-updates             │
│                                            │  │ (serialized tensors)     │
│                                            ▼  └──────────────────────►   │
│                                  ┌──────────────────┐  ┌──────────────┐ │
│                                  │   trainer.py     │  │ inference.py │ │
│                                  │                  │  │              │ │
│                                  │ Base Model       │  │ Base Model   │ │
│                                  │ + LoRA Adapter   │  │ + LoRA       │ │
│                                  │ (QLoRA training) │  │   (hot-swap) │ │
│                                  │                  │  │              │ │
│                                  │ → Pushes weights │  │ Flask REST   │ │
│                                  │   every N secs   │  │ API :5000    │ │
│                                  └──────────────────┘  └──────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

### Three Loosely-Coupled Services

| Service | File | Kafka Role | Direction |
|---|---|---|---|
| **Producer** | `producer.py` | Kafka **Producer** | Writes to `training-data` topic |
| **Trainer** | `trainer.py` | Kafka **Consumer** (reads data) + Kafka **Producer** (writes weights) | Reads training data, writes LoRA weights |
| **Inference Server** | `inference.py` | Kafka **Consumer** | Reads LoRA weight updates |

All three services are stateless with respect to each other: they communicate **only through Kafka topics**. This means you can restart or scale any one of them independently.

---

## 4. Repository Structure

```
Infinitune-Realtime-LLM-Fine-Tuning-Framework/
│
├── producer.py           # Service 1: Streams dataset samples to Kafka
├── trainer.py            # Service 2: Consumes data, fine-tunes model with QLoRA
├── inference.py          # Service 3: Serves REST API, hot-swaps LoRA weights
├── evaluate.py           # Offline evaluation: loads checkpoints, runs eval metrics
│
├── configs/
│   ├── imdb_config.yaml          # IMDb sentiment classification (quantitative)
│   ├── imdb_qualitative.yaml     # IMDb review generation (qualitative eval)
│   ├── gsm8k_config.yaml         # GSM8K math reasoning task
│   └── e2e_qualitative.yaml      # E2E NLG slot-to-text generation
│
├── utils/
│   ├── stream_filter.py          # Data quality filter used by producer.py
│   ├── plot_metrics.py           # Standalone CLI tool: regenerates training plots from CSV
│   ├── eval_metrics_train.py     # Evaluator class: perplexity, accuracy, F1, overlap F1
│   ├── eval_qualitative.py       # QualitativeEvaluator: slot coverage, consistency, TTR
│   └── checkpoint_manager.py    # CheckpointManager: hierarchical LoRA checkpoint save/load
│
├── docs/
│   ├── Infinitune_Project_Context.md   # This file — full architecture reference
│   ├── imdb_qualitative_guide.md       # IMDb generation task guide
│   └── e2e_qualitative_guide.md        # E2E NLG task guide (slot coverage + consistency)
│
├── output/               # Auto-created runtime directory
│   ├── logs/
│   │   └── <run_name>/<timestamp>_<uid>/
│   │       ├── metrics.csv          # All columns (many sparse depending on config)
│   │       ├── metrics_clean.csv    # Only populated columns (recommended for analysis)
│   │       ├── run_params.json      # Snapshot of all config params at training start
│   │       ├── verbose_samples.md   # Markdown table of eval samples (if verbose=true)
│   │       └── *.png                # Auto-generated metric plots
│   └── checkpoints/
│       └── <model>__<dataset>/
│           └── run_<YYYYMMDD-HHMMSS>_<uid>/   # Unique dir per training run
│               ├── step_000200/
│               │   ├── adapter_model.safetensors
│               │   ├── adapter_config.json
│               │   └── checkpoint_meta.json
│               └── final/
│
├── requirements.txt      # Python package dependencies
└── README.md             # Setup and quickstart guide
```

---

## 5. Core Concepts

### 5.1 QLoRA / LoRA Adapters

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning (PEFT) method. Instead of updating all the weights in a model (which can be billions of parameters), LoRA inserts small, trainable low-rank matrices into selected layers.

For a weight matrix `W` (frozen), LoRA adds:
```
W' = W + (B × A) × (alpha / r)
```
Where:
- `A` and `B` are small trainable matrices (the "adapter")
- `r` is the rank (e.g., 8) — controls the number of trainable parameters
- `alpha` is a scaling factor (e.g., 64) — `alpha/r` acts as an effective learning rate multiplier

**Why this matters for InfiniTune:**
- The adapter tensors are tiny (e.g., a few MB for a GPT-2 class model vs. hundreds of MB for the full model)
- Only adapters are serialized and sent over Kafka — this is fast
- The base model stays frozen and is shared between training and inference
- Memory footprint is dramatically smaller than full fine-tuning

In InfiniTune, LoRA is applied via the `peft` library using `get_peft_model()`. The target modules (attention layers) are configured per-task in the YAML config.

**QLoRA** refers to running LoRA on a quantized (4-bit or 8-bit) base model. Currently the configs use `fp16` precision, which is the standard half-precision variant.

### 5.2 Apache Kafka as the Data Backbone

**Kafka** is a distributed event streaming platform. InfiniTune uses it as a message queue / data bus between services.

Key Kafka concepts used:
- **Topic**: A named log of records. InfiniTune uses two: `training-data` and `lora-updates`.
- **Producer**: Writes records to a topic (`producer.py`, `trainer.py` for weights).
- **Consumer**: Reads records from a topic (`trainer.py` for data, `inference.py` for weights).
- **Consumer Group**: A group of consumers sharing work from a topic. InfiniTune uses separate groups for trainer and inference so they each receive all messages independently.
- **Offset**: Position within a topic. On startup, the trainer uses `auto_offset_reset="earliest"` (replay all data) or seeks to the end in test mode.

**Why Kafka instead of a simple queue?**
- Records are persistent — the trainer can be restarted and replay data from the beginning
- Kafka handles backpressure naturally — the producer can be faster or slower than the trainer
- Kafka's durable log is perfect for weight updates — the inference server can catch up if it was briefly unavailable
- Topic-level separation makes it trivial to add new consumers (e.g., a second inference server)

### 5.3 Online (Streaming) Learning

Traditional training processes a fixed dataset in multiple epochs. InfiniTune uses **online learning**: the model is updated on each mini-batch as it arrives from the stream, without storing the full dataset.

Benefits:
- The model continuously adapts to new data without waiting for a full dataset to accumulate
- Memory is proportional to a single mini-batch + the model, not the entire dataset
- The model improves immediately as data flows in, making it suitable for live or rapidly evolving data sources

---

## 6. Component Deep-Dives

### 6.1 `producer.py` — Data Streamer

**Role:** Loads a HuggingFace dataset, applies quality filtering, and publishes each sample as a JSON record to the Kafka `training-data` topic.

#### Startup Sequence
1. Parse `--config` argument and load YAML.
2. Initialize `StreamFilter` with filtering rules from config.
3. Create a `KafkaProducer` (JSON-serialized values, SHA-256-hashed keys).
4. **Send a verification message** (`{"_verify": True}`) and block until it is acknowledged by Kafka — this ensures the broker is reachable before streaming begins.
5. Iterate through all examples from `generate_training_examples()`.

#### `generate_training_examples(config)`
```python
# Pseudocode
load HuggingFace dataset (path, split, optional config_name)
optionally shuffle with a seed (critical for sorted datasets like IMDb)
for each example:
    map input_col → "input", target_col → "target"
    apply label_map if defined (e.g., 0 → "negative", 1 → "positive")
    yield {"input": ..., "target": ...}
```
This normalizes every dataset to the same `{"input": ..., "target": ...}` interface, making the rest of the pipeline dataset-agnostic.

#### Filtering via `StreamFilter`
Before sending each sample, `stream_filter.validate(raw_record, extracted_text)` is called. If it fails, the sample is dropped and a drop reason is logged. Telemetry (ingested count, drop rate, recent drop reasons) is printed every 1000 records or 60 seconds.

#### Message Key
The Kafka message key is a **SHA-256 hash** of the input text (the configured `hash_column`). This makes keys stable and unique — enabling Kafka log compaction (dedup) if needed.

#### End-of-Stream Signal
After the dataset is exhausted, the producer sends `{"_eof": True}` with key `__eof__`. The trainer uses this signal to know it should stop the training loop after draining remaining data.

#### Error Handling
- Delivery errors are tracked in `_delivery_errors`. If more than 10 accumulate, the producer aborts.
- Progress is logged every 5 seconds.

---

### 6.2 `trainer.py` — Streaming Trainer

This is the core of InfiniTune. It is the largest file (937 lines) and contains all training logic.

#### Top-Level Components Inside `trainer.py`

| Class / Function | Purpose |
|---|---|
| `MetricsLogger` | Writes per-step CSV metrics and generates PNG plots |
| `LoRAProducer` | Serializes and pushes LoRA adapter weights to Kafka |
| `tokenize_with_label_masking()` | Tokenizes prompt + response, masks prompt tokens in labels |
| `pad_batch()` | Pads a list of tokenized samples into a batch tensor |
| `Evaluator` | Runs periodic model evaluation (perplexity, accuracy, F1, MCC, kappa) |
| `build_lr_scheduler()` | Builds a `LambdaLR` scheduler from config |
| `train_model(config)` | Main training loop — orchestrates everything |

#### `MetricsLogger`

Writes a CSV file with columns:
```
step, loss, lr, eval_loss, perplexity, accuracy, f1, mcc, kappa,
exact_match, grad_norm, tokens_per_sec, step_time_s, records_used_total
```
Design decision: The CSV file is **opened, written, and closed on every `log()` call** (not kept open). This is intentional for Windows compatibility — Windows blocks readers on open file handles.

At the end of training, `generate_plots()` reads the CSV and saves a PNG for each metric using matplotlib (non-interactive `Agg` backend).

Also writes `run_params.json` — a snapshot of all config params at startup — for reproducibility.

#### `LoRAProducer`

- Serializes each LoRA adapter tensor using `torch.save()` into a `BytesIO` buffer and publishes it to the `lora-updates` Kafka topic.
- Message key = layer name (e.g., `"base_model.model.transformer.h.0.attn.c_attn.lora_A.default.weight"`).
- After training completes, sends a `__done__` sentinel with key `"__done__"` so the inference server knows to stop listening.

#### `tokenize_with_label_masking(tokenizer, prompt_text, response_text, max_seq_length)`

This function is critical to training quality:

```
Input:  prompt_text = "Review: This movie was great.\nSentiment:"
        response_text = " positive"

Tokenize separately:
  prompt_ids  = tokenizer.encode(prompt_text)        # with special tokens
  response_ids = tokenizer.encode(response_text)     # without special tokens
  response_ids += [EOS]               # always append EOS

Truncation strategy:
  max_prompt_len = max_seq_length - len(response_ids)
  prompt_ids = prompt_ids[:max_prompt_len]   # truncate PROMPT, never response

Concatenate:
  input_ids = prompt_ids + response_ids
  labels    = [-100] * len(prompt_ids) + response_ids
              # -100 = ignored by cross-entropy loss (HuggingFace convention)
```

**Why mask the prompt?** The model should learn to produce the *response*, not the prompt. If prompt tokens were included in the loss, the model would waste capacity learning to memorize the prompt template rather than the answer.

#### `pad_batch(samples, pad_token_id, device)`

Pads all samples in a batch to the length of the longest sample:
- `input_ids`: padded with `pad_token_id`
- `attention_mask`: padded with `0` (masked out)
- `labels`: padded with `-100` (ignored in loss)

Returns a dict of `torch.Tensor` objects on the target device.

#### `Evaluator`

The evaluator runs in-process (on the same model, after switching to `.eval()` mode) every `eval_interval` optimizer steps.

**Sliding window evaluation:** Instead of always using the same fixed evaluation set, the evaluator maintains a cursor over the `eval_pool_size` samples. On each eval call, it advances by `eval_batch_size` samples (wrapping around). This ensures the full eval pool is seen over time without evaluating all of it at once.

**Evaluation strategies:**

| Strategy | How it works | When to use |
|---|---|---|
| `perplexity` | Forward pass only — computes `exp(avg_loss)` | Always computed regardless of strategy |
| `class_match` | Generate text, compare first N words to target label | Classification tasks (IMDb) |
| `regex_extract` | Generate text, apply a regex to extract answer, compare to gold | Generative/math tasks (GSM8K) |

**Metrics computed:**

| Metric | Description |
|---|---|
| `eval_loss` | Average cross-entropy loss on eval samples (response tokens only) |
| `perplexity` | `exp(eval_loss)` — lower is better |
| `accuracy` | Fraction of predictions exactly matching gold labels |
| `exact_match` | Accuracy after stripping punctuation (more lenient) |
| `f1` | Macro F1 across all classes (from confusion matrix) |
| `mcc` | Matthews Correlation Coefficient — robust to class imbalance |
| `kappa` | Cohen's Kappa — agreement beyond chance |

All metrics are computed from scratch using only numpy/math (no sklearn dependency) using a confusion matrix built in pure Python.

**Verbose mode:** If `verbose: true` in config, each prediction is printed with a ✓/✗ mark alongside the expected and actual output.

#### `build_lr_scheduler(optimizer, config)`

Builds a `torch.optim.lr_scheduler.LambdaLR` with one of three schedules:

| Type | Behavior |
|---|---|
| `constant` | No scheduling — `lambda = 1.0` throughout |
| `linear` | Linear warmup from 0 → base LR over `warmup_steps`, then linear decay to `min_lr_ratio × base_lr` over `T_max` steps |
| `cosine_with_warmup` | Linear warmup, then cosine decay: `LR = min_lr + (1 - min_lr) × 0.5 × (1 + cos(π × progress))` |

#### `train_model(config)` — The Main Loop

```
1. Parse config sections (model, lora, training, kafka, preprocessing).
2. Initialize MetricsLogger and write run_params.json.
3. Build LoraConfig and wrap the base model with get_peft_model().
4. Detect device (CUDA > MPS > CPU) and move model to device.
5. Create KafkaConsumer for training-data topic.
   - In test_mode: seek to end of topic (consume only new data from this run).
6. Set up optimizer (AdamW) and LR scheduler.
7. Instantiate LoRAProducer and Evaluator.
8. Enter the streaming training loop:

   WHILE optimization_step < max_steps AND not stop_requested:
     a. Poll Kafka for messages, assemble a mini-batch.
        - Skip _verify and _eof control messages.
        - On _eof: set eof_received = True.
        - After EOF + 2s idle: set stop_requested = True.
     b. Tokenize each sample with label masking.
     c. Pad the batch and move to device.
     d. Forward pass → compute loss.
     e. Scale loss by 1/grad_accum_steps, backward pass.
     f. Every grad_accum_steps micro-batches:
        - optimizer.step(), scheduler.step(), optimizer.zero_grad()
        - Log step metrics (loss, LR, grad norm, tok/s, step time).
        - Every eval_interval steps: run Evaluator.evaluate().
     g. Every weight_push_interval seconds:
        - Send LoRA adapter state dict to Kafka via LoRAProducer.

9. On training complete (or Ctrl-C / error):
   a. Send final LoRA weights.
   b. Send __done__ sentinel.
   c. Run final evaluation (unless interrupted).
   d. Generate plots from metrics CSV.
```

**test_mode behavior:**
When `test_mode: true`, the trainer seeks to the **end** of the Kafka topic on startup, so it only consumes records produced *after* the current run starts. It then trains until the EOF marker arrives, rather than stopping at `max_steps`. This enables a reproducible "train on the full dataset, then stop" run useful for benchmarking.

**Heartbeat logging:** While waiting for the producer to send data, the trainer logs every 5 seconds with idle time and batch progress, so it never looks "stuck".

**Gradient norm tracking:** The raw (pre-clip) L2 norm of all gradients is accumulated across micro-batches and logged at each optimizer step. This is useful for diagnosing training instability (exploding/vanishing gradients).

**Token throughput:** Tokens per second is computed as `response_tokens_in_batch / step_wall_time`, giving a hardware-specific throughput metric.

---

### 6.3 `inference.py` — Live Inference Server

**Role:** Loads the base model + initial LoRA adapter, serves a Flask REST API for generation, and continuously hot-swaps LoRA weights in the background as the trainer publishes updates.

#### Startup Sequence
1. Load config from `--config` argument.
2. Detect device (CUDA > MPS > CPU).
3. Load base model with `device_map="auto"` (uses `accelerate` for multi-GPU/MPS support).
4. Apply initial LoRA config with `get_peft_model()`.
5. Set model to `.eval()` mode.
6. Create a `queue.Queue` (the weight update queue) and a `threading.Lock` (the model lock).
7. Start two background daemon threads:
   - `kafka_consumer_thread` — reads LoRA weight updates from Kafka
   - `weight_application_thread` — applies updates to the model
8. Start the Flask server (blocking).

#### Background Threading Architecture

```
Main Thread (Flask):
  ┌─────────────────────────────────────────┐
  │  handle_generate() → generate_text()    │
  │  acquires model_lock before inference   │
  └─────────────────────────────────────────┘
         ▲ model_lock (mutex)
         │
  ┌──────┴──────────────────────────────────┐
  │  weight_application_thread              │
  │  drains update_queue, acquires lock,    │
  │  calls model.load_state_dict()          │
  └──────────────────────────────────────────┘
         ▲ update_queue (thread-safe Queue)
         │
  ┌──────┴──────────────────────────────────┐
  │  kafka_consumer_thread                  │
  │  polls Kafka lora-updates topic,        │
  │  puts (layer_name, tensor) into queue   │
  └─────────────────────────────────────────┘
```

#### `kafka_consumer_thread(update_queue, config)`

- Polls the `lora-updates` Kafka topic continuously.
- Each message is a `(key=layer_name, value=serialized_tensor)` pair.
- Deserializes tensors using `torch.load()` from a `BytesIO` buffer.
- Puts `(layer_name, tensor)` tuples into `update_queue`.
- On receiving `__done__` signal: puts `_DONE_SENTINEL = ("__done__", None)` into the queue and exits.
- Uses `consumer_timeout_ms` and `poll_timeout_ms` from config for responsiveness tuning.
- Uses `auto_offset_reset="latest"` — the inference server only cares about the newest weights, not historical ones.

#### `weight_application_thread(model, update_queue, model_lock, device)`

The key challenge here is applying weight updates **without blocking ongoing inference requests** for longer than necessary. The thread does this cleverly:

1. **Blocking wait** for the first item in the queue (to avoid busy-looping).
2. **Non-blocking drain**: immediately after getting the first item, drain all remaining items currently in the queue in a tight non-blocking loop.
3. Only **then** acquire the model lock and call `model.load_state_dict(updates, strict=False)`.

This batching approach means that if the trainer pushes 10 tensor updates rapidly, they are applied in a single lock acquisition rather than 10 sequential ones — minimizing the time the Flask server's inference is blocked.

`strict=False` in `load_state_dict` means only the adapter keys present in the update dict are updated; unrelated model weights are unchanged.

#### `generate_text(prompt, model, tokenizer, model_lock, device, inference_cfg)`

- Acquires `model_lock` for the entire duration of tokenization + generation.
- Tokenizes the prompt, records `prompt_token_len`.
- Runs `model.generate()` with a `GenerationConfig` object (from config: `max_new_tokens`, `do_sample`, `temperature`, `top_p`).
- **Slices off the prompt tokens** from the output: `generated_ids = outputs[0, prompt_token_len:]`. This ensures only the model's *new* tokens are decoded and returned, not the echoed input prompt.

### 6.4 `utils/stream_filter.py` — Data Quality Gate

`StreamFilter` implements a **two-tier, short-circuit filtering pipeline** for the producer. All filtering rules are read from the config YAML at construction time.

**Ordering principle (fastest checks first):**

| Tier | Complexity | Checks |
|---|---|---|
| 1 | O(1) | `min_chars`, `max_chars` — simple `len()` checks |
| 2 | O(N) | `require_numeric_content` — `any(ch.isdigit())`, `min_alphanumeric_ratio` — count alphanumeric chars |
| 3 | O(N) | `max_repetition_ratio` — zlib compression ratio (compressed/original ≤ threshold = repetitive text) |
| 4 | O(N) | Chat structure validation (for dialogue datasets) |
| 5 | O(N×R) | Custom regex patterns (`must_match`, `must_not_match`) — most expensive, always last |

**Repetition detection via zlib:** When text is highly repetitive (e.g., "A A A A A A"), zlib compresses it very efficiently, producing a very low `compressed_size / original_size` ratio. InfiniTune rejects any text where this ratio falls below `max_repetition_ratio` (e.g., 0.2).

**Chat-structure filter:** For datasets formatted as lists of `{"role": ..., "content": ...}` dicts, the filter can check:
- Minimum number of turns (`min_turns`)
- Whether the last message is from the assistant (`require_assistant_final`)

**Returns:** `(is_valid: bool, reason: str | None)` — the reason string is used for telemetry/debugging but never exposed to the model.

**Safety:** If the `validate()` function crashes for any reason, it returns `(True, None)` — fail open, to avoid silently starving the trainer.

### 6.5 `utils/plot_metrics.py` — Offline Plot Utility

A standalone CLI script for regenerating training plots from a `metrics.csv` file.

```bash
# Regenerate plots after the run
python utils/plot_metrics.py "output/imdb/logs/infinitune-imdb-sentiment/20240315-120000/metrics.csv"

# Save to a custom directory
python utils/plot_metrics.py metrics.csv --out-dir ./my_analysis_plots
```

Generates one PNG per metric:
- `train_loss.png`, `eval_loss.png`, `perplexity.png`
- `accuracy.png`, `f1.png`, `mcc.png`, `kappa.png`, `exact_match.png`
- `grad_norm.png`, `tokens_per_sec.png`

Useful after a crash (metrics CSV is written incrementally), after a Ctrl-C interrupt, or for sharing results without re-running training.

---

## 7. Configuration System

All three services share a single YAML config file. Every config section maps directly to one part of the system.

### 7.1 Schema Reference

```yaml
# ── Project ──────────────────────────────────────────────
project:
  name: "my-run"          # Used as folder name for output logs
  output_dir: "./output"  # Root output directory

# ── Model ────────────────────────────────────────────────
model:
  name: "distilgpt2"      # HuggingFace model ID (e.g., "gpt2", "meta-llama/Llama-2-7b-hf")
  task_type: "CAUSAL_LM"  # CAUSAL_LM | SEQ_2_SEQ_LM | TOKEN_CLS | SEQ_CLS
  precision: "fp16"       # fp16 | fp32 | 4bit
  max_seq_length: 256     # Max tokens per sample (prompt + response)

# ── LoRA Adapter ─────────────────────────────────────────
lora:
  r: 8                    # Adapter rank. Higher = more capacity, more params. Typical: 4–64.
  alpha: 64               # Scaling factor. Effective LR multiplier = alpha / r.
  dropout: 0.05           # Dropout on the adapter layers during training
  bias: "none"            # "none" | "all" | "lora_only"
  target_modules:         # Which weight matrices to add adapters to.
    - "c_attn"            # GPT-2: combined Q/K/V projection
    - "c_proj"            # GPT-2: output projection
    # For LLaMA: ["q_proj", "v_proj"] is typical

# ── Dataset ──────────────────────────────────────────────
dataset:
  name: "imdb"            # HuggingFace dataset path/name
  config_name: null       # Optional dataset config (e.g., "main" for gsm8k)
  split: "train"          # Which split the producer streams
  eval_split: "test"      # Which split the evaluator loads
  column_mapping:
    input_col: "text"     # Column to use as the prompt input
    target_col: "label"   # Column to use as the target/answer
  label_map:              # Optional: map raw label values → string labels
    0: "negative"
    1: "positive"
  shuffle: true           # Shuffle dataset before streaming (critical for sorted datasets)
  shuffle_seed: 42        # Reproducible shuffle; null = random

# ── Preprocessing ────────────────────────────────────────
preprocessing:
  prompt_template: "Review: {{ input }}\nSentiment:"   # Jinja2 template for the prompt
  response_template: " {{ target }}"                   # Jinja2 template for the response
  hash_column: "input"   # Which column to SHA-256-hash as the Kafka message key

# ── Data Filtering ───────────────────────────────────────
data:
  filtering:
    universal:
      min_chars: 15             # Drop samples shorter than this
      max_chars: 4000           # Drop samples longer than this
      min_alphanumeric_ratio: 0.5  # Drop if <50% of chars are alphanumeric
      max_repetition_ratio: 0.2    # Drop if zlib_ratio <= 0.2 (highly repetitive)
    domain_specific:
      require_numeric_content: null  # true = drop if no digits present
      custom_regex_must_match: null  # List of regex patterns that must match
      custom_regex_must_not_match: null # List of patterns that must NOT match
      chat_structure:
        min_turns: null              # Minimum dialogue turns
        require_assistant_final: null  # Last message must be from assistant

# ── Kafka ────────────────────────────────────────────────
kafka:
  bootstrap_servers:
    - "localhost:9092"          # Kafka broker address(es)
  training_topic: "training-data-imdb"    # Topic for training samples
  lora_updates_topic: "lora-updates-imdb" # Topic for LoRA weight updates
  producer_send_interval: 0.1   # Seconds between samples (throttle speed)
  consumer_group_trainer: "trainer-group"           # Kafka consumer group for trainer
  consumer_group_inference: "inference-api-group"   # Kafka consumer group for inference
  poll_timeout_ms: 1000         # How long consumer.poll() waits before returning empty
  consumer_timeout_ms: 1000     # KafkaConsumer session timeout

# ── Training ─────────────────────────────────────────────
training:
  test_mode: true               # If true: seek to end, train entire dataset, stop on EOF
  batch_size: 8                 # Micro-batch size
  gradient_accumulation_steps: 4  # Effective batch = batch_size × grad_accum = 32
  learning_rate: 1e-4
  max_steps: 2000               # Max optimizer steps (ignored in test_mode)
  logging_steps: 1              # Log metrics every N optimizer steps
  weight_push_interval: 60      # Push LoRA weights to Kafka every N seconds
  lr_scheduler:
    type: "cosine_with_warmup"  # constant | linear | cosine_with_warmup
    warmup_steps: 50
    min_lr_ratio: 0.01          # Floor LR = base_lr × min_lr_ratio
    T_max: 1000                 # Scheduler period in optimizer steps

# ── Inference ────────────────────────────────────────────
inference:
  host: "localhost"
  port: 5000
  max_new_tokens: 6             # Max tokens to generate per request
  do_sample: false              # false = greedy decoding; true = sampling
  temperature: 0.7              # Sampling temperature (only used if do_sample=true)
  top_p: 0.9                    # Nucleus sampling (only used if do_sample=true)

# ── Evaluation ───────────────────────────────────────────
evaluation:
  enabled: true
  strategy: "class_match"       # perplexity | class_match | regex_extract
  eval_interval: 50             # Evaluate every N optimizer steps
  eval_pool_size: 5000          # Number of eval samples to load at startup
  eval_batch_size: 100          # Samples to evaluate per eval call (sliding window)
  verbose: false                # Print per-sample predictions if true
  answer_regex: null            # Regex to extract answer (used with regex_extract)
```

### 7.2 IMDb Config Walkthrough

**Task:** Sentiment classification — given a movie review, output "positive" or "negative".

Key config choices:
- `model.max_seq_length: 256` — IMDb reviews can be long; 256 tokens is a reasonable truncation for DistilGPT-2's context window.
- `label_map: {0: "negative", 1: "positive"}` — converts integer labels to English words the LM can generate.
- `prompt_template: "Review: {{ input }}\nSentiment:"` — a simple instruction prompt.
- `response_template: " {{ target }}"` — the expected completion is a single word.
- `evaluation.strategy: "class_match"` — compare the first word of generated output to the target label.
- `inference.max_new_tokens: 6` — classification only needs a few tokens; this avoids the model rambling.
- `inference.do_sample: false` — greedy decoding for deterministic classification at inference time.
- `lora.target_modules: ["c_attn", "c_proj"]` — adapts both the attention and projection layers in GPT-2.

### 7.3 GSM8K Config Walkthrough

**Task:** Math reasoning — given a grade-school math word problem, generate a step-by-step solution ending with `#### <answer>`.

Key config choices:
- `model.max_seq_length: 512` — math solutions can be longer than sentiment labels.
- `label_map: null` — no label remapping; the target is a free-form text solution.
- `prompt_template: "Question: {{ input }}\nAnswer:"` — direct question-answer format.
- `evaluation.strategy: "regex_extract"` — extract the final numeric answer using `#### (\d+)` regex.
- `evaluation.answer_regex: "#### (\\d+)"` — applied to both the gold answer and the model's output.
- `inference.max_new_tokens: 150` — longer completions needed for step-by-step reasoning.
- `inference.do_sample: true` — sampling allows the model to explore different reasoning paths.
- `training.batch_size: 2` — smaller batch because math sequences are longer and use more GPU memory.
- `lora.target_modules: ["c_attn"]` — only the attention QKV layer (simpler config for the reasoning task).

---

### 7.4 E2E NLG Config Walkthrough

**Task:** Structured data-to-text NLG — convert a meaning representation (MR) containing restaurant slot-value pairs into a fluent English sentence.

**Dataset:** [E2E NLG Challenge](https://huggingface.co/datasets/e2e_nlg) — ~33,525 training samples, each with an `mr` (structured input) and `human_reference` (reference text).

**Model:** `distilgpt2` — lightweight enough for fast iteration on Colab T4.

Key config choices:
- `evaluation.strategy: "perplexity"` — no exact-match classification; the model generates free-form text, so perplexity is the primary quantitative signal.
- `evaluation.metrics.compute_answer_overlap_f1: true` — token-level F1 between the generated output and the reference `human_reference`. This approximates ROUGE-1 without requiring the `rouge_score` library.
- `testing_strategy.method: "structured_slot_coverage"` — the qualitative evaluator dynamically parses `slot[value]` pairs from the `mr` field and checks if each value appears in the generated output. No pre-defined keyword list needed.
- `testing_strategy.consistency_runs: 10` — runs model.generate() 10 times per MR to measure output stability. Higher = more reliable consistency score, but 10× the generation time at each eval interval.
- `testing_strategy.eval_interval: 200` — aligned with `save_every_steps: 200` so each checkpoint coincides with a qualitative eval snapshot.
- `training.test_mode: true` — the trainer trains until all 33,525 streaming samples are consumed (EOF signal), not until `max_steps`. `max_steps: 2100` is a safety cap only.
- `kafka.producer_send_interval: 0.05` — fast streaming (20 msg/sec), so the producer finishes in ~28 minutes; the trainer catches up over the full training run.
- `lora.r: 8, alpha: 32` — standard LoRA config for a generation task. Increase `r` to 16 for more capacity if slot coverage plateaus.

**Kafka reliability settings (trainer.py):**
- `max_poll_interval_ms: 1800000` (30 min) — qualitative eval at `consistency_runs=10` takes 15–25 minutes. This prevents the consumer from being kicked from the group during eval.
- Unique consumer group per run — ensures no stale committed offsets from previous incomplete runs.
- Offset-based EOF draining — trainer only terminates after verifying all Kafka offsets are consumed, not just after a fixed number of empty polls.

> See `docs/e2e_qualitative_guide.md` for the full operational guide including startup instructions, expected metric ranges, and troubleshooting.

---

## 8. Data Flow — Step-by-Step

Below is a detailed walkthrough of a single training example moving through the system.

```
[IMDb Test Example]
Raw HuggingFace record:
  {"text": "An absolutely incredible film. The acting was superb...", "label": 1}

STEP 1 — producer.py: generate_training_examples()
  Apply label_map: 1 → "positive"
  Emit: {"input": "An absolutely incredible film...", "target": "positive"}

STEP 2 — producer.py: StreamFilter.validate()
  len("An absolutely incredible film...") = 512 chars → passes min_chars, max_chars
  alphanumeric_ratio = 0.81 → passes min_alphanumeric_ratio=0.5
  zlib_ratio = 0.61 → passes max_repetition_ratio=0.2 (not repetitive)
  Result: (True, None) → sample is accepted

STEP 3 — producer.py: KafkaProducer.send()
  key = SHA-256("An absolutely incredible film...")  → "3f4a9bc..."
  value = json.dumps({"input": "...", "target": "positive"}).encode("utf-8")
  Published to Kafka topic: "training-data-imdb"

STEP 4 — trainer.py: KafkaConsumer.poll()
  Receives the JSON record: {"input": "...", "target": "positive"}
  Renders Jinja2 templates:
    prompt_text   = "Review: An absolutely incredible film...\nSentiment:"
    response_text = " positive"

STEP 5 — trainer.py: tokenize_with_label_masking()
  prompt_ids   = [50256, 36819, 25, 1052, ...]    # ~240 tokens
  response_ids = [2068, 50256]                    # "positive" + EOS
  labels       = [-100, -100, ..., 2068, 50256]   # mask all prompt tokens

STEP 6 — trainer.py: pad_batch() + forward pass
  Batch assembled from 8 samples. Padded to max_len.
  outputs = model(input_ids, attention_mask, labels)
  loss = outputs.loss                              # only response tokens contribute
  scaled_loss = loss / 4  (gradient_accumulation_steps)
  scaled_loss.backward()

STEP 7 — trainer.py: optimizer step (every 4 micro-batches)
  optimizer.step() → updates LoRA adapter weights
  scheduler.step() → adjusts learning rate

STEP 8 — trainer.py: LoRAProducer.send_weights() (every 60 seconds)
  adapter_state_dict = get_peft_model_state_dict(model)
  For each layer tensor:
    Serialize to BytesIO via torch.save()
    Publish to Kafka topic "lora-updates-imdb"
    key = "base_model.model.h.5.attn.c_attn.lora_A.default.weight"

STEP 9 — inference.py: kafka_consumer_thread
  Receives (layer_name, tensor) from "lora-updates-imdb"
  Puts into update_queue

STEP 10 — inference.py: weight_application_thread
  Drains update_queue, acquires model_lock
  model.load_state_dict({layer_name: tensor, ...}, strict=False)
  Live model is now updated — no restart

STEP 11 — inference.py: Flask API
  POST /generate {"prompt": "Review: This movie was terrible.\nSentiment:"}
  generate_text() → acquires model_lock → model.generate()
  Returns: {"generated_text": "negative"}
```

---

## 9. Key Design Decisions & Engineering Notes

| Decision | Rationale |
|---|---|
| **Kafka as the transport layer** | Persistent, replayable, handles backpressure, supports multiple consumers, broker-mediated so services don't need to know each other's addresses. |
| **LoRA-only weight transfer** | Full model weights would be GB-scale and impractically slow over Kafka. LoRA adapters are MB-scale and transfer in seconds. |
| **Prompt-only label masking** | Prevents the model from wasting capacity learning to predict the (known, templated) prompt. Loss is computed exclusively on response tokens. |
| **Truncate prompt, not response** | The response (target) must always fit fully so the model sees the complete answer during training. The prompt can be truncated because the model still gets the critical context. |
| **EOS token appended to every response** | Teaches the model to stop generating after the answer, preventing infinite repetition at inference time. |
| **CSV opened/closed per write** | Windows-specific: An open file handle blocks other processes from reading the file. Close after every write for Windows compatibility. |
| **SHA-256 hash as Kafka message key** | Enables Kafka log compaction (keeps only the latest record per key). Also deduplicates re-sent data. |
| **Verification message on producer startup** | Detects broker connectivity issues immediately, rather than silently failing after streaming has begun. |
| **test_mode: seek to end of topic** | Prevents re-processing data from a previous producer run when the topic already has messages. |
| **Sliding window evaluation** | Evaluating the same fixed 100 samples every time would be uninformative if they were all of one class. The sliding window ensures the full eval pool is covered over time. |
| **StreamFilter: short-circuit ordering** | O(1) checks first, O(N) scans second, O(N×R) regex last. This minimizes CPU time on the hot path (thousands of samples per minute). |
| **Weight application thread batching** | Block-wait for first update, then drain the queue non-blockingly before acquiring the model lock. Minimizes lock contention and the duration that inference is blocked. |
| **Heartbeat logging in trainer** | Makes the trainer appear "alive" while waiting for producer data. Critical for debugging — distinguishes "stuck" from "waiting". |
| **Fail-open in StreamFilter** | If `validate()` crashes unexpectedly, it returns `(True, None)`. This prevents one malformed record from permanently starving the trainer. |
| **Auto-detect Kafka API version** | Pinning `api_version=(0, 10)` causes silent delivery failures on Kafka 3.x+. The producer intentionally uses auto-detect. |

---

## 10. Training Internals

### 10.1 Label Masking

The HuggingFace convention for causal language model training is:
- `labels` tensor has the same shape as `input_ids`
- Tokens with label `-100` are **ignored** by the cross-entropy loss function
- InfiniTune masks all prompt tokens to `-100`, so only the response tokens contribute to the loss

This is sometimes called "instruction fine-tuning" or "supervised fine-tuning (SFT)" in the literature.

### 10.2 Gradient Accumulation

The effective batch size = `batch_size × gradient_accumulation_steps`.

In the IMDb config: `8 × 4 = 32` effective batch size.

The loss is scaled by `1 / gradient_accumulation_steps` before `.backward()` so that gradients are averaged (not summed) across micro-batches. This exactly simulates training with a larger batch size when GPU memory is insufficient to hold `batch_size * grad_accum` samples at once.

### 10.3 LR Scheduler

The scheduler is stepped once per **optimizer step** (every `gradient_accumulation_steps` micro-batches), not once per micro-batch.

For `cosine_with_warmup`:
```
if step < warmup_steps:
    lambda = step / warmup_steps
else:
    progress = (step - warmup_steps) / (T_max - warmup_steps)
    lambda = min_lr_ratio + (1 - min_lr_ratio) × 0.5 × (1 + cos(π × progress))
```

The `min_lr_ratio` floor prevents the LR from decaying to zero, which can cause the model to stop learning even if new data arrives.

### 10.4 Evaluation Strategies

**`class_match`** strategy detail:
```python
target_words = gold.split()           # e.g., ["positive"]
response_words = response.lower().split()
pred_words = response_words[:len(target_words)]  # take first N words
pred = " ".join(pred_words)
```
This handles multi-word labels (e.g., "very positive") without being fooled by the model repeating itself (e.g., "positive positive positive" → correctly extracts just "positive").

**`regex_extract`** strategy detail:
```python
# For GSM8K: answer_regex = "#### (\d+)"
pred_match = re.search(answer_regex, response)
gold_match = re.search(answer_regex, sample['target'])
pred = pred_match.group(1).strip().lower()  # the captured number
gold = gold_match.group(1).strip().lower()
```
Applied to both the model's output and the gold answer, so both are normalized before comparison.

**Greedy decoding for evaluation:** The evaluator always uses `do_sample=False` (greedy) regardless of the inference config. This ensures deterministic, reproducible evaluation results.

### 10.5 Metrics Logging & Plots

**CSV structure:** The metrics CSV has one row per event (training step or eval step). Not all columns are filled on every row — training step rows fill `loss, lr, grad_norm, tokens_per_sec, step_time_s`; eval step rows fill `eval_loss, perplexity, accuracy, f1, mcc, kappa, exact_match`. The `plot_metrics.py` script handles missing values gracefully by skipping blanks.

**Plot generation timing:**
- Auto-generated at the end of every training run (in the CSV's directory).
- Can be regenerated at any time from a saved CSV using `utils/plot_metrics.py`.
- Uses non-interactive `Agg` backend so they work on headless servers.

---

## 11. Inference Server Internals

### 11.1 Hot-Swap Mechanism

The core hot-swap mechanism is `model.load_state_dict(updates, strict=False)`:

- `strict=False` means keys missing from `updates` are silently ignored — only the present adapter weight tensors are updated.
- After applying, the next call to `model.generate()` uses the updated adapter weights automatically.
- The update is atomic from the perspective of a single generation call — the lock ensures no generation spans a weight update.

### 11.2 Thread Safety

The `model_lock` (a `threading.Lock`) is the only synchronization primitive. It is acquired:
1. By `weight_application_thread` during `model.load_state_dict()`.
2. By `generate_text()` for the entire duration of tokenization + `model.generate()`.

This means:
- A weight update cannot happen during a running inference request (safe).
- A second inference request must wait for the first to complete (single-threaded generation — a performance limitation, not a correctness issue).
- Flask runs with `threaded=True` so multiple HTTP requests can be queued, but only one will generate at a time (due to the model lock).

### 11.3 REST API Reference

**`POST /generate`**

Request body (JSON):
```json
{"prompt": "Review: This movie was absolutely terrible.\nSentiment:"}
```

Response (JSON):
```json
{"generated_text": "negative"}
```

Error responses:
- `400` — Missing `prompt` field, or non-JSON body
- `503` — Model not initialized yet
- `500` — Internal server error during generation

**`GET /health`**

Response:
```json
{"status": "ok"}
```

---

## 12. Dependencies & Tech Stack

| Package | Version Constraint | Purpose |
|---|---|---|
| `torch` | Any (auto-selects CUDA/MPS/CPU) | Tensor ops, model training, inference |
| `transformers` | Any | HuggingFace model + tokenizer loading, `GenerationConfig` |
| `peft` | Any | LoRA adapter implementation (`LoraConfig`, `get_peft_model`, `get_peft_model_state_dict`) |
| `datasets` | Any | HuggingFace dataset loading + shuffling |
| `kafka-python` | Any | Kafka producer and consumer clients |
| `flask` | Any | REST API server for inference |
| `accelerate` | Any | `device_map="auto"` multi-device model loading |
| `trl` | Any | Imported but not directly used in current code (available for future SFT utilities) |
| `wandb` | Any | In requirements but not integrated in current codebase (future experiment tracking) |
| `matplotlib` | Any | Training metric plot generation |
| `jinja2` | (transitive dep) | Prompt/response template rendering |
| `zlib` | (stdlib) | Compression-based repetition detection in StreamFilter |
| `hashlib` | (stdlib) | SHA-256 hashing for Kafka message keys |

**Platform support:**
- **CUDA GPU (Linux/Windows):** Primary target. FP16 training enabled.
- **Apple Silicon MPS (macOS):** Supported. Device auto-detected. FP16 not enabled on MPS (falls through to FP32 or MPS native).
- **CPU:** Fallback. Training will be very slow but functional.

---

## 13. Setup & Running the System

### Prerequisites

1. **Java JDK 11+** — required by Kafka. Set `JAVA_HOME` environment variable.
2. **Apache Kafka 3.3+** — KRaft mode (no Zookeeper). Download from [kafka.apache.org](https://kafka.apache.org/downloads).
3. **Python 3.9+** — with dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

### Start Kafka (Windows, KRaft mode)

```bat
cd C:\kafka
:: One-time setup (first use only):
.\bin\windows\kafka-storage.bat format -t <UUID> -c .\config\server.properties

:: Start the broker:
.\bin\windows\kafka-server-start.bat .\config\server.properties
```

### Launch InfiniTune (4 terminals)

```bash
# Terminal 1 — Start inference server FIRST (must be ready to receive weight updates)
python inference.py --config configs/imdb_config.yaml

# Terminal 2 — Start trainer (waits for data, logs ">>> Start the producer now <<<")
python trainer.py --config configs/imdb_config.yaml

# Terminal 3 — Start producer AFTER trainer is ready
python producer.py --config configs/imdb_config.yaml

# Terminal 4 — Test the API
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Review: One of the best films I have ever seen.\nSentiment:"}'
```

### Startup Order Matters

| Order | Reason |
|---|---|
| Inference first | Uses `auto_offset_reset="latest"` — it only wants weight updates sent _after_ it starts. If the trainer runs first and sends weights before inference starts, those weights will be missed. |
| Trainer second | In `test_mode`, it seeks to the end of the training topic. This must happen before the producer starts publishing so no data is missed. |
| Producer last | The trainer logs a "Start the producer now" message when ready. This is the cue. |

---

## 14. Extending InfiniTune to a New Task

To add a new task (e.g., news category classification), only a new YAML config file is needed:

1. **Create `configs/my_task_config.yaml`** by copying an existing config.
2. **Set `dataset.name`** to the HuggingFace dataset (e.g., `"ag_news"`).
3. **Set `dataset.column_mapping`** to map the dataset's columns to `input_col` and `target_col`.
4. **Set `dataset.label_map`** if labels are integers.
5. **Update `preprocessing.prompt_template`** and `preprocessing.response_template` using Jinja2 `{{ input }}` / `{{ target }}` placeholders.
6. **Set `kafka.training_topic`** and `kafka.lora_updates_topic`** to unique topic names (avoids mixing data from different tasks).
7. **Choose `evaluation.strategy`**: `class_match` for classification, `regex_extract` for extractive generation, or `perplexity` only for open-ended generation.
8. **Adjust `lora.target_modules`** based on the model architecture (e.g., `["q_proj", "v_proj"]` for LLaMA-family models).
9. **Adjust `inference.max_new_tokens`** to match expected output length.

No Python code changes are required for new tasks — the system is fully config-driven.

---

## 15. Glossary

| Term | Definition |
|---|---|
| **LoRA** | Low-Rank Adaptation — fine-tuning method that inserts small trainable matrices into frozen model layers |
| **QLoRA** | LoRA applied to a quantized (4-bit/8-bit) base model to further reduce memory usage |
| **PEFT** | Parameter-Efficient Fine-Tuning — umbrella term for methods like LoRA that train far fewer parameters than the full model |
| **KafkaProducer** | Client that publishes records to Kafka topics |
| **KafkaConsumer** | Client that reads records from Kafka topics |
| **Consumer Group** | A set of consumers sharing a Kafka topic; each partition is assigned to one consumer per group |
| **auto_offset_reset** | Consumer config: `"earliest"` = start from beginning of topic, `"latest"` = only new messages |
| **Label masking** | Setting label tokens to `-100` so cross-entropy loss ignores them; used to train on response-only |
| **Gradient accumulation** | Summing gradients over N micro-batches before calling `optimizer.step()`, simulating a larger batch |
| **EOS token** | End-of-sequence token; appended to every response so the model learns to stop generating |
| **Hot-swap** | Updating model weights at runtime without restarting the serving process |
| **Jinja2 template** | A text template with `{{ variable }}` placeholders used to format prompts and responses |
| **LambdaLR** | PyTorch LR scheduler that applies a user-defined function to compute the learning rate multiplier each step |
| **Online learning** | Training on a stream of examples one at a time (or in small batches), rather than on a fixed dataset |
| **Perplexity** | `exp(cross_entropy_loss)` — measure of how "surprised" the model is by the data; lower is better |
| **MCC** | Matthews Correlation Coefficient — classification quality metric robust to class imbalance |
| **Cohen's Kappa** | Agreement metric that corrects for chance agreement between predicted and actual labels |
| **Macro F1** | Average F1 score across all classes, treating each class equally regardless of frequency |
| **Sliding window eval** | Evaluating on a rotating window of the eval pool rather than the same fixed subset every time |
| **KRaft mode** | Kafka Raft consensus — Kafka without Zookeeper, available since Kafka 3.3 |
| **zlib compression ratio** | `len(zlib.compress(text)) / len(text)` — used to detect repetitive/spam text efficiently |
| **`strict=False` in load_state_dict** | Only update keys present in the state dict; ignore missing keys in the model |
| **`device_map="auto"`** | HuggingFace `accelerate` feature: automatically distributes model layers across available devices |
