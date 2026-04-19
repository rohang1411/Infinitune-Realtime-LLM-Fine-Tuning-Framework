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
   - [Decoupled Evaluation Architecture](#54-decoupled-evaluation-architecture)
   - [Hierarchical Checkpoint Management](#55-hierarchical-checkpoint-management)
6. [Component Deep-Dives](#6-component-deep-dives)
   - [producer.py — Data Streamer](#61-producerpy--data-streamer)
   - [trainer.py — Streaming Trainer](#62-trainerpy--streaming-trainer)
   - [inference.py — Live Inference Server](#63-inferencepy--live-inference-server)
   - [evaluate.py — Decoupled Offline Evaluator](#64-evaluatepy--decoupled-offline-evaluator)
   - [utils/stream_filter.py — Data Quality Gate](#65-utilsstream_filterpy--data-quality-gate)
   - [utils/checkpoint_manager.py — Checkpoint Manager](#66-utilscheckpoint_managerpy--checkpoint-manager)
   - [utils/eval_metrics_train.py — Quantitative Evaluator](#67-utilseval_metrics_trainpy--quantitative-evaluator)
   - [utils/eval_qualitative.py — Qualitative Evaluator](#68-utilseval_qualitativepy--qualitative-evaluator)
   - [utils/evaluation_artifacts.py — Artifact Orchestrator](#69-utilsevaluation_artifactspy--artifact-orchestrator)
   - [utils/plot_metrics.py — Offline Plot & Report Utility](#610-utilsplot_metricspy--offline-plot--report-utility)
7. [Configuration System](#7-configuration-system)
   - [Config Naming Convention](#71-config-naming-convention)
   - [Schema Reference (all keys explained)](#72-schema-reference)
   - [IMDb Quantitative Config Walkthrough](#73-imdb-quantitative-config-walkthrough)
   - [IMDb Qualitative Config Walkthrough](#74-imdb-qualitative-config-walkthrough)
   - [GSM8K Quantitative Config Walkthrough](#75-gsm8k-quantitative-config-walkthrough)
   - [GSM8K Qualitative Config Walkthrough](#76-gsm8k-qualitative-config-walkthrough)
   - [Alpaca Qualitative Config Walkthrough](#77-alpaca-qualitative-config-walkthrough)
   - [E2E NLG Config Walkthrough](#78-e2e-nlg-config-walkthrough)
8. [Data Flow — Step-by-Step](#8-data-flow--step-by-step)
9. [Key Design Decisions & Engineering Notes](#9-key-design-decisions--engineering-notes)
10. [Training Internals](#10-training-internals)
    - [Label Masking](#101-label-masking)
    - [Gradient Accumulation](#102-gradient-accumulation)
    - [LR Scheduler](#103-lr-scheduler)
    - [Gradient Clipping & Stability](#104-gradient-clipping--stability)
    - [Gradient Checkpointing](#105-gradient-checkpointing)
    - [Quantitative Evaluation Strategies](#106-quantitative-evaluation-strategies)
    - [Qualitative Evaluation Strategies](#107-qualitative-evaluation-strategies)
    - [Metrics Catalog](#108-metrics-catalog)
    - [Metrics Logging & Plots](#109-metrics-logging--plots)
    - [Evaluation Artifact Bundle](#1010-evaluation-artifact-bundle)
11. [Inference Server Internals](#11-inference-server-internals)
    - [Hot-Swap Mechanism](#111-hot-swap-mechanism)
    - [Thread Safety](#112-thread-safety)
    - [Streaming vs Static Checkpoint Mode](#113-streaming-vs-static-checkpoint-mode)
    - [REST API Reference](#114-rest-api-reference)
12. [Optimizations & Engineering Hardening](#12-optimizations--engineering-hardening)
    - [Apple Silicon / Unified Memory Sweep](#121-apple-silicon--unified-memory-sweep)
    - [Precision Policy — fp32 as Stable Default](#122-precision-policy--fp32-as-stable-default)
    - [Gradient Clipping](#123-gradient-clipping)
    - [Batched Qualitative Generation](#124-batched-qualitative-generation)
    - [GPU Multi-Sequence Consistency Runs](#125-gpu-multi-sequence-consistency-runs)
    - [Kafka Consumer Long-Eval Reliability](#126-kafka-consumer-long-eval-reliability)
    - [UUID Run Isolation](#127-uuid-run-isolation)
    - [Producer Topic Hygiene](#128-producer-topic-hygiene)
    - [Decoupled Flag — Removing Eval from the Hot Path](#129-decoupled-flag--removing-eval-from-the-hot-path)
    - [Balanced Sliding-Window Evaluation](#1210-balanced-sliding-window-evaluation)
    - [Left-Padding for Batched Generation](#1211-left-padding-for-batched-generation)
    - [Lazy-Loaded Heavy Models](#1212-lazy-loaded-heavy-models)
13. [Dependencies & Tech Stack](#13-dependencies--tech-stack)
14. [Setup & Running the System](#14-setup--running-the-system)
15. [Extending InfiniTune to a New Task](#15-extending-infinitune-to-a-new-task)
16. [Evolution & Modernization Timeline](#16-evolution--modernization-timeline)
17. [Glossary](#17-glossary)

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
| **Continuously updating classifiers** | A sentiment/intent/toxicity classifier that must track evolving language, slang, or domain-specific terminology without ever going offline. | `imdb_quantitative.yaml` |
| **Adaptive math / reasoning models** | A model that fine-tunes on a stream of math problems for targeted reasoning skill improvement. | `gsm8k_quantitative.yaml` |
| **Domain language adoption** | A model that learns to generate domain-specific vocabulary and style from a continuous stream of in-domain text. | `imdb_qualitative.yaml` |
| **Structured data-to-text NLG** | A model that learns to convert structured meaning representations (key-value slot pairs) into fluent natural language descriptions. | `e2e_qualitative.yaml` |
| **Chain-of-Thought reasoning** | A model that is proven to adopt step-by-step logical structure, not just arrive at correct answers by pattern matching. | `gsm8k_qualitative.yaml` |
| **Instruction following** | A model that semantically converges toward golden responses on open-ended instruction-following tasks. | `alpaca_qualitative.yaml` |
| **Real-time personalization** | An LLM that adapts to a specific user's writing style, topics of interest, or interaction history as new conversations arrive. | (Custom config) |
| **Research / Experimentation** | Researchers who want to study online/continual learning dynamics without building their own training infrastructure from scratch. | Any config |

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

### Standard Streaming Training Mode

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
│                                  │ → Saves checkpts │  │ Flask REST   │ │
│                                  │ → Pushes weights │  │ API :5000    │ │
│                                  └────────┬─────────┘  └──────────────┘ │
│                                           │                              │
│                                    output/checkpoints/                   │
│                                           │                              │
│                                           ▼                              │
│                               ┌─────────────────────┐                   │
│                               │     evaluate.py      │                   │
│                               │  (No Kafka required) │                   │
│                               │  Loads any checkpoint│                   │
│                               │  Runs full eval suite│                   │
│                               └─────────────────────┘                   │
└──────────────────────────────────────────────────────────────────────────┘
```

### Decoupled Evaluation Path

When `evaluation.decoupled: true` is set in the config, the trainer skips inline evaluation and only saves checkpoints to disk. `evaluate.py` can then be run independently — no Kafka required — to score any checkpoint at any time.

```
trainer.py (decoupled: true)
  │  trains on Kafka stream, skips inline eval
  └──► checkpoints/<model>__<dataset>/run_<ts>_<uid>/
           step_000100/ step_000200/ ... final/
                │
                └──► python evaluate.py --config configs/imdb_quantitative.yaml
                          │  [--step 200 | --all-checkpoints | --list | --checkpoint-dir <path>]
                          └──► output/<project>/eval_results/...
                                    eval_results.json
                                    eval_config.json
                                    evaluation_artifacts/
                                        artifact_<ts>_<uid>/
                                            report.html, dashboards/, plots/
```

### Static Inference Path (No Kafka Required)

When `kafka.enable_lora_streaming: false` or `--checkpoint` is provided to `inference.py`, the Kafka consumer thread is never started. The LoRA adapter is loaded directly from disk via `PeftModel.from_pretrained`.

```
inference.py --config configs/imdb_quantitative.yaml --checkpoint latest
  │
  ├── CheckpointManager.resolve_checkpoint_path("latest")
  │       → output/imdb/checkpoints/distilgpt2__imdb/run_<ts>_<uid>/final/
  │
  ├── PeftModel.from_pretrained(base_model, checkpoint_path)
  │       → loads LoRA adapter from disk, no Kafka
  │
  └── Flask REST API :5000
        POST /generate {"prompt": "..."} → {"generated_text": "..."}
```

### Three Loosely-Coupled Services

| Service | File | Kafka Role | Direction |
|---|---|---|---|
| **Producer** | `producer.py` | Kafka **Producer** | Writes to `training-data` topic |
| **Trainer** | `trainer.py` | Kafka **Consumer** (reads data) + Kafka **Producer** (writes weights, optional) | Reads training data, writes LoRA weights to `lora-updates` topic |
| **Inference Server** | `inference.py` | Kafka **Consumer** (optional) | Reads LoRA weight updates if streaming enabled |
| **Evaluator** | `evaluate.py` | None | Reads checkpoints from disk only |

All three services are stateless with respect to each other: they communicate **only through Kafka topics** (for the streaming path). This means you can restart or scale any one of them independently.

---

## 4. Repository Structure

```text
Infinitune-Realtime-LLM-Fine-Tuning-Framework/
|
|-- producer.py           # Service 1: Streams dataset samples to Kafka
|-- trainer.py            # Service 2: Consumes data, fine-tunes model with QLoRA
|-- inference.py          # Service 3: Serves REST API, hot-swaps LoRA weights
|-- evaluate.py           # Decoupled evaluator: loads any checkpoint, runs full eval
|
|-- configs/
|   |-- imdb_quantitative.yaml    # IMDb sentiment classification (distilgpt2, class_match)
|   |-- imdb_qualitative.yaml     # IMDb domain generation (Qwen2.5-1.5B, keyword_density)
|   |-- gsm8k_quantitative.yaml   # GSM8K math reasoning (Qwen2.5-3B, regex_extract)
|   |-- gsm8k_qualitative.yaml    # GSM8K CoT structure (Qwen2.5-3B, structural_cot)
|   |-- alpaca_qualitative.yaml   # Alpaca instruction following (Qwen2.5-1.5B, semantic_similarity)
|   `-- e2e_qualitative.yaml      # E2E NLG slot-to-text (gpt2-medium, structured_slot_coverage)
|
|-- utils/
|   |-- stream_filter.py          # Data quality filter used by producer.py
|   |-- checkpoint_manager.py     # CheckpointManager: hierarchical LoRA checkpoint save/load
|   |-- eval_metrics_train.py     # Evaluator class: quantitative metrics (perplexity, accuracy, F1, etc.)
|   |-- eval_qualitative.py       # QualitativeEvaluator: slot coverage, CoT, semantic similarity, keyword density
|   |-- evaluation_artifacts.py   # Artifact orchestrator: versioned bundles, manifest, index
|   |-- plot_metrics.py           # Standalone CLI: regenerates evaluation artifact bundle from CSV
|   |-- report_html.py            # Generates offline self-contained Plotly HTML report
|   `-- report_utils.py           # Shared presentation-spec layer and chart utilities
|
|-- docs/
|   |-- Infinitune_Project_Context.md   # This file - full architecture reference
|   |-- README.md                       # Guide index + config chooser
|   |-- infinitune_modernization_changelog.md  # Changelog: old vs new implementation pairs
|   |-- imdb_quantitative_guide.md      # IMDb sentiment classification guide
|   |-- imdb_qualitative_guide.md       # IMDb domain generation guide
|   |-- gsm8k_quantitative_guide.md     # GSM8K quantitative reasoning guide
|   |-- gsm8k_qualitative_guide.md      # GSM8K CoT structure guide
|   |-- alpaca_qualitative_guide.md     # Alpaca instruction-following guide
|   `-- e2e_qualitative_guide.md        # E2E NLG task guide (slot coverage + consistency)
|
|-- output/               # Auto-created runtime directory (git-ignored)
|   |-- <project>/
|   |   |-- checkpoints/<model>__<dataset>/
|   |   |   `-- run_<YYYYMMDD-HHMMSS>_<uid>/    # unique per training run
|   |   |       |-- step_000050/
|   |   |       |   |-- adapter_model.safetensors
|   |   |       |   |-- adapter_config.json
|   |   |       |   |-- checkpoint_meta.json
|   |   |       |   `-- config_snapshot.yaml
|   |   |       |-- step_000100/
|   |   |       `-- final/
|   |   |-- logs/<run_name>/<timestamp>_<uid>/   # unique per training run
|   |   |   |-- metrics.csv                      # all columns (sparse per config)
|   |   |   |-- metrics_clean.csv                # only non-empty columns
|   |   |   |-- run_params.json                  # config snapshot at startup
|   |   |   |-- verbose_samples.md               # per-sample predictions (if verbose=true)
|   |   |   `-- evaluation_artifacts/
|   |   |       |-- index.json
|   |   |       `-- artifact_<timestamp>_<uid>/
|   |   |           |-- manifest.json
|   |   |           |-- generation_log.json
|   |   |           |-- metrics/
|   |   |           |-- dashboards/
|   |   |           |   |-- dashboard_dark.png
|   |   |           |   `-- dashboard_light.png
|   |   |           |-- insights/
|   |   |           |-- plots/
|   |   |           `-- report.html
|   |   `-- eval_results/<model>__<dataset>/<checkpoint>/eval_<timestamp>_<uid>/
|   |       |-- eval_results.json
|   |       |-- eval_config.json
|   |       `-- evaluation_artifacts/
|   |           `-- artifact_<timestamp>_<uid>/
|   |               `-- (same bundle structure as above)
|
|-- requirements.txt
`-- README.md             # Setup and quickstart guide
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
- `r` is the rank (e.g., 8 or 16) — controls the number of trainable parameters
- `alpha` is a scaling factor (e.g., 32) — `alpha/r` acts as an effective learning rate multiplier

**Why this matters for InfiniTune:**
- The adapter tensors are tiny (e.g., 5–20 MB for a GPT-2 class model vs. hundreds of MB for the full model)
- Only adapters are serialized and sent over Kafka — this is fast
- The base model stays frozen and is shared between training and inference
- Memory footprint is dramatically smaller than full fine-tuning

In InfiniTune, LoRA is applied via the `peft` library using `get_peft_model()`. The target modules (attention layers) are configured per-task in the YAML config.

**QLoRA** refers to running LoRA on a quantized (4-bit or 8-bit) base model. Currently the configs use `fp32` or `fp16` precision; `4bit` is a valid option if the hardware supports it.

#### LoRA Target Modules — Evolution

> **Previously (GPT-2 era):** All configs used `target_modules: ["c_attn", "c_proj"]` — the combined Q/K/V projection and output projection layers in GPT-2's MHA implementation. These are GPT-2-specific module names.
>
> **Now (Qwen-native):** Migrating to `Qwen/Qwen2.5-x` models caused PEFT injection failures because Qwen lacks `c_attn` and `c_proj` entirely. All Qwen configs now use `["q_proj", "k_proj", "v_proj", "o_proj"]` (the four standard attention projection layers common to LLaMA/Qwen/Mistral architectures). Some domain adaptation configs additionally include `["gate_proj", "up_proj", "down_proj"]` from the MLP block. **The `e2e_qualitative.yaml` config uses `gpt2-medium` and retains the original `["c_attn", "c_proj"]` targets.**

Architecture-specific reference:
| Model family | Correct `target_modules` |
|---|---|
| GPT-2, DistilGPT-2, GPT-2-medium | `["c_attn", "c_proj"]` |
| Qwen2.5, LLaMA, Mistral, Gemma | `["q_proj", "k_proj", "v_proj", "o_proj"]` |
| Qwen2.5 (domain adaptation) | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` |

### 5.2 Apache Kafka as the Data Backbone

**Kafka** is a distributed event streaming platform. InfiniTune uses it as a message queue / data bus between services.

Key Kafka concepts used:
- **Topic**: A named log of records. InfiniTune uses two per task: `training-data-<task>` and `lora-updates-<task>`.
- **Producer**: Writes records to a topic (`producer.py`, `trainer.py` for weights).
- **Consumer**: Reads records from a topic (`trainer.py` for data, `inference.py` for weights when streaming is enabled).
- **Consumer Group**: A group of consumers sharing work from a topic. InfiniTune uses separate groups for trainer and inference so they each receive all messages independently.
- **Offset**: Position within a topic. On startup, the trainer uses `auto_offset_reset="earliest"` (replay all data) or seeks to the end in `test_mode`.

**Why Kafka instead of a simple queue?**
- Records are persistent — the trainer can be restarted and replay data from the beginning
- Kafka handles backpressure naturally — the producer can be faster or slower than the trainer
- Kafka's durable log is perfect for weight updates — the inference server can catch up if it was briefly unavailable
- Topic-level separation makes it trivial to add new consumers (e.g., a second inference server)

#### Kafka API Version — Critical Note

> **Previously:** Some early versions pinned `api_version=(0, 10)` explicitly in the `KafkaProducer` constructor.
>
> **Now:** This is explicitly forbidden. Pinning `api_version=(0, 10)` causes **silent delivery failures on modern Kafka 3.x+ brokers** — messages appear to send but are never received. The producer always uses auto-negotiation. This is enforced via a code comment directly above the `KafkaProducer` instantiation.

#### Kafka Consumer Long-Eval Reliability

For qualitative evaluation tasks (especially `structured_slot_coverage` with many consistency runs), a single eval call can take 15–25 minutes. Kafka's default consumer session timeout is far shorter, causing the broker to evict the consumer as "dead" and trigger a rebalance that breaks training.

The following settings are applied to `KafkaConsumer` in `trainer.py`:

| Parameter | Default Value | Purpose |
|---|---|---|
| `max_poll_interval_ms` | `1,800,000` (30 min) | Maximum time between `poll()` calls before the broker considers the consumer dead |
| `session_timeout_ms` | `30,000` (30 sec) | Heartbeat timeout; broker waits this long after missing a heartbeat |
| `heartbeat_interval_ms` | `10,000` (10 sec) | How often the consumer sends heartbeats to the coordinator |

All three can be overridden in `kafka:` config block.

### 5.3 Online (Streaming) Learning

Traditional training processes a fixed dataset in multiple epochs. InfiniTune uses **online learning**: the model is updated on each mini-batch as it arrives from the stream, without storing the full dataset.

In `test_mode: true` (used by all current configs), the trainer processes the full streaming dataset **exactly once** — a single pass through the Kafka topic — stopping when it receives the `_eof` sentinel. This is equivalent to a single epoch of streaming training, not multi-epoch replay.

Benefits:
- The model continuously adapts to new data without waiting for a full dataset to accumulate
- Memory is proportional to a single mini-batch + the model, not the entire dataset
- The model improves immediately as data flows in, making it suitable for live or rapidly evolving data sources

### 5.4 Decoupled Evaluation Architecture

**The Problem:** In-process quantitative and qualitative evaluation was originally always inline — it ran inside `trainer.py` at every `eval_interval` optimizer steps. For qualitative tasks (e.g., `structured_slot_coverage` with `consistency_runs: 10`), a single eval pass over 334 samples took **40+ minutes**, freezing the Kafka consumer and blocking training ingestion.

> **Previously (inline-only):** All evaluation ran synchronously inside `trainer.py`. Setting `eval_interval: 200` meant every 200 optimizer steps, training would pause for 15–40 minutes while qualitative metrics were computed. This made real-time training impractical for complex eval configurations.
>
> **Now (decoupled flag):** A `evaluation.decoupled: true` config flag completely bypasses inline evaluation. The trainer focuses on streaming training and checkpoint saving at `save_every_steps`. After training (or even during it), `evaluate.py` can be run independently at any time against any saved checkpoint, producing the full evaluation suite without affecting training throughput.

The `evaluate.py` script is the dedicated offline evaluator:
- No Kafka connection required
- Loads any saved LoRA adapter checkpoint
- Runs the full quantitative + qualitative evaluation suite using the same `Evaluator` and `QualitativeEvaluator` infrastructure as inline eval
- Writes results to a versioned `eval_results/` directory

### 5.5 Hierarchical Checkpoint Management

> **Previously (flat layout):** Checkpoints were saved as `step_*/` and `final/` directories directly under `output/checkpoints/<model>__<dataset>/`. Rapid training restarts would overwrite `final/` from a previous run and step directories could collide between runs.
>
> **Now (hierarchical layout):** Each training run creates a unique `run_<YYYYMMDD-HHMMSS>_<uid>/` subdirectory. All checkpoints for that run live inside it. Multiple training runs stack independently and never overwrite each other. `list_checkpoints()` scans all runs and also understands the legacy flat layout for backward compatibility.

Each checkpoint directory contains:
- `adapter_model.safetensors` — the LoRA adapter weights
- `adapter_config.json` — PEFT adapter configuration (architecture metadata)
- `checkpoint_meta.json` — provenance: step, timestamp, model name, dataset, config path, training loss
- `config_snapshot.yaml` — full YAML config copy, making every checkpoint self-contained for re-evaluation

---

## 6. Component Deep-Dives

### 6.1 `producer.py` — Data Streamer

**Role:** Loads a HuggingFace dataset, applies quality filtering, and publishes each sample as a JSON record to the Kafka `training-data` topic.

#### Startup Sequence
1. Parse `--config` argument and load YAML.
2. Optionally call `clear_kafka_topic()` to remove stale records from previous runs.
3. Initialize `StreamFilter` with filtering rules from config.
4. Create a `KafkaProducer` (JSON-serialized values, SHA-256-hashed keys, **no `api_version` pin**).
5. **Send a verification message** (`{"_verify": True}`) with key `"__verify__"` and block until it is acknowledged — this ensures the broker is reachable before streaming begins.
6. Iterate through all examples from `generate_training_examples()`.

#### `generate_training_examples(config)`

```python
# Pseudocode
load HuggingFace dataset (path, split, optional config_name)
# For 'parquet' datasets: load from data_files URLs (e.g., HuggingFace parquet CDN)
optionally shuffle with a seed (critical for sorted datasets like IMDb)
for each example:
    map input_col → "input", target_col → "target"
    apply label_map if defined (e.g., 0 → "negative", 1 → "positive")
    yield {"input": ..., "target": ...}
```

This normalizes every dataset to the same `{"input": ..., "target": ...}` interface, making the rest of the pipeline dataset-agnostic.

#### Topic Cleanup — `clear_kafka_topic()`

Before streaming, the producer can purge the training topic by deleting and recreating it. This prevents the trainer from replaying stale records from a previous producer run when the topic already contains messages. This is particularly important when running multiple experiments back-to-back using `test_mode: true`.

#### Filtering via `StreamFilter`

Before sending each sample, `stream_filter.validate(raw_record, extracted_text)` is called. If it fails, the sample is dropped and a drop reason is logged. Telemetry (ingested count, drop rate, recent drop reasons) is printed every 1000 records or 60 seconds.

#### Message Key

The Kafka message key is a **SHA-256 hash** of the input text (the configured `hash_column`). This makes keys stable and unique — enabling Kafka log compaction (dedup) if needed. The SHA-256 hash is always 64 hex characters regardless of input length.

#### Control Messages

| Key | Value | Purpose |
|---|---|---|
| `"__verify__"` | `{"_verify": True}` | Sent on startup; confirms broker connectivity before streaming starts |
| `"__eof__"` | `{"_eof": True}` | Sent after the last sample; signals the trainer to stop after draining remaining data |

#### Error Handling
- Delivery errors are tracked. If more than 10 accumulate, the producer aborts.
- Progress is logged every 5 seconds (heartbeat).

---

### 6.2 `trainer.py` — Streaming Trainer

This is the core of InfiniTune. It contains all training logic.

#### Top-Level Components Inside `trainer.py`

| Class / Function | Purpose |
|---|---|
| `MetricsLogger` | Writes per-step CSV metrics and orchestrates artifact generation |
| `LoRAProducer` | Serializes and pushes LoRA adapter weights to Kafka (when streaming enabled) |
| `tokenize_with_label_masking()` | Tokenizes prompt + response, masks prompt tokens in labels |
| `pad_batch()` | Pads a list of tokenized samples into a batch tensor |
| `build_lr_scheduler()` | Builds a `LambdaLR` scheduler from config |
| `train_model(config)` | Main training loop — orchestrates everything |

#### `MetricsLogger`

Writes a CSV file with all columns from `MetricsLogger.COLUMNS`. The full column list (see §10.8 for detailed descriptions):

```
step, loss, lr,
eval_loss, perplexity, accuracy, aauc, average_accuracy,
f1, mcc, kappa, exact_match,
qafacteval, answer_overlap_f1,
forgetting_max, update_latency_s, eval_cycle_time_s, backward_transfer,
grad_norm, tokens_per_sec, step_time_s, records_used_total,
qual_semantic_similarity, qual_keyword_density,
qual_type_token_ratio, qual_hapax_ratio,
qual_cot_anchor_count_mean, qual_cot_step_length_mean, qual_cot_coverage_rate,
qual_mean_response_length, qual_repetition_rate, qual_non_empty_rate,
qual_slot_coverage_mean, qual_consistency_score_mean, qual_perfect_coverage_rate,
qual_slot_familyFriendly_inversion_rate,
qual_pinned_slot_coverage_mean, qual_pinned_perfect_coverage_rate, qual_pinned_consistency_score
```

> **Design decision — CSV opened/closed per write:** The CSV file is opened, written, and closed on every `log()` call (not kept open). This is intentional for Windows compatibility — Windows blocks readers on open file handles.

Also writes `run_params.json` — a snapshot of all config params at startup — for reproducibility.

At the end of training, the evaluation artifact orchestrator reads the CSV and generates a versioned artifact bundle.

#### `LoRAProducer`

- Serializes each LoRA adapter tensor using `torch.save()` into a `BytesIO` buffer and publishes it to the `lora-updates` Kafka topic.
- Message key = layer name (e.g., `"base_model.model.transformer.h.0.attn.c_attn.lora_A.default.weight"`).
- After training completes, sends a `__done__` sentinel with key `"__done__"` so the inference server knows to stop listening.
- **Only active when `kafka.enable_lora_streaming: true`.** When `false`, LoRA weights are still saved to disk checkpoints but never broadcast to Kafka.

#### `tokenize_with_label_masking(tokenizer, prompt_text, response_text, max_seq_length)`

This function is critical to training quality:

```
Input:  prompt_text  = "Review: This movie was great.\nSentiment:"
        response_text = " positive"

Tokenize separately:
  prompt_ids   = tokenizer.encode(prompt_text)        # with special tokens
  response_ids = tokenizer.encode(response_text)      # without special tokens
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

#### `build_lr_scheduler(optimizer, config)`

Builds a `torch.optim.lr_scheduler.LambdaLR` with one of three schedules:

| Type | Behavior |
|---|---|
| `constant` | No scheduling — `lambda = 1.0` throughout |
| `linear` | Linear warmup from 0 → base LR over `warmup_steps`, then linear decay to `min_lr_ratio × base_lr` over `T_max` steps |
| `cosine_with_warmup` | Linear warmup, then cosine decay: `LR = min_lr + (1 - min_lr) × 0.5 × (1 + cos(π × progress))` |

#### `train_model(config)` — The Main Loop

```
1.  Parse config sections (model, lora, training, kafka, preprocessing).
2.  Initialize MetricsLogger and write run_params.json.
3.  Build LoraConfig and wrap the base model with get_peft_model().
4.  Optionally enable gradient_checkpointing (use_reentrant=False).
5.  Detect device (CUDA > MPS > CPU) and move model to device.
6.  Create KafkaConsumer for training-data topic with long-eval timeout settings.
    - In test_mode: seek to END of topic (consume only new data from this run).
    - UUID-suffixed consumer group in test_mode to avoid stale committed offsets.
7.  Set up optimizer (AdamW) and LR scheduler.
8.  Instantiate LoRAProducer (if enable_lora_streaming) and Evaluator.
9.  Instantiate CheckpointManager for periodic checkpoint saving.
10. Enter the streaming training loop:

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
           i.  clip_grad_norm_(max_norm=1.0) — before optimizer.step()
           ii. optimizer.step(), scheduler.step()
           iii.optimizer.zero_grad(set_to_none=True) — aggressive memory release
           iv. del outputs, loss, scaled_loss, batch → gc.collect() → empty_cache()
           v.  Log step metrics (loss, LR, grad_norm, tok/s, step time).
           vi. Every eval_interval steps AND NOT decoupled:
               run Evaluator.evaluate() + QualitativeEvaluator.run()
           vii.Every save_every_steps steps:
               CheckpointManager.save(model, tokenizer, step, loss)
      g. Every weight_push_interval seconds AND enable_lora_streaming:
         - Send LoRA adapter state dict to Kafka via LoRAProducer.

11. On training complete (or Ctrl-C / error):
    a. Send final LoRA weights (if streaming enabled).
    b. Send __done__ sentinel (if streaming enabled).
    c. Save final checkpoint via CheckpointManager.save(model, tokenizer, "final")
    d. Run final evaluation (unless interrupted or decoupled).
    e. Generate evaluation artifact bundle from metrics CSV.
    f. Log: "To run decoupled evaluation: python evaluate.py --config <config> [--step N | --all-checkpoints]"
```

**`test_mode` behavior:**
When `test_mode: true`, the trainer seeks to the **end** of the Kafka topic on startup so it only consumes records produced *after* the current run starts. It trains until the EOF marker arrives, rather than stopping at `max_steps`. `max_steps` serves as a safety cap only.

**Heartbeat logging:** While waiting for the producer to send data, the trainer logs every 5 seconds with idle time and batch progress, so it never looks "stuck".

**Gradient norm tracking:** The raw (pre-clip) L2 norm of all gradients is accumulated across micro-batches and logged at each optimizer step as `grad_norm`. This is useful for diagnosing training instability.

**Token throughput:** Tokens per second is computed as `response_tokens_in_batch / step_wall_time`, giving a hardware-specific throughput metric.

---

### 6.3 `inference.py` — Live Inference Server

**Role:** Loads the base model + LoRA adapter (from disk or Kafka), serves a Flask REST API for generation, and optionally hot-swaps LoRA weights in the background as the trainer publishes updates.

#### Startup Modes

| Mode | Trigger | Behavior |
|---|---|---|
| **Static checkpoint** | `--checkpoint <path\|step\|"latest">` | Loads LoRA adapter via `PeftModel.from_pretrained`. Kafka consumer never started. |
| **Kafka streaming** | No `--checkpoint`, `enable_lora_streaming: true` | Starts Kafka consumer thread + weight application thread. Hot-swaps weights as trainer publishes. |
| **Base model only** | No `--checkpoint`, `enable_lora_streaming: false` | Loads base model with empty LoRA adapter. No Kafka. |

#### Startup Sequence (Streaming Mode)
1. Load config from `--config` argument.
2. Detect device (CUDA > MPS > CPU).
3. Load base model. Uses `device_map={"": device_global}` for **contiguous device mapping** (all layers on the same device — avoids split-layer issues on MPS).
4. If `--checkpoint` provided: `PeftModel.from_pretrained(base_model, resolved_path)` — Kafka threads skipped.
5. Else: Apply initial LoRA config with `get_peft_model()`.
6. Set model to `.eval()` mode.
7. Create a `queue.Queue` (the weight update queue) and a `threading.Lock` (the model lock).
8. Start two background daemon threads (streaming mode only):
   - `kafka_consumer_thread` — reads LoRA weight updates from Kafka
   - `weight_application_thread` — applies updates to the model
9. Start the Flask server (blocking).

> **Previously (`device_map="auto"`):** The inference server used `device_map="auto"` from `accelerate`, which can split model layers across devices for large models. On MPS with multi-GPU setups this caused layer-placement conflicts.
>
> **Now (`device_map={"": device_global}`):** All layers are explicitly assigned to the same device. This is more predictable and avoids cross-device tensor operations.

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
- Uses `auto_offset_reset="latest"` — the inference server only cares about the newest weights, not historical ones.

#### `weight_application_thread(model, update_queue, model_lock, device)`

The key challenge is applying weight updates **without blocking ongoing inference requests** for longer than necessary:

1. **Blocking wait** for the first item in the queue (avoids busy-looping).
2. **Non-blocking drain**: immediately drain all remaining items currently in the queue in a tight non-blocking loop.
3. Only **then** acquire the model lock and call `model.load_state_dict(updates, strict=False)`.

This batching means that if the trainer pushes 10 tensor updates rapidly, they are applied in a single lock acquisition rather than 10 sequential ones — minimizing the time inference is blocked.

`strict=False` in `load_state_dict` means only the adapter keys present in the update dict are updated; unrelated model weights are unchanged.

#### `generate_text(prompt, model, tokenizer, model_lock, device, inference_cfg)`

- Acquires `model_lock` for the entire duration of tokenization + generation.
- Tokenizes the prompt, records `prompt_token_len`.
- Runs `model.generate()` with a `GenerationConfig` object (from config: `max_new_tokens`, `do_sample`, `temperature`, `top_p`).
- **Slices off the prompt tokens** from the output: `generated_ids = outputs[0, prompt_token_len:]`. This ensures only the model's *new* tokens are decoded and returned.

---

### 6.4 `evaluate.py` — Decoupled Offline Evaluator

**Role:** Loads any saved LoRA checkpoint and runs the full evaluation suite (quantitative + qualitative) independently of training and without requiring a Kafka connection.

#### CLI Reference

```bash
# Evaluate the final checkpoint (default if no other flag given)
python evaluate.py --config configs/imdb_quantitative.yaml

# Evaluate the checkpoint saved at optimizer step 500
python evaluate.py --config configs/imdb_quantitative.yaml --step 500

# Evaluate a specific checkpoint directory directly
python evaluate.py --config configs/imdb_quantitative.yaml \
  --checkpoint-dir output/imdb/checkpoints/distilgpt2__imdb/run_20260413-153020_a3f2/step_000500

# Evaluate ALL saved checkpoints sequentially (includes baseline comparison)
# Produces one row per checkpoint → combined CSV + plots showing full training arc
python evaluate.py --config configs/imdb_quantitative.yaml --all-checkpoints

# List available checkpoints and exit (no evaluation)
python evaluate.py --config configs/imdb_quantitative.yaml --list
```

| Flag | Type | Description |
|---|---|---|
| `--config` | str (required) | Path to the YAML config file |
| `--step` | int | Evaluate the checkpoint at this optimizer step (finds latest run containing it) |
| `--checkpoint-dir` | str | Path to a specific checkpoint directory (overrides `--step`) |
| `--all-checkpoints` | flag | Evaluate ALL checkpoints sequentially; adds a `base_model` (no adapter) entry for baseline |
| `--list` | flag | Print available checkpoints and exit without evaluating |

#### `--all-checkpoints` Mode

When `--all-checkpoints` is used, `evaluate.py` automatically prepends a **base model evaluation** (no LoRA adapter loaded) to the checkpoint list. This provides a step-0 baseline, making it easy to visualize the full learning arc from the untrained model to the final checkpoint.

#### Output

Results are written to:
```
output/<project>/eval_results/<model>__<dataset>/<checkpoint_name>/eval_<timestamp>_<uid>/
    eval_results.json          # full metrics dict
    eval_config.json           # config snapshot used for this eval
    evaluation_artifacts/      # same artifact bundle structure as training
        index.json
        artifact_<timestamp>_<uid>/
            manifest.json
            dashboards/
            plots/
            report.html
```

For `--all-checkpoints`, results for all checkpoints are additionally combined into a single `all_checkpoints_results.csv` that can be fed to `utils/plot_metrics.py` to produce a complete learning-curve plot.

#### Execution Flow

```python
# Pseudocode for evaluate.py
load config from --config
init CheckpointManager(config)

if --list:
    print checkpoints; exit

if --all-checkpoints:
    checkpoints = [base_model_entry] + manager.list_checkpoints()
elif --step:
    checkpoints = [manager.resolve_checkpoint_path(step)]
elif --checkpoint-dir:
    checkpoints = [{"path": args.checkpoint_dir, ...}]
else:
    checkpoints = [most_recent_final_checkpoint]

for ckpt in checkpoints:
    if ckpt["path"] == "base_model":
        model = load_base_model_only(config)   # no adapter
    else:
        base = load_base_model(config)
        model = PeftModel.from_pretrained(base, ckpt["path"])

    model.eval()
    evaluator = Evaluator(config, tokenizer, device, ...)
    metrics = evaluator.evaluate(model, step=ckpt["step"])

    if testing_strategy.enabled:
        qual_evaluator = QualitativeEvaluator(config, ...)
        qual_metrics = qual_evaluator.run(model, tokenizer, ...)
        metrics.update(qual_metrics)

    write eval_results.json, generate artifacts
```

---

### 6.5 `utils/stream_filter.py` — Data Quality Gate

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

The `chat_structure` block is fully optional (`null` on all fields disables it).

**Returns:** `(is_valid: bool, reason: str | None)` — the reason string is used for telemetry/debugging but never exposed to the model.

**Safety:** If the `validate()` function crashes for any reason, it returns `(True, None)` — fail open, to avoid silently starving the trainer.

---

### 6.6 `utils/checkpoint_manager.py` — Checkpoint Manager

**Role:** Manages the full lifecycle of LoRA adapter checkpoints — versioned saving, discovery across all past runs, path resolution, and evaluation scheduling.

#### Directory Layout

```
output/<project>/checkpoints/<model_slug>__<dataset_slug>/   ← checkpoint_root
    run_20260413-153020_a3f2/      ← per-run subdir (created at init)
        step_000050/
            adapter_model.safetensors
            adapter_config.json
            checkpoint_meta.json
            config_snapshot.yaml
        step_000100/
        ...
        final/
    run_20260414-090015_b7c1/      ← second run; never collides
        step_000050/
        ...
```

`<model_slug>` and `<dataset_slug>` are created by `_slugify()` — lowercased, non-alphanumeric characters replaced with underscores (e.g., `"Qwen/Qwen2.5-1.5B"` → `"qwen_qwen2_5_1_5b"`).

#### Key Methods

```python
# Pseudocode

CheckpointManager.save(model, tokenizer, step, loss=None):
    # step = int (optimizer step) or "final"
    # "final" always overwrites within the same run
    # Numbered steps never overwrite (a step can only appear once per run)
    dir_name = "step_000200" if step==200 else "final"
    model.save_pretrained(save_path)          # writes adapter_model.safetensors
    tokenizer.save_pretrained(save_path)      # writes tokenizer files
    write checkpoint_meta.json                # step, timestamp, model, dataset, lora config, loss
    write config_snapshot.yaml                # full YAML config for self-contained eval

CheckpointManager.list_checkpoints():
    # Scans ALL run_* subdirs (new layout) AND step_* flat dirs (legacy)
    # Returns list of dicts:
    # { "name": "step_000200", "step": 200, "path": "/abs/path",
    #   "run": "run_20260413-153020_a3f2", "timestamp": "2026-04-13T15:32:00" }
    # Sorted: numbered steps ascending, then "final" last

CheckpointManager.resolve_checkpoint_path(step):
    # step = int, "final", "step_000200", or "latest"
    # For "latest": returns the path of the most-recently-saved checkpoint
    # For an int: scans all runs, returns the latest run containing that step
    # Returns None if not found

CheckpointManager.get_evaluation_schedule(config):
    # Returns a dict indicating which eval types are scheduled
    # { "quantitative_enabled": bool, "qualitative_enabled": bool, ... }

CheckpointManager.select_evaluation_checkpoints(all_ckpts, schedule):
    # Filters the checkpoint list to those that fall on configured cadence
    # Used by evaluate.py --all-checkpoints to determine which to score
```

#### Legacy Flat Layout Support

Pre-versioning checkpoints had `step_*/` and `final/` directories directly under `<model>__<dataset>/` (no `run_*/` wrapper). `list_checkpoints()` detects both layouts during its directory scan and returns them with `"run": null` for legacy entries. This means `evaluate.py` works correctly on checkpoints from older InfiniTune versions.

---

### 6.7 `utils/eval_metrics_train.py` — Quantitative Evaluator

**Role:** The `Evaluator` class runs quantitative evaluation using one of three configurable strategies. It is used both inline by `trainer.py` and standalone by `evaluate.py`.

#### Evaluator Initialization

```python
evaluator = Evaluator(config, tokenizer, device, tokenize_fn, pad_fn)
# config: full YAML dict
# tokenizer: HuggingFace tokenizer
# device: "cuda", "mps", or "cpu"
# tokenize_fn: reference to tokenize_with_label_masking from trainer.py
# pad_fn: reference to pad_batch from trainer.py
```

On initialization, the evaluator:
1. Loads `eval_pool_size` samples from `dataset.eval_split`.
2. Renders each sample through the Jinja2 prompt/response templates.
3. Calls `_refresh_class_match_label_space()` to build the known class-label set for `class_match` strategy.
4. Merges user `evaluation.metrics` flags onto strategy defaults via `_merge_metric_flags()`.
5. Lazily instantiates `_QAFactEvalScorer` (does **not** download the model until the first call to `evaluate()`).

#### Evaluation Loop

```python
# Pseudocode: Evaluator.evaluate(model, step)
model.eval()
advance sliding window by eval_batch_size samples (wraps around eval_pool)

# 1. Perplexity (always computed)
for batch in eval_samples:
    loss = forward_pass(batch)          # labels masked on prompt tokens
eval_loss = mean(all_losses)
perplexity = exp(eval_loss)

# 2. Generation-based metrics (if strategy != "perplexity")
predictions = _generate_batch_records(model, eval_samples, ...)
# Left-padded batches of up to generation_batch_size samples sent to GPU together

for pred, gold in zip(predictions, eval_samples):
    if strategy == "class_match":
        normalized_pred = _normalize_class_match_prediction(pred)
        # Prefix-matches first N tokens against known labels
        # Falls back to other_label ("other") if no known label matched
    elif strategy == "regex_extract":
        pred_match = re.search(answer_regex, pred)
        gold_match = re.search(answer_regex, gold["target"])

# Compute metrics from confusion matrix (pure numpy/math, no sklearn)
accuracy, exact_match, f1, mcc, kappa
# Update AAUC history, compute backward_transfer, forgetting_max
```

#### Strategy-Default Metric Flags (`_default_metric_flags`)

| Strategy | Defaults ON | Defaults OFF |
|---|---|---|
| `class_match` | accuracy, exact_match, f1, mcc, kappa, backward_transfer | qafacteval, answer_overlap_f1, forgetting, eval_cycle_time |
| `regex_extract` | accuracy, exact_match, backward_transfer | f1, mcc, kappa, qafacteval, answer_overlap_f1 |
| `perplexity` | (loss only) | everything else |

User flags in `evaluation.metrics` override these defaults. Unknown user keys are also forwarded (forward-compatibility).

#### Batched Generation with Left-Padding

> **Previously:** Generation was unbatched — one sample at a time passed to `model.generate()`. This severely underutilized GPU parallel capacity.
>
> **Now:** `_generate_batch_records()` groups samples into chunks of up to `generation_batch_size` (default: `min(16, eval_pool_size)`). Before batching, the tokenizer is temporarily switched to **left-padding** — required for decoder-only causal models during batch generation so that all samples are right-aligned in the padded batch. After generation, right-padding is restored.

#### Special Metrics

**AAUC (Average Accuracy Under the Curve):**
Computed via `_normalized_aauc_from_history()`. Uses trapezoidal integration over `(step, accuracy)` pairs divided by the total step span. Reports as both `aauc` (legacy name) and `average_accuracy` (current name, used by plot_metrics.py). A single data point returns that accuracy directly.

**Forgetting-Max (`forgetting_max`):**
Tracks the peak value of each `forgetting_track_metrics` scalar and reports the maximum drop from peak to current value. Meaningful when using a stable eval pool (sliding window can conflate distribution shift with genuine forgetting).

**Backward Transfer (`backward_transfer`):**
For each sample that the model *currently gets wrong*, checks if it was correct at a previous eval step. The BWT score is the mean "did we get this wrong after having been right" across all samples. A negative BWT signals catastrophic forgetting; zero means no regression.

**QAFactEval:**
Factual consistency score between the source input and the generated response. Uses an extractive QA model (`deepset/minilm-uncased-squad2` by default, ~120 MB, **lazy-loaded on CPU**). For each sentence-span in the source, asks the QA model to find that span in the generated text. Scores are averaged across spans using token-level F1. Range: 0 (fully inconsistent) → 1 (fully consistent).

**Answer Overlap F1:**
Token-level F1 between generated text and reference target, computed without requiring the `rouge_score` library. Approximates ROUGE-1.

**Update Latency (`update_latency_s`):**
Wall-clock seconds between consecutive optimizer steps — measures how fast the training loop itself is running.

**Eval Cycle Time (`eval_cycle_time_s`):**
Wall-clock seconds between the end of the previous eval and the start of the current eval. Captures total elapsed time including training steps and any I/O.

---

### 6.8 `utils/eval_qualitative.py` — Qualitative Evaluator

**Role:** `QualitativeEvaluator` runs one of four qualitative proxy strategies to measure model behavior that cannot be captured by exact-match or loss metrics.

The module is zero-dependency on `eval_metrics_train.py`. All strategies output metrics prefixed with `"qual_"` so they integrate cleanly with `MetricsLogger.COLUMNS` and never collide with quantitative names.

#### Universal Metrics (`_compute_universal_qualitative_metrics`)

Run on every strategy at zero additional cost (operating on already-generated text):

| Metric | Description |
|---|---|
| `qual_mean_response_length` | Mean word count across non-empty responses — detects output collapse or verbosity |
| `qual_repetition_rate` | Mean fraction of repeated bigrams per response — detects stuck/repetitive generation |
| `qual_non_empty_rate` | Fraction of predictions that are non-empty strings |

#### Strategy 1: `SemanticSimilarityMetric`

```python
# Pseudocode
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
# Always CPU — never competes with LLM for VRAM
# Lazy-loaded on first compute() call (~90 MB RAM)

pred_embs = model.encode(predictions)
ref_embs  = model.encode(references)
cos_sims  = cosine_similarity(pred_embs, ref_embs, dim=1)
return {"qual_semantic_similarity": mean(cos_sims)}   # range [0, 1]
```

**Use case:** `alpaca_qualitative.yaml` — proves instruction-following responses are semantically converging toward the golden `output` column over training steps.

**Hardware note:** `sentence-transformers` is always loaded with `device="cpu"` regardless of GPU availability. This is intentional — the embedding model (~90 MB RAM) must not compete with the training LLM for MPS/CUDA VRAM.

**Requires:** `pip install sentence-transformers`

#### Strategy 2: `KeywordDensityMetric`

```python
# Pseudocode — reference-free (no golden answers needed)
for pred in predictions:
    words = pred.lower().split()
    keyword_density = count(kw in pred for kw in keywords) / len(keywords)
    ttr = len(set(words)) / len(words)           # type-token ratio
    hapax = count(words appearing exactly once) / len(words)

return {
    "qual_keyword_density":   mean(keyword_density),
    "qual_type_token_ratio":  mean(ttr),
    "qual_hapax_ratio":       mean(hapax),
}
```

**Use case:** `imdb_qualitative.yaml` — proves the model is absorbing movie-review vocabulary and style. A curated list of ~150 film-criticism keywords (cinematography, protagonist, ensemble, narrative, etc.) is defined in the config. A rising `qual_keyword_density` over training steps proves domain vocabulary adoption.

**Hardware note:** Pure string analysis — zero GPU or additional model requirements.

#### Strategy 3: `StructuralCoTMetric`

```python
# Pseudocode
patterns = [re.compile(p, IGNORECASE) for p in logic_anchors]

for pred in predictions:
    positions = all anchor match positions in pred
    anchor_count = len(positions)
    # mean chars between consecutive anchors = "step length"
    step_lengths = [segment length between anchor i and i+1]
    step_length_mean = mean(step_lengths)

return {
    "qual_cot_anchor_count_mean": mean(anchor_counts),
    "qual_cot_step_length_mean":  mean(step_length_means),
    "qual_cot_coverage_rate":     fraction(responses with ≥1 anchor),
}
```

**Use case:** `gsm8k_qualitative.yaml` — proves the model is adopting Chain-of-Thought structure by counting reasoning markers ("First,", "Therefore,", "Step 1:", "Because", etc.) and measuring the length of steps between them. A rising `qual_cot_anchor_count_mean` with a rising `qual_cot_step_length_mean` proves the model generates actual reasoning content, not just inserts anchors as prefixes.

**Hardware note:** Pure regex — zero GPU or additional model requirements.

#### Strategy 4: `StructuredSlotCoverageMetric`

The most sophisticated qualitative strategy. Evaluates whether the model correctly verbalizes all attribute slots from a structured Meaning Representation (MR).

```python
# Pseudocode
def _parse_mr(mr_string) → dict:
    # Extracts slot[value] pairs from MR strings like:
    # "name[The Punter], eatType[coffee shop], priceRange[less than £20]"
    # Returns: {"name": "The Punter", "eatType": "coffee shop", ...}

def _check_slot(slot_name, slot_value, generated_text, checker_type) → bool:
    if checker_type == "boolean_negation":
        # For slots like familyFriendly[yes/no]:
        # "yes" → checks that text contains positive phrasing (kid-friendly, child-friendly, etc.)
        #         AND does NOT contain negative phrasing (not family friendly, etc.)
        # "no"  → reverse: must have negative phrasing, must not have positive
        return _check_boolean_negation(slot_value, generated_text)
    else:
        # Default: simple case-insensitive substring match
        return slot_value.lower() in generated_text.lower()

def _is_valid_restaurant_description(text) → bool:
    # Fluency gate: rejects hallucinated or incoherent generations
    # Checks for minimum length, no excessive repetition, 
    # contains at least some expected restaurant-domain vocabulary
    # Score of 0.0 means the generation is ignored (logged as [HALLUCINATION IGNORED])

# Per-sample scoring
for sample in eval_window:
    slots = _parse_mr(sample["input"])
    coverage_scores = []
    for slot, value in slots.items():
        checker = slot_checkers.get(slot, "default")
        covered = _check_slot(slot, value, generated, checker)
        coverage_scores.append(1.0 if covered else 0.0)
    slot_coverage = mean(coverage_scores)    # 0.0 → 1.0
    perfect_coverage = (slot_coverage == 1.0)
```

**Key sub-methods:**
- `compute_consistency()`: runs the same MR through `consistency_runs` generation passes and measures variance in coverage score
- `compute_pinned()`: evaluates a fixed set of "pinned anchor" MRs at every checkpoint, enabling direct cross-step comparison on identical inputs

**Outputs:**
```
qual_slot_coverage_mean              — mean coverage across eval window
qual_consistency_score_mean          — mean consistency across consistency_runs
qual_perfect_coverage_rate           — fraction of samples where ALL slots are covered
qual_slot_familyFriendly_inversion_rate — fraction where boolean negation is inverted
qual_pinned_slot_coverage_mean       — coverage on pinned anchors
qual_pinned_perfect_coverage_rate    — perfect coverage on pinned anchors
qual_pinned_consistency_score        — consistency on pinned anchors
```

**Per-slot tracking:** If `e2e_nlg_options.track_per_slot` is configured, a separate `qual_slot_<slot_name>_coverage` metric is produced for each listed slot, enabling per-attribute learning curves.

**GPU batching:** Uses `num_return_sequences=consistency_runs` inside `model.generate()` — all consistency runs execute in parallel as a single GPU operation rather than as a Python loop (see §12.5).

---

### 6.9 `utils/evaluation_artifacts.py` — Artifact Orchestrator

**Role:** Centralizes the creation of versioned evaluation output bundles. Called at the end of training and by `evaluate.py`.

#### Artifact Bundle Structure

```
evaluation_artifacts/
    index.json                          # list of all artifact bundles for this run
    artifact_<timestamp>_<uid>/
        manifest.json                   # bundle metadata: timestamp, config, metric summary
        generation_log.json             # per-sample generation records (if verbose)
        metrics/
            resolved_metrics.csv        # clean copy of metrics CSV with sparse columns dropped
            <source_metrics_csv>        # original metrics.csv copy
        dashboards/
            dashboard_dark.png
            dashboard_light.png
            dashboard_dark.svg          # (if generated — additive to PNG)
            dashboard_light.svg
        insights/<theme>/...            # grouped insight charts
        plots/<theme>/...               # per-metric PNG charts
        report.html                     # offline self-contained Plotly HTML dashboard
```

`evaluation_artifacts.py` orchestrates:
1. Creating the versioned `artifact_<timestamp>_<uid>/` directory.
2. Writing `manifest.json` and `generation_log.json`.
3. Copying and resolving the metrics CSV.
4. Calling `plot_metrics.py` routines to generate PNGs/SVGs.
5. Calling `report_html.py` to generate the self-contained `report.html`.
6. Updating `index.json` with the new bundle entry.

Each `evaluate.py` invocation and each training run creates a **new** bundle directory. Existing bundles are never overwritten.

---

### 6.10 `utils/plot_metrics.py` — Offline Plot & Report Utility

A standalone CLI script for regenerating the full evaluation artifact bundle from a `metrics.csv` or `metrics_clean.csv` file.

```bash
# Regenerate from latest training run
python utils/plot_metrics.py "output/imdb/logs/infinitune-imdb-sentiment/20260315-120000/metrics.csv" \
  --config configs/imdb_quantitative.yaml

# Save the bundle under a custom root directory
python utils/plot_metrics.py metrics.csv \
  --config configs/imdb_quantitative.yaml \
  --out-dir ./my_analysis_plots
```

Each run creates a fresh `evaluation_artifacts/artifact_<timestamp>_<uid>/` bundle and does not overwrite prior dashboards or reports.

#### What Is Generated

| Output | Description |
|---|---|
| `plots/<theme>/...` | Per-metric line plots (training loss, accuracy, perplexity, etc.) |
| `insights/<theme>/...` | Grouped insight charts (e.g., combined learning curve, forgetting diagnostics) |
| `dashboards/dashboard_dark.png` | Dark-theme summary dashboard with KPI cards and hero chart |
| `dashboards/dashboard_light.png` | Light-theme variant |
| `report.html` | Offline self-contained Plotly HTML dashboard — the primary analysis surface |
| `manifest.json` | Bundle metadata and chart index |
| `generation_log.json` | Per-sample generation records |

#### HTML Report Design (`report_html.py` + `report_utils.py`)

> **Previously:** `report.html` embedded the PNG dashboard image and then repeated the same KPIs/charts below it. KPI text overlapped due to fixed geometry. Charts were a generic gallery regardless of eval task.
>
> **Now:** `report.html` is an **independently optimized offline Plotly dashboard** that does not embed the PNG dashboard at all. It uses native interactive Plotly charts (self-contained, no CDN), a shared `PresentationSpec` layer that both the HTML and PNG renderers consume (eliminating filename-based gallery assembly), usecase-aware hero/support chart layouts, KPI cards with explicit comparison baseline and delta labels, and collapsed detail accordions. The PNG/SVG dashboards are rebuilt as true card-based layouts with bounded typography.

Key design properties:
- **Offline self-contained**: all Plotly JS embedded inline — works without internet access
- **Usecase-aware layouts**: classification gets accuracy/MCC/forgetting hero; structured NLG gets slot-coverage arc; CoT gets anchor-count trajectory
- **KPI semantics**: every card shows metric name + current/peak value + delta + baseline text + short supporting caption
- **Dark and light themes**: separate CSS surfaces for both; light mode rebalanced for higher contrast

Useful after a crash (metrics CSV is written incrementally), after a Ctrl-C interrupt, or for sharing results without re-running training.

---

## 7. Configuration System

All four services share a single YAML config file. Every config section maps directly to one part of the system.

### 7.1 Config Naming Convention

> **Previously:** Config files had ad-hoc names like `imdb_config.yaml` and `gsm8k_config.yaml`. When a dataset needed both a classification config and a generation config, the naming became unclear.
>
> **Now:** All configs follow the strict `[dataset]_[eval-mode].yaml` convention. "quantitative" configs use exact-match or classification metrics; "qualitative" configs use generation-based proxy metrics. A dataset can have both.

| Config | Dataset | Eval Mode | Task |
|---|---|---|---|
| `imdb_quantitative.yaml` | IMDb | quantitative | Sentiment classification |
| `imdb_qualitative.yaml` | IMDb | qualitative | Domain vocabulary adoption |
| `gsm8k_quantitative.yaml` | GSM8K | quantitative | Exact-answer math reasoning |
| `gsm8k_qualitative.yaml` | GSM8K | qualitative | CoT structure quality |
| `alpaca_qualitative.yaml` | Alpaca | qualitative | Instruction-following convergence |
| `e2e_qualitative.yaml` | E2E NLG | qualitative | Structured slot verbalization |

**Note:** A "quantitative" config can still have `testing_strategy.enabled: true` to layer qualitative metrics on top. The naming refers to the *primary* eval signal.

### 7.2 Schema Reference

```yaml
# ── Project ──────────────────────────────────────────────
project:
  name: "my-run"          # Used as folder name for output logs
  output_dir: "./output"  # Root output directory

# ── Model ────────────────────────────────────────────────
model:
  name: "distilgpt2"      # HuggingFace model ID
                          # e.g., "Qwen/Qwen2.5-1.5B", "gpt2-medium"
  task_type: "CAUSAL_LM"  # CAUSAL_LM | SEQ_2_SEQ_LM | TOKEN_CLS | SEQ_CLS
  precision: "fp32"       # fp32 | fp16 | bf16 | 4bit
                          # fp32 is the stable default (prevents NaN on MPS)
  max_seq_length: 512     # Max tokens per sample (prompt + response)

# ── LoRA Adapter ─────────────────────────────────────────
lora:
  r: 8                    # Adapter rank. Higher = more capacity. Typical: 4–64.
  alpha: 32               # Scaling factor. Effective multiplier = alpha / r.
  dropout: 0.05           # Dropout on adapter layers during training
  bias: "none"            # "none" | "all" | "lora_only"
  target_modules:         # Which weight matrices to inject adapters into.
    - "q_proj"            # Qwen/LLaMA/Mistral: use q/k/v/o projections
    - "k_proj"
    - "v_proj"
    - "o_proj"
    # GPT-2 / DistilGPT-2 / gpt2-medium: use ["c_attn", "c_proj"] instead

# ── Dataset ──────────────────────────────────────────────
dataset:
  name: "imdb"            # HuggingFace dataset path/name, or "parquet"
  data_files:             # Only used when name == "parquet"
    train: "https://huggingface.co/.../train/0000.parquet"
    validation: "https://huggingface.co/.../validation/0000.parquet"
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
  prompt_template: "Review: {{ input }}\nSentiment:"   # Jinja2 template
  response_template: " {{ target }}"                   # Jinja2 template
  hash_column: "input"   # Which column to SHA-256-hash as the Kafka message key

# ── Data Filtering ───────────────────────────────────────
data:
  filtering:
    universal:
      min_chars: 15                     # Drop samples shorter than this
      max_chars: 4000                   # Drop samples longer than this
      min_alphanumeric_ratio: 0.5       # Drop if <50% of chars are alphanumeric
      max_repetition_ratio: 0.2         # Drop if zlib_ratio <= 0.2 (repetitive)
    domain_specific:
      require_numeric_content: null     # true = drop if no digits present
      custom_regex_must_match: null     # List of regex patterns that must match
      custom_regex_must_not_match: null # List of patterns that must NOT match
      chat_structure:
        min_turns: null                 # Minimum dialogue turns
        require_assistant_final: null   # Last message must be from assistant

# ── Kafka ────────────────────────────────────────────────
kafka:
  bootstrap_servers:
    - "localhost:9092"
  training_topic: "training-data-imdb"
  lora_updates_topic: "lora-updates-imdb"
  enable_lora_streaming: false  # false = trainer saves checkpoints but does NOT
                                # broadcast weights to inference.py via Kafka.
                                # Set true to enable live hot-swap.
  producer_send_interval: 0.1  # Seconds between samples (throttle speed)
  consumer_group_trainer: "trainer-group"
  consumer_group_inference: "inference-api-group"
  poll_timeout_ms: 1000
  consumer_timeout_ms: 1000
  max_poll_interval_ms: 1800000  # 30 min — prevents eviction during long eval
  session_timeout_ms: 30000      # 30 sec heartbeat timeout
  heartbeat_interval_ms: 10000   # 10 sec heartbeat send frequency

# ── Training ─────────────────────────────────────────────
training:
  test_mode: true               # true = seek to end, train full dataset, stop on EOF
  batch_size: 8
  gradient_accumulation_steps: 4  # Effective batch = batch_size × grad_accum
  gradient_checkpointing: false  # true = save VRAM at cost of ~20% more compute
  learning_rate: 1e-4
  max_steps: 2000               # Safety cap (ignored in test_mode until EOF)
  logging_steps: 1
  weight_push_interval: 60      # Push LoRA weights to Kafka every N seconds
  lr_scheduler:
    type: "cosine_with_warmup"  # constant | linear | cosine_with_warmup
    warmup_steps: 50
    min_lr_ratio: 0.01          # Floor LR = base_lr × min_lr_ratio
    T_max: 1000                 # Scheduler period in optimizer steps
  save_checkpoints:
    enabled: true
    save_every_steps: 100       # Save adapter to disk every N optimizer steps
    save_final: true            # Always save a 'final' checkpoint at end

# ── Inference ────────────────────────────────────────────
inference:
  host: "localhost"
  port: 5000
  max_new_tokens: 6
  do_sample: false              # false = greedy; true = sampling
  temperature: 0.7
  top_p: 0.9
  enable_lora_streaming: true   # (read at inference.py runtime; mirrors kafka block)

# ── Evaluation ───────────────────────────────────────────
evaluation:
  enabled: true
  decoupled: false              # true = skip inline eval, save checkpoints only
                                # Run evaluate.py separately for scoring
  strategy: "class_match"       # perplexity | class_match | regex_extract
  eval_interval: 50             # Evaluate every N optimizer steps
  eval_pool_size: 5000          # Number of eval samples to load at startup
  eval_batch_size: 100          # Samples per eval call (sliding window)
                                # OR "full_pool" / "full" / "all" / "entire_pool"
                                # to evaluate the entire pool at once
  verbose: false                # Print per-sample predictions if true
  answer_regex: null            # Regex to extract answer (regex_extract only)
  other_label: "other"          # Fallback bucket for class_match generations
                                # that don't start with a known label
  plotting:
    rolling_average_enabled: true
    rolling_average_window: 11
    rolling_average_include:
      - "accuracy"
      - "f1"
      - "mcc"
  metrics:
    compute_loss: true
    compute_accuracy: true
    compute_exact_match: true
    compute_f1: true
    compute_mcc: true
    compute_kappa: true
    compute_answer_overlap_f1: false  # Token-level F1 vs reference target
    max_distinct_labels_for_structure_metrics: 32  # Skip F1/MCC/kappa if > N labels
    compute_qafacteval: false         # Factual consistency (lazy ~120 MB download)
    qafacteval_model: "deepset/minilm-uncased-squad2"
    compute_forgetting: true          # Peak-vs-current drop tracking
    forgetting_track_metrics:         # Which metrics to track for forgetting
      - "accuracy"
      - "f1"
      - "eval_loss"
    compute_eval_cycle_time: true     # Wall-clock seconds between eval calls
    compute_update_latency: true      # Wall-clock seconds per optimizer step
    compute_aauc: true                # Area under accuracy curve (normalized)
    compute_backward_transfer: true   # Fraction of samples correct before, wrong now

# ── Qualitative Evaluation ───────────────────────────────
testing_strategy:
  enabled: true
  method: "semantic_similarity"  # semantic_similarity | keyword_density
                                 # structural_cot | structured_slot_coverage
  eval_interval: 50
  eval_samples: 20               # Samples per eval window
  eval_pool_size: 100            # Pool to rotate over
  max_new_tokens: 150
  consistency_runs: 3            # Number of stochastic generation passes per sample
                                 # Uses num_return_sequences for GPU parallelism
  consistency_temperature: 0.7
  eval_batch_size: 32            # GPU batch size for generation

  # Method-specific fields (null = unused for this method)
  sentence_model: "sentence-transformers/all-MiniLM-L6-v2"
  keywords: []                   # Domain keyword list (keyword_density only)
  logic_anchors: []              # Regex patterns for CoT anchors (structural_cot only)
  slot_keywords: null            # Overridden automatically from MR (structured_slot_coverage)

  # E2E NLG specific — only read when method == "structured_slot_coverage"
  e2e_nlg_options:
    track_per_slot:              # Produce per-slot qual_slot_<name>_coverage metric
      - "name"
      - "eatType"
      - "familyFriendly"
    slot_checkers:               # Override default substring matching for specific slots
      familyFriendly: "boolean_negation"
    pinned_anchors:              # Fixed MRs evaluated at every checkpoint
      - "name[The Punter], eatType[coffee shop], priceRange[less than £20]"
```

---

### 7.3 IMDb Quantitative Config Walkthrough

**File:** `configs/imdb_quantitative.yaml`
**Task:** Sentiment classification — given a movie review, output "positive" or "negative".
**Dataset:** [IMDb](https://huggingface.co/datasets/imdb) — 25,000 training samples, 25,000 test samples.
**Model:** `distilgpt2` (82M parameters) — fast testing baseline.

**Key config choices:**

- `model.max_seq_length: 512` — IMDb reviews average ~300 words (~400 tokens); 256 was truncating many reviews mid-sentence.
- `model.precision: "fp32"` — prevents NaN loss on Apple Silicon MPS under small-batch variance.
- `lora.r: 16, alpha: 32` — ratio of 2 for better capacity and stable gradient scaling.
- `lora.target_modules: ["c_attn", "c_proj"]` — GPT-2 native targets (distilgpt2 is GPT-2 architecture).
- `label_map: {0: "negative", 1: "positive"}` — converts integer labels to English words.
- `evaluation.strategy: "class_match"` — compare prefix of generated output to target label.
- `evaluation.decoupled: true` — training focuses purely on throughput; run `evaluate.py` separately.
- `evaluation.eval_batch_size: "full_pool"` — evaluates all 4,000 pool samples at each eval call, removing window-composition noise from sliding windows.
- `evaluation.other_label: "other"` — captures off-label generations without breaking confusion matrix computation.
- `evaluation.compute_forgetting: true` — tracks peak-vs-current accuracy drop (catastrophic forgetting signal).
- `training.gradient_checkpointing: false` — disabled to maximize throughput; distilgpt2 is small enough that VRAM is not a constraint.
- `training.max_steps: 1563` — exactly `25000 / (4 * 4) = 1562.5` ≈ one full epoch with batch 4 and grad_accum 4.
- `inference.max_new_tokens: 6` — classification only needs a few tokens.
- `inference.do_sample: false` — greedy decoding for deterministic classification.

**Run recipe:**
```bash
python producer.py --config configs/imdb_quantitative.yaml
python trainer.py  --config configs/imdb_quantitative.yaml
python evaluate.py --config configs/imdb_quantitative.yaml --all-checkpoints
```

**Expected learning curve:** Accuracy climbs from ~50% (random) toward ~85–90% over 1,563 steps. MCC rises from ~0 toward ~0.70–0.80. Loss decays smoothly with cosine schedule.

---

### 7.4 IMDb Qualitative Config Walkthrough

**File:** `configs/imdb_qualitative.yaml`
**Task:** Unconditional movie review generation — learning to *write* movie reviews, not classify them.
**Dataset:** [IMDb](https://huggingface.co/datasets/imdb) — same dataset, different task framing.
**Model:** `Qwen/Qwen2.5-1.5B` (1.5B parameters).

**Key config choices:**

- `dataset.column_mapping: {input_col: "text", target_col: "text"}` — **self-supervised**: both input and target are the same review text. The model is trained to continue writing a review given a prompt. This is the unconditional LM technique — no class label is involved.
- `preprocessing.prompt_template: "Write a detailed movie review:\n\n"` — open-ended generation prompt.
- `preprocessing.response_template: " {{ input }}"` — the actual review text is the response.
- `model.max_seq_length: 768` — reviews average ~400 tokens; 512 was truncating many mid-sentence.
- `model.precision: "fp16"` — Qwen2.5-1.5B on Apple Silicon uses fp16 safely; the model is large enough to benefit from memory savings.
- `lora.target_modules` includes `gate_proj, up_proj, down_proj` in addition to attention projections — broader adapter coverage for domain style learning.
- `evaluation.strategy: "perplexity"` — no class to match; perplexity is the only quantitative signal.
- `evaluation.decoupled: true` — perplexity-only inline would be fast, but decoupled is set for consistency.
- `testing_strategy.method: "keyword_density"` — 150+ film-criticism keywords curated from IMDb/Rotten Tomatoes style vocabulary. A rising `qual_keyword_density` proves domain style absorption.
- `training.learning_rate: 2e-5` — 1.5B models require lower LR than 82M models; `1e-4` caused oscillating loss.
- `training.batch_size: 4` — smaller than IMDb quantitative because full reviews are longer sequences.

**Run recipe:**
```bash
python producer.py --config configs/imdb_qualitative.yaml
python trainer.py  --config configs/imdb_qualitative.yaml
python evaluate.py --config configs/imdb_qualitative.yaml --all-checkpoints
```

**Expected learning curve:** Perplexity decays steadily. `qual_keyword_density` rises over training steps as the model absorbs movie-review vocabulary. `qual_type_token_ratio` and `qual_hapax_ratio` also trend upward, indicating increasing lexical diversity.

---

### 7.5 GSM8K Quantitative Config Walkthrough

**File:** `configs/gsm8k_quantitative.yaml`
**Task:** Math reasoning — given a grade-school math word problem, generate a step-by-step solution ending with `#### <answer>`.
**Dataset:** [GSM8K](https://huggingface.co/datasets/gsm8k) (config: `"main"`) — 7,473 train / 1,319 test.
**Model:** `Qwen/Qwen2.5-3B` (3B parameters) — selected because `distilgpt2` cannot perform basic arithmetic reasoning.

**Key config choices:**

- `dataset.config_name: "main"` — GSM8K requires explicit config name; omitting this causes a dataset loading error.
- `dataset.column_mapping: {input_col: "question", target_col: "answer"}`.
- `model.max_seq_length: 512` — math solutions with CoT reasoning can be 200–300 tokens.
- `model.precision: "fp32"` — arithmetic stability on MPS.
- `lora.target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]` — Qwen native.
- `evaluation.strategy: "regex_extract"` — extract the final numeric answer using `#### (\d+)`.
- `evaluation.answer_regex: "#### (\\d+)"` — applied to both gold and predicted output.
- `evaluation.compute_f1: false` — with thousands of distinct numeric answers, macro F1 is meaningless.
- `evaluation.compute_forgetting: false` — high-cardinality extracted answers make peak tracking uninterpretable.
- `evaluation.decoupled: false` — regex_extract is fast; inline eval is fine.
- `inference.max_new_tokens: 150` — full step-by-step solutions need 100–150 tokens.
- `inference.do_sample: true` — sampling enables exploration of different reasoning paths.
- `training.batch_size: 2` — smaller batch because math sequences are longer.

**Run recipe:**
```bash
python producer.py --config configs/gsm8k_quantitative.yaml
python trainer.py  --config configs/gsm8k_quantitative.yaml
python evaluate.py --config configs/gsm8k_quantitative.yaml --all-checkpoints
```

**Expected learning curve:** Accuracy (exact-match on extracted answer) climbs from near-0% at step 0 toward 20–40% by step 500 (~26% of dataset). Perplexity decays steadily. The model learns to produce `####` markers and correctly compute simple arithmetic.

---

### 7.6 GSM8K Qualitative Config Walkthrough

**File:** `configs/gsm8k_qualitative.yaml`
**Task:** Proves the model is adopting Chain-of-Thought *reasoning structure*, not just guessing numeric answers.
**Dataset:** [GSM8K](https://huggingface.co/datasets/gsm8k) — same dataset as quantitative.
**Model:** `Qwen/Qwen2.5-3B`.

**Key config choices:**

- `evaluation.strategy: "regex_extract"` — both quantitative and qualitative eval are enabled simultaneously; the quantitative side still checks if the final answer is correct.
- `testing_strategy.method: "structural_cot"` — the qualitative side checks for reasoning markers.
- `testing_strategy.logic_anchors` — 30+ regex patterns compiled at init with `IGNORECASE`. Includes sequential markers ("First,", "Second,"), logical connectives ("Therefore,", "Thus,", "Because"), math-specific patterns ("Let us", "Since \w+", "Adding these"), and the `####` final-answer marker. A rising `qual_cot_anchor_count_mean` proves structural CoT adoption; rising `qual_cot_step_length_mean` proves the model generates actual reasoning content between anchors.
- `preprocessing.prompt_template: "Question: {{ input }}\nAnswer: Let's solve this step by step."` — the "Let's solve this step by step" suffix biases the model toward structured generation.
- `testing_strategy.eval_interval: 50` — aligned with `evaluation.eval_interval: 50`.
- `evaluation.decoupled: false` — structural_cot is pure regex (no GPU generation needed for the qualitative pass), so inline eval is fast.

**Relationship to gsm8k_quantitative.yaml:**
The quantitative config checks *if* the answer is correct; the qualitative config checks *how* the model reasons. A model that both gets correct answers AND shows rising CoT anchor counts is genuinely learning to reason, not just pattern-matching final digits.

**Run recipe:**
```bash
python producer.py --config configs/gsm8k_qualitative.yaml
python trainer.py  --config configs/gsm8k_qualitative.yaml
```

**Expected learning curve:** `qual_cot_anchor_count_mean` rises from ~1–2 per response to 4–8. `qual_cot_coverage_rate` rises toward 90%+. `qual_cot_step_length_mean` trends upward as reasoning steps grow longer with more content.

---

### 7.7 Alpaca Qualitative Config Walkthrough

**File:** `configs/alpaca_qualitative.yaml`
**Task:** Instruction following — proves responses converge semantically toward golden outputs.
**Dataset:** [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) — 52,000 instruction-following examples.
**Model:** `Qwen/Qwen2.5-1.5B`.

**Key config choices:**

- **Why Alpaca instead of UltraChat or similar conversational datasets:** UltraChat's target column (`messages`) is a list of dicts. The producer would stringify it via `str()` into garbage like `"[{'role': 'user', 'content': '...'}]"`. Alpaca's columns (`instruction`, `output`) are clean strings that flow through the data-agnostic producer pipeline without transformation.
- `dataset.column_mapping: {input_col: "instruction", target_col: "output"}` — ~40% of examples have a non-empty `input` field providing additional context; the prompt template handles this gracefully.
- `dataset.eval_split: "train"` — Alpaca has no test split; eval uses a shuffled subset of the train split.
- `preprocessing.prompt_template: "### Instruction:\n{{ input }}\n\n### Response:"` — standard Alpaca instruction format.
- `testing_strategy.method: "semantic_similarity"` — uses cosine similarity between generated and golden responses via `sentence-transformers/all-MiniLM-L6-v2` on CPU.
- `testing_strategy.sentence_model: "sentence-transformers/all-MiniLM-L6-v2"` — ~90 MB RAM, CPU-only.
- `evaluation.strategy: "perplexity"` — exact-match is meaningless for open-ended instruction following.
- `evaluation.decoupled: false` — perplexity eval is fast enough for inline use.
- `training.learning_rate: 2e-4` — standard for Qwen2.5-1.5B on instruction following.

**Run recipe:**
```bash
python producer.py --config configs/alpaca_qualitative.yaml
python trainer.py  --config configs/alpaca_qualitative.yaml
python evaluate.py --config configs/alpaca_qualitative.yaml --all-checkpoints
```

**Expected learning curve:** `qual_semantic_similarity` rises from ~0.3–0.4 (baseline MiniLM similarity on random outputs) toward ~0.55–0.70. Perplexity decays from ~300+ toward ~150–200. `qual_mean_response_length` stabilizes as the model learns response length norms.

---

### 7.8 E2E NLG Config Walkthrough

**File:** `configs/e2e_qualitative.yaml`
**Task:** Structured data-to-text NLG — convert a meaning representation (MR) with restaurant slot-value pairs into a fluent English sentence.
**Dataset:** [GEM/e2e_nlg](https://huggingface.co/datasets/GEM/e2e_nlg) — 42,061 train / 4,672 validation records. Loaded as **parquet** from HuggingFace CDN URLs.
**Model:** `gpt2-medium` (345M parameters) — GPT-2 architecture, retained for E2E because the task is short (MR inputs + descriptions fit in 128 tokens).

**Key config choices:**

- `dataset.name: "parquet"` — the GEM/e2e_nlg dataset is loaded via direct parquet file URLs rather than the standard HuggingFace `load_dataset("GEM/e2e_nlg")` path.
- `dataset.column_mapping: {input_col: "meaning_representation", target_col: "target"}`.
- `lora.target_modules: ["c_attn", "c_proj"]` — gpt2-medium is GPT-2 architecture; Qwen targets would fail.
- `model.max_seq_length: 128` — MR strings + descriptions are very short; 128 is sufficient.
- `model.precision: "fp32"` — arithmetic stability.
- `training.gradient_checkpointing: true` — enabled because gpt2-medium with full validation pool (4,672 samples) puts pressure on VRAM during eval.
- `training.max_steps: 2629` — `42,061 / (8 × 2) ≈ 2,629` steps for one full epoch.
- `evaluation.strategy: "perplexity"` — primary quantitative signal; no class to match.
- `evaluation.compute_answer_overlap_f1: true` — token-level F1 between generated and reference, approximating ROUGE-1.
- `evaluation.decoupled: true` — qualitative eval with `consistency_runs: 3` would slow training.
- `evaluation.eval_batch_size: 334` — `334 × 14 evals ≈ 4,676` ≥ 4,672 pool size, ensuring full coverage.
- `testing_strategy.method: "structured_slot_coverage"`.
- `testing_strategy.consistency_runs: 3` — reduced from the original 10 for faster inline iterations (3 is sufficient to detect instability).
- `testing_strategy.eval_batch_size: 32` — GPU batch size for generation.
- `e2e_nlg_options.track_per_slot: [name, eatType, food, priceRange, customer rating, area, familyFriendly, near]` — produces 8 separate per-slot coverage metrics.
- `e2e_nlg_options.slot_checkers.familyFriendly: "boolean_negation"` — the `familyFriendly` slot requires negation-aware checking: `yes` → model must say "child-friendly", `no` → model must say "not suitable for children" (not just "friendly").
- `e2e_nlg_options.pinned_anchors` — 10 fixed MRs spanning 2-slot (easy) to 7-slot (hard) complexity, including both `familyFriendly[yes]` and `familyFriendly[no]` variants. These are evaluated at every checkpoint, enabling direct cross-step comparison.

**Kafka Reliability (important for E2E):**

`max_poll_interval_ms: 1,800,000` is essential. With `consistency_runs: 3` across 150 eval samples, a single qualitative eval pass takes 2–5 minutes. Without this setting, Kafka would evict the consumer during eval, terminating the training run.

**Run recipe:**
```bash
python producer.py --config configs/e2e_qualitative.yaml
python trainer.py  --config configs/e2e_qualitative.yaml
python evaluate.py --config configs/e2e_qualitative.yaml --all-checkpoints
```

**Expected learning curve:** `qual_slot_coverage_mean` rises from ~0.30 (base model zero-shot) toward ~0.75–0.85. `qual_perfect_coverage_rate` (all slots covered in one generation) rises toward 30–50%. `qual_consistency_score_mean` rises, indicating stable multi-run outputs. `qual_slot_familyFriendly_inversion_rate` falls toward 0 as the model learns negation-aware verbalization.

> See `docs/e2e_qualitative_guide.md` for the full operational guide including startup instructions, expected metric ranges, and troubleshooting.

---

## 8. Data Flow — Step-by-Step

### 8.1 Standard Streaming Flow (IMDb Example)

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
  key = SHA-256("An absolutely incredible film...")  → "3f4a9bc..." (64 hex chars)
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
  Batch assembled from 4 samples. Padded to max_len.
  outputs = model(input_ids, attention_mask, labels)
  loss = outputs.loss                              # only response tokens contribute
  scaled_loss = loss / 4  (gradient_accumulation_steps)
  scaled_loss.backward()

STEP 7 — trainer.py: optimizer step (every 4 micro-batches)
  clip_grad_norm_(parameters, max_norm=1.0)       # before optimizer.step()
  optimizer.step() → updates LoRA adapter weights
  optimizer.zero_grad(set_to_none=True)            # aggressive memory release
  del outputs, loss, scaled_loss, batch            # explicit graph termination
  gc.collect(); torch.mps.empty_cache()            # OS-level memory redistribution

STEP 8 — trainer.py: CheckpointManager.save() (every 50 steps)
  model.save_pretrained(step_000050/)
  write checkpoint_meta.json + config_snapshot.yaml

STEP 9 (optional, if enable_lora_streaming=true):
  trainer.py: LoRAProducer.send_weights() (every 60 seconds)
  adapter_state_dict = get_peft_model_state_dict(model)
  For each layer tensor:
    Serialize to BytesIO via torch.save()
    Publish to Kafka topic "lora-updates-imdb"

STEP 10 — inference.py: kafka_consumer_thread (streaming mode)
  Receives (layer_name, tensor) from "lora-updates-imdb"
  Puts into update_queue

STEP 11 — inference.py: weight_application_thread
  Drains update_queue, acquires model_lock
  model.load_state_dict({layer_name: tensor, ...}, strict=False)
  Live model is now updated — no restart

STEP 12 — inference.py: Flask API
  POST /generate {"prompt": "Review: This movie was terrible.\nSentiment:"}
  generate_text() → acquires model_lock → model.generate()
  Returns: {"generated_text": "negative"}
```

### 8.2 Decoupled Evaluation Path

```
[Decoupled Eval — after training completes]

STEP 1 — trainer.py has finished:
  output/imdb/checkpoints/distilgpt2__imdb/run_20260413-153020_a3f2/
    step_000050/ step_000100/ ... step_001550/ final/

STEP 2 — User runs:
  python evaluate.py --config configs/imdb_quantitative.yaml --all-checkpoints

STEP 3 — evaluate.py: CheckpointManager.list_checkpoints()
  Discovers all checkpoints across all runs
  Prepends base_model (no adapter) for baseline comparison

STEP 4 — For each checkpoint:
  Load base model (from HuggingFace cache)
  PeftModel.from_pretrained(base_model, checkpoint_path)  # no Kafka
  model.eval()

STEP 5 — Evaluator.evaluate(model, step) + QualitativeEvaluator.run()
  Same code path as inline eval
  Returns metrics dict

STEP 6 — Write results:
  output/imdb/eval_results/distilgpt2__imdb/final/eval_20260414-090015_b7c1/
    eval_results.json
    eval_config.json
    evaluation_artifacts/artifact_.../report.html

STEP 7 — evaluation_artifacts.py builds versioned artifact bundle
  Plots learning curves from all checkpoints
  Generates dashboard_dark.png, dashboard_light.png, report.html
```

### 8.3 Static Inference Path

```
[Static inference — no Kafka required]

STEP 1 — User runs:
  python inference.py --config configs/imdb_quantitative.yaml --checkpoint latest

STEP 2 — inference.py: resolve checkpoint
  CheckpointManager.resolve_checkpoint_path("latest")
  → output/imdb/checkpoints/distilgpt2__imdb/run_20260413-153020_a3f2/final/

STEP 3 — Load model from disk:
  base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
  model = PeftModel.from_pretrained(base_model, resolved_path)
  model.eval()
  # Kafka consumer thread is NOT started

STEP 4 — Flask API starts at localhost:5000
  POST /generate {"prompt": "Review: This was a disaster.\nSentiment:"}
  → model.generate() → slices prompt tokens → decodes
  → {"generated_text": "negative"}
```

---

## 9. Key Design Decisions & Engineering Notes

| Decision | Rationale |
|---|---|
| **Kafka as the transport layer** | Persistent, replayable, handles backpressure, supports multiple consumers, broker-mediated so services don't need to know each other's addresses. |
| **LoRA-only weight transfer** | Full model weights would be GB-scale and impractically slow over Kafka. LoRA adapters are 5–20 MB and transfer in seconds. |
| **Prompt-only label masking** | Prevents the model from wasting capacity learning to predict the (known, templated) prompt. Loss is computed exclusively on response tokens. |
| **Truncate prompt, not response** | The response (target) must always fit fully so the model sees the complete answer during training. The prompt can be truncated because the model still gets critical context. |
| **EOS token appended to every response** | Teaches the model to stop generating after the answer, preventing infinite repetition at inference time. |
| **CSV opened/closed per write** | Windows-specific: an open file handle blocks other processes from reading the file. Close after every write for Windows compatibility. |
| **SHA-256 hash as Kafka message key** | Enables Kafka log compaction (keeps only the latest record per key). Also deduplicates re-sent data. 64-char deterministic key. |
| **Verification message on producer startup** | Detects broker connectivity issues immediately, rather than silently failing after streaming has begun. |
| **`test_mode`: seek to end of topic** | Prevents re-processing data from a previous producer run when the topic already has messages. |
| **Sliding window evaluation** | Evaluating the same fixed batch every time would be uninformative if it were all one class. The window ensures the full eval pool is covered over time. |
| **`full_pool` eval_batch_size alias** | Enables evaluating the entire pool at once for classification tasks, eliminating window-composition noise from accuracy measurements. |
| **`other_label` bucket** | Captures model generations that don't start with any known label (e.g., the model starts talking instead of classifying). Without it, off-label outputs would incorrectly inflate one class's miss count. |
| **StreamFilter: short-circuit ordering** | O(1) checks first, O(N) scans second, O(N×R) regex last. Minimizes CPU time on the hot path (thousands of samples per minute). |
| **Weight application thread batching** | Block-wait for first update, then drain queue non-blockingly before acquiring model lock. Minimizes lock contention and inference blocking duration. |
| **Heartbeat logging in trainer** | Makes the trainer appear "alive" while waiting for producer data. Distinguishes "stuck" from "waiting". |
| **Fail-open in StreamFilter** | If `validate()` crashes unexpectedly, returns `(True, None)`. Prevents one malformed record from permanently starving the trainer. |
| **Auto-detect Kafka API version** | Pinning `api_version=(0, 10)` causes silent delivery failures on Kafka 3.x+. The producer uses auto-negotiate. |
| **`device_map={"": device_global}` in inference** | Forces all model layers onto the same device. Avoids cross-device tensor errors on MPS that can occur with `device_map="auto"`. |
| **`enable_lora_streaming: false` default** | New configs default to saving checkpoints without Kafka broadcasting. This is safer for development and decoupled workflows. Live streaming is opt-in. |
| **Hierarchical checkpoint layout** | `run_<ts>_<uid>/step_*/` layout means rapid restarts never collide. Legacy flat layout still works for backward compatibility. |
| **`decoupled` eval flag** | Removes the bottleneck of qualitative eval from the training hot path. Training stays fast; evaluation is async and unlimited. |
| **Qwen-native LoRA target modules** | `["q_proj", "k_proj", "v_proj", "o_proj"]` is required for Qwen2.5; GPT-2 targets cause PEFT injection failures on modern architectures. |
| **fp32 as default precision** | fp16 on MPS under small batches produces NaN losses. fp32 is stable on all platforms. fp16/bf16 available as opt-in for throughput when hardware supports it. |
| **Gradient clipping at max_norm=1.0** | Small batch sizes → high gradient variance → AdamW divergence. Clipping prevents norm explosion while still tracking the raw (pre-clip) norm in `grad_norm` for diagnostics. |
| **`set_to_none=True` in `zero_grad()`** | Sets parameter gradients to `None` instead of zero tensors. Saves one allocation cycle per step and prevents PyTorch from holding gradient tensors in the MPS unified memory pool between steps. |
| **Explicit `del` + `empty_cache()`** | Terminates PyTorch computational graphs immediately after backward pass. Prevents MPS unified memory from accumulating 25+ GB of retained graph buffers during Kafka `poll()` wait periods. |
| **`num_return_sequences` for consistency runs** | Moves Python-level `for run in range(N)` into a single `model.generate(num_return_sequences=N)` call — all runs execute in parallel as one GPU matrix operation. |
| **Left-padding for batch generation** | Decoder-only causal models require all sequences to be right-aligned in the padded batch. Tokenizer is temporarily switched to left-padding for `_generate_batch_records()`. |
| **QAFactEval and sentence-transformers on CPU** | Both heavy models lazy-loaded on CPU, never touching training GPU VRAM. Prevents OOM errors that would occur if they competed with the training LLM for device memory. |
| **Versioned artifact bundles (never overwrite)** | Each training run and each `evaluate.py` invocation gets a unique `artifact_<ts>_<uid>/` directory. Allows side-by-side comparison of different eval runs on the same checkpoint. |
| **UUID-per-run output dirs** | `<timestamp>_<uuid>` directory suffix prevents collision when two training runs start in the same second. |

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

In the IMDb quantitative config: `4 × 4 = 16` effective batch size.

The loss is scaled by `1 / gradient_accumulation_steps` before `.backward()` so that gradients are averaged (not summed) across micro-batches. This exactly simulates training with a larger batch size when GPU memory is insufficient to hold the full effective batch at once.

The **LR scheduler is stepped once per optimizer step** (every `gradient_accumulation_steps` micro-batches), not once per micro-batch. This ensures the LR schedule is consistent regardless of the accumulation setting.

### 10.3 LR Scheduler

The scheduler is built via `build_lr_scheduler()` as a `torch.optim.lr_scheduler.LambdaLR`.

For `cosine_with_warmup`:
```
if step < warmup_steps:
    lambda = step / warmup_steps            # linear warmup from 0 → 1.0
else:
    progress = (step - warmup_steps) / (T_max - warmup_steps)
    lambda = min_lr_ratio + (1 - min_lr_ratio) × 0.5 × (1 + cos(π × progress))
```

The `min_lr_ratio` floor prevents the LR from decaying to zero, which can cause the model to stop learning even if new data arrives. At `min_lr_ratio: 0.01`, the LR floor is 1% of the base LR.

### 10.4 Gradient Clipping & Stability

> **Previously:** No gradient clipping. With small batch sizes (e.g., `batch_size: 2` for GSM8K math sequences), stochastic gradients could spike by 10–100×, corrupting AdamW's momentum estimates and causing sudden loss explosions.
>
> **Now:** `torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)` is applied immediately **before** `optimizer.step()` on every optimizer step. This rescales the entire gradient vector to have L2 norm ≤ 1.0 if it exceeds that threshold.

**Why clip before `optimizer.step()`:** The raw gradient norm is still logged as `grad_norm` in `MetricsLogger` (accumulated across micro-batches during gradient accumulation). A high `grad_norm` in logs indicates a potentially unstable step even after clipping. This is useful for diagnosing issues without preventing the training signal.

**`max_norm=1.0`** is the standard value for LLM fine-tuning. Smaller values (e.g., 0.5) are more aggressive; larger values (e.g., 5.0) are more permissive.

### 10.5 Gradient Checkpointing

Configured via `training.gradient_checkpointing: true/false`. When enabled:

```python
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
```

**Trade-off:** Instead of storing all intermediate activations during the forward pass (which uses O(layers × batch × seq_len) VRAM), gradient checkpointing recomputes activations during the backward pass. This trades ~15–30% more compute for a significant reduction in VRAM usage.

**When to enable:** Recommended for large sequence lengths (e.g., 512+ tokens with a 345M+ parameter model) or when VRAM pressure causes OOM errors. For small models like `distilgpt2`, it's disabled (`false`) to maximize throughput.

**`use_reentrant=False`:** The newer non-reentrant implementation. Compatible with PEFT/LoRA and avoids some edge cases with custom `autograd` functions.

### 10.6 Quantitative Evaluation Strategies

| Strategy | Generation? | When to use | Primary metrics |
|---|---|---|---|
| `perplexity` | No (forward pass only) | Open-ended generation, unconditional LM | `eval_loss`, `perplexity` |
| `class_match` | Yes | Classification (fixed label set) | `accuracy`, `f1`, `mcc`, `kappa`, `exact_match` |
| `regex_extract` | Yes | Extractive generation (numeric/structured answer) | `accuracy`, `exact_match`, `answer_overlap_f1` |

**`class_match` strategy detail:**
```python
# Build known label space from eval targets + other_label
known_labels = set(canonicalize(target) for target in eval_targets)
known_labels.add(other_label)   # "other" by default

# For each prediction:
response_tokens = tokenize(generated_text)
for label in known_labels:
    if response_tokens starts with label_tokens:
        return label
return other_label   # fallback if no known label matched
```

Handles multi-word labels (e.g., "very positive") without being fooled by repetition.

**`regex_extract` strategy detail:**
```python
# For GSM8K: answer_regex = "#### (\d+)"
pred_match = re.search(answer_regex, generated_response)
gold_match = re.search(answer_regex, gold_target)
pred = pred_match.group(1).strip().lower() if pred_match else ""
gold = gold_match.group(1).strip().lower() if gold_match else ""
```

Applied to both model output and gold answer, so both are normalized before comparison.

**Greedy decoding for evaluation:** The evaluator always uses `do_sample=False` (greedy decoding) regardless of the inference config. This ensures deterministic, reproducible evaluation results.

**`max_distinct_labels_for_structure_metrics`:** If the number of distinct gold + predicted labels exceeds this threshold, F1/MCC/kappa are skipped. This safety guard prevents these metrics from being computed on open-ended text where hundreds of unique strings are "labels" — the results would be meaningless.

### 10.7 Qualitative Evaluation Strategies

| Strategy | Reference required? | Output metrics | Use case |
|---|---|---|---|
| `semantic_similarity` | Yes (golden responses) | `qual_semantic_similarity` | Instruction following, conversational |
| `keyword_density` | No (reference-free) | `qual_keyword_density`, `qual_type_token_ratio`, `qual_hapax_ratio` | Domain adaptation, style learning |
| `structural_cot` | No (pattern matching) | `qual_cot_anchor_count_mean`, `qual_cot_step_length_mean`, `qual_cot_coverage_rate` | Math reasoning, step-by-step tasks |
| `structured_slot_coverage` | No (MR parsing) | `qual_slot_coverage_mean`, `qual_consistency_score_mean`, `qual_perfect_coverage_rate`, `qual_slot_<name>_coverage` (per-slot), pinned anchor metrics | Structured NLG, slot-filling tasks |

All strategies additionally produce universal metrics: `qual_mean_response_length`, `qual_repetition_rate`, `qual_non_empty_rate`.

**Hardware alignment:**
- `semantic_similarity`: sentence-transformers runs on CPU (~90 MB RAM), never competing with LLM VRAM.
- `keyword_density`, `structural_cot`: pure string/regex operations — zero GPU or model requirements.
- `structured_slot_coverage`: uses the training LLM for generation (on same device), plus string analysis.

### 10.8 Metrics Catalog

Complete reference for every column in `metrics.csv`. Sparse columns are `NaN` when not applicable.

| Column | Source | Config Flag | Description |
|---|---|---|---|
| `step` | trainer | always | Optimizer step number |
| `loss` | trainer | always | Training loss at this step |
| `lr` | trainer | always | Learning rate at this step |
| `eval_loss` | Evaluator | `compute_loss: true` | Cross-entropy loss on eval samples (response tokens only) |
| `perplexity` | Evaluator | `compute_loss: true` | `exp(eval_loss)` — lower is better |
| `accuracy` | Evaluator | `compute_accuracy: true` | Fraction of predictions exactly matching gold labels |
| `aauc` | Evaluator | legacy alias | Same as `average_accuracy` (older column name, kept for backward compat) |
| `average_accuracy` | Evaluator | `compute_aauc: true` | Normalized area under accuracy-vs-step curve (trapezoidal) |
| `f1` | Evaluator | `compute_f1: true` | Macro F1 across all classes (from confusion matrix) |
| `mcc` | Evaluator | `compute_mcc: true` | Matthews Correlation Coefficient — robust to class imbalance |
| `kappa` | Evaluator | `compute_kappa: true` | Cohen's Kappa — agreement beyond chance |
| `exact_match` | Evaluator | `compute_exact_match: true` | Accuracy after punctuation stripping (more lenient) |
| `qafacteval` | Evaluator | `compute_qafacteval: true` | Factual consistency score [0, 1] via extractive QA (CPU, lazy) |
| `answer_overlap_f1` | Evaluator | `compute_answer_overlap_f1: true` | Token-level F1 between generated and reference target |
| `forgetting_max` | Evaluator | `compute_forgetting: true` | Max drop from peak across `forgetting_track_metrics` scalars |
| `update_latency_s` | Evaluator | `compute_update_latency: true` | Wall-clock seconds between optimizer steps |
| `eval_cycle_time_s` | Evaluator | `compute_eval_cycle_time: true` | Wall-clock seconds from previous eval end to current eval start |
| `backward_transfer` | Evaluator | `compute_backward_transfer: true` | Fraction of samples correct before but wrong now (forgetting signal) |
| `grad_norm` | trainer | always | Raw (pre-clip) L2 gradient norm — diagnostic for instability |
| `tokens_per_sec` | trainer | always | Response tokens processed per wall-clock second |
| `step_time_s` | trainer | always | Wall-clock seconds per optimizer step |
| `records_used_total` | trainer | always | Cumulative count of training records consumed |
| `qual_semantic_similarity` | QualitativeEvaluator | `method: semantic_similarity` | Mean cosine similarity between generated and golden responses [0, 1] |
| `qual_keyword_density` | QualitativeEvaluator | `method: keyword_density` | Fraction of configured keywords present in mean response |
| `qual_type_token_ratio` | QualitativeEvaluator | `method: keyword_density` | Unique tokens / total tokens (lexical diversity) |
| `qual_hapax_ratio` | QualitativeEvaluator | `method: keyword_density` | Fraction of words appearing exactly once (lexical richness) |
| `qual_cot_anchor_count_mean` | QualitativeEvaluator | `method: structural_cot` | Mean count of logic anchor matches per response |
| `qual_cot_step_length_mean` | QualitativeEvaluator | `method: structural_cot` | Mean character length of segments between anchors |
| `qual_cot_coverage_rate` | QualitativeEvaluator | `method: structural_cot` | Fraction of responses containing ≥1 logic anchor |
| `qual_mean_response_length` | QualitativeEvaluator | any strategy | Mean word count of generated responses |
| `qual_repetition_rate` | QualitativeEvaluator | any strategy | Mean fraction of repeated bigrams per response |
| `qual_non_empty_rate` | QualitativeEvaluator | any strategy | Fraction of non-empty generated responses |
| `qual_slot_coverage_mean` | QualitativeEvaluator | `method: structured_slot_coverage` | Mean fraction of MR slots verbalized in generated text |
| `qual_consistency_score_mean` | QualitativeEvaluator | `method: structured_slot_coverage` | Mean consistency of slot coverage across consistency_runs |
| `qual_perfect_coverage_rate` | QualitativeEvaluator | `method: structured_slot_coverage` | Fraction of samples where ALL slots are covered |
| `qual_slot_familyFriendly_inversion_rate` | QualitativeEvaluator | boolean_negation checker | Fraction where familyFriendly yes/no is inverted in generation |
| `qual_pinned_slot_coverage_mean` | QualitativeEvaluator | `pinned_anchors` configured | Mean slot coverage on the fixed pinned MR set |
| `qual_pinned_perfect_coverage_rate` | QualitativeEvaluator | `pinned_anchors` configured | Perfect coverage rate on pinned anchors |
| `qual_pinned_consistency_score` | QualitativeEvaluator | `pinned_anchors` configured | Consistency score on pinned anchors |

### 10.9 Metrics Logging & Plots

**CSV structure:** The metrics CSV has one row per event (training step or eval step). Not all columns are filled on every row — training step rows fill `loss, lr, grad_norm, tokens_per_sec, step_time_s, records_used_total`; eval rows fill the eval and qual columns. `plot_metrics.py` handles missing values (NaN) gracefully by skipping blanks.

**Two CSV files are maintained:**
- `metrics.csv` — all columns (includes sparse NaN-filled columns)
- `metrics_clean.csv` — only columns that were populated at least once during the run (recommended for analysis)

**Plot generation timing:**
- Auto-generated at the end of every training run.
- Can be regenerated at any time from a saved CSV using `utils/plot_metrics.py`.
- Uses non-interactive `Agg` backend so it works on headless servers.

### 10.10 Evaluation Artifact Bundle

Generated by `utils/evaluation_artifacts.py` at the end of training (from `MetricsLogger`) and by `evaluate.py` after each eval run.

```
evaluation_artifacts/
    index.json                    # master index of all artifact bundles
    artifact_<timestamp>_<uid>/   # one bundle per generation call (never overwritten)
        manifest.json             # bundle metadata, chart registry, config snapshot
        generation_log.json       # per-sample generation records
        metrics/
            resolved_metrics.csv  # clean CSV with sparse columns dropped
            <original_metrics.csv>
        dashboards/
            dashboard_dark.png    # summary card + hero chart, dark theme
            dashboard_light.png   # light theme variant
        insights/<theme>/         # grouped insight charts (learning curve, forgetting, etc.)
        plots/<theme>/            # individual per-metric PNG line charts
        report.html               # offline self-contained Plotly interactive dashboard
```

**No-overwrite guarantee:** Every bundle has a UUID suffix. Re-running `utils/plot_metrics.py` on the same CSV creates a new bundle alongside existing ones.

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

### 11.3 Streaming vs Static Checkpoint Mode

> **Previously:** Launching `inference.py` always required a running Kafka broker. There was no way to use a saved checkpoint for static inference without the full streaming infrastructure.
>
> **Now:** Two independent mechanisms allow static operation: the `kafka.enable_lora_streaming: false` config flag (suppresses the consumer threads) and the `--checkpoint` CLI argument (loads a specific adapter from disk via `PeftModel.from_pretrained`).

| Aspect | Streaming Mode | Static Checkpoint Mode |
|---|---|---|
| **Activation** | Default; `enable_lora_streaming: true` | `--checkpoint <path\|step\|"latest">` |
| **Kafka required?** | Yes | No |
| **Initial weights** | Empty LoRA adapter (base model); updated via Kafka | LoRA adapter loaded from disk |
| **Weight updates** | Continuous, from `lora-updates` topic | None (static) |
| **Background threads** | `kafka_consumer_thread` + `weight_application_thread` | Neither started |
| **Use case** | Live fine-tuning demo; production streaming | Offline demos; checkpoint evaluation; A/B testing |

**Using `--checkpoint`:**
```bash
# Load the 'final' checkpoint automatically
python inference.py --config configs/imdb_quantitative.yaml --checkpoint latest

# Load a specific step
python inference.py --config configs/imdb_quantitative.yaml --checkpoint 500
# or equivalently:
python inference.py --config configs/imdb_quantitative.yaml --checkpoint step_000500

# Load from an absolute path
python inference.py --config configs/imdb_quantitative.yaml \
  --checkpoint output/imdb/checkpoints/distilgpt2__imdb/run_xxx/final
```

`CheckpointManager.resolve_checkpoint_path()` handles the `"latest"` alias, integer step numbers, `"step_NNNNNN"` strings, and absolute paths.

### 11.4 REST API Reference

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

## 12. Optimizations & Engineering Hardening

This chapter consolidates all optimizations implemented during the modernization sprint. Each subsection describes the problem, the implementation, and the measured impact.

### 12.1 Apple Silicon / Unified Memory Sweep

**Context:** Apple Silicon Macs use a unified memory architecture — CPU and GPU share the same physical RAM. PyTorch's MPS backend does not immediately release memory to the OS after a tensor is freed in Python. During the Kafka `consumer.poll()` wait between training steps, PyTorch was silently retaining fully-executed forward/backward computational graphs in unified memory. With `batch_size: 8` and `max_seq_length: 512`, these retained graphs accumulated to **25+ GB**, grinding the OS to a halt via SSD swapping.

**Implementation** (every optimizer step in `trainer.py`):
```python
# 1. Explicitly delete all live tensors referencing the current graph
del outputs, loss, scaled_loss, batch

# 2. zero_grad with set_to_none=True — releases gradient tensors immediately
#    (instead of zeroing them, which still holds the allocation)
optimizer.zero_grad(set_to_none=True)

# 3. Python garbage collector sweep
import gc
gc.collect()

# 4. Force OS-level memory redistribution
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
elif torch.cuda.is_available():
    torch.cuda.empty_cache()
```

**Impact:** Unified memory usage stabilizes at **~10 GB** regardless of training duration or Kafka poll wait lengths. Multi-task workloads (training + evaluation + Kafka) become viable on 16 GB M-series machines.

---

### 12.2 Precision Policy — fp32 as Stable Default

**Context:** All early configs used `precision: "fp16"`. On Apple Silicon MPS with small batch sizes (2–8 samples), the reduced float range of fp16 caused gradient computations to overflow to `NaN` under specific batch compositions. Once a `NaN` appears in the loss, AdamW momentum estimates corrupt permanently — the model cannot recover without a restart.

> **Previously:** Hard-coded `precision: "fp16"` in all configs. NaN losses were a frequent failure mode on MPS, particularly early in training when gradients are largest.
>
> **Now:** `precision` is a per-config configurable field defaulting to `"fp32"`. All Qwen configs use `"fp32"`. The `imdb_qualitative.yaml` config uses `"fp16"` because Qwen2.5-1.5B is large enough that the memory savings are important and the model is more stable at that scale. `"bf16"` is available as an alternative — wider exponent range than fp16, narrower mantissa than fp32.

**Impact:** Zero NaN-related training failures since the switch. No measurable quality difference between fp32 and fp16 for LoRA fine-tuning at these scales.

---

### 12.3 Gradient Clipping

**Context:** Small batch sizes (2–4 samples) produce high gradient variance. With AdamW's adaptive learning rates, a single batch with abnormally high loss can push gradient norms to 10–100×, corrupting the optimizer's moment estimates and causing divergence.

**Implementation:**
```python
# Applied before optimizer.step() on every optimizer step
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

The raw gradient norm (before clipping) is still logged as `grad_norm`. Persistent high `grad_norm` values in the training log indicate instability even with clipping and may warrant reducing `learning_rate` or increasing `gradient_accumulation_steps`.

**Impact:** Eliminates parameter corruption from gradient explosions. Training is stable even at `batch_size: 2` on GSM8K math sequences.

---

### 12.4 Batched Qualitative Generation

**Context:** The original `_generate_responses()` in `eval_qualitative.py` was completely unbatched — one sample at a time:
```python
# Old implementation
for i, sample in enumerate(window):
    output_ids = model.generate(**inputs, ...)   # batch_size=1
```

With `eval_samples: 334` and `consistency_runs: 10`, this was **3,340 individual GPU calls**. On a Colab T4, this took **40 minutes** per eval interval — freezing training ingestion for the entire duration.

> **Previously:** Unbatched generation. 40 minutes for 3,340 calls at step 200.
>
> **Now:** `_generate_batch_records()` groups samples into chunks of `eval_batch_size` (e.g., 32). Chunks are tokenized together, left-padded (see §12.11), and sent to the GPU as a single `model.generate()` call.

**Impact:** ~10–15× speedup. The same 3,340 calls complete in **3–4 minutes**.

---

### 12.5 GPU Multi-Sequence Consistency Runs

**Context:** Even after batching individual samples, consistency runs were still implemented as a Python loop over `consistency_runs`:
```python
# Old implementation
for run_idx in range(self._consistency_runs):
    preds, refs = self._generate_responses(...)
```

This forced the GPU to re-tokenize the same prompts `consistency_runs` times, re-transfer tensors, and recompute KV-cache from scratch on each pass.

> **Previously:** Python-level loop. Generating 3 consistency runs ≈ 3× the time of 1 run.
>
> **Now:** `num_return_sequences=consistency_runs` is injected directly into `model.generate()`. All stochastic branches are generated in parallel as a single GPU matrix operation on `(batch_size × consistency_runs)` sequences.

**Impact:** 60–70% further reduction in generation time. Generating 3 or 10 consistency runs is nearly as fast as generating 1.

---

### 12.6 Kafka Consumer Long-Eval Reliability

**Context:** Kafka's default consumer session timeout (`session_timeout_ms=10000`, 10 seconds) is far shorter than the time needed for a qualitative eval pass. When `trainer.py` was blocked inside `Evaluator.evaluate()` for 15–25 minutes, the Kafka broker marked the consumer as dead and triggered a group rebalance. On return from eval, the consumer had been removed from the group and had to rejoin — causing offset reset and data replay.

**Implementation:** Three custom timeout settings now applied to `KafkaConsumer`:
```python
KafkaConsumer(
    max_poll_interval_ms=int(kafka_cfg.get('max_poll_interval_ms', 1_800_000)),  # 30 min
    session_timeout_ms=int(kafka_cfg.get('session_timeout_ms', 30_000)),          # 30 sec
    heartbeat_interval_ms=int(kafka_cfg.get('heartbeat_interval_ms', 10_000)),    # 10 sec
)
```

All three are configurable in the `kafka:` YAML block and default to the above values.

**Impact:** No consumer evictions or rebalances during eval, regardless of eval duration.

---

### 12.7 UUID Run Isolation

**Context:** Two separate collision problems existed:
1. **Output directory collision:** If two training runs started in the same second, they would write to the same `<timestamp>/` output directory, mixing metrics CSV rows.
2. **Consumer group offset collision:** In `test_mode`, if a training run crashed and was restarted, the Kafka consumer would resume from the last committed offset instead of seeking to the end — causing the trainer to re-process old data rather than new data from the current producer.

> **Previously:** Fixed `<timestamp>` directory suffix; shared consumer group ID across runs.
>
> **Now:** (1) `MetricsLogger` appends a 4-char `uuid.uuid4().hex[:4]` suffix to the run directory name — `<timestamp>_<uid>/`. (2) `CheckpointManager` creates `run_<timestamp>_<uid>/` per training run. (3) In `test_mode`, the Kafka consumer group ID is suffixed with a 4-char UUID, ensuring each test run has a fresh consumer with no committed offsets.

**Impact:** Zero collision between rapid restarts. `test_mode` always starts clean.

---

### 12.8 Producer Topic Hygiene

**Context:** In iterative testing, stale records from previous producer runs accumulated in Kafka topics. On restart, the trainer (with `auto_offset_reset="earliest"`) would replay all historical data — mixing old experimental batches with the current run's data.

**Implementation:** `clear_kafka_topic(kafka_cfg, topic)` function in `producer.py`. Deletes the topic and recreates it before streaming begins. Called optionally at producer startup based on configuration.

**Kafka API version (critical):**
```python
# DO NOT set api_version explicitly — let kafka-python auto-negotiate with the broker.
# Pinning api_version=(0,10) causes silent delivery failures on modern Kafka 3.x+ brokers.
# This is enforced via comment directly in the KafkaProducer construction code.
```

**Impact:** Clean slate on every producer restart. Silent delivery failures on Kafka 3.x+ eliminated.

---

### 12.9 Decoupled Flag — Removing Eval from the Hot Path

**Context:** (See §5.4 for full context.) Inline qualitative eval blocked Kafka ingestion for 40+ minutes per eval interval. Setting `eval_interval: 200` with `consistency_runs: 10` would only allow training to process ~200 batches between 40-minute pauses — completely defeating the "real-time" premise.

**Implementation:** `evaluation.decoupled: true` in config causes `trainer.py` to:
1. Skip instantiation of `Evaluator` and `QualitativeEvaluator`.
2. Focus entirely on training forward/backward passes and checkpoint saving.
3. Log at training end: `"To run decoupled evaluation: python evaluate.py --config <config>"`

**Impact:** Training throughput is maximized. Evaluation can be run asynchronously at any time, against any checkpoint, without affecting training.

---

### 12.10 Balanced Sliding-Window Evaluation

**Context:** The original sliding window advanced by `eval_batch_size` samples on each call. For class-imbalanced datasets (e.g., a window that happened to contain 95% "positive" reviews), accuracy could spike or crash due to window composition rather than model quality.

**Implementation:**
- `eval_batch_size: "full_pool"` alias (also accepts `"full"`, `"all"`, `"entire_pool"`) evaluates the entire shuffled pool at once — eliminating window noise.
- `other_label` fallback bucket absorbs off-label predictions instead of incorrectly assigning them to the nearest known label.
- `_normalize_class_match_labels()` collapses unexpected predicted labels back into the known label space before confusion matrix computation.
- `max_distinct_labels_for_structure_metrics` guard prevents confusion-matrix metrics from running on open-ended generative tasks.

**Impact:** Stable, comparable accuracy measurements across eval calls. No false accuracy spikes from lucky window sampling.

---

### 12.11 Left-Padding for Batched Generation

**Context:** HuggingFace tokenizers default to right-padding for training (important for loss computation). However, decoder-only causal models require all sequences to be **right-aligned** in a padded batch for generation — i.e., left-padded. If right-padded batches are sent to `model.generate()`, the model generates from the padding position rather than the end of the prompt.

**Implementation** (inside `_generate_batch_records()` and the qualitative evaluator):
```python
# Switch to left-padding for generation
original_padding_side = tokenizer.padding_side
tokenizer.padding_side = "left"
try:
    inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt")
    outputs = model.generate(**inputs, generation_config=gen_config)
    # ... decode and record outputs
finally:
    tokenizer.padding_side = original_padding_side   # restore for training
```

**Impact:** Correct batched generation. Without left-padding, batch generation produces garbage outputs (model generates from a padding position).

---

### 12.12 Lazy-Loaded Heavy Models

**Context:** Both `QAFactEval` (via `transformers.pipeline("question-answering", ...)`, ~120 MB) and `SemanticSimilarityMetric` (via `sentence-transformers`, ~90 MB) were candidates for eager loading at evaluator initialization. On hardware with tight VRAM, loading these at init time even when the metric is disabled would waste memory and slow startup.

**Implementation:**
- `_QAFactEvalScorer._pipeline` is `None` at init and populated only on the first call to `score()`. The pipeline is always loaded on **CPU** via the default `pipeline` device.
- `SemanticSimilarityMetric._model` is `None` at init and populated on the first call to `compute()`, always with `device="cpu"`.
- Both models remain on CPU for the entire session — they are never moved to GPU/MPS.

**Impact:**
- Startup is fast even when these metrics are enabled in config.
- First eval call triggers a one-time download/load (logged explicitly).
- Neither model competes with the training LLM for GPU VRAM — critical on systems with limited VRAM (e.g., 16 GB M-series Mac running a 3B parameter model).

---

## 13. Dependencies & Tech Stack

| Package | Version | Purpose |
|---|---|---|
| `torch` | Any (auto-selects CUDA/MPS/CPU) | Tensor ops, model training, inference |
| `transformers` | Any | HuggingFace model + tokenizer loading, `GenerationConfig`, `pipeline` |
| `peft` | Any | LoRA adapter implementation (`LoraConfig`, `get_peft_model`, `PeftModel`) |
| `datasets` | Any | HuggingFace dataset loading + shuffling, parquet file loading |
| `kafka-python` | Any | Kafka producer and consumer clients |
| `flask` | Any | REST API server for inference |
| `accelerate` | Any | Device-aware model loading utilities |
| `matplotlib` | Any | PNG/SVG training metric plot generation |
| `plotly` | Any | Interactive offline HTML charts in `report.html` |
| `sentence-transformers` | Any | Semantic similarity evaluation (MiniLM, **CPU-only**, ~90 MB) |
| `jinja2` | (transitive) | Prompt/response template rendering |
| `zlib` | (stdlib) | Compression-based repetition detection in StreamFilter |
| `hashlib` | (stdlib) | SHA-256 hashing for Kafka message keys |
| `scipy` | (transitive) | Statistics utilities (transitive via transformers/datasets) |
| `numpy` | (transitive) | Array operations for confusion matrix metrics |
| `trl` | Any | In `requirements.txt`; not directly used in current code (available for future SFT trainers) |
| `wandb` | Any | In `requirements.txt`; not integrated in current codebase (available for future experiment tracking) |

**QAFactEval note:** `QAFactEval` uses `transformers.pipeline("question-answering", model="deepset/minilm-uncased-squad2")` — no separate package installation required. The model (~120 MB) is downloaded from HuggingFace on first use, lazy-loaded to CPU.

**Platform support:**

| Platform | Status | Notes |
|---|---|---|
| **CUDA GPU (Linux/Windows)** | Fully supported | Primary target. `fp16`/`bf16` training available. |
| **Apple Silicon MPS (macOS)** | Fully supported | Officially tested with memory sweep (§12.1). Stable at ~10 GB for 3B models. `fp32` recommended. |
| **CPU** | Fallback | Functional but slow. Not recommended for models > 500M params. |

---

## 14. Setup & Running the System

### 14.1 Prerequisites

1. **Java JDK 11+** — required by Kafka. Set `JAVA_HOME` environment variable.
2. **Apache Kafka 3.3+** — KRaft mode (no Zookeeper). Download from [kafka.apache.org](https://kafka.apache.org/downloads).
3. **Python 3.9+** — with dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

### 14.2 Starting Kafka

#### macOS (KRaft mode — No Zookeeper)

```bash
# Install
brew install kafka

# Add to PATH (add to ~/.zshrc)
export PATH="/opt/homebrew/opt/kafka/bin:$PATH"
source ~/.zshrc

# One-time storage format (generates a cluster UUID):
KAFKA_CLUSTER_ID="$(kafka-storage random-uuid)"
kafka-storage format -t $KAFKA_CLUSTER_ID -c /opt/homebrew/etc/kafka/server.properties
# If you see "Log directory is already formatted" — skip this step.

# Start Kafka:
brew services start kafka

# Stop Kafka:
brew services stop kafka

# Verify (list topics):
kafka-topics --bootstrap-server localhost:9092 --list
```

#### Windows (KRaft mode — No Zookeeper)

```bat
cd C:\kafka

:: One-time storage format:
.\bin\windows\kafka-storage.bat random-uuid
:: Copy the printed UUID, then:
.\bin\windows\kafka-storage.bat format -t <YOUR_UUID> -c .\config\server.properties

:: Start the broker:
.\bin\windows\kafka-server-start.bat .\config\server.properties

:: Verify:
.\bin\windows\kafka-topics.bat --bootstrap-server localhost:9092 --list
```

#### Windows (Legacy — with Zookeeper)

> Use only for Kafka versions below 3.3.

```bat
:: Terminal 1 — Zookeeper
.\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties

:: Terminal 2 — Kafka broker
.\bin\windows\kafka-server-start.bat .\config\server.properties
```

### 14.3 Standard 3-Terminal Training Run

Open **3 terminals** in the project root and start in this order:

```bash
# Terminal 1 — Start inference server FIRST
# (if enable_lora_streaming: true — otherwise optional for training-only runs)
python inference.py --config configs/imdb_quantitative.yaml
```

```bash
# Terminal 2 — Start trainer
# (waits for data, logs ">>> Start the producer now <<<")
python trainer.py --config configs/imdb_quantitative.yaml
```

```bash
# Terminal 3 — Start producer LAST (after trainer is ready)
python producer.py --config configs/imdb_quantitative.yaml
```

**Startup order matters:**

| Order | Reason |
|---|---|
| Inference first | Uses `auto_offset_reset="latest"` — only wants weight updates sent *after* it starts. If trainer runs first, early weight pushes will be missed. |
| Trainer second | In `test_mode`, it seeks to the end of the training topic. This must happen before the producer starts publishing so no data is missed. |
| Producer last | The trainer logs a "Start the producer now" message when ready. This is the cue. |

**Test the API:**
```bash
curl -s -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Review: One of the best films I have ever seen.\nSentiment:"}' \
  | python3 -m json.tool
```

### 14.4 Static Inference (No Kafka Required)

Set `kafka.enable_lora_streaming: false` in your config (already the default for all current configs), then:

```bash
# Load the final/latest checkpoint automatically
python inference.py --config configs/imdb_quantitative.yaml --checkpoint latest

# Load a specific optimizer step
python inference.py --config configs/imdb_quantitative.yaml --checkpoint 500

# Load from an explicit directory path
python inference.py --config configs/imdb_quantitative.yaml \
  --checkpoint output/imdb/checkpoints/distilgpt2__imdb/run_20260413-153020_a3f2/step_000500
```

No Kafka broker needs to be running for static inference.

### 14.5 Decoupled Evaluation

After training completes (or even during training), run `evaluate.py` against any saved checkpoint:

```bash
# Evaluate the final checkpoint (default)
python evaluate.py --config configs/imdb_quantitative.yaml

# Evaluate checkpoint at step 500
python evaluate.py --config configs/imdb_quantitative.yaml --step 500

# Evaluate from a specific directory
python evaluate.py --config configs/imdb_quantitative.yaml \
  --checkpoint-dir output/imdb/checkpoints/distilgpt2__imdb/run_xxx/step_000500

# Evaluate ALL checkpoints + base model baseline (produces learning-arc CSV + plots)
python evaluate.py --config configs/imdb_quantitative.yaml --all-checkpoints

# List available checkpoints
python evaluate.py --config configs/imdb_quantitative.yaml --list
```

Results are written to `output/<project>/eval_results/...` — no Kafka required.

### 14.6 Regenerating Evaluation Artifacts from a CSV

Use `utils/plot_metrics.py` to regenerate the full artifact bundle from any existing metrics CSV:

```bash
python utils/plot_metrics.py "output/imdb/logs/infinitune-imdb-sentiment/20260315-120000/metrics.csv" \
  --config configs/imdb_quantitative.yaml

# Custom output directory
python utils/plot_metrics.py metrics.csv \
  --config configs/imdb_quantitative.yaml \
  --out-dir ./my_analysis_plots
```

**PowerShell helpers (from `docs/README.md`):**

```powershell
# Latest inline training run
$Config  = "configs/imdb_quantitative.yaml"
$LogRoot = "output/imdb/logs/infinitune-imdb-sentiment"
$Run = Get-ChildItem $LogRoot -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$Csv = Join-Path $Run.FullName "metrics_clean.csv"
if (-not (Test-Path $Csv)) { $Csv = Join-Path $Run.FullName "metrics.csv" }
python utils/plot_metrics.py $Csv --config $Config

# Latest single-checkpoint decoupled eval
$Config   = "configs/imdb_quantitative.yaml"
$EvalRoot = "output/imdb/eval_results/distilgpt2__imdb"
$EvalRun  = Get-ChildItem $EvalRoot -Directory -Recurse | Where-Object { $_.Name -like "eval_*" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$Csv = Join-Path $EvalRun.FullName "plots\eval_metrics.csv"
python utils/plot_metrics.py $Csv --config $Config

# Latest all-checkpoints comparison
$Config   = "configs/imdb_quantitative.yaml"
$EvalRoot = "output/imdb/eval_results/distilgpt2__imdb\all_checkpoints"
$EvalRun  = Get-ChildItem $EvalRoot -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$Csv      = Join-Path $EvalRun.FullName "all_checkpoints_results.csv"
python utils/plot_metrics.py $Csv --config $Config
```

---

## 15. Extending InfiniTune to a New Task

To add a new task (e.g., news category classification, toxicity detection, domain-specific NLG), only a new YAML config file is needed:

1. **Create `configs/my_task_[eval_mode].yaml`** by copying the closest existing config.
2. **Set `dataset.name`** to the HuggingFace dataset (e.g., `"ag_news"`).
3. **Set `dataset.column_mapping`** to map the dataset's columns to `input_col` and `target_col`.
4. **Set `dataset.label_map`** if labels are integers.
5. **Update `preprocessing.prompt_template`** and `preprocessing.response_template` using Jinja2 `{{ input }}` / `{{ target }}` placeholders.
6. **Set `kafka.training_topic`** and `kafka.lora_updates_topic` to unique topic names (avoids mixing data from different tasks).
7. **Choose `evaluation.strategy`**: `class_match` for classification, `regex_extract` for extractive generation, `perplexity` for open-ended generation.
8. **Adjust `lora.target_modules`** based on the model architecture:
   - **Qwen2.5, LLaMA, Mistral, Gemma:** `["q_proj", "k_proj", "v_proj", "o_proj"]`
   - **GPT-2, DistilGPT-2, gpt2-medium:** `["c_attn", "c_proj"]`
   - For broader domain adaptation with Qwen: add `["gate_proj", "up_proj", "down_proj"]`
9. **Adjust `inference.max_new_tokens`** to match expected output length.
10. **Choose `evaluation.decoupled`**: `true` if qualitative eval is expected to be slow; `false` if perplexity/class_match is fast enough for inline use.
11. **Optionally add `testing_strategy`**: choose a qualitative method and configure method-specific fields. Set unused fields to `null`.

No Python code changes are required for new tasks — the system is fully config-driven.

---

## 16. Evolution & Modernization Timeline

A chronological record of all major "old → new" transitions. Each entry corresponds to a subsection in §12.

| # | Area | Previously | Now |
|---|---|---|---|
| 1 | Config naming | Ad-hoc (`imdb_config.yaml`) | Strict `[dataset]_[eval-mode].yaml` convention |
| 2 | Base models | `distilgpt2` only — cannot reason on GSM8K or generate quality text | `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen2.5-3B`, `gpt2-medium` per task; `distilgpt2` retained as fast baseline |
| 3 | LoRA target_modules | GPT-2 specific `["c_attn", "c_proj"]` — PEFT injection failures on Qwen | Qwen-native `["q_proj", "k_proj", "v_proj", "o_proj"]`; GPT-2 targets retained for gpt2-medium configs |
| 4 | Documentation layout | Everything in `README.md` (bloated, hard to navigate) | Dedicated `docs/` with 6 per-config guides + context document + changelog |
| 5 | Evaluation timing | Always inline — blocked Kafka ingestion for 40+ min during qualitative eval | `evaluation.decoupled: true/false` flag; async `evaluate.py` for offline scoring |
| 6 | Inference launch | Required running Kafka broker for any inference | `kafka.enable_lora_streaming: false` + `--checkpoint` arg; `PeftModel.from_pretrained` from disk |
| 7 | Checkpoint layout | Flat `step_*/final/` under `<model>__<dataset>/` — runs could collide | Hierarchical `run_<ts>_<uid>/step_*/final/` with `checkpoint_meta.json` + `config_snapshot.yaml` |
| 8 | Precision | Hard-coded `fp16` — NaN losses on MPS under small-batch variance | Configurable `model.precision`; `fp32` as stable default; `fp16`/`bf16` opt-in |
| 9 | MPS memory | 25+ GB unified RAM blowup during `poll()` waits | Explicit `del` + `set_to_none=True` + `gc.collect()` + `torch.mps.empty_cache()` → steady ~10 GB |
| 10 | Gradient stability | Raw gradients; small batches → exploding norms | `torch.nn.utils.clip_grad_norm_(max_norm=1.0)` pre-`optimizer.step()` |
| 11 | VRAM on large seq | Always stored full activations | Optional `training.gradient_checkpointing` (`use_reentrant=False`) — VRAM vs compute trade-off |
| 12 | Qualitative generation | Unbatched (one sample at a time) — 40 min for 3,340 calls | Batched generation with left-padding, chunk size 32 — 10-15× speedup (~3–4 min) |
| 13 | Consistency runs | Python `for run in range(N)` — N sequential forward passes | `num_return_sequences=N` in `model.generate()` — parallel on GPU, 60-70% further reduction |
| 14 | Kafka long-eval | Default timeouts → consumer evicted during 15–25 min eval | `max_poll_interval_ms=1,800,000`, `session_timeout_ms=30,000`, `heartbeat_interval_ms=10,000` |
| 15 | Run isolation | Shared consumer group; fixed timestamp dir → collisions on rapid restart | UUID-suffixed consumer group in `test_mode`; `<ts>_<uuid>` output dirs |
| 16 | Kafka topic hygiene | Stale records from previous runs replayed on restart | `clear_kafka_topic()` removes stale records before streaming |
| 17 | Kafka API version | Pinned `api_version=(0,10)` → silent delivery failures on Kafka 3.x+ | Auto-negotiation; pinning explicitly forbidden via code comment |
| 18 | Eval sample balance | Sliding window could be class-imbalanced | `full_pool` alias, `other_label` bucket, `_normalize_class_match_labels()` |
| 19 | Heavy eval models | Eager loading at evaluator init | Lazy-load on first `score()`/`compute()` call, CPU-only, ~120 MB / ~90 MB |
| 20 | Report surface | PNG dashboard embedded in `report.html`; KPI text overlapped | Independent Plotly HTML + reconstructed card-based PNG dashboard; usecase-aware KPIs |

---

## 17. Glossary

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
| **Gradient clipping** | Rescaling the gradient vector so its L2 norm does not exceed `max_norm`; prevents exploding gradients |
| **Gradient checkpointing** | Trading compute for VRAM by recomputing activations during backward pass instead of storing them |
| **EOS token** | End-of-sequence token; appended to every response so the model learns to stop generating |
| **Hot-swap** | Updating model weights at runtime without restarting the serving process |
| **Jinja2 template** | A text template with `{{ variable }}` placeholders used to format prompts and responses |
| **LambdaLR** | PyTorch LR scheduler that applies a user-defined function to compute the LR multiplier each step |
| **Online learning** | Training on a stream of examples one at a time (or in small batches), rather than on a fixed dataset |
| **Perplexity** | `exp(cross_entropy_loss)` — measure of how "surprised" the model is by the data; lower is better |
| **MCC** | Matthews Correlation Coefficient — classification quality metric robust to class imbalance |
| **Cohen's Kappa** | Agreement metric that corrects for chance agreement between predicted and actual labels |
| **Macro F1** | Average F1 score across all classes, treating each class equally regardless of frequency |
| **Sliding window eval** | Evaluating on a rotating window of the eval pool rather than the same fixed subset every time |
| **`full_pool` eval** | Setting `eval_batch_size: "full_pool"` evaluates the entire shuffled pool at once, removing window-composition noise |
| **`other_label`** | Fallback class bucket in `class_match` for model generations that don't start with any known label |
| **KRaft mode** | Kafka Raft consensus — Kafka without Zookeeper, available since Kafka 3.3 |
| **zlib compression ratio** | `len(zlib.compress(text)) / len(text)` — used to detect repetitive/spam text efficiently |
| **`strict=False` in load_state_dict** | Only update keys present in the state dict; ignore missing keys in the model |
| **`device_map="auto"`** | HuggingFace `accelerate` feature: automatically distributes model layers across available devices |
| **`device_map={"": device}`** | Forces all model layers onto a single explicit device; prevents cross-device tensor issues |
| **Decoupled Eval** | Running `evaluate.py` independently from training; requires no Kafka connection; scores any saved checkpoint |
| **Static Checkpoint Mode** | Launching `inference.py --checkpoint <path>` to load a LoRA adapter from disk without Kafka |
| **AAUC** | Average Accuracy Under the Curve — normalized trapezoidal area under the accuracy-vs-step curve; reported as both `aauc` and `average_accuracy` |
| **Backward Transfer (BWT)** | Fraction of eval samples that were correct at a previous checkpoint but are now wrong; negative BWT signals catastrophic forgetting |
| **Forgetting-Max** | Maximum drop from peak value across any tracked `forgetting_track_metrics` scalar; zero means no regression |
| **Update Latency** | `update_latency_s` — wall-clock seconds between consecutive optimizer steps; measures training throughput |
| **Eval Cycle Time** | `eval_cycle_time_s` — wall-clock seconds from the previous eval's end to the current eval's start; includes training steps + I/O |
| **QAFactEval** | Factual consistency score [0, 1] computed by asking a QA model to extract source key spans from the generated text; lazy-loaded on CPU via `deepset/minilm-uncased-squad2` (~120 MB) |
| **Answer Overlap F1** | Token-level F1 between generated text and reference target; approximates ROUGE-1 without requiring the `rouge_score` library |
| **Consistency Runs** | `consistency_runs` stochastic generation passes per input; implemented via `num_return_sequences` for GPU parallelism |
| **Pinned Anchors** | A fixed set of input MRs evaluated at every checkpoint to enable direct cross-step comparison on identical inputs |
| **Slot Coverage** | Fraction of meaning representation (MR) attribute slots that are correctly verbalized in the generated text |
| **Boolean Negation Checker** | Slot checker for yes/no attributes (e.g., `familyFriendly`): detects positive vs negative phrasing, not just substring presence |
| **Fluency Gate** | `_is_valid_restaurant_description()` — rejects hallucinated or incoherent E2E NLG generations before scoring; logged as `[HALLUCINATION IGNORED]` |
| **Semantic Similarity** | Cosine similarity between sentence embeddings of generated and golden responses; computed via `sentence-transformers/all-MiniLM-L6-v2` on CPU |
| **Keyword Density** | Fraction of a configured domain keyword list present in a generated text; measures vocabulary adoption |
| **Type-Token Ratio (TTR)** | Unique tokens / total tokens; measures lexical diversity. 1.0 = every word unique; 0 = single word repeated |
| **Hapax Ratio** | Fraction of words appearing exactly once in a text; measures lexical richness |
| **Logic Anchor** | A compiled regex pattern (e.g., `r"\bfirst\b"`, `r"Step\s*\d+[:\.]"`) used to detect Chain-of-Thought reasoning markers in generated text |
| **`set_to_none=True`** | Parameter in `optimizer.zero_grad()`: sets gradient tensors to `None` instead of zero, releasing the allocation and preventing MPS from retaining gradient buffers |
| **MPS `empty_cache()`** | `torch.mps.empty_cache()` — forces the MPS allocator to release all cached (but currently unused) memory back to the OS; prevents unified RAM accumulation |
| **`checkpoint_meta.json`** | Per-checkpoint JSON file storing provenance: step, timestamp, model name, dataset name, LoRA config, training loss |
| **`config_snapshot.yaml`** | Full YAML config copy stored inside each checkpoint directory; makes every checkpoint self-contained for re-evaluation without needing the original config path |
| **Evaluation Artifact Bundle** | Versioned directory (`artifact_<timestamp>_<uid>/`) containing `report.html`, PNG dashboards, per-metric plots, `manifest.json`, and `generation_log.json` |
| **Left-padding** | Tokenizer mode where padding tokens are added to the left of sequences; required for correct batch generation with decoder-only causal models |
| **PeftModel.from_pretrained** | PEFT library function that loads a LoRA adapter from a checkpoint directory onto an existing base model |
