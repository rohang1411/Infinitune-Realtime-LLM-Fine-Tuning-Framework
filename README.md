<div align="center">

# ♾️ InfiniTune
### Realtime LLM Fine-Tuning Framework

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg?style=flat)](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Kafka](https://img.shields.io/badge/Apache_Kafka-231F20?style=flat&logo=apache-kafka&logoColor=white)](https://kafka.apache.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://makeapullrequest.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A distributed framework for **continuously fine-tuning Large Language Models in real time** using Kafka data streams and LoRA (Low-Rank Adaptation). As new training data arrives, the model adapts on-the-fly and the inference server receives updated adapter weights automatically — no restarts required.

<br>

[**Architecture**](#architecture) • [**Getting Started**](#running-infinitune) • [**Configurations**](#available-configs) • [**Evaluation Suite**](#qualitative-evaluation-suite)

<br>

**🚀 Zero-Downtime Hot-Swaps** &nbsp;|&nbsp; **🧠 Consumer Hardware Friendly** &nbsp;|&nbsp; **⚡ Live Streaming Data** &nbsp;|&nbsp; **📊 Qualitative & Quantitative Eval**

</div>

---

## Table of Contents

1. [What is InfiniTune?](#what-is-infinitune)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Dependencies](#dependencies)
5. [Kafka Setup](#kafka-setup)
   - [macOS (KRaft mode)](#macos-kraft-mode--no-zookeeper)
   - [Windows (KRaft mode)](#windows-kraft-mode--no-zookeeper)
   - [Windows (Legacy — with Zookeeper)](#windows-legacy--with-zookeeper)
6. [Running InfiniTune](#running-infinitune)
   - [Quick Start](#quick-start)
   - [Available Configs](#available-configs)
   - [Key Config Options](#key-config-options)
7. [Checkpoint Saving](#checkpoint-saving)
8. [Evaluation Modes](#evaluation-modes)
   - [Inline Evaluation (during training)](#inline-evaluation-during-training)
   - [Decoupled Evaluation (after training)](#decoupled-evaluation-after-training)
9. [Qualitative Evaluation Suite](#qualitative-evaluation-suite)
   - [The Three Proxy Strategies](#the-three-proxy-strategies)
   - [Running the Qualitative Configs](#running-the-qualitative-configs)
   - [Interpreting the Metrics](#interpreting-the-metrics)
10. [Output Directory Structure](#output-directory-structure)
11. [Regenerating Plots](#regenerating-plots)

---

## What is InfiniTune?

Traditional fine-tuning requires a static dataset, an offline training run, and a manual deployment step. **InfiniTune removes all three bottlenecks.**

It is built around three decoupled services that communicate over Kafka:

| Service | Script | Role |
|---|---|---|
| **Producer** | `producer.py` | Streams training samples from a HuggingFace dataset to a Kafka topic |
| **Trainer** | `trainer.py` | Consumes data from Kafka, fine-tunes the model with LoRA, saves checkpoints, and pushes updated LoRA adapter weights back to Kafka |
| **Inference Server** | `inference.py` | Loads the base model + LoRA adapter, serves a REST API, and hot-swaps adapter weights in real time as the trainer pushes updates |

A fourth standalone script handles post-training evaluation:

| Script | Role |
|---|---|
| `evaluate.py` | Loads any saved checkpoint and runs the full evaluation suite (quantitative + qualitative) without re-training |

Key properties:
- **Online (streaming) learning** — the model improves continuously as data flows in
- **Memory-efficient** — uses LoRA adapters (only a fraction of model parameters are trained)
- **Config-driven** — all hyperparameters, dataset settings, and evaluation logic are defined in a single YAML file
- **Multi-task** — pre-built configs for IMDb, GSM8K, Alpaca, and qualitative evaluation variants
- **Dual evaluation modes** — inline (during training) and decoupled (after training via `evaluate.py`)
- **Versioned outputs** — training logs, checkpoints, and evaluation results are all versioned and never overwrite previous runs

---

## Architecture

```
┌─────────────┐   training data    ┌────────────────────┐   LoRA weights    ┌───────────────────┐
│  Producer   │ ─────────────────► │   Kafka Broker     │ ────────────────► │  Inference Server │
│ producer.py │                    │  (localhost:9092)  │                   │  inference.py     │
└─────────────┘                    └──────────┬─────────┘                   └───────────────────┘
                                              │ training data
                                              ▼
                                   ┌───────────────────────────┐
                                   │        Trainer             │
                                   │       trainer.py           │
                                   │      (LoRA + AdamW)        │
                                   │                            │
                                   │  ┌─────────────────────┐  │
                                   │  │  Inline: Quant.Eval  │  │  ← eval_metrics_train.py
                                   │  └─────────────────────┘  │
                                   │  ┌─────────────────────┐  │
                                   │  │  Inline: Qual. Eval  │  │  ← eval_qualitative.py
                                   │  └─────────────────────┘  │
                                   │  ┌─────────────────────┐  │
                                   │  │  CheckpointManager   │  │  ← utils/checkpoint_manager.py
                                   │  └──────────┬──────────┘  │
                                   └─────────────┼─────────────┘
                                                 │ saves LoRA adapter
                                                 ▼
                                      output/<project>/checkpoints/
                                         step_000100/
                                         step_000200/
                                         final/
                                                 │
                                    ┌────────────┘
                                    │  loads checkpoint
                                    ▼
                           ┌─────────────────┐
                           │   evaluate.py    │
                           │  (no Kafka, no  │
                           │   re-training)  │
                           │                 │
                           │  Quant. + Qual. │
                           │  full-pool eval  │
                           └─────────────────┘
                                    │ saves results
                                    ▼
                           output/<project>/eval_results/
                               step_000200/
                                 eval_20260412-150230_a3f2/
                                   eval_results.json
                                   plots/*.png
```

**Data flow during training:**
1. `producer.py` reads a HuggingFace dataset, applies filtering and templating, and publishes samples to the `training-data` Kafka topic.
2. `trainer.py` consumes samples from Kafka, runs a forward+backward pass, and every N steps:
   - Saves the LoRA adapter checkpoint to `output/<project>/checkpoints/`
   - Pushes updated LoRA adapter weights to the `lora-updates` Kafka topic
3. `inference.py` hot-applies weight updates to the live model while serving generation requests on a Flask REST API at `http://localhost:5000`.
4. Both quantitative and qualitative evaluation run **inside the trainer** on configurable intervals, writing results to a unified CSV log.

**After training:**
5. `evaluate.py` loads any saved checkpoint, runs the full evaluation suite, and saves versioned results to `output/<project>/eval_results/`.

---

## Project Structure

```
InfiniTune/
├── producer.py                    # Data streaming service
├── trainer.py                     # Training + inline eval + checkpoint saving
├── inference.py                   # REST API + live LoRA weight updates
├── evaluate.py                    # Standalone decoupled evaluation script
│
├── configs/
│   ├── imdb_quantitative.yaml           # IMDb sentiment classification
│   ├── gsm8k_quantitative.yaml          # GSM8K math reasoning
│   ├── alpaca_qualitative.yaml     # Alpaca instruction following
│   ├── imdb_qualitative.yaml   # IMDb unconditional language model
│   └── gsm8k_qualitative.yaml # GSM8K + CoT structure metrics
│
├── utils/
│   ├── checkpoint_manager.py      # Versioned LoRA adapter save/load
│   ├── eval_metrics_train.py      # Quantitative evaluation (Evaluator class)
│   ├── eval_qualitative.py        # Qualitative evaluation (strategies + orchestrator)
│   ├── plot_metrics.py            # Standalone plot regeneration utility
│   └── stream_filter.py           # Kafka data quality filtering
│
├── output/                        # All generated artefacts (git-ignored)
│   └── <project>/
│       ├── checkpoints/           # Saved LoRA adapters
│       ├── logs/                  # Training metrics CSVs + plots
│       └── eval_results/          # Decoupled evaluation results
│
└── requirements.txt
```

---

## Dependencies

### System Requirements

- **Python** 3.9+
- **Apache Kafka** 3.3+ (KRaft mode) or 4.x
- **Java JDK** 11+ (required by Kafka)

### Python Packages

Install all Python dependencies with:

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `torch` | Model training and inference |
| `transformers` | HuggingFace model/tokenizer loading |
| `peft` | LoRA adapter implementation |
| `datasets` | HuggingFace dataset loading |
| `kafka-python` | Kafka producer/consumer client |
| `flask` | REST API for the inference server |
| `accelerate` | Device-aware model loading |
| `trl` | Trainer utilities |
| `matplotlib` | Training metrics plots |
| `sentence-transformers` | Semantic similarity evaluation (MiniLM, CPU-only) |
| `jinja2` | Prompt and response templating |

> **`sentence-transformers`** is only needed for the `semantic_similarity` qualitative strategy. The other two strategies use pure string operations. The import is guarded — if the package is absent, only the semantic similarity strategy will fail at runtime.

> **macOS (Apple Silicon):** PyTorch MPS backend is used automatically when CUDA is unavailable. All qualitative metrics are CPU-only and do not compete with training for MPS memory.

---

## Kafka Setup

InfiniTune uses Kafka as the data backbone. Kafka 3.3+ supports **KRaft mode** (no Zookeeper required).

### macOS (KRaft mode — No Zookeeper)

#### 1. Install Kafka

```bash
brew install kafka
```

#### 2. Add Kafka to PATH

Add to your `~/.zshrc`:

```bash
export PATH="/opt/homebrew/opt/kafka/bin:$PATH"
```

```bash
source ~/.zshrc
```

#### 3. Format Storage (One-time only)

```bash
KAFKA_CLUSTER_ID="$(kafka-storage random-uuid)"
kafka-storage format -t $KAFKA_CLUSTER_ID -c /opt/homebrew/etc/kafka/server.properties
```

> If you get `"Log directory is already formatted"`, skip this step.

#### 4. Start Kafka

**Background service (recommended):**
```bash
brew services stop zookeeper  # ensure zookeeper is off
brew services start kafka
```

**Foreground (for debugging):**
```bash
kafka-server-start /opt/homebrew/etc/kafka/server.properties
```

#### 5. Verify

```bash
kafka-topics --bootstrap-server localhost:9092 --list
```

#### Stop Kafka

```bash
brew services stop kafka
```

---

### Windows (KRaft mode — No Zookeeper)

#### 1. Install Java

Download **JDK 11+** from [adoptium.net](https://adoptium.net/) and set `JAVA_HOME`.

#### 2. Download Kafka

Download the latest binary from [kafka.apache.org/downloads](https://kafka.apache.org/downloads) and extract to `C:\kafka`.

#### 3. Format Storage (One-time only)

```bat
cd C:\kafka
.\bin\windows\kafka-storage.bat random-uuid
```

Copy the UUID, then:

```bat
.\bin\windows\kafka-storage.bat format -t <YOUR_UUID_HERE> -c .\config\server.properties
```

#### 4. Start Kafka

```bat
cd C:\kafka
.\bin\windows\kafka-server-start.bat .\config\server.properties
```

#### 5. Verify

```bat
.\bin\windows\kafka-topics.bat --bootstrap-server localhost:9092 --list
```

---

### Windows (Legacy — with Zookeeper)

> Use only for Kafka < 3.3.

```bat
# Terminal 1
.\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties

# Terminal 2
.\bin\windows\kafka-server-start.bat .\config\server.properties
```

---

## Running InfiniTune

### Quick Start

Make sure Kafka is running, then open **three separate terminals**:

**Terminal 1 — Inference Server** *(start first — it listens for weight updates)*
```bash
python inference.py --config configs/imdb_quantitative.yaml
```

**Terminal 2 — Trainer** *(loads model, connects to Kafka, waits for data)*
```bash
python trainer.py --config configs/imdb_quantitative.yaml
```

> The trainer logs `>>> Start the producer now (if not already running). <<<` when it is ready.

**Terminal 3 — Producer** *(streams training data)*
```bash
python producer.py --config configs/imdb_quantitative.yaml
```

**Terminal 4 — Test the API** *(optional)*
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Review: This movie was absolutely terrible.\nSentiment:"}'

curl http://localhost:5000/health
# Expected: {"status": "ok"}
```

### Available Configs

| Config | Task | Dataset | Eval Type |
|---|---|---|---|
| `configs/imdb_quantitative.yaml` | Sentiment classification | IMDb (25k reviews) | Quantitative (Accuracy, F1, MCC) |
| `configs/gsm8k_quantitative.yaml` | Math reasoning | GSM8K (8.5k problems) | Quantitative (Exact Match) |
| `configs/alpaca_qualitative.yaml` | Instruction following | Alpaca (52k) | Qualitative (Semantic Similarity) |
| `configs/imdb_qualitative.yaml` | Domain adaptive generation | IMDb (unconditional LM) | Qualitative (Keyword Density + TTR) |
| `configs/gsm8k_qualitative.yaml` | Math reasoning + CoT structure | GSM8K | Qualitative (CoT Adherence) + Quantitative |

### Key Config Options

```yaml
project:
  name: "my-run"
  output_dir: "./output/my-run"    # All logs, checkpoints, and eval results go here

model:
  name: "distilgpt2"               # Any HuggingFace CausalLM model
  max_seq_length: 512

lora:
  r: 8                             # Adapter rank (higher = more capacity, more memory)
  alpha: 64                        # Scaling factor (effective LR = alpha/r × learning_rate)
  target_modules: [c_attn, c_proj] # Which linear layers to adapt

training:
  batch_size: 8
  gradient_accumulation_steps: 4   # Effective batch = batch_size × grad_accum
  learning_rate: 1e-4
  max_steps: 2000
  test_mode: true                  # true = train on entire dataset, stop on EOF
  lr_scheduler:
    type: "cosine_with_warmup"     # cosine_with_warmup | linear | constant
    warmup_steps: 50
    T_max: 2000                    # Must match max_steps for full cosine decay
  save_checkpoints:
    enabled: true                  # Save LoRA adapter periodically (default: true)
    save_every_steps: 200          # Checkpoint every N optimizer steps
    save_final: true               # Always save a 'final' checkpoint at training end

evaluation:                        # Inline quantitative eval during training
  enabled: true
  strategy: "class_match"          # class_match | regex_extract | perplexity
  eval_interval: 50
  eval_pool_size: 5000
  eval_batch_size: 100
```

---

## Checkpoint Saving

During training, the **LoRA adapter** (not the full base model) is saved to disk automatically. Adapter files are tiny (~5–20 MB) compared to the full model (~500 MB for distilgpt2).

**What is saved at each checkpoint:**
- `adapter_model.safetensors` — LoRA adapter weights
- `adapter_config.json` — PEFT adapter configuration
- `checkpoint_meta.json` — step number, timestamp, model name, dataset name, and loss

**Directory layout:**
```
output/<project>/checkpoints/<model>__<dataset>/
    step_000100/         ← saved at step 100
    step_000200/         ← saved at step 200
    ...
    final/               ← always saved at training end (overwrites previous 'final')
```

**No-overwrite policy:**
- Step directories (`step_000100/`, etc.) are **never overwritten**. If a checkpoint already exists for a step from a previous run, the current run skips it and logs a message.
- The `final/` checkpoint is the only exception — it always reflects the latest training endpoint.

**Config:**
```yaml
training:
  save_checkpoints:
    enabled: true           # Set to false to skip all checkpoint saving
    save_every_steps: 100   # How often to save
    save_final: true        # Whether to save 'final' at the end
```

---

## Evaluation Modes

InfiniTune supports two evaluation modes that are **fully independent** — you can use one or both.

### Inline Evaluation (during training)

Runs automatically inside `trainer.py` on a configurable interval. Useful for monitoring training progress in real time.

- **Quantitative** (`evaluation` block): every `eval_interval` steps, evaluates loss, perplexity, accuracy, F1, exact match, etc.
- **Qualitative** (`testing_strategy` block): every `testing_strategy.eval_interval` steps, generates responses and computes proxy metrics.
- Results are written to `output/<project>/logs/.../metrics.csv` after each eval event.
- Plots are generated automatically at the end of the training run.

**To disable inline evaluation** (for maximum training speed):
```yaml
evaluation:
  enabled: false

testing_strategy:
  enabled: false
```

---

### Decoupled Evaluation (after training)

Run `evaluate.py` to evaluate any saved checkpoint without re-training. Useful when:
- You want to evaluate with updated metrics logic
- You want to compare multiple checkpoints
- You want definitive full-pool scores (not windowed)

**Key differences from inline evaluation:**
| Feature | Inline | Decoupled |
|---|---|---|
| When it runs | During training | After training |
| Eval pool coverage | Sliding window (eval_batch_size samples) | Full pool (all samples) |
| Kafka required | Yes | No |
| Results location | `logs/.../metrics.csv` | `eval_results/.../eval_results.json` |
| Re-runnable | No (runs once per step) | Yes (each run creates a new versioned directory) |

#### Usage

```bash
# Evaluate the 'final' checkpoint (default)
python evaluate.py --config configs/imdb_quantitative.yaml

# Evaluate a specific step
python evaluate.py --config configs/imdb_quantitative.yaml --step 200

# Evaluate a specific checkpoint directory
python evaluate.py --config configs/imdb_quantitative.yaml \
    --checkpoint-dir output/imdb/checkpoints/distilgpt2__stanfordnlp_imdb/step_000200

# Evaluate ALL checkpoints (produces per-checkpoint results + combined plots + CSV)
python evaluate.py --config configs/imdb_quantitative.yaml --all-checkpoints

# List available checkpoints without evaluating
python evaluate.py --config configs/imdb_quantitative.yaml --list
```

#### Output

Each evaluation run creates a **new versioned directory** — never overwriting previous runs:

```
output/<project>/eval_results/<model>__<dataset>/
    step_000200/
        eval_20260412-150230_a3f2/      ← timestamp + random suffix
            eval_results.json           ← all metric values
            eval_config.json            ← config used for this eval
            plots/
                accuracy.png
                perplexity.png
                qual_semantic_similarity.png
                ...
    final/
        eval_20260412-160010_b7e1/
            ...
    all_checkpoints/                    ← only with --all-checkpoints
        eval_20260412-161500_c2d9/
            all_checkpoints_results.csv ← one row per checkpoint
            plots/
                accuracy.png            ← accuracy vs step across all checkpoints
                ...
```

#### Reading `eval_results.json`

```json
{
  "checkpoint_path": "output/imdb/checkpoints/distilgpt2__stanfordnlp_imdb/step_000200",
  "eval_timestamp": "2026-04-12T15:02:30",
  "metrics": {
    "eval_loss": 1.2456,
    "perplexity": 3.472,
    "accuracy": 0.73,
    "f1": 0.724,
    "mcc": 0.467,
    "exact_match": 0.71
  }
}
```

---

## Qualitative Evaluation Suite

### Overview

The Qualitative Evaluation Suite measures improvement in tone, style, and reasoning structure — without LLM API calls and with minimal RAM overhead.

**Design constraints:**
- Zero LLM API calls (no GPT-4, no Claude)
- All proxy metrics run on CPU — no VRAM competition with training
- SentenceTransformers (MiniLM) is explicitly pinned to `device="cpu"` (~90 MB RAM)
- 0 extra MB for keyword density and CoT structure strategies
- Independent eval interval from quantitative eval

---

### The Three Proxy Strategies

#### 1. Semantic Similarity — `alpaca_qualitative.yaml`

**Dataset:** `tatsu-lab/alpaca` (52k instruction-following examples)  
**Model:** `sentence-transformers/all-MiniLM-L6-v2` (CPU, ~90 MB)

The model generates a response to an instruction. That response is compared to the golden output using **cosine similarity of sentence embeddings**.

```
Instruction ─► model generates response
                        │
Generated: "Python is a programming language..."
Reference: "Python is a high-level language used for..."
                        │
         Cosine similarity = 0.73   ← qual_semantic_similarity
```

**Rising similarity over training = model learning aligned instruction-following responses.**

**Blind spots:** Does not measure fluency or grammar.

---

#### 2. Keyword Density + TTR — `imdb_qualitative.yaml`

**Dataset:** `stanfordnlp/imdb` (unconditional language modeling — model learns to *write* reviews)  
**Reference:** None required (reference-free)

Model is prompted with `"Write a movie review:\n"` and generates freely. Output is analysed for:
- **Keyword Density** — fraction of words from domain vocabulary (cinematography, screenplay, pacing, …)
- **Type-Token Ratio (TTR)** — unique words / total words (lexical diversity)
- **Hapax Ratio** — fraction of words used exactly once per response

```
Generated: "The cinematography was stunning. The pacing felt off
            though the screenplay had some brilliant moments..."
                              │
qual_keyword_density  = 0.068   (6.8% domain keywords)
qual_type_token_ratio = 0.71    (71% unique words)
qual_hapax_ratio      = 0.58
```

> **Note:** `imdb_qualitative.yaml` trains for *unconditional LM* (write reviews). `imdb_quantitative.yaml` trains for *sentiment classification* (predict positive/negative). Different tasks, same dataset.

---

#### 3. Structural CoT Adherence — `gsm8k_qualitative.yaml`

**Dataset:** `gsm8k` (grade school math, same as `gsm8k_quantitative.yaml`)  
**Complements:** quantitative `exact_match`

Model generates a math reasoning chain. Output is analysed for **logic anchors** — regex patterns marking structured reasoning steps:

```
Generated: "First, we find the total apples.
            Step 1: 5 apples × 3 baskets = 15 apples
            Therefore, the answer is 15.
            #### 15"
                     │
qual_cot_anchor_count_mean = 4.0     (4 anchors found)
qual_cot_step_length_mean  = 38.5    (avg 38.5 chars between anchors)
qual_cot_coverage_rate     = 1.0     (100% of responses had ≥1 anchor)
```

`exact_match` tells you if the final answer is right. Structural CoT tells you if the reasoning *process* is structured. **Both rising together = genuine CoT learning.**

---

### Running the Qualitative Configs

All qualitative configs follow the same three-terminal launch pattern:

**Conversational (Alpaca):**
```bash
python inference.py --config configs/alpaca_qualitative.yaml
python trainer.py   --config configs/alpaca_qualitative.yaml
python producer.py  --config configs/alpaca_qualitative.yaml
```

**Domain Adaptation (IMDb unconditional LM):**
```bash
python inference.py --config configs/imdb_qualitative.yaml
python trainer.py   --config configs/imdb_qualitative.yaml
python producer.py  --config configs/imdb_qualitative.yaml
```

**Reasoning + CoT (GSM8K):**
```bash
python inference.py --config configs/gsm8k_qualitative.yaml
python trainer.py   --config configs/gsm8k_qualitative.yaml
python producer.py  --config configs/gsm8k_qualitative.yaml
```

> **First run with `alpaca_qualitative.yaml`:** MiniLM (~90 MB) is downloaded from HuggingFace and cached. You'll see: `Loading sentence embedding model 'all-MiniLM-L6-v2' on CPU...`. Subsequent runs use the local cache instantly.

---

### Interpreting the Metrics

All qualitative metrics in the CSV are prefixed with `qual_` to distinguish them from quantitative metrics.

#### Universal Metrics (all three strategies)

| Metric | Range | Healthy value |
|---|---|---|
| `qual_non_empty_rate` | 0.0–1.0 | Should stay at 1.0; drops → training instability or mode collapse |
| `qual_mean_response_length` | words | Should increase or stay stable; sharp drops → collapse |
| `qual_repetition_rate` | 0.0–1.0 | Should stay below 0.10; spikes → "the the the" degeneration |

#### Semantic Similarity (chat config)

| Metric | Range | Healthy value |
|---|---|---|
| `qual_semantic_similarity` | 0.0–1.0 | Rising trend. Untrained baseline ~0.1–0.2; well-trained model should reach 0.5+ |

#### Keyword Density (domain config)

| Metric | Range | Healthy value |
|---|---|---|
| `qual_keyword_density` | 0.0–1.0 | Rising. Untrained ~0.005–0.01; adapted model should reach 0.05+ |
| `qual_type_token_ratio` | 0.0–1.0 | Should stay above 0.5; drops → repeated generation |
| `qual_hapax_ratio` | 0.0–1.0 | For rich writing: 0.5–0.8 |

#### Structural CoT (reasoning config)

| Metric | Range | Healthy value |
|---|---|---|
| `qual_cot_anchor_count_mean` | 0–unbounded | Rising trend; untrained ~0, well-trained CoT model should reach 3–6 |
| `qual_cot_step_length_mean` | chars | Rising; short values (<20 chars) = anchors without reasoning content |
| `qual_cot_coverage_rate` | 0.0–1.0 | Should approach 1.0 as training progresses |

---

## Output Directory Structure

After a training run, the `output/` directory looks like this:

```
output/<project_output_dir>/
│
├── checkpoints/                              ← LoRA adapter snapshots
│   └── <model>__<dataset>/                  ← e.g., distilgpt2__stanfordnlp_imdb
│       ├── step_000100/
│       │   ├── adapter_model.safetensors
│       │   ├── adapter_config.json
│       │   └── checkpoint_meta.json         ← step, timestamp, model, loss
│       ├── step_000200/
│       └── final/                           ← always present after training
│
├── logs/                                     ← Inline training metrics
│   └── <project-name>/
│       └── 20260412-143500_a3f2/            ← timestamp + UUID suffix (never collides)
│           ├── metrics.csv                  ← all metrics, one row per eval event
│           ├── run_params.json              ← config snapshot
│           └── *.png                        ← auto-generated plots
│
└── eval_results/                             ← Decoupled evaluation results
    └── <model>__<dataset>/
        └── final/                           ← or step_000200/, etc.
            └── eval_20260412-150230_a3f2/   ← timestamp + UUID (never overwrites)
                ├── eval_results.json
                ├── eval_config.json
                └── plots/
                    └── *.png
```

**No-overwrite guarantees:**
- **Training logs:** Each run gets a unique `<timestamp>_<uuid4[:4]>` directory
- **Step checkpoints:** Existing directories are skipped (never overwritten)
- **Final checkpoint:** Always overwrites (represents latest training endpoint)
- **Eval results:** Each `evaluate.py` run creates a new timestamped+UUID directory

---

## Regenerating Plots

Plots are automatically generated at the end of every training run. To regenerate from an existing CSV:

```bash
python utils/plot_metrics.py "output/imdb/logs/infinitune-imdb-sentiment/20260412-143500_a3f2/metrics.csv"

# Save plots to a different directory
python utils/plot_metrics.py path/to/metrics.csv --out-dir ./my_plots
```

Available plot files (generated if the corresponding data column is non-empty):

| Plot file | What it shows |
|---|---|
| `train_loss.png` | Training loss over steps |
| `eval_loss.png` | Evaluation loss |
| `perplexity.png` | Model perplexity |
| `accuracy.png` | Classification accuracy |
| `f1.png` | Macro F1 score |
| `mcc.png` | Matthews Correlation Coefficient |
| `kappa.png` | Cohen's Kappa |
| `exact_match.png` | Exact match rate (generative tasks) |
| `qual_semantic_similarity.png` | Semantic similarity (chat) |
| `qual_keyword_density.png` | Domain keyword adoption (domain) |
| `qual_type_token_ratio.png` | Lexical diversity (domain) |
| `qual_cot_anchor_count.png` | CoT anchor count (reasoning) |
| `qual_cot_step_length.png` | Reasoning step length (reasoning) |
| `qual_cot_coverage.png` | Fraction of responses with anchors (reasoning) |
| `qual_mean_response_length.png` | Response length trend — collapse detector (all) |
| `qual_repetition_rate.png` | Bigram repetition — degeneration detector (all) |
| `qual_non_empty_rate.png` | Training stability indicator (all) |

---

## `testing_strategy` Config Block Reference

Add this block to any config to enable qualitative evaluation. All method-specific fields must be present in the YAML; set unused ones to `null`:

```yaml
testing_strategy:
  enabled: true                         # Master switch (false = zero overhead)
  method: "structural_cot"              # semantic_similarity | keyword_density | structural_cot
  eval_interval: 50                     # Qualitative eval every N optimizer steps
  eval_samples: 20                      # Samples evaluated per window (sliding)
  eval_pool_size: 100                   # Total eval samples loaded at startup (≥ eval_samples)
  max_new_tokens: 250                   # Max tokens generated per eval sample

  # Superset schema: null out fields that don't apply to your chosen method
  sentence_model: null                  # semantic_similarity: HuggingFace model ID
  keywords: null                        # keyword_density: list of domain keywords
  logic_anchors:                        # structural_cot: list of regex strings
    - "First[,\\s]"
    - "Therefore[,\\s]"
    - "Step\\s*\\d+[:\\.]"
    - "####\\s*\\d+"
```