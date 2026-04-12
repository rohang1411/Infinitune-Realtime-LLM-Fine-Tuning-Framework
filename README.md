# InfiniTune — Realtime LLM Fine-Tuning Framework

A distributed framework for **continuously fine-tuning Large Language Models in real time** using Kafka data streams and QLoRA (Quantized Low-Rank Adaptation). As new training data arrives, the model adapts on-the-fly and the inference server receives updated adapter weights automatically — no restarts required.

---

## Table of Contents

1. [What is InfiniTune?](#what-is-infinitune)
2. [Architecture](#architecture)
3. [Dependencies](#dependencies)
4. [Kafka Setup](#kafka-setup)
   - [macOS (KRaft mode — No Zookeeper)](#macos-kraft-mode--no-zookeeper)
   - [Windows (KRaft mode — No Zookeeper)](#windows-kraft-mode--no-zookeeper)
   - [Windows (Legacy — with Zookeeper)](#windows-legacy--with-zookeeper)
5. [Running InfiniTune](#running-infinitune)
6. [Qualitative Evaluation Suite](#qualitative-evaluation-suite)
   - [Overview](#overview)
   - [The Three Proxy Strategies](#the-three-proxy-strategies)
   - [Running the Qualitative Configs](#running-the-qualitative-configs)
   - [Interpreting the Metrics](#interpreting-the-metrics)
   - [Reading the CSV and Plots](#reading-the-csv-and-plots)

---

## What is InfiniTune?

Traditional fine-tuning requires a static dataset, an offline training run, and a manual deployment step. **InfiniTune removes all three bottlenecks.**

It is built around three decoupled services that communicate over Kafka:

| Service | Script | Role |
|---|---|---|
| **Producer** | `producer.py` | Streams training samples from a HuggingFace dataset to a Kafka topic |
| **Trainer** | `trainer.py` | Consumes data from Kafka, fine-tunes the model with QLoRA, and pushes updated LoRA adapter weights back to Kafka |
| **Inference Server** | `inference.py` | Loads the base model + LoRA adapter, serves a REST API, and hot-swaps adapter weights in real time as the trainer pushes updates |

Key properties:
- **Online (streaming) learning** — the model improves continuously as data flows in
- **Memory-efficient** — uses LoRA adapters (only a fraction of model params are trained)
- **Config-driven** — all hyperparameters, dataset settings, and topology are defined in a single YAML file
- **Multi-task** — pre-built configs for IMDb, GSM8K, UltraChat, and qualitative evaluation variants
- **Dual evaluation** — quantitative metrics (Accuracy, Perplexity, Exact Match) AND qualitative proxy metrics (Semantic Similarity, Keyword Density, CoT Structure)

---

## Architecture

```
┌─────────────┐   training data    ┌───────────────────┐   LoRA weights    ┌───────────────────┐
│  Producer   │ ─────────────────► │  Kafka Broker     │ ────────────────► │  Inference Server │
│ producer.py │                    │  (localhost:9092)  │                   │  inference.py     │
└─────────────┘                    └────────┬──────────┘                   └───────────────────┘
                                            │ training data
                                            ▼
                                   ┌─────────────────┐
                                   │    Trainer       │
                                   │   trainer.py     │
                                   │  (QLoRA + AdamW) │
                                   │                  │
                                   │  ┌────────────┐  │
                                   │  │ Quant.Eval │  │  ← eval_metrics_train.py
                                   │  └────────────┘  │
                                   │  ┌────────────┐  │
                                   │  │ Qual. Eval │  │  ← eval_qualitative.py
                                   │  └────────────┘  │
                                   └─────────────────┘
```

**Data flow:**
1. `producer.py` reads a dataset, applies filtering/templating, and publishes samples to the `training-data` Kafka topic.
2. `trainer.py` consumes samples from Kafka, runs a forward+backward pass, and every N seconds pushes updated LoRA adapter weights to the `lora-updates` Kafka topic.
3. `inference.py` listens to the `lora-updates` topic in a background thread and hot-applies weight updates to the live model, while serving generation requests on a Flask REST API at `http://localhost:5000`.
4. Both quantitative and qualitative evaluation run **inside the trainer** on configurable intervals, writing results to a unified CSV log.

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

`requirements.txt` includes:

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

> **`sentence-transformers`** is only required for the `semantic_similarity` qualitative strategy (`config_qualitative_chat.yaml`). The other qualitative configs (`keyword_density`, `structural_cot`) use pure string operations with zero extra dependencies. The import is guarded — if the package is absent, only the semantic similarity strategy will fail at runtime.

> **macOS (Apple Silicon):** PyTorch MPS backend is used automatically when CUDA is unavailable. No extra steps needed.

---

## Kafka Setup

InfiniTune uses Kafka as the data backbone. Kafka 3.3+ supports **KRaft mode** (no Zookeeper required), which is the recommended approach.

### macOS (KRaft mode — No Zookeeper)

#### 1. Install Kafka

```bash
brew install kafka
```

#### 2. Add Kafka to PATH

Add this line to your `~/.zshrc` (or `~/.bashrc`):

```bash
export PATH="/opt/homebrew/opt/kafka/bin:$PATH"
```

Then apply it:

```bash
source ~/.zshrc
```

#### 3. Format Storage (One-time only)

This initialises the KRaft metadata log. Only needed on first use.

```bash
# Generate a unique cluster ID
KAFKA_CLUSTER_ID="$(kafka-storage random-uuid)"

# Format the storage directory
kafka-storage format -t $KAFKA_CLUSTER_ID -c /opt/homebrew/etc/kafka/server.properties
```

> **Note:** If you get `"Log directory is already formatted"`, you can skip this step — the storage is already initialised.

#### 4. Start Kafka

**Option A — Background service (recommended):**
```bash
# Stop Zookeeper if it was previously started
brew services stop zookeeper

# Start Kafka as a background service
brew services start kafka
```

**Option B — Foreground (useful for debugging, see live logs):**
```bash
kafka-server-start /opt/homebrew/etc/kafka/server.properties
```

#### 5. Verify

```bash
kafka-topics --bootstrap-server localhost:9092 --list
```

If Kafka is running, you'll see a (possibly empty) list of topics with no errors.

#### Stop Kafka

```bash
brew services stop kafka
```

---

### Windows (KRaft mode — No Zookeeper)

Kafka 3.3+ on Windows also supports KRaft mode. This is the recommended approach.

#### 1. Install Java

Download and install **JDK 11+** from [adoptium.net](https://adoptium.net/) or [Oracle](https://www.oracle.com/java/technologies/downloads/).

Set the environment variable:
```
JAVA_HOME = C:\Program Files\Eclipse Adoptium\jdk-21.x.x  (or your JDK path)
```

#### 2. Download Kafka

1. Go to [kafka.apache.org/downloads](https://kafka.apache.org/downloads)
2. Download the latest binary (`.tgz`) and extract it to `C:\kafka`

#### 3. Format Storage (One-time only)

Open a Command Prompt and run:

```bat
cd C:\kafka

.\bin\windows\kafka-storage.bat random-uuid
```

Copy the UUID printed, then format the storage:

```bat
.\bin\windows\kafka-storage.bat format -t <YOUR_UUID_HERE> -c .\config\server.properties
```

#### 4. Start Kafka

Open a Command Prompt:

```bat
cd C:\kafka
.\bin\windows\kafka-server-start.bat .\config\server.properties
```

Keep this window open while using InfiniTune.

#### 5. Verify

Open a new Command Prompt:

```bat
cd C:\kafka
.\bin\windows\kafka-topics.bat --bootstrap-server localhost:9092 --list
```

---

### Windows (Legacy — with Zookeeper)

> Use this only if you are on Kafka < 3.3 or require Zookeeper for another reason.

#### 1. Start Zookeeper

```bat
cd C:\kafka
.\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties
```

#### 2. Start Kafka (in a new terminal)

```bat
cd C:\kafka
.\bin\windows\kafka-server-start.bat .\config\server.properties
```

> **Troubleshooting:** If you encounter issues with `wmic` on Windows 11, consider using **WSL 2** or **Docker** for better compatibility.

---

## Running InfiniTune

### 1. Prerequisites

Make sure:
- Kafka is running on `localhost:9092`
- Python dependencies are installed (`pip install -r requirements.txt`)
- You have chosen a config file from `configs/` (e.g., `configs/imdb_config.yaml`)

### 2. Launch the Services

Open **three separate terminals** and run each command in order:

**Terminal 1 — Start the Inference Server first** *(starts listening for weight updates)*
```bash
python inference.py --config configs/imdb_config.yaml
```

**Terminal 2 — Start the Trainer** *(loads model, waits for data from Kafka)*
```bash
python trainer.py --config configs/imdb_config.yaml
```

**Terminal 3 — Start the Producer** *(streams training data into Kafka)*
```bash
python producer.py --config configs/imdb_config.yaml
```

> The trainer logs `>>> Start the producer now (if not already running). <<<` when it is ready. Start the producer after you see this.

### 3. Test the Inference API

**Terminal 4 — Send a generation request:**
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Review: This movie was absolutely terrible.\nSentiment:"}'
```

**Health check:**
```bash
curl http://localhost:5000/health
```

Expected response: `{"status": "ok"}`

### Available Configs

| Config | Task | Dataset | Eval Type |
|---|---|---|---|
| `configs/imdb_config.yaml` | Sentiment classification | IMDb (25k reviews) | Quantitative (Accuracy, F1) |
| `configs/gsm8k_config.yaml` | Math reasoning | GSM8K (grade school math) | Quantitative (Exact Match) |
| `configs/config_qualitative_chat.yaml` | Instruction following | Alpaca (52k) | Qualitative (Semantic Similarity) + Perplexity |
| `configs/config_qualitative_domain.yaml` | Domain adaptive generation | IMDb (unconditional LM) | Qualitative (Keyword Density + TTR) + Perplexity |
| `configs/config_qualitative_reasoning.yaml` | Math reasoning + CoT structure | GSM8K | Qualitative (CoT Adherence) + Quantitative (Exact Match) |

### Key Config Options

Edit the config YAML to tune the framework. The most useful fields:

```yaml
training:
  batch_size: 8                    # mini-batch size
  gradient_accumulation_steps: 4   # effective batch = batch_size x grad_accum
  learning_rate: 1e-4
  max_steps: 2000
  lr_scheduler:
    type: "cosine_with_warmup"     # cosine_with_warmup | linear | constant
    warmup_steps: 50
    T_max: 1000

lora:
  r: 8                             # adapter rank (higher = more capacity, more memory)
  alpha: 64                        # scaling factor (effective LR multiplier = alpha/r)
  target_modules: [c_attn, c_proj] # which layers to adapt

evaluation:
  eval_interval: 50                # evaluate every N steps
  eval_pool_size: 5000             # total eval samples loaded at startup
  eval_batch_size: 100             # samples evaluated per step (sliding window)
  verbose: false                   # set to true to print per-sample predictions
```

---

## Qualitative Evaluation Suite

### Overview

InfiniTune includes a **Qualitative Evaluation Suite** that measures improvement in tone, style, and reasoning structure — without any LLM API calls and with minimal RAM overhead. This is particularly useful when the model is learning to *write* rather than *classify*, where traditional accuracy metrics are meaningless.

**Design constraints met:**
- Zero LLM API calls (no GPT-4, no Claude)
- All proxy metrics run entirely on CPU
- The SentenceTransformers model (for semantic similarity) is explicitly pinned to `device="cpu"` — it never touches GPU VRAM
- Total extra RAM for qualitative eval: ~90 MB (MiniLM model) for semantic similarity only; 0 extra MB for the other two strategies
- Non-blocking: qualitative eval runs at its own `testing_strategy.eval_interval` cadence, fully independent from the quantitative eval loop

The implementation follows the **Strategy Pattern**: `utils/eval_qualitative.py` contains an abstract base class (`QualitativeMetric`) and three concrete strategy classes. A `QualitativeEvaluator` factory reads the `testing_strategy.method` field from the YAML config and instantiates the correct class at startup.

---

### The Three Proxy Strategies

#### 1. Semantic Similarity — `config_qualitative_chat.yaml`

**Dataset:** `tatsu-lab/alpaca` (52k instruction-following examples)
**Model:** `sentence-transformers/all-MiniLM-L6-v2` (CPU, ~90 MB)

The model generates a response to an instruction prompt. That response is compared to the golden ground-truth output from the dataset using **cosine similarity of sentence embeddings**.

```
Instruction ─► model generates response
                        |
Generated: "Python is a programming language..."
Reference: "Python is a high-level language used for..."
                        |
         Cosine similarity = 0.73
```

A higher similarity score over training steps proves the model is learning to produce semantically aligned instruction-following responses.

**Blind spots:** Does not measure fluency or grammar. A semantically similar but grammatically broken answer scores the same as a polished one.

---

#### 2. Keyword Density + TTR — `config_qualitative_domain.yaml`

**Dataset:** `imdb` (unconditional language modeling — model learns to *write* reviews)
**Reference:** None required (reference-free)

The model is prompted with `"Write a movie review:\n"` and generates freely. The output is analysed for:
- **Keyword Density**: fraction of words that are domain-specific film criticism vocabulary (e.g., "cinematography", "screenplay", "pacing")
- **Type-Token Ratio (TTR)**: unique words / total words — measures lexical diversity
- **Hapax Ratio**: fraction of words used exactly once — a finer diversity measure for longer texts

```
Generated: "The cinematography was stunning. The pacing felt off
            though the screenplay had some brilliant moments..."
                              |
keyword_density  = 0.068   (6.8% domain keywords)
type_token_ratio = 0.71    (71% unique words)
hapax_ratio      = 0.58
```

**Note on this config vs `imdb_config.yaml`:** `imdb_config.yaml` trains for *sentiment classification* (predicts "positive"/"negative"). `config_qualitative_domain.yaml` trains the model for *unconditional language modeling* — it learns to write movie reviews from scratch. These are fundamentally different tasks.

---

#### 3. Structural CoT Adherence — `config_qualitative_reasoning.yaml`

**Dataset:** `gsm8k` (same task as `gsm8k_config.yaml`)
**Complements:** quantitative `exact_match` from the `evaluation` block

The model generates a math reasoning chain. The output is analysed for **"logic anchors"** — regex patterns that mark structured reasoning steps:

```
Generated: "First, we need to find the total apples.
            Step 1: There are 5 apples per basket, and 3 baskets.
            Therefore, 5 x 3 = 15 apples total.
            The answer is #### 15"
                     |
cot_anchor_count_mean = 4.0    (4 anchors found)
cot_step_length_mean  = 38.5   (avg 38.5 chars between anchors)
cot_coverage_rate     = 1.0    (100% of responses had at least 1 anchor)
```

This metric is **additive** to quantitative evaluation — `exact_match` tells you if the final answer is right; structural CoT tells you if the reasoning process is structured. A model achieving both is genuinely learning chain-of-thought reasoning, not just pattern-matching the final number.

---

### Running the Qualitative Configs

All qualitative configs follow the same three-terminal launch pattern:

**Conversational (UltraChat):**
```bash
# Terminal 1
python inference.py --config configs/config_qualitative_chat.yaml

# Terminal 2
python trainer.py --config configs/config_qualitative_chat.yaml

# Terminal 3
python producer.py --config configs/config_qualitative_chat.yaml
```

**Domain Adaptation (IMDb unconditional LM):**
```bash
python inference.py --config configs/config_qualitative_domain.yaml
python trainer.py  --config configs/config_qualitative_domain.yaml
python producer.py --config configs/config_qualitative_domain.yaml
```

**Reasoning + CoT (GSM8k):**
```bash
python inference.py --config configs/config_qualitative_reasoning.yaml
python trainer.py  --config configs/config_qualitative_reasoning.yaml
python producer.py --config configs/config_qualitative_reasoning.yaml
```

> **First run with `config_qualitative_chat.yaml`:** The MiniLM model (~90 MB) will be downloaded from HuggingFace and cached locally on the first qualitative eval call. You will see: `[QUAL_EVAL] Loading sentence embedding model 'sentence-transformers/all-MiniLM-L6-v2' on CPU...`. Subsequent runs use the local cache instantly.

---

### Interpreting the Metrics

All qualitative metrics in the CSV and log output are prefixed with `qual_` to distinguish them from quantitative metrics at a glance.

#### Universal Metrics (present for all three strategies)

| Metric | Range | What it means | Healthy value |
|---|---|---|---|
| `qual_non_empty_rate` | 0.0 – 1.0 | Fraction of eval prompts where the model generated at least one token | Should stay at 1.0; drops indicate training instability or mode collapse |
| `qual_mean_response_length` | words | Average word count of generated responses | Should increase or stay stable; sharp drops indicate collapse |
| `qual_repetition_rate` | 0.0 – 1.0 | Fraction of bigrams that repeat within a single response | Should stay below 0.10; spikes indicate degenerate "the the the" patterns |

#### Semantic Similarity (chat config only)

| Metric | Range | What it means | Healthy value |
|---|---|---|---|
| `qual_semantic_similarity` | 0.0 – 1.0 | Mean cosine similarity between generated and golden responses | Rising trend. Raw distilgpt2 baseline ~0.1–0.2; well-trained model should reach 0.5+ |

**Reading the trend:** If `qual_semantic_similarity` plateaus early, the model may be learning the response format but not the content. Check `qual_mean_response_length` — if both plateau together, consider increasing `lora.r` to add capacity.

#### Keyword Density (domain config only)

| Metric | Range | What it means | Healthy value |
|---|---|---|---|
| `qual_keyword_density` | 0.0 – 1.0 | Fraction of output words that are domain keywords | Rising trend; untrained baseline ~0.005–0.01, well-adapted model should reach 0.05+ |
| `qual_type_token_ratio` | 0.0 – 1.0 | Unique words / total words | Should stay above 0.5; drops indicate repeated generation |
| `qual_hapax_ratio` | 0.0 – 1.0 | Fraction of words used exactly once per response | For rich writing, expect 0.5–0.8 |

**Reading the trend:** Rising `qual_keyword_density` without falling `qual_type_token_ratio` is the ideal signal — domain vocabulary is being added while maintaining lexical diversity.

#### Structural CoT (reasoning config only)

| Metric | Range | What it means | Healthy value |
|---|---|---|---|
| `qual_cot_anchor_count_mean` | 0 – unbounded | Mean logic anchors per response | Rising trend; untrained ~0, well-trained CoT model should reach 3–6 |
| `qual_cot_step_length_mean` | chars | Mean characters between anchors | Rising trend; short values (<20 chars) mean anchors without actual reasoning |
| `qual_cot_coverage_rate` | 0.0 – 1.0 | Fraction of responses with at least one anchor | Should approach 1.0 as training progresses |

**Reading the trend:** `qual_cot_anchor_count_mean` rising in lock-step with `qual_cot_step_length_mean` is the key signal — structural adoption with actual reasoning content. Combined with rising `exact_match` from the quantitative block, this is proof of genuine CoT learning.

---

### Reading the CSV and Plots

#### CSV Location

After a training run, metrics are saved to:

```
output/<run_name>/logs/<run_name>/<timestamp>/metrics.csv
```

The CSV has one row per evaluation event. Quantitative and qualitative evaluations each write their own rows (columns that don't apply to that eval type are left empty). Example:

```
step | eval_loss | accuracy | ... | qual_cot_anchor_count_mean | qual_cot_step_length_mean
50   | 3.21      | 0.54     | ... |                            |
50   |           |          | ... | 1.2                        | 18.4
100  | 3.05      | 0.58     | ... |                            |
100  |           |          | ... | 2.8                        | 24.1
```

#### Regenerating Plots

Plots are automatically generated at the end of every training run. To regenerate from an existing CSV:

```bash
python utils/plot_metrics.py "output/infinitune-qual-reasoning/logs/.../metrics.csv"

# Save to a specific directory
python utils/plot_metrics.py path/to/metrics.csv --out-dir ./my_plots
```

Qualitative plots are automatically included when the CSV contains the corresponding columns. The following qualitative plot files will be generated:

| Plot file | What it shows |
|---|---|
| `qual_semantic_similarity.png` | Semantic similarity trend (chat) |
| `qual_keyword_density.png` | Domain keyword adoption (domain) |
| `qual_type_token_ratio.png` | Lexical diversity (domain) |
| `qual_hapax_ratio.png` | Word uniqueness (domain) |
| `qual_cot_anchor_count.png` | CoT anchor count (reasoning) |
| `qual_cot_step_length.png` | Reasoning step length (reasoning) |
| `qual_cot_coverage.png` | Fraction of responses with anchors (reasoning) |
| `qual_mean_response_length.png` | Response length trend — collapse detector (all) |
| `qual_repetition_rate.png` | Bigram repetition — degeneration detector (all) |
| `qual_non_empty_rate.png` | Training stability indicator (all) |

#### The `testing_strategy` Config Block

Add the `testing_strategy` block to any config to enable qualitative evaluation. All three method-specific fields must be present; set unused ones to `null`:

```yaml
testing_strategy:
  enabled: true                         # Master switch (false = zero overhead)
  method: "structural_cot"              # semantic_similarity | keyword_density | structural_cot
  eval_interval: 50                     # Qualitative eval every N optimizer steps
  eval_samples: 20                      # Samples per eval window (sliding)
  max_new_tokens: 250                   # Max tokens generated per eval sample

  # Superset schema: null out fields that don't apply
  sentence_model: null                  # For semantic_similarity: HuggingFace model ID
  keywords: null                        # For keyword_density: list of domain keywords
  logic_anchors:                        # For structural_cot: list of regex strings
    - "First[,\\s]"
    - "Therefore[,\\s]"
    - "Step\\s*\\d+[:\\.]"
```