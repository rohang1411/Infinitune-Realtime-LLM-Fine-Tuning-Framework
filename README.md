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
- **Multi-task** — pre-built configs for IMDB sentiment classification and GSM8K math reasoning; easy to extend

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
                                   └─────────────────┘
```

**Data flow:**
1. `producer.py` reads a dataset, applies filtering/templating, and publishes samples to the `training-data` Kafka topic.
2. `trainer.py` consumes samples from Kafka, runs a forward+backward pass, and every N seconds pushes updated LoRA adapter weights to the `lora-updates` Kafka topic.
3. `inference.py` listens to the `lora-updates` topic in a background thread and hot-applies weight updates to the live model, while serving generation requests on a Flask REST API at `http://localhost:5000`.

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

Open **four separate terminals** and run each command in order:

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

| Config | Task | Dataset |
|---|---|---|
| `configs/imdb_config.yaml` | Sentiment classification | IMDb (25k reviews) |
| `configs/gsm8k_config.yaml` | Math reasoning | GSM8K (grade school math) |

### Key Config Options

Edit the config YAML to tune the framework. The most useful fields:

```yaml
training:
  batch_size: 8                    # mini-batch size
  gradient_accumulation_steps: 4   # effective batch = batch_size × grad_accum
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