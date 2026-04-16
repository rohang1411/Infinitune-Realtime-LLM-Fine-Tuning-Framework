# IMDb Sentiment Analysis — Testing Guide
**Config:** `configs/imdb_quantitative.yaml`

---

## Overview

This pipeline fine-tunes **Qwen2.5-1.5B** on the Stanford IMDb dataset to perform **binary sentiment classification**. The model learns to read a movie review and output either `positive` or `negative`.

This is the cleanest demo of quantitative learning in InfiniTune: you get hard accuracy, F1, and MCC numbers that rise visibly over training steps, providing unambiguous proof that the model is absorbing signal from the data stream.

| Property | Value |
|---|---|
| **Model** | `Qwen/Qwen2.5-1.5B` |
| **Dataset** | Stanford IMDb (~25k train / 25k test reviews) |
| **Task** | Binary sentiment classification |
| **Eval Strategy** | `class_match` (generate token → match against known labels) |
| **Estimated Runtime** | ~45 minutes (1,000 steps on M4 Pro, inline eval enabled) |
| **Checkpoints** | Every 100 steps + final |

---

## What the Model Learns

### Training Data Flow

The producer reads each IMDb review and formats it as:

```
Review: This film was a masterpiece of visual storytelling...
Sentiment:
```

The target token is ` positive` or ` negative` (with a leading space matching the response template). The trainer tokenizes this entire sequence and masks the prompt portion so that **only the sentiment token is in the loss**. The model is never rewarded for memorizing the review text — only for predicting the correct label.

### What Learning Looks Like

| Training Phase | Model Behaviour |
|---|---|
| Steps 0–50 (warmup) | Model outputs random tokens. Accuracy ~50% (random binary chance). Loss ~0.7 |
| Steps 50–200 | Model learns that short tokens like "positive"/"negative" follow "Sentiment:". Accuracy rises to 60–65%. |
| Steps 200–600 | Model picks up on strong sentiment words ("terrible", "masterpiece") in the review text. Accuracy 70–80%. |
| Steps 600–1000 | Model generalises across review styles and lengths. Peak accuracy 80–88%. |

---

## Prerequisites

1. Kafka is running on `localhost:9092`
2. Python dependencies installed: `pip install -r requirements.txt`
3. Approximately **4 GB free disk space** for HuggingFace model cache

**First-run note:** `Qwen/Qwen2.5-1.5B` (~3 GB) will be downloaded from HuggingFace on the first run and cached at `~/.cache/huggingface/`. Subsequent runs start instantly.

---

## How to Run

Open **3 terminals** in the project root. Order matters — start them top to bottom.

### Terminal 1 — Inference Server

Starts the REST API and waits for LoRA weight updates from the trainer.

```bash
python inference.py --config configs/imdb_quantitative.yaml
```

**Expected output:**
```
[INFERENCE] Loading base model: Qwen/Qwen2.5-1.5B
[INFERENCE] Flask server running on http://localhost:5000
[INFERENCE] Listening for LoRA updates on topic: lora-updates-imdb
```

### Terminal 2 — Trainer

Loads the model, sets up LoRA, and waits for data to arrive from Kafka.

```bash
python trainer.py --config configs/imdb_quantitative.yaml
```

**Wait for this line before starting the producer:**
```
>>> Start the producer now (if not already running). <<<
```

### Terminal 3 — Producer

Streams IMDb reviews into Kafka. Training begins as soon as the first batch is consumed.

```bash
python producer.py --config configs/imdb_quantitative.yaml
```

**Expected output:**
```
[PRODUCER] Loaded dataset: stanfordnlp/imdb (train split, 25000 samples)
[PRODUCER] Sending sample 1 to topic: training-data-imdb
[PRODUCER] Sending sample 2 to topic: training-data-imdb
...
```

---

## Test the Live Inference API

While training is running, open a 4th terminal and query the model in real time:

```bash
# Positive review
curl -s -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Review: This movie was an absolute masterpiece. The acting was superb.\nSentiment:"}' \
  | python3 -m json.tool
```

```bash
# Negative review
curl -s -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Review: I wasted two hours of my life on this garbage film.\nSentiment:"}' \
  | python3 -m json.tool
```

**Early in training** (step ~50): The model may return random tokens or the wrong label.  
**After step ~300**: The model should reliably return `positive`/`negative` for obvious reviews.  
**After step ~700**: The model handles subtle reviews correctly.

---

## Metrics Explained

Evaluation runs automatically every **50 steps** against a shuffled pool of **5,000 test samples**, evaluating **100 samples per window**.

### Primary Metrics

| Metric | Column in CSV | What it measures | Healthy progression |
|---|---|---|---|
| **Training Loss** | `train_loss` | Cross-entropy on the training batch | Should fall continuously from ~2.0 → ~0.3 |
| **Eval Loss** | `eval_loss` | Cross-entropy on held-out test samples | Should fall with training loss, slightly higher |
| **Perplexity** | `perplexity` | `exp(eval_loss)` — how "surprised" the model is | Falls from ~7 → ~1.5 as the model learns |
| **Accuracy** | `accuracy` | Fraction of reviews classified correctly | Rises from ~50% → 80–88% |
| **Exact Match** | `exact_match` | Same as accuracy for classification | Tracks accuracy |
| **Macro F1** | `f1` | Harmonic mean of precision/recall across both classes | Important: ensures the model isn't just predicting "positive" for everything |
| **MCC** | `mcc` | Matthews Correlation Coefficient — the most reliable single number for binary classification | Rises from ~0 → 0.6–0.75. Random baseline is 0.0. |
| **Cohen's Kappa** | `kappa` | Agreement beyond chance | Rises from ~0 → 0.6–0.7 |

### Diagnostic Metrics

| Metric | Column in CSV | What it tells you |
|---|---|---|
| **Forgetting** | `forgetting_accuracy`, `forgetting_f1` | Peak accuracy minus current accuracy. Should stay near 0. A spike means the model has forgotten previously learned patterns (catastrophic forgetting). |
| **Update Latency** | `update_latency` | Wall-clock seconds between consecutive eval events. Useful for profiling. |

### Why MCC is the Most Trustworthy Metric

Accuracy can be misleading on imbalanced datasets. If 60% of reviews are positive, a model that always outputs "positive" scores 60% accuracy while being completely useless. **MCC penalises this behaviour** — it requires the model to perform well on both classes simultaneously. An MCC of 0.6+ is strong evidence that the model has genuinely learned to discriminate sentiment.

---

## Reading the Learning Curves

Plots are saved automatically at training end to:
```
output/imdb/logs/infinitune-imdb-sentiment/<timestamp>/
```

### Loss Curve (`train_loss.png`, `eval_loss.png`)

```
Loss
2.0 │▓▓
    │  ▓▓
1.0 │    ▓▓▓
    │       ▓▓▓
0.3 │           ▓▓▓▓▓▓▓▓▓▓▓  ← converging
    └──────────────────────── Steps
       0   200  400  600  800 1000
```

- **A fast, smooth drop** means the model is clearly learning from the data stream.
- **A plateau after step 200** is normal — the easy patterns are learned first.
- **If loss spikes upward**: the learning rate may be too high, or a bad batch arrived.

### Accuracy Curve (`accuracy.png`)

- Should start near **50%** (random binary guessing) and rise steadily.
- A value that stays at 50% after 100+ steps means the model is stuck — check that label_map is correctly configured.
- **F1 should track accuracy closely.** A wide gap (e.g., accuracy 75%, F1 55%) means class imbalance — the model is biased toward one class.

### MCC Curve (`mcc.png`)

- Starts near **0** (no better than chance).
- MCC > 0.4 by step 300 is a healthy milestone.
- MCC > 0.6 by step 700 means the model is reliably discriminating both positive and negative reviews.

---

## Decoupled Evaluation (Post-Training)

By default, the framework evaluates the model inline during training. If you want maximum training speed (or prefer to evaluate later), you can disable inline evaluation natively in your config:

```yaml
evaluation:
  decoupled: true   # true = skip inline evaluation during training
```

After training completes (whether you trained with `decoupled: true` or not), you can run evaluation on any checkpoint without re-training:

```bash
# Evaluate the final checkpoint
python evaluate.py --config configs/imdb_quantitative.yaml

# Evaluate a specific step
python evaluate.py --config configs/imdb_quantitative.yaml --step 500

# Evaluate every checkpoint and compare
python evaluate.py --config configs/imdb_quantitative.yaml --all-checkpoints

# See all available checkpoints
python evaluate.py --config configs/imdb_quantitative.yaml --list
```

Results are saved to:
```
output/imdb/eval_results/Qwen2.5-1.5B__imdb/final/eval_<timestamp>/
    eval_results.json       ← all metric values (full pool, not windowed)
    eval_config.json        ← config snapshot
    plots/
        accuracy.png
        f1.png
        mcc.png
        perplexity.png
```

**Decoupled eval uses the full 5,000-sample pool** (not a 100-sample sliding window), giving you definitive, unsampled scores.

---

## Regenerating Plots

```bash
python utils/plot_metrics.py output/imdb/logs/infinitune-imdb-sentiment/<timestamp>/metrics.csv

# Optional: specify output directory
python utils/plot_metrics.py output/imdb/logs/infinitune-imdb-sentiment/<timestamp>/metrics.csv \
  --out-dir ./my_plots
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| Accuracy stuck at exactly 50% | `label_map` not resolving | Verify `label_map: {0: "negative", 1: "positive"}` in config |
| Loss not decreasing after step 100 | Learning rate too low | Try `learning_rate: 2e-4` |
| Model outputs tokens like `"neg"` instead of `"negative"` | `max_new_tokens: 6` is trimming | Already set to 6, check tokenizer output |
| Out of memory on M4 Pro | Batch size too large | Reduce `batch_size: 2` and keep `gradient_accumulation_steps: 8` |
| Kafka connection refused | Broker not running | Run `brew services start kafka` then verify with `kafka-topics --bootstrap-server localhost:9092 --list` |


## Standalone Inference via Checkpoints

If you prefer to serve the model strictly off a saved checkpoint rather than using real-time Kafka streaming, you can use the decoupled inference mode. 

**Step 1: Disable LoRA Streaming (Optional but Recommended)**
In your YAML configuration file under the `kafka` block, ensure streaming is turned off:
```yaml
kafka:
  enable_lora_streaming: false
```
This reduces networking overhead and focuses the trainer entirely on saving local checkpoints.

**Step 2: Locate your Checkpoint**
After training is complete, your adapter weights are saved under the project's `output_dir`.
- Locate the final checkpoint: `output/infinitune-imdb-sentiment/checkpoint-final`

**Step 3: Run Inference Server**
Launch `inference.py` and pass the mapped checkpoint using the `--checkpoint` flag. This natively bypasses Kafka and locks the adapter statically:
```bash
python inference.py --config configs/imdb_quantitative.yaml --checkpoint output/infinitune-imdb-sentiment/checkpoint-final
```

**Step 4: Test the Endpoint**
```bash
curl -X POST http://localhost:5000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Your test prompt here"}'
```
