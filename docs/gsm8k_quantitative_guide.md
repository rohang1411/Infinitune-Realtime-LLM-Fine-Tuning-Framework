# GSM8K Math Reasoning — Testing Guide
**Config:** `configs/gsm8k_quantitative.yaml`

---

## Overview

This pipeline fine-tunes **Qwen2.5-3B** on the GSM8K dataset (Grade School Math 8K) to solve elementary math word problems. The model learns to generate a step-by-step solution and produce a final numeric answer in the format `#### <number>`.

This is the hardest quantitative task in InfiniTune. Math reasoning requires internal arithmetic capacity that smaller models simply lack — which is why this config uses the 3B model. A rising Exact Match score is one of the most compelling demonstrations that a model is actively learning from a data stream.

| Property | Value |
|---|---|
| **Model** | `Qwen/Qwen2.5-3B` |
| **Dataset** | GSM8K `main` split (~7.5k train / 1.3k test) |
| **Task** | Math word problem solving (generate full solution + final answer) |
| **Eval Strategy** | `regex_extract` — extract `#### <number>` from output and compare to gold |
| **Estimated Runtime** | ~60 minutes (500 steps on M4 Pro, inline eval enabled) |
| **Checkpoints** | Every 100 steps + final |

---

## What the Model Learns

### Training Data Flow

Each GSM8K problem is formatted as:

```
Question: Natalia sold clips to 48 of her friends in April, and then she sold
half as many clips in May. How many clips did Natalia sell altogether in April
and May?
Answer:  Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72
```

The **entire answer** — including intermediate reasoning steps and the `#### 72` final answer marker — is set as the target. The model must learn to:

1. Parse the question and identify the numerical quantities involved.
2. Generate intermediate calculation steps in natural language.
3. Produce the final answer in the exact format `#### <number>` so the regex evaluator can extract it.

The loss is computed over the **answer tokens only**. The model is trained to predict the full solution given the question, not just the final number.

### What Learning Looks Like

| Training Phase | Model Behaviour |
|---|---|
| Steps 0–50 (warmup) | Output is random text. Exact match ~0%. Eval loss is high (~4–5). |
| Steps 50–150 | Model learns to produce `####` as a terminal sequence. Exact match jumps to 2–5%. |
| Steps 150–300 | Model starts generating plausible arithmetic. Exact match reaches 10–20%. |
| Steps 300–500 | Model generalises over question structures. Exact match 25–45% (remarkable for online learning). |

> **Note:** Even 25% exact match on GSM8K represents strong learning — this is a hard benchmark. Full fine-tuning of 7B models typically achieves 40–60% after many epochs.

---

## Prerequisites

1. Kafka is running on `localhost:9092`
2. Python dependencies installed: `pip install -r requirements.txt`
3. Approximately **7 GB free disk space** for HuggingFace model cache (Qwen2.5-3B is ~6 GB)

**First-run note:** `Qwen/Qwen2.5-3B` (~6 GB) will be downloaded and cached on first run. Allow 10–15 minutes for download depending on your connection speed.

---

## How to Run

Open **3 terminals** in the project root in order:

### Terminal 1 — Inference Server

```bash
python inference.py --config configs/gsm8k_quantitative.yaml
```

**Expected output:**
```
[INFERENCE] Loading base model: Qwen/Qwen2.5-3B
[INFERENCE] Flask server running on http://localhost:5000
[INFERENCE] Listening for LoRA updates on topic: lora-updates-gsm8k
```

### Terminal 2 — Trainer

```bash
python trainer.py --config configs/gsm8k_quantitative.yaml
```

**Wait for this line before starting the producer:**
```
>>> Start the producer now (if not already running). <<<
```

### Terminal 3 — Producer

```bash
python producer.py --config configs/gsm8k_quantitative.yaml
```

**Expected output:**
```
[PRODUCER] Loaded dataset: gsm8k (train split, 7473 samples)
[PRODUCER] Sending sample 1 to topic: training-data-gsm8k
...
```

---

## Test the Live Inference API

Query the model mid-training to watch it improve in real time:

```bash
# Simple single-step problem
curl -s -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Question: James has 6 apples. He gives 2 to his sister. How many does he have left?\nAnswer:"}' \
  | python3 -m json.tool
```

```bash
# Multi-step problem
curl -s -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Question: A store sells notebooks for $3 each and pens for $1.50 each. Maria buys 4 notebooks and 6 pens. How much does she spend in total?\nAnswer:"}' \
  | python3 -m json.tool
```

**Early (step ~50):** Random output, no `####` marker present.  
**Mid-training (~250):** Generates partial reasoning, sometimes produces `#### <wrong_number>`.  
**Late (~500):** Generates structured reasoning and correct `#### <answer>` for simple problems.

---

## Metrics Explained

Evaluation runs every **50 steps** against **200 samples** from the GSM8K test split (1,319 total), evaluating **20 samples per window**.

### Primary Metrics

| Metric | Column in CSV | What it measures | Healthy progression |
|---|---|---|---|
| **Training Loss** | `train_loss` | Cross-entropy on answer tokens | Falls from ~4.0 → ~1.5 |
| **Eval Loss** | `eval_loss` | Cross-entropy on test answer tokens | Tracks training loss |
| **Perplexity** | `perplexity` | `exp(eval_loss)` — model's surprise | Falls from ~55 → ~5 |
| **Accuracy** | `accuracy` | Fraction of answers that exactly match gold after regex extraction | Rises from ~0% → 25–45% |
| **Exact Match** | `exact_match` | Same as accuracy here — the extracted `#### N` must exactly equal the gold `#### N` | The key metric to watch |
| **Update Latency** | `update_latency` | Seconds between eval events | Profiling only |

### Why F1/MCC Are Disabled

This config disables `compute_f1` and `compute_mcc`. Each problem has a unique numeric answer (e.g., "72", "144", "3"). With thousands of distinct answer classes, precision/recall per class is meaningless. **Exact match is the only honest metric for this task.**

### Understanding Exact Match vs Accuracy

For `regex_extract`, both `accuracy` and `exact_match` measure whether the extracted answer from the model's output matches the gold answer extracted from the label. They are effectively the same metric here.

The regex pattern `#### (\d+)` extracts the first number after `####`. If the model says:

```
Natalia sold 24 clips in May. Together she sold 72 clips. #### 72
```

The evaluator extracts `72` and compares it to gold `72` → **Match**.

If the model says:

```
The answer is probably 72 or 48. #### 48
```

The evaluator extracts `48` and compares to gold `72` → **No match**.

---

## Reading the Learning Curves

Plots are saved to: `output/gsm8k/logs/infinitune-gsm8k-math/<timestamp>/`

### Loss Curve

```
Loss
4.5 │▓▓
    │  ▓▓
2.5 │    ▓▓▓
    │       ▓▓
1.5 │         ▓▓▓▓▓▓▓▓▓▓▓▓  ← converging
    └──────────────────────── Steps
       0   100  200  300  400  500
```

- An initial loss of **~4–5** is normal for a cold model on math problems.
- The loss should fall steeply in the first 150 steps as the model internalises the output format.
- A persistent plateau above 2.0 suggests the model is struggling — consider reducing `batch_size` to 1 and doubling `gradient_accumulation_steps`.

### Exact Match Curve (`accuracy.png`, `exact_match.png`)

```
Exact
Match
45% │                      ▓▓
    │                 ▓▓▓▓▓
20% │           ▓▓▓▓▓
    │      ▓▓▓
 5% │  ▓▓▓
 0% │▓▓
    └──────────────────────── Steps
       0   100  200  300  400  500
```

- **Starts at exactly 0%** — a cold model cannot format `#### N` at all.
- A jump from 0% to 5%+ in the first 100 steps confirms the model has learned the output format.
- Sustained growth to 20–45% is a strong quantitative proof of learning.
- **A flat line at 0% after 150 steps** means the model never learned the format — check `max_new_tokens: 150` is large enough.

### Perplexity Curve

- Falls from ~55 to ~5. This is a large drop and is one of the most visually dramatic plots.
- Perplexity below 5 on math reasoning text is excellent for online / streaming fine-tuning.

---

## Decoupled Evaluation (Post-Training)

```bash
# Evaluate the final checkpoint on full test pool (200 samples)
python evaluate.py --config configs/gsm8k_quantitative.yaml

# Compare all checkpoints to see learning trajectory
python evaluate.py --config configs/gsm8k_quantitative.yaml --all-checkpoints

# See what's saved
python evaluate.py --config configs/gsm8k_quantitative.yaml --list
```

Results saved to:
```
output/gsm8k/eval_results/Qwen2.5-3B__gsm8k/final/eval_<timestamp>/
    eval_results.json
    plots/
        accuracy.png
        exact_match.png
        perplexity.png
```

**The `--all-checkpoints` flag is especially powerful here** — you get a CSV table and a single combined plot showing exact match as a function of step, giving a clean learning trajectory from step 100 to final.

---

## Regenerating Plots

```bash
python utils/plot_metrics.py output/gsm8k/logs/infinitune-gsm8k-math/<timestamp>/metrics.csv
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| Exact match stays at 0% after step 150 | `max_new_tokens` too small — model can't reach `####` | Increase `max_new_tokens: 250` in `inference` block |
| Loss falls but exact match stays low | Model writes correct reasoning but wrong number | Normal early-stage behaviour; more steps needed |
| OOM on 24 GB M4 Pro | 3B model in fp16 + gradient states exceeds budget | Reduce `batch_size: 1`, `gradient_accumulation_steps: 8` |
| Kafka consumer lag warning | Producer too fast | Increase `producer_send_interval: 0.2` |
| Model outputs decimals instead of integers | GSM8K gold answers are always integers | This is a model error, not a framework issue — more training needed |
