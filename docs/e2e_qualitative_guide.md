# E2E NLG Domain Adaptation — Testing Guide
**Config:** `configs/e2e_qualitative.yaml`

---

## Overview

This pipeline fine-tunes **DistilGPT-2** on the [E2E NLG Challenge](https://huggingface.co/datasets/e2e_nlg) dataset in **structured slot-to-text generation** mode. The model learns to convert structured meaning representations (MRs) — key-value slot pairs describing a restaurant — into fluent natural language descriptions.

Unlike sentiment classification or review generation, this task has **verifiable targets**: each MR has reference sentences that mention specific slots (name, cuisine, price range, area, rating). We measure whether the model reliably surfaces those slots in its output.

| Property | Value |
|---|---|
| **Model** | `distilgpt2` |
| **Dataset** | E2E NLG Challenge (~33,525 train samples) |
| **Task** | Structured data-to-text NLG (MR → restaurant description) |
| **Eval (Quant)** | `perplexity` + `answer_overlap_f1` (token-level F1 vs. reference) |
| **Eval (Qual)** | `structured_slot_coverage` — slot coverage + consistency score |
| **Estimated Runtime** | ~3–4 hours on Colab T4 (2,100 steps over 33.5k samples) |
| **Checkpoints** | Every 200 steps + final |

---

## Dataset Format

Each E2E sample contains:
- **`mr`**: A meaning representation string — e.g., `"name[The Golden Palace], food[Chinese], priceRange[cheap], area[city centre], familyFriendly[yes]"`
- **`human_reference`**: One or more reference sentences — e.g., `"The Golden Palace is a cheap Chinese restaurant in the city centre and is family-friendly."`

### What the MR Looks Like After Parsing

The producer formats each sample using the Jinja2 prompt template:

```
Generate a restaurant description for the following attributes:
name[The Golden Palace], food[Chinese], priceRange[cheap], area[city centre], familyFriendly[yes]

Description:
 The Golden Palace is a cheap Chinese restaurant in the city centre and is family-friendly.
```

The model's task: given the structured slots, produce a grammatically fluent, slot-faithful sentence.

---

## What the Model Learns

### Training Data Flow

The producer streams all 33,525 training samples through Kafka with a 50ms inter-message interval (~28 minutes of streaming). The trainer reads in batches of 8, performing gradient accumulation every 2 steps (effective batch size = 16).

**Key training progression:**

| Training Phase | Expected Model Behaviour |
|---|---|
| Steps 0–100 | Generic sentence completions. Slots poorly surfaced. Slot coverage <30%. |
| Steps 200–500 | Model begins echoing slot values. Sentence fluency irregular. Coverage rising. |
| Steps 500–1000 | Consistent slot coverage >70%. Sentences grammatically coherent. |
| Steps 1000–2100 | High consistency (>0.75). Near-complete slot coverage. Fluent template following. |

### Evaluation Signal Explained

**`answer_overlap_f1`** (quantitative, every 50 steps):
Token-level F1 between the model's generated output and the reference `human_reference`. Measures lexical overlap. Expected range: 0.1 (baseline) → 0.5+ (well-trained).

**`qual_slot_coverage_mean`** (qualitative, every 200 steps):
Fraction of MR slots that appear in the model's output. The system dynamically parses `name[X]` style slots from the MR and checks if the value `X` appears in the generated text. Expected range: 0.2 → 0.85+.

**`qual_consistency_score_mean`** (qualitative, every 200 steps):
Average pairwise semantic similarity across `consistency_runs=10` independent generations for the same MR. Measures whether the model produces **repeatable, non-random** outputs. Expected range: 0.3 → 0.8+.

---

## ⚠️ Important Operational Notes

### Producer / Trainer Startup Order

**Either order works.** The trainer uses a unique consumer group per run (`auto_offset_reset=earliest`). Whether the producer starts first or the trainer starts first, all 33,525 messages will be consumed.

> Unlike previous versions, the trainer no longer calls `seek_to_end()`. This was the root cause of earlier runs consuming only ~12,900 of 33,525 samples.

### Qualitative Eval Duration

At step 200 (and every 200 steps thereafter), the system runs:
- `eval_samples=50` generations × `consistency_runs=10` = **500 total model.generate() calls**

At ~2–3 seconds per call on a Colab T4, this takes **~15–25 minutes**. The Kafka consumer is configured with `max_poll_interval_ms=1800000` (30 minutes) to survive this pause without being kicked from the group. **Do not reduce `consistency_runs` below 3** or the consistency score becomes unreliable.

### Kafka Consumer Group

Each run uses a unique group ID: `trainer-group-<8-char-hex>`. This prevents stale committed offsets from previous incomplete runs. You do not need to reset the consumer group between runs.

---

## Config Walkthrough (`e2e_qualitative.yaml`)

### Dataset

```yaml
dataset:
  name: "e2e_nlg"
  split: "train"
  input_col: "mr"
  target_col: "human_reference"
  test_split: "test"
  test_size: 0.05      # ~1,676 samples held for final testing
```

The `test_split` reserves 5% of the HuggingFace test partition for final evaluation. Training only sees `split: train`.

### Training

```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 2   # Effective batch = 16
  learning_rate: 3e-4
  max_steps: 2100                   # Safety cap; EOF will stop training first
  optimizer: "adamw"
  scheduler: "cosine"
  warmup_steps: 50
  T_max: 2100
```

### Evaluation — Quantitative

```yaml
evaluation:
  enabled: true
  strategy: "perplexity"
  eval_interval: 50                # Every 50 optimizer steps
  eval_pool_size: 500              # 500 samples in the eval pool
  eval_batch_size: 50
  metrics:
    compute_loss: true
    compute_answer_overlap_f1: true   # Token-level F1 vs. reference
```

### Evaluation — Qualitative

```yaml
testing_strategy:
  enabled: true
  method: "structured_slot_coverage"
  eval_interval: 200               # Every 200 steps (checkpoint-aligned)
  eval_samples: 50                 # 50 MRs evaluated
  consistency_runs: 10             # 10 generations per MR for consistency
```

### Checkpoints

```yaml
training:
  save_checkpoints:
    enabled: true
    save_every_steps: 200          # Aligns with qual eval interval
    save_final: true
```

Checkpoints are saved under:
```
output/checkpoints/distilgpt2__e2e_nlg/run_<YYYYMMDD-HHMMSS>_<uid>/step_000200/
```

Each run gets its own `run_*` subdirectory. Previous runs are never overwritten.

---

## Expected Output Files

After a complete training run, the following files are generated:

```
output/logs/<run_name>/<ts>_<uid>/
  metrics.csv              # All columns (many sparse for E2E perplexity config)
  metrics_clean.csv        # Only populated columns (recommended for analysis)
  run_params.json          # Full training config snapshot
  verbose_samples.md       # Markdown table of eval samples (if verbose=true)
  train_loss.png
  eval_loss.png
  perplexity.png
  answer_overlap_f1.png    # Token-level F1 trend over training
  qual_slot_coverage_mean.png
  qual_consistency_score_mean.png
```

### `metrics_clean.csv` vs. `metrics.csv`

`metrics.csv` contains all COLUMNS defined in `MetricsLogger` — most are empty for the E2E perplexity config (e.g., `accuracy`, `f1`, `mcc` are disabled). `metrics_clean.csv` is generated at the end of training and contains **only the columns with at least one value**. Use `metrics_clean.csv` for analysis and sharing.

---

## Running the Pipeline

### Step 1: Start Trainer (Colab Cell 1)

```python
import subprocess, threading

def run_trainer():
    subprocess.run(["python", "trainer.py", "--config", "configs/e2e_qualitative.yaml"])

t = threading.Thread(target=run_trainer, daemon=True)
t.start()
```

### Step 2: Start Producer (Colab Cell 2)

```python
subprocess.run(["python", "producer.py", "--config", "configs/e2e_qualitative.yaml"])
```

> **Order doesn't matter.** Either cell can run first. The trainer will consume all messages regardless of start order.

### Step 3: Monitor Progress

Watch for these log patterns:

```
[TRAINER] Test mode: unique consumer group 'trainer-group-a3f2b1c9' (no stale offset risk)
[TRAINER] Step 200: loss = 2.341, lr = 2.97e-04, grad_norm = 0.8821, tok/s = 142.3
[TRAINER] Checkpoint saved [step 200] → .../run_20260413-153020_a3f2/step_000200
[TRAINER] Slot coverage evaluation at step 200: slot_coverage_mean=0.412, consistency=0.551
[TRAINER] answer_overlap_f1: 0.2341
```

### Step 4: Evaluate Checkpoints

```bash
# Evaluate a specific checkpoint
python evaluate.py --config configs/e2e_qualitative.yaml --step 1000

# Evaluate all checkpoints from all runs
python evaluate.py --config configs/e2e_qualitative.yaml --all-checkpoints
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| Trainer stops at `~12,900` messages | Old code with `seek_to_end()` | Upgrade to current `trainer.py` |
| Consumer kicked from group mid-eval | `max_poll_interval_ms` too low | Current config sets 30 min; verify Kafka broker allows it |
| `qual_slot_coverage_mean` all zeros | `slot_keywords: null` not dynamically parsed | Check `eval_qualitative.py` version; null means dynamic parse |
| `answer_overlap_f1` not in CSV | `compute_answer_overlap_f1: false` in config | Set to `true` in `e2e_qualitative.yaml` metrics block |
| Perplexity stuck >200 | Learning rate too low or LoRA rank too small | Increase `lora.r` from 8 to 16, or `learning_rate` to 5e-4 |
| OOM during consistency runs | 500 generations too many for small GPU | Reduce `consistency_runs` to 3 (min for reliable score) |

---

## Interpreting Results

### Sample Progression

**Untrained (step 0):**
```
MR:    name[Loch Fyne], food[seafood], priceRange[£20-25], area[riverside]
Model: "Loch Fyne is a great place to eat fish and chips. The staff were ..."
Slot coverage: 1/4 (only name)
```

**Well-trained (step 1500+):**
```
MR:    name[Loch Fyne], food[seafood], priceRange[£20-25], area[riverside]
Model: "Loch Fyne is a seafood restaurant near the riverside, with prices in the £20-25 range."
Slot coverage: 4/4 (all slots)
Consistency: 0.81 (very similar outputs across 10 runs)
```

### Key Metrics Table

| Metric | Baseline (step 0) | Target (step 2100) |
|---|---|---|
| `eval_loss` | ~3.5 | <2.0 |
| `perplexity` | ~33 | <8 |
| `answer_overlap_f1` | ~0.05 | >0.45 |
| `qual_slot_coverage_mean` | ~0.2 | >0.80 |
| `qual_consistency_score_mean` | ~0.3 | >0.75 |

---

*Last updated: April 2026 — reflects InfiniTune v2 hierarchical checkpointing, offset-based EOF draining, and `metrics_clean.csv` generation.*
