# GSM8K Reasoning + Chain-of-Thought — Testing Guide
**Config:** `configs/gsm8k_qualitative.yaml`

---

## Overview

This pipeline fine-tunes **Qwen2.5-3B** on the GSM8K dataset to develop **structured Chain-of-Thought (CoT) reasoning**. While `gsm8k_quantitative.yaml` measures whether the final numeric answer is correct, this config measures **whether the model's reasoning process itself is structured** — a deeper proof of genuine learning.

The two metrics are complementary and designed to be run together or independently:

- **Quantitative (exact match):** Did the model get the right answer?
- **Qualitative (structural CoT):** Is the model reasoning in a structured, step-by-step way?

A model that achieves both is genuinely learning Chain-of-Thought reasoning, not just pattern-matching the final digit.

| Property | Value |
|---|---|
| **Model** | `Qwen/Qwen2.5-3B` |
| **Dataset** | GSM8K `main` split (~7.5k train / 1.3k test) |
| **Task** | Math word problem solving with structured reasoning |
| **Eval (Quant)** | `regex_extract` — exact match of `#### N` |
| **Eval (Qual)** | `structural_cot` — logic anchor counting, step length analysis |
| **Estimated Runtime** | ~65 minutes (500 steps on M4 Pro, both inline eval modes active) |
| **Checkpoints** | Every 100 steps + final |

---

## What the Model Learns

### Training Data Flow

The producer formats each GSM8K sample with an explicit CoT prompt:

```
Question: A store had 120 apples. They sold 45 apples on Monday and 30 on
Tuesday. How many apples remain?
Answer: Let's solve this step by step.  On Monday, the store sold 45 apples.
After Monday: 120 - 45 = <<120-45=75>>75 apples.
On Tuesday, they sold 30 more apples.
After Tuesday: 75 - 30 = <<75-30=45>>45 apples.
#### 45
```

The prompt template `"Answer: Let's solve this step by step."` is a deliberate CoT elicitation. The model learns that after this phrase, it should produce sequential reasoning rather than a direct answer.

### What Learning Looks Like

| Training Phase | Quantitative Signal | Qualitative Signal |
|---|---|---|
| Steps 0–50 | Exact match ~0%. Loss high (~4.5). | CoT anchor count ~0. Model outputs random text. |
| Steps 50–150 | Exact match jumps to 3–8%. Model learns `####` format. | Anchors begin appearing. `cot_coverage_rate` > 0 for first time. |
| Steps 150–300 | Exact match 12–25%. Model attempts multi-step solutions. | Anchor count rises to 1.5–3.0. Step length grows (actual text between anchors). |
| Steps 300–500 | Exact match 25–40%. Strong arithmetic on simple problems. | Anchor count 3–5. Coverage rate approaches 0.8–1.0. Step length >30 chars. |

The critical insight: **a model can reach 15% exact match without any structured reasoning** (by learning to guess common answer patterns). The structural CoT metrics reveal whether the model is actually reasoning or just pattern-matching.

---

## Prerequisites

1. Kafka running on `localhost:9092`
2. Python dependencies: `pip install -r requirements.txt`
3. ~7 GB disk for Qwen2.5-3B model cache

> No extra dependencies for structural CoT — it uses pure Python regex matching. No additional model downloads beyond the base LLM.

---

## How to Run

Open **3 terminals** in the project root:

### Terminal 1 — Inference Server

```bash
python inference.py --config configs/gsm8k_qualitative.yaml
```

**Expected output:**
```
[INFERENCE] Loading base model: Qwen/Qwen2.5-3B
[INFERENCE] Flask server running on http://localhost:5000
[INFERENCE] Listening for LoRA updates on topic: lora-updates-qual-reasoning
```

### Terminal 2 — Trainer

```bash
python trainer.py --config configs/gsm8k_qualitative.yaml
```

**Wait for:**
```
>>> Start the producer now (if not already running). <<<
```

First eval at step 50 produces both quantitative and qualitative output:
```
[EVAL] strategy=regex_extract | exact_match=0.000 | eval_loss=4.321
[QUAL_EVAL] Running structural_cot evaluation (20 samples)...
[QUAL_EVAL] qual_cot_anchor_count_mean: 0.0
[QUAL_EVAL] qual_cot_step_length_mean: 0.0
[QUAL_EVAL] qual_cot_coverage_rate: 0.0
```

By step 300, you should see:
```
[EVAL] exact_match=0.187 | eval_loss=1.842
[QUAL_EVAL] qual_cot_anchor_count_mean: 2.8
[QUAL_EVAL] qual_cot_step_length_mean: 34.2
[QUAL_EVAL] qual_cot_coverage_rate: 0.85
```

### Terminal 3 — Producer

```bash
python producer.py --config configs/gsm8k_qualitative.yaml
```

---

## Test the Live Inference API

This config generates the longest outputs — use `max_new_tokens: 250` to capture full reasoning chains.

```bash
# Simple addition problem
curl -s -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Question: Sarah has 15 red marbles and 8 blue marbles. How many marbles does she have in total?\nAnswer: Let'\''s solve this step by step."}' \
  | python3 -m json.tool
```

```bash
# Multi-step word problem
curl -s -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Question: A baker makes 3 batches of cookies with 24 cookies per batch. She sells 2/3 of them. How many cookies does she have left?\nAnswer: Let'\''s solve this step by step."}' \
  | python3 -m json.tool
```

**Early (step ~50):** Outputs random text with no structure.
```json
{"response": "Answer: Let's solve this step by step. The cookies are all there #### 12"}
```

**Mid-training (~300):** Structured but may have arithmetic errors.
```json
{"response": "Answer: Let's solve this step by step. First, we find the total cookies. 3 × 24 = 72 cookies. Then, she sells 2/3 of them. 72 × 2/3 = 48 cookies sold. Therefore, she has 72 - 48 = 24 cookies left. #### 24"}
```

**Late training (~500):** Correct and fully structured.

---

## Metrics Explained

### Quantitative — Regex Extract

Runs every **50 steps** on a pool of **200 test samples**, evaluating **20 per window**.

| Metric | Column | What it measures | Healthy progression |
|---|---|---|---|
| **Eval Loss** | `eval_loss` | Cross-entropy on full reasoning chain tokens | Falls from ~4.5 → ~1.5 |
| **Perplexity** | `perplexity` | `exp(eval_loss)` | Falls from ~90 → ~4.5 |
| **Accuracy / Exact Match** | `accuracy`, `exact_match` | Fraction where regex-extracted answer = gold answer | Rises from 0% → 25–40% |

### Qualitative — Structural CoT

Runs every **50 steps** on a sliding window of **20 samples** from a pool of **100**.

The model is prompted and generates up to **250 tokens**. The text is then analysed for **logic anchors** — regex patterns that mark structured reasoning transitions.

#### Logic Anchor Categories

The config includes anchors in four semantic groups:

| Category | Example Anchors | Purpose |
|---|---|---|
| **Sequential markers** | `First,` `Second,` `Next,` `Finally,` | Checks for numbered steps |
| **Logical connectives** | `Therefore,` `Thus,` `Hence,` `This means` | Checks for causal reasoning links |
| **Problem decomposition** | `Step 1:` `Let's` `We know` `We need` | Checks for explicit decomposition |
| **Math-specific** | `Since X` `Because X` `The total` `#### N` | Checks for math-reasoning language |

#### Metrics Table

| Metric | Column | Definition | Untrained Baseline | Target |
|---|---|---|---|---|
| **Anchor Count** | `qual_cot_anchor_count_mean` | Mean number of logic anchor matches per response | ~0 | 3–6 |
| **Step Length** | `qual_cot_step_length_mean` | Mean characters between consecutive anchors | ~0 | >30 chars |
| **Coverage Rate** | `qual_cot_coverage_rate` | Fraction of responses with at least 1 anchor | ~0 | Close to 1.0 |
| **Non-Empty Rate** | `qual_non_empty_rate` | Fraction of prompts generating any output | 1.0 | Must stay 1.0 |
| **Repetition Rate** | `qual_repetition_rate` | Fraction of repeated bigrams | <0.05 | Must stay <0.15 |

#### The Critical Metric Pair: Anchor Count + Step Length

Anchor count alone is not enough. A degenerate model could spam `"First, First, First, First, Therefore,"` and score an anchor count of 5 with zero actual reasoning content.

**Step length is the quality gate.** It measures the average character count *between* consecutive anchors:

```
"First, we find the total. Step 1: 3 × 24 = 72. Therefore, 72 - 48 = 24. #### 24"
         │                         │                 │                 │
       anchor 1                 anchor 2          anchor 3          anchor 4
         ←── 22 chars ──────────────→ ←── 19 chars ──→ ←── 8 chars ──→
         
         step_length_mean = (22 + 19 + 8) / 3 = 16.3 chars
```

A rising step length means the model is filling the space between anchors with actual reasoning content, not just repeating markers.

**The ideal signal: `anchor_count` and `step_length` both rising together.**

---

## Reading the Learning Curves

Plots saved to: `output/qual_reasoning/logs/infinitune-qual-reasoning/<timestamp>/`

### Exact Match Curve (`accuracy.png`, `exact_match.png`)

Same interpretation as in `gsm8k_quantitative_guide.md` — starts at 0%, target 25–40%.

### CoT Anchor Count Curve (`qual_cot_anchor_count.png`)

```
Anchors
per
response
5.0 │                              ▓▓▓▓
    │                        ▓▓▓▓▓
2.5 │                  ▓▓▓▓▓
    │            ▓▓▓
1.0 │      ▓▓▓
0.0 │▓▓▓▓▓
    └──────────────────────── Steps
       0   100  200  300  400  500
```

### CoT Step Length Curve (`qual_cot_step_length.png`)

```
Mean chars
between
anchors
40 │                              ▓▓▓▓
   │                        ▓▓▓▓▓
20 │                  ▓▓▓▓▓
   │            ▓▓▓
10 │      ▓▓▓
 0 │▓▓▓▓▓ ← model outputs anchors with no content between them
   └──────────────────────── Steps
      0   100  200  300  400  500
```

**Step length below 10 chars early** is expected — the model generates anchors as random filler. Step length growing beyond 25+ chars means reasoning content is appearing between the structural markers.

### Coverage Rate Curve (`qual_cot_coverage.png`)

- Should approach 1.0 as training progresses.
- A coverage rate of 0.0 for the first 100 steps means the model hasn't learned to use the CoT format at all — check that the prompt template includes `"Let's solve this step by step."`.
- A coverage rate of 1.0 with an anchor count of 1 means the model is using one anchor per response — it has adopted the format minimally. A count of 3–5 is the target.

---

## Decoupled Evaluation (Post-Training)

By default, the framework evaluates the model inline during training. If you want maximum training speed (or prefer to evaluate later), you can disable inline evaluation natively in your config:

```yaml
evaluation:
  decoupled: true   # true = skip inline evaluation during training
```

After training completes (whether you trained with `decoupled: true` or not), you can run evaluation on any checkpoint without re-training:

```bash
# Evaluate final checkpoint
python evaluate.py --config configs/gsm8k_qualitative.yaml

# Compare all checkpoints — best for showing CoT development
python evaluate.py --config configs/gsm8k_qualitative.yaml --all-checkpoints

# Evaluate a specific step
python evaluate.py --config configs/gsm8k_qualitative.yaml --step 300

# List available checkpoints
python evaluate.py --config configs/gsm8k_qualitative.yaml --list
```

Results saved to: `output/qual_reasoning/eval_results/Qwen2.5-3B__gsm8k/<checkpoint>/eval_<timestamp>_<uid>/`

The `--all-checkpoints` combined CSV is particularly powerful for this config — it lets you show a table correlating exact_match (answer correctness) with cot_anchor_count (reasoning structure) across steps, proving that both improve together.

### PowerShell Copy-Paste Commands

```powershell
# Train with inline evaluation enabled
python trainer.py --config configs/gsm8k_qualitative.yaml

# Evaluate final checkpoint
python evaluate.py --config configs/gsm8k_qualitative.yaml

# Evaluate checkpoint step 300
python evaluate.py --config configs/gsm8k_qualitative.yaml --step 300

# Evaluate all checkpoints and build the combined comparison bundle
python evaluate.py --config configs/gsm8k_qualitative.yaml --all-checkpoints

# List saved checkpoints
python evaluate.py --config configs/gsm8k_qualitative.yaml --list
```

---

## Regenerating Evaluation Artifacts

Use `python utils/plot_metrics.py ...` as the canonical regenerate command. It builds a fresh bundle containing plots, dashboards, insights, and `report.html`.

### Latest inline training run -> fresh bundle

```powershell
$Run = Get-ChildItem "output/qual_reasoning/logs/infinitune-qual-reasoning" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$Csv = Join-Path $Run.FullName "metrics_clean.csv"
if (-not (Test-Path $Csv)) { $Csv = Join-Path $Run.FullName "metrics.csv" }
python utils/plot_metrics.py $Csv --config configs/gsm8k_qualitative.yaml
```

### Latest inline training run -> fresh bundle in a custom directory

```powershell
$Run = Get-ChildItem "output/qual_reasoning/logs/infinitune-qual-reasoning" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$Csv = Join-Path $Run.FullName "metrics_clean.csv"
if (-not (Test-Path $Csv)) { $Csv = Join-Path $Run.FullName "metrics.csv" }
python utils/plot_metrics.py $Csv --config configs/gsm8k_qualitative.yaml --out-dir .\my_gsm8k_qual_run_plots
```

### Latest decoupled single-checkpoint eval -> fresh bundle

```powershell
$EvalRun = Get-ChildItem "output/qual_reasoning/eval_results/Qwen2.5-3B__gsm8k" -Directory -Recurse | Where-Object { $_.Name -like "eval_*" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$Csv = Join-Path $EvalRun.FullName "plots\eval_metrics.csv"
python utils/plot_metrics.py $Csv --config configs/gsm8k_qualitative.yaml
```

### Latest all-checkpoints comparison -> fresh bundle

```powershell
$EvalRun = Get-ChildItem "output/qual_reasoning/eval_results/Qwen2.5-3B__gsm8k\all_checkpoints" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$Csv = Join-Path $EvalRun.FullName "all_checkpoints_results.csv"
python utils/plot_metrics.py $Csv --config configs/gsm8k_qualitative.yaml
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `cot_coverage_rate` stays at 0 for 200+ steps | Prompt template missing the CoT elicitation phrase | Verify `prompt_template: "Question: {{ input }}\nAnswer: Let's solve this step by step."` |
| Anchor count rises but step length stays near 0 | Model spams anchors without content | Normal in early training; should resolve by step 200. If not, lower `temperature: 0.4`. |
| Exact match plateau at ~5% | Model learns format but not arithmetic | More training steps needed; consider `max_steps: 750` |
| OOM during qualitative eval | 3B model + 250-token generation is heavy | Reduce `eval_samples: 10` in `testing_strategy` block |
| Both evals running simultaneously cause slowdowns | Quant and qual eval windows interleave | Set `testing_strategy.eval_interval: 100` to stagger them |


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
- Easiest option: let InfiniTune resolve the newest saved adapter automatically with `--checkpoint latest`
- If you need an explicit path, checkpoints live under `output/qual_reasoning/checkpoints/Qwen2.5-3B__gsm8k/run_<timestamp>_<uid>/final`

**Step 3: Run Inference Server**
Launch `inference.py` and pass the mapped checkpoint using the `--checkpoint` flag. This natively bypasses Kafka and locks the adapter statically:
```bash
python inference.py --config configs/gsm8k_qualitative.yaml --checkpoint latest
```

**Step 4: Test the Endpoint**
```bash
curl -X POST http://localhost:5000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Your test prompt here"}'
```
