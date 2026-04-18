# Alpaca Instruction Following — Testing Guide
**Config:** `configs/alpaca_qualitative.yaml`

---

## Overview

This pipeline fine-tunes **Qwen2.5-1.5B** on the Alpaca dataset — 52,000 instruction-following examples. The model learns to read an instruction and generate a helpful, on-topic response.

Instruction following is an **open-ended generative task** — there is no single "correct" answer. You cannot measure quality with accuracy. Instead, this config uses **Semantic Similarity** as a qualitative proxy: the model's generated response is compared to the golden Alpaca answer using sentence embeddings, and cosine similarity is tracked over training steps. A rising similarity curve is evidence that the model is learning to produce semantically aligned responses.

| Property | Value |
|---|---|
| **Model** | `Qwen/Qwen2.5-1.5B` |
| **Dataset** | `tatsu-lab/alpaca` (~52k instruction-response pairs) |
| **Task** | Open-ended instruction following |
| **Eval Strategy** | `perplexity` (quantitative baseline) + `semantic_similarity` (qualitative) |
| **Semantic Model** | `sentence-transformers/all-MiniLM-L6-v2` (CPU, ~90 MB, no VRAM) |
| **Estimated Runtime** | ~30 minutes (500 steps on M4 Pro) |
| **Checkpoints** | Every 100 steps + final |

---

## What the Model Learns

### Training Data Flow

Each Alpaca sample has three fields: `instruction`, `input` (optional context), and `output`. The framework maps `instruction → input_col` and `output → target_col`.

The producer formats each sample as:

```
### Instruction:
Explain the difference between supervised and unsupervised learning in simple terms.

### Response:  In supervised learning, the model is trained on labelled examples...
```

The model is trained to predict the **response tokens** given the instruction. The loss is masked so only the response contributes to the gradient — the instruction is a free prompt.

### What Learning Looks Like

| Training Phase | Model State |
|---|---|
| Steps 0–50 | Model ignores the instruction and generates random text. Similarity ~0.1–0.15 (noise floor). |
| Steps 50–150 | Model learns the `### Response:` output format. Starts producing coherent sentences. Similarity grows to 0.2–0.3. |
| Steps 150–350 | Model begins aligning its responses with the semantic content of the instruction. Similarity 0.3–0.45. |
| Steps 350–500 | Model produces fluent, on-topic responses that semantically converge with golden answers. Similarity 0.45–0.60+. |

---

## Prerequisites

1. Kafka is running on `localhost:9092`
2. Python dependencies: `pip install -r requirements.txt` (includes `sentence-transformers`)
3. ~3 GB disk for Qwen2.5-1.5B model cache, ~90 MB for MiniLM sentence model

**First-run downloads:**
- `Qwen/Qwen2.5-1.5B` — downloaded on first `trainer.py` / `inference.py` run
- `sentence-transformers/all-MiniLM-L6-v2` (~90 MB) — downloaded on first qualitative eval (step 50)

You will see this message when MiniLM loads:
```
[QUAL_EVAL] Loading sentence embedding model 'all-MiniLM-L6-v2' on CPU...
[QUAL_EVAL] Model loaded. Device: cpu
```

---

## How to Run

Open **3 terminals** in the project root:

### Terminal 1 — Inference Server

```bash
python inference.py --config configs/alpaca_qualitative.yaml
```

**Expected output:**
```
[INFERENCE] Loading base model: Qwen/Qwen2.5-1.5B
[INFERENCE] Flask server running on http://localhost:5000
[INFERENCE] Listening for LoRA updates on topic: lora-updates-qual-chat
```

### Terminal 2 — Trainer

```bash
python trainer.py --config configs/alpaca_qualitative.yaml
```

**Wait for this before starting the producer:**
```
>>> Start the producer now (if not already running). <<<
```

At step 50, the first qualitative eval fires. You'll see:
```
[QUAL_EVAL] Running semantic_similarity evaluation (20 samples)...
[QUAL_EVAL] qual_semantic_similarity: 0.142
[QUAL_EVAL] qual_non_empty_rate: 1.0
[QUAL_EVAL] qual_mean_response_length: 12.3 words
```

### Terminal 3 — Producer

```bash
python producer.py --config configs/alpaca_qualitative.yaml
```

---

## Test the Live Inference API

```bash
# Factual instruction
curl -s -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "### Instruction:\nExplain what photosynthesis is in one sentence.\n\n### Response:"}' \
  | python3 -m json.tool
```

```bash
# Creative instruction
curl -s -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "### Instruction:\nWrite a short motivational quote about perseverance.\n\n### Response:"}' \
  | python3 -m json.tool
```

**Early training:** Responses will be incoherent or off-topic.  
**Mid-training:** Responses become coherent but may drift from the specific instruction.  
**Late training:** Responses directly and specifically answer the instruction.

---

## Metrics Explained

### Quantitative (Perplexity Baseline)

Runs every **50 steps** on a pool of **50 samples**, evaluating **10 per window**.

| Metric | Column | What it measures | Healthy value |
|---|---|---|---|
| **Eval Loss** | `eval_loss` | Cross-entropy on response tokens | Falls from ~3.5 → ~1.5 |
| **Perplexity** | `perplexity` | `exp(eval_loss)` | Falls from ~33 → ~4.5 |
| **Update Latency** | `update_latency` | Seconds between eval events | Profiling only |

> Accuracy/F1/MCC are disabled — there is no correct class label for open-ended instruction following.

### Qualitative — Semantic Similarity

Runs every **50 steps** on a sliding window of **20 samples** from a pool of **100**.

| Metric | Column | What it measures | Healthy range | What "learning" looks like |
|---|---|---|---|---|
| **Semantic Similarity** | `qual_semantic_similarity` | Cosine similarity between model's response embedding and golden Alpaca response embedding | Rising from ~0.10 → 0.55+ | A consistent upward trend. The baseline (untrained model) is ~0.10–0.15 because random text has near-zero cosine similarity with coherent answers. |
| **Non-Empty Rate** | `qual_non_empty_rate` | Fraction of prompts where the model generated at least one token | Should be 1.0 | If it drops, the model is entering mode collapse |
| **Mean Response Length** | `qual_mean_response_length` | Average word count of generated responses | Should increase from ~5 → 40+ words | Short responses early → model learning to elaborate later |
| **Repetition Rate** | `qual_repetition_rate` | Fraction of repeated bigrams in a response | Should stay below 0.10 | Spikes above 0.20 = degeneration ("the the the...") |

### How Semantic Similarity is Computed

For each eval sample:

```
Instruction: "Explain machine learning in simple terms."

Model Output: "Machine learning is a way for computers to learn from examples
               without being explicitly programmed for each task."

Golden Answer: "Machine learning allows computers to learn automatically from
                data and improve their predictions over time without programming."

Embedding(model output) · Embedding(golden answer)
─────────────────────────────────────────────────── = 0.72 cosine similarity
|Embedding(model output)| × |Embedding(golden answer)|
```

The metric is averaged across all 20 samples in the window.

**Why this works:** Two semantically similar sentences will have high cosine similarity in embedding space even if they use different words. A rising average similarity means the model's responses are converging toward the semantic content of the golden answers.

**What this does not measure:** Fluency, grammar, factual accuracy, or response style. A model could score 0.8 by producing semantically similar but grammatically broken text.

---

## Reading the Learning Curves

Plots saved to: `output/qual_chat/logs/infinitune-qual-chat/<timestamp>/`

### Semantic Similarity Curve (`qual_semantic_similarity.png`)

```
Similarity
0.60 │                              ▓▓▓▓▓
     │                        ▓▓▓▓▓
0.40 │                  ▓▓▓▓▓
     │            ▓▓▓▓▓
0.20 │      ▓▓▓▓▓
     │▓▓▓▓▓
0.10 │ ← baseline (untrained noise floor)
     └──────────────────────── Steps
        0   100  200  300  400  500
```

- The curve should rise **monotonically** (with some noise from windowing).
- A flat curve at ~0.10 after 100 steps means the model is not learning to follow instructions. Check that the prompt template exactly matches what the Alpaca dataset contains.
- A curve that peaks at ~0.25 and plateaus early suggests the model has learned the output format but not the semantic content — more steps or a higher LoRA rank (`r: 16`) may help.

### Response Length Curve (`qual_mean_response_length.png`)

- Starts low (~5–10 words) as the cold model generates near-empty outputs.
- Should grow to **30–80 words** as the model learns to produce full responses.
- **If it drops sharply mid-training:** mode collapse. Watch `qual_repetition_rate.png`.

### Repetition Rate Curve (`qual_repetition_rate.png`)

- Should stay **below 0.10** throughout training.
- A spike above 0.20 indicates degeneration — the model is producing loops.
- If you see this, reduce `temperature` in the `inference` block from 0.7 to 0.5.

---

## Decoupled Evaluation (Post-Training)

By default, the framework evaluates the model inline during training. If you want maximum training speed (or prefer to evaluate later), you can disable inline evaluation natively in your config:

```yaml
evaluation:
  decoupled: true   # true = skip inline evaluation during training
```

After training completes (whether you trained with `decoupled: true` or not), you can run evaluation on any checkpoint without re-training:

```bash
# Full-pool evaluation on final checkpoint
python evaluate.py --config configs/alpaca_qualitative.yaml

# Evaluate a specific checkpoint step
python evaluate.py --config configs/alpaca_qualitative.yaml --step 300

# Compare all checkpoints
python evaluate.py --config configs/alpaca_qualitative.yaml --all-checkpoints

# List saved checkpoints
python evaluate.py --config configs/alpaca_qualitative.yaml --list
```

Results saved to: `output/qual_chat/eval_results/Qwen2.5-1.5B__alpaca/<checkpoint>/eval_<timestamp>_<uid>/`

### PowerShell Copy-Paste Commands

```powershell
# Train with inline evaluation enabled
python trainer.py --config configs/alpaca_qualitative.yaml

# Evaluate final checkpoint
python evaluate.py --config configs/alpaca_qualitative.yaml

# Evaluate checkpoint step 300
python evaluate.py --config configs/alpaca_qualitative.yaml --step 300

# Evaluate all checkpoints and build the combined comparison bundle
python evaluate.py --config configs/alpaca_qualitative.yaml --all-checkpoints

# List saved checkpoints
python evaluate.py --config configs/alpaca_qualitative.yaml --list
```

---

## Regenerating Evaluation Artifacts

Use `python utils/plot_metrics.py ...` as the canonical regenerate command. It builds a fresh bundle containing plots, dashboards, insights, and `report.html`.

### Latest inline training run -> fresh bundle

```powershell
$Run = Get-ChildItem "output/qual_chat/logs/infinitune-qual-chat" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$Csv = Join-Path $Run.FullName "metrics_clean.csv"
if (-not (Test-Path $Csv)) { $Csv = Join-Path $Run.FullName "metrics.csv" }
python utils/plot_metrics.py $Csv --config configs/alpaca_qualitative.yaml
```

### Latest inline training run -> fresh bundle in a custom directory

```powershell
$Run = Get-ChildItem "output/qual_chat/logs/infinitune-qual-chat" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$Csv = Join-Path $Run.FullName "metrics_clean.csv"
if (-not (Test-Path $Csv)) { $Csv = Join-Path $Run.FullName "metrics.csv" }
python utils/plot_metrics.py $Csv --config configs/alpaca_qualitative.yaml --out-dir .\my_alpaca_qual_run_plots
```

### Latest decoupled single-checkpoint eval -> fresh bundle

```powershell
$EvalRun = Get-ChildItem "output/qual_chat/eval_results/Qwen2.5-1.5B__alpaca" -Directory -Recurse | Where-Object { $_.Name -like "eval_*" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$Csv = Join-Path $EvalRun.FullName "plots\eval_metrics.csv"
python utils/plot_metrics.py $Csv --config configs/alpaca_qualitative.yaml
```

### Latest all-checkpoints comparison -> fresh bundle

```powershell
$EvalRun = Get-ChildItem "output/qual_chat/eval_results/Qwen2.5-1.5B__alpaca\all_checkpoints" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$Csv = Join-Path $EvalRun.FullName "all_checkpoints_results.csv"
python utils/plot_metrics.py $Csv --config configs/alpaca_qualitative.yaml
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `qual_semantic_similarity` stuck at ~0.10 | Model not following instruction format | Verify `prompt_template` exactly matches `### Instruction:\n{{ input }}\n\n### Response:` |
| MiniLM not loading | `sentence-transformers` not installed | `pip install sentence-transformers` |
| Responses are just repetitions of the instruction | Prompt template applied to wrong column | Check `input_col: "instruction"` and `target_col: "output"` in config |
| Perplexity not decreasing | `eval_pool_size: 50` too small — lucky/unlucky samples | Increase to `eval_pool_size: 200` |
| Response length stays at 5–10 words | `max_new_tokens: 150` — check inference block | Increase to `max_new_tokens: 200` |


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
- If you need an explicit path, checkpoints live under `output/qual_chat/checkpoints/Qwen2.5-1.5B__alpaca/run_<timestamp>_<uid>/final`

**Step 3: Run Inference Server**
Launch `inference.py` and pass the mapped checkpoint using the `--checkpoint` flag. This natively bypasses Kafka and locks the adapter statically:
```bash
python inference.py --config configs/alpaca_qualitative.yaml --checkpoint latest
```

**Step 4: Test the Endpoint**
```bash
curl -X POST http://localhost:5000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Your test prompt here"}'
```
