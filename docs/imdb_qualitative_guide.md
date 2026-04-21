# IMDb Domain Adaptation — Testing Guide
**Config:** `configs/imdb_qualitative.yaml`

---

## Overview

This pipeline fine-tunes **Qwen2.5-1.5B** on IMDb reviews in **unconditional language modeling** mode. Unlike `imdb_quantitative.yaml` (which classifies sentiment), this config trains the model to **write** movie reviews from scratch.

The model has no "correct answer" to match against. Instead, we measure **Keyword Density** — whether the model's free-form output absorbs the specific vocabulary of film criticism — and **Type-Token Ratio (TTR)** — whether it generates diverse, non-repetitive text.

This is the most "vibes-based" task in InfiniTune, and it demonstrates that quantitative metrics are not always appropriate or meaningful. Domain vocabulary adoption is a direct signal that the model has internalised the style and lexicon of a corpus.

| Property | Value |
|---|---|
| **Model** | `Qwen/Qwen2.5-1.5B` |
| **Dataset** | Stanford IMDb (~25k train reviews) |
| **Task** | Unconditional language modeling — generate movie reviews |
| **Eval (Quant)** | `perplexity` only (accuracy meaningless for generation) |
| **Eval (Qual)** | `keyword_density` + TTR + Hapax Ratio |
| **Estimated Runtime** | ~45 minutes (1,000 steps on M4 Pro) |
| **Checkpoints** | Every 100 steps + final |

---

## ⚠️ Key Difference from `imdb_quantitative.yaml`

Both configs use the IMDb dataset, but they are **completely different tasks**:

| Config | Task | Input | Target |
|---|---|---|---|
| `imdb_quantitative.yaml` | Sentiment classification | Review text | `"positive"` or `"negative"` |
| `imdb_qualitative.yaml` | Review generation | `"Write a movie review:\n"` | The full review text |

In this config, the **review text itself is the training target**. The model learns from the review body, not from a label.

---

## What the Model Learns

### Training Data Flow

The producer formats each IMDb sample as:

```
Write a movie review:
 This film reminded me of the classic noir thrillers of the 1940s. The
cinematography was dark and moody, perfectly complementing the screenplay's
labyrinthine plot. The lead performance was nuanced and compelling...
```

Both `input_col` and `target_col` point to `"text"` — the model learns to continue a movie-review prompt with genuine review prose. There is no label; the signal comes entirely from next-token prediction over the review body.

The model learns: the vocabulary of film criticism, the typical sentence structures of reviews, the emotional valence markers, and the analytical language used by critics.

### What Learning Looks Like

| Training Phase | Model Behaviour |
|---|---|
| Steps 0–50 | Model generates generic text when prompted. Keyword density ~0.005 (basically none). |
| Steps 50–200 | Model starts embedding film-adjacent words. Keyword density creeps up to 0.015–0.025. |
| Steps 200–600 | Model adopts review structure and vocabulary. Density reaches 0.04–0.06. TTR stays high (diverse language). |
| Steps 600–1000 | Model writes coherent, styled reviews with dense film-criticism vocabulary. Density 0.06–0.10+. |

---

## Prerequisites

1. Kafka running on `localhost:9092`
2. Python dependencies: `pip install -r requirements.txt`
3. ~3 GB disk for Qwen2.5-1.5B

> **No extra dependencies beyond the base requirements** — keyword density and TTR are computed with pure Python string operations. No model downloads beyond the base LLM.

---

## How to Run

Open **3 terminals** in the project root:

### Terminal 1 — Inference Server

```bash
python inference.py --config configs/imdb_qualitative.yaml
```

**Expected output:**
```
[INFERENCE] Loading base model: Qwen/Qwen2.5-1.5B
[INFERENCE] Flask server running on http://localhost:5000
[INFERENCE] Listening for LoRA updates on topic: lora-updates-qual-domain
```

### Terminal 2 — Trainer

```bash
python trainer.py --config configs/imdb_qualitative.yaml
```

**Wait for:**
```
>>> Start the producer now (if not already running). <<<
```

At step 50, the first qualitative eval fires:
```
[QUAL_EVAL] Running keyword_density evaluation (30 samples)...
[QUAL_EVAL] qual_keyword_density: 0.007
[QUAL_EVAL] qual_type_token_ratio: 0.81
[QUAL_EVAL] qual_hapax_ratio: 0.64
```

### Terminal 3 — Producer

```bash
python producer.py --config configs/imdb_qualitative.yaml
```

---

## Test the Live Inference API

This is uniquely interesting — you can watch the model's **writing style** evolve:

```bash
# Prompt the model to write a review (short generation)
curl -s -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a movie review:\n"}' \
  | python3 -m json.tool
```

**Early training (~step 50):** Response might be:
```json
{"response": "Write a movie review:\n I liked it very much and it was good and fun."}
```

**Mid-training (~step 400):** Response becomes:
```json
{"response": "Write a movie review:\n The film boasts stunning cinematography and a compelling screenplay. The director's pacing is masterful, though the third act feels somewhat formulaic."}
```

**Late training (~step 900):** The model writes in a genuinely critical, film-savvy voice with domain vocabulary dense throughout.

---

## Metrics Explained

### Quantitative (Perplexity Baseline)

Runs every **50 steps** on a pool of **100 samples**, evaluating **20 per window**.

| Metric | Column | What it measures | Healthy value |
|---|---|---|---|
| **Eval Loss** | `eval_loss` | Cross-entropy on review text tokens | Falls from ~3.5 → ~1.8 |
| **Perplexity** | `perplexity` | `exp(eval_loss)` | Falls from ~33 → ~6 |

> F1/accuracy/exact_match are completely disabled. There is no correct class to predict.

### Qualitative — Keyword Density Suite

Runs every **50 steps** on **30 samples** from a pool of **150**. The model is prompted with `"Write a movie review:\n"` and generates freely (up to 200 tokens). The generated text is then analysed.

#### Domain Keyword List

The config includes 50 curated film-criticism vocabulary words spanning four categories:
- **Cinematography & Visual:** `cinematography`, `visuals`, `lighting`, `color palette`, `frame`, `shot`
- **Narrative & Structure:** `screenplay`, `narrative`, `pacing`, `arc`, `twist`, `subplot`
- **Performance:** `performance`, `portrayal`, `protagonist`, `ensemble`, `nuanced`, `compelling`
- **Direction & Production:** `director`, `editing`, `masterpiece`, `genre`, `sequel`
- **Evaluation Language:** `gripping`, `formulaic`, `engaging`, `captivating`, `disappointing`

#### Metrics Table

| Metric | Column | Definition | Baseline (untrained) | Target (well-trained) |
|---|---|---|---|---|
| **Keyword Density** | `qual_keyword_density` | `(# domain keywords in output) / (total words)` | ~0.005–0.01 | 0.06–0.10+ |
| **Type-Token Ratio** | `qual_type_token_ratio` | `(unique words) / (total words)` | ~0.85 (cold model repeats little) | Should stay above 0.55 |
| **Hapax Ratio** | `qual_hapax_ratio` | `(words appearing exactly once) / (total words)` | ~0.70 | Should stay 0.50–0.80 |
| **Non-Empty Rate** | `qual_non_empty_rate` | Fraction of prompts with at least one generated token | 1.0 | Must stay 1.0 |
| **Mean Response Length** | `qual_mean_response_length` | Average words per generated review | ~10–15 words | 40–100 words |
| **Repetition Rate** | `qual_repetition_rate` | Fraction of repeated bigrams | <0.05 | Must stay <0.15 |

#### Why TTR Drops Over Training (and That's Normal)

At step 0, the cold model generates short, incoherent bursts with high word diversity (because it's random). As training progresses:

1. Responses get **longer** (more total words)
2. The model starts using domain-specific words **repeatedly** across samples

This can push TTR from 0.85 → 0.60. This is **healthy** — it means the model has a consistent domain vocabulary rather than random noise. The warning sign is TTR dropping below 0.40, which indicates degeneration.

---

## Reading the Learning Curves

Plots saved to: `output/qual_domain/logs/infinitune-qual-domain/<timestamp>/`

### Keyword Density Curve (`qual_keyword_density.png`)

```
Keyword
Density
0.08 │                              ▓▓▓▓
     │                        ▓▓▓▓▓
0.04 │                  ▓▓▓▓▓
     │            ▓▓▓▓▓
0.02 │      ▓▓▓▓▓
     │▓▓▓▓▓
0.01 │ ← untrained noise floor (random words, no film vocab)
     └──────────────────────── Steps
        0   200  400  600  800 1000
```

- The **slope** matters more than the absolute value. A consistent upward trend confirms learning.
- Flat line at 0.005–0.008 after 200 steps = model is not absorbing the domain. Check that `target_col: "text"` is set correctly and that the reviews are actually being sent to Kafka.

### TTR Curve (`qual_type_token_ratio.png`)

- Starts high (~0.80–0.85) as random text is naturally diverse.
- Acceptable drop to ~0.55–0.65 as the model develops a consistent vocabulary.
- **Below 0.40:** degeneration. Stop training and inspect `qual_repetition_rate.png`.

### Hapax Ratio Curve (`qual_hapax_ratio.png`)

- Should track TTR shape but be slightly lower.
- In rich, well-trained review generation: 0.50–0.75.
- Very low hapax ratio (<0.30) means the model is fixating on a small word set.

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
python evaluate.py --config configs/imdb_qualitative.yaml

# Compare all checkpoints (best for showing learning trajectory)
python evaluate.py --config configs/imdb_qualitative.yaml --all-checkpoints

# List saved checkpoints
python evaluate.py --config configs/imdb_qualitative.yaml --list
```

Results saved to: `output/qual_domain/eval_results/Qwen2.5-1.5B__imdb/<checkpoint>/eval_<timestamp>_<uid>/`

The `--all-checkpoints` combined CSV is especially useful here — it gives you a table of `keyword_density` vs step number, which is the cleanest way to present domain adaptation learning.

### PowerShell Copy-Paste Commands

```powershell
# Train with inline evaluation enabled
python trainer.py --config configs/imdb_qualitative.yaml

# Evaluate final checkpoint
python evaluate.py --config configs/imdb_qualitative.yaml

# Evaluate checkpoint step 500
python evaluate.py --config configs/imdb_qualitative.yaml --step 500

# Evaluate all checkpoints and build the combined comparison bundle
python evaluate.py --config configs/imdb_qualitative.yaml --all-checkpoints

# List saved checkpoints
python evaluate.py --config configs/imdb_qualitative.yaml --list
```

---

## Regenerating Evaluation Artifacts

Use `python utils/plot_metrics.py ...` as the canonical regenerate command. It builds a fresh bundle containing plots, dashboards, insights, and `report.html`.

### Latest inline training run -> fresh bundle

```powershell
$Run = Get-ChildItem "output/qual_domain/logs/infinitune-qual-domain" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$Csv = Join-Path $Run.FullName "metrics_clean.csv"
if (-not (Test-Path $Csv)) { $Csv = Join-Path $Run.FullName "metrics.csv" }
python utils/plot_metrics.py $Csv --config configs/imdb_qualitative.yaml
```

### Latest inline training run -> fresh bundle in a custom directory

```powershell
$Run = Get-ChildItem "output/qual_domain/logs/infinitune-qual-domain" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$Csv = Join-Path $Run.FullName "metrics_clean.csv"
if (-not (Test-Path $Csv)) { $Csv = Join-Path $Run.FullName "metrics.csv" }
python utils/plot_metrics.py $Csv --config configs/imdb_qualitative.yaml --out-dir .\my_imdb_qual_run_plots
```

### Latest decoupled single-checkpoint eval -> fresh bundle

```powershell
$EvalRun = Get-ChildItem "output/qual_domain/eval_results/Qwen2.5-1.5B__imdb" -Directory -Recurse | Where-Object { $_.Name -like "eval_*" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$Csv = Join-Path $EvalRun.FullName "plots\eval_metrics.csv"
python utils/plot_metrics.py $Csv --config configs/imdb_qualitative.yaml
```

### Latest all-checkpoints comparison -> fresh bundle

```powershell
$EvalRun = Get-ChildItem "output/qual_domain/eval_results/Qwen2.5-1.5B__imdb\all_checkpoints" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$Csv = Join-Path $EvalRun.FullName "all_checkpoints_results.csv"
python utils/plot_metrics.py $Csv --config configs/imdb_qualitative.yaml
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| Keyword density flat at ~0.005 | Reviews not reaching trainer | Check Kafka producer is running and `training_topic: "training-data-qual-domain"` matches |
| TTR drops below 0.30 immediately | Mode collapse / degeneration | Reduce `temperature` in testing_strategy from 0.8 to 0.6; check `max_repetition_ratio` filter |
| `qual_non_empty_rate` drops below 1.0 | Model producing empty outputs | Check `max_new_tokens: 200` in testing_strategy block |
| Perplexity not decreasing | `eval_pool_size: 100` is small | Increase to 500 for more stable estimates |
| Generated reviews are always negative | IMDb has ~50/50 split but random sampling variance | Normal; not a problem since this is an unconditional generation task |


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
- If you need an explicit path, checkpoints live under `output/qual_domain/checkpoints/Qwen2.5-1.5B__imdb/run_<timestamp>_<uid>/final`

**Step 3: Run Inference Server**
Launch `inference.py` and pass the mapped checkpoint using the `--checkpoint` flag. This natively bypasses Kafka and locks the adapter statically:
```bash
python inference.py --config configs/imdb_qualitative.yaml --checkpoint latest
```

**Step 4: Test the Endpoint**
```bash
curl -X POST http://localhost:5000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Your test prompt here"}'
```
