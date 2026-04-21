# InfiniTune - Testing Guides

This directory contains detailed per-config testing guides for every InfiniTune configuration. Each guide covers what to run, what the model is expected to learn, how to interpret the metrics, and how to regenerate the full evaluation artifact bundle after training or offline evaluation.

For a version-oriented summary of the current major release, see the [Infinitune v2 Release Notes](releases/infinitune-v2.md).

---

## Quick Reference

| Config File | Task | Model | Guide |
|---|---|---|---|
| `configs/imdb_quantitative.yaml` | Sentiment classification | `distilgpt2` | [imdb_quantitative_guide.md](imdb_quantitative_guide.md) |
| `configs/gsm8k_quantitative.yaml` | Math reasoning (exact match) | `Qwen/Qwen2.5-3B` | [gsm8k_quantitative_guide.md](gsm8k_quantitative_guide.md) |
| `configs/alpaca_qualitative.yaml` | Instruction following (semantic similarity) | `Qwen/Qwen2.5-1.5B` | [alpaca_qualitative_guide.md](alpaca_qualitative_guide.md) |
| `configs/imdb_qualitative.yaml` | Domain-adapted review generation (keyword density) | `Qwen/Qwen2.5-1.5B` | [imdb_qualitative_guide.md](imdb_qualitative_guide.md) |
| `configs/gsm8k_qualitative.yaml` | Math reasoning + Chain-of-Thought structure | `Qwen/Qwen2.5-3B` | [gsm8k_qualitative_guide.md](gsm8k_qualitative_guide.md) |
| `configs/e2e_qualitative.yaml` | Structured NLG (slot coverage + consistency) | `gpt2-medium` | [e2e_qualitative_guide.md](e2e_qualitative_guide.md) |

---

## Choosing a Config

```text
I want to demonstrate...

  Clear, hard numbers for classification
  -> imdb_quantitative_guide.md

  Exact-match math improvement
  -> gsm8k_quantitative_guide.md

  Instruction-following quality
  -> alpaca_qualitative_guide.md

  Domain vocabulary adaptation
  -> imdb_qualitative_guide.md

  Structured reasoning behavior
  -> gsm8k_qualitative_guide.md

  Structured data-to-text learning with slot coverage
  -> e2e_qualitative_guide.md
```

---

## Estimated Runtimes

| Config | Training Steps | Inline Eval Time | Decoupled Eval / Checkpoint |
|---|---|---|---|
| `imdb_quantitative.yaml` | 1,000 | ~45 min | ~2 min |
| `gsm8k_quantitative.yaml` | 500 | ~60 min | ~8 min |
| `alpaca_qualitative.yaml` | 500 | ~30 min | ~4 min |
| `imdb_qualitative.yaml` | 1,000 | ~45 min | ~5 min |
| `gsm8k_qualitative.yaml` | 500 | ~65 min | ~10 min |
| `e2e_qualitative.yaml` | 2,100 | ~3-4 hr on Colab T4 | ~15-25 min |

---

## Common Prerequisites

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Kafka (macOS)
brew services start kafka

# 3. Verify Kafka is up
kafka-topics --bootstrap-server localhost:9092 --list
```

For detailed Kafka setup on Windows or legacy Kafka deployments, see the [main README](../README.md#kafka-setup).

---

## Regenerating Plots, Dashboard, and HTML

Use `python utils/plot_metrics.py ...` as the canonical regenerate command. It rebuilds the full evaluation artifact bundle from a saved `metrics.csv` or `metrics_clean.csv`.

```bash
python utils/plot_metrics.py "<path_to_metrics.csv_or_metrics_clean.csv>" \
  --config configs/<your_config>.yaml \
  --out-dir "<path_to_write_artifact_root>"
```

What it generates:
- `evaluation_artifacts/artifact_<timestamp>_<uid>/plots/<theme>/...`
- `evaluation_artifacts/artifact_<timestamp>_<uid>/insights/<theme>/...`
- `evaluation_artifacts/artifact_<timestamp>_<uid>/dashboards/dashboard_dark.png`
- `evaluation_artifacts/artifact_<timestamp>_<uid>/dashboards/dashboard_light.png`
- `evaluation_artifacts/artifact_<timestamp>_<uid>/report.html`
- `evaluation_artifacts/artifact_<timestamp>_<uid>/manifest.json`
- `evaluation_artifacts/artifact_<timestamp>_<uid>/generation_log.json`

Notes:
- If `--out-dir` is omitted, the bundle is created beside the CSV.
- Bundles are append-only. Existing dashboards, plots, and reports are never overwritten.
- `report.html` is best-effort and still renders when some image artifacts are missing.
- For E2E runs, `metrics_clean.csv` is usually the better input because many quantitative columns are intentionally empty.

---

## PowerShell Copy-Paste Pattern

Every per-config guide includes exact PowerShell commands for:
- `python trainer.py --config ...`
- `python evaluate.py --config ...`
- `python evaluate.py --config ... --step N`
- `python evaluate.py --config ... --all-checkpoints`
- regenerating a bundle from the latest inline run
- regenerating a bundle from the latest decoupled single-checkpoint eval
- regenerating a bundle from the latest all-checkpoints comparison

Those commands auto-resolve the latest timestamped run directory so users do not need to manually browse `output/...`.
