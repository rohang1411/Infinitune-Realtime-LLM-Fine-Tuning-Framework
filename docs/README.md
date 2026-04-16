# InfiniTune — Testing Guides

This directory contains detailed per-config testing guides for every InfiniTune configuration. Each guide covers the full lifecycle: what to run, what the model learns, how to interpret every metric, and how to read the learning curves.

---

## Quick Reference

| Config File | Task | Model | Guide |
|---|---|---|---|
| `configs/imdb_quantitative.yaml` | Sentiment classification | Qwen2.5-1.5B | [imdb_quantitative_guide.md](imdb_quantitative_guide.md) |
| `configs/gsm8k_quantitative.yaml` | Math reasoning (exact match) | Qwen2.5-3B | [gsm8k_quantitative_guide.md](gsm8k_quantitative_guide.md) |
| `configs/alpaca_qualitative.yaml` | Instruction following (semantic similarity) | Qwen2.5-1.5B | [alpaca_qualitative_guide.md](alpaca_qualitative_guide.md) |
| `configs/imdb_qualitative.yaml` | Domain-adapted review generation (keyword density) | Qwen2.5-1.5B | [imdb_qualitative_guide.md](imdb_qualitative_guide.md) |
| `configs/gsm8k_qualitative.yaml` | Math reasoning + Chain-of-Thought structure | Qwen2.5-3B | [gsm8k_qualitative_guide.md](gsm8k_qualitative_guide.md) |

---

## Choosing a Config

```
I want to demonstrate...

  Clear, hard numbers (accuracy goes from 50% → 85%)
  └─► imdb_quantitative_guide.md

  Math solving (exact match goes from 0% → 35%)
  └─► gsm8k_quantitative_guide.md

  Instruction following quality (semantic similarity rises)
  └─► alpaca_qualitative_guide.md

  Domain vocabulary absorption (plot the keyword density curve)
  └─► imdb_qualitative_guide.md

  Structured reasoning proof (CoT anchors + exact match together)
  └─► gsm8k_qualitative_guide.md
```

---

## Estimated Runtimes (M4 Pro, 24 GB)

| Config | Training Steps | Inline Eval Time | Decoupled Eval / Checkpoint |
|---|---|---|---|
| `imdb_quantitative.yaml` | 1,000 | ~45 min | ~2 min |
| `gsm8k_quantitative.yaml` | 500 | ~60 min | ~8 min |
| `alpaca_qualitative.yaml` | 500 | ~30 min | ~4 min |
| `imdb_qualitative.yaml` | 1,000 | ~45 min | ~5 min |
| `gsm8k_qualitative.yaml` | 500 | ~65 min | ~10 min |

---

## Common Prerequisites (All Configs)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Kafka (macOS)
brew services start kafka

# 3. Verify Kafka is up
kafka-topics --bootstrap-server localhost:9092 --list
```

For detailed Kafka setup (Windows, KRaft vs Zookeeper), see the [main README](../README.md#kafka-setup).
