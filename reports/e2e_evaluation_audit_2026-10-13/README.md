# E2E metric audit — 2026-10-13

These are real recalculations from retained generated text, not new model runs.
`recalculated_scores.json` records the input SHA-256, package version, original
benchmark code fingerprint, reference-selection rule, and per-example scores.

## Recomputed text-overlap scores

Input: `reports/benchmark_runs/verified_smoke_20261006/predictions/e2e.jsonl`.
There are **four unique meaning representations**. For each ROUGE variant, choose
the reference with the highest per-example F1, take precision/recall/F1 from that
same reference, then average each score across the four examples. Normalization
uses `rouge-score==0.1.2` with Porter stemming. Mean F1 is not reconstructed from
the mean precision and recall. The recomputed ROUGE-L F1 exactly matches the
original manifest, and the prediction SHA-256 matches its saved checksum.

| Metric | Precision | Recall | F1 |
|---|---:|---:|---:|
| ROUGE-1 (word overlap) | 88.72% | 83.93% | 85.90% |
| ROUGE-2 (two-word phrase overlap) | 57.96% | 54.50% | 56.12% |
| ROUGE-L (ordered word overlap) | 67.83% | 65.75% | 66.64% |

These are **four-example smoke diagnostics**, not production-scale quality claims.
ROUGE recall is reference-text recall, not recall of input facts. None of these
scores is classification accuracy, factual precision, or a contradiction rate.
Do not choose the largest score and rename it generic F1.

## Audit of the proposed benchmark claim

Historical input: `output/e2e/outputs/run5/metrics_clean.csv`.

- Mean heuristic coverage: **19.5817% at step 0**, **89.8786% at final**.
  The windows change, and their per-example outputs/IDs are not retained. This
  cannot establish a matched before/after factual-recall result.
- Final complete-coverage proxy: **17.3333%**. Mean 89.9% coverage does not mean
  that 89.9% of descriptions include every requested detail.
- Family-friendliness inversion proxy: initial **0%**, peak **15.7303% at step
  400**, final **1.2048%**. The proposed 15.8% baseline is unsupported; even
  rounding the observed peak yields 15.7%. The metric tests only this one
  attribute and counts examples with a majority of sampled outputs inverted.
  It does not measure all factual contradictions.
- The current `no` checker accepts an omitted family-friendliness statement as
  covered. Substring/alias matching is also not a validated semantic fact
  extractor. Therefore coverage should stay labeled a heuristic proxy.
- `generation_log.json` in historical report bundles contains report-generation
  messages, not the model's generated descriptions. It cannot supply missing
  prediction evidence.
- An automated alert-delivery path was not found in the surveyed trainer,
  inference, benchmark, or utility code. Plot thresholds are not alert delivery.
  Do not include “with alerting” without separate implementation/run evidence.
- Monitoring reports the training trend; the retained evidence does not show
  that telemetry itself caused the model-quality improvement.

## Recommended metric and wording

Use **required-detail recall**: correctly expressed supplied facts / supplied
facts. A clear, plain-language explanation is “percentage of requested restaurant
details correctly included.” Pair it with factual precision (supported generated
facts / all generated facts) or factual F1 so added unsupported facts are penalized.
An even simpler score is **fully correct description rate**: all requested facts
are correctly expressed, with no added unsupported facts or contradictions.

Before using those scores in project reports, evaluate the base and adapted model on
the same fixed held-out MRs with identical decoding, retain predictions and
expected/generated fact labels, verify synonyms/negations against human-reviewed
examples, and report denominators and uncertainty. The runner's full E2E profile
covers 1,484 unique validation MRs, but its current heuristic coverage alone is
not a validated factual precision/recall/F1 implementation.

Safe current wording:

> Built an LLM evaluation pipeline with dashboards and per-example traces to
> track missing restaurant details, family-friendliness errors, and generated-text quality.

Measured-result template, **only after the required evaluation**:

> Built an LLM evaluation pipeline for restaurant descriptions, increasing
> recall of requested details from X% to Y% and reducing descriptions with
> incorrect facts from A% to B% on a fixed test set of N inputs.

## Reproduce the calculation

From the repository root:

```powershell
.\.venv-eval\Scripts\python.exe reports/e2e_evaluation_audit_2026-10-13/recalculate.py
```

No model download or new generation is required. The script writes only this
audit's JSON, preserving the original benchmark and historical files.

Metric references: [Google Research ROUGE implementation](https://github.com/google-research/google-research/blob/master/rouge/rouge_scorer.py),
[official E2E metric suite](https://github.com/tuetschek/e2e-metrics).
The official suite and this repository's SacreBLEU/ROUGE protocol differ;
these numbers are not direct E2E leaderboard comparisons.
