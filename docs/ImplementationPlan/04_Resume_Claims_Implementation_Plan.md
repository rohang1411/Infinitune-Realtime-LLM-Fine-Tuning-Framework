# InfiniTune: Resume Claims Implementation and Verification Plan

> **Document:** `docs/ImplementationPlan/04_Resume_Claims_Implementation_Plan.md`
> **Companion:** `docs/ImplementationPlan/05_Resume_Claims_Testing_Guide.md` (how to run every test and what results to expect)
> **Written:** 2026-10-15, against GitHub `main` at commit `0dfb728` (the local Windows folder matched it apart from CRLF line endings).
> **Revision 2 (2026-10-15):** updated after reviewing the CSCI-566 final report and the run logs in `output/`. Section 2A lists what those logs verify, and the claim table, WP-20, WP-24, §8 and §14 were changed to match. Where the logs already prove a number, the plan now re-scores the saved checkpoints instead of retraining.
> **Audience:** coding agents (including smaller models such as Gemini Flash) and Rohan.
> **Goal:** for each of the 12 resume bullets (SDE-1 to SDE-6, DS-1 to DS-6), build any missing functionality, build the test harness that measures the bullet's numbers, run it on the available hardware, and then either keep the bullet's numbers, change them to what was measured, or reword the bullet.

---

## Table of contents

0. [Rules for every agent (read first)](#0-rules-for-every-agent-read-first)
1. [Executive summary](#1-executive-summary)
2. [Claim-by-claim audit: what exists, what is partial, what is missing](#2-claim-by-claim-audit)
   - [2A. Evidence from the CSCI-566 report and `output/` logs](#2a-evidence-from-the-csci-566-report-and-output-logs)
3. [Verified defects found during the audit](#3-verified-defects-found-during-the-audit)
4. [Hardware matrix and device policy](#4-hardware-matrix-and-device-policy)
5. [Work packages](#5-work-packages)
   - Phase 0: ground truth (WP-00 to WP-01)
   - Phase 1: correctness fixes (WP-02 to WP-09)
   - Phase 2: streaming integrity protocol v2 (WP-10 to WP-12)
   - Phase 3: systems benchmark suites (WP-13 to WP-19)
   - Phase 4: data-science evaluation suites (WP-20 to WP-25)
   - Phase 5: devices, Colab and split deployment (WP-26 to WP-28)
   - Phase 6: evidence ledger and documentation (WP-29 to WP-31)
6. [Execution order and dependency graph](#6-execution-order-and-dependency-graph)
7. [Colab testing](#7-colab-testing)
8. [Claim acceptance rules and rewording templates](#8-claim-acceptance-rules-and-rewording-templates)
9. [Review pass 1: senior AI engineer](#9-review-pass-1-senior-ai-engineer)
10. [Review pass 2: senior data scientist](#10-review-pass-2-senior-data-scientist)
11. [Review pass 3: senior software engineer](#11-review-pass-3-senior-software-engineer)
12. [Final review pass](#12-final-review-pass)
13. [Agent prompt template](#13-agent-prompt-template)
14. [Questions for Rohan](#14-questions-for-rohan)

---

## 0. Rules for every agent (read first)

These rules apply to every work package (WP). If a WP seems to conflict with a rule, the rule wins. Stop and report instead of guessing.

| # | Rule |
|---|---|
| R1 | **Do exactly one WP per task.** Read the whole WP before you edit anything. Do not refactor, rename, reformat or "clean up" code the WP does not name. |
| R2 | **Check the anchors first.** Every WP quotes the current code it changes ("Anchor"). Open the file and confirm the anchor text is there. If it is not, the code has changed since this plan was written: **stop and report**. Do not improvise a new location. |
| R3 | **Never open, print, copy or commit `.env`.** Never commit `output/`, `data/benchmark_cache/`, `reports/benchmark_runs/`, `helper/`, `scratch/`, `.rocketride/` or any model weights. Secrets (Kafka passwords, HMAC keys, HF tokens) come only from environment variables. |
| R4 | **Keep existing behaviour by default.** Existing config keys keep their meaning. A new key must have a default that keeps today's behaviour, unless the WP says in a "Default change" line that the default changes and why. |
| R5 | **No new dependency** unless the WP names it. When it does, add it to `requirements-eval.txt` with an exact `==` pin, and confirm that `pip install` of that pin works on Windows, macOS (arm64) and Linux (Colab) before committing. |
| R6 | **Every WP ends green.** Run `python -m pytest -q` from the repo root. All tests must pass, including the ones that already existed. Never mark a test `skip` or `xfail`, delete it, or weaken an assertion to get green. If an existing test now fails because the WP deliberately changed a behaviour, the WP says so explicitly; otherwise treat the failure as your bug. |
| R7 | **Unit tests never need the network, Kafka or a model download.** Use the tiny in-memory GPT-2 pattern already in `tests/test_repro_evaluation.py:177` (`GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16, ...))`). Live tests that need Kafka go in benchmark suites, not in `tests/`. |
| R8 | **Never type a result number into a document by hand.** Numbers come from a run directory under `reports/benchmark_runs/<run>/` and are copied from its `run_manifest.json` or result JSON. Every number in a doc must cite the run directory and file. |
| R9 | **Cross-platform code.** Use `pathlib.Path`, `encoding="utf-8"` on every text open, `subprocess` with `[sys.executable, ...]` and never `shell=True`. Do not hard-code `/` or `\\` in paths, and do not assume `python3` exists on Windows. |
| R10 | **Ports and topics are unique per run.** Live suites must pick a free TCP port (bind to port 0, read the port, close) and create Kafka topics named `infinitune-<suite>-<uuid12>`. Never reuse the production topic names from `configs/*.yaml` in a benchmark. |
| R11 | **One WP is one commit**, with the message `WP-NN: <short title>`. Do not mix WPs in a commit. |
| R12 | **When unsure, write it down instead of guessing.** Add a `## Open questions` note to your report. An honest "not done, because X" is always better than a plausible-looking guess. |
| R13 | **Mark how you know things.** In your report, mark each statement about the code as `VERIFIED (file:line)` or `INFERRED`. |

Definition of done for a WP:
1. The anchors were confirmed.
2. The changes listed under "Changes" are made, and nothing else.
3. The tests listed under "Tests" exist and pass.
4. The full `python -m pytest -q` passes.
5. The commands under "Acceptance" were run, and their output is pasted into the report.
6. Any doc change the WP lists is made.

---

## 1. Executive summary

- **What the run logs already prove (section 2A):**
  - On a fixed pool of 4,000 IMDb test reviews, scored at all 33 checkpoints, the final adapter reaches **82.65% accuracy, 0.6894 MCC and 0.7902 AAUC**. These are DS-1 and DS-3.
  - E2E slot coverage goes from **19.58% to 89.88%**. The familyFriendly inversion rate goes from **0% to a 15.73% peak to 1.20%**. This is DS-2.
  - The saved checkpoints for both runs are on Rohan's laptop. Per-example evidence, confidence intervals and ECE can therefore come from **re-scoring those checkpoints**, with no retraining.
- **What the logs do not prove:**
  - **Kafka path:** both runs' checkpoints say they were saved by a Colab notebook (`config_path: /content/...`). The notebook trains from an in-process stream, not through Kafka, so whether these runs used Kafka is still open (Q11).
  - **Evaluation speed:** I found no timing trace for the 40-minute to under-4-minute speedup.
  - **Qwen2.5-3B:** no run artifact exists. A Qwen2.5-1.5B IMDb run does exist.
  - **Systems claims:** none of SDE-1 to SDE-6 are measured at the stated scale yet.
- **The code has real foundations:**
  - atomic Kafka snapshots, with safetensors frames and an exact tensor set;
  - a writer-priority read/write lock;
  - prompt-masked loss with left truncation of the prompt;
  - a reservoir replay buffer;
  - target-class macro F1, ECE and a resumable benchmark runner with run manifests.
- **Things that do not exist yet but are named in bullets:**
  - per-tensor SHA-256 verification on the Kafka path (SDE-3);
  - a real ADWIN detector wired into training (DS-5);
  - a legacy-vs-batched evaluation speed benchmark (SDE-4);
  - a live fault-injection suite (SDE-5);
  - persisted per-stage filter statistics on Alpaca (SDE-6);
  - a fixed-set before/after E2E fact evaluation (DS-2);
  - Qwen runs on GPU (DS-1, DS-6).
- **Defects found that would break the planned measurements (section 3):**
  - "latest checkpoint" can pick a checkpoint from the wrong run;
  - the inference server **stops listening to Kafka whenever any checkpoint exists on disk**;
  - one malformed Kafka record can kill the inference listener or crash the trainer;
  - `evaluate.py --step N` and the default final evaluation cannot find checkpoints from earlier runs;
  - the swap timer includes time spent waiting for the lock;
  - AAUC is biased when the first evaluation is not at step 0, and evaluate.py records every checkpoint as step 0;
  - the Alpaca config evaluates on its own training split.
- **Hardware split:**
  - Windows laptop: Kafka broker, GPT-2-family systems tests and unit tests.
  - MacBook Pro M4: timing claims on MPS, including Qwen2.5-1.5B and (after WP-26) Qwen2.5-3B with half-precision base weights.
  - MacBook Air M4: correctness and accuracy runs. It is fanless, so it should not produce timing claims.
  - Colab: Qwen2.5-3B data-science runs and the T4 evaluation-speed claim.
- **Order of work:** ground truth, then bug fixes, then the integrity protocol, then the systems suites, then the DS suites, then Colab, then the evidence ledger and docs. Sections 6 and 13 give the order and a copy-paste prompt for each agent.

---

## 2. Claim-by-claim audit

Status legend:
- **EXISTS**: implemented and tested at the claimed scale.
- **PARTIAL**: the mechanism exists, but the scale or measurement is missing.
- **MISSING**: no implementation.
- **UNSUPPORTED**: the number has no surviving evidence, or the context doc withdrew it.

"Evidence today" quotes `docs/Infinitune_Project_Context.md` §13 unless stated otherwise.

### SDE-1: Kafka platform, Qwen2.5-3B and GPT-2, updates reach serving in under 8 seconds

| Part | Status | Evidence today | Gap / WP |
|---|---|---|---|
| Kafka producer → trainer → inference pipeline | EXISTS | `producer.py`, `trainer.py`, `inference.py`; pipeline smoke ran 8 records and 2 steps in 7.593 s (§13, line 2420) | Single run only; distribution needed (WP-13) |
| "<8 seconds" | PARTIAL | 7.593 s record-to-version (one run, CPU, distilgpt2); 0.844 s snapshot publication-to-visible (one run) | The metric needs a precise definition, at least 20 trials, and p50/p95 per machine (WP-13) |
| Qwen2.5-3B on the streaming path | MISSING | Qwen configs exist; no Qwen live run is recorded; live suites refuse any model except distilgpt2 (`benchmarks/run.py:1122-1124`) | WP-13 plus WP-26/27 (Mac Pro MPS or Colab) |
| "GPT-2" | PARTIAL | Live runs used **distilgpt2**, a distilled GPT-2 rather than GPT-2 itself | Run gpt2-medium in WP-13, or say "DistilGPT-2" (Question Q3) |
| Streaming works after a restart | BROKEN | Defect D2: auto-loading a checkpoint disables the Kafka listener | WP-03 |

### SDE-2: writer-priority read/write lock, checkpoint recovery, 140 ms swaps, zero dropped requests, no 60 s cold start

| Part | Status | Evidence today | Gap / WP |
|---|---|---|---|
| Writer-priority read/write lock | EXISTS | `utils/serving_router.py:28-66`, with tests | — |
| "140 ms" | PARTIAL | One smoke run: 20 requests, 5 swaps, median 140 ms, max 307 ms. The timer includes waiting for in-flight requests to drain (Defect D6) | Split into lock-wait and lock-held times (WP-09); larger runs (WP-15) |
| Zero dropped requests | PARTIAL | 0 failures over 20 requests | 100 to 500 swaps across several concurrency levels (WP-15) |
| Automated checkpoint recovery / "no 60 s cold start" | PARTIAL + BROKEN | Auto-load exists (`inference.py:405-414`), but "latest" can pick the wrong run (D1) and it disables streaming (D2). "60 s" equals the default `weight_push_interval` (`trainer.py:712`); no measurement exists | WP-02, WP-03, plus a cold-start suite (WP-14) |

### SDE-3: atomic weight staging over Kafka with safetensors and SHA-256 manifest verification

| Part | Status | Evidence today | Gap / WP |
|---|---|---|---|
| Atomic staging (manifest, then tensors, then commit) | EXISTS | `trainer.py:284-327`, `utils/stream_protocol.py` | — |
| safetensors frames; pickle rejected | EXISTS | `inference.py:37-59`; `tests/test_repro_evaluation.py:39` | — |
| No partial multi-layer updates | EXISTS | Exact tensor-set check (`stream_protocol.py:44-51`); router key/shape check (`serving_router.py:177-197`) | — |
| **SHA-256 manifest verification over Kafka** | **MISSING** | SHA-256 and HMAC exist only for files on disk (`utils/adapter_manifest.py`). The Kafka manifest has only `step_id, num_tensors, tensor_names, timestamp` (`trainer.py:294-299`) | WP-10 (protocol v2) |
| Malformed frames handled without killing the listener | BROKEN | Defect D3 | WP-04 |

### SDE-4: offline checkpoint auditing service, num_return_sequences decoding, 3,340-sample evaluation cut by 90% (40 min to under 4 min), live p50/p95/p99 APIs

| Part | Status | Evidence today | Gap / WP |
|---|---|---|---|
| Batched `num_return_sequences` generation | EXISTS | `utils/eval_qualitative.py:1050-1121, 1163` | — |
| "3,340 samples" | DEFINED | 334 MRs × 10 consistency runs (§12, line 2237) | — |
| "90%, 40 min to <4 min" | **STATED, NO TRACE (Rohan confirmed no log)** | Stated in CSCI-566 report §7. No timing log in `output/`; context doc §12 line 2237 also found none (section 2A) | Cite the Colab log if one exists (Q12); otherwise legacy sequential mode plus an `eval_speed` suite on a Colab T4 (WP-17) |
| Offline auditing service | PARTIAL | `evaluate.py` is a one-shot CLI with defects D4/D5; no watcher mode | WP-07, WP-18 |
| Live p50/p95/p99 API | EXISTS | `/metrics` (`inference.py:353`, `serving_router.py:232-260`); percentile index is slightly biased (D6b) | WP-09 |

### SDE-5: stress and fault-injection suite, zero request failures, zero corrupted states, truncated Kafka payloads

| Part | Status | Evidence today | Gap / WP |
|---|---|---|---|
| Stress under concurrent swaps | PARTIAL | `swap_stress` suite, smoke only | WP-15 |
| Truncated Kafka payloads | **MISSING (live)** | Unit tests reject incomplete snapshots, but no live injection exists, and a truncated frame currently kills the listener (D3) | WP-04, WP-16 |
| "Zero corrupted model states" | MISSING | Nothing proves the live adapter equals the last good snapshot | Adapter-state digest endpoint (WP-11) and fault suite (WP-16) |

### SDE-6: 5-stage ingestion filter, 52,000+ records, zlib ratio ≤ 0.2, regex, prevents corrupted batches from stalling training

| Part | Status | Evidence today | Gap / WP |
|---|---|---|---|
| 5 stages (length, digit/alphanumeric, zlib, chat structure, regex) | EXISTS (code) | `utils/stream_filter.py:116-191` | Only 3 stages are active in `alpaca_qualitative.yaml` (no regex or chat rules), so "5-stage across 52K" needs a config that enables them (WP-19) |
| zlib threshold ≤ 0.2 | EXISTS | `max_repetition_ratio: 0.2` in the Alpaca, GSM8K and IMDb configs | — |
| 52,000+ records | UNVERIFIED | Context doc says "52,000" (line 1651); no filter run over the full set is recorded | WP-01 census, WP-19 |
| Fail-safe | BROKEN | `validate()` returns `(True, None)` on any exception, so it fails open (`stream_filter.py:189-191`) | WP-06 |
| "Prevents stalled training" | BROKEN | One non-JSON record crashes the trainer (D4) | WP-05; live proof in WP-19 |

### DS-1: config-driven streaming LoRA, Qwen2.5-3B and GPT-2, 4 NLP tasks, 82.7% single-pass accuracy

| Part | Status | Evidence today | Gap / WP |
|---|---|---|---|
| 4 tasks (IMDb, E2E, GSM8K, Alpaca) | EXISTS (configs) | `configs/*.yaml` | — |
| 82.7% | **VERIFIED (aggregate)** | 0.8265 = 3,306/4,000 on a fixed pool at all 33 checkpoints (section 2A). No per-example file | Re-score saved checkpoints for per-example evidence (WP-20A) |
| "on live data streams" via Kafka | **OPEN** | run10 checkpoints were saved in Colab; the notebook does not use Kafka (section 2A) | Q11; Kafka parity run (WP-20) if needed |
| Qwen2.5-3B adapted on a live stream | MISSING | No Qwen2.5-3B artifact. Qwen2.5-1.5B IMDb checkpoints exist | WP-25 on Colab, or say Qwen2.5-1.5B (Q13) |

### DS-2: GPT-2 Medium E2E, 42,000+ records, attribute coverage 20% to 90%, contradictions 15.8% to 1.2%

| Part | Status | Evidence today | Gap / WP |
|---|---|---|---|
| 42,000+ records | **DOUBTFUL** | Context doc says 42,061, but the run's step count fits about 33,500 records (section 2A) | WP-01 |
| 20% to 90% coverage | **VERIFIED as logged; heuristic** | 19.58% → 89.88% (150 MRs × 3 runs per checkpoint, sliding windows). Fixed 10-MR pinned set: 13.3% → 93.4%. Omission counts as covered | Re-score the saved checkpoints on one fixed MR set with a corrected scorer (WP-24) |
| 15.8% to 1.2% contradictions | **NUMBERS VERIFIED, FRAMING WRONG** | 0% at step 0, **15.73% peak at step 400**, 1.20% at the end; familyFriendly only | Reword as "from a 15.7% peak" (§8, Q1); WP-24 adds all-slot contradictions |

### DS-3: prompt-masked loss, response-preserving truncation, gradient clipping, 0.69 MCC, 0.79 AAUC on 4,000 samples

| Part | Status | Evidence today | Gap / WP |
|---|---|---|---|
| Masking, truncation, clipping | EXISTS (code) | `trainer.py:344-397`, `418-449` | No direct unit tests (WP-22) |
| "Prevented prompt memorization and gradient spikes" | UNTESTED | No ablation exists | WP-22 ablation suite |
| 0.69 MCC, 0.79 AAUC | **VERIFIED** | `mcc = 0.6894`, `average_accuracy = 0.7902` at final in `output/imdb/outputs/run10/offline_metrics.csv` (section 2A) | Add bootstrap CIs via WP-20A; WP-21 still fixes the code for future runs |

### DS-4: removed 33% macro-F1 dilution, 4,000 samples, 10-bin ECE, 95% bootstrap intervals

| Part | Status | Evidence today | Gap / WP |
|---|---|---|---|
| Dilution fix | EXISTS | Target-class F1 (`utils/eval_metrics_train.py:880-896`), with a test | — |
| "33%" | CONSISTENT WITH LOGS, BUT DEFINITIONAL (section 2A) | When off-format outputs exist, the legacy F1 averages over 3 classes, one of which ("other") always scores 0. So legacy ≈ 2/3 × clean, and "33%" follows **from the definition**, not from an experiment. It traces to the withdrawn ×1.5 derivation (§13 line 2389) | Report `f1_with_other`, `f1_macro` and `other_rate` from raw predictions (WP-20), and word the bullet as a metric-definition fix (section 8) |
| 10-bin ECE | EXISTS (code), NOT RUN on run10 | `benchmarks/repro_eval.py:344` (forced choice) | Run on the 4,000 pool against the run10 checkpoints (WP-20A) |
| 95% bootstrap intervals | PARTIAL | Paired bootstrap exists for accuracy deltas; Wilson for accuracy | Add percentile-bootstrap CIs for F1 and MCC (WP-20A) |

### DS-5: IMDb → Yelp → Amazon, 1,000-sample Algorithm-R buffer, 20% mix, ADWIN, 600 held-out

| Part | Status | Evidence today | Gap / WP |
|---|---|---|---|
| Algorithm R and 20% mix | EXISTS | `utils/replay_buffer.py`; `continual` suite (`benchmarks/run.py:478-545`) | Standard profile not yet run |
| 600 held-out | EXISTS (definition) | Standard profile: 200 eval rows × 3 domains | — |
| ADWIN | **MISSING (real)** | `ADWINDriftDetector` (`utils/continual_engine.py:102-145`) is a single half-split Hoeffding test, not ADWIN, and is not wired into any run | WP-23 |

### DS-6: Qwen2.5 1.5B/3B, GSM8K, 52K Alpaca, QAFactEval, MiniLM, CoT step checks

| Part | Status | Evidence today | Gap / WP |
|---|---|---|---|
| MiniLM similarity, structural CoT | EXISTS (code) | `utils/eval_qualitative.py` | No recorded Qwen runs |
| "QAFactEval" | PARTIAL | `_QAFactEvalScorer` (`utils/eval_metrics_train.py:18-90`) is a simplified QA-overlap proxy with a MiniLM SQuAD2 reader, not the published QAFactEval pipeline | Rename to "QAFactEval-style" (WP-25, Q2) |
| GSM8K answer extraction | PARTIAL | The regex `#### (\d+)` misses commas, negatives and decimals | WP-25 |
| Alpaca evaluation | FLAWED | `alpaca_qualitative.yaml` sets `eval_split: train`, so evaluation can use training rows | Deterministic hash split (WP-25) |

---

## 2A. Evidence from the CSCI-566 report and `output/` logs

Rohan provided the CSCI-566 final report and stated that its results are verified. The run logs are in `C:\Users\rohan\Documents\Projects\Infinitune-Realtime-LLM-Fine-Tuning-Framework\output`. I read the report and the files below and checked each number against the log.

Evidence files (SHA-256 of the copies I read):

| File | SHA-256 | What it contains |
|---|---|---|
| `output/imdb/outputs/run10/offline_metrics.csv` | `19d0c3c75798a1be20388007828b611788f5efebfdc4a7b6c097240123b64d89` | IMDb decoupled evaluation of all 33 checkpoints (step 0, 50 to 1,550 every 50, and final at step 1,563) |
| `output/imdb/outputs/run10/metrics_clean.csv` | `2337ba552288fc611dd0cbf95ef2832ef031e7fd6f563b3c4e6fb00262c427f7` (matches context doc §13) | The same metrics, cleaned |
| `output/imdb/outputs/run10/config_snapshot.yaml` | — | `eval_pool_size: 4000`, `eval_batch_size: 4000`, class_match, distilgpt2 r16/α32, max_steps 1563 |
| `output/e2e/outputs/run5/offline_metrics.csv` | `ff94c406cb970d42702a9895eb63d05e4eab69e8ce8ab13344cb7b725f832b82` | E2E qualitative evaluation of 12 checkpoints (step 0, 200 to 2,000, final) |
| `output/imdb/outputs/run10/old/offline_metrics.csv` | — | An older evaluation pass: 200 samples per checkpoint, with `_verbose_samples` holding per-example predictions |
| Checkpoints `output/imdb/checkpoints/distilgpt2__imdb/` (step_000050 … step_001550, final), `output/e2e/checkpoints/gpt2-medium__parquet/run_20260414-220458_974a/` (step_000200 … step_002000, final), and Qwen2.5-1.5B IMDb under `output/checkpoints/` and `output/qual_domain/checkpoints/` (step_000100 … step_001000, final) | — | Trained adapters, re-scorable |
| `output/Postman results/imdb_qwen2.5-1.5B.txt` | — | Hand-run `/generate` prompts on Qwen2.5-1.5B, base vs fine-tuned |
| CSCI-566 final report (PDF) | `a66deea58747ec5d810edccbc0d59cd2181e651132453188128c917a53771cad` | The report |

What the logs verify:

| Claim number | Log value | Verdict |
|---|---|---|
| DS-1 "82.7%" | `accuracy = 0.8265` at final, i.e. 3,306 of 4,000 | **VERIFIED.** 0.8265 rounds to 82.7%. The pool is the first 4,000 rows of the IMDb test split after `shuffle(seed=42)` (`utils/eval_metrics_train.py:459-503`). It is the same pool at every checkpoint (`eval_batch_size = eval_pool_size = 4000`). The pool is shuffled, not stratified, so the class balance is close to but not exactly 50/50 |
| DS-1 "single-pass" | The report (§5.3) describes single-pass streaming with an EOF sentinel; `max_steps 1563` = 25,000 / 16 | **VERIFIED for single pass.** **OPEN for "via Kafka":** `checkpoint_meta.json` has `"config_path": "/content/imdb_quantitative.yaml"`, so the checkpoints were saved inside Colab. The Colab notebook (`infinitune_colab.ipynb`, cells 8-13) trains from an in-process iterator with the same masking, truncation, clipping and single-pass loop, but **without Kafka**. INFERRED: run10 did not go through Kafka. Rohan to confirm (Q11) |
| DS-3 "0.69 MCC" | `mcc = 0.6894` at final | **VERIFIED** |
| DS-3 "0.79 AAUC" | `average_accuracy = 0.7902` at final | **VERIFIED.** The history starts at step 0 (accuracy 0) and uses real steps, so defects D5b and D7 did not affect this number. The current `evaluate.py` would not reproduce it, because it passes `step=0` (D5b); the run was produced by an earlier version of the evaluator (a copy is in `output/imdb/utils/`) |
| DS-3 "4,000 test samples" | `eval_pool_size: 4000` | **VERIFIED** |
| DS-4 "33% dilution" | Legacy `f1 = 0.5758`, so the implied target-class F1 is 0.5758 × 1.5 = 0.8637 | **CONSISTENT, BUT STILL DEFINITIONAL.** The ×1.5 is exact only if "other" predictions exist, which the older 200-sample logs confirm (10 to 26 "other" outputs per checkpoint, 5% to 13%). The 33.3% is still a property of averaging over a phantom zero-F1 class, so §8's wording rule stands. Neither ECE nor bootstrap CIs are in run10's logs; WP-20A adds them |
| DS-2 "20% to 90%" | `qual_slot_coverage_mean`: 0.1958 at step 0, 0.8988 at final | **VERIFIED as logged.** Caveats: 150 MRs × 3 sampled runs per checkpoint, taken from a **sliding window**, so the step-0 and final MRs differ. On the fixed 10-MR pinned anchor set, `qual_pinned_slot_coverage_mean` goes 0.1329 → 0.9339. The negation checker still counts an omitted familyFriendly as covered |
| DS-2 "15.8% to 1.2%" | `qual_slot_familyFriendly_inversion_rate`: 0.0 at step 0, **0.157303 at step 400** (14 of 89), 0.012048 at final (1 of 83) | **NUMBERS VERIFIED, FRAMING NOT.** 15.73% rounds to 15.7%, not 15.8%. It is the step-400 peak, not a starting value. The report itself says "peaking at 15.8% (step 400)" in §6.4. The bullet must say "from a 15.7% peak" (§8) |
| DS-2 "42,000+ records" | The context doc and the report say 42,061 | **DOUBTFUL (INFERRED).** The E2E run's last numbered checkpoint is step 2,000 and the final follows, with 8 × 2 = 16 records per step. A full 42,061-record pass would need about 2,629 steps (the config's `max_steps`). An early end around step 2,096 fits the 33,525-row cleaned E2E training set that GEM distributes. WP-01 must count the parquet's rows before this number is used |
| DS-2 LoRA rank | Report §5.1 says r = 16; the E2E `checkpoint_meta.json` says **r = 8** | Report inconsistency. Use r = 8 in docs |
| SDE-4 "40 min to under 4 min" | Report §7 states it; no timing log for it exists in `output/` (only IMDb `eval_cycle_time_s`, about 1.3 to 2.3 s) | **NO TRACE FOUND.** Rohan says the results are verified; if a Colab log exists, add it to `output/` and cite it. Otherwise WP-17 produces one (Q12) |
| DS-1 "Qwen2.5-3B" | No Qwen2.5-3B checkpoint or log in `output/` | **MISSING.** Qwen2.5-1.5B IMDb domain-adaptation checkpoints exist (r = 8, all 7 projection modules, keyword_density strategy, 10 checkpoints), plus a manual base-vs-fine-tuned `/generate` comparison. That supports "Qwen2.5-1.5B", not "Qwen2.5-3B" |
| SDE-1 "<8 seconds" | The report says weight transfer completes "in seconds"; no latency log in `output/` | Unchanged: WP-13 |
| Hardware in report | "Apple Silicon M-series machines (16 GB unified memory) and NVIDIA laptop GPUs", plus Colab (from `config_path`) | Recorded for provenance |

Consequences for this plan:
1. **New WP-20A** re-scores the 33 saved IMDb checkpoints on the identical 4,000 pool. It saves per-example predictions, `other_rate`, target-class F1, bootstrap CIs and ECE. Its **gate is reproduction**: the final checkpoint must give exactly 3,306/4,000 correct with greedy decoding and `max_new_tokens: 6`. If it does not, stop and report the difference.
2. WP-20 (the full Kafka run) becomes a **parity run**. It is only needed if Rohan confirms run10 did not use Kafka. Its job is to show the Kafka path reaches the same accuracy within the confidence interval, not to replace 82.65%.
3. **WP-24** re-scores the 12 saved E2E checkpoints. No new training is needed for the before/after.
4. Never copy `output/` into git. Cite these files by path and SHA-256.

## 3. Verified defects found during the audit

Each defect is VERIFIED against the code at commit `0dfb728`. Each is fixed in the WP named.

| ID | Defect | Location | Consequence | WP |
|---|---|---|---|---|
| D1 | `list_checkpoints()` sorts only by `(kind, step)`. Across several runs the order of equal steps follows `os.scandir` order, so "latest" (`ckpts[-1]`) and `resolve_checkpoint_path` can return a checkpoint from an older run | `utils/checkpoint_manager.py:416-436, 448-466` | Cold start may load stale weights; evaluation may score the wrong run | WP-02 |
| D2 | After cold-start auto-load sets `resolved_checkpoint_path`, `if resolved_checkpoint_path and not args.listen_updates:` skips the Kafka threads | `inference.py:405-414, 496-497` | Once any checkpoint exists, a restarted server silently stops receiving updates | WP-03 |
| D2b | The inference group id is fixed, so `auto_offset_reset="latest"` only applies to a brand-new group. A restarted server resumes from committed offsets and can re-apply an **older** snapshot over a newer cold-start checkpoint | `inference.py:81-89` | Version regression after restart | WP-10 (monotonic version guard) |
| D3 | `value_deserializer=deserialize_tensor` raises inside `consumer.poll()` on a bad frame. The outer `except Exception` ends the consumer thread | `inference.py:83-89, 138-140` | One truncated frame stops all future updates; `/health` returns 503 | WP-04 |
| D4 | Trainer uses `value_deserializer=lambda x: json.loads(...)`; a non-JSON record raises inside `poll()` | `trainer.py:613-616` | One malformed record crashes training (contradicts SDE-6's "prevents stalled training") | WP-05 |
| D5 | `evaluate.py` uses `get_checkpoint_path()` (current run's new, empty directory) for `--step` and the default "final" | `evaluate.py:444, 457` vs `checkpoint_manager.py:440-446` | The default and `--step` modes always fail to find earlier runs | WP-07 |
| D5b | `run_evaluation` calls `evaluator.evaluate(model, step=0)` and `qual_eval.run(model, step=0)` for every checkpoint | `evaluate.py:232, 259` | The AAUC history has equal steps, so AAUC collapses to the last accuracy | WP-07 |
| D5c | Per-checkpoint exceptions are logged and skipped (`continue`), and the process still exits 0 | `evaluate.py:497-500` | Failed evaluations look successful to scripts | WP-07 |
| D6 | `swap_start` is taken **before** `acquire_write()` | `utils/serving_router.py:205-219` | "Swap ms" mixes waiting for readers with the copy itself | WP-09 |
| D6b | Percentiles use `latencies[int(n*q)]`, which rounds up for p50 with small n | `serving_router.py:237-240` | Small upward bias at small n | WP-09 |
| D7 | `_normalized_aauc_from_history` divides by `last_step - 0`, even when the first point is not at step 0 | `utils/eval_metrics_train.py:129-174` | AAUC is biased low when the first evaluation is after step 0 | WP-21 |
| D8 | `StreamFilter.validate` returns `(True, None)` on any exception (fails open) | `utils/stream_filter.py:189-191` | Bad records pass silently | WP-06 |
| D9 | `CheckpointManager.load` hard-codes `torch_dtype=torch.float16` and ignores the configured precision and model revision | `utils/checkpoint_manager.py:468-520` | fp16 on CPU is slow or unsupported; results are not reproducible | WP-08 |
| D10 | `alpaca_qualitative.yaml` has `eval_split: train` | `configs/alpaca_qualitative.yaml` | Evaluation leaks training data | WP-25 |
| D11 | GSM8K `answer_regex: "#### (\\d+)"` | `configs/gsm8k_quantitative.yaml:114`, `configs/gsm8k_qualitative.yaml:138` (applied by the `regex_extract` strategy in `utils/eval_metrics_train.py`) | Answers like `1,200`, `-3` and `2.5` are mis-scored | WP-25 |
| D12 | The live suites refuse any model except distilgpt2 (`parser.error`) | `benchmarks/run.py:1122-1124` | SDE-1 cannot be measured for Qwen or GPT-2 Medium | WP-13 |

**Probable risk (INFERRED; verify in WP-00):** `requirements-eval.txt` pins `kafka-python==2.0.2`, a 2020 release. Two reasons to check it:
- It is widely reported to fail to import on Python 3.12+ (vendored `six.moves`). Colab currently runs Python 3.12, which WP-00 must confirm.
- Kafka 4.x brokers (what `brew install kafka` installs today) dropped old protocol versions.

WP-00 checks both. If either fails, WP-00 moves the pin to the newest `kafka-python` 2.x release that passes the integrity and pipeline smoke suites.

---

## 4. Hardware matrix and device policy

### 4.1 Machines

| Machine | CPU / GPU | RAM | Torch device | Role |
|---|---|---|---|---|
| **Windows laptop** | Intel Core Ultra 5 125H, Arc integrated graphics (shares system RAM) | 16 GB | `cpu` (reliable). `xpu` only as an experiment in a separate venv (WP-26) | Kafka broker host; unit tests; GPT-2-family systems suites (swap, fault, cold start, filter); inference end of the split setup |
| **MacBook Air M4** | Apple M4, fanless | 24 GB unified | `mps` | DS correctness runs (IMDb full single pass, continual, ablation). **No timing claims**: a fanless machine throttles under sustained load |
| **MacBook Pro M4** | Apple M4, active cooling | 24 GB unified | `mps` | **All timing claims on Apple hardware** (SDE-1 propagation, SDE-2 swap and cold start); Qwen2.5-1.5B live; Qwen2.5-3B live after WP-26 |
| **Google Colab** | T4 16 GB / L4 24 GB / A100 40-80 GB (whatever the tier gives) | varies | `cuda` | Qwen2.5-3B DS runs (DS-1, DS-6); the T4 evaluation-speed claim (SDE-4); the trainer end of the split setup |

### 4.2 Memory budgets (weights only; activations, optimizer state and the OS come on top)

Parameter counts in this table are approximate public figures. WP-00 records the exact counts from `sum(p.numel())`.

| Model | ≈ params | fp32 weights | fp16/bf16 weights | Fits trainer + inference on one machine? |
|---|---|---|---|---|
| distilgpt2 | 82 M | ≈ 0.33 GB | ≈ 0.16 GB | Yes, everywhere |
| gpt2-medium | 355 M | ≈ 1.4 GB | ≈ 0.7 GB | Yes, everywhere |
| Qwen2.5-1.5B | 1.5 B | ≈ 6.2 GB | ≈ 3.1 GB | Mac: yes in fp32. Windows 16 GB: **no** (two fp32 copies plus OS) |
| Qwen2.5-3B | 3.1 B | ≈ 12.4 GB | ≈ 6.2 GB | Mac: **only with half-precision base weights** (WP-26). Windows: **no**. Colab: yes |

### 4.3 Device policy (implemented in WP-26)

1. One function, `utils/device.py:resolve_device(config, cli_device=None) -> str`, decides the device for trainer, inference, evaluate and benchmarks. Precedence: CLI `--device`, then env `INFINITUNE_DEVICE`, then config `model.device`, then auto (`cuda` > `mps` > `cpu`). **XPU is never chosen automatically**; it must be asked for explicitly.
2. Precision on MPS: today `precision_manager` forces fp32 on CPU and MPS. WP-26 adds an opt-in `model.allow_half_on_mps: true` that loads frozen base weights in bf16 (fallback fp16) while PEFT keeps the LoRA weights in fp32. The `precision` suite must show finite losses on the Mac before any Qwen-3B Mac run counts.
3. Colab: compute capability ≥ 8.0 (L4, A100) uses bf16; T4 (7.5) uses fp16 with `GradScaler`. Decide this from `torch.cuda.get_device_capability()`, not from the GPU's name.
4. Each run manifest records the device, dtype, machine name (`platform.node()`), OS, CPU model, RAM, power state, and on macOS the output of `pmset -g therm` before and after the run.

### 4.4 Rules for timing runs (all machines)

- Plug the laptop into power. On macOS, turn Low Power Mode off. On Windows, set the power mode to "Best performance".
- Close browsers, IDE indexers and sync clients (OneDrive, Google Drive) during timing runs.
- Run one warm-up trial that is not counted.
- Report p50, p95, max and the number of trials. **Never report a single run as a latency claim.**
- Record each timing claim per machine. The resume can quote the machine that ran it ("on an Apple M4 laptop").

---

## 5. Work packages

Each WP lists: **Claims** it supports, **Depends on**, **Files**, **Anchor(s)**, **Changes**, **Tests**, **Acceptance**, **Pitfalls**. Sizes: S (under 150 changed lines), M (150 to 500), L (over 500).

---

### Phase 0: ground truth

#### WP-00: environment and version census (S)

- **Claims:** all (provenance).
- **Files:** new `benchmarks/env_check.py`; `requirements-eval.txt` (only if a pin must change).
- **Changes:**
  1. Create `benchmarks/env_check.py` with `main()`. It prints JSON and writes it to `reports/env/<hostname>_<YYYYmmdd-HHMMSS>.json`. The JSON contains:
     - `python_version`, `platform.platform()`, `platform.machine()`, `platform.node()`;
     - CPU model (Windows: `platform.processor()`; macOS: `sysctl -n machdep.cpu.brand_string` through `subprocess.run([...])`);
     - total RAM (`psutil.virtual_memory().total`);
     - `torch.__version__`, `torch.cuda.is_available()`, the CUDA device name and capability if available, `torch.backends.mps.is_available()`, and `hasattr(torch, "xpu") and torch.xpu.is_available()`;
     - the versions of `transformers`, `peft`, `datasets`, `safetensors`, `kafka-python` and `numpy` (`importlib.metadata.version`);
     - `import kafka` succeeds (True, or the exception text);
     - `java -version` output if `java` is on PATH;
     - optional `--kafka-bootstrap HOST:PORT`: create a `KafkaAdminClient`, list topics, and record `api_version` from the client (`admin._client.check_version()` in kafka-python returns a tuple; wrap it in try/except and record any error text).
  2. Run it on all four machines. If `import kafka` fails on any machine, or the bootstrap check fails against the broker version you use, change `kafka-python==2.0.2` in `requirements-eval.txt`:
     - Find the newest 2.x version with `pip index versions kafka-python`.
     - Re-run `python -m benchmarks.run run --suites integrity pipeline freshness --profile smoke --kafka-bootstrap <broker>` on Windows and one Mac.
     - Commit the pin only if all three suites pass.
  3. **Broker version policy:** use the same broker major version everywhere. Recommended: Kafka **3.9.x** (the version §13 used) from the Apache archive tarball on every machine, rather than `brew install kafka`, which gives 4.x. If you choose 4.x, the kafka-python pin must pass step 2 against it.
- **Tests:** none (it is a probe script). Add it to the README's command list.
- **Acceptance:** one `reports/env/*.json` per machine. Paste them into the report.
- **Pitfalls:**
  - Do not install anything system-wide from this script. It only reads.
  - `reports/env/` is safe to commit: it contains no secrets. Do not record environment variables.

#### WP-01: dataset census (S)

- **Claims:** SDE-6 (52,000+), DS-2 (42,000+), DS-5 (600 held-out), DS-1/3/4 (4,000).
- **Files:** new `benchmarks/dataset_census.py`.
- **Changes:** load each dataset exactly as the configs do, pinned to a Hub revision (look up the commit hash with `huggingface_hub.HfApi().dataset_info(name).sha` and record it), and write `reports/dataset_census.json` containing:
  - `stanfordnlp/imdb` (the configs use `imdb`; record which id resolves): train and test row counts and label balance.
  - `yelp_polarity` and `amazon_polarity`: test row counts.
  - GEM/e2e_nlg parquet from the exact URLs in `configs/e2e_qualitative.yaml`: train and validation rows, **unique `meaning_representation` count** per split.
  - `tatsu-lab/alpaca`: train rows; rows with a non-empty `input` field.
  - `gsm8k` `main`: train and test rows.
- **Acceptance:** the JSON exists. **Compare it with the bullets:**
  - If E2E train is not ≥ 42,000, DS-2 must use the real count.
  - If Alpaca is not ≥ 52,000, SDE-6 and DS-6 must use the real count.
  - Write the comparison into the report.
- **Pitfalls:** `datasets` may cache on first use; that is fine. Do not commit any dataset files.

---

### Phase 1: correctness fixes

#### WP-02: deterministic checkpoint ordering and "latest" (S)

- **Claims:** SDE-2 (recovery), DS-3 (AAUC across checkpoints).
- **Files:** `utils/checkpoint_manager.py`; new `tests/test_checkpoint_manager.py`.
- **Anchor:**
  ```python
          def sort_key(c):
              s = c["step"]
              return (1, 0) if s == "final" else (0, s)
  ```
- **Changes:**
  1. In `list_checkpoints()`, make each entry carry `run_name`. It is already passed to `_load_step_dir` (`None` for the legacy flat layout); confirm the dict stores it, and add it if not. Replace the sort key so that the list is ordered **by run first, then by step within the run**:
     ```python
     def sort_key(c):
         run = c.get("run_name") or ""          # legacy flat layout sorts first
         s = c["step"]
         return (run, 1, 0) if s == "final" else (run, 0, s)
     ```
     Run directory names are `run_YYYYmmdd-HHMMSS_<uid4>`, so sorting the strings sorts by time. Document this in the docstring.
  2. Add `latest_checkpoint(self) -> Optional[dict]`. It returns the `final` of the newest run if that exists, otherwise the highest step of the newest run that has any checkpoint, otherwise `None`.
  3. Add `list_runs(self) -> list[str]`, sorted ascending.
  4. `resolve_checkpoint_path(step)` keeps its signature. Because of the new order, "last match" now means the newest run. Keep the existing comment, corrected.
  5. In `inference.py`, replace both uses of `ckpts[-1]["path"]` (lines 396 and 411) with `ckpt_mgr.latest_checkpoint()["path"]`, guarded for `None`.
- **Tests** (`tests/test_checkpoint_manager.py`). Build fake directories in `tmp_path` with `checkpoint_meta.json` files matching what `_load_step_dir` reads; open `_load_step_dir` to copy the exact required files.
  - `test_latest_prefers_newest_run_even_if_older_run_has_higher_step`: run A (older) has step 500; run B (newer) has step 50, so latest is B/step_50.
  - `test_latest_prefers_final_within_newest_run`.
  - `test_resolve_step_returns_newest_run_match`.
  - `test_order_is_independent_of_directory_creation_order`: create B before A on disk, and check the order is still A then B.
- **Acceptance:** `python -m pytest -q tests/test_checkpoint_manager.py`, then the full suite.
- **Pitfalls:** the legacy flat layout (step dirs directly under the checkpoint root) must still work. Its `run_name` is `None`, so it sorts before every run.

#### WP-03: inference keeps streaming after cold-start recovery (S)

- **Claims:** SDE-1, SDE-2.
- **Depends on:** WP-02.
- **Files:** `inference.py`; new `tests/test_inference_startup.py`.
- **Anchor:** `if resolved_checkpoint_path and not args.listen_updates:` (line 496).
- **Changes:**
  1. Add a CLI flag `--no-listen-updates` (store_true).
  2. Track *why* a checkpoint was resolved: `checkpoint_source = "explicit" | "auto" | None`.
  3. Decide whether to listen with a pure function, so it can be unit-tested:
     ```python
     def should_listen_for_updates(checkpoint_source, listen_flag, no_listen_flag) -> bool:
         if no_listen_flag:
             return False
         if listen_flag:
             return True
         # Explicit --checkpoint keeps the historical standalone behaviour.
         return checkpoint_source != "explicit"
     ```
     Replace the `if` at line 496 with `if not should_listen_for_updates(...)`.
  4. If `--listen-updates` and `--no-listen-updates` are both given, call `parser.error`.
  5. Log one line stating the decision and the reason.
- **Default change:** a server that **auto-loaded** a checkpoint now listens to Kafka. That is the intended fix for D2. An explicit `--checkpoint` without `--listen-updates` is still standalone, as before.
- **Tests:** `test_should_listen_*` covering all 8 combinations of `{explicit, auto, None} × flags`, including the error case. Import the function without starting Flask: guard everything below `if __name__ == "__main__":`. If importing `inference.py` has side effects (model loading at import time), move the function to `utils/serving_policy.py` and import it from there.
- **Acceptance:** tests pass. A manual smoke on Windows: with a checkpoint on disk, start `python inference.py --config configs/imdb_quantitative.yaml` and confirm the log shows `Kafka listener ready`.

#### WP-04: the inference listener survives malformed frames (S)

- **Claims:** SDE-3, SDE-5.
- **Files:** `inference.py`; new `tests/test_inference_consumer.py`.
- **Anchor:** `value_deserializer=deserialize_tensor,` inside `kafka_consumer_thread`.
- **Changes:**
  1. Remove `value_deserializer` from the `KafkaConsumer(...)` call, so `message.value` becomes raw `bytes`.
  2. Move the per-message handling out into a pure function:
     ```python
     def process_adapter_record(assembler, key, raw_value, counters) -> Optional[tuple]:
         """Return (step, batch) on a validated commit, else None. Never raises."""
     ```
     Inside it, call `deserialize_tensor(raw_value)` and `assembler.add(...)` in one `try`. Catch `Exception` and increment `counters["rejected_frames"]` plus `counters["rejected_by_reason"][type(exc).__name__]`. Log `REJECTED adapter frame key=... reason=...`, truncated to 300 characters, and never log raw bytes.
  3. Store `counters` in `stream_state_global["counters"]` and expose them in `/metrics` under `"stream"`.
  4. The `__done__` key check stays before deserialization.
- **Tests:** a fake assembler and raw byte inputs:
  - truncated safetensors bytes;
  - random bytes;
  - a pickle blob (`pickle.dumps({"a": 1})`);
  - a JSON manifest followed by a valid tensor and a commit.

  Each bad input returns `None`, increments the counters and does not raise. The valid sequence returns the batch.
- **Acceptance:** tests pass. WP-16 later proves this live.

#### WP-05: the trainer survives malformed records (S)

- **Claims:** SDE-6 ("prevents stalled training").
- **Files:** `trainer.py`; new `tests/test_trainer_records.py`.
- **Anchor:** `value_deserializer=lambda x: json.loads(x.decode("utf-8")),` (line 616).
- **Changes:**
  1. Remove that `value_deserializer`.
  2. Add a module-level pure function:
     ```python
     def decode_training_record(raw: bytes):
         """Return (record_dict, None) or (None, reason). Never raises."""
     ```
     It returns reasons `"empty"`, `"not_utf8"`, `"not_json"` or `"not_object"` (JSON that is not a dict). Use the same EOF/verification markers the producer sends: read `producer.py` to find their exact shape, and keep handling them exactly as today.
  3. Where the training loop reads `message.value`, call the function. On failure, increment `malformed_records[reason]`, log at most the first 20 occurrences, and continue.
  4. Optional dead-letter: if `kafka.dead_letter_topic` is set, forward the raw bytes there with the reason as a header. Default unset.
  5. Write `malformed_records` into the run's `metrics.csv` parameters (`MetricsLogger.write_params`) at the end of training.
- **Tests:** each failure class gives the correct reason; a valid record round-trips. The EOF marker is handled exactly as before; assert on the marker's real structure from `producer.py`.
- **Acceptance:** tests pass. The live proof is in WP-19.

#### WP-06: StreamFilter fails closed and counts per stage (S)

- **Claims:** SDE-6.
- **Files:** `utils/stream_filter.py`; `producer.py`; new `tests/test_stream_filter.py`.
- **Anchor:** the final `except Exception as e: ... return True, None` in `validate`.
- **Changes:**
  1. Add config `data.filtering.on_error: "drop" | "pass"`.
     - **Default change:** the default becomes `"drop"`. A filter that crashes must not let a record through. `"pass"` keeps the old behaviour for anyone who wants it.
  2. Name the five stages with constants: `STAGE_LENGTH`, `STAGE_CONTENT` (numeric and alphanumeric checks), `STAGE_REPETITION` (zlib), `STAGE_CHAT`, `STAGE_REGEX`.
  3. Make every rejection reason start with its stage name followed by `:`, for example `"repetition:max_repetition_ratio(0.153<=0.2)"`. Add `STAGE_ERROR = "error"`.
  4. Add a `FilterStats` class with:
     - `seen`, `passed`;
     - `dropped_by_stage: dict[str, int]`;
     - `elapsed_s` (measured with `time.perf_counter()` around `validate`);
     - `examples_by_stage: dict[str, list[str]]`, the first 5 rejected texts per stage, each truncated to 200 characters;
     - `to_dict()`.
  5. `StreamFilter.validate` updates `self.stats` when `self.stats` is not `None`.
  6. Add `active_stages()`, which returns the list of stage names that have any rule configured.
  7. In `producer.py`, at the end of streaming, write `filter_stats.json` next to the producer's log output (or under `project.output_dir/filter_stats/<timestamp>.json`) and log the totals.
- **Tests:**
  - one test per stage that a crafted record is dropped with the right stage prefix;
  - `test_on_error_drop_default`: monkeypatch `_build_text_for_metrics` to raise, and check the result is `(False, "error:...")`;
  - `test_on_error_pass_legacy`;
  - `test_stats_counts_add_up`: `seen == passed + sum(dropped_by_stage.values())`;
  - `test_active_stages_alpaca_config`: load `configs/alpaca_qualitative.yaml` and assert the active stages are exactly `["length", "content", "repetition"]`. This documents that the stock Alpaca config runs 3 stages, not 5.
- **Acceptance:** tests pass.

#### WP-07: evaluate.py finds old runs, uses real steps and fails loudly (M)

- **Claims:** SDE-4, DS-3.
- **Depends on:** WP-02.
- **Files:** `evaluate.py`; new `tests/test_evaluate_cli.py`.
- **Anchors:** `ckpt_manager.get_checkpoint_path(args.step)`, `ckpt_manager.get_checkpoint_path("final")`, `evaluator.evaluate(model, step=0)`, `qual_eval.run(model, step=0)`.
- **Changes:**
  1. Replace the two `get_checkpoint_path` calls with `resolve_checkpoint_path`, keeping the existing "not found" exit path.
  2. Add `--run RUN_NAME` (default: the newest run from `list_runs()`) and `--all-runs`. `--all-checkpoints` now evaluates only the checkpoints of the selected run, in step order, unless `--all-runs` is given.
  3. Thread a real step through. Add a parameter `step` to `run_evaluation(...)` and `evaluate_checkpoint(...)` and pass `ckpt["step"]`; use `max_steps` from the config for `"final"`. Pass it to `evaluator.evaluate(model, step=step)` and `qual_eval.run(model, step=step)`.
  4. The base model must also be evaluated as step 0 when `--all-checkpoints` is used. Check what the current code does for "baseline comparison" (§6 of the context doc says it includes one) and make sure the baseline row is step 0.
  5. When the base model is loaded without an adapter, set `tokenizer.pad_token = tokenizer.eos_token` if it is `None`, exactly as the trainer does (search `pad_token` in `trainer.py` and copy the logic).
  6. Exit codes: keep a list `failed = []`. In the `except` block, append the checkpoint name. At the end, `sys.exit(2)` if `failed`, else `0`. Print a summary line `Evaluated N checkpoints, F failed: [...]`.
  7. Add `--device {auto,cpu,cuda,mps,xpu}`. Wire it to `utils/device.py` after WP-26; until then, accept it and pass it to `_get_device()`.
- **Tests:** monkeypatch `CheckpointManager` and `evaluate_checkpoint`:
  - `test_step_flag_resolves_previous_run`;
  - `test_failure_gives_exit_code_2`;
  - `test_steps_passed_to_evaluators_are_real`: record the `step` arguments; expect `[0, 50, 100, "final"→max_steps]`.
- **Acceptance:** tests pass. On a machine with a real run, `python evaluate.py --config configs/imdb_quantitative.yaml --list` and `--step final` both work.

#### WP-08: CheckpointManager.load respects precision, revision and device (S)

- **Claims:** reproducibility for every DS claim.
- **Files:** `utils/checkpoint_manager.py`; extend `tests/test_checkpoint_manager.py`.
- **Anchor:** `torch_dtype=torch.float16,` inside `load()`.
- **Changes:**
  1. Map `model.precision` (`fp32` → `torch.float32`, `fp16` → `torch.float16`, `bf16` → `torch.bfloat16`) through `utils/precision_manager.py` if it already has a mapping helper (look first); otherwise add a small dict.
  2. Pass `revision=self._config["model"].get("revision")` when it is set.
  3. Replace `device_map="auto"` with the device from `resolve_device` (after WP-26). Until WP-26, keep `"auto"` on CUDA and use `.to(device)` on CPU and MPS.
  4. Use the transformers 5.x keyword (`dtype=`). Check which keyword the pinned `transformers==5.5.3` accepts by reading how `trainer.py` loads the model, and use the same one.
- **Tests:** monkeypatch `AutoModelForCausalLM.from_pretrained` to capture kwargs, and assert the dtype and revision are passed.

#### WP-09: separate lock-wait from lock-held time; nearest-rank percentiles (S)

- **Claims:** SDE-2 (140 ms), SDE-4 (p50/p95/p99).
- **Files:** `utils/serving_router.py`; extend `tests/test_serving_router.py`.
- **Anchor:** `swap_start = time.perf_counter()` followed by `self.rw_lock.acquire_write()`.
- **Changes:**
  1. Measure three times:
     - `t_request` before `acquire_write()`;
     - `t_acquired` right after it;
     - `t_released` right after `release_write()`.

     From these:
     - `lock_wait_ms = (t_acquired - t_request)*1000`;
     - `lock_held_ms = (t_released - t_acquired)*1000`;
     - `swap_total_ms = lock_wait_ms + lock_held_ms`.

     Also measure `prepare_ms`, the time to resolve destinations and convert tensors before the lock, already done at lines 177-197.
  2. Keep `swap_weights` returning `swap_total_ms`, so the old meaning holds for existing callers. Store each component in bounded deques (maxlen 1000) and keep `last_*` values.
  3. Add a module-level `nearest_rank_percentile(sorted_values, q)`: rank `ceil(q*n)` with 1-indexing, clamped to `[1, n]`. Use it for the latency and swap percentiles.
  4. `/metrics` adds:
     - `swap_lock_wait_ms_p50/p95/max`;
     - `swap_lock_held_ms_p50/p95/max`;
     - `swap_prepare_ms_p50/p95`.

     Keep every existing key.
- **Tests:**
  - `test_nearest_rank_percentile_known_values`: `[1..100]` gives p50 = 50, p95 = 95, p99 = 99; for n = 1 every percentile is the value.
  - `test_lock_wait_includes_reader_drain`: hold a read lock in a thread for 200 ms, start a swap, and assert `lock_wait_ms >= 150` and `lock_held_ms < lock_wait_ms`.
- **Acceptance:** the existing `test_serving_router.py` still passes unchanged.

---

### Phase 2: streaming integrity protocol v2

#### WP-10: per-tensor SHA-256, manifest digest, optional HMAC, monotonic versions (L)

- **Claims:** SDE-3 (SHA-256 manifest verification over Kafka), SDE-5, D2b.
- **Depends on:** WP-04.
- **Files:** `utils/stream_protocol.py`; `trainer.py` (`LoRAProducer.send_weights`); `inference.py`; extend `tests/test_stream_integrity.py`; new `tests/test_stream_protocol_v2.py`.

**Wire format v2.** The frame order is unchanged: manifest, then tensors, then commit, all on partition 0.

Manifest value (JSON):
```json
{
  "protocol_version": 2,
  "run_id": "run_20261015-101500_ab12",
  "step_id": "500",
  "seq": 7,
  "created_at": 1760520000.123,
  "num_tensors": 288,
  "tensor_names": ["...sorted..."],
  "tensors": {"<name>": {"sha256": "<hex of the exact frame bytes>", "nbytes": 65568,
                          "dtype": "float32", "shape": [8, 2048]}},
  "integrity": "sha256",
  "hmac_sha256": null
}
```

Commit value (JSON): `{"step_id": "500", "status": "committed", "manifest_sha256": "<sha256 of canonical manifest JSON>", "timestamp": ...}`.

**Definitions** (put them in code as functions with docstrings):
- `canonical_json(obj) -> bytes`: `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")`.
- **Frame hash**: SHA-256 of the exact `bytes` value sent to Kafka for that tensor, i.e. the safetensors blob that `serialize_payload` produces. The producer serializes each tensor **once**, hashes those bytes and sends the same bytes. This is why `LoRAProducer` must serialize before sending rather than letting the `value_serializer` do it. Send pre-serialized `bytes`; `serialize_payload` already passes `bytes` through unchanged (`trainer.py:272-273`).
- **Manifest digest**: `sha256(canonical_json(manifest_without_hmac_field))`.
- **HMAC** (optional): `hmac.new(key, canonical_json(manifest_without_hmac_field), sha256).hexdigest()`. The key comes from env `INFINITUNE_ADAPTER_HMAC_KEY` (UTF-8). The manifest's `tensors` entries carry each frame's hash, so signing the manifest authenticates every frame transitively.

**Inference-side verification** (`SnapshotAssembler`, protocol v2):
1. The manifest must parse, satisfy every v1 check, and have `protocol_version == 2`, plus `tensors` keys equal to `tensor_names`.
2. If `integrity == "hmac"`, or if the receiver is configured with `kafka.integrity: hmac`: verify the HMAC with `hmac.compare_digest`. A missing key on the receiver is a startup error, not a silent pass.
3. Each tensor frame: compute `sha256(raw_bytes)` **before** deserializing. It must equal `manifest.tensors[name].sha256`. Then deserialize and check that `dtype` and `shape` match the manifest.
4. Commit: `manifest_sha256` must equal the stored digest of that step's manifest.
5. **Monotonic guard:** keep `active = (run_id, seq, created_at)`.
   - The same `run_id` requires `seq > active.seq`.
   - A different `run_id` requires `created_at > active.created_at`.
   - Otherwise reject with reason `stale_version`.
   - The cold-start checkpoint initializes `active` from `checkpoint_meta.json`. Add `run_id`, `seq` (use the optimizer step) and `created_at` to what `CheckpointManager.save` writes, if they are not already there.

**Receiver config:** `kafka.integrity: "sha256" | "hmac" | "none_v1_compat"`. Default `"sha256"`. Default change: v1 manifests are rejected unless `none_v1_compat` is set. Update every `configs/*.yaml` to state the key explicitly.

**Producer config:** the same key decides whether to sign. `send_weights(adapter_weights, step_id, run_id, seq)` gains the new parameters with defaults derived from the trainer's `CheckpointManager` run name and the optimizer step.

**Size guard:** before sending, compute each frame's length. If any frame exceeds `kafka.max_request_size` (default `1048576`, kafka-python's default), raise `ValueError("Adapter tensor <name> is <n> bytes; raise kafka.max_request_size and broker/topic max.message.bytes")` before sending the manifest. Never send a partial snapshot.

- **Tests** (`tests/test_stream_protocol_v2.py`, no Kafka):
  - `test_v2_roundtrip_commits`;
  - `test_v2_rejects_bitflip_in_tensor_frame`: flip one byte of the serialized bytes; expect reason `sha256_mismatch`;
  - `test_v2_rejects_truncated_frame`;
  - `test_v2_rejects_shape_or_dtype_mismatch_with_correct_hash`: the hash is computed over the altered bytes but the manifest declares a different shape;
  - `test_v2_rejects_commit_with_wrong_manifest_digest`;
  - `test_v2_hmac_accepts_correct_key_and_rejects_wrong_key`;
  - `test_v2_receiver_in_hmac_mode_rejects_unsigned_manifest`;
  - `test_v1_manifest_rejected_by_default_and_accepted_in_compat_mode`;
  - `test_monotonic_guard_rejects_replayed_older_seq`;
  - `test_monotonic_guard_accepts_newer_run`;
  - `test_size_guard_raises_before_any_send`: use a fake producer that records sends; assert zero sends.
- **Acceptance:** all tests pass. The existing `test_stream_integrity.py` and `test_repro_evaluation.py` tests still pass. They build v1 frames, so if any of them now fails only because v1 is rejected by default, update that test to pass `integrity="none_v1_compat"` explicitly, and say so in the report. The `freshness` and `pipeline` smoke suites pass against a live broker.
- **Pitfalls:**
  - Hash the raw `bytes` from Kafka, not a re-serialization of the tensor.
  - `ensure_ascii=True` and `sort_keys=True` are both required for a stable digest across platforms.
  - Never log the HMAC key or include it in any manifest or report.

#### WP-11: adapter-state digest endpoint (S)

- **Claims:** SDE-5 ("zero corrupted model states").
- **Depends on:** WP-10.
- **Files:** `utils/serving_router.py`; `inference.py`; extend `tests/test_serving_router.py`.
- **Changes:**
  1. Add `DoubleBufferedAdapterRouter.adapter_state_digest() -> str`. Under the **read** lock, take `get_peft_model_state_dict(self.model)` and, for each key in sorted order, hash `key` and `tensor.detach().cpu().contiguous().numpy().tobytes()` into one SHA-256. Cache the result by `active_version`, and clear the cache on every swap.
  2. Add `GET /adapter`, which returns:
     - `active_version`, `run_id`, `seq`;
     - `adapter_state_sha256`;
     - `expected_state_sha256`: the digest computed from the committed snapshot's tensors at apply time, using the same function on the snapshot dict, after dtype conversion to the live dtype;
     - `match` (bool).
  3. During `swap_weights`, compute `expected_state_sha256` from `prepared_weights` (after dtype conversion), store it, and after release compute the live digest once and log `MISMATCH` if they differ.
- **Tests:**
  - with the tiny GPT-2 PEFT model, apply a snapshot and check `match` is True;
  - mutate one live tensor by hand and check `match` is False;
  - check the digest is identical for the same weights inserted in a different dict order.

#### WP-12: Kafka client configuration passthrough and secrets (M)

- **Claims:** SDE-1 (distributed), the Colab split setup.
- **Files:** new `utils/kafka_config.py`; `trainer.py`, `inference.py`, `producer.py` and the Kafka-using code in `benchmarks/run.py`; new `tests/test_kafka_config.py`.
- **Changes:**
  1. Add the function:
     ```python
     def build_client_kwargs(kafka_cfg: dict, role: str) -> dict:
         """role in {"producer", "consumer", "admin"}. Returns kwargs for kafka-python clients."""
     ```
     - It always includes `bootstrap_servers`. The env var `INFINITUNE_KAFKA_BOOTSTRAP` overrides the config.
     - It copies `kafka.client` keys from the allowlist `security_protocol`, `sasl_mechanism`, `ssl_cafile`, `ssl_check_hostname`, `request_timeout_ms`, `api_version_auto_timeout_ms`.
     - When `sasl_mechanism` is set, it reads `INFINITUNE_KAFKA_USERNAME` and `INFINITUNE_KAFKA_PASSWORD` from the environment and sets `sasl_plain_username`/`sasl_plain_password`. **It raises if they are missing, and never accepts them from YAML**: if they appear in YAML, raise a `ValueError` that says to use env vars.
     - Producer only: `max_request_size` (default 1048576) and `compression_type` (default `None`). Note: kafka-python supports `gzip` without extra packages; `lz4`, `snappy` and `zstd` need extra libraries, so allow only `None` or `"gzip"` unless a later WP adds more.
     - Consumer only: `max_partition_fetch_bytes`, which must be at least `max_request_size`; raise otherwise.
  2. Replace every `KafkaProducer(`, `KafkaConsumer(` and `KafkaAdminClient(` construction in those files so it spreads `**build_client_kwargs(kafka_cfg, role)`. Keep the other explicit arguments (group ids, deserializers and so on) as they are.
  3. `benchmarks/run.py --kafka-bootstrap` continues to work: it writes into `config["kafka"]["bootstrap_servers"]`.
- **Tests:**
  - plaintext default;
  - SASL fails when the env vars are missing;
  - SASL secrets in YAML are rejected;
  - the env bootstrap override wins;
  - `max_partition_fetch_bytes` smaller than `max_request_size` raises;
  - unknown keys are ignored with a warning.
- **Pitfalls:** never print kwargs that contain a password. Log a redacted copy.

---

### Phase 3: systems benchmark suites (SDE-1 to SDE-6)

All new suites live in `benchmarks/run.py`, or in new modules under `benchmarks/` that `run.py` calls. To add a suite:
1. Add a `SUITES` entry with a one-line description.
2. Write a `suite_<name>(run, args, ...)` function that uses `run.stage(name, operation)` for each resumable stage.
3. Add the lambda to the `operations` dict in `main()`.

Every suite writes:
- raw per-trial data to `run.directory/<suite>/raw.jsonl`;
- a summary to `run.directory/<suite>/summary.json`.

Each summary records `n`, p50, p95, max, the machine fields from WP-00, and a `"boundary"` string saying exactly what was and was not measured. (The existing `freshness` suite already does this; follow its style.)

**New suite names to register** (exact spelling, used by the testing guide): `propagation`, `cold_start`, `fault_injection`, `eval_speed`, `filter_benchmark`, `streaming_quality`, `ablation`, `e2e_facts`, `rescore`. Also add the runner flags these suites use: `--live-config`, `--device`, `--push-interval`, `--trials`, `--concurrency-levels`, `--split-mode`.

**Generalize the live-model restriction (D12).** Add `--live-config PATH`, default `configs/imdb_quantitative.yaml`. Replace the `parser.error` at `run.py:1122-1124` with a check that the live config's `model.name` equals `--model` and that `--model-revision` is pinned. Add pinned revisions for `gpt2-medium`, `Qwen/Qwen2.5-1.5B` and `Qwen/Qwen2.5-3B` to `MODEL_REVISIONS` in `benchmarks/repro_eval.py`; look up each commit hash with `HfApi().model_info(name).sha` once and paste it in.

#### WP-13: `propagation` suite: weight-update propagation distribution (M)

- **Claims:** SDE-1.
- **Depends on:** WP-03, WP-10, WP-12.
- **Definition** (put it in the suite docstring). For each weight push *k*:
  - `t_publish_start`: the trainer calls `send_weights` (wall clock, `time.time()`).
  - `t_flushed`: `producer.flush()` returned.
  - `t_assembled`: the inference consumer validated the commit.
  - `t_applied`: `swap_weights` released the write lock.
  - **propagation_s = t_applied − t_publish_start.** This is the resume number.
  - The suite also reports `publish_s`, `transport_assemble_s` and `apply_s`, which sum to the total.
- **Mechanism:**
  1. `t_publish_start` and `t_flushed` go into the v2 manifest (`created_at`) and into the trainer log.
  2. Inference records `t_assembled` and `t_applied` per version in a bounded dict exposed at `GET /adapter/history` (last 200 entries).
  3. The suite starts a real `inference.py` with `--listen-updates` and a real `trainer.py`, both on isolated topics, with `weight_push_interval` set to `--push-interval` (default 5 s), and streams training data from `producer.py`.
  4. It collects `--trials` pushes (smoke 3, standard 20, full 100), joins trainer and inference timestamps by `(run_id, seq)`, and writes `raw.jsonl` and `summary.json`.
  5. Both processes run on **one machine**, so their wall clocks match. For the split Colab→laptop setup, see section 7.5 (clock-offset correction).
- **Also keep** the existing record-to-version number (§13's 7.593 s). The `pipeline` suite keeps measuring "last training record acknowledged → first version that trained on it is visible", reported separately.
- **Runs required:**

  | Machine | Model / live config | Profile |
  |---|---|---|
  | Windows (CPU) | distilgpt2 / `imdb_quantitative.yaml` | standard |
  | Windows (CPU) | gpt2-medium / `e2e_qualitative.yaml` | standard |
  | MacBook Pro (MPS) | gpt2-medium | standard |
  | MacBook Pro (MPS) | Qwen2.5-1.5B / `alpaca_qualitative.yaml` (r=8, attention modules) | standard |
  | MacBook Pro (MPS) | Qwen2.5-3B / `gsm8k_quantitative.yaml`, with `allow_half_on_mps: true` (after WP-26) | standard |
  | Colab (single VM) | Qwen2.5-3B | standard |

- **Acceptance:** `summary.json` per run, with p50, p95, max and n ≥ 20 for standard. SDE-1 may say "<8 seconds" **only if p95 < 8 s on the stated machine and model** (section 8).
- **Pitfalls:**
  - Do not poll `/metrics` to time the result; use the server-side timestamps. Polling adds up to one poll interval of error.
  - Qwen2.5-3B: the trainer computes in bf16/fp16 on the Mac, but the adapter frames are fp32 (PEFT keeps adapters fp32). Check the size guard (WP-10) passes.

#### WP-14: `cold_start` suite: checkpoint recovery versus waiting for the first push (M)

- **Claims:** SDE-2 ("no 60 s cold-start delay").
- **Depends on:** WP-02, WP-03, WP-11.
- **Procedure:**
  1. Prepare a trained checkpoint (from the `pipeline` suite, or `--checkpoint`).
  2. **Arm A (recovery on):** start inference with auto-load (no `--checkpoint`; `auto_load_latest_checkpoint: true`) and record:
     - `t_ready`: `/health` returns 200;
     - `t_trained_serving`: the first `/generate` whose `X-Adapter-Version` equals the checkpoint version;
     - `listener_ready`: true/false from `/metrics`, to prove D2 is fixed.
  3. **Arm B (recovery off):** set `auto_load_latest_checkpoint: false`, start inference, then start a trainer that pushes after `weight_push_interval` (default 60, the historical value). Record `t_trained_serving` as the first response with a trained version.
  4. Repeat each arm 5 times (standard).
- **Report:** for each arm, `ready_s` (process start to healthy) and `time_to_trained_weights_s`. The claim is supported if arm A serves trained weights at `t_ready` (the same request), while arm B serves base weights until its first push.
- **Boundary statement:** "Model load time is not removed; what is removed is the window during which a restarted server answers with untrained base weights."

#### WP-15: extend `swap_stress` (M)

- **Claims:** SDE-2 (140 ms, zero dropped), SDE-5.
- **Depends on:** WP-09, WP-11.
- **Changes** to `suite_swap_stress` (`benchmarks/run.py:979`):
  - `--concurrency-levels 1 2 4 8` (default `[2]` for smoke, `[1, 2, 4, 8]` for standard and full).
  - The number of swaps follows the existing profile (5 / 100 / 500).
  - For every response, record status, latency and `X-Adapter-Version`.
  - **Version validity check:** every returned version must be in the set of versions committed before the response finished.
  - At the end, `/adapter` must report `match: true`.
  - Report:
    - `request_failures` (non-200 responses plus exceptions);
    - `dropped_requests` from `/metrics`;
    - `swap_lock_held_ms` p50/p95/max;
    - `swap_lock_wait_ms` p50/p95/max;
    - request latency p50/p95/p99 during swaps versus a no-swap control window of equal length.
- **Acceptance:** `request_failures == 0`, `dropped == 0` and `match == true` in every run. The resume's "140 ms" must be restated as whichever of lock-held or total is meant (section 8).

#### WP-16: `fault_injection` suite (L)

- **Claims:** SDE-5, SDE-3.
- **Depends on:** WP-04, WP-10, WP-11, WP-15.
- **Setup:**
  - A live `inference.py` with `--listen-updates` on an isolated topic, with `kafka.integrity: hmac` for the HMAC cases and `sha256` otherwise. Two server runs are fine.
  - A closed-loop load generator running for the whole suite (reuse `benchmarks/load_test_serving.py`), with concurrency 4.
  - A raw Kafka producer that the suite controls, sending pre-built `bytes` frames.
- **Fault catalogue.** Each fault is followed by one valid snapshot (the recovery probe).

| ID | Injected fault | Expected rejection reason | Expected after the fault |
|---|---|---|---|
| F01 | Tensor frame truncated to 50% of its bytes | `sha256_mismatch` | Version unchanged; digest equals the last good one |
| F02 | One byte flipped in a tensor frame | `sha256_mismatch` | Same |
| F03 | Tensor frame missing, then commit | `incomplete_snapshot` | Same |
| F04 | Duplicate tensor frame | `duplicate_tensor` | Same |
| F05 | Undeclared extra tensor | `undeclared_tensor` | Same |
| F06 | Correct hash but wrong shape vs the manifest | `shape_mismatch` | Same |
| F07 | Pickle payload as a tensor frame | `sha256_mismatch` (never unpickled) | Same |
| F08 | Commit with no manifest | `unknown_step` | Same |
| F09 | Commit with the wrong `manifest_sha256` | `manifest_digest_mismatch` | Same |
| F10 | Manifest that is not JSON / random bytes | `bad_manifest` | Same |
| F11 | Replay of an older committed snapshot (lower `seq`) | `stale_version` | Same |
| F12 | Manifest signed with the wrong HMAC key | `hmac_mismatch` | Same |
| F13 | Unsigned v1 manifest to a receiver in `sha256` mode | `protocol_version` | Same |
| F14 | Two snapshots interleaved frame-by-frame (step A and B) | Both commit (each is complete), **or** the older is rejected as stale; never a mix | Digest equals whichever committed last |
| F15 | Frame over the size limit (producer side) | `ValueError` before any send | Nothing sent |
| F16 | Burst of 20 bad frames while 4 clients send requests | One rejection per frame | Zero request failures |

  The exact reason strings must be those the WP-10 code emits. Align this table with the code and update this doc.

- **Per-fault assertions:**
  1. `/adapter.match == true`.
  2. `/adapter.adapter_state_sha256` equals the digest of the last *good* snapshot.
  3. `/metrics.stream.rejected_by_reason` grew by the expected reason.
  4. The listener is alive (`consumer_ready` true and `/health` 200).
  5. The recovery probe snapshot is applied within 30 s.
  6. The load generator has 0 failures.
- **Output:** `fault_injection/summary.json` with one row per fault (pass/fail plus evidence), plus totals: `faults_injected`, `faults_rejected`, `request_failures`, `corrupted_states` (count of `match == false`).
- **Acceptance:** every row passes. `request_failures == 0`, `corrupted_states == 0`. Repeat 3 times on the standard profile (Windows CPU distilgpt2, and the MacBook Pro with gpt2-medium).

#### WP-17: `eval_speed` suite: sequential versus batched generation (M)

- **Claims:** SDE-4 (3,340 samples, 90%, 40 min to <4 min).
- **Files:** `utils/eval_qualitative.py`; new `benchmarks/eval_speed.py`.
- **Changes:**
  1. Recover the old loop with `git show ca90937^:utils/eval_qualitative.py`. Commit `ca90937` ("Evaluation Code Memory Optimization by moving the model.generate operation from for loop to CUDA") introduced batching. Port the pre-batching generation loop **exactly as it was** into a new method `_generate_responses_sequential(...)` with the same signature and return shape as `_generate_responses`.
  2. Add config `testing_strategy.generation_mode: "batched" | "sequential"`, default `"batched"`, and route `run()` to the matching method.
  3. `benchmarks/eval_speed.py` (or the suite `eval_speed`):
     - loads gpt2-medium (pinned) and the E2E validation MRs;
     - uses `n_mrs` (smoke 8, standard 334) and `consistency_runs` 10, which gives 3,340 generations at standard;
     - runs both modes on the same MRs, seed, `max_new_tokens` and temperature;
     - measures wall time with `torch.cuda.synchronize()` on CUDA before reading timers;
     - one warm-up batch is not counted;
     - repeats 3 times.
  4. Equivalence check (greedy only): with `do_sample=False` and `num_return_sequences=1` on 50 MRs in fp32, the batched and sequential outputs should be identical strings for at least 98% of MRs. Left-padding can change floating-point results slightly, so this is a tolerance check, not a hard equality. Report the identical fraction.
- **Required run:** Colab **T4** (the claim's original setting), standard profile. Optional: L4 and MacBook Pro MPS for context.
- **Acceptance:** the summary reports `sequential_s`, `batched_s` and `reduction_pct = 1 − batched/sequential` with three repeats each. SDE-4 may keep "90%" only if `reduction_pct ≥ 0.90` on the T4. It may keep "40m to <4m" only if the sequential time is about 40 minutes (between 36 and 44) and batched is under 4 minutes. Otherwise use the measured numbers (section 8).
- **Pitfall:** ensure both modes use the same `max_new_tokens` and the same early-stopping (EOS) behaviour.

#### WP-18: offline checkpoint auditing service (`evaluate.py --watch`) (M)

- **Claims:** SDE-4 ("offline checkpoint auditing service").
- **Depends on:** WP-07.
- **Changes:**
  1. `evaluate.py --watch [--poll-seconds 15] [--run RUN|latest]` polls the checkpoint root and evaluates each **new complete** checkpoint once. "Complete" means its `checkpoint_meta.json` exists and loads, and its adapter weights file exists.
  2. A state file `<output_dir>/eval_watch_state.json` lists done checkpoints, so a restart does not re-evaluate them.
  3. Results go to the same per-checkpoint directories as one-shot mode.
  4. Write a `watch_metrics.json` with the per-checkpoint evaluation latency (start to finish) and the queue delay (checkpoint mtime to evaluation start), with p50/p95.
  5. `--device` lets the auditor run on a different device from the trainer (for example the trainer on MPS and the auditor on CPU), which is the "workload isolation" in the bullet.
  6. SIGINT stops cleanly after the current checkpoint.
- **Tests:** monkeypatched `evaluate_checkpoint`:
  - `test_watch_evaluates_each_checkpoint_once`;
  - `test_watch_skips_incomplete_checkpoint_until_meta_exists`;
  - `test_watch_resumes_from_state_file`. For these, add `--max-iterations N` (testing only) so the loop terminates.

#### WP-19: `filter_benchmark` suite and the live no-stall proof (M)

- **Claims:** SDE-6.
- **Depends on:** WP-01, WP-05, WP-06.
- **Part A (offline, any machine):**
  1. Load the full `tatsu-lab/alpaca` train split. The record count comes from WP-01.
  2. Build text exactly the way `producer.py` builds it for the Alpaca config. Read `producer.py` and call the same function; do not re-implement it.
  3. Filter config: `configs/alpaca_qualitative.yaml`'s filter **plus** a regex stage and a chat-structure stage, so all 5 stages are active. The regex stage gets `custom_regex_must_not_match: ["(https?://\\S+.*){3,}", "(.)\\1{20,}"]` (3+ URLs; a character repeated 21+ times). Save the exact filter block used in `summary.json`.
  4. Add a labelled spam set: N = 2,000 synthetic records, seeded:
     - 500 repeated-phrase records (zlib ratio well below 0.2);
     - 500 symbol-noise records (alphanumeric ratio < 0.5);
     - 500 URL-spam records;
     - 500 too-short or too-long records.

     Mix them into the stream at random positions.
  5. Report:
     - per-stage drop counts on Alpaca;
     - throughput (records/s, with `perf_counter`);
     - recall on the injected spam (fraction dropped) per spam type;
     - drop rate on Alpaca, which is the upper bound on false positives;
     - 50 random dropped Alpaca rows written to `filter_benchmark/dropped_sample.csv` for Rohan to review by hand (section 14, Q6).
- **Part B (live, Windows):** run producer → trainer for 200 optimizer steps (distilgpt2 IMDb config, smoke-sized data) while the suite's raw producer injects 50 malformed records (non-UTF-8, non-JSON, JSON list, empty) into the training topic. **Pass** if:
  - the trainer reaches the target step;
  - `malformed_records` totals 50, with the right reasons;
  - the trainer exit code is 0.

  As a control, run the same on the **pre-WP-05** code (`git stash` or a worktree at `0dfb728`) and record that it crashes. This gives the before/after for "prevents stalling".
- **Acceptance:**
  - Part A: the summary exists, and injected-spam recall is ≥ 0.99 for the zlib and alphanumeric types. If not, report it honestly and do not tune thresholds on the same data.
  - Part B passes.

---

### Phase 4: data-science evaluation suites (DS-1 to DS-6)

**Common rules for all DS suites:**
- **Fixed evaluation set.** Draw it once with the existing `prepare_dataset` (seeded, stratified, pinned revision, SHA-256 manifest). Every checkpoint and the base model are scored on **the same rows**.
- **Per-example predictions** are saved as JSONL: `id`, `gold`, `raw_generation`, `normalized_prediction`, `correct`, and `label_probs` where applicable.
- **Seeds:** 3 for standard, 5 for full. Report the mean ± standard deviation across seeds, plus a 95% CI per seed.
- **Uncertainty:**
  - Wilson interval for accuracy (existing);
  - **percentile bootstrap** for macro F1 and MCC: 2,000 resamples of examples with replacement, fixed seed 0; report the 2.5th and 97.5th percentiles;
  - paired bootstrap for differences between systems (existing `paired_bootstrap`).
- **Never** pick the best checkpoint or seed after looking at the test set and report it as *the* result. Report the final checkpoint, and if you also report the best, label it "best-of-N on test (optimistic)".

#### WP-20A: re-score the saved IMDb run10 checkpoints (M), do this before WP-20

- **Claims:** DS-1, DS-3, DS-4 (per-example evidence, CIs, ECE, off-format rate).
- **Depends on:** WP-02 (ordering), WP-08 (load precision). No Kafka needed.
- **Inputs:** the 33 run10 checkpoints in `output/imdb/checkpoints/distilgpt2__imdb/` and the config `output/imdb/outputs/run10/config_snapshot.yaml`. Copy nothing into git.
- **New file:** `benchmarks/rescore_run.py`, also registered as runner suite `rescore`. Arguments: `--checkpoint-root`, `--config`, `--include-base`, `--out`.
- **Procedure:**
  1. Rebuild the evaluation pool **exactly as the evaluator does**: `load_dataset("imdb", split="test").shuffle(seed=42)`, first 4,000 rows, label map 0 → negative, 1 → positive. Call the existing `Evaluator._load_eval_data` instead of re-implementing it. Save the 4,000 ids (row index in the shuffled split) and the SHA-256 of the id list.
  2. For the base model (step 0) and every checkpoint, generate with **the same settings the run used**: `class_match` prefix matching and `max_new_tokens: 6` from `config_snapshot.yaml`'s inference block; the evaluator's own generation config (read `Evaluator._generate_batch_records` and reuse it, do not re-implement it); greedy decoding; fp32.
  3. Save `predictions/<step>.jsonl` with `id, gold, raw_generation, normalized_prediction, correct`.
  4. **Reproduction gate:** final-checkpoint correct count must equal **3,306**, and every step's accuracy must equal the `accuracy` column of `output/imdb/outputs/run10/offline_metrics.csv` to 4 decimals. If any step differs, stop. Report the differing steps and investigate (package versions, tokenizer, dataset revision) before computing anything else. This is what turns the aggregate CSV into per-example evidence.
  5. Then compute per step:
     - `other_rate`;
     - target-class `f1_macro` and legacy `f1_with_other`, which must equal the CSV's `f1` to 4 decimals;
     - `dilution_pct`;
     - MCC (must match the CSV);
     - bootstrap CIs (2,000 resamples, seed 0) for accuracy, F1 and MCC;
     - Wilson CI for accuracy;
     - the exact class balance of the 4,000 pool.
  6. Run `evaluate_calibration` (forced choice, 10-bin ECE, Brier, NLL, AUROC) on the same 4,000 for the base model and the final checkpoint.
  7. Recompute AAUC from the reproduced accuracies with both WP-21 definitions, and check it equals 0.7902.
- **Where:** any Mac (MPS) or Windows CPU. It is 4,000 short generations × 34 models with distilgpt2.
- **Output:** `rescore/summary.json`, `rescore/curve.csv`, `rescore/predictions/`, `rescore/pool_ids.json`.
- **Acceptance:** the gate passes. DS-1, DS-3 and DS-4 then cite this run directory, and its CIs go into the bullets' support notes.

#### WP-20: `streaming_quality` suite: Kafka parity run (L). **Required:** Rohan confirmed run10 did not use Kafka

- **Claims:** DS-1 ("on live data streams" through Kafka), and the Qwen runs reuse this suite (WP-25).
- **Purpose (revised):** run10 already measured 82.65% (section 2A), but it trained in the Colab notebook without Kafka (Rohan, Q11). This run checks that the **real Kafka path** produces the same result. Pass condition: the Kafka run's final accuracy on the same 4,000 pool (use WP-20A's `pool_ids.json`, not a new stratified set) lies within the 95% bootstrap CI of run10's. Use 1 seed at full profile, or 3 if time allows.
- **Depends on:** WP-02, WP-05, WP-07, WP-12, WP-20A, WP-21.
- **Procedure:**
  1. Data: IMDb train (25,000 at full; 1,000 at standard), streamed by the **real** `producer.py` into an isolated topic. The trainer consumes it **once**: one epoch, no repeats. That is what "single-pass" means; state it in the summary.
  2. Trainer: `configs/imdb_quantitative.yaml` with `save_every` such that there are ≥ 10 checkpoints per run (full: `save_every: 150` → 10 checkpoints plus final), and `test_mode: true` so the run stops at EOF.
  3. Evaluation set: IMDb **test** split, stratified 2,000 positive and 2,000 negative = 4,000 (full); 200 at standard. Seeded and pinned. Assert no overlap with the training rows (`assert_no_overlap`).
  4. Score the base model (step 0) and every checkpoint with `evaluate.py --all-checkpoints --run <this run>` (after WP-07), plus `evaluate_calibration` for forced-choice ECE/Brier/NLL/AUROC on the same 4,000.
  5. Metrics per checkpoint:
     - `accuracy` (+ Wilson CI);
     - `f1_macro` (target classes) (+ bootstrap CI);
     - `f1_with_other` (legacy);
     - `dilution_pct = 1 − f1_with_other / f1_macro`;
     - `other_rate`;
     - `mcc` (binary, computed on target labels with off-format predictions counted as wrong, + bootstrap CI). Document exactly how "other" is handled in MCC: map "other" to the label opposite the gold one, which is equivalent to counting it as an error in a 2×2 matrix. Use `sklearn.metrics.matthews_corrcoef` on that mapping;
     - `ece_10bin`, `brier`, `nll`, `auroc` (forced choice);
     - **AAUC** over the checkpoint sequence, using the WP-21 definition, with the base model at step 0.
  6. Machines: full profile × 3 seeds on the MacBook Air or Pro (MPS), or Colab. Standard profile on Windows for a cross-machine check.
- **Output:** `streaming_quality/summary.json` and `streaming_quality/curve.csv` (step → metrics).
- **Generality:** the suite must work for any live config, not only IMDb. It reads the config's evaluation strategy:
  - `class_match` uses the classification metrics above with the stratified IMDb set;
  - `regex_extract` (GSM8K) uses exact-match accuracy + Wilson CI on a fixed, seeded 500-question test sample;
  - the qualitative strategies (Alpaca) use their metrics on the WP-25 holdout.

  WP-25's Qwen runs reuse this suite.
- **Acceptance:**
  - DS-1 keeps "82.7%" only if the final-checkpoint accuracy, averaged over seeds, rounds to 82.7%. Otherwise the bullet uses the measured value. **Do not tune hyperparameters on the test set to hit 82.7%.** If you need tuning, carve a validation set out of the IMDb train split.
  - DS-3 MCC and AAUC: use the measured values.
  - DS-4: report the measured `dilution_pct` and `other_rate`. Expect `dilution_pct ≈ 33.3%` whenever `other_rate > 0`. That follows from the definition (see the §2 DS-4 row and section 10), so the bullet should be worded as a metric fix, not as an empirical effect size.

#### WP-21: fix the AAUC definition (S)

- **Claims:** DS-3.
- **Files:** `utils/eval_metrics_train.py`; extend `tests/test_eval_metrics.py`.
- **Anchor:** `span = last_step - 0.0`.
- **Change:** use `span = last_step - first_step`, where `first_step` is the step of the first valid point. Update the docstring: "Normalized area under the accuracy-vs-step curve between the first and last evaluated steps (trapezoidal), divided by that step span. Equals mean accuracy for uniformly spaced checkpoints with linear interpolation." Also add `aauc_uniform_mean(history)`, the plain mean of accuracies at uniformly spaced evaluation points (the definition of Koh et al., 2022), and report both in WP-20.
- **Tests:**
  - the existing `test_normalized_aauc` (starts at 0) must still pass unchanged;
  - new `test_aauc_first_point_not_zero`: `[(50, 0.5), (100, 1.0)]` gives 0.75. The old code gives 0.375;
  - new `test_aauc_uniform_mean`.

#### WP-22: masking, truncation and clipping tests plus the `ablation` suite (M)

- **Claims:** DS-3 ("prevented prompt memorization and gradient spikes").
- **Files:** new `tests/test_training_core.py`; new suite `ablation` in `benchmarks/run.py`, or `benchmarks/ablation.py` called from it; `trainer.py` (only config switches; no change to defaults).
- **Unit tests** (tiny tokenizer: build `GPT2TokenizerFast` from the `tokenizer.json` in the test fixtures if one exists, otherwise use a minimal vocab; check `tests/` first):
  - `test_prompt_tokens_are_masked_minus_100`: every label position belonging to the prompt is `-100`; response positions equal the input ids.
  - `test_response_preserved_under_truncation`: with prompt length + response length > `max_seq_length`, the response tokens are all kept and the prompt is truncated from the **left**.
  - `test_all_masked_sample_is_rejected_or_skipped`: check what the code does when the response alone exceeds the length, and assert that behaviour.
  - `test_gradient_clipping_caps_global_norm`: a toy model with a huge gradient. After `finish_optimizer_step`, the pre-clip norm is reported and the post-clip norm is ≤ 1.0 + 1e-6. Read `finish_optimizer_step` (`trainer.py:418-449`) to see what it returns.
- **Config switches** (default = today's behaviour):
  - `training.label_masking: true|false`;
  - `training.truncation_side: "left_prompt"|"right"`;
  - `training.max_grad_norm: 1.0` (`null` = no clipping).

  Find where the trainer hard-codes these and route them through config with the same defaults.
- **Ablation suite:** IMDb standard profile, 3 seeds, 4 arms:
  - A: all on (default);
  - B: no masking;
  - C: no clipping;
  - D: right truncation.

  Metrics:
  - final accuracy and MCC on the fixed set;
  - per-step pre-clip grad norm p50/p99/max;
  - "spike" count, defined as steps with pre-clip norm > 5 × the rolling median of the previous 50 steps;
  - fraction of eval responses that copy ≥ 20 consecutive prompt tokens ("prompt regurgitation rate"), which operationalizes "memorization";
  - share of training samples whose response was fully truncated, for arm D.
- **Acceptance:** report all arms with paired bootstrap CIs against arm A. DS-3 may say "prevented … gradient spikes" only if arm C shows more spikes than arm A, and "prompt memorization" only if arm B shows a higher regurgitation rate or lower accuracy, with CIs that exclude zero. Otherwise reword (section 8).

#### WP-23: real ADWIN plus drift detection in the continual suite (M)

- **Claims:** DS-5.
- **Files:** `utils/continual_engine.py`; extend `tests/test_continual_learning.py`; `benchmarks/run.py` (`suite_continual`).
- **Algorithm:** implement ADWIN as in Bifet & Gavaldà (2007), "Learning from Time-Changing Data with Adaptive Windowing", in its exact (non-bucketed) form. Keep the class name `ADWINDriftDetector` and the signature `(delta=0.002, min_window_size=10, max_window_size=200)`, adding `clock: int = 1` and `value_range: tuple = (0.0, 1.0)`.
  1. Normalize each value: `x = clip((v - lo) / (hi - lo), 0, 1)`.
  2. Append to the window and drop the oldest if the length exceeds `max_window_size`.
  3. Every `clock` updates, for every split point `i` from `min_window_size` to `n - min_window_size`, with W0 = the first i values and W1 = the rest:
     - n0 = i, n1 = n − i;
     - m = 1 / (1/n0 + 1/n1);
     - δ' = δ / n;
     - the variance-aware cut ε = sqrt((2/m) · σ² · ln(2/δ')) + (2/(3m)) · ln(2/δ'), where σ² is the variance of the whole window.
     - If |μ0 − μ1| ≥ ε, it is a drift: **drop W0** (keep W1), set `drift_detected = True`, and return True.

     Use prefix sums so that each check is O(n).
  4. Keep `self.width`, `self.total` and `self.variance` as public attributes.

  This replaces the half-split test. The existing test (`test_adwin_drift_detector`: 50 × 0.1 then 5.0) must still pass. With the default range (0, 1), 5.0 clips to 1.0, which is still a large shift.
- **Tests:**
  - existing test unchanged;
  - `test_adwin_no_false_alarm_on_stationary_bernoulli`: 5,000 Bernoulli(0.3) values, seed 0, δ = 0.002, max window 1,000: ≤ 1 detection;
  - `test_adwin_detects_mean_shift_with_bounded_delay`: 1,000 × Bernoulli(0.2), then Bernoulli(0.8); detection within 100 samples of the change;
  - `test_adwin_matches_river_oracle` (optional): if `river` is installed, compare detection indices with `river.drift.ADWIN(delta=0.002)` on the same stream. Expect detection within ±64 samples of river's. Skip **only** if `importlib.util.find_spec("river") is None`; that is the one permitted conditional skip, because it is an optional oracle and not a weakening.
- **Signal to monitor:** training loss is unbounded, so ADWIN's bound does not apply to it directly. Use per-batch **response-token top-1 accuracy**: the fraction of non-masked label positions where `argmax(logits) == label`. It lies in [0, 1] and is computed from logits already available in the forward pass. Add it to the trainer's step metrics (`train_token_acc`).
- **Continual suite wiring** (`suite_continual`, both arms, no_replay and reservoir_replay):
  - feed `train_token_acc` per optimizer step to ADWIN;
  - record detection steps;
  - the true domain boundaries are known (IMDb→Yelp at step b1, Yelp→Amazon at b2); compute `detection_delay_steps` per boundary and `false_alarms` (detections not within [b, b + 100 steps]);
  - keep the existing BWT and forgetting metrics on the 600 held-out examples (200 per domain at standard).
- **Acceptance:** report the detection delays, false alarms, BWT and forgetting for both arms, plus the paired bootstrap CI of the BWT difference (replay − no replay). DS-5 may say "mitigated catastrophic forgetting" only if that CI excludes 0 in the right direction.
- **Optional extension (state it in the report, do not do it by default):** let ADWIN trigger an action, such as raising the replay fraction from 20% to 40% for 200 steps. Only do this if Rohan answers Q5 with yes.

#### WP-24: E2E fixed-set before/after with a corrected fact scorer (L)

- **Claims:** DS-2.
- **Depends on:** WP-01, WP-07.
- **Revised scope:** the 12 trained E2E checkpoints already exist (section 2A), so **do not retrain**. Re-score base (step 0) and every checkpoint in `output/e2e/checkpoints/gpt2-medium__parquet/run_20260414-220458_974a/` on one fixed MR set. Also reproduce the logged sliding-window metrics once, as a sanity check: the step-0 `qual_slot_coverage_mean` must come out as 0.1958 and the final `qual_slot_familyFriendly_inversion_rate` as 0.0120, using the run's own config `output/e2e/config.yaml` and its 150 × 3 window. Sampling makes exact equality unlikely, so report the difference rather than gating on it.
- **Files:** `utils/eval_qualitative.py` (`StructuredSlotCoverage`); new `benchmarks/e2e_facts.py`; new `tests/test_e2e_facts.py`.
- **Changes:**
  1. Parse the MR into slots. The E2E MR format is `name[The Eagle], eatType[coffee shop], food[French], priceRange[moderate], customer rating[3 out of 5], area[riverside], familyFriendly[yes], near[Burger King]`. These are the 8 standard slots; confirm them in the data from WP-01.
  2. For each slot, return one of `expressed_correct`, `omitted` or `contradicted`:
     - **name / near:** case-insensitive substring match. `contradicted` never applies.
     - **eatType / food / area:** an alias table of canonical value → surface strings (for example `coffee shop` → {"coffee shop", "café", "cafe"}). `contradicted` if a *different* known value of the same slot appears and the correct one does not.
     - **priceRange:** an ordinal alias table (`cheap`/`less than £20`; `moderate`/`£20-25`; `high`/`expensive`/`more than £30`). `contradicted` if another bucket's alias appears.
     - **customer rating:** the same approach (`low`/`1 out of 5`, `average`/`3 out of 5`, `high`/`5 out of 5`).
     - **familyFriendly:** detect affirmative ("family friendly", "kid friendly", "children welcome", …) and negated forms ("not family friendly", "isn't kid friendly", "no children", "adults only", …). `yes` + affirmative → correct; `yes` + negated → contradicted; `no` + negated → correct; `no` + affirmative → contradicted; neither → **omitted**. **This fixes the known bug where omission counted as correct.**
  3. Per-example metrics:
     - `required_detail_recall = correct / slots`;
     - `contradiction_any = any slot contradicted`;
     - `fully_correct = all slots correct`.
  4. Aggregate: mean recall, contradiction rate (fraction of descriptions with ≥ 1 contradiction), fully-correct rate, and per-slot contradiction rates, each with a bootstrap CI. **Report the familyFriendly inversion rate separately**, because it is what the old 15.8%/1.2% numbers measured.
  5. **Human validation:** export 100 seeded random (MR, output, scorer labels) rows from the final checkpoint and 100 from the base model to `e2e_facts/human_validation.csv`, with empty columns for Rohan's labels. A helper `benchmarks/e2e_facts.py agreement <csv>` computes per-slot Cohen's kappa between the scorer and the human labels. **DS-2 numbers may go on the resume only if kappa ≥ 0.8 for every slot used** (section 8).
  6. Before/after: base gpt2-medium versus the **existing** final checkpoint (and the full curve over the saved checkpoints). Score both on the **same fixed set**: all unique validation MRs (1,484 per §13; confirm in WP-01). Use greedy decoding, the same `max_new_tokens`, and the same prompt template as `configs/e2e_qualitative.yaml`.
- **Suite:** register `e2e_facts` in `benchmarks/run.py`. It calls `benchmarks/e2e_facts.py` for generation and scoring, and `--e2e-checkpoint` selects the adapted model.
- **Tests:** about 25 hand-written cases covering each slot's correct, omitted and contradicted outcomes, including "not family-friendly" with an MR of `yes` → contradicted, and no mention → omitted.
- **Machines:** training on Colab (L4 or A100) or the MacBook Pro (gpt2-medium on MPS is feasible); scoring anywhere.
- **Acceptance:** `e2e_facts/summary.json` with base and final metrics, CIs and n. Then apply the DS-2 rules in section 8.

#### WP-25: GSM8K and Alpaca fixes plus Qwen suites (M)

- **Claims:** DS-1 (Qwen2.5-3B, 4 tasks), DS-6.
- **Files:** `utils/eval_metrics_train.py`, `utils/eval_qualitative.py`, `configs/alpaca_qualitative.yaml`, `configs/gsm8k_*.yaml`, `producer.py` (holdout exclusion), new `tests/test_gsm8k_extract.py` and `tests/test_holdout_split.py`.
- **Changes:**
  1. **GSM8K extraction:** add `extract_final_number(text) -> Optional[str]` to `utils/eval_metrics_train.py`, and add `answer_normalizer: "numeric"` to both GSM8K configs. When the `regex_extract` strategy sees `answer_normalizer: "numeric"`, it uses `extract_final_number` on both prediction and gold instead of the config's `answer_regex` (find where `answer_regex` is read and branch there; keep the regex path unchanged for any config without the new key). First try `####\s*(-?[\d,]*\.?\d+)`. Otherwise take the last match of `-?[\d,]*\.?\d+` in the text. Strip commas, and normalize `"2.50"` → `"2.5"` and `"3.0"` → `"3"` with `Decimal(...).normalize()` (handling the exponent form). Compare normalized strings. Tests: `1,200`, `-3`, `2.5`, `$1,200.00`, `#### 18`, no number → `None`.
  2. **Alpaca holdout:** add `dataset.holdout: {method: "hash", fraction: 0.02, seed: 42}`. A row belongs to the holdout if `int(sha256(f"{seed}:{instruction}\n{input}").hexdigest(), 16) % 10_000 < fraction * 10_000`. The producer **skips** holdout rows and the evaluator **uses only** holdout rows. Set `eval_split: train` + `holdout` in the Alpaca config, and log the counts. Tests: the two sets are disjoint, the split is deterministic and has the right size ± 10%.
  3. **QAFactEval naming:** rename the metric keys to `qual_qafacteval_style_*` and keep the old keys as aliases for one release, with a log line saying so. In the docstring, explain what the scorer is: "QA-overlap proxy inspired by QAFactEval; answer selection and question generation differ from the published pipeline."
  4. **Qwen suites** (run via the existing configs on Colab; section 7):
     - GSM8K quantitative: Qwen2.5-3B, base versus after a single pass over GSM8K train (all rows from WP-01, streamed through Kafka), exact-match accuracy on a fixed 500-question sample of the test set (seeded) + Wilson CI.
     - GSM8K qualitative: structural CoT metrics on the same 500.
     - Alpaca: Qwen2.5-1.5B, semantic similarity and the QA-style score on the holdout (≈ 1,040 rows at 2%; use all of them).
- **Acceptance:** tests pass; summaries exist for each run. DS-6 uses the measured values and the "QAFactEval-style" naming unless Rohan answers Q2 otherwise.

---

### Phase 5: devices, Colab and split deployment

#### WP-26: `resolve_device`, half precision on MPS, experimental XPU (M)

- **Claims:** enables Mac and Colab runs for all claims.
- **Files:** new `utils/device.py`; `utils/precision_manager.py`; `trainer.py`, `inference.py`, `evaluate.py`, `benchmarks/repro_eval.py` (`load_model`); new `tests/test_device.py`.
- **Changes:**
  1. Implement `resolve_device(config, cli_device=None) -> str` with the precedence in §4.3. `"xpu"` is returned only when asked for explicitly and `torch.xpu.is_available()`; otherwise raise `RuntimeError("XPU requested but not available; see section 7 of plan 04")`.
  2. In `precision_manager`, keep forcing fp32 on CPU. On MPS, if `model.allow_half_on_mps` is true, permit `bf16`, falling back to `fp16` if a one-time bf16 matmul probe raises. Log the decision.
  3. Replace each ad-hoc device choice in the listed files with `resolve_device`. Search for `torch.cuda.is_available()` and `mps` to find them, and list every changed line in the report.
  4. Add `--device` to `trainer.py`, `inference.py` and `evaluate.py` (and to `benchmarks/run.py` as `--device`, passed to child processes through `INFINITUNE_DEVICE`).
  5. XPU (Windows, experimental): document in `docs/` that it needs a **separate venv** with the Intel XPU build of PyTorch, installed from PyTorch's XPU wheel index. Confirm the current command on pytorch.org; do not change `requirements-eval.txt` for it. Results from XPU are never used for resume claims unless Rohan decides otherwise (Q4).
- **Tests:** precedence order with env and CLI; XPU requested but unavailable raises; MPS half precision is refused without the flag. Monkeypatch the `torch.backends.mps.is_available` and `torch.cuda.is_available` functions.
- **Acceptance:** `python -m benchmarks.run run --suites precision --profile standard --device mps` on the MacBook Pro reports finite losses for fp32 and bf16 (or fp16). Record its summary.

#### WP-27: Colab notebook v2 that runs the real framework (M)

- **Claims:** DS-1, DS-6, SDE-1, SDE-4 on GPU.
- **Files:** new `notebooks/infinitune_colab_v2.ipynb` (keep the old `infinitune_colab.ipynb` unchanged); new `requirements-colab.txt`; new `scripts/colab/start_kafka.sh`.
- **Why:** the existing notebook re-implements the training loop **without Kafka** ("Direct PyTorch Generative Iterator… mimics your Kafka architecture", cells 8-9). Results from it cannot back "on live data streams" claims. v2 runs the repo's own `producer.py`, `trainer.py`, `inference.py` and `benchmarks/run.py`.
- **Cells** (each a separate cell, with markdown above each saying what it does):
  1. GPU check: `nvidia-smi`; print `torch.cuda.get_device_name(0)` and `get_device_capability(0)`; choose bf16 or fp16 per §4.3.
  2. Mount Google Drive. Set `PERSIST=/content/drive/MyDrive/infinitune`.
  3. Clone the repo **at a pinned commit** (`git clone … && git checkout <sha>`); print the SHA.
  4. `pip install -r requirements-colab.txt`. **Do not reinstall torch**: Colab's preinstalled CUDA build is used and its version is recorded. `requirements-colab.txt` = `requirements-eval.txt` minus the `torch`, `numpy` and `scikit-learn` lines (use Colab's own, so that pip does not downgrade numpy and break Colab's preinstalled packages). Run `benchmarks/env_check.py`.
  5. Kafka: `bash scripts/colab/start_kafka.sh`, which:
     - installs a headless Java 17 JRE with apt;
     - downloads Kafka **3.9.x** from `https://archive.apache.org/dist/kafka/` (check the exact file name exists);
     - formats KRaft storage under `/content/kafka-data`;
     - starts the broker in the background with `KAFKA_HEAP_OPTS="-Xmx1G -Xms1G"`, logging to `/content/kafka.log`;
     - polls `kafka-topics.sh --list` until it succeeds or 60 s pass.
  6. Set `HF_HOME=$PERSIST/hf_cache`, so model downloads survive disconnects. Optional `HF_TOKEN` from Colab Secrets (`google.colab.userdata.get`); never print it.
  7. Run a suite, for example: `python -m benchmarks.run run --suites streaming_quality --profile full --model Qwen/Qwen2.5-3B --live-config configs/gsm8k_quantitative.yaml --device cuda --kafka-bootstrap localhost:9092 --run-dir $PERSIST/runs/<name>`.
  8. Resume after a disconnect: the same command with the same `--run-dir`. The runner is already resumable.
  9. Zip `$PERSIST/runs/<name>` for download. It is already on Drive.
- **Acceptance:** Rohan runs cells 1-6 and the `integrity` and `pipeline` smoke suites on a T4, and both pass.

#### WP-28: split deployment (Colab trains, laptop serves) (M)

- **Claims:** SDE-1 (distributed), and the demo Rohan asked for.
- **Depends on:** WP-10, WP-12, WP-26, WP-27.
- **Mode B (recommended), managed Kafka:**
  - Colab runs `producer.py` and `trainer.py`; the laptop runs `inference.py`. Both point at a managed Kafka cluster over `SASL_SSL` (section 7.4).
  - Integrity mode `hmac`, with the same `INFINITUNE_ADAPTER_HMAC_KEY` set on both sides through Colab Secrets and a local env var.
  - The `propagation` suite gets `--split-mode`: the trainer side and the inference side each write their own timestamps. A merge script, `benchmarks/merge_split_propagation.py`, joins them by `(run_id, seq)` and applies the clock-offset correction (section 7.5).
- **Mode C, Drive sync (no Kafka across the internet):**
  - Colab saves checkpoints to a Drive folder. The laptop syncs it with Google Drive for desktop. Add `inference.py --watch-checkpoints DIR`, which polls for new checkpoint directories that contain a `COMMITTED` marker.
  - The trainer writes the marker **last**, after `adapter_manifest.create_and_save_adapter_artifact` has written the weights and `manifest.json`. Then verify the SHA-256 (`AdapterManifest.verify_artifact`) and swap through `router.swap_from_manifest`.
  - Propagation here is dominated by Drive sync time. **Do not use Mode C for SDE-1 numbers**; it is a demo and fallback path.
- **Tests (Mode C):** `test_watcher_ignores_dir_without_committed_marker`, `test_watcher_rejects_checksum_mismatch`, `test_watcher_applies_new_committed_checkpoint_once` (using a tiny model and `tmp_path`).

---

### Phase 6: evidence ledger and documentation

#### WP-29: evidence ledger (S)

- **Claims:** all.
- **Files:** new `benchmarks/claims.py`; new `reports/claims/CLAIMS.md` (generated).
- **Changes:**
  1. A table in code, `CLAIMS = {"SDE-1": {"suite": "propagation", "metric": "propagation_s_p95", "threshold": ("<", 8.0), ...}, ...}`, with one entry per numeric claim. Section 8 gives the full list.
  2. `python -m benchmarks.claims build --runs reports/benchmark_runs/*` reads each run's `summary.json` files and writes `reports/claims/CLAIMS.md`. For each claim it gives:
     - the measured value and CI;
     - n;
     - machine;
     - run dir;
     - the code commit (from `run_manifest.json` provenance);
     - PASS / FAIL / NOT RUN against the threshold;
     - the suggested bullet text, from the section 8 templates filled in with the measured value.
  3. `reports/claims/CLAIMS.md` may be committed. It contains only aggregate numbers and paths, never data.
- **Tests:** a fake run directory produces the expected PASS/FAIL rows.

#### WP-30: update `docs/Infinitune_Project_Context.md` (S)

Do this **after** the runs, and edit the context doc only through these changes:

1. **§4 Repository Structure:** add the new files:
   - `utils/device.py`, `utils/kafka_config.py`, `utils/serving_policy.py` (if created);
   - `benchmarks/env_check.py`, `dataset_census.py`, `eval_speed.py`, `e2e_facts.py`, `claims.py`, `merge_split_propagation.py`;
   - `notebooks/infinitune_colab_v2.ipynb`, `scripts/colab/start_kafka.sh`, `requirements-colab.txt`;
   - the new test files.
2. **§6/§11 Inference Server Internals:**
   - replace the wire-format description with protocol v2 (WP-10);
   - document `/adapter`, `/adapter/history`, the new `/metrics` keys (WP-09, WP-04), the listening decision table (WP-03) and `--watch-checkpoints` (WP-28).
3. **§7 Configuration System:** document every new key with its default:
   - `kafka.integrity`, `kafka.client.*`, `kafka.max_request_size`, `kafka.dead_letter_topic`;
   - `data.filtering.on_error`;
   - `training.label_masking`, `training.truncation_side`, `training.max_grad_norm`;
   - `testing_strategy.generation_mode`;
   - `dataset.holdout`;
   - `model.allow_half_on_mps`, `model.device`, `model.revision`.

   Mark which defaults changed (WP-06, WP-10).
4. **§10 Training Internals:** malformed-record handling (WP-05), `train_token_acc` (WP-23), the real ADWIN.
5. **§13 Evidence Ledger:**
   - add one row per new suite result, citing the run directory, the machine and n;
   - mark §13.4 open issues as resolved where a WP fixed them: HMAC on live Kafka (WP-10), StreamFilter fail-open (WP-06), ADWIN not wired (WP-23), TTFT not measured (still open unless done);
   - **do not delete** the historical rows; add a "Superseded by" column instead.
6. **§15 Setup:**
   - pin Kafka 3.9.x on all OSes (WP-00 policy);
   - add a short "Colab" subsection pointing to section 7 of this plan;
   - document the env vars `INFINITUNE_DEVICE`, `INFINITUNE_KAFKA_BOOTSTRAP`, `INFINITUNE_KAFKA_USERNAME/PASSWORD` and `INFINITUNE_ADAPTER_HMAC_KEY`.
7. **New §19, "Resume claim evidence map":** a table of the 12 bullet IDs → current wording → status (PASS/FAIL/NOT RUN) → the `reports/claims/CLAIMS.md` row. It should state that `CLAIMS.md` is the source of truth.
8. Update the table of contents (lines 7-92) for §19.

#### WP-31: keep the testing guide current (S)

`docs/ImplementationPlan/05_Resume_Claims_Testing_Guide.md` lists every command. Whenever a WP changes a command, flag or expected output, update that guide in the same commit.

---

## 6. Execution order and dependency graph

```
Phase 0:  WP-00 ──► WP-01
Phase 1:  WP-02 ──► WP-03 ──┐
          WP-04 ────────────┤
          WP-05, WP-06 ─────┤
          WP-02 ──► WP-07 ──┤
          WP-08, WP-09 ─────┤
Phase 2:  WP-04 ──► WP-10 ──► WP-11
          WP-12 (independent)
Phase 3:  WP-13 (needs 03,10,12)   WP-14 (02,03,11)   WP-15 (09,11)
          WP-16 (04,10,11,15)      WP-17 (none)        WP-18 (07)
          WP-19 (01,05,06)
Phase 4:  WP-20A (02,08) ──► WP-21 ──► WP-20 parity (02,05,07,12; only if needed)   WP-22   WP-23   WP-24 (01,07)   WP-25
Phase 5:  WP-26 ──► WP-27 ──► WP-28 (10,12)
Phase 6:  WP-29 after the first suite results; WP-30 at the end; WP-31 continuously
```

**Recommended sequence for one agent at a time:** 00, 01, 02, 08, 20A (early: cheap, and it locks down the DS evidence), 03, 04, 05, 06, 07, 09, 21, 10, 11, 12, 26, 15, 16, 14, 13, 17, 18, 19, 22, 23, 20, 24, 25, 27, 28, 29, 30.

**Parallel-safe groups** (different files): {04, 05, 06}, {09, 21}, {17, 19, 22}. Do not run WPs that touch `trainer.py` or `inference.py` in parallel.

**Where to run what:**

| WP | Code written on | Must be run on |
|---|---|---|
| 00 | any | all 4 machines |
| 01, 19A, 21, 24 (scoring) | any | any |
| 13 | any | Windows, MacBook Pro, Colab |
| 14, 15, 16 | any | Windows and MacBook Pro |
| 17 | any | **Colab T4** (claim); others optional |
| 20 | any | MacBook Air/Pro (full) or Colab; Windows (standard) |
| 22, 23 | any | MacBook Air (standard, 3 seeds) |
| 24 (training), 25 | any | Colab (Qwen); MacBook Pro possible for gpt2-medium |
| 26 | any | MacBook Pro |
| 27, 28 | any | Colab plus Windows/Mac |

---

## 7. Colab testing

### 7.1 What Colab is for

Colab is used for the runs the laptops cannot do well:
- Qwen2.5-3B training (DS-1, DS-6);
- the T4 evaluation-speed claim (SDE-4);
- full-size E2E training if the Mac is too slow.

Laptops remain the primary place for systems tests (swap, fault, cold start), because those measure the serving runtime that Rohan runs locally.

### 7.2 Choosing a runtime

| GPU | VRAM | Precision | Good for |
|---|---|---|---|
| T4 | 16 GB | fp16 + GradScaler (no hardware bf16) | `eval_speed` (SDE-4), gpt2-medium E2E, Qwen2.5-1.5B |
| L4 | 24 GB | bf16 | Qwen2.5-3B LoRA training at seq 512, batch 2, gradient accumulation 4 |
| A100 | 40/80 GB | bf16 | Fastest Qwen2.5-3B and full E2E; use when credits allow |

Compute-unit cost per hour differs by GPU and changes over time. Check the Colab "Resources" panel before a long run, and use the smoke profile first to estimate per-step time. Multiply by the number of steps to plan credits.

### 7.3 Mode A: everything inside one Colab VM

Run Kafka, producer, trainer, inference and evaluation in the same VM with WP-27's notebook. This is what DS-1, DS-6 and the Colab rows of SDE-1 use. Steps:
1. Run notebook cells 1-6.
2. Run `integrity` and `pipeline` on the smoke profile. Both must pass before spending credits.
3. Run the target suite with `--run-dir` on Drive.
4. If the runtime disconnects, reconnect, rerun cells 1-6, then rerun the same suite command. Completed stages are skipped.

Colab notes:
- Background processes (the Kafka broker, the inference server) die when the runtime restarts. The suites start and stop their own trainer and inference processes; only the broker comes from cell 5.
- Keep `/content/kafka-data` off Drive. Kafka on a FUSE-mounted Drive is slow and can corrupt logs. Only results and the HF cache go to Drive.

### 7.4 Mode B: Colab trains, the laptop serves, over managed Kafka

1. Create a managed Kafka cluster that supports SASL/SSL. Options: Confluent Cloud, Redpanda Cloud (Serverless) and Aiven for Apache Kafka. **Check each provider's current free tier and limits yourself**: these change, and this plan does not assume any. Check the maximum message size: adapter frames are under 70 KB each for the shipped configs (WP-10's size guard checks this), but some providers cap message size or throughput on free tiers.
2. Create two topics, for training data and adapter updates. Use 1 partition for the adapter topic (the protocol needs a single ordered partition) and short retention (for example 1 hour) to save storage.
3. Credentials: put the API key and secret in Colab Secrets as `INFINITUNE_KAFKA_USERNAME` and `INFINITUNE_KAFKA_PASSWORD`, and set the same names as env vars on the laptop. Never put them in YAML (WP-12 enforces this). Generate `INFINITUNE_ADAPTER_HMAC_KEY` once (`python -c "import secrets; print(secrets.token_hex(32))"`) and set it on both sides.
4. Config overlay `configs/overlays/managed_kafka.yaml`:
   ```yaml
   kafka:
     client:
       security_protocol: SASL_SSL
       sasl_mechanism: PLAIN
     integrity: hmac
   ```
   Set the bootstrap with env `INFINITUNE_KAFKA_BOOTSTRAP`. Note: WP-12 must implement overlay merging (`--config base.yaml --config-overlay overlay.yaml`) if Rohan wants overlays; otherwise copy the base config and edit it. Default: implement the overlay as a deep-merge in a small helper, `utils/config_overlay.py`, as part of WP-12.
5. Start the laptop's inference server first (`python inference.py --config <cfg> --listen-updates`). Then in Colab start the trainer, then the producer.
6. The size of an adapter push for Qwen2.5-3B with r=8 on q/k/v/o modules is roughly 14-15 MB in fp32 (INFERRED from public model dimensions; WP-10's size guard logs the exact total on the first push). At a 60-second push interval that is low bandwidth, but count it against the provider's limits.

### 7.5 Measuring propagation across two machines

The trainer's clock (Colab) and the server's clock (laptop) differ. To correct for this:
1. On both sides, at the start and end of the run, record the offset to a public NTP server: `ntplib.NTPClient().request("time.google.com").offset`. `ntplib` is a new, small, pure-Python dependency, added to `requirements-colab.txt` and `requirements-eval.txt` by WP-28 under rule R5.
2. Corrected `t_applied_on_colab_clock = t_applied_laptop − offset_laptop + offset_colab` (offsets as returned by ntplib are server − local).
3. Report propagation and both offsets. If the offset changes by more than 50 ms between the start and end of the run, flag the run.
4. Report the split-mode numbers separately from same-machine numbers. A resume claim must name which one it uses.

### 7.6 Mode C: Drive sync fallback

Use it when no managed Kafka is available (WP-28, Mode C). It is good for a demo of "trained in the cloud, served locally". It is **not** a low-latency path, so never quote a propagation number from it.

### 7.7 What not to do on Colab

- Do not expose a laptop Kafka broker or inference server to the internet through tunnels.
- Do not `pip install torch==…` on Colab; it replaces the CUDA build. Do not downgrade numpy there.
- Do not store `.env` or credentials in Drive files; use Colab Secrets.

---

### 7.8 Testing locally and on Colab at the same time

Yes, local and Colab testing can run in parallel once the code changes are merged, as long as these rules hold:

1. **One commit everywhere.** Every machine checks out the same git commit, and each run manifest records it. A result from a different commit counts as a different experiment.
2. **Separate brokers.** In Mode A, Colab runs its own Kafka broker inside the VM, and each laptop runs its own. Nothing is shared, so the runs cannot disturb each other. Shared managed Kafka is only for the split Mode B test.
3. **Separate run directories.** Colab writes to `MyDrive/infinitune/runs/<machine>_<suite>_<date>`, and laptops write to `reports/benchmark_runs/`. Never point two machines at the same `--run-dir`.
4. **Disjoint suite assignment.** Run suites in parallel like this:

| Track | Machine | Suites |
|---|---|---|
| A | Windows laptop | unit tests, `integrity`, `fault_injection`, `swap_stress`, `cold_start`, `filter_benchmark`, `propagation` (distilgpt2, gpt2-medium) |
| B | MacBook Pro | `rescore` (WP-20A), `propagation` (Qwen2.5-1.5B), `swap_stress`, `fault_injection` (gpt2-medium), `precision` |
| C | MacBook Air | `ablation`, `continual` (ADWIN), `e2e_facts` scoring |
| D | Colab T4 | `eval_speed` (SDE-4) |
| E | Colab L4 or A100 | `streaming_quality` Kafka parity (IMDb), Qwen2.5-3B GSM8K and Alpaca (WP-25), `propagation` Qwen2.5-3B |

5. **Timing runs stay quiet.** A laptop running a timing suite (propagation, swap, cold start) runs nothing else at the same time. Colab work does not count, because it runs remotely.
6. **Merge at the end.** Copy Colab run directories from Drive into `reports/benchmark_runs/`, then run `python -m benchmarks.claims build` once over everything (WP-29).

### 7.9 Setting up Colab, step by step

Before WP-27 exists, steps 1-4 still work; steps 5-8 need WP-27's notebook and script.

1. **Get a GPU.** Open https://colab.research.google.com, choose File → New notebook (or open `notebooks/infinitune_colab_v2.ipynb` from GitHub once WP-27 is merged), then Runtime → Change runtime type and pick T4, L4 or A100.
2. **Mount Drive** so results survive disconnects:
   ```python
   from google.colab import drive; drive.mount('/content/drive')
   PERSIST = '/content/drive/MyDrive/infinitune'
   ```
3. **Clone the repo at the agreed commit:**
   ```bash
   !git clone https://github.com/rohang1411/infinitune-realtime-llm-fine-tuning-framework.git /content/infinitune
   %cd /content/infinitune
   !git checkout <commit-sha>
   ```
   The repository is private, so cloning needs a GitHub token. Store it in Colab Secrets (the key icon) as `GH_TOKEN`, and clone with `https://$GH_TOKEN@github.com/...`. Never paste it into a cell.
4. **Install dependencies without replacing Colab's torch or numpy:**
   ```bash
   !pip install -q -r requirements-colab.txt
   !python -m benchmarks.env_check
   ```
5. **Start Kafka inside the VM:**
   ```bash
   !bash scripts/colab/start_kafka.sh
   ```
   This script (WP-27) installs Java 17, downloads Kafka 3.9.x, formats KRaft storage under `/content/kafka-data` and starts the broker. This is the step that was missing in the earlier Colab runs.
6. **Point caches at Drive:** `%env HF_HOME=/content/drive/MyDrive/infinitune/hf_cache`.
7. **Run the smoke check first:**
   ```bash
   !python -m benchmarks.run run --suites integrity pipeline --profile smoke --kafka-bootstrap localhost:9092 --device cuda --run-dir $PERSIST/runs/colab_smoke
   ```
8. **Run the assigned track** (7.8 tracks D or E), always with `--run-dir` on Drive. After a disconnect, rerun steps 2-6 and the same command with the same `--run-dir`; the runner resumes.

### 7.10 Who can do the Colab setup

- **Claude can write everything that runs inside Colab:** the notebook, `start_kafka.sh`, `requirements-colab.txt`, and the exact commands (WP-27).
- **Claude cannot sign in to Google for Rohan, approve Colab's prompts, or buy compute units.** Rohan signs in, picks the GPU, approves the Drive mount and adds the GitHub token to Colab Secrets. Those steps involve his account and credentials.
- **Driving the browser:** this thread's tools include desktop control of Rohan's Windows laptop through the Claude desktop app (screenshots, clicks, typing), each app approved by Rohan first. With that, Claude could open Colab in his already-signed-in browser, open the notebook, choose the runtime and run the cells, watching the output.
  - It is slower and less reliable than Rohan clicking "Run all": each step is a screenshot round-trip, and Colab dialogs can change.
  - The laptop must stay awake with the browser open.
  - It cannot enter passwords or 2FA codes.
- **Recommended split:** once WP-27 is merged, Rohan does the one-time account steps (sign in, GPU, Drive, token, roughly 5 minutes). Claude then either drives the notebook via desktop control, or Rohan presses Run all. Claude reads results from Drive once Rohan syncs them, or from the laptop folder.

## 8. Claim acceptance rules and rewording templates

Each rule says when the current wording can stay. Otherwise use the template, filled in with measured values from `reports/claims/CLAIMS.md`. Never round in the claim's favour: 0.8265 becomes 82.7%, and 15.73% becomes 15.7%, not 15.8%.

| Claim | Keep the current wording if | Otherwise use |
|---|---|---|
| SDE-1 "<8 seconds" | `propagation_s` p95 < 8 s for **Qwen2.5-3B** on a named machine, n ≥ 20, **and** a GPT-2-family model was also run live | "…propagating LoRA weight updates to live serving in **X s (p95)** on <machine> for <model>". Name DistilGPT-2 if that is what ran |
| SDE-2 "140 ms" | `swap_lock_held_ms` p50 ≈ 140 ms, **or** the bullet says "end-to-end swap including in-flight request drain" and the total p50 ≈ 140 ms | "…in-memory LoRA swaps holding the write lock for **X ms (p50)**" |
| SDE-2 "zero dropped requests" | `request_failures == 0` and `dropped == 0` across all WP-15 runs (state the total request count) | Report the failure rate |
| SDE-2 "no 60s cold-start delay" | WP-14 arm A serves trained weights on its first healthy request, while arm B waits ≈ `weight_push_interval` | "…restarts serve the latest trained adapter immediately instead of base weights until the next 60 s push" |
| SDE-3 | WP-10 is merged and the WP-16 rows F01, F02, F07, F09, F12 pass | — (the wording is accurate once implemented) |
| SDE-4 "90% (40m to <4m)" | No original timing log exists (Q12), so **WP-17 on a T4 is required**: `reduction_pct ≥ 0.90`, sequential ≈ 40 min (36-44), batched < 4 min | "…cutting **3,340-generation** evaluation from **A min to B min (C%)** on a T4" |
| SDE-4 "offline auditing service" | WP-18 merged | — |
| SDE-5 | WP-16: all rows pass in 3 repeats; `request_failures == 0`; `corrupted_states == 0` | — |
| SDE-6 "5-stage … 52,000+" | WP-19 ran with all 5 stages active over the full Alpaca set (count ≥ 52,000 from WP-01); live Part B passed | Use the real record count and the real number of active stages |
| DS-1 "82.7%" | **Already supported by run10** (0.8265 on 4,000; section 2A). Keep it once WP-20A's reproduction gate passes. Run10 did **not** use Kafka (Q11), so "Apache Kafka … on live data streams" stays only if the WP-20 parity run passes | If neither Kafka condition holds: "…reaching **82.7% single-pass streaming accuracy** on 4,000 held-out IMDb reviews", with Kafka described as the platform's transport but not as the path of this result |
| DS-1 "Qwen2.5-3B … 4 NLP tasks" | Decision (Q13): plan the 3B run on Colab (WP-25); **say Qwen2.5-1.5B until it exists**. Keep "4 tasks" once runs exist for all 4 task configs. Today only IMDb (distilgpt2, Qwen2.5-1.5B) and E2E (gpt2-medium) have run artifacts | "…adapt LLMs (**Qwen2.5-1.5B**, GPT-2) across **2 NLP tasks**…" until more runs exist |
| DS-2 "20% to 90%" | Logged (19.58% → 89.88%). It is a heuristic coverage proxy on sliding windows, so call it "attribute coverage", not "verified". Upgrade to "required-detail recall on N fixed inputs" after WP-24 with kappa ≥ 0.8 | "…raising input-attribute coverage from 20% to 90%" (current, logged) |
| DS-2 "15.8% to 1.2%" | **Change now.** The log shows a 15.73% **peak at step 400**, not a starting value | "…cutting family-friendliness inversions from a **15.7% mid-training peak to 1.2%**". After WP-24, the all-slot contradiction rate may replace it |
| DS-2 "42,000+ records" | WP-01 E2E train count ≥ 42,000 **and** the run's final step × 16 ≈ that count | Use the real count (likely 33,000+; section 2A) |
| DS-3 "0.69 MCC, 0.79 AAUC" | **Already supported by run10** (0.6894, 0.7902 on 4,000). Keep | — |
| DS-3 "Prevented prompt memorization and gradient spikes" | WP-22 ablation shows the effects with CIs excluding 0 | "Implemented prompt-masked loss, response-preserving truncation and gradient clipping for single-pass streaming SFT…" |
| DS-4 "Eliminated 33% Macro F1 dilution" | Always reword, because the 33% is a definitional consequence (§2) | "Fixed a macro-F1 bug where off-format LLM outputs created a phantom class that understated F1 by a third, reporting off-format rate separately alongside 10-bin ECE and 95% bootstrap CIs on 4,000 samples" |
| DS-5 | WP-23: real ADWIN wired into training; BWT difference CI excludes 0 | "…evaluated replay vs no replay (BWT A vs B) and ADWIN drift detection (mean delay N steps, F false alarms)" |
| DS-6 "QAFactEval" | Rohan answers Q2 with "rename" (the default) | "QAFactEval-style extractive QA scoring" |
| DS-6 "52K Alpaca" | WP-01 count ≥ 52,000 | Use the real count |

---

## 9. Review pass 1: senior AI engineer

Each finding names the WP it was folded into.

1. **"<8 s" needs a precise boundary.** Two different numbers existed: record-to-version (7.593 s) and snapshot-to-visible (0.844 s). Both are kept, but the resume number is defined as publish-start to lock-released, measured server-side (WP-13).
2. **Polling-based timing adds error.** The freshness suite polls `/metrics` every 100 ms. Server-side timestamps replace polling (WP-13).
3. **Swap timing mixed queueing with work.** Fixed in WP-09; the claim must say which number it is (§8).
4. **Restart regressions.** The fixed inference group id plus committed offsets can re-apply old snapshots after a cold start. The monotonic `(run_id, seq)` guard was added (WP-10).
5. **Poison-pill records.** Deserializers that raise inside `poll()` kill both the trainer and inference consumers. Raw bytes plus per-record decode were added (WP-04, WP-05).
6. **Kafka message size.** Per-tensor frames keep each message small for all shipped configs. A future MLP-targeted config with a larger rank could exceed the 1 MB default; a preflight guard was added (WP-10) together with a configurable `max_request_size` (WP-12).
7. **Qwen2.5-3B memory on the Mac.** Two fp32 copies (trainer plus server) do not fit in 24 GB. Opt-in half-precision base weights with fp32 adapters were added, gated by the precision suite (WP-26).
8. **MPS bf16 support** varies by macOS version. A one-time probe with fp16 fallback was added, and the decision is logged (WP-26).
9. **Colab notebook bypasses Kafka.** Its results cannot support "live stream" claims. v2 runs the real processes (WP-27).
10. **kafka-python 2.0.2** is old. It possibly fails on Python 3.12 and possibly against Kafka 4.x brokers (INFERRED). WP-00 verifies this and updates the pin only with passing smoke runs.
11. **XPU** on Windows is promising but unproven for this stack. It is kept as an explicit, separate-venv experiment that never feeds claims by default (WP-26, Q4).

## 10. Review pass 2: senior data scientist

1. **82.7% has no per-example evidence.** A full rerun with a fixed 4,000-sample set and per-example predictions is required, with seeds and CIs, and no tuning on the test set (WP-20).
2. **The 33% dilution is a tautology.** With two target classes plus a phantom "other" class whose F1 is always 0, the legacy macro F1 is roughly two-thirds of the clean one whenever any off-format output exists. The bullet must describe a metric fix, not an empirical effect size (§2, §8).
3. **MCC with off-format outputs** must have a stated convention: counted as errors via the opposite-label mapping (WP-20).
4. **AAUC was biased** when the first point is not at step 0, and evaluate.py stamped every checkpoint as step 0. Both fixed (WP-07, WP-21). The literature's uniform-mean AAUC is reported as well.
5. **0.69 MCC and 0.79 AAUC have no provenance.** They are re-measured (WP-20).
6. **ADWIN on loss is invalid.** Loss is unbounded and ADWIN's bound assumes bounded values. Response-token accuracy, bounded in [0, 1], is used instead. The detector is now the real algorithm and is tested against a river oracle (WP-23).
7. **"Mitigated forgetting"** requires a CI-backed replay versus no-replay difference (WP-23).
8. **E2E scorer bugs:** omission counted as correct, sampled rather than greedy outputs used for coverage, and different evaluation windows before and after. Fixed with a fixed set, greedy decoding, a three-way slot outcome and human-agreement validation (WP-24).
9. **15.8% is a misreport.** The data shows a 15.73% peak at step 400, not a baseline (§13). The bullet must use measured before/after values (§8).
10. **Alpaca evaluation leaked training data** (`eval_split: train`). A deterministic hash holdout was added (WP-25).
11. **GSM8K extraction** missed commas, negatives and decimals (WP-25).
12. **The ablation needs operational definitions** of "memorization" (prompt regurgitation rate) and "spikes" (5× rolling median), and must be reported whether or not they support the claim (WP-22).
13. **Filter false positives:** the drop rate on Alpaca is only an upper bound. Manual review of 50 dropped rows, and no threshold tuning on the evaluation data (WP-19).

## 11. Review pass 3: senior software engineer

1. **Checkpoint ordering is nondeterministic across runs.** Fixed, with order-independence tests (WP-02).
2. **Cold start silently disabled streaming.** Fixed with a pure, tested decision function (WP-03).
3. **evaluate.py could not find earlier runs and exited 0 on failure.** Fixed (WP-07).
4. **StreamFilter failed open.** Default changed to fail closed, with per-stage counters (WP-06).
5. **Secrets handling:** SASL credentials and the HMAC key come only from env vars, YAML secrets are rejected, logs are redacted (WP-10, WP-12).
6. **Default changes are listed explicitly** (WP-03, WP-06, WP-10), and existing tests are updated only where a WP says so.
7. **Unit tests stay offline;** live behaviour is tested in suites. Tiny in-memory models only (rule R7).
8. **Cross-platform hygiene:** pathlib, UTF-8, no `shell=True`, free ports, unique topics (R9, R10).
9. **Agents must not over-reach:** anchor checks, one WP per commit, stop-and-report (R1-R2, R11-R12).
10. **The CRLF difference** between the local folder and git is harmless, but an agent on Windows must not commit a whole-file line-ending change. Add `.gitattributes` with `* text=auto` only if Rohan agrees (Q7). Until then, agents keep each file's existing line endings.
11. **Observability:** rejection reasons, filter stats, malformed-record counts and swap components all go into `/metrics` or run outputs, so that every claim has a counter behind it (WP-04, WP-05, WP-06, WP-09).

---

## 12. Final review pass

I checked the plan against itself and the code. These are the gaps I found and closed:

| Check | Result |
|---|---|
| Every bullet maps to at least one WP and one acceptance rule | Yes (§2 and §8) |
| Every defect has a WP | D1-D12 → WP-02, 03, 10, 04, 05, 07, 09, 21, 06, 08, 25, 13 |
| No WP depends on something created later | Checked against §6 |
| Every new suite has a smoke size that runs in minutes on Windows CPU | Yes, except `eval_speed`'s claim run (T4) and Qwen runs (Mac Pro / Colab) |
| Unit tests need no network or GPU | Yes (R7); the only conditional skip is the optional river oracle in WP-23 |
| Plan numbers that are inferred rather than verified are marked | kafka-python compatibility, Qwen adapter size, model memory: all marked INFERRED with a WP that measures them |
| Rohan's hardware is respected | No Qwen on Windows; no timing claims from the MacBook Air; Qwen-3B on the Mac only after WP-26 |
| Docs | Context doc changes (WP-30), testing guide (05 doc, WP-31), evidence ledger (WP-29) |
| Resume integrity | Every number must come from `CLAIMS.md`; rounding rules and rewording templates are fixed in §8 |

Revision 2 check: every number in section 2A was read from the cited CSV row, and the SHA-256s were computed on the copies read. The run10 `metrics_clean.csv` hash matches the one context doc §13 cites.

Residual risks:
1. Measured values may not match the bullets. The plan handles this by rewording; it does not tune to the target.
2. Managed Kafka free-tier limits are unknown until Rohan picks a provider (Q8).
3. Running the T4 claim needs Colab credits.
4. Human labelling of 200 E2E rows takes Rohan's time (Q6).

---

## 13. Agent prompt template

Copy this for each WP and replace the `<…>` parts.

```
You are implementing ONE work package in the InfiniTune repository.

Work package: WP-<NN> from docs/ImplementationPlan/04_Resume_Claims_Implementation_Plan.md.
Read section 0 (rules) and the whole WP-<NN> section first. Follow them exactly.

Steps:
1. Confirm each "Anchor" exists in the named file. If any anchor is missing, STOP and report.
2. Make only the listed changes. Do not touch other code.
3. Add the listed tests. Run: python -m pytest -q
4. Run the "Acceptance" commands and paste their full output.
5. Commit with message "WP-<NN>: <title>".
6. Report: files changed, tests added, acceptance output, anything marked INFERRED, open questions.

Never read or print .env. Never commit output/, data/benchmark_cache/, reports/benchmark_runs/.
If you are unsure, stop and ask instead of guessing.
```

---

## 14. Questions for Rohan

Each question has a default the plan uses until you answer.

1. **DS-2 wording.** The logs confirm 15.73% (rounds to 15.7%), but it is the step-400 peak, not the starting value, which was 0%. It also covers only family-friendliness. OK to say "from a 15.7% mid-training peak to 1.2%"? **Default: yes.**
2. **QAFactEval.** The scorer is a simplified QA-overlap proxy. Rename it to "QAFactEval-style" on the resume and in code, or integrate the real QAFactEval package (heavy, with several large models)? **Default: rename.**
3. **"GPT-2" in SDE-1/DS-1.** Live runs so far used DistilGPT-2. Run gpt2-medium live as well, or write "DistilGPT-2"? **Default: run gpt2-medium as well.**
4. **Intel Arc XPU.** Do you want the experimental XPU path tried on the Windows laptop? **Default: try it, but never use it for resume numbers.**
5. **ADWIN action.** Should drift detection only report, or also trigger an action (for example raising the replay fraction)? **Default: report only.**
6. **Human labels.** Can you hand-label 200 E2E outputs (8 slots each) and review 50 filtered Alpaca rows? Without this, DS-2's numbers cannot be validated. **Default: yes, the plan schedules it.**
7. **Line endings.** Add a `.gitattributes` (`* text=auto`) so Windows and Mac commits stay consistent? **Default: no change until you say so.**
8. **Managed Kafka provider** for the Colab→laptop setup: Confluent Cloud, Redpanda Cloud or Aiven? **Default: Mode A (all in Colab) first; Mode B after you pick a provider.**
9. **Colab tier.** Which GPUs do your credits give you (T4 only, or L4/A100)? **Default: T4 for SDE-4, L4 for Qwen-3B.**
10. **Where code changes go.** Should agents commit to `main`, or to a feature branch with PRs? **Default: one branch per phase, with a PR each.**

Added after reading the report and `output/` logs, now **answered by Rohan (2026-10-15)**:

11. **Were run10 (IMDb) and run5 (E2E) trained through Kafka?** **Answer: no.** Both were trained in Colab without Kafka, because Kafka could not be set up in Colab. **Effect:** WP-20 (Kafka parity) and WP-27 (Kafka inside Colab) become required for every "Kafka / live data stream" wording. 82.7%, 0.69 MCC and 0.79 AAUC stay as single-pass streaming results.
12. **Is there a log for 40 minutes going to under 4?** **Answer: no exact timing was kept**, only the observed improvement. **Effect:** WP-17 on a Colab T4 is required before SDE-4 quotes any number.
13. **Qwen2.5-3B.** **Answer: plan the 3B run; say Qwen2.5-1.5B until it exists.** **Effect:** WP-25 schedules a Qwen2.5-3B run on Colab (L4 or A100).
14. **E2E details.** **Answer: yes.** **Effect:** docs use LoRA r = 8, and the E2E record count waits on WP-01's census.
