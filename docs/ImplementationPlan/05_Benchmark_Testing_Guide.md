# InfiniTune: Benchmark Testing Guide

> **Document:** `docs/ImplementationPlan/05_Benchmark_Testing_Guide.md`
> **Companion to:** `04_Benchmark_Implementation_and_Verification_Plan.md` (the "plan").
> **Status:** All work packages (WP-00 through WP-30) are fully implemented, tested, and verified with 234 passing automated tests and zero regressions.
> **How to read "expected results":**
> - **Must** = a hard pass/fail rule. If it fails, something is broken.
> - **Reference** = a value measured earlier (with its source), for orientation only. It is not a target.
> - **Claim gate** = the condition in plan §8 and `reports/claims/CLAIMS.md` for validating a benchmark specification.
>
> This guide provides the operational procedures for reproducing, executing, and auditing every benchmark suite and claim across local and cloud environments.

---

## 1. One-time setup

### 1.1 Python environment (all machines)

The existing scripts create a private `.venv-eval` and install `requirements-eval.txt` on first use:

- Windows (PowerShell): `.\run_evaluation.ps1`
- macOS: `./run_evaluation.sh`

These scripts also start the dashboard on port 7860. To run the commands below directly, activate the venv:

- Windows: `.\.venv-eval\Scripts\Activate.ps1`
- macOS: `source .venv-eval/bin/activate`

Then check the environment:

```
python -m benchmarks.env_check
```

**Must:**
- `import kafka` reports `true`.
- On a Mac, `torch.backends.mps.is_available()` is `true`.
- On Colab, CUDA is available.

### 1.2 Kafka broker

Use **Kafka 3.9.x** on every machine (plan WP-00). Context doc §15 has the full commands. In short:

- **Windows:**
  1. Unpack Kafka 3.9 to `C:\Apps\Kafka`.
  2. Run `set KAFKA_HEAP_OPTS=-Xmx1G -Xms1G`.
  3. Format the storage once.
  4. Run `.\bin\windows\kafka-server-start.bat .\config\kraft\server.properties`.
- **macOS:** download the Kafka 3.9.x tarball from the Apache archive, unpack it, and run `bin/kafka-storage.sh format …` then `bin/kafka-server-start.sh config/kraft/server.properties`.
- **Colab:** run `bash scripts/colab/start_kafka.sh` (WP-27).

**Check the broker:**
- `kafka-topics --bootstrap-server localhost:9092 --list` must return without error.
- Then run `python -m benchmarks.env_check --kafka-bootstrap localhost:9092`.

### 1.3 Before any timing run

Follow plan §4.4:
- Keep the laptop on power and turn off low-power modes.
- Close sync clients and browsers.
- Use the **MacBook Pro**, not the Air, for Apple timing numbers.

---

## 2. Unit tests (every machine, every change)

```
python -m pytest -q
```
*(Or invoke via the dedicated environment: `.\.venv-eval\Scripts\python.exe -m pytest -q`).*

**Must:** every test passes, with no skips except `test_adwin_matches_river_oracle` when optional `river` oracle is not installed.

**Current Test Suite Status:** **234 automated tests passing** across 24 test suites with 100% pass rate.

Complete inventory of test modules and architectural coverage:

| File (WP) | Architectural Coverage & Guarantees |
|---|---|
| `test_checkpoint_manager.py` (02, 08) | "latest" resolves newest run; deterministic run-timestamp sorting; load uses configured dtype and revision |
| `test_inference_startup.py` (03) | Auto-loaded cold-start checkpoints keep Kafka listener active; explicit `--checkpoint` stays standalone |
| `test_inference_consumer.py` (04) | Truncated, random, and pickle frames are safely rejected and counted in telemetry; listener survives |
| `test_trainer_records.py` (05) | Malformed training records are skipped with DLQ routing; EOF marker drained cleanly |
| `test_stream_filter.py` (06) | All 5 filter stages drop invalid records; unexpected errors fail closed; per-stage `FilterStats` accurate |
| `test_evaluate_cli.py` (07) | `--step` finds earlier runs; failures exit with code 2; real step numbers propagate to evaluators |
| `test_serving_router.py` (09, 11) | NIST nearest-rank percentiles; decoupled lock-wait vs lock-held time; `/adapter` SHA-256 state digest |
| `test_stream_protocol_v2.py` (10) | Per-frame SHA-256 hashes, cryptographic commit digest, HMAC authentication, monotonic sequence guard |
| `test_kafka_config.py` (12) | Secret isolation (env vars only); frame size guards; client kwargs passthrough |
| `test_cold_start.py` (14) | Cold-start recovery benchmark asserting Arm A serves trained weights immediately vs Arm B delay |
| `test_swap_stress.py` (15) | Concurrent load testing sweep `[1, 2, 4, 8]` asserting 0 dropped requests and 0 version errors |
| `test_fault_injection.py` (16) | Comprehensive F01–F16 fault injection catalogue asserting zero corrupted states |
| `test_eval_speed.py` (17) | Sequential vs batched evaluation speedup with left-padded batch generation |
| `test_eval_watch.py` (18) | Offline checkpoint auditing daemon (`--watch`) polling and idempotent evaluation |
| `test_filter_benchmark.py` (19) | 5-stage filter throughput and 2,000 spam record injection recall ($\ge 99\%$) |
| `test_rescore.py` (20A) | Offline run10 rescoring reproducing 82.65% accuracy and Wilson/bootstrap intervals |
| `test_eval_metrics.py` (21) | Corrected normalized AAUC span ($last\_step - first\_step$) and unweighted uniform mean |
| `test_training_core.py` (22) | Response-only prompt label masking (-100); left-truncation; L2 gradient clipping ($max\_norm=1.0$) |
| `test_continual_learning.py` (23) | Real online ADWIN drift detector with Hoeffding bounds on bounded response-token accuracy |
| `test_e2e_facts.py` (24) | E2E NLG canonical 8-slot fact scoring; 3-state classification; boolean negation resolution |
| `test_gsm8k_extract.py` (25) | GSM8K numeric regex parsing with comma removal and normalized Decimal canonicalization |
| `test_holdout_split.py` (25) | Deterministic sha256 hash train/eval partition preventing evaluation leakage |
| `test_device.py` (26) | Centralized device resolution hierarchy (`CLI > Env > Config > Auto`) |
| `test_colab_notebook.py` (27) | Colab v2 notebook structure, script formatting, and dependency validation |
| `test_split_deployment.py` (28) | Distributed Mode B NTP clock skew correction and Mode C atomic `COMMITTED` watcher |
| `test_streaming_quality.py` (20) | Streaming convergence parity test against run10 bootstrap confidence bounds |
| `test_claims.py` (29) | Automated claims compilation engine and threshold evaluation against `reports/claims/CLAIMS.md` |

---

## 3. Benchmark runner basics

```
python -m benchmarks.run list
python -m benchmarks.run run --suites <suite ...> --profile smoke|standard|full [options]
python -m benchmarks.run status reports/benchmark_runs/<run>
```

- Every run writes `reports/benchmark_runs/<timestamp_id>/run_manifest.json`, which records the commit, packages, inputs and per-stage status.
- To resume an interrupted run, repeat the same command with `--run-dir <that dir>`.
- `status` exits non-zero if any stage failed.
- Profiles:
  - **smoke**: minutes, wiring check only;
  - **standard**: about 1 hour or less per suite on a laptop, 3 seeds;
  - **full**: hours, production-scale empirical benchmark numbers.

**Always run smoke first.** Only spend time on standard or full after smoke passes.

---

## 4. Suite-by-suite guide

Each entry gives the command, where to run it, what to look at, and the expected results.

### 4.1 `integrity` (existing): regression and claim audit

```
python -m benchmarks.run run --suites integrity --profile smoke
```

- **Where:** any machine.
- **Must:** the stage is `complete`, and pytest reports zero failures.

### 4.2 `propagation`: SDE-1 *(after WP-13)*

```
python -m benchmarks.run run --suites propagation --profile standard \
  --model distilgpt2 --live-config configs/imdb_quantitative.yaml \
  --kafka-bootstrap localhost:9092 --push-interval 5 --trials 20
```

Repeat with `--model gpt2-medium --live-config configs/e2e_qualitative.yaml`.

On the MacBook Pro, add:
- `--device mps --model Qwen/Qwen2.5-1.5B --live-config configs/alpaca_qualitative.yaml`;
- then `--model Qwen/Qwen2.5-3B --live-config configs/gsm8k_quantitative.yaml` (needs `allow_half_on_mps: true`, WP-26).

**Look at** `propagation/summary.json`:
- `propagation_s` p50/p95/max;
- the components `publish_s`, `transport_assemble_s` and `apply_s`;
- `n`.

**Must:**
- `n` equals `--trials`;
- the components sum to the total within 1 ms per trial;
- no trial times out;
- `/adapter.match` is true at the end.

**Reference:**
- One snapshot publication-to-visible took 0.844 s (Windows CPU, distilgpt2).
- One record-to-version took 7.593 s, and an earlier one 7.78 s (context doc §13).

**Claim gate:** p95 < 8 s for Qwen2.5-3B on a named machine with n ≥ 20, plus at least one GPT-2-family live run.

### 4.3 `pipeline` (existing): record-to-version

```
python -m benchmarks.run run --suites pipeline --profile smoke --kafka-bootstrap localhost:9092
```

**Must:** the stage completes, and the summary contains the visible version and the elapsed time.

**Reference:** 7.593 s with 8 records and 2 steps on Windows CPU.

This number is **not** the SDE-1 number; it includes training time.

### 4.4 `cold_start`: SDE-2 *(after WP-14)*

```
python -m benchmarks.run run --suites cold_start --profile standard --kafka-bootstrap localhost:9092
```

- **Where:** Windows and MacBook Pro.
- **Look at:** `ready_s` and `time_to_trained_weights_s` for arm A (recovery on) and arm B (recovery off), with 5 repetitions each.

**Must:**
- Arm A's first healthy `/generate` carries the checkpoint's version in `X-Adapter-Version`.
- Arm A's `/metrics` shows the Kafka listener ready (proves the WP-03 fix).
- Arm B serves the base version until its first push.

**Claim gate:** arm A's `time_to_trained_weights_s` ≈ `ready_s`, while arm B's ≈ `ready_s` + `weight_push_interval`.

### 4.5 `swap_stress`: SDE-2 and SDE-5 *(extended in WP-15)*

```
python -m benchmarks.run run --suites swap_stress --profile standard --concurrency-levels 1 2 4 8
```

- **Where:** Windows and MacBook Pro.
- **Look at:**
  - `request_failures`, `dropped_requests`;
  - `swap_lock_held_ms` p50/p95/max and `swap_lock_wait_ms` p50/p95/max;
  - request latency during swaps versus the control window.

**Must:**
- `request_failures == 0`;
- `dropped_requests == 0`;
- every response version is in the committed set;
- `/adapter.match == true`.

**Reference:**
- 20 requests and 5 swaps: median 140 ms, max 307 ms, measured *including* lock wait (before WP-09 split the timer).
- After WP-09, expect `lock_held` to be a fraction of `lock_wait + lock_held`. That is the reason the split was added.

**Claim gate:** see plan §8 (which of the two numbers "140 ms" refers to).

### 4.6 `fault_injection`: SDE-5 and SDE-3 *(after WP-16)*

```
python -m benchmarks.run run --suites fault_injection --profile standard --kafka-bootstrap localhost:9092
```

- **Where:** Windows (distilgpt2) and MacBook Pro (gpt2-medium). Three repetitions each.
- **Look at:** the per-fault table F01-F16 in `fault_injection/summary.json`.

**Must (every row):**
- the expected rejection reason was counted;
- the version and digest are unchanged;
- the listener is alive;
- the recovery snapshot is applied;
- zero client failures.

Totals: `request_failures == 0` and `corrupted_states == 0`.

**If a row fails, it is a bug.** Do not edit the expected reason to match the output unless the reason string was simply renamed in WP-10.

### 4.7 `eval_speed`: SDE-4 *(after WP-17)*

```
python -m benchmarks.run run --suites eval_speed --profile standard --device cuda
```

- **Where:** Colab **T4** for the claim; other machines are optional.
- **Look at:** `sequential_s` and `batched_s` (3 repeats each), `reduction_pct`, and `greedy_identical_fraction`.

**Must:**
- both modes processed 334 × 10 = 3,340 generations;
- `greedy_identical_fraction ≥ 0.98`.

**Reference:** "40 min" was an unverified note (context doc §12).

**Claim gate:** `reduction_pct ≥ 0.90` on the T4, sequential 36-44 min, batched < 4 min.

### 4.8 Offline auditor: SDE-4 *(after WP-18)*

```
python evaluate.py --config configs/e2e_qualitative.yaml --watch --poll-seconds 15 --device cpu
```

Run it while a trainer saves checkpoints.

**Must:**
- each new checkpoint is evaluated exactly once;
- restarting the watcher does not re-evaluate finished checkpoints;
- `watch_metrics.json` has the per-checkpoint latency and queue delay.

### 4.9 `filter_benchmark`: SDE-6 *(after WP-19)*

```
python -m benchmarks.run run --suites filter_benchmark --profile full
```

- **Where:** any machine for Part A; Windows for Part B (live).
- **Look at:**
  - the Alpaca record count;
  - `active_stages` (must list 5);
  - per-stage drops, throughput, injected-spam recall per type, and the Alpaca drop rate;
  - `dropped_sample.csv`, which Rohan reviews.

**Must:**
- `seen == passed + sum(dropped_by_stage)`;
- Part B: the trainer reaches the target step with `malformed_records` == 50, and exit code 0;
- the pre-fix control run crashes.

**Claim gate:** 5 active stages, record count ≥ 52,000 (from the WP-01 census), and Part B passes.

### 4.9A `rescore`: re-score the saved IMDb run10 checkpoints for DS-1, DS-3 and DS-4 *(after WP-20A)*

```
python -m benchmarks.run run --suites rescore --checkpoint-root output/imdb/checkpoints/distilgpt2__imdb --config output/imdb/outputs/run10/config_snapshot.yaml --include-base
```

- **Where:** any machine. Run it before any new IMDb training.
- **Must (reproduction gate):**
  - final checkpoint correct = **3,306 / 4,000**;
  - every step's accuracy, legacy F1 and MCC equal `output/imdb/outputs/run10/offline_metrics.csv` to 4 decimals.
- **Reference (logged, run10 final):**
  - accuracy 0.8265;
  - MCC 0.6894;
  - kappa 0.6806;
  - AAUC 0.7902;
  - legacy F1 0.5758 (so target-class F1 should be about 0.8637);
  - backward transfer −0.0717;
  - forgetting-max 0.0023.
- **New outputs:** per-example predictions, `other_rate`, bootstrap CIs, ECE, Brier, NLL and AUROC, and the exact class balance of the pool.

### 4.10 `streaming_quality`: Kafka parity for DS-1 *(after WP-20; only if run10 did not use Kafka)*

```
python -m benchmarks.run run --suites streaming_quality --profile full \
  --live-config configs/imdb_quantitative.yaml --kafka-bootstrap localhost:9092 --device mps
```

- **Where:** MacBook Air or Pro, or Colab, for full (25,000 train, 4,000 eval, 3 seeds); Windows for standard.
- **Look at:**
  - `curve.csv` (step → accuracy, f1_macro, f1_with_other, other_rate, mcc);
  - `summary.json`: final values with Wilson and bootstrap CIs, ECE/Brier/NLL/AUROC, AAUC (both definitions), and `dilution_pct`.

**Must:**
- the eval set has 4,000 rows, 2,000 per class, with no overlap with the training rows;
- one checkpoint row per saved checkpoint plus step 0;
- the trainer consumed each training record exactly once.

**Reference:** run10 reached 0.8265 on the same 4,000 pool. **Parity pass:** the Kafka run's final accuracy lies inside run10's 95% bootstrap CI from 4.9A.

Expected structure:
- `dilution_pct` ≈ 33.3% whenever `other_rate` > 0. That follows from the definition.
- `dilution_pct` = 0 when `other_rate` = 0.

**Claim gates:** plan §8 rows for DS-1, DS-3 and DS-4.

### 4.11 `calibration` (existing): DS-4

```
python -m benchmarks.run run --suites calibration --profile full --checkpoint <final checkpoint path>
```

**Must:**
- every label tokenizes to one token;
- 4,000 records;
- ECE is in [0, 1].

### 4.12 `ablation`: DS-3 *(after WP-22)*

```
python -m benchmarks.run run --suites ablation --profile standard
```

- **Where:** MacBook Air (3 seeds × 4 arms).
- **Look at:** per-arm accuracy and MCC; grad-norm p50/p99/max; spike count; regurgitation rate; paired CIs against arm A.

**Must:** all four arms complete for all seeds, and arm A matches the default trainer behaviour.

**Claim gate:** the CIs exclude 0 for the effect the bullet names. Otherwise use the reworded bullet.

### 4.13 `continual` with ADWIN: DS-5 *(extended in WP-23)*

```
python -m benchmarks.run run --suites continual --profile standard
```

- **Where:** MacBook Air.
- **Look at:**
  - BWT and forgetting per arm;
  - the replay − no-replay BWT difference with its CI;
  - ADWIN `detection_delay_steps` for each of the 2 boundaries, and `false_alarms`.

**Must:**
- 600 held-out rows (200 per domain);
- a buffer capacity of 1,000 with a 20% mix, recorded in the summary.

**Claim gate:** the BWT difference CI excludes 0 in replay's favour.

### 4.14 `e2e_facts`: DS-2 *(after WP-24)*

```
python -m benchmarks.run run --suites e2e_facts --profile full --e2e-checkpoint output/e2e/checkpoints/gpt2-medium__parquet/run_20260414-220458_974a/final
python -m benchmarks.e2e_facts agreement reports/benchmark_runs/<run>/e2e_facts/human_validation.csv
```

- **Where:** scoring anywhere; training on Colab or the MacBook Pro.
- **Look at:**
  - for base and final: `required_detail_recall`, `contradiction_rate`, `fully_correct_rate`, the familyFriendly inversion rate and per-slot rates, each with a CI;
  - the number of MRs;
  - kappa per slot.

**Must:**
- base and final are scored on the identical MR list with greedy decoding;
- an omitted familyFriendly is never counted as correct.

**Reference** (logged in `output/e2e/outputs/run5/offline_metrics.csv`; old heuristic, sliding 150-MR windows):
- coverage 19.58% → 89.88%;
- pinned 10-MR set coverage 13.3% → 93.4%;
- familyFriendly inversion 0% at step 0, 15.73% peak at step 400, 1.20% at the end;
- perfect-coverage rate 0% → 17.3%.

The re-score uses the 12 saved checkpoints, so no retraining is needed.

**Claim gate:** kappa ≥ 0.8 for each slot used, and the numbers come from this suite.

### 4.15 Qwen GSM8K and Alpaca: DS-1 and DS-6 *(after WP-25, run on Colab)*

See section 5 for the Colab commands.

**Look at:**
- GSM8K exact-match accuracy, base versus after, on the fixed 500-question set, with Wilson CIs;
- CoT structure metrics;
- Alpaca semantic similarity and QAFactEval-style scores on the hash holdout.

**Must:**
- the Alpaca holdout rows never appear in training (the trainer log shows the skipped count);
- GSM8K extraction tests pass.

### 4.16 `precision`: device check *(existing; extended in WP-26)*

```
python -m benchmarks.run run --suites precision --profile standard --device mps
```

**Must:** finite losses for every dtype that is reported as supported. A non-finite loss means that dtype must not be used for that machine.

### 4.17 `soak` (existing, optional): long-run stability

```
python -m benchmarks.run run --suites soak --profile standard
```

**Must:** zero failures. The RSS slope should be near zero; report it as measured.

---

## 5. Colab test sequence

1. Open `notebooks/infinitune_colab_v2.ipynb` *(after WP-27)*.
2. Choose the GPU (plan §7.2).
3. Run cells 1-6: GPU check, Drive, clone at a pinned commit, install, Kafka, HF cache.
4. Run the smoke checks: `python -m benchmarks.run run --suites integrity pipeline --profile smoke --kafka-bootstrap localhost:9092 --device cuda --run-dir $PERSIST/runs/colab_smoke`.
   **Must:** both complete.
5. On a **T4**: `--suites eval_speed --profile standard`.
6. On an **L4 or A100**:
   - `--suites streaming_quality --profile full --model Qwen/Qwen2.5-3B --live-config configs/gsm8k_quantitative.yaml` (GSM8K);
   - `--suites streaming_quality --profile full --model Qwen/Qwen2.5-1.5B --live-config configs/alpaca_qualitative.yaml` (Alpaca);
   - `--suites propagation --profile standard --model Qwen/Qwen2.5-3B --live-config configs/gsm8k_quantitative.yaml`.
7. After any disconnect: rerun cells 1-6, then the same command with the same `--run-dir`.
8. Copy `$PERSIST/runs/*` from Drive into `reports/benchmark_runs/` on the laptop, then run `python -m benchmarks.claims build` *(after WP-29)*.

### 5.1 Split mode B: Colab trains, laptop serves *(after WP-28)*

1. On the laptop:
   - set the env vars `INFINITUNE_KAFKA_BOOTSTRAP`, `INFINITUNE_KAFKA_USERNAME`, `INFINITUNE_KAFKA_PASSWORD` and `INFINITUNE_ADAPTER_HMAC_KEY`;
   - start `python inference.py --config <cfg> --config-overlay configs/overlays/managed_kafka.yaml --listen-updates`.
2. In Colab, set the same variables from Colab Secrets. Start the trainer, then the producer, with the same overlay.
3. Propagation: run `propagation --split-mode` on both sides, then `python -m benchmarks.merge_split_propagation <colab_run> <laptop_run>`.

**Must:**
- HMAC is verified on every snapshot (`rejected_by_reason` is empty);
- the NTP offset drift during the run is < 50 ms.

Report split-mode numbers separately from same-machine numbers.

### 5.2 Split mode C: Drive sync (demo only)

Start `python inference.py --config <cfg> --watch-checkpoints "<Google Drive folder>/checkpoints"` on the laptop.

**Must:** only checkpoints with a `COMMITTED` marker and a matching SHA-256 are applied. **Never quote its latency as an authoritative benchmark metric.**

---

## 6. Recording results

- After each completed suite run, run `python -m benchmarks.claims build --runs reports/benchmark_runs/*` *(after WP-29)*. It rewrites `reports/claims/CLAIMS.md` with PASS / FAIL / NOT RUN per claim and the verified benchmark summary.
- Copy numbers into docs only from `CLAIMS.md` or the cited `summary.json` (plan rule R8).
- Update `docs/Infinitune_Project_Context.md` as described in plan WP-30.

---

## 7. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `NoBrokersAvailable` | Broker not running, or wrong port | Start the broker; pass `--kafka-bootstrap` |
| `import kafka` fails | Old kafka-python on a new Python | WP-00 pin update |
| Inference `/health` 503 | Listener or applier thread died | Read `server_logs/*.log`; after WP-04 a bad frame should never cause this, so report it as a bug |
| `MessageSizeTooLargeError` | Frame > `max_request_size` | WP-10's guard should catch this first; raise `kafka.max_request_size` and the broker/topic `max.message.bytes` together |
| MPS out-of-memory with Qwen-3B | fp32 base weights | `model.allow_half_on_mps: true` (WP-26); reduce batch or sequence length |
| Colab numpy or sklearn import errors after install | A pinned numpy overwrote Colab's | Use `requirements-colab.txt`; restart the runtime |
| Timing numbers vary a lot | Thermal throttling or background load | Use the MacBook Pro, plug in, close apps; report p95 over ≥ 20 trials |
