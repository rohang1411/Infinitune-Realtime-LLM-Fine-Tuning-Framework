# InfiniTune: Critical Audit Report and Agent Work Orders

Scope: review of the updated `Infinitune_Project_Context.md` (2,965 lines), mainly sections 6.7-6.18, 10.8, 11, 12, 13, 14, 17, checked against the CSCI-566 final report.
Limit: I reviewed the document only, not the code or raw run outputs. Anything I say about the code is inferred from how the document describes it. Where a claim depends on code, I say "verify in code".
Reviewer lens: staff engineer (infra/serving), senior data scientist, senior AI engineer at a frontier lab.

---

## 0. Ground rules for the agent fixing this (read first)

1. Do not write, edit, round or "restore" any number in any document unless it was produced by a run you can point to. Every metric needs a raw artifact: JSON/CSV path, git commit SHA, exact command, hardware, library versions, seed(s), timestamp.
2. If a number in the current document cannot be traced to such an artifact, delete it from the document. Do not replace it with an expected value.
3. Remove all "Ideal Target" columns. Self-assigned targets that every result happens to beat are a credibility problem (see F6). Replace them with a "Baseline" column (naive/legacy implementation measured under identical conditions).
4. Every metric row must state: model, hardware, batch size, sequence length, QPS/concurrency, n, number of seeds.
5. Prefer "I measured X vs baseline Y under identical conditions" over any claim against an invented SLA.

---

## 1. Verdict

The engineering direction is right and the self-audit in 13.1 shows real insight: phantom-class F1, torn updates, pickle RCE, a global lock, within-task "forgetting", and cold start are real problems, and finding them is the strongest thing in this project.

But in the current state the document would hurt you in an interview, for three reasons:

1. Several headline numbers are internally inconsistent or implausible as written (F3, F4), and a reader can show this with arithmetic from your own table.
2. Several measurement methods do not measure what their label says (swap pause, TTFT, "prequential" AAUC, freshness, FWT).
3. Several claimed measurements have no harness, no implementation description, or no raw artifact (24 h soak, ECE, BLEU/SER, ADWIN, canary gate, 500+ swaps).

Fixing this is days, not weeks. The fixed version is a much stronger resume project than the current one, because the story becomes: "I found my own system's flaws, fixed them, and proved each fix with a controlled before/after measurement."

---

## 2. Findings

Severity: S0 = can end an interview if unverified. S1 = methodology flaw a senior reviewer will probe. S2 = documentation hygiene.

### S0 findings

**F1. Provenance of the "Measured" numbers cannot be established from the document.**
Tells a reviewer will notice:
- All 13 metrics in tables 13.2/13.3 with a target land just inside it: p50 28.4 vs <30, p95 52.1 vs <60, p99 84.6 vs <100, freshness 11.4 s vs <15 s, lag knee 620 vs >500, ECE 0.042 vs <0.05, BWT -0.041 vs >= -0.05, SER 0.078 vs <0.08, BLEU 0.674 vs >0.65, AAUC 0.782 vs >0.75, other_rate 0.4% vs <1%. The only miss is IMDb accuracy.
- The "measured" latencies (28.4/52.1/84.6 ms) and swap time (2.14 ms) are identical to the example `/metrics` JSON in section 11.6.
- Final accuracy 82.65% is identical to two decimals to the earlier single-seed result, now labelled a 3-seed mean after substantial code changes.
- Section 13.4's reproduction commands cover four benchmarks. There is no command or script for the 24 h soak, the model-size scaling, chaos/failure tests, offline baselines, or multi-broker runs, yet the soak result is reported as measured.
- The TOC lists sections 10.11 (ECE) and 10.12 (MFU) that do not exist in the body. `MetricsLogger.COLUMNS` and the 10.8 catalog have no `ece`, `mfu`, `other_rate` or `f1_macro` column.

Action: for every number, attach the raw artifact. Anything that cannot be attached gets deleted. If the numbers did come from real runs, this is quick to fix and the problem disappears.

**F2. Hot-swap: the mechanism description and the headline metric do not line up.**
- The document shows the swap as `self._active_weights, self._shadow_weights = self._shadow_weights, self._active_weights` under a write lock. Swapping two Python references does not change what `model.generate()` computes unless the LoRA parameters inside the PEFT module are rebound or copied into. So either a copy step exists that the document omits (and that copy is the real cost), or the live model never sees the new weights. Verify in code.
- The writer-priority lock grants the write lock only after active readers drain, and blocks new readers while the writer waits. The user-visible pause is therefore (time waiting for the longest in-flight generation) + (copy/bind) + (flip). The reported "<2.5 ms pause" appears to time only the section inside the lock, so it excludes the lock-wait, which is the part that hurts tail latency.
- Section 11.1 says in-flight requests "conclude on the prior adapter" while new requests bind to the new one. That is impossible under an exclusive lock that requires zero readers, unless adapters are bound per request, which the document does not show. Pick one semantics and document it.
- "Lock-free" appears in 11.1, in timeline item 24 and in the diagram. It is a reader-writer lock. Remove the term.
- "Concurrent readers generate in parallel": Python threads share the GIL and one GPU. Parallel speed-up on HF `generate()` is usually well below linear. Show the throughput curve (RPS vs concurrency 1..64), do not assert parallelism.
- The 180 ms "legacy" figure needs an A/B under identical conditions. Keep both code paths behind a flag and run both.

**F3. Throughput, MFU and the lag knee contradict each other.**
Arithmetic from your own table (14,800 tokens/s, MFU 41.2%, knee 620 records/s):
- MFU = 6 x P x tokens/s / peak. For distilgpt2 (82M) that is 7.3 TFLOPs, about 2.3% of an A100 (312 TFLOPs bf16 dense) and about 4.4% of a 4090. 41.2% is only coherent for a model near 1.4-1.5B parameters on an A100 (about 44%). For Qwen2.5-3B it would exceed 87% of A100 peak, which is not credible. The table mixes models (the other rows are IMDb/distilgpt2) without saying so.
- `tokens_per_sec` is defined in the doc as response tokens only. On IMDb the response is a one-word label plus EOS, about 2-3 tokens per sample, so this metric (and MFU built on it) is meaningless for classification. MFU must use all processed non-padding tokens (prompt + response).
- 6P overstates FLOPs for LoRA (frozen base weights need no weight-gradient computation). State the FLOP model used, and use `torch.utils.flop_counter.FlopCounterMode` or the profiler for the real count.
- 620 records/s as a "consumer lag saturation knee" cannot be the trainer's limit if the trainer processes 14,800 tokens/s: an IMDb review is roughly 230-300 tokens (capped at 512), so 14,800 tokens/s is about 30-65 records/s. A knee of 620 records/s is about 10x too high unless it was measured on the producer-to-broker leg only. Define which consumer's lag it is.
- 41% MFU would also be exceptional for an HF + PEFT loop fed by Kafka with dynamic padding and poll gaps. Expect reviewers to ask for a profiler trace. An honest 15-25% with an analysis of padding waste and poll idle time is more credible than 41%.
- The "2.1x throughput, 48% VRAM" claim bundles bf16, `torch.compile(max-autotune)` and possibly other changes. Provide an ablation (fp32 / bf16 / bf16+compile) with recompile counts. With variable-length padded batches, `max-autotune` can recompile repeatedly, and Triton support on Windows is non-standard. The code falls back to eager silently, so verify compile actually ran. Halving VRAM from fp32 to bf16 is expected and should not be presented as an engineering win.

**F4. The classification numbers do not reconcile.**
- Section 13.1 explains the old 0.576 as 0.864 x 2/3, i.e. the true binary macro-F1 would be 0.864. The "fixed" value is reported as 0.825. They cannot both be right. Also 0.825 x 2/3 = 0.550, not 0.576. With accuracy 82.65%, one of 0.864 / 0.825 is wrong or the two numbers come from different runs.
- If other_rate at the end is 0.4%, macro-F1 over target classes should be within about 0.5 point of accuracy. 0.864 would need roughly 5% off-label outputs. State the other_rate at the final step of the old run.
- Define explicitly how off-label ("other") predictions are treated in the confusion matrix for F1 (counted as errors for recall of the true class), and add a unit test with a hand-computed matrix.
- "Wilson 95% CI" is labelled Wilson but the formula given is Wald. Wilson for p=0.8265, n=4000 is [81.45%, 83.79%] (asymmetric); Wald is +/-1.17% (the doc says 1.18).
- Either interval is only the binomial sampling error of the test set for one model. It does not include seed-to-seed variance. Report mean +/- std across seeds separately. "3 seeds" is the minimum; use 5.
- "Eliminates single-seed randomness; proves statistical significance" is wrong. A CI proves nothing about significance without a comparison. For model-vs-baseline claims use a paired test (McNemar) on the same 4,000 items. At n=4,000 an unpaired difference needs about 1.7 points to be distinguishable at 95%.
- Confirm seeds change everything that should change: producer shuffle seed, LoRA init, dropout, data order. The producer is a separate process with its own `shuffle_seed`.

**F5. Metrics reported with no documented implementation or harness.**
- ECE 0.042 (and the "0.218 overconfident baseline"). The evaluator described in 6.7 is generation + prefix match; no probabilities are described. ECE needs a confidence value (e.g. softmax over the label tokens at the answer position). Section 10.11 is missing. Specify bins, binary vs multiclass handling, and what the 0.218 baseline is (zero-shot with 100% off-label output has no meaningful calibration).
- BLEU-4 0.674 and SER 0.078. Section 6.8 describes substring slot checks only. It has no BLEU computation, no reference handling and no detection of added/hallucinated slots, which SER requires. E2E has multiple references per MR; use the official e2e-metrics scripts, and say which split (official test, 4,693) you used. Also note the original report shows perfect-coverage at 17.3% while this table shows SER 0.078; reconcile or explain.
- ADWIN: nothing says what it changes ("adapt training hyper-parameters" is not specified) and no detection-delay or false-positive numbers exist. As described it is decorative.
- Golden-canary gate: no results (catch rate, false-reject rate, rollback time). The glossary claims "sub-50ms rollbacks" with no measurement.
- 24 h soak (flat 10.2 GB MPS / 5.8 GB CUDA): no harness (see F1).
- "0.00% drops across 500+ swaps under 50 QPS": the runbook says a 60 s test at concurrency 16. Swaps at a 10 s push interval give about 6 per minute. 500+ swaps imply either a long run or forced synthetic swaps. State which, how swaps were forced, and whether the swapped weights actually differed.

**F6. Self-assigned "ideal targets".**
Columns titled "Ideal Target (Big Tech SLA)" and "(Frontier Lab)" are invented by the document. No company has one universal SLA, and results that beat every self-set target look curated. Remove the columns. Also remove audience-labelled framing in headings ("Staff SDE at Big Tech", "Senior DS"); present metrics neutrally.

### S1 findings

**M1. Serving benchmark methodology**
- "TTFT" is mislabelled. `/generate` is a non-streaming Flask endpoint returning the full text, so time-to-first-token is not observable from the client. Call it request latency and state `max_new_tokens`, prompt length, model, GPU.
- The `/metrics` percentiles are server-side. Report client-side latencies, which include queueing.
- Closed-loop load (fixed concurrency threads) understates tail latency because slow responses suppress new arrivals (coordinated omission). Use open-loop constant-arrival-rate load (wrk2, k6 constant-arrival-rate, or an asyncio generator that schedules sends by clock) for SLA-style claims, and record latency from the intended send time.
- Flask dev server on Windows is not a production serving stack. Re-run on Linux behind gunicorn/uvicorn with an explicit worker/thread model, and say so. Port mismatch: the runbook uses `localhost:8000`, the rest of the doc `localhost:5000`.
- Return `adapter_version` in every response (and a header). Without it you cannot prove per-request consistency or monotonic version progression under swaps.
- `/metrics` returns JSON, not Prometheus exposition format. Either emit real Prometheus format or drop the word.

**M2. Freshness probe**
- A single canary record cannot trigger an optimizer step when batch size x grad-accum (4 x 4 = 16 in the IMDb config) is larger than one, and one step at lr 1e-4 will not flip a learned behaviour. Document exactly how many canary records are sent, how they are made to cross the accumulation boundary, and what behaviour flips.
- Freshness is dominated by the intentional `weight_push_interval` (uniformly 0 to 10 s wait). A single number "11.4 s at 10 s interval" is mostly a configuration value. Decompose into: produce -> consumed -> accumulate -> optimizer step -> wait for push tick -> serialize -> transfer -> stage -> swap -> first response showing the change. Report p50/p95 over 30+ trials, and a curve of freshness and served accuracy vs push interval (1, 5, 10, 30, 60 s).

**M3. Continual-learning benchmark validity**
- IMDb -> Yelp -> Amazon is one task (binary sentiment, same label space) across three review domains. That is domain shift, a mild test. Forgetting will be small by construction and FWT will be dominated by format learning. Add a real task shift (different label space or task, e.g. topic classification, NLI, E2E NLG) and measure general-capability retention (WikiText perplexity and 2-3 lm-eval-harness tasks) before/after, which is what matters for LLM adaptation.
- FWT = mean(R_{i-1,i} - b_i): the baseline b_i must be defined. Zero-shot base accuracy is near zero because of format non-compliance, which inflates FWT; a format-matched control is required. Better: report forward transfer as sample-efficiency (steps to reach X% on the new domain with vs without prior training).
- Missing reference arms: joint/multitask training (upper bound), sequential fine-tuning without replay (lower bound), and ideally one regularization baseline (EWC or LwF).
- Report the full R matrix, average accuracy after the last phase, forgetting per task (max over previous minus final, Chaudhry-style), and plasticity (R_{i,i}, how well each new task is learned). Replay's cost in plasticity must be visible.
- Compute-match the comparison. Replay at ratio 0.25 changes the number of new-data samples per step. Report equal-gradient-steps and equal-new-samples variants.
- Sweep replay ratio (0, 0.1, 0.25, 0.5) and buffer size (250, 1000, 4000) and plot the stability-plasticity frontier. Single operating points (-0.184 vs -0.041) are less persuasive than a frontier.
- Fixed held-out set per domain; state sizes; 5 seeds; mean +/- std for BWT/forgetting/accuracy.

**M4. Two different things named "backward transfer".**
6.7 and 10.8 still describe `backward_transfer` as the fraction of samples correct before and wrong now (a regression/forgetting-event rate), and say "negative signals catastrophic forgetting". The final report said negative means improvement. 6.18 uses the standard cross-task BWT where negative means forgetting. Rename the per-sample metric (`regression_rate`) and keep one definition of BWT.

**M5. AAUC is not prequential.**
Table 13.3 calls AAUC "prequential learning velocity". AAUC is computed from periodic eval on a held-out pool, which is not prequential (test-then-train on the incoming stream). Real prequential accuracy/loss is free in a single-pass trainer (the loss computed before the update). Report prequential loss, cumulative regret vs the offline baseline, and the held-out AAUC under its correct name.

**M6. Slot-coverage evaluation (E2E)**
- Substring matching undercounts valid paraphrases (customer_rating "high" vs "highly rated"; priceRange "less than 20 pounds" vs "cheap"). Add normalization/synonym handling, or validate the checker against a human-labelled sample of 100 outputs and report its precision/recall.
- `_is_valid_restaurant_description` returns 0.0 and the sample is "ignored" ([HALLUCINATION IGNORED]). If ignored means excluded from the mean, coverage is inflated by survivorship. Count fluency failures as failures, and report the fluency-gate rate.
- SER needs added-slot detection. Perfect-coverage 17.3% is weak and needs an explanation.

**M7. Adapter protocol and security**
- A SHA-256 carried in the same manifest as the payload detects corruption, not tampering: anyone who can write to the topic can write a matching hash. For authentication use HMAC-SHA256 with a shared secret, or Ed25519 signatures, and reject unsigned snapshots. The manifest field `signature: Optional[str]` suggests this is not enforced. Verify in code.
- "Zero-copy" is wrong when loading from a bytes object (it copies). Drop the term.
- Multi-partition ordering: keys are `<step>:<layer>`, so layers hash to different partitions and a commit marker can arrive before some layers. The doc says commit plus all expected tensors are required, which is right, but specify: timeout and garbage collection of incomplete snapshots (memory leak otherwise), behavior when a newer step's manifest arrives before an older one completes, monotonic-step enforcement (never apply an older step), and idempotency for duplicate delivery (at-least-once). Test each.
- Cold start loads the latest local disk checkpoint, so staleness is bounded by `save_every_steps`, not eliminated, and a new replica on another machine has no local checkpoint. Better: a compacted `lora-latest` topic (or seek to the last `__commit__`), plus measure "steps behind" at startup.

**M8. Hardware and memory statements**
- Report says 16 GB Apple Silicon; 13.4 says M2 Max 32 GB. Section 14 says ~10 GB stable "for 3B models", but Qwen2.5-3B weights alone are about 12.4 GB in fp32 and all Qwen configs are fp32. State memory per model and precision.
- 13.4 lists A100/4090 as "reference environments" and Windows as "primary host". Attribute each number to the machine that produced it.
- Section 14 lists dependency versions as "Any" while 13.4 pins versions. Add a pinned `requirements.lock` and list `safetensors`, `scikit-learn`, any benchmark dependency.

**M9. Items from the earlier plan that are still absent**
Offline baselines (zero-shot with a proper prompt or label-logprob scoring, few-shot, offline LoRA 1 and 3 epochs); model-size scaling of swap time and adapter bytes; chaos tests (kill trainer, inference, broker); 3-broker KRaft with replication factor 3; multi-replica swap skew; soak harness; prequential metrics; learning-rate/rank sweeps.

### S2 findings (documentation hygiene)
- Section 4 repo tree lacks `adapter_manifest.py`, `serving_router.py`, `precision_manager.py`, `replay_buffer.py`, `continual_engine.py`, `benchmarks/`, `tests/`.
- 10.8 catalog and the CSV schema are stale (no `f1_macro`, `other_rate`, `ece`, `mfu`, per-seed outputs). 7.2 has no replay/precision/compile/canary keys.
- "QLoRA" remains in the architecture diagram, in 5.1 and in the trainer comment. The code is LoRA; `precision: 4bit` is listed but `bitsandbytes` is not a dependency. Remove QLoRA and the 4bit option, or implement it.
- 12.10 is titled "Balanced Sliding-Window Evaluation" but implements `full_pool`; 13.4 claims "balanced sampling across classes", which is not what 12.10 describes.
- Typos/tone: "Concurreny", exclamation marks in a technical table.
- The `tests/` content is only referenced in 13.5: describe what the 28 tests are, and whether the concurrency test uses a real model or a mock.

---

## 3. What is genuinely good (keep, and lead with it)

- Self-found, self-fixed failure modes with root cause: phantom-class F1 (once the numbers reconcile), torn updates, pickle deserialization, global lock, cold start, within-task "forgetting".
- Versioned manifest + commit protocol and safetensors with integrity check (after the HMAC fix, this becomes a strong systems story).
- Reader-writer lock with writer preference to avoid starvation (the idea is correct; fix the measurement).
- Separating format compliance (`other_rate`) from task correctness.
- A formal CL protocol with an R matrix, seeds, and a reservoir replay buffer (Algorithm R is described correctly).
- The decoupled evaluator and the 10x evaluation speed-up (40 min -> under 4 min), the memory-sweep fix (25+ GB -> ~10 GB) and the fp32/NaN fix. These are small but concrete and credible.
- Honest framing that this is single-pass streaming and what that costs.

---

## 4. Claims ledger for the resume

| Claim | Status | Condition |
|---|---|---|
| 40 min -> under 4 min eval, 3,340 generations | Safe | Keep. State model/hardware |
| Memory 25+ GB -> ~10 GB | Safe after fix | Per-model table, fp32 vs fp16, soak time-series |
| Zero NaN after fp32 switch | Safe | Say "on MPS, small batches" |
| Versioned atomic adapter snapshots (no torn updates) | Safe after fix | Edge-case tests (M7), show torn-update repro before / none after |
| Safetensors + checksum | Safe | Do not say "tamper-proof/signed" until HMAC |
| Hot-swap pause < 2.5 ms | Cut or fix | Replace with: lock-wait, copy, flip, plus p99 request latency during swaps vs steady state |
| 0 dropped requests over N swaps | Safe after fix | Raw log, open-loop load, real weight changes, per-response `adapter_version` |
| p50/p95/p99 "TTFT" | Cut or relabel | Request latency, client-side, open-loop, full context |
| 14,800 tok/s, MFU 41.2%, 2.1x, -48% VRAM | Cut until re-measured | F3 |
| Lag knee 620 rec/s | Cut until re-measured | Define which consumer, sweep producer rate |
| Freshness 11.4 s | Relabel | Decompose, p50/p95, interval curve |
| 24 h soak flat memory | Cut until a harness and a run exist | WO-6 |
| 82.65% +/- 1.18% IMDb | Safe after fix | Baselines, 5 seeds mean +/- std, paired test |
| Macro-F1 0.825 | Safe after fix | Reconcile 0.864 vs 0.825 first |
| ECE 0.042 | Cut until implemented/documented | WO-11 |
| BWT -0.041 with replay vs -0.184 | Safe after fix | WO-9, plasticity and baselines |
| FWT +0.068 | Cut | Format-matched baseline, or report sample-efficiency instead |
| BLEU 0.674 / SER 0.078 | Cut until official scripts | WO-12 |
| ADWIN / canary gate | Cut until evaluated | WO-10 |
| "Distributed" | Cut unless 3-broker run exists | WO-5 |
| QLoRA | Remove | LoRA only |

---

## 5. Work orders for the agent (priority order)

Each work order has acceptance criteria. Do not claim completion without the raw artifacts in `reports/<work_order>/` plus a `run_manifest.json` (git SHA, command, hardware, library versions, seeds, start/end time).

**WO-1. Provenance harness (0.5 d).**
Add a `bench` entry point that writes `run_manifest.json` plus raw per-run JSON for every benchmark. Add a docs lint that fails if a number in section 13 has no matching artifact reference. Accept: every table cell links to an artifact. Delete untraceable numbers.

**WO-2. Serving A/B and correct latency measurement (1-1.5 d).**
- Add `--serving-mode {legacy_lock,router}`; run both under identical load.
- Open-loop constant-arrival-rate load generator; QPS sweep from low load to saturation; latency measured from the intended send time; client-side p50/p95/p99/p99.9; error rate; goodput.
- Return `adapter_version` in the response body/header. Assertions: versions seen by each client are non-decreasing, and every version served was fully committed.
- Instrument and report separately: writer lock-wait, copy/bind time, flip time, and request latency in the window around each swap (vs steady state).
- Swaps: (a) natural trainer pushes, (b) forced swaps with real changed weights at 1/s, 0.2/s. Report which.
- Run on Linux behind gunicorn/uvicorn; record the GPU. Concurrency sweep 1..64: report RPS and latency vs concurrency (parallel efficiency).
Accept: a table "legacy vs router" with p99 during swaps, drop rate, plus lock-wait distribution.

**WO-3. Freshness decomposition (0.5-1 d).**
Timestamp every stage (producer send, trainer consume, optimizer step, push tick, serialize, send, receive, stage, swap, first probe seeing the change). Canary design: N records sufficient to cross grad-accum; document N and the behaviour tested. 30+ trials per push interval in {1, 5, 10, 30, 60} s. Report p50/p95 and each stage. Add served-accuracy vs interval for IMDb.
Accept: stacked-bar stage breakdown and a freshness-vs-interval curve.

**WO-4. Trainer throughput and MFU done correctly (1 d).**
Define tokens as processed non-pad tokens (prompt + response); keep response-tokens as a separate metric. Use FlopCounterMode/profiler for FLOPs and state peak-FLOPs assumption (dense bf16). Per-model table (distilgpt2, gpt2-medium, Qwen 1.5B, 3B): tokens/s, MFU, peak memory, step time, fraction of wall time spent in Kafka poll wait. Ablation: fp32 / bf16 / bf16 + compile with recompile counts and a check that compile did not fall back to eager. Measure the trainer's real ingest limit by sweeping producer rate and recording consumer-lag slope.
Accept: reconcile tokens/s, records/s and MFU in one coherent table.

**WO-5. Failure tolerance and Kafka topology (1.5-2 d).**
- Protocol edge tests: out-of-order steps, duplicate delivery, commit before last layer (multi-partition), abandoned manifest (trainer crash), two snapshots interleaved. Add TTL/GC for incomplete snapshots and a monotonic-step guard.
- Chaos: kill trainer mid-run (time to resume, steps lost), kill inference replica (time to serving the latest adapter), kill a broker on a 3-broker KRaft cluster, replication factor 3 (docker compose). Report recovery times and request error rate.
- Multi-replica: N inference replicas, each with its own consumer group; report swap-propagation skew (first vs last replica serving step k).
Accept: table of failure mode -> detection -> recovery time -> data/steps lost.

**WO-6. Soak (1 d, mostly unattended).**
Script that runs trainer + producer + inference with continuous load for 24 h, sampling RSS, GPU memory, open fds, threads, consumer lag, error counts every 10 s; fit memory slope. Per model and device.
Accept: time-series CSV plus slope estimate and confidence interval.

**WO-7. Evaluation correctness and statistics (1 d).**
- Unit tests for macro-F1 with hand-computed matrices including off-label predictions; log both `f1_macro` (target classes) and `accuracy` (off-label as wrong) and `other_rate` at every eval step. Reconcile 0.864 vs 0.825.
- Correct Wilson interval (or bootstrap), per seed, plus mean +/- std across 5 seeds; verify all seeds vary producer shuffle, init and dropout. Paired McNemar for every model-vs-baseline comparison.
- Rename the per-sample metric `regression_rate`; one BWT definition.
- Report prequential loss/accuracy and regret; rename held-out AAUC correctly.
Accept: per-seed JSON, mean/std table, p-values.

**WO-8. Baselines (1-1.5 d).**
Zero-shot (proper prompt and label-logprob scoring so format is not the failure), few-shot, offline LoRA 1 epoch and 3 epochs (same data, same eval pool), full fine-tuning if affordable, plus sequential FT and joint training for CL. Same eval items for all.
Accept: one table: method, trainable %, wall-clock, accuracy +/- std, MCC, paired p-value vs streaming.

**WO-9. Continual-learning benchmark v2 (2 d).**
Add a task-shift sequence (e.g. IMDb -> topic classification -> E2E NLG) and keep the domain-shift sequence. Report R matrix, final average accuracy, forgetting per task, BWT, plasticity, FWT with a format-matched baseline (or sample-efficiency), general-capability retention (WikiText perplexity, 2-3 lm-eval tasks, Qwen runs). Replay ratio and buffer-size sweeps; compute-matched variants; joint and sequential reference arms; 5 seeds.
Accept: stability-plasticity frontier plot and a table with std.

**WO-10. Drift detector and promotion gate evaluation (1 d).**
ADWIN: known-boundary streams, detection delay, false-positive rate on stationary streams; say what it actually triggers. Canary gate: inject bad updates (5% and 20% flipped labels, lr spike, NaN), report catch rate, false-reject rate on normal updates, time to rollback, served-accuracy dip. Reconsider gate vs domain shift (a stationary old-domain canary will reject legitimate adaptation; define the policy).
Accept: confusion table for the gate and a delay/FPR table for ADWIN. Remove "sub-50ms rollback" unless measured.

**WO-11. Calibration (0.5 d).**
Compute ECE from label-token probabilities at the answer position; 15 equal-mass bins; reliability diagram; before/after training; document in a new 10.11. Say what the baseline is.

**WO-12. E2E evaluation done to standard (1 d).**
Official e2e-metrics (BLEU, NIST, METEOR, ROUGE-L, CIDEr, multi-reference) on the official test set; SER with added-slot detection; slot normalization validated against 100 hand-labelled outputs; fluency gate counted as failure; compare with an offline LoRA baseline run by you on the same split.

**WO-13. Protocol security and cold start (0.5-1 d).**
HMAC-SHA256 (or Ed25519) over manifest + payload, reject unsigned/invalid, test with tampered bytes and replayed old steps. Cold start via a compacted `lora-latest` topic or seeking to the last commit; measure steps-behind at startup.

**WO-14. Documentation hygiene (0.5 d).**
Fix every S2 item. Add the missing sections 10.11/10.12 once implemented, update repo tree, metrics catalog, schema, pinned deps, remove QLoRA, fix ports, rename the "balanced sliding window" section, remove "lock-free", "Prometheus" (unless real), "TTFT", "zero-copy", audience-labelled headings and all "Ideal Target" columns.

Total: about 11-14 working days. Highest value per day: WO-1, WO-2, WO-7, WO-8, WO-9, then WO-3, WO-4, WO-5.

---

## 6. Questions a senior reviewer will ask, and what you need to be able to say

1. "How does a pointer swap change what the model computes?" You must be able to walk through the exact code path and what the copy costs.
2. "Your swap is 2 ms. What is p99 latency for a request that arrives during a swap?" Lock-wait included.
3. "Closed-loop or open-loop load? Where does coordinated omission show up?"
4. "Why Flask and threads instead of vLLM or continuous batching? What is your throughput ceiling?"
5. "What is the zero-shot baseline with a properly constrained prompt? Offline LoRA on the same data?"
6. "How many seeds, what varies, what is the std? Is this difference significant (paired)?"
7. "IMDb -> Yelp -> Amazon is the same task. What did you do about task shift and general-capability forgetting?"
8. "What baseline defines your FWT? Is that just format learning?"
9. "How does the commit protocol behave with multiple partitions, a crashed trainer, or duplicate delivery?"
10. "A hash in the same message does not authenticate the sender. What does?"
11. "Your MFU, tokens/s and lag knee do not match. Which model, hardware, token definition?"
12. "How do you know the 24-hour soak was flat? Show the series."
13. "Who built what in the six-person team?" Claim the training loop, label masking, evaluation design, decoupled evaluator and the audit/fix work you did yourself, and be ready to say who built the producer and Flask server.

---

## 7. Resume bullet skeletons (fill only from verified artifacts)

Systems
- Built a three-service streaming LoRA fine-tuning pipeline (producer, trainer, inference) on Kafka with step-versioned, HMAC-authenticated adapter snapshots; [X] torn updates across [N] swaps (vs [Y] with the original per-layer protocol).
- Replaced a global inference lock with a reader-writer double-buffered router; p99 request latency during swaps [X] ms vs [Y] ms under the legacy lock at [Q] QPS (open-loop, [GPU]), [0] failed requests over [N] swaps.
- Measured end-to-end freshness (record produced to behaviour changed at the endpoint) at p50 [X] s / p95 [Y] s, [Z] s of it intentional push-interval wait; recovered from trainer/broker/replica kills in [X]/[Y]/[Z] s.

Modelling
- Single-pass streaming LoRA on distilgpt2 reached [X]% +/- [S] (5 seeds) on IMDb, within [D] points of an offline [N]-epoch LoRA baseline (paired p=[P]), training [T]% of parameters.
- Under [domain/task] shift, reservoir replay cut forgetting from [X] to [Y] points at [Z]% plasticity cost ([N] seeds); retained [R]% of base-model general capability on [benchmarks].
- Reduced E2E slot error rate to [X]% (BLEU [Y], official scripts) against [Z] for an offline LoRA baseline on the same split.

Do not use: "0% to 82.65%", "lock-free", "zero downtime" (without WO-2), "distributed" (without WO-5), "QLoRA", any MFU/throughput figure before WO-4, any soak figure before WO-6.
