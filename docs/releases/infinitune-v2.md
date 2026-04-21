# Infinitune v2 Release Notes

## Overview
Infinitune v2 is a major upgrade to the platform's training, evaluation, and runtime infrastructure. This release expands Infinitune from a narrower fine-tuning workflow into a more configurable and observable system, with stronger support for task-specific evaluation, richer metric tracking, cleaner evaluation architecture, and safer streaming behavior.

At a high level, v2 improves three core areas:

- how model quality is measured
- how training and inference behavior are configured and monitored
- how streaming data is filtered and managed during evaluation workflows

This release should be treated as a major platform step rather than a routine feature update.

## Expanded Evaluation Framework
A major focus of Infinitune v2 is the evaluation system. The platform now supports a broader and more useful set of metrics for both model performance and training behavior.

Support for F1, Matthews Correlation Coefficient (MCC), Cohen's Kappa, exact match, gradient norm, and tokens-per-second tracking was introduced in [`0bd844f`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/0bd844f7647a97a2d62b6401f1ecf720a3819b6a), giving training runs much stronger observability and making it easier to compare classification and generation-style workloads. That work was extended in [`62c279a`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/62c279aadd326166a58c6696bf5237b94b04222c), which integrated F1, MCC, Kappa, and exact match more deeply into trainer behavior and task configuration.

Infinitune v2 also broadens its support for continual-learning and longitudinal evaluation. Backward transfer tracking was added in [`801f3c5`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/801f3c58607c614aeba498da65ada264316bbdf7), and that foundation was expanded in [`bed7547`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/bed75477cc828ef1222a56f8840f03bae511a1a0), which introduced forgetting and update latency metrics. Together, these additions make the framework better suited for understanding not just raw task performance, but also how model behavior evolves over time.

Qualitative evaluation was also strengthened. QAFactEval-based evaluation with graph support was added in [`0463bff`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/0463bfffc3c6f29af6cdf04b3b916669b3f3b49a), making it easier to assess output quality beyond basic scalar metrics. That coverage was further extended with the addition of the AUAC metric in [`87f9c3c`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/87f9c3cd268263a2ad6432d0f6a5fa6fec95652e).

The result is a more complete evaluation framework that can capture task accuracy, output quality, training efficiency, and learning dynamics in one system.

## Evaluation Architecture and Maintainability
Infinitune v2 improves not only what is measured, but also how evaluation logic is organized in the codebase.

The Evaluator class was moved into a dedicated utility module in [`d87ee53`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/d87ee532f8b1130f4084cb4085adf77c13e49e46), which is an important structural cleanup for long-term maintainability. This refactor reduces coupling inside the trainer, makes metric computation easier to extend, and creates a clearer separation between orchestration logic and evaluation logic.

That refactoring work is reinforced by the expansion of `utils/eval_metrics_train.py` and `utils/plot_metrics.py` across [`d87ee53`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/d87ee532f8b1130f4084cb4085adf77c13e49e46), [`0bd844f`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/0bd844f7647a97a2d62b6401f1ecf720a3819b6a), [`bed7547`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/bed75477cc828ef1222a56f8840f03bae511a1a0), [`0463bff`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/0463bfffc3c6f29af6cdf04b3b916669b3f3b49a), and [`87f9c3c`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/87f9c3cd268263a2ad6432d0f6a5fa6fec95652e). These changes make the evaluation pipeline more modular and give the platform a stronger foundation for future metrics and reporting features.

The addition of `patch_eval.py` in [`801f3c5`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/801f3c58607c614aeba498da65ada264316bbdf7) also supports this broader shift toward a more explicit and extensible evaluation workflow.

## Training Improvements and Runtime Observability
Training in Infinitune v2 is more configurable, more transparent, and better aligned with the needs of real experimentation.

Adaptive learning rate support was added in [`45272a6`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/45272a65db8c1d789429c0a51f1e906cf99a9730), allowing the training loop to respond more flexibly to model behavior over time. The same change also made training evaluation metrics configurable, which is especially useful when different tasks require different monitoring strategies or when experiments need to focus on a subset of available metrics.

The trainer itself was expanded over several commits to support richer metric calculation and persistence. Metric integration work in [`62c279a`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/62c279aadd326166a58c6696bf5237b94b04222c), [`bed7547`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/bed75477cc828ef1222a56f8840f03bae511a1a0), [`0463bff`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/0463bfffc3c6f29af6cdf04b3b916669b3f3b49a), and [`87f9c3c`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/87f9c3cd268263a2ad6432d0f6a5fa6fec95652e) turns the trainer into a much stronger hub for evaluation-aware experimentation.

Infinitune v2 also improves run stability. Inference bug fixes and general runtime cleanup were introduced in [`2411f33`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/2411f339aa6285996cb232509d4fb38c33e3168e), while metrics persistence and lifecycle reliability were improved in [`0001b28`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/0001b283dc1727fddbac4e6cf92bc6887f18893c). Together, these changes make the system easier to trust during repeated training and evaluation cycles.

## Streaming, Filtering, and Kafka Lifecycle Handling
Another major improvement in v2 is better control over streaming data and test-mode behavior.

Basic filtering support was introduced in the producer path in [`c6afdd3`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/c6afdd3afa82306f6a5cac822730a87ce9639c92), along with the addition of `utils/stream_filter.py`. This created a reusable foundation for filtering streamed content before it enters downstream processing.

That work was expanded in [`0001b28`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/0001b283dc1727fddbac4e6cf92bc6887f18893c), which refined filtering behavior and fixed several operational issues around test mode lifecycle handling, stale Kafka data, and metrics persistence. These updates improve both correctness and usability, especially in scenarios where repeated test runs or older stream data could previously interfere with clean evaluation.

This means Infinitune v2 is better equipped for iterative experimentation in streaming settings, where stale data or inconsistent lifecycle handling can otherwise make results difficult to interpret.

## Task-Specific Configuration Support
Infinitune v2 adds more explicit support for benchmark- and task-specific workflows, especially for GSM8K and IMDb.

Initial config additions and runtime fixes landed in [`2411f33`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/2411f339aa6285996cb232509d4fb38c33e3168e), which introduced key configuration work alongside inference fixes. This was built on by [`c6afdd3`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/c6afdd3afa82306f6a5cac822730a87ce9639c92), [`62c279a`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/62c279aadd326166a58c6696bf5237b94b04222c), [`bed7547`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/bed75477cc828ef1222a56f8840f03bae511a1a0), and [`0463bff`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/0463bfffc3c6f29af6cdf04b3b916669b3f3b49a), which progressively expanded config coverage and aligned those configs with the evolving evaluation framework.

By making these task-specific paths more explicit, v2 makes experimentation more reproducible and reduces the amount of ad hoc setup needed to run meaningful comparisons.

## Visualization and Reporting Improvements
As the evaluation framework grows, visibility becomes more important. Infinitune v2 improves the ability to inspect and interpret results through updated plotting and metrics surfaces.

Plotting support was expanded beginning with [`0bd844f`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/0bd844f7647a97a2d62b6401f1ecf720a3819b6a), then extended in [`bed7547`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/bed75477cc828ef1222a56f8840f03bae511a1a0), [`0463bff`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/0463bfffc3c6f29af6cdf04b3b916669b3f3b49a), and [`87f9c3c`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/87f9c3cd268263a2ad6432d0f6a5fa6fec95652e). These changes ensure that the newly added metrics are not just computed, but also surfaced in ways that are easier to reason about and compare.

This is an important part of the v2 story: the release does not just add metrics, it makes them operationally useful.

## Documentation and Usability
Infinitune v2 also improves the overall developer and user experience through documentation updates.

The README was refreshed across [`2411f33`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/2411f339aa6285996cb232509d4fb38c33e3168e) and [`45272a6`](https://github.com/rohang1411/Infinitune-Realtime-LLM-Fine-Tuning-Framework/commit/45272a65db8c1d789429c0a51f1e906cf99a9730), helping better explain the expanded configuration model, training behavior, and evaluation capabilities introduced in this release.

For a release of this size, this matters. The new capabilities in v2 are substantial, and clearer documentation makes them much easier to adopt effectively.

## Impact
Infinitune v2 should be viewed as a major platform release.

Users upgrading from the previous version can expect:

- broader and more meaningful evaluation coverage
- better visibility into model behavior during training
- improved configuration support for benchmark-driven experimentation
- stronger runtime stability in inference and streaming workflows
- cleaner internal structure for future extension

In practical terms, v2 makes Infinitune more useful both as an experimentation framework and as a more production-aware research platform.
