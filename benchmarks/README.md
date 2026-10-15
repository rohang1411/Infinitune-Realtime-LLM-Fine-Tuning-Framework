# Benchmark source and saved runs

This folder contains **code**, not the output of executed evaluations. Local results are under [`../reports/benchmark_runs/`](../reports/benchmark_runs/); the run folders are Git ignored.

From the repository root, open the local result browser with `./run_evaluation.ps1 dashboard` on Windows or `bash run_evaluation.sh dashboard` on macOS/Linux. It lists earlier CLI runs and new dashboard runs, separates data science from engineering stages, and links to raw evidence. The local verified smoke manifest is `../reports/benchmark_runs/verified_smoke_20261006/run_manifest.json`. A fresh clone will not contain that ignored run directory.

See [Project Context §13](../docs/Infinitune_Project_Context.md#13-reproducible-evaluation-evidence-ledger-and-external-audit) for methods, claims, failure handling, and commands.

### Environment Census
Run `python -m benchmarks.env_check` (or with `--kafka-bootstrap localhost:9092`) to record hardware, PyTorch capability, and dependency versions in `reports/env/`.

