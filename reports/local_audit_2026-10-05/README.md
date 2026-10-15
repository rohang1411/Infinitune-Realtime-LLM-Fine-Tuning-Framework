# Local serving audit (2026-10-05 PDT)

These four JSON files are single-run observations from commit `06e3e17147c8bcb3ef6ebd5a70a5e501b322dcd4` on a Windows CPU host. Python 3.10.16, PyTorch 2.10.0+cpu, Transformers 5.5.3, and PEFT 0.18.1 were installed. All runs loaded the same local `output/imdb/checkpoints/distilgpt2__imdb/final` checkpoint, used the load tester's default prompt, and sent 20 requests to `localhost:8011/generate`. The server was restarted between serving modes. No Kafka traffic or live adapter swaps occurred.

| Mode | Concurrent clients | Successful requests | Requests/s | Client p95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Router | 2 | 20/20 | 0.80 | 3219.59 |
| Legacy lock | 2 | 20/20 | 0.88 | 2481.18 |
| Router | 4 | 20/20 | 1.52 | 2702.53 |
| Legacy lock | 4 | 20/20 | 1.70 | 2729.51 |

Commands (run from the repository root with the Python 3.10 environment):

```powershell
python inference.py --config output/imdb/config.yaml --checkpoint final --serving-mode router --port 8011
python benchmarks/load_test_serving.py --url http://localhost:8011/generate --requests 20 --concurrency 2 --output_json reports/local_audit_2026-10-05/router_c2.json
python benchmarks/load_test_serving.py --url http://localhost:8011/generate --requests 20 --concurrency 4 --output_json reports/local_audit_2026-10-05/router_c4.json
# Stop the server, then restart with --serving-mode legacy_lock.
python benchmarks/load_test_serving.py --url http://localhost:8011/generate --requests 20 --concurrency 2 --output_json reports/local_audit_2026-10-05/legacy_c2.json
python benchmarks/load_test_serving.py --url http://localhost:8011/generate --requests 20 --concurrency 4 --output_json reports/local_audit_2026-10-05/legacy_c4.json
```

These short closed-loop bursts do not estimate sustainable capacity, tail-latency SLAs, or swap reliability. They also cannot isolate a stable mode effect without repeated, alternated, same-load trials. The legacy mode reports `v0_base` in its response header even though the server loaded the `final` checkpoint; its `/metrics` endpoint reports the latter because it reads the separately initialized router state.
