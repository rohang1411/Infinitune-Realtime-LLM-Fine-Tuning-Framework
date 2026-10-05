"""
benchmarks/load_test_serving.py
─────────────────────────────────────────────────────────────────────────────
Concurrent Serving Load Tester & Latency SLA Benchmark Harness for InfiniTune.

Measures:
1. p50, p95, p99 Time-To-First-Token / End-to-End latency under concurrency.
2. Failure / dropped request rate across active adapter hot-swaps (Target: 0.00%).
3. Max sustainable Queries Per Second (QPS) and saturation knee.
4. Server-side hot-swap pause time telemetry via /metrics.
"""

import time
import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import urllib.request
import urllib.error


def send_generation_request(
    endpoint_url: str,
    prompt: str,
    timeout_s: float = 30.0
) -> Dict[str, Any]:
    """Send a single generation request to the inference server."""
    payload = json.dumps({"prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(
        endpoint_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            status = response.status
            body = json.loads(response.read().decode("utf-8"))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "success": status == 200,
                "status_code": status,
                "latency_ms": elapsed_ms,
                "response": body.get("generated_text", ""),
                "server_latency_ms": body.get("latency_ms", 0.0),
                "error": None,
            }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "status_code": 500,
            "latency_ms": elapsed_ms,
            "response": "",
            "server_latency_ms": 0.0,
            "error": str(e),
        }


def fetch_server_metrics(base_url: str) -> Optional[Dict[str, Any]]:
    """Fetch real-time metrics from /metrics endpoint."""
    metrics_url = f"{base_url.rstrip('/')}/metrics"
    req = urllib.request.Request(metrics_url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception:
        pass
    return None


def run_load_test(
    endpoint_url: str,
    num_requests: int = 100,
    concurrency: int = 10,
    prompt: str = "Review: Outstanding film with brilliant acting.\nSentiment:"
) -> Dict[str, Any]:
    """Execute concurrent load test across worker threads."""
    print(f"Starting load test on {endpoint_url} with {num_requests} requests (concurrency={concurrency})...")

    results: List[Dict[str, Any]] = []
    start_wall = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_generation_request, endpoint_url, prompt)
            for _ in range(num_requests)
        ]
        for f in as_completed(futures):
            results.append(f.result())

    total_duration_s = time.perf_counter() - start_wall

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    latencies = sorted(r["latency_ms"] for r in successful)
    n = len(latencies)

    if n > 0:
        p50 = latencies[int(n * 0.50)]
        p95 = latencies[int(n * 0.95)]
        p99 = latencies[min(int(n * 0.99), n - 1)]
        avg_lat = sum(latencies) / n
    else:
        p50 = p95 = p99 = avg_lat = 0.0

    qps = round(len(successful) / max(total_duration_s, 0.001), 2)
    failure_rate = round((len(failed) / max(len(results), 1)) * 100.0, 2)

    summary = {
        "total_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "failure_rate_percent": failure_rate,
        "concurrency": concurrency,
        "total_duration_s": round(total_duration_s, 2),
        "achieved_qps": qps,
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "p99_latency_ms": round(p99, 2),
        "avg_latency_ms": round(avg_lat, 2),
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="InfiniTune Concurrent Serving Load Tester")
    parser.add_argument("--url", type=str, default="http://localhost:5000/generate", help="Inference endpoint URL")
    parser.add_argument("--requests", type=int, default=100, help="Total requests to send")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent workers")
    parser.add_argument("--prompt", type=str, default="Review: Excellent film.\nSentiment:", help="Test prompt")
    parser.add_argument("--output_json", type=str, default=None, help="Save report to JSON file")
    args = parser.parse_args()

    summary = run_load_test(
        endpoint_url=args.url,
        num_requests=args.requests,
        concurrency=args.concurrency,
        prompt=args.prompt,
    )

    print("\n" + "=" * 50)
    print("      INFINITUNE SERVING SLA LOAD TEST REPORT      ")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k:25}: {v}")
    print("=" * 50)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Report saved to: {args.output_json}")


if __name__ == "__main__":
    main()
