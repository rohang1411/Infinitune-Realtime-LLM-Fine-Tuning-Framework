"""
benchmarks/load_test_serving.py
─────────────────────────────────────────────────────────────────────────────
Concurrent Serving Load Tester & Latency SLA Benchmark Harness for InfiniTune.

Measures closed-loop HTTP end-to-end latency and achieved throughput at a fixed
client concurrency. It does not measure TTFT, saturation knee, or active swaps;
those require streaming/open-loop or dedicated swap experiments.
"""

import time
import json
import argparse
import os
import sys
import platform
import subprocess
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import urllib.request
import urllib.error
import math


def get_git_info() -> Dict[str, str]:
    """Retrieve git commit and branch for provenance tracking."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        commit = "unknown"
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        branch = "unknown"
    return {"git_commit": commit, "git_branch": branch}


def send_generation_request(
    endpoint_url: str,
    prompt: str,
    timeout_s: float = 60.0
) -> Dict[str, Any]:
    """Send a single generation request to the inference server and record version headers."""
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
            header_version = response.headers.get("X-Adapter-Version")
            body = json.loads(response.read().decode("utf-8"))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            
            adapter_version = header_version or body.get("adapter_version", "unknown")
            serving_mode = body.get("serving_mode", "unknown")
            server_latency_ms = body.get("latency_ms", 0.0)

            return {
                "success": status == 200,
                "status_code": status,
                "latency_ms": elapsed_ms,
                "server_latency_ms": server_latency_ms,
                "adapter_version": adapter_version,
                "serving_mode": serving_mode,
                "response": body.get("generated_text", ""),
                "error": None,
            }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "status_code": 500,
            "latency_ms": elapsed_ms,
            "server_latency_ms": 0.0,
            "adapter_version": "none",
            "serving_mode": "none",
            "response": "",
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
    prompt: str = "Review: Outstanding film with brilliant acting.\nSentiment:",
    raw_output: Optional[str] = None,
    progress=None,
    timeout_s: float = 60.0,
) -> Dict[str, Any]:
    """Execute concurrent load test across worker threads."""
    print(f"Starting load test on {endpoint_url} with {num_requests} requests (concurrency={concurrency})...")

    results: List[Dict[str, Any]] = []
    start_wall = time.perf_counter()

    if num_requests < 1 or concurrency < 1:
        raise ValueError("requests and concurrency must be positive")
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_generation_request, endpoint_url, prompt, timeout_s)
            for _ in range(num_requests)
        ]
        raw_file = open(raw_output, "w", encoding="utf-8", buffering=1) if raw_output else None
        try:
            for index, f in enumerate(as_completed(futures), 1):
                result = f.result()
                result["completion_index"] = index
                results.append(result)
                if raw_file:
                    raw_file.write(json.dumps(result) + "\n")
                if progress:
                    progress(index, num_requests)
        finally:
            if raw_file:
                raw_file.close()

    total_duration_s = time.perf_counter() - start_wall

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    client_latencies = sorted(r["latency_ms"] for r in successful)
    server_latencies = sorted(r["server_latency_ms"] for r in successful if r.get("server_latency_ms", 0.0) > 0)
    n = len(client_latencies)

    def percentile(values, fraction):
        if not values:
            return 0.0
        position = (len(values) - 1) * fraction
        low = math.floor(position)
        high = math.ceil(position)
        return values[low] + (values[high] - values[low]) * (position - low)

    if n > 0:
        p50 = percentile(client_latencies, .50)
        p95 = percentile(client_latencies, .95)
        p99 = percentile(client_latencies, .99)
        avg_lat = sum(client_latencies) / n
        min_lat = client_latencies[0]
        max_lat = client_latencies[-1]
    else:
        p50 = p95 = p99 = avg_lat = min_lat = max_lat = 0.0

    sn = len(server_latencies)
    if sn > 0:
        srv_p50 = percentile(server_latencies, .50)
        srv_p95 = percentile(server_latencies, .95)
        srv_p99 = percentile(server_latencies, .99)
        srv_avg = sum(server_latencies) / sn
    else:
        srv_p50 = srv_p95 = srv_p99 = srv_avg = 0.0

    qps = round(len(successful) / max(total_duration_s, 0.001), 2)
    failure_rate = round((len(failed) / max(len(results), 1)) * 100.0, 2)

    unique_versions = sorted(list({r["adapter_version"] for r in successful if r.get("adapter_version")}))
    detected_modes = sorted(list({r["serving_mode"] for r in successful if r.get("serving_mode")}))

    # Base URL for metrics
    parts = endpoint_url.rsplit("/", 1)
    base_url = parts[0] if len(parts) > 1 else endpoint_url
    server_metrics = fetch_server_metrics(base_url)

    git_info = get_git_info()

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["git_commit"],
        "git_branch": git_info["git_branch"],
        "environment": {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
        },
        "endpoint_url": endpoint_url,
        "serving_modes_detected": detected_modes,
        "unique_adapter_versions": unique_versions,
        "total_requests": len(results),
        "raw_requests_path": raw_output,
        "latency_definition": "client end-to-end; TTFT is unavailable from non-streaming /generate",
        "p99_reliable": len(successful) >= 1000,
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "failure_rate_percent": failure_rate,
        "concurrency": concurrency,
        "total_duration_s": round(total_duration_s, 2),
        "achieved_qps": qps,
        "client_latency_ms": {
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "avg": round(avg_lat, 2),
            "min": round(min_lat, 2),
            "max": round(max_lat, 2),
        },
        "server_latency_ms": {
            "p50": round(srv_p50, 2),
            "p95": round(srv_p95, 2),
            "p99": round(srv_p99, 2),
            "avg": round(srv_avg, 2),
        },
        "server_telemetry": server_metrics or {},
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="InfiniTune Concurrent Serving Load Tester")
    parser.add_argument("--url", type=str, default="http://localhost:8000/generate", help="Inference endpoint URL")
    parser.add_argument("--requests", type=int, default=100, help="Total requests to send")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent workers")
    parser.add_argument("--prompt", type=str, default="Review: Excellent film.\nSentiment:", help="Test prompt")
    parser.add_argument("--output_json", type=str, default=None, help="Save report to JSON file")
    parser.add_argument("--raw-output", type=str, default=None, help="JSONL for every request")
    args = parser.parse_args()

    summary = run_load_test(
        endpoint_url=args.url,
        num_requests=args.requests,
        concurrency=args.concurrency,
        prompt=args.prompt,
        raw_output=args.raw_output,
    )

    print("\n" + "=" * 60)
    print("        INFINITUNE SERVING SLA LOAD TEST REPORT        ")
    print("=" * 60)
    print(f"  Timestamp UTC      : {summary['timestamp_utc']}")
    print(f"  Git Commit         : {summary['git_commit'][:10]}")
    print(f"  Detected Mode      : {summary['serving_modes_detected']}")
    print(f"  Adapter Versions   : {summary['unique_adapter_versions']}")
    print(f"  Total Requests     : {summary['total_requests']}")
    print(f"  Successful         : {summary['successful_requests']}")
    print(f"  Failed             : {summary['failed_requests']} ({summary['failure_rate_percent']}%)")
    print(f"  Duration           : {summary['total_duration_s']} s")
    print(f"  Achieved QPS       : {summary['achieved_qps']}")
    print(f"  Client Latency p50 : {summary['client_latency_ms']['p50']} ms")
    print(f"  Client Latency p95 : {summary['client_latency_ms']['p95']} ms")
    print(f"  Client Latency p99 : {summary['client_latency_ms']['p99']} ms")
    print(f"  Client Latency Avg : {summary['client_latency_ms']['avg']} ms")
    print("=" * 60)

    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Report saved to: {args.output_json}")


if __name__ == "__main__":
    main()
