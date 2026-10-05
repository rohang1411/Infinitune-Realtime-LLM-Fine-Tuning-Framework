"""
benchmarks/canary_freshness_probe.py
─────────────────────────────────────────────────────────────────────────────
End-to-End Pipeline Freshness Probe for InfiniTune.

Measures the headline ML Systems latency:
Exact wall-clock duration from the moment a training record is published
by the producer to the moment the corresponding behavior change is observable
on the live /generate inference endpoint.
"""

import time
import json
import argparse
from typing import Dict, Any, Optional
import urllib.request
import urllib.error


def poll_endpoint_for_keyword(
    endpoint_url: str,
    probe_prompt: str,
    expected_target: str,
    poll_interval_s: float = 0.5,
    timeout_s: float = 120.0
) -> Dict[str, Any]:
    """
    Poll the /generate endpoint until the response contains expected_target
    or timeout expires.
    """
    t0 = time.time()
    attempts = 0
    payload = json.dumps({"prompt": probe_prompt}).encode("utf-8")

    while (time.time() - t0) < timeout_s:
        attempts += 1
        req = urllib.request.Request(
            endpoint_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    generated = data.get("generated_text", "").strip().lower()
                    if expected_target.lower() in generated:
                        elapsed = time.time() - t0
                        return {
                            "success": True,
                            "freshness_latency_s": round(elapsed, 2),
                            "attempts": attempts,
                            "observed_response": generated,
                            "active_version": data.get("version", "unknown"),
                        }
        except Exception:
            pass

        time.sleep(poll_interval_s)

    return {
        "success": False,
        "freshness_latency_s": round(time.time() - t0, 2),
        "attempts": attempts,
        "observed_response": "TIMEOUT",
        "active_version": "unknown",
    }


def main():
    parser = argparse.ArgumentParser(description="InfiniTune Freshness Probe Harness")
    parser.add_argument("--url", type=str, default="http://localhost:5000/generate", help="Inference URL")
    parser.add_argument("--prompt", type=str, required=True, help="Probe prompt text")
    parser.add_argument("--target", type=str, required=True, help="Expected keyword target")
    parser.add_argument("--timeout", type=float, default=60.0, help="Max wait seconds")
    parser.add_argument("--interval", type=float, default=0.5, help="Poll interval in seconds")
    args = parser.parse_args()

    print(f"Starting freshness probe on {args.url} (target='{args.target}')...")
    res = poll_endpoint_for_keyword(
        endpoint_url=args.url,
        probe_prompt=args.prompt,
        expected_target=args.target,
        poll_interval_s=args.interval,
        timeout_s=args.timeout,
    )

    print("\n" + "=" * 45)
    print("      FRESHNESS PROBE RESULT      ")
    print("=" * 45)
    for k, v in res.items():
        print(f"  {k:22}: {v}")
    print("=" * 45)


if __name__ == "__main__":
    main()
