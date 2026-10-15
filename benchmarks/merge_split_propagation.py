"""
Merge split-mode propagation logs across distributed hosts (e.g. Colab trainer -> Laptop serving).
Applies NTP clock-offset correction, flags timing drift, and calculates propagation distributions.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    import ntplib
    HAS_NTPLIB = True
except ImportError:
    HAS_NTPLIB = False

from utils.serving_router import nearest_rank_percentile


def query_ntp_offset(host: str = "time.google.com", timeout: float = 2.0) -> Optional[float]:
    """Query a public NTP server for local clock offset. Returns seconds (server - local)."""
    if not HAS_NTPLIB:
        return None
    try:
        client = ntplib.NTPClient()
        response = client.request(host, version=3, timeout=timeout)
        return float(response.offset)
    except Exception:
        return None


def merge_split_propagation_records(
    trainer_records: List[Dict[str, Any]],
    server_records: List[Dict[str, Any]],
    trainer_offset_start: float = 0.0,
    trainer_offset_end: float = 0.0,
    server_offset_start: float = 0.0,
    server_offset_end: float = 0.0,
) -> Dict[str, Any]:
    """
    Join trainer and server event records by (run_id, seq), applying clock skew correction.
    
    Formula:
      offset = server_ntp - local_clock
      t_applied_corrected = t_applied_server - server_offset + trainer_offset
      propagation_s = t_applied_corrected - t_publish_start_trainer
    """
    trainer_drift = abs(trainer_offset_end - trainer_offset_start)
    server_drift = abs(server_offset_end - server_offset_start)
    drift_flag = (trainer_drift > 0.050) or (server_drift > 0.050)

    mean_trainer_offset = (trainer_offset_start + trainer_offset_end) / 2.0
    mean_server_offset = (server_offset_start + server_offset_end) / 2.0

    # Index server events by (run_id, seq) or seq
    server_map = {}
    for rec in server_records:
        key = (str(rec.get("run_id", "")), rec.get("seq"))
        server_map[key] = rec
        if rec.get("seq") is not None:
            server_map[rec.get("seq")] = rec

    merged_items = []
    latencies = []

    for t_rec in trainer_records:
        seq = t_rec.get("seq")
        run_id = str(t_rec.get("run_id", ""))
        s_rec = server_map.get((run_id, seq)) or server_map.get(seq)
        if not s_rec:
            continue

        t_pub = t_rec.get("t_publish_start") or t_rec.get("t_start") or t_rec.get("timestamp")
        t_app = s_rec.get("t_applied") or s_rec.get("timestamp")

        if t_pub is None or t_app is None:
            continue

        # Correct for clock skew across the distributed hosts
        t_app_corrected = t_app - mean_server_offset + mean_trainer_offset
        prop_s = max(0.0, t_app_corrected - t_pub)
        latencies.append(prop_s)

        merged_items.append({
            "run_id": run_id,
            "seq": seq,
            "t_publish_trainer": t_pub,
            "t_applied_server_raw": t_app,
            "t_applied_server_corrected": round(t_app_corrected, 6),
            "propagation_s": round(prop_s, 6),
        })

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    return {
        "status": "complete" if n > 0 else "no_matched_records",
        "matched_count": n,
        "trainer_offset_start": round(trainer_offset_start, 6),
        "trainer_offset_end": round(trainer_offset_end, 6),
        "trainer_clock_drift_s": round(trainer_drift, 6),
        "server_offset_start": round(server_offset_start, 6),
        "server_offset_end": round(server_offset_end, 6),
        "server_clock_drift_s": round(server_drift, 6),
        "drift_warning_flag": drift_flag,
        "propagation_s_p50": nearest_rank_percentile(latencies_sorted, 0.50) if n else 0.0,
        "propagation_s_p95": nearest_rank_percentile(latencies_sorted, 0.95) if n else 0.0,
        "propagation_s_p99": nearest_rank_percentile(latencies_sorted, 0.99) if n else 0.0,
        "propagation_s_min": min(latencies_sorted) if n else 0.0,
        "propagation_s_max": max(latencies_sorted) if n else 0.0,
        "propagation_s_mean": round(sum(latencies_sorted) / n, 6) if n else 0.0,
        "items": merged_items,
    }


def main():
    parser = argparse.ArgumentParser(description="Merge distributed split-mode propagation records.")
    parser.add_argument("--trainer-log", type=Path, required=True, help="JSON or JSONL file from trainer host")
    parser.add_argument("--server-log", type=Path, required=True, help="JSON or JSONL file from server host")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--trainer-offset-start", type=float, default=0.0)
    parser.add_argument("--trainer-offset-end", type=float, default=0.0)
    parser.add_argument("--server-offset-start", type=float, default=0.0)
    parser.add_argument("--server-offset-end", type=float, default=0.0)
    args = parser.parse_args()

    def load_records(path: Path) -> List[Dict[str, Any]]:
        text = path.read_text(encoding="utf-8").strip()
        if text.startswith("["):
            return json.loads(text)
        elif text.startswith("{"):
            data = json.loads(text)
            if "items" in data and isinstance(data["items"], list):
                return data["items"]
            return [data]
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    trainer_recs = load_records(args.trainer_log)
    server_recs = load_records(args.server_log)

    summary = merge_split_propagation_records(
        trainer_recs,
        server_recs,
        trainer_offset_start=args.trainer_offset_start,
        trainer_offset_end=args.trainer_offset_end,
        server_offset_start=args.server_offset_start,
        server_offset_end=args.server_offset_end,
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Summary written to {args.out}")
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
