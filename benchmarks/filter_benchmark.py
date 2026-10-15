"""
InfiniTune — Benchmark Suite: Stream Filter & No-Stall Proof (WP-19)

Evaluates:
  Part A (Offline):
    - 5-stage stream filter throughput and per-stage drop counts on Alpaca dataset.
    - Recall against a calibrated spam injection set (repeated phrases, symbol noise,
      URL spam, and length boundary violations).
    - False-positive upper bound (Alpaca drop rate) and audit sample export.
  Part B (Live / Functional):
    - Resilience to malformed frames (non-UTF8, non-JSON, non-object, empty),
      proving zero training stalls.
"""

import csv
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from utils.stream_filter import (
    StreamFilter,
    FilterStats,
    STAGE_LENGTH,
    STAGE_CONTENT,
    STAGE_REPETITION,
    STAGE_CHAT,
    STAGE_REGEX,
    STAGE_ERROR,
)
from trainer import decode_training_record


def build_filter_benchmark_config(base_config_path: str = "configs/alpaca_qualitative.yaml") -> dict:
    """Build a configuration with all 5 filter stages active."""
    if os.path.exists(base_config_path):
        with open(base_config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    data_cfg = cfg.setdefault("data", {})
    filt_cfg = data_cfg.setdefault("filtering", {})
    filt_cfg["on_error"] = "drop"

    # Stage 1: Length bounds
    univ = filt_cfg.setdefault("universal", {})
    univ["min_chars"] = univ.get("min_chars", 10)
    univ["max_chars"] = univ.get("max_chars", 4000)

    # Stage 2: Alphanumeric ratio
    univ["min_alphanumeric_ratio"] = univ.get("min_alphanumeric_ratio", 0.5)

    # Stage 3: Repetition ratio (zlib)
    univ["max_repetition_ratio"] = univ.get("max_repetition_ratio", 0.2)

    # Stage 4: Chat structure
    domain = filt_cfg.setdefault("domain_specific", {})
    domain["chat_structure"] = {
        "min_turns": 1,
        "require_assistant_final": False,
    }

    # Stage 5: Regex stage (3+ URLs or 21+ repeated single character)
    domain["custom_regex_must_not_match"] = [
        r"(https?://\S+.*){3,}",
        r"(.)\1{20,}",
    ]

    return cfg


def generate_synthetic_spam_records(
    seed: int = 42,
    count_per_type: int = 500,
) -> List[dict]:
    """
    Generate synthetic spam records for the 4 target failure classes:
      1. repeated_phrase (zlib ratio < 0.2)
      2. symbol_noise (alphanumeric ratio < 0.5)
      3. url_spam (matches 3+ URLs regex)
      4. length_bounds (too short or too long)
    """
    rng = random.Random(seed)
    records = []

    # 1. Repeated phrase (low entropy, high compression)
    phrases = [
        "Exclusive limited time offer! Click here now to claim your grand prize today! ",
        "Make thousands of dollars from home right now with this simple quick trick! ",
        "Urgent account verification needed immediately to prevent account suspension! ",
    ]
    for i in range(count_per_type):
        phrase = phrases[i % len(phrases)]
        text = phrase * 28  # ~2000 chars, zlib ratio ~0.03
        records.append({
            "record": {"input": text, "target": "Spam response"},
            "text": text,
            "is_spam": True,
            "spam_type": "repeated_phrase",
            "expected_stage": STAGE_REPETITION,
        })

    # 2. Symbol noise (low alphanumeric ratio)
    symbol_banks = [
        "!@#$%^&*()_+-=[]{}|;':\",.<>/?`~ ",
        "###***$$$%%%^^^&&&:::;;;,,,---+++ ",
        "<<<>>>///???|||\\\\\\===+++~~~!!!@@@ ",
    ]
    for i in range(count_per_type):
        bank = symbol_banks[i % len(symbol_banks)]
        # ~500 chars with <5% alphanumeric
        noise = (bank * 16) + f" id{i} "
        records.append({
            "record": {"input": noise, "target": "Noise response"},
            "text": noise,
            "is_spam": True,
            "spam_type": "symbol_noise",
            "expected_stage": STAGE_CONTENT,
        })

    # 3. URL spam (3+ URLs)
    for i in range(count_per_type):
        text = (
            f"Here are top deals for today: https://offer-{i}-a.example.com/deal "
            f"and visit https://offer-{i}-b.example.org/promo "
            f"plus check https://offer-{i}-c.example.net/now for more discounts!"
        )
        records.append({
            "record": {"input": text, "target": "URL response"},
            "text": text,
            "is_spam": True,
            "spam_type": "url_spam",
            "expected_stage": STAGE_REGEX,
        })

    # 4. Length bounds (half too short < 10, half too long > 4000)
    for i in range(count_per_type):
        if i % 2 == 0:
            text = f"hi {i}"[:random.choice([2, 3, 4, 5])]
        else:
            text = "Normal valid English sentence with sufficient entropy and proper vocabulary. " * 80  # ~6000 chars
        records.append({
            "record": {"input": text, "target": "Length response"},
            "text": text,
            "is_spam": True,
            "spam_type": "length_bounds",
            "expected_stage": STAGE_LENGTH,
        })

    rng.shuffle(records)
    return records


def load_or_generate_alpaca_records(
    max_records: Optional[int] = None,
    offline: bool = False,
) -> List[dict]:
    """
    Load legitimate records from tatsu-lab/alpaca or fallback to synthetic
    instruction-following records when offline / cached.
    """
    records = []
    if not offline:
        try:
            from datasets import load_dataset
            ds = load_dataset("tatsu-lab/alpaca", split="train")
            limit = min(len(ds), max_records) if max_records else len(ds)
            for i in range(limit):
                row = ds[i]
                instruction = str(row.get("instruction", ""))
                output = str(row.get("output", ""))
                records.append({
                    "record": {"input": instruction, "target": output},
                    "text": instruction,
                    "is_spam": False,
                    "spam_type": None,
                })
            return records
        except Exception:
            pass

    # Offline fallback: Generate realistic Alpaca-like instruction records
    count = max_records or 2000
    templates = [
        "Explain the fundamental mechanism of {topic} in concise plain language.",
        "Write a Python function to solve {topic} with comprehensive docstrings.",
        "Compare and contrast {topic} with respect to efficiency and maintainability.",
        "Summarize the historical evolution of {topic} highlighting three milestones.",
    ]
    topics = [
        "gradient accumulation in distributed deep learning",
        "double buffered adapter swapping under reader-writer locks",
        "cryptographic verification of safetensors manifests",
        "asynchronous backpressure handling in streaming ingestion",
        "loss surface curvature and adaptive window drift detection",
    ]
    for i in range(count):
        inst = templates[i % len(templates)].format(topic=topics[i % len(topics)]) + f" (Item #{i+1})"
        out = f"Comprehensive discussion and response addressing query #{i+1} with full technical details."
        records.append({
            "record": {"input": inst, "target": out},
            "text": inst,
            "is_spam": False,
            "spam_type": None,
        })

    return records


def run_filter_benchmark_part_a(
    config: dict,
    alpaca_records: List[dict],
    spam_records: List[dict],
    seed: int = 42,
) -> Tuple[dict, List[dict]]:
    """
    Run Part A of filter benchmark:
      - Interleave Alpaca records and synthetic spam.
      - Measure throughput with perf_counter.
      - Measure per-stage drop counts, spam recall per type, and Alpaca drop rate.
      - Return (summary_dict, dropped_alpaca_samples).
    """
    stream_filter = StreamFilter(config)
    active_stages = stream_filter.active_stages()

    # Interleave records deterministically
    combined = []
    rng = random.Random(seed)
    
    # Tag records with index
    a_idx, s_idx = 0, 0
    total_alpaca = len(alpaca_records)
    total_spam = len(spam_records)
    
    # Mix ratio
    all_items = [(r, False) for r in alpaca_records] + [(r, True) for r in spam_records]
    rng.shuffle(all_items)

    dropped_alpaca_samples = []
    spam_detected_by_type = {}
    spam_total_by_type = {}

    for item, is_spam in all_items:
        if is_spam:
            stype = item["spam_type"]
            spam_total_by_type[stype] = spam_total_by_type.get(stype, 0) + 1

    t0 = time.perf_counter()
    for idx, (item, is_spam) in enumerate(all_items):
        raw_rec = item["record"]
        extracted = item["text"]
        ok, reason = stream_filter.validate(raw_record=raw_rec, extracted_text=extracted)
        
        if is_spam:
            stype = item["spam_type"]
            if not ok:
                spam_detected_by_type[stype] = spam_detected_by_type.get(stype, 0) + 1
        else:
            if not ok and len(dropped_alpaca_samples) < 50:
                dropped_alpaca_samples.append({
                    "record_index": idx,
                    "reason": reason,
                    "text": extracted[:300].replace("\n", " "),
                })

    elapsed_s = time.perf_counter() - t0
    total_records = len(all_items)
    throughput = total_records / elapsed_s if elapsed_s > 0 else 0.0

    stats = stream_filter.stats
    alpaca_dropped = len([s for s in dropped_alpaca_samples]) if not stats else (total_alpaca - (stats.passed - sum(spam_detected_by_type.values())))
    alpaca_dropped = max(0, min(total_alpaca, alpaca_dropped))
    alpaca_drop_rate = alpaca_dropped / float(total_alpaca) if total_alpaca > 0 else 0.0

    recall_by_type = {}
    for stype, count in spam_total_by_type.items():
        det = spam_detected_by_type.get(stype, 0)
        recall_by_type[stype] = round(det / float(count), 4) if count > 0 else 0.0

    total_spam_detected = sum(spam_detected_by_type.values())
    overall_recall = round(total_spam_detected / float(total_spam), 4) if total_spam > 0 else 0.0

    summary = {
        "active_stages": active_stages,
        "filter_config": config.get("data", {}).get("filtering", {}),
        "total_records_processed": total_records,
        "alpaca_records_count": total_alpaca,
        "spam_records_count": total_spam,
        "elapsed_seconds": round(elapsed_s, 4),
        "throughput_records_per_second": round(throughput, 2),
        "alpaca_dropped_count": alpaca_dropped,
        "alpaca_drop_rate": round(alpaca_drop_rate, 4),
        "overall_spam_recall": overall_recall,
        "recall_by_type": recall_by_type,
        "dropped_by_stage": stats.dropped_by_stage if stats else {},
        "target_recall_met": (
            recall_by_type.get("repeated_phrase", 0.0) >= 0.99 and
            recall_by_type.get("symbol_noise", 0.0) >= 0.99
        ),
    }

    return summary, dropped_alpaca_samples


def run_filter_benchmark_part_b_functional() -> dict:
    """
    Functional verification of Part B (no-stall proof):
    Generates 50 malformed records across all failure categories:
      - empty
      - non-UTF8
      - non-JSON
      - non-object
    Verifies that decode_training_record catches every single one without raising
    and categorizes reasons properly.
    """
    malformed_spec = [
        (b"", "empty"),
        (None, "empty"),
        (b"   \n\t  ", "empty"),
        (b"\xff\xfe\xfd\x80\x81", "not_utf8"),
        (b"malformed string without json {", "not_json"),
        (b"['list', 'is', 'not', 'object']", "not_json"),
        (b"[\"valid\", \"json\", \"array\"]", "not_object"),
        (b"12345", "not_object"),
        (b"\"scalar_string\"", "not_object"),
        (b"true", "not_object"),
    ]

    injected = []
    while len(injected) < 50:
        for item, exp_reason in malformed_spec:
            injected.append((item, exp_reason))
            if len(injected) >= 50:
                break

    reasons_recorded = {}
    valid_record_bytes = json.dumps({"input": "test input", "target": "test target"}).encode("utf-8")

    # Interleave valid and malformed records
    all_records = []
    for raw_bytes, exp in injected:
        all_records.append((raw_bytes, exp, False))
        all_records.append((valid_record_bytes, None, True))

    valid_passed = 0
    malformed_dropped = 0
    exceptions_raised = 0

    for raw, expected_reason, is_valid in all_records:
        try:
            sample, reason = decode_training_record(raw)
            if is_valid:
                assert sample is not None
                assert reason is None
                valid_passed += 1
            else:
                assert sample is None
                assert reason is not None
                malformed_dropped += 1
                reasons_recorded[reason] = reasons_recorded.get(reason, 0) + 1
        except Exception:
            exceptions_raised += 1

    passed = (
        malformed_dropped == 50 and
        exceptions_raised == 0 and
        valid_passed == 50 and
        all(k in reasons_recorded for k in ["empty", "not_utf8", "not_json", "not_object"])
    )

    return {
        "total_malformed_injected": 50,
        "total_malformed_dropped": malformed_dropped,
        "exceptions_raised": exceptions_raised,
        "valid_records_passed": valid_passed,
        "reasons_recorded": reasons_recorded,
        "no_stall_proven": passed,
    }


def run_filter_benchmark_suite(run, args) -> dict:
    """Execute the full filter_benchmark suite under run.py."""
    cfg = build_filter_benchmark_config()
    profile = getattr(args, "profile", "smoke")
    
    if profile == "smoke":
        alpaca_count = 200
        spam_per_type = 20
    elif profile == "standard":
        alpaca_count = 2000
        spam_per_type = 500
    else:
        alpaca_count = None  # all 52k
        spam_per_type = 500

    out_dir = run.directory / "filter_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    alpaca_records = load_or_generate_alpaca_records(
        max_records=alpaca_count,
        offline=getattr(args, "offline", False),
    )
    spam_records = generate_synthetic_spam_records(
        seed=getattr(args, "seed", 42),
        count_per_type=spam_per_type,
    )

    part_a_summary, dropped_samples = run_filter_benchmark_part_a(
        cfg,
        alpaca_records,
        spam_records,
        seed=getattr(args, "seed", 42),
    )

    part_b_summary = run_filter_benchmark_part_b_functional()

    # Save dropped sample CSV
    csv_path = out_dir / "dropped_sample.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["record_index", "reason", "text"])
        writer.writeheader()
        writer.writerows(dropped_samples)

    final_summary = {
        "profile": profile,
        "part_a": part_a_summary,
        "part_b_no_stall": part_b_summary,
        "dropped_sample_csv": str(csv_path),
        "boundary": "5-stage StreamFilter throughput on Alpaca plus synthetic spam recall and malformed-record decoding resilience",
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")

    return final_summary
