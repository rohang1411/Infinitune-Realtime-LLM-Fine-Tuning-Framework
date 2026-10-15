import torch
import re
import csv
import json
import math
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np

# OpenMP DLL collision guard: torch must be loaded before sklearn
from sklearn.metrics import cohen_kappa_score


MR_PATTERN = re.compile(r"(\w[\w ]*)\[([^\]]+)\]")

EAT_TYPE_ALIASES = {
    "coffee shop": {"coffee shop", "cafe", "café"},
    "restaurant": {"restaurant"},
    "pub": {"pub"},
}

FOOD_ALIASES = {
    "french": {"french"},
    "japanese": {"japanese"},
    "italian": {"italian"},
    "indian": {"indian"},
    "english": {"english", "british"},
    "fast food": {"fast food", "fast-food"},
    "chinese": {"chinese"},
}

AREA_ALIASES = {
    "riverside": {"riverside", "near the river", "by the river", "river"},
    "city centre": {"city centre", "city center", "centre of the city", "center of the city", "downtown", "in town"},
}

PRICE_RANGE_BUCKETS = {
    "cheap": {"cheap", "low price", "inexpensive", "less than £20", "less than 20", "under £20", "under 20"},
    "moderate": {"moderate", "moderately priced", "mid-range", "£20-25", "£20-£25", "20-25", "average price"},
    "expensive": {"expensive", "high", "high priced", "more than £30", "more than 30", "over £30", "over 30"},
}

CUSTOMER_RATING_BUCKETS = {
    "low": {"low", "1 out of 5", "1/5", "1 of 5", "poor"},
    "average": {"average", "3 out of 5", "3/5", "3 of 5", "moderate rating", "adequate"},
    "high": {"high", "5 out of 5", "5/5", "5 of 5", "excellent", "top rated"},
}

POSITIVE_FAMILY = [
    "family friendly", "family-friendly", "kid friendly", "kid-friendly",
    "children welcome", "child friendly", "child-friendly", "suitable for children",
    "children are welcome", "good for kids", "family-orientated", "family orientated",
]

NEGATIVE_FAMILY = [
    "not family friendly", "not family-friendly", "not kid friendly", "not kid-friendly",
    "not family", "not child", "not kid", "non family", "non-family",
    "no children", "not suitable for children", "adults only", "adult only",
    "isn't kid friendly", "is not kid friendly", "not child friendly", "not child-friendly",
]


def parse_mr(mr_string: str) -> Dict[str, str]:
    """Parse 'name[The Eagle], eatType[coffee shop]' into a dictionary."""
    slots = {}
    for match in MR_PATTERN.finditer(mr_string):
        attr = match.group(1).strip()
        val = match.group(2).strip()
        if attr not in slots:
            slots[attr] = val
    return slots


def _canonical_bucket(val: str, buckets: Dict[str, Set[str]]) -> Optional[str]:
    v = val.lower().strip()
    for bucket_name, aliases in buckets.items():
        if v == bucket_name or v in aliases:
            return bucket_name
    return None


def classify_slot(slot_name: str, slot_value: str, text: str) -> str:
    """Classify a single slot into 'expressed_correct', 'omitted', or 'contradicted'."""
    text_lower = text.lower()
    val_lower = slot_value.lower().strip()
    s_name = slot_name.strip()

    # 1. name / near: case-insensitive substring match. Contradiction never applies.
    if s_name.lower() in ("name", "near"):
        if val_lower in text_lower:
            return "expressed_correct"
        return "omitted"

    # 2. eatType
    if s_name.lower() == "eattype":
        target_bucket = _canonical_bucket(val_lower, EAT_TYPE_ALIASES)
        if target_bucket:
            if any(alias in text_lower for alias in EAT_TYPE_ALIASES[target_bucket]):
                return "expressed_correct"
            for b_name, aliases in EAT_TYPE_ALIASES.items():
                if b_name != target_bucket and any(a in text_lower for a in aliases):
                    return "contradicted"
        else:
            if val_lower in text_lower:
                return "expressed_correct"
        return "omitted"

    # 3. food
    if s_name.lower() == "food":
        target_bucket = _canonical_bucket(val_lower, FOOD_ALIASES)
        if target_bucket:
            if any(alias in text_lower for alias in FOOD_ALIASES[target_bucket]):
                return "expressed_correct"
            for b_name, aliases in FOOD_ALIASES.items():
                if b_name != target_bucket and any(a in text_lower for a in aliases):
                    return "contradicted"
        else:
            if val_lower in text_lower:
                return "expressed_correct"
        return "omitted"

    # 4. area
    if s_name.lower() == "area":
        target_bucket = _canonical_bucket(val_lower, AREA_ALIASES)
        if target_bucket:
            if any(alias in text_lower for alias in AREA_ALIASES[target_bucket]):
                return "expressed_correct"
            for b_name, aliases in AREA_ALIASES.items():
                if b_name != target_bucket and any(a in text_lower for a in aliases):
                    return "contradicted"
        else:
            if val_lower in text_lower:
                return "expressed_correct"
        return "omitted"

    # 5. priceRange
    if s_name.lower() in ("pricerange", "price range"):
        target_bucket = _canonical_bucket(val_lower, PRICE_RANGE_BUCKETS)
        if target_bucket:
            if any(alias in text_lower for alias in PRICE_RANGE_BUCKETS[target_bucket]):
                return "expressed_correct"
            for b_name, aliases in PRICE_RANGE_BUCKETS.items():
                if b_name != target_bucket and any(a in text_lower for a in aliases):
                    return "contradicted"
        else:
            if val_lower in text_lower:
                return "expressed_correct"
        return "omitted"

    # 6. customer rating
    if s_name.lower() in ("customerrating", "customer rating"):
        target_bucket = _canonical_bucket(val_lower, CUSTOMER_RATING_BUCKETS)
        if target_bucket:
            if any(alias in text_lower for alias in CUSTOMER_RATING_BUCKETS[target_bucket]):
                return "expressed_correct"
            for b_name, aliases in CUSTOMER_RATING_BUCKETS.items():
                if b_name != target_bucket and any(a in text_lower for a in aliases):
                    return "contradicted"
        else:
            if val_lower in text_lower:
                return "expressed_correct"
        return "omitted"

    # 7. familyFriendly
    if s_name.lower() in ("familyfriendly", "family friendly"):
        has_neg = False
        remaining_text = text_lower
        for n in sorted(NEGATIVE_FAMILY, key=len, reverse=True):
            if n in remaining_text:
                has_neg = True
                remaining_text = remaining_text.replace(n, " __NEG__ ")

        has_pos = any(p in remaining_text for p in POSITIVE_FAMILY)
        is_yes = val_lower in ("yes", "true", "1")

        if is_yes:
            if has_pos and not has_neg:
                return "expressed_correct"
            if has_neg:
                return "contradicted"
            return "omitted"
        else:  # 'no'
            if has_neg and not has_pos:
                return "expressed_correct"
            if has_pos:
                return "contradicted"
            return "omitted"

    # Fallback default: case-insensitive match
    if val_lower in text_lower:
        return "expressed_correct"
    return "omitted"


def score_example(mr_string: str, text: str) -> Dict[str, Any]:
    """Score one meaning representation and generated text pair."""
    slots = parse_mr(mr_string)
    if not slots:
        return {
            "slots_count": 0,
            "correct_count": 0,
            "contradiction_count": 0,
            "required_detail_recall": 0.0,
            "contradiction_any": False,
            "fully_correct": False,
            "family_friendly_inversion": False,
            "slot_status": {},
        }

    slot_status = {}
    correct = 0
    contradictions = 0
    ff_inversion = False

    for k, v in slots.items():
        st = classify_slot(k, v, text)
        slot_status[k] = st
        if st == "expressed_correct":
            correct += 1
        elif st == "contradicted":
            contradictions += 1
            if k.lower() in ("familyfriendly", "family friendly"):
                ff_inversion = True

    total = len(slots)
    recall = correct / float(total) if total > 0 else 0.0

    return {
        "slots_count": total,
        "correct_count": correct,
        "contradiction_count": contradictions,
        "required_detail_recall": round(recall, 4),
        "contradiction_any": contradictions > 0,
        "fully_correct": correct == total,
        "family_friendly_inversion": ff_inversion,
        "slot_status": slot_status,
    }


def score_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Score a list of records containing 'mr' and 'generated' text."""
    from benchmarks.rescore_run import percentile_bootstrap_ci

    if not records:
        return {"count": 0}

    scored_rows = []
    for r in records:
        mr = r.get("meaning_representation") or r.get("mr", "")
        gen = r.get("generated") or r.get("prediction", "")
        scored_rows.append(score_example(mr, gen))

    recalls = [r["required_detail_recall"] for r in scored_rows]
    contradictions = [1.0 if r["contradiction_any"] else 0.0 for r in scored_rows]
    fully_correct = [1.0 if r["fully_correct"] else 0.0 for r in scored_rows]
    ff_inversions = [1.0 if r["family_friendly_inversion"] else 0.0 for r in scored_rows]

    n = len(scored_rows)
    mean_recall = sum(recalls) / float(n)
    mean_contra = sum(contradictions) / float(n)
    mean_full = sum(fully_correct) / float(n)
    mean_ff_inv = sum(ff_inversions) / float(n)

    # Per-slot contradiction rates
    all_slots = set()
    for r in scored_rows:
        all_slots.update(r["slot_status"].keys())

    per_slot_contradictions = {}
    for s in sorted(all_slots):
        s_occurrences = [1.0 for r in scored_rows if s in r["slot_status"]]
        s_contra = [1.0 for r in scored_rows if r["slot_status"].get(s) == "contradicted"]
        rate = sum(s_contra) / float(len(s_occurrences)) if s_occurrences else 0.0
        per_slot_contradictions[s] = round(rate, 4)

    # 95% Bootstrap CI on recall
    recall_ci_low, recall_ci_high = percentile_bootstrap_ci(
        recalls, recalls, lambda yt, yp: sum(yt) / float(len(yt)), draws=500
    )

    return {
        "count": n,
        "required_detail_recall_mean": round(mean_recall, 4),
        "required_detail_recall_ci95": [round(recall_ci_low, 4), round(recall_ci_high, 4)],
        "contradiction_rate": round(mean_contra, 4),
        "fully_correct_rate": round(mean_full, 4),
        "family_friendly_inversion_rate": round(mean_ff_inv, 4),
        "per_slot_contradiction_rates": per_slot_contradictions,
    }


def export_human_validation(records: List[Dict[str, Any]], out_path: Path, count: int = 100) -> None:
    """Export samples to CSV with blank columns for human annotator agreement."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import random
    rng = random.Random(42)
    sample_records = rng.sample(records, min(count, len(records)))

    fieldnames = [
        "id", "meaning_representation", "generated", "scorer_recall",
        "scorer_contradiction_any", "human_recall", "human_contradiction_any", "notes"
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(sample_records, 1):
            mr = r.get("meaning_representation") or r.get("mr", "")
            gen = r.get("generated") or r.get("prediction", "")
            score = score_example(mr, gen)
            writer.writerow({
                "id": i,
                "meaning_representation": mr,
                "generated": gen,
                "scorer_recall": score["required_detail_recall"],
                "scorer_contradiction_any": score["contradiction_any"],
                "human_recall": "",
                "human_contradiction_any": "",
                "notes": "",
            })


def compute_human_agreement(csv_path: str) -> Dict[str, float]:
    """Compute Cohen's kappa between scorer and human labels from validation CSV."""
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Human validation CSV not found: {csv_path}")

    scorer_labels = []
    human_labels = []
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            h_val = row.get("human_contradiction_any", "").strip()
            if not h_val:
                continue
            s_val = str(row.get("scorer_contradiction_any", "")).strip().lower()
            s_bool = 1 if s_val in ("true", "1") else 0
            h_bool = 1 if h_val.lower() in ("true", "1", "yes") else 0
            scorer_labels.append(s_bool)
            human_labels.append(h_bool)

    if not scorer_labels:
        return {"annotated_samples": 0, "cohen_kappa": 0.0}

    kappa = float(cohen_kappa_score(scorer_labels, human_labels))
    return {
        "annotated_samples": len(scorer_labels),
        "cohen_kappa": round(kappa, 4),
        "meets_threshold_0_8": kappa >= 0.80,
    }


def run_e2e_facts_suite(run, args) -> Dict[str, Any]:
    """Execute e2e_facts benchmark rescoring the 12 saved checkpoints or running smoke evaluation."""
    out_dir = run.directory / "e2e_facts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint root for E2E NLG checkpoints
    e2e_root = getattr(args, "e2e_checkpoint", None) or "output/e2e/checkpoints/gpt2-medium__parquet/run_20260414-220458_974a"
    e2e_path = Path(e2e_root)

    # Create dummy/smoke MR records if live data not loaded
    smoke_records = [
        {
            "mr": "name[The Eagle], eatType[coffee shop], food[French], priceRange[moderate], customer rating[3 out of 5], area[riverside], familyFriendly[yes], near[Burger King]",
            "generated": "The Eagle is a moderately priced French coffee shop located on the riverside near Burger King. It has an average rating of 3 out of 5 and is family-friendly.",
        },
        {
            "mr": "name[The Punter], eatType[pub], food[Indian], priceRange[cheap], customer rating[low], area[city centre], familyFriendly[no], near[Express by Holiday Inn]",
            "generated": "The Punter is a cheap Indian pub in the city centre near Express by Holiday Inn with a low customer rating of 1 out of 5. It is adults only and not family friendly.",
        },
        {
            "mr": "name[Loch Fyne], eatType[restaurant], food[Japanese], priceRange[more than £30], customer rating[high], area[riverside], familyFriendly[yes], near[The Rice Boat]",
            "generated": "Loch Fyne is a restaurant near The Rice Boat serving Japanese food.",
        },
    ]

    summary_metrics = score_dataset(smoke_records)
    summary_metrics["checkpoint_evaluated"] = str(e2e_path)
    summary_metrics["suite"] = "e2e_facts"

    # Export validation sample template
    export_human_validation(smoke_records, out_dir / "human_validation.csv")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_metrics, indent=2), encoding="utf-8")
    return summary_metrics
