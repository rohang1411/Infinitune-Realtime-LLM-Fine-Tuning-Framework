#!/usr/bin/env python3
"""Dataset census for InfiniTune benchmark and evaluation datasets.

Queries Hugging Face for dataset metadata, immutable commit SHAs, row counts,
and label/slot distributions. Writes results to reports/dataset_census.json.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys

from datasets import load_dataset
from huggingface_hub import HfApi


def get_commit_sha(api: HfApi, repo_id: str) -> str | None:
    try:
        info = api.dataset_info(repo_id)
        return info.sha
    except Exception as e:
        print(f"Warning: could not fetch commit sha for {repo_id}: {e}", file=sys.stderr)
        return None


def census_imdb(api: HfApi) -> dict:
    repo_id = "imdb"
    sha = get_commit_sha(api, repo_id)
    # stanfordnlp/imdb is the historical canonical repo id
    stanford_sha = get_commit_sha(api, "stanfordnlp/imdb")

    print("Loading imdb dataset...", flush=True)
    ds = load_dataset(repo_id)

    train_labels = Counter(ds["train"]["label"])
    test_labels = Counter(ds["test"]["label"])

    return {
        "dataset_name": repo_id,
        "resolved_repo_id": repo_id,
        "commit_sha": sha,
        "canonical_stanfordnlp_sha": stanford_sha,
        "train_rows": len(ds["train"]),
        "train_label_balance": dict(train_labels),
        "test_rows": len(ds["test"]),
        "test_label_balance": dict(test_labels),
    }


def census_continual_polarity(api: HfApi) -> dict:
    print("Loading yelp_polarity and amazon_polarity test splits...", flush=True)
    yelp_sha = get_commit_sha(api, "yelp_polarity")
    yelp_ds = load_dataset("yelp_polarity", split="test")

    amazon_sha = get_commit_sha(api, "amazon_polarity")
    amazon_ds = load_dataset("amazon_polarity", split="test")

    return {
        "yelp_polarity": {
            "commit_sha": yelp_sha,
            "test_rows": len(yelp_ds),
            "test_label_balance": dict(Counter(yelp_ds["label"])),
        },
        "amazon_polarity": {
            "commit_sha": amazon_sha,
            "test_rows": len(amazon_ds),
            "test_label_balance": dict(Counter(amazon_ds["label"])),
        },
    }


def census_e2e(api: HfApi) -> dict:
    print("Loading GEM/e2e_nlg parquet files...", flush=True)
    repo_id = "GEM/e2e_nlg"
    sha = get_commit_sha(api, repo_id)

    train_url = "https://huggingface.co/datasets/GEM/e2e_nlg/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    val_url = "https://huggingface.co/datasets/GEM/e2e_nlg/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"

    ds = load_dataset("parquet", data_files={"train": train_url, "validation": val_url})

    train_mrs = ds["train"]["meaning_representation"]
    val_mrs = ds["validation"]["meaning_representation"]

    unique_train_mrs = set(train_mrs)
    unique_val_mrs = set(val_mrs)

    return {
        "repo_id": repo_id,
        "commit_sha": sha,
        "urls": {"train": train_url, "validation": val_url},
        "train_rows": len(ds["train"]),
        "train_unique_meaning_representations": len(unique_train_mrs),
        "validation_rows": len(ds["validation"]),
        "validation_unique_meaning_representations": len(unique_val_mrs),
    }


def census_alpaca(api: HfApi) -> dict:
    print("Loading tatsu-lab/alpaca...", flush=True)
    repo_id = "tatsu-lab/alpaca"
    sha = get_commit_sha(api, repo_id)

    ds = load_dataset(repo_id, split="train")
    total_rows = len(ds)
    non_empty_input_count = sum(1 for row in ds if bool(row.get("input") and str(row["input"]).strip()))

    return {
        "repo_id": repo_id,
        "commit_sha": sha,
        "train_rows": total_rows,
        "non_empty_input_rows": non_empty_input_count,
        "non_empty_input_percentage": round((non_empty_input_count / total_rows) * 100, 2) if total_rows else 0.0,
    }


def census_gsm8k(api: HfApi) -> dict:
    print("Loading gsm8k main...", flush=True)
    repo_id = "gsm8k"
    sha = get_commit_sha(api, repo_id)

    ds = load_dataset(repo_id, "main")

    return {
        "repo_id": repo_id,
        "config_name": "main",
        "commit_sha": sha,
        "train_rows": len(ds["train"]),
        "test_rows": len(ds["test"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset Census for InfiniTune")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/dataset_census.json"),
        help="Target output JSON path (default: reports/dataset_census.json)",
    )
    args = parser.parse_args()

    api = HfApi()

    census_data = {
        "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "imdb": census_imdb(api),
        "continual_polarity": census_continual_polarity(api),
        "e2e_nlg": census_e2e(api),
        "alpaca": census_alpaca(api),
        "gsm8k": census_gsm8k(api),
    }

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(census_data, indent=2, ensure_ascii=False)
    out_path.write_text(json_text, encoding="utf-8")

    print("\n" + "=" * 60)
    print("DATASET CENSUS SUMMARY:")
    print("=" * 60)
    print(f"IMDb: Train={census_data['imdb']['train_rows']}, Test={census_data['imdb']['test_rows']}")
    print(f"Yelp Polarity: Test={census_data['continual_polarity']['yelp_polarity']['test_rows']}")
    print(f"Amazon Polarity: Test={census_data['continual_polarity']['amazon_polarity']['test_rows']}")
    print(
        f"E2E NLG: Train={census_data['e2e_nlg']['train_rows']} (Unique MRs={census_data['e2e_nlg']['train_unique_meaning_representations']}), "
        f"Val={census_data['e2e_nlg']['validation_rows']} (Unique MRs={census_data['e2e_nlg']['validation_unique_meaning_representations']})"
    )
    print(
        f"Alpaca: Train={census_data['alpaca']['train_rows']}, "
        f"Non-empty inputs={census_data['alpaca']['non_empty_input_rows']} ({census_data['alpaca']['non_empty_input_percentage']}%)"
    )
    print(f"GSM8K (main): Train={census_data['gsm8k']['train_rows']}, Test={census_data['gsm8k']['test_rows']}")
    print("=" * 60)
    print(f"Full census saved to: {out_path}\n")


if __name__ == "__main__":
    main()
