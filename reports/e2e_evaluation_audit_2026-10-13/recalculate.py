"""Re-score retained E2E text; never convert aggregate coverage into factual F1.

Run from the repository root with .venv-eval/Scripts/python.exe. No model,
dataset download, or generation is needed. Existing benchmark files are read-only.
"""

import csv
import hashlib
from importlib import metadata
import json
from pathlib import Path
from statistics import mean

# Import order matches the working Windows benchmark runtime.
import torch  # noqa: F401
from rouge_score.rouge_scorer import RougeScorer

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "reports/benchmark_runs/verified_smoke_20261006/predictions/e2e.jsonl"
HISTORICAL = ROOT / "output/e2e/outputs/run5/metrics_clean.csv"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    records = [json.loads(line) for line in PREDICTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or len({row["id"] for row in records}) != len(records):
        raise ValueError("Expected nonempty predictions with unique meaning representations")
    scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    results = []
    for record in records:
        if not record.get("references"):
            raise ValueError(f"Missing references: {record['id']}")
        reference_scores = [scorer.score(reference, record["generation"])
                            for reference in record["references"]]
        metrics = {}
        for metric in scorer.rouge_types:
            # Select by highest F1, exactly as the existing ROUGE-L benchmark.
            # Precision and recall come from that SAME reference, rather than
            # selecting their maxima separately and manufacturing a triplet.
            selected = max(range(len(reference_scores)),
                           key=lambda index: reference_scores[index][metric].fmeasure)
            score = reference_scores[selected][metric]
            metrics[metric] = {"precision": score.precision, "recall": score.recall,
                               "f1": score.fmeasure, "reference_index": selected}
        results.append({"id": record["id"], "scores": metrics})
    summary = {metric: {field: mean(row["scores"][metric][field] for row in results)
                        for field in ("precision", "recall", "f1")}
               for metric in scorer.rouge_types}
    manifest = json.loads((PREDICTIONS.parent.parent / "run_manifest.json").read_text(encoding="utf-8"))
    prior = manifest["stages"]["e2e/generation"]["result"]
    if sha256(PREDICTIONS) != prior["predictions_sha256"]:
        raise ValueError("Saved predictions do not match their benchmark checksum")
    if abs(summary["rougeL"]["f1"] - prior["rouge_l_f1_max_reference"]) > 1e-12:
        raise ValueError("Recomputed ROUGE-L F1 does not match original manifest")
    with HISTORICAL.open(encoding="utf-8-sig", newline="") as stream:
        historical = list(csv.DictReader(stream))
    initial = next(row for row in historical if row["step"] == "0")
    final = next(row for row in historical if row["step"] == "final")
    inversion_key = "qual_slot_familyFriendly_inversion_rate"
    peak = max(historical, key=lambda row: float(row[inversion_key]))
    report = {
        "recalculation_date": "2026-10-13", "n_unique_mrs": len(records),
        "source_predictions": str(PREDICTIONS), "predictions_sha256": sha256(PREDICTIONS),
        "source_manifest_code_fingerprint": manifest["code_fingerprint"],
        "rouge_score_version": metadata.version("rouge-score"),
        "aggregation": "mean per-MR score; best-F1 reference selected independently per ROUGE variant",
        "tokenization": "rouge-score default normalization, Porter stemming enabled",
        "scores": summary, "per_example": results,
        "existing_sacrebleu": prior["bleu"],
        "limits": ["Four-example smoke diagnostic; not production-scale quality evidence",
                   "ROUGE precision/recall/F1 measure reference-text overlap, not factual correctness",
                   "Mean F1 is the mean of per-example F1, not F1 reconstructed from averaged P/R",
                   "No base-vs-adapter E2E generations are retained for a matched quality-improvement claim"],
        "historical_aggregates": {
            "source_csv": str(HISTORICAL), "csv_sha256": sha256(HISTORICAL),
            "initial_mean_coverage": float(initial["qual_slot_coverage_mean"]),
            "final_mean_coverage": float(final["qual_slot_coverage_mean"]),
            "final_perfect_coverage_rate": float(final["qual_perfect_coverage_rate"]),
            "initial_family_friendly_inversion_rate": float(initial[inversion_key]),
            "peak_family_friendly_inversion_step": peak["step"],
            "peak_family_friendly_inversion_rate": float(peak[inversion_key]),
            "final_family_friendly_inversion_rate": float(final[inversion_key]),
            "limits": ["Aggregate CSV with changing evaluation windows and no retained per-example predictions",
                       "Coverage is heuristic; boolean no checker treats omission as satisfied",
                       "Family-friendliness inversion is a narrow majority-vote proxy, not an all-facts error rate",
                       "Observed peak is not the untrained baseline; monitoring did not establish the cause of improvement"]}}
    output = OUT / "recalculated_scores.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"n": len(records), "scores": summary,
                      "original_rougeL_f1_reproduced": True, "saved": str(output)}, indent=2))


if __name__ == "__main__":
    main()
