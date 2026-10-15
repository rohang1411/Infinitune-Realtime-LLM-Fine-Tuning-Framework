"""E2E NLG fixed-MR evaluation with multi-reference BLEU and ROUGE-L."""

from collections import defaultdict
import json
from pathlib import Path
import random
import time

from benchmarks.repro_eval import MODEL_REVISIONS, file_sha256, load_model, read_rows


def prepare_e2e(count, seed, data_dir, offline=False):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"e2e_validation_mr_n{count}_s{seed}.jsonl"
    manifest_path = path.with_suffix(".manifest.json")
    if path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("sha256") == file_sha256(path)
                and manifest.get("seed") == seed and manifest.get("requested_mrs") == count):
            return path, manifest
        raise ValueError(f"E2E cache mismatch: {path}")
    if offline:
        raise FileNotFoundError(f"E2E cache missing: {path}")
    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info("GEM/e2e_nlg", revision="refs/convert/parquet").sha
    url = ("https://huggingface.co/datasets/GEM/e2e_nlg/resolve/"
           f"{revision}/default/validation/0000.parquet")
    source = load_dataset("parquet", data_files={"validation": url},
                          split="validation", streaming=True)
    grouped = defaultdict(set)
    for item in source:
        mr = str(item["meaning_representation"])
        grouped[mr].update(str(x) for x in (item.get("references") or [item["target"]])
                           if x)
    mrs = sorted(grouped)
    random.Random(seed).shuffle(mrs)
    selected = mrs[:min(count, len(mrs))]
    with path.open("w", encoding="utf-8") as out:
        for mr in selected:
            out.write(json.dumps({"id": mr, "mr": mr,
                                  "references": sorted(grouped[mr])}, ensure_ascii=False) + "\n")
    manifest = {"source": "GEM/e2e_nlg", "revision": revision, "url": url,
                "split": "validation", "sampling": "unique MR, seeded shuffle",
                "requested_mrs": count, "available_mrs": len(mrs),
                "selected_mrs": len(selected), "seed": seed,
                "sha256": file_sha256(path)}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path, manifest


def evaluate_e2e(rows, output_path, *, adapter, progress=None):
    # On Windows the NLTK/NumPy dependency chain can load an OpenMP runtime
    # before PyTorch and make c10.dll initialization fail. Import torch first.
    import torch
    import sacrebleu
    from rouge_score.rouge_scorer import RougeScorer
    from transformers import GenerationConfig

    output_path = Path(output_path)
    if output_path.exists():
        records = read_rows(output_path)
        if [r["id"] for r in records] != [r["id"] for r in rows]:
            raise RuntimeError(f"Incomplete E2E predictions: {output_path}")
    else:
        model, tokenizer, device = load_model("gpt2-medium", adapter)
        tokenizer.truncation_side = "left"
        records = []
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", buffering=1) as out:
            for index, row in enumerate(rows, 1):
                prompt = f"MR: {row['mr']}\nDescription:"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                                   max_length=128).to(device)
                with torch.inference_mode():
                    generated = model.generate(**inputs, generation_config=GenerationConfig(
                        max_new_tokens=60, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id))
                text = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True).strip()
                record = {"id": row["id"], "generation": text,
                          "references": row["references"]}
                records.append(record)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                if progress:
                    progress(index, len(rows))
    hypotheses = [r["generation"] for r in records]
    max_refs = max(len(r["references"]) for r in records)
    # SacreBLEU expects reference streams by index. Empty slots are ignored.
    references = [[r["references"][j] if j < len(r["references"]) else ""
                   for r in records] for j in range(max_refs)]
    metric = sacrebleu.metrics.BLEU()
    bleu = metric.corpus_score(hypotheses, references)
    scorer = RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = sum(max(scorer.score(ref, hyp)["rougeL"].fmeasure
                      for ref in row["references"])
                  for row, hyp in zip(records, hypotheses)) / len(records)
    from utils.eval_qualitative import StructuredSlotCoverageMetric
    proxy = StructuredSlotCoverageMetric(e2e_nlg_options={
        "slot_checkers": {"familyFriendly": "boolean_negation"},
        "track_per_slot": ["name", "eatType", "food", "priceRange",
                           "customer rating", "area", "familyFriendly", "near"],
    }).compute(hypotheses, [r["references"][0] for r in records],
               [r["id"] for r in records])
    return {"n_unique_mrs": len(records), "bleu": bleu.score,
            "base_model_revision": MODEL_REVISIONS["gpt2-medium"],
            "bleu_signature": str(metric.get_signature()),
            "rouge_l_f1_max_reference": rouge_l,
            "mean_references_per_mr": sum(len(r["references"]) for r in records)/len(records),
            "slot_coverage_proxy": proxy,
            "predictions_path": str(output_path), "predictions_sha256": file_sha256(output_path),
            "method": "greedy 60-token generation; MR-grouped multi-reference corpus BLEU"}
