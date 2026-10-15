"""Fixed-data, per-example benchmark primitives for binary sentiment tasks.

The offline training routines exercise model quality and continual learning.
They do not measure Kafka delivery or live serving; those have separate suites.
"""

from collections import Counter
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time


DATASETS = {
    "imdb": ("imdb", "text", "label"),
    "yelp": ("yelp_polarity", "text", "label"),
    "amazon": ("amazon_polarity", "content", "label"),
}
LABELS = ("negative", "positive")
# Immutable Hub commits used by the checked-in benchmark configurations.
# Custom --model values require an explicit --model-revision commit.
MODEL_REVISIONS = {
    "distilgpt2": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
    "gpt2-medium": "6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_dataset(name, split, count, seed, data_dir, offline=False):
    """Cache a deterministic sample with source revision and content hash.

    Training and held-out data use the upstream train/test splits. A cache is
    reused only when its manifest and file digest match the request.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown task: {name}")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{name}_{split}_n{count}_s{seed}.jsonl"
    manifest_path = path.with_suffix(".manifest.json")
    if path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("count") == count and manifest.get("seed") == seed
                and manifest.get("split") == split and manifest.get("task") == name
                and manifest.get("sampling") == "stratified_binary_v2_unique_text"
                and manifest.get("sha256") == file_sha256(path)):
            return path, manifest
        if offline:
            raise ValueError(f"Corrupt or outdated offline dataset cache: {path}")
        suffix = f".invalid_{int(time.time())}"
        path.rename(path.with_name(path.name + suffix))
        manifest_path.rename(manifest_path.with_name(manifest_path.name + suffix))
        print(f"Archived outdated dataset cache before re-download: {path}", flush=True)
    elif path.exists() or manifest_path.exists():
        raise ValueError(f"Incomplete dataset cache pair: {path} and {manifest_path}")
    if offline:
        raise FileNotFoundError(f"Offline dataset cache is missing: {path}")

    from datasets import load_dataset
    from huggingface_hub import HfApi

    hub_id, text_col, label_col = DATASETS[name]
    # Record the immutable Hub commit and load that exact revision. If the Hub
    # is unavailable, fail rather than silently substituting a moving dataset.
    revision = HfApi().dataset_info(hub_id).sha
    stream = load_dataset(hub_id, split=split, revision=revision, streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=10000)
    # Balanced fixed sets avoid misleading small-profile scores when upstream
    # splits are label-sorted or shuffle buffers happen to favor one class.
    quotas = {0: count // 2, 1: count - count // 2}
    buckets = {0: [], 1: []}
    seen_ids = set()
    for item in stream:
        text = str(item[text_col]).strip()
        label = int(item[label_col])
        sample_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if (text and sample_id not in seen_ids and label in quotas
                and len(buckets[label]) < quotas[label]):
            buckets[label].append({"id": sample_id, "text": text, "label": label})
            seen_ids.add(sample_id)
        if all(len(buckets[label]) == quotas[label] for label in quotas):
            break
    rows = buckets[0] + buckets[1]
    random.Random(seed).shuffle(rows)
    if len(rows) != count:
        raise RuntimeError(f"{hub_id}/{split}: requested {count}, found {len(rows)}")
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
    manifest = {"task": name, "source": hub_id, "revision": revision,
                "split": split, "seed": seed, "count": count,
                "sampling": "stratified_binary_v2_unique_text",
                "label_counts": dict(Counter(row["label"] for row in rows)),
                "sha256": file_sha256(path)}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path, manifest


def read_rows(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def device_name():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name, adapter=None, train=False):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device_name()
    revision = (os.environ.get("INFINITUNE_MODEL_REVISION")
                if os.environ.get("INFINITUNE_MODEL_NAME") == model_name
                else MODEL_REVISIONS.get(model_name))
    if not revision:
        raise ValueError(f"No pinned revision for {model_name}; pass --model-revision")
    # fp32 avoids platform-specific fp16 behavior; record the actual device.
    base = AutoModelForCausalLM.from_pretrained(model_name, revision=revision,
                                               dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    tokenizer.pad_token = tokenizer.eos_token
    if adapter:
        model = PeftModel.from_pretrained(base, str(adapter), is_trainable=train)
    elif train:
        model = get_peft_model(base, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            target_modules=["c_attn", "c_proj"], task_type="CAUSAL_LM"))
    else:
        model = base
    model.to(device)
    model.train(train)
    return model, tokenizer, device


def train_phase(rows, output_dir, model_name, seed, *, adapter=None,
                max_steps=None, batch_size=4, accumulation=4,
                max_seq_length=256, replay_rows=None, progress=None):
    """Train a single LoRA phase; atomic directory rename marks completion.

    Interrupted phases restart from the previous completed adapter. This keeps
    run identity and training order deterministic without trusting a partial
    optimizer checkpoint.
    """
    import torch
    from trainer import pad_batch, tokenize_with_label_masking
    from jinja2 import Template

    output_dir = Path(output_dir)
    if (output_dir / "adapter_config.json").exists():
        return {"checkpoint": str(output_dir), "reused": True}
    if output_dir.exists():
        raise RuntimeError(f"Incomplete checkpoint directory requires inspection: {output_dir}")
    random.seed(seed)
    torch.manual_seed(seed)
    model, tokenizer, device = load_model(model_name, adapter, train=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    pad_id = tokenizer.pad_token_id
    prepared = list(rows)
    if replay_rows:
        prepared += list(replay_rows)
    # The caller controls whether the stream is shuffled. Preserve that order.
    prompt = Template("Review: {{ input }}\nSentiment:")
    response = Template(" {{ target }}")
    samples = [tokenize_with_label_masking(
        tokenizer, prompt.render(input=row["text"]),
        response.render(target=LABELS[row["label"]]), max_seq_length)
        for row in prepared]
    micro_batches = math.ceil(len(samples) / batch_size)
    planned_steps = math.ceil(micro_batches / accumulation)
    steps = min(planned_steps, max_steps) if max_steps else planned_steps
    if steps < 1:
        raise ValueError("Training phase has no optimizer steps")
    warmup = min(50, max(0, steps//10))
    def lr_factor(step):
        if warmup and step < warmup:
            return max(.01, (step+1)/warmup)
        fraction = (step-warmup)/max(1, steps-warmup)
        return .01 + .99 * .5 * (1+math.cos(math.pi*min(1, fraction)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    optimizer.zero_grad(set_to_none=True)
    began = time.perf_counter()
    completed = 0
    losses = []
    for step in range(steps):
        chunk = samples[step * batch_size * accumulation:
                        (step + 1) * batch_size * accumulation]
        if not chunk:
            break
        groups = [chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size)]
        for group in groups:
            batch = pad_batch(group, pad_id, device)
            loss = model(**batch).loss
            (loss / len(groups)).backward()
            losses.append(float(loss.detach().cpu()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        completed += 1
        if progress:
            progress(completed, steps)
    staging = output_dir.with_name(output_dir.name + ".staging")
    if staging.exists():
        raise RuntimeError(f"Stale incomplete training output: {staging}")
    staging.mkdir(parents=True)
    model.save_pretrained(staging, safe_serialization=True)
    tokenizer.save_pretrained(staging)
    (staging / "training_meta.json").write_text(json.dumps({
        "seed": seed, "device": device, "examples": len(prepared),
        "optimizer_steps": completed, "mean_train_loss": sum(losses) / len(losses),
        "duration_s": time.perf_counter() - began,
        "method": "offline sequential LoRA; no Kafka",
        "scheduler": {"type": "cosine_with_warmup", "warmup_steps": warmup,
                      "min_lr_ratio": .01, "planned_steps": steps},
    }, indent=2), encoding="utf-8")
    staging.replace(output_dir)
    del model
    return {"checkpoint": str(output_dir), "reused": False,
            "optimizer_steps": completed, "examples": len(prepared),
            "mean_train_loss": sum(losses) / len(losses),
            "duration_s": time.perf_counter() - began}


def _label_from_generation(text):
    import re
    words = re.findall(r"[a-z]+", text.lower())
    return words[0] if words and words[0] in LABELS else "other"


def score_predictions(records):
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
    from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support
    from statistics import NormalDist

    gold = [LABELS[r["label"]] for r in records]
    pred = [r["prediction"] for r in records]
    n = len(records)
    if n == 0:
        raise ValueError("No predictions")
    correct = sum(p == g for p, g in zip(pred, gold))
    p = correct / n
    z = NormalDist().inv_cdf(0.975)
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    precision, recall, _, support = precision_recall_fscore_support(
        gold, pred, labels=list(LABELS), zero_division=0)
    return {"n": n, "correct": correct, "accuracy": p,
            "accuracy_wilson_95": [center-half, center+half],
            "f1_macro_target_classes": f1_score(gold, pred, labels=list(LABELS),
                                                 average="macro", zero_division=0),
            "other_rate": pred.count("other") / n,
            "mcc": matthews_corrcoef(gold, pred),
            "kappa": cohen_kappa_score(gold, pred),
            "per_class": {name: {"precision": float(precision[i]),
                                  "recall": float(recall[i]),
                                  "support": int(support[i])}
                          for i, name in enumerate(LABELS)},
            "confusion": {g: {q: sum(x == g and y == q for x, y in zip(gold, pred))
                              for q in (*LABELS, "other")} for g in LABELS}}


def evaluate_rows(rows, output_path, model_name, *, adapter=None,
                  max_seq_length=256, progress=None):
    """Greedy generated-label evaluation; each decision is saved immediately."""
    import torch
    from transformers import GenerationConfig

    output_path = Path(output_path)
    if output_path.exists():
        saved = read_rows(output_path)
        if len(saved) == len(rows) and [r["id"] for r in saved] == [r["id"] for r in rows]:
            return score_predictions(saved)
        raise RuntimeError(f"Incomplete prediction file: {output_path}")
    model, tokenizer, device = load_model(model_name, adapter, train=False)
    tokenizer.truncation_side = "left"
    records = []
    began = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", buffering=1) as out:
        for index, row in enumerate(rows, 1):
            prompt = f"Review: {row['text']}\nSentiment:"
            # Left truncate long prompts while retaining the task instruction.
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=max_seq_length).to(device)
            with torch.inference_mode():
                generated = model.generate(**inputs, generation_config=GenerationConfig(
                    max_new_tokens=6, do_sample=False, pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id))
            answer = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True).strip()
            pred = _label_from_generation(answer)
            record = {"id": row["id"], "label": row["label"],
                      "prediction": pred, "raw_generation": answer}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            if progress:
                progress(index, len(rows))
    score = score_predictions(records)
    score.update({"duration_s": time.perf_counter() - began,
                  "model": model_name, "adapter": str(adapter) if adapter else None,
                  "device": device, "method": "greedy generated-label class match",
                  "predictions_sha256": file_sha256(output_path)})
    return score


def paired_bootstrap(a_records, b_records, seed=42, draws=2000):
    """Paired bootstrap CI of accuracy difference on identical sample IDs."""
    import numpy as np
    if [r["id"] for r in a_records] != [r["id"] for r in b_records]:
        raise ValueError("Paired comparison requires identical ordered sample IDs")
    a = np.array([r["prediction"] == LABELS[r["label"]] for r in a_records])
    b = np.array([r["prediction"] == LABELS[r["label"]] for r in b_records])
    diff = b.astype(float) - a.astype(float)
    rng = np.random.default_rng(seed)
    samples = np.array([diff[rng.integers(0, len(diff), len(diff))].mean()
                        for _ in range(draws)])
    return {"accuracy_delta": float(diff.mean()),
            "paired_bootstrap_95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
            "draws": draws, "seed": seed,
            "discordant_a_only": int(((a == 1) & (b == 0)).sum()),
            "discordant_b_only": int(((a == 0) & (b == 1)).sum())}


def evaluate_calibration(rows, output_path, model_name, *, adapter=None,
                         max_seq_length=256, progress=None):
    """Score conditional *single-token* label probabilities and calibration.

    This is a forced-choice probe, distinct from free-form generation accuracy.
    Both labels must be one token under the selected tokenizer, or the metric
    fails rather than silently comparing unequal tokenizations.
    """
    import torch
    from sklearn.metrics import roc_auc_score

    output_path = Path(output_path)
    if output_path.exists():
        records = read_rows(output_path)
        if len(records) != len(rows) or [r["id"] for r in records] != [r["id"] for r in rows]:
            raise RuntimeError(f"Incomplete calibration file: {output_path}")
    else:
        model, tokenizer, device = load_model(model_name, adapter)
        tokenizer.truncation_side = "left"
        token_ids = [tokenizer.encode(" " + label, add_special_tokens=False)
                     for label in LABELS]
        if any(len(ids) != 1 for ids in token_ids):
            raise ValueError(f"Forced-choice labels must each tokenize to one token: {token_ids}")
        records = []
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", buffering=1) as out:
            for index, row in enumerate(rows, 1):
                prompt = f"Review: {row['text']}\nSentiment:"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                                   max_length=max_seq_length).to(device)
                with torch.inference_mode():
                    logits = model(**inputs).logits[0, -1, [ids[0] for ids in token_ids]]
                    probability = torch.softmax(logits.float(), dim=-1)[1].item()
                record = {"id": row["id"], "label": row["label"],
                          "p_positive": probability,
                          "prediction": int(probability >= .5)}
                records.append(record)
                out.write(json.dumps(record) + "\n")
                if progress:
                    progress(index, len(rows))
    n = len(records)
    ece = 0.0
    for bin_index in range(10):
        low, high = bin_index / 10, (bin_index+1) / 10
        group = [r for r in records if low <= max(r["p_positive"], 1-r["p_positive"])
                 < high or (bin_index == 9 and max(r["p_positive"], 1-r["p_positive"]) == 1)]
        if group:
            confidence = sum(max(r["p_positive"], 1-r["p_positive"]) for r in group) / len(group)
            accuracy = sum(r["prediction"] == r["label"] for r in group) / len(group)
            ece += len(group)/n * abs(confidence-accuracy)
    probabilities = [r["p_positive"] for r in records]
    labels = [r["label"] for r in records]
    nll = -sum(math.log(max(p if y else 1-p, 1e-12))
               for p, y in zip(probabilities, labels)) / n
    result = {"n": n, "forced_choice_accuracy": sum(r["prediction"] == r["label"]
                                                   for r in records) / n,
              "ece_10_equal_width": ece,
              "brier_score": sum((p-y)**2 for p, y in zip(probabilities, labels)) / n,
              "conditional_label_nll": nll,
              "conditional_label_perplexity": math.exp(nll),
              "roc_auc": roc_auc_score(labels, probabilities) if len(set(labels)) == 2 else None,
              "probabilities_path": str(output_path),
              "probabilities_sha256": file_sha256(output_path),
              "method": "two-token restricted softmax; single-token labels verified"}
    return result
