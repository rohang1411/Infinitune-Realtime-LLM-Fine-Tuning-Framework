"""Measured LoRA optimizer-step timing on the available PyTorch device."""

import json
from pathlib import Path
from statistics import median
import time


def run_precision_probe(output_path, *, model_name="distilgpt2", seq_length=128,
                        steps=3, requested_precision="fp32", compile_model=False,
                        progress=None):
    import psutil
    import torch
    from benchmarks.repro_eval import device_name, load_model
    from trainer import finish_optimizer_step
    from utils.precision_manager import PrecisionManager

    output_path = Path(output_path)
    if output_path.exists():
        records = [json.loads(x) for x in output_path.read_text(encoding="utf-8").splitlines()]
        if len(records) == steps:
            return _summary(records, output_path, requested_precision)
        raise RuntimeError(f"Incomplete precision trace: {output_path}")
    device = device_name()
    manager = PrecisionManager(requested_precision, device, enable_compile=compile_model)
    if manager.active_precision != requested_precision:
        return {"status": "unsupported", "requested_precision": requested_precision,
                "active_precision": manager.active_precision, "device": device}
    model, tokenizer, device = load_model(model_name, train=True)
    model = manager.compile_model_if_enabled(model)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    ids = torch.randint(0, tokenizer.vocab_size, (2, seq_length),
                        device=device, dtype=torch.long)
    attention_mask = torch.ones_like(ids)
    labels = ids.clone()
    labels[:, :seq_length//2] = -100
    process = psutil.Process()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with output_path.open("w", encoding="utf-8", buffering=1) as out:
        for index in range(steps):
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with manager.autocast_context():
                loss = model(input_ids=ids, attention_mask=attention_mask,
                             labels=labels).loss
            if manager.scaler:
                manager.scaler.scale(loss).backward()
            else:
                loss.backward()
            did_step, norm = finish_optimizer_step(model, optimizer, scheduler, manager)
            if device == "cuda":
                torch.cuda.synchronize()
            record = {"step": index+1, "duration_s": time.perf_counter()-start,
                      "loss": float(loss.detach().cpu()), "grad_norm": norm,
                      "did_step": did_step,
                      "rss_bytes": process.memory_info().rss,
                      "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
                      if device == "cuda" else None}
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Nonfinite loss at step {index+1}")
            records.append(record)
            out.write(json.dumps(record) + "\n")
            if progress:
                progress(index+1, steps)
    result = _summary(records, output_path, requested_precision)
    result.update({"device": device, "active_precision": manager.active_precision,
                   "compiled": manager.is_compiled,
                   "sequence_length": seq_length, "batch_size": 2,
                   "all_steps_applied": all(r["did_step"] for r in records)})
    return result


def _summary(records, path, requested_precision):
    # First step includes model/kernel warmup. Report both full and steady-state.
    steady = records[1:] if len(records) > 1 else records
    duration = sum(r["duration_s"] for r in steady)
    return {"requested_precision": requested_precision,
            "steps": len(records), "median_step_s": median(r["duration_s"] for r in steady),
            "max_rss_bytes": max(r["rss_bytes"] for r in records),
            "peak_cuda_allocated_bytes": max((r["cuda_peak_allocated_bytes"] or 0)
                                             for r in records),
            "trace_path": str(path), "warmup_steps_excluded": 1 if len(records)>1 else 0,
            "mean_loss": sum(r["loss"] for r in records)/len(records),
            "steady_steps_per_s": len(steady)/duration,
            "method": "synthetic fixed-token LoRA optimizer steps; no model-quality claim"}
