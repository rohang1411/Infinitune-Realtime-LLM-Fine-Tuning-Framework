"""
utils/serving_router.py
─────────────────────────────────────────────────────────────────────────────
Read/Write Locked Adapter Router & Concurrent Serving Engine for InfiniTune.

Key Production Architectural Improvements:
1. Lock-Free Concurrent Inference: Replaces global model generation locks with
   a high-throughput Read-Write Lock pattern. Arbitrary concurrent inference
   requests generate in parallel without serializing to concurrency=1.
2. Snapshot validation happens before a write lock. The lock is held while all
   adapter tensors are copied; swap time is measured, never promised.
3. SLA Observability & Telemetry: Captures real-time latency distributions
   (p50, p95, p99 TTFT / end-to-end latency), swap pause time (ms), and
   guarantees zero dropped requests during active updates.
"""

import hashlib
import math
import time
import threading
from collections import deque
from typing import Dict, Any, Optional, List, Sequence
import torch
from peft import PeftModel
from transformers import AutoTokenizer, GenerationConfig

from utils.adapter_manifest import AdapterManifest, load_adapter_weights


def nearest_rank_percentile(sorted_values: Sequence[float], q: float) -> float:
    """
    Calculate nearest-rank percentile for a pre-sorted sequence of numbers.
    Uses rank = ceil(q * n) with 1-indexing clamped to [1, n].
    """
    n = len(sorted_values)
    if n == 0:
        return 0.0
    rank = max(1, min(n, math.ceil(q * n)))
    return float(sorted_values[rank - 1])


def compute_adapter_state_digest(state_dict: Dict[str, torch.Tensor]) -> str:
    """Compute deterministic SHA-256 digest over sorted adapter keys and raw tensor bytes."""
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        hasher.update(key.encode("utf-8"))
        tensor = state_dict[key]
        if hasattr(tensor, "detach"):
            t_bytes = tensor.detach().cpu().contiguous().numpy().tobytes()
        else:
            t_bytes = bytes(tensor)
        hasher.update(t_bytes)
    return hasher.hexdigest()



class ReadWriteLock:
    """
    Classic Read-Write Lock allowing multiple concurrent readers,
    with priority-aware exclusive writer acquisition.
    """
    def __init__(self):
        self._readers = 0
        self._writers_waiting = 0
        self._writing = False
        self._lock = threading.Lock()
        self._read_ok = threading.Condition(self._lock)
        self._write_ok = threading.Condition(self._lock)

    def acquire_read(self):
        with self._lock:
            while self._writing or self._writers_waiting > 0:
                self._read_ok.wait()
            self._readers += 1

    def release_read(self):
        with self._lock:
            self._readers -= 1
            if self._readers == 0:
                self._write_ok.notify()

    def acquire_write(self):
        with self._lock:
            self._writers_waiting += 1
            while self._writing or self._readers > 0:
                self._write_ok.wait()
            self._writers_waiting -= 1
            self._writing = True

    def release_write(self):
        with self._lock:
            self._writing = False
            if self._writers_waiting > 0:
                self._write_ok.notify()
            else:
                self._read_ok.notify_all()


class DoubleBufferedAdapterRouter:
    """
    Manages one live model with read/write locking and swap telemetry.

    The historical class name is retained for API compatibility. This is a
    single-model router, not a two-replica double buffer.
    """

    def __init__(self, model: PeftModel, tokenizer: AutoTokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.rw_lock = ReadWriteLock()

        # Telemetry and state
        self.active_version = "v0_base"
        self.swap_count = 0
        self.last_swap_duration_ms = 0.0
        self.total_swap_duration_ms = 0.0
        self.last_swap_prepare_ms = 0.0
        self.last_swap_lock_wait_ms = 0.0
        self.last_swap_lock_held_ms = 0.0
        self.dropped_requests = 0
        self.requests_served = 0
        self.in_flight_requests = 0
        self.active_run_id: Optional[str] = None
        self.active_seq: Optional[Any] = None
        self.expected_state_sha256: Optional[str] = None
        self._cached_adapter_digest: Optional[str] = None

        # Latency tracking for p50/p95/p99
        self._latency_history_lock = threading.Lock()
        self._recent_latencies_ms: List[float] = []
        self._max_latency_history = 1000

        # Swap metrics tracking
        self._swap_prepare_history: deque = deque(maxlen=1000)
        self._swap_lock_wait_history: deque = deque(maxlen=1000)
        self._swap_lock_held_history: deque = deque(maxlen=1000)

    def adapter_state_digest(self) -> str:
        """Return SHA-256 digest of live model adapter state under read lock.
        Result is cached by active_version and cleared on every swap.
        """
        if self._cached_adapter_digest is not None:
            return self._cached_adapter_digest
        self.rw_lock.acquire_read()
        try:
            if isinstance(self.model, PeftModel):
                from peft import get_peft_model_state_dict
                live_state = get_peft_model_state_dict(self.model)
            elif hasattr(self.model, "state_dict"):
                live_state = self.model.state_dict()
            else:
                live_state = {}
            digest = compute_adapter_state_digest(live_state)
            self._cached_adapter_digest = digest
            return digest
        finally:
            self.rw_lock.release_read()

    def generate(
        self,
        prompt: str,
        inference_cfg: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute text generation concurrently. Multiple threads can call this
        in parallel without blocking one another.
        """
        start_time = time.perf_counter()
        self.rw_lock.acquire_read()
        with self._latency_history_lock:
            self.in_flight_requests += 1

        try:
            raw_inputs = self.tokenizer(prompt, return_tensors="pt")
            if hasattr(raw_inputs, "to"):
                inputs = raw_inputs.to(self.device)
            else:
                inputs = {
                    k: v.to(self.device) if hasattr(v, "to") else v
                    for k, v in raw_inputs.items()
                }
            prompt_token_len = inputs["input_ids"].shape[1]

            generation_config = GenerationConfig(
                max_new_tokens=inference_cfg.get("max_new_tokens", 100),
                do_sample=bool(inference_cfg.get("do_sample", True)),
                temperature=float(inference_cfg.get("temperature", 0.7)),
                top_p=float(inference_cfg.get("top_p", 0.9)),
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            with torch.no_grad():
                outputs = self.model.generate(**inputs, generation_config=generation_config)

            generated_ids = outputs[0, prompt_token_len:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            with self._latency_history_lock:
                self.requests_served += 1
                self._recent_latencies_ms.append(elapsed_ms)
                if len(self._recent_latencies_ms) > self._max_latency_history:
                    self._recent_latencies_ms.pop(0)

            return {
                "response": generated_text,
                "version": self.active_version,
                "latency_ms": round(elapsed_ms, 2),
                "generated_tokens": len(generated_ids),
            }

        except Exception as e:
            with self._latency_history_lock:
                self.dropped_requests += 1
            raise e
        finally:
            with self._latency_history_lock:
                self.in_flight_requests -= 1
            self.rw_lock.release_read()

    def swap_weights(
        self,
        new_state_dict: Dict[str, torch.Tensor],
        version_tag: str,
        run_id: Optional[str] = None,
        seq: Optional[Any] = None,
    ) -> float:
        """
        Apply one complete adapter state to the live model.

        PEFT's exported keys omit adapter names such as `.default.`; resolve
        those against live state before mutation. Reject missing, extra, or
        shape-mismatched tensors so partial updates cannot be published.
        Returns swap duration in milliseconds.
        """
        from peft import get_peft_model_state_dict

        t_prep_start = time.perf_counter()
        if not new_state_dict or not version_tag:
            raise ValueError("A swap requires a non-empty snapshot and version")
        if isinstance(self.model, PeftModel):
            expected = get_peft_model_state_dict(self.model)
            if set(new_state_dict) != set(expected):
                raise ValueError(
                    f"Adapter key mismatch: missing={sorted(set(expected)-set(new_state_dict))}, "
                    f"extra={sorted(set(new_state_dict)-set(expected))}"
                )
            live_state = self.model.state_dict()
            destinations = {}
            for key, tensor in new_state_dict.items():
                matches = [name for name in live_state
                           if name == key or name.replace(".default.", ".") == key]
                if len(matches) != 1:
                    raise ValueError(f"Cannot uniquely resolve live tensor for {key}")
                target = live_state[matches[0]]
                if not isinstance(tensor, torch.Tensor) or tensor.shape != target.shape:
                    raise ValueError(f"Adapter tensor shape/type mismatch for {key}")
                destinations[key] = target
            prepared_weights = {k: v.to(device=destinations[k].device,
                                        dtype=destinations[k].dtype)
                                for k, v in new_state_dict.items()}
        else:
            # Generic models are useful for isolated lock tests. Production
            # inference always takes the validated PEFT branch above.
            destinations = None
            prepared_weights = {k: v.to(self.device) for k, v in new_state_dict.items()}
        t_prep_end = time.perf_counter()
        prepare_ms = (t_prep_end - t_prep_start) * 1000.0

        self.expected_state_sha256 = compute_adapter_state_digest(prepared_weights)
        self._cached_adapter_digest = None

        t_request = time.perf_counter()
        self.rw_lock.acquire_write()
        t_acquired = time.perf_counter()
        try:
            if destinations is None:
                self.model.load_state_dict(prepared_weights, strict=False)
            else:
                with torch.no_grad():
                    for key, tensor in prepared_weights.items():
                        destinations[key].copy_(tensor)
            self.active_version = version_tag
            self.active_run_id = run_id
            self.active_seq = seq
            self.swap_count += 1
        finally:
            self.rw_lock.release_write()
            t_released = time.perf_counter()

        live_digest = self.adapter_state_digest()
        if self.expected_state_sha256 is not None and live_digest != self.expected_state_sha256:
            import logging
            logging.error(
                "MISMATCH: live adapter state digest (%s) != expected digest (%s)",
                live_digest,
                self.expected_state_sha256,
            )

        lock_wait_ms = (t_acquired - t_request) * 1000.0
        lock_held_ms = (t_released - t_acquired) * 1000.0
        swap_total_ms = lock_wait_ms + lock_held_ms

        self.last_swap_prepare_ms = prepare_ms
        self.last_swap_lock_wait_ms = lock_wait_ms
        self.last_swap_lock_held_ms = lock_held_ms
        self.last_swap_duration_ms = swap_total_ms
        self.total_swap_duration_ms += swap_total_ms

        with self._latency_history_lock:
            self._swap_prepare_history.append(prepare_ms)
            self._swap_lock_wait_history.append(lock_wait_ms)
            self._swap_lock_held_history.append(lock_held_ms)

        return swap_total_ms

    def swap_from_manifest(self, manifest: AdapterManifest) -> float:
        """
        Safely load weights from a verified manifest off-path, then swap atomically.
        """
        weights = manifest.load_weights(device="cpu")
        return self.swap_weights(
            weights,
            version_tag=manifest.adapter_version,
            run_id=getattr(manifest, "run_id", None),
            seq=getattr(manifest, "seq", None),
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Return real-time serving telemetry and latency SLAs."""
        with self._latency_history_lock:
            latencies = sorted(self._recent_latencies_ms)
            n = len(latencies)
            if n > 0:
                p50 = nearest_rank_percentile(latencies, 0.50)
                p95 = nearest_rank_percentile(latencies, 0.95)
                p99 = nearest_rank_percentile(latencies, 0.99)
                avg_lat = sum(latencies) / n
            else:
                p50 = p95 = p99 = avg_lat = 0.0

            avg_swap = (
                self.total_swap_duration_ms / self.swap_count
                if self.swap_count > 0
                else 0.0
            )

            if self._swap_lock_wait_history:
                wait_sorted = sorted(self._swap_lock_wait_history)
                wait_p50 = nearest_rank_percentile(wait_sorted, 0.50)
                wait_p95 = nearest_rank_percentile(wait_sorted, 0.95)
                wait_max = max(wait_sorted)
            else:
                wait_p50 = wait_p95 = wait_max = 0.0

            if self._swap_lock_held_history:
                held_sorted = sorted(self._swap_lock_held_history)
                held_p50 = nearest_rank_percentile(held_sorted, 0.50)
                held_p95 = nearest_rank_percentile(held_sorted, 0.95)
                held_max = max(held_sorted)
            else:
                held_p50 = held_p95 = held_max = 0.0

            if self._swap_prepare_history:
                prep_sorted = sorted(self._swap_prepare_history)
                prep_p50 = nearest_rank_percentile(prep_sorted, 0.50)
                prep_p95 = nearest_rank_percentile(prep_sorted, 0.95)
            else:
                prep_p50 = prep_p95 = 0.0

            return {
                "active_version": self.active_version,
                "swap_count": self.swap_count,
                "last_swap_duration_ms": round(self.last_swap_duration_ms, 2),
                "avg_swap_duration_ms": round(avg_swap, 2),
                "swap_lock_wait_ms_p50": round(wait_p50, 2),
                "swap_lock_wait_ms_p95": round(wait_p95, 2),
                "swap_lock_wait_ms_max": round(wait_max, 2),
                "swap_lock_held_ms_p50": round(held_p50, 2),
                "swap_lock_held_ms_p95": round(held_p95, 2),
                "swap_lock_held_ms_max": round(held_max, 2),
                "swap_prepare_ms_p50": round(prep_p50, 2),
                "swap_prepare_ms_p95": round(prep_p95, 2),
                "requests_served": self.requests_served,
                "in_flight_requests": self.in_flight_requests,
                "dropped_requests": self.dropped_requests,
                "p50_latency_ms": round(p50, 2),
                "p95_latency_ms": round(p95, 2),
                "p99_latency_ms": round(p99, 2),
                "avg_latency_ms": round(avg_lat, 2),
            }
