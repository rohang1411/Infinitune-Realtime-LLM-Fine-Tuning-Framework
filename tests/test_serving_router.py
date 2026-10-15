"""
tests/test_serving_router.py
-----------------------------------------------------------------------------
Unit tests for DoubleBufferedAdapterRouter and ReadWriteLock concurrency.
"""

import time
import threading
from unittest.mock import MagicMock
import pytest
import torch

from utils.serving_router import (
    ReadWriteLock,
    DoubleBufferedAdapterRouter,
    nearest_rank_percentile,
    compute_adapter_state_digest,
)


def test_read_write_lock_concurrent_readers():
    lock = ReadWriteLock()
    active_readers = []
    max_concurrent = 0
    state_lock = threading.Lock()

    def reader_worker():
        nonlocal max_concurrent
        lock.acquire_read()
        with state_lock:
            active_readers.append(1)
            if len(active_readers) > max_concurrent:
                max_concurrent = len(active_readers)
        time.sleep(0.05)
        with state_lock:
            active_readers.pop()
        lock.release_read()

    threads = [threading.Thread(target=reader_worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Proves multiple readers executed concurrently
    assert max_concurrent >= 2


def test_read_write_lock_writer_exclusion():
    lock = ReadWriteLock()
    in_write = False
    read_while_write = False
    lock_checked = threading.Lock()

    def writer_worker():
        nonlocal in_write
        lock.acquire_write()
        with lock_checked:
            in_write = True
        time.sleep(0.05)
        with lock_checked:
            in_write = False
        lock.release_write()

    def reader_worker():
        nonlocal read_while_write
        time.sleep(0.01)  # start slightly after writer
        lock.acquire_read()
        with lock_checked:
            if in_write:
                read_while_write = True
        lock.release_read()

    t_writer = threading.Thread(target=writer_worker)
    t_reader = threading.Thread(target=reader_worker)

    t_writer.start()
    t_reader.start()

    t_writer.join()
    t_reader.join()

    # Reader must never observe writer in active state
    assert read_while_write is False


def test_double_buffered_router_hot_swap_telemetry():
    # Mock model and tokenizer
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.decode.return_value = "generated text"

    # Make model.generate return a mock tensor
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

    router = DoubleBufferedAdapterRouter(mock_model, mock_tokenizer, device="cpu")

    # Generate once
    cfg = {"max_new_tokens": 10, "do_sample": False}
    res = router.generate("prompt", cfg)
    assert res["response"] == "generated text"
    assert res["version"] == "v0_base"
    assert res["generated_tokens"] == 2

    # Swap weights to v1
    dummy_new_weights = {"layer.weight": torch.randn(4, 4)}
    swap_ms = router.swap_weights(dummy_new_weights, version_tag="v1_step_50")
    assert swap_ms >= 0.0

    # Generate again
    res2 = router.generate("prompt 2", cfg)
    assert res2["version"] == "v1_step_50"

    metrics = router.get_metrics()
    assert metrics["active_version"] == "v1_step_50"
    assert metrics["swap_count"] == 1
    assert metrics["requests_served"] == 2
    assert metrics["dropped_requests"] == 0
    assert metrics["p50_latency_ms"] >= 0.0


def test_concurrent_generation_during_swap():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2]])}
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.decode.return_value = "test"
    mock_model.generate.return_value = torch.tensor([[1, 2, 3]])

    router = DoubleBufferedAdapterRouter(mock_model, mock_tokenizer, device="cpu")
    cfg = {"max_new_tokens": 5, "do_sample": False}

    errors = []

    def request_loop(thread_id):
        for _ in range(15):
            try:
                res = router.generate(f"prompt {thread_id}", cfg)
                assert "response" in res
            except Exception as e:
                errors.append(e)
            time.sleep(0.005)

    def swap_loop():
        for i in range(5):
            time.sleep(0.01)
            router.swap_weights({"w": torch.randn(2, 2)}, version_tag=f"v_{i}")

    threads = [threading.Thread(target=request_loop, args=(i,)) for i in range(4)]
    swap_thread = threading.Thread(target=swap_loop)

    for t in threads:
        t.start()
    swap_thread.start()

    for t in threads:
        t.join()
    swap_thread.join()

    # Zero errors and zero dropped requests during live hot-swaps
    assert len(errors) == 0
    metrics = router.get_metrics()
    assert metrics["dropped_requests"] == 0
    assert metrics["requests_served"] == 60
    assert metrics["swap_count"] == 5


def test_nearest_rank_percentile_known_values():
    values = list(range(1, 101))
    assert nearest_rank_percentile(values, 0.50) == 50.0
    assert nearest_rank_percentile(values, 0.95) == 95.0
    assert nearest_rank_percentile(values, 0.99) == 99.0

    single = [42.0]
    for q in [0.0, 0.1, 0.5, 0.95, 0.99, 1.0]:
        assert nearest_rank_percentile(single, q) == 42.0

    assert nearest_rank_percentile([], 0.50) == 0.0


def test_lock_wait_includes_reader_drain():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    router = DoubleBufferedAdapterRouter(mock_model, mock_tokenizer, device="cpu")
    reader_acquired = threading.Event()

    def slow_reader():
        router.rw_lock.acquire_read()
        reader_acquired.set()
        try:
            time.sleep(0.20)  # Hold read lock for 200ms
        finally:
            router.rw_lock.release_read()

    t_reader = threading.Thread(target=slow_reader)
    t_reader.start()

    assert reader_acquired.wait(timeout=2.0)
    # Reader is currently holding the read lock; swap_weights must wait for reader to drain
    swap_total_ms = router.swap_weights({"w": torch.randn(2, 2)}, version_tag="v_drain_test")
    t_reader.join()

    assert router.last_swap_lock_wait_ms >= 150.0
    assert router.last_swap_lock_held_ms < router.last_swap_lock_wait_ms
    assert swap_total_ms == router.last_swap_lock_wait_ms + router.last_swap_lock_held_ms

    metrics = router.get_metrics()
    assert metrics["swap_lock_wait_ms_max"] >= 150.0
    assert metrics["swap_lock_held_ms_max"] < metrics["swap_lock_wait_ms_max"]
    assert "swap_prepare_ms_p50" in metrics


def test_compute_adapter_state_digest_order_invariance():
    t_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    t_b = torch.tensor([5.0, 6.0, 7.0])
    t_c = torch.tensor([[8.0]])

    d1 = {"layer_b": t_b, "layer_a": t_a, "layer_c": t_c}
    d2 = {"layer_a": t_a, "layer_c": t_c, "layer_b": t_b}
    d3 = {"layer_c": t_c, "layer_b": t_b, "layer_a": t_a}

    digest1 = compute_adapter_state_digest(d1)
    digest2 = compute_adapter_state_digest(d2)
    digest3 = compute_adapter_state_digest(d3)

    assert digest1 == digest2 == digest3
    assert len(digest1) == 64

    d_modified = {"layer_a": t_a + 0.01, "layer_b": t_b, "layer_c": t_c}
    assert compute_adapter_state_digest(d_modified) != digest1


def test_adapter_state_digest_peft_match_and_mutation():
    from transformers import GPT2Config, GPT2LMHeadModel
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

    config = GPT2Config(
        vocab_size=100,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
    )
    base_model = GPT2LMHeadModel(config)
    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["c_attn"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, lora_config)
    mock_tokenizer = MagicMock()

    router = DoubleBufferedAdapterRouter(peft_model, mock_tokenizer, device="cpu")

    # Snapshot with initial state
    initial_weights = {k: v.clone() for k, v in get_peft_model_state_dict(peft_model).items()}
    router.swap_weights(initial_weights, version_tag="step_000010", run_id="run_test", seq=10)

    # Digests should match
    live_sha = router.adapter_state_digest()
    assert live_sha == router.expected_state_sha256
    assert router.active_version == "step_000010"
    assert router.active_run_id == "run_test"
    assert router.active_seq == 10

    # Invalidate cache and mutate one live tensor in-place
    router._cached_adapter_digest = None
    mutated = False
    for name, param in peft_model.named_parameters():
        if "lora" in name:
            with torch.no_grad():
                param.data.add_(1.0)
            mutated = True
            break
    assert mutated, "Should have mutated at least one LoRA parameter"

    mutated_sha = router.adapter_state_digest()
    assert mutated_sha != router.expected_state_sha256


def test_adapter_endpoint_schema_and_match(monkeypatch):
    import inference
    from transformers import GPT2Config, GPT2LMHeadModel
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

    config = GPT2Config(
        vocab_size=100,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
    )
    base_model = GPT2LMHeadModel(config)
    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["c_attn"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, lora_config)
    mock_tokenizer = MagicMock()

    router = DoubleBufferedAdapterRouter(peft_model, mock_tokenizer, device="cpu")
    weights = {k: v.clone() for k, v in get_peft_model_state_dict(peft_model).items()}
    router.swap_weights(weights, version_tag="step_000020", run_id="run_endpoint", seq=20)

    monkeypatch.setattr(inference, "router_global", router)

    client = inference.app.test_client()
    resp = client.get("/adapter")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["active_version"] == "step_000020"
    assert data["run_id"] == "run_endpoint"
    assert data["seq"] == 20
    assert data["match"] is True
    assert data["adapter_state_sha256"] == data["expected_state_sha256"]

    # Mutate in place
    router._cached_adapter_digest = None
    for name, param in peft_model.named_parameters():
        if "lora" in name:
            with torch.no_grad():
                param.data.add_(2.0)
            break

    resp_mutated = client.get("/adapter")
    assert resp_mutated.status_code == 200
    data_mutated = resp_mutated.get_json()
    assert data_mutated["match"] is False
    assert data_mutated["adapter_state_sha256"] != data_mutated["expected_state_sha256"]


