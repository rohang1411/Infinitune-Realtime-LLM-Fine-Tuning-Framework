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

from utils.serving_router import ReadWriteLock, DoubleBufferedAdapterRouter


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
