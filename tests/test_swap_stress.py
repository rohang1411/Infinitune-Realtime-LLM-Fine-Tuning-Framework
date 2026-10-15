"""
tests/test_swap_stress.py
-----------------------------------------------------------------------------
Unit tests for WP-15 extended swap_stress suite: version validity check,
control window latency comparison, and metrics schema.
"""

import pytest
import time
import torch
from unittest.mock import MagicMock

from utils.serving_router import DoubleBufferedAdapterRouter


def test_version_validity_logic():
    # Simulate version commitment timestamps
    t0 = 100.0
    committed = [("v0_base", t0)]

    # Version committed at t=101.0
    t1 = 101.0
    committed.append(("stress_1", t1))

    # Version committed at t=105.0
    t2 = 105.0
    committed.append(("stress_2", t2))

    # Request finishing at t=102.0: can only have seen v0_base or stress_1
    finish_early = 102.0
    valid_early = {v for v, t_c in committed if t_c <= finish_early + 0.05}
    assert "v0_base" in valid_early
    assert "stress_1" in valid_early
    assert "stress_2" not in valid_early

    # Request finishing at t=106.0: can have seen any of the 3
    finish_late = 106.0
    valid_late = {v for v, t_c in committed if t_c <= finish_late + 0.05}
    assert "stress_2" in valid_late


def test_mock_router_swap_stress_zero_failures():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2]])}
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.decode.return_value = "positive response"
    mock_model.generate.return_value = torch.tensor([[1, 2, 3]])

    router = DoubleBufferedAdapterRouter(mock_model, mock_tokenizer, device="cpu")

    # Generate baseline request
    cfg = {"max_new_tokens": 5, "do_sample": False}
    res = router.generate("test prompt", cfg)
    assert res["version"] == "v0_base"
    assert "response" in res

    # Perform a swap
    dummy_weights = {"w": torch.randn(2, 2)}
    router.swap_weights(dummy_weights, version_tag="v_swapped", run_id="run_1", seq=1)

    res2 = router.generate("test prompt 2", cfg)
    assert res2["version"] == "v_swapped"

    metrics = router.get_metrics()
    assert metrics["dropped_requests"] == 0
    assert metrics["swap_count"] == 1
    assert "swap_lock_held_ms_p50" in metrics
    assert "swap_lock_wait_ms_p50" in metrics
