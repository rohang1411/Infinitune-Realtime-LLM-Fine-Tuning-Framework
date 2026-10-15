"""
tests/test_precision_manager.py
-----------------------------------------------------------------------------
Unit tests for PrecisionManager: mixed-precision resolution, backward execution,
and MFU utilization calculation.
"""

import pytest
import torch
import torch.nn as nn

from utils.precision_manager import PrecisionManager


def test_precision_resolution_cpu():
    # On CPU, all precision modes must resolve to fp32 for stability
    mgr_bf16 = PrecisionManager(precision="bf16", device="cpu")
    assert mgr_bf16.active_precision == "fp32"
    assert mgr_bf16.torch_dtype == torch.float32

    mgr_fp16 = PrecisionManager(precision="fp16", device="cpu")
    assert mgr_fp16.active_precision == "fp32"


def test_autocast_context_cpu():
    mgr = PrecisionManager(precision="fp32", device="cpu")
    with mgr.autocast_context():
        x = torch.randn(2, 4)
        y = x * 2.0
        assert y.shape == (2, 4)


def test_backward_and_step_execution():
    mgr = PrecisionManager(precision="fp32", device="cpu")
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    x = torch.randn(3, 4)
    target = torch.randn(3, 2)
    output = model(x)
    loss = nn.functional.mse_loss(output, target)

    telemetry = mgr.backward_and_step(
        loss=loss,
        optimizer=optimizer,
        max_grad_norm=1.0,
        model=model,
    )

    assert "grad_norm" in telemetry
    assert telemetry["overflow_occurred"] is False
    assert telemetry["grad_norm"] > 0.0


def test_mfu_calculation():
    # Model parameters = 1,000,000,000 (1B)
    # Tokens/sec = 10,000
    # Expected FLOPs/sec = 10,000 * 6 * 1e9 = 6e13 = 60 TFLOP/s
    # On a 300 TFLOP/s GPU, MFU should be (60 / 300) * 100 = 20.0%
    mfu = PrecisionManager.calculate_mfu(
        tokens_per_sec=10000.0,
        num_parameters=1000000000,
        custom_peak_tflops=300.0,
    )
    assert abs(mfu - 20.0) < 1e-2


def test_memory_telemetry_keys():
    telemetry = PrecisionManager.get_memory_telemetry(device="cpu")
    assert "vram_allocated_mb" in telemetry
    assert "vram_reserved_mb" in telemetry
    assert "vram_max_allocated_mb" in telemetry
