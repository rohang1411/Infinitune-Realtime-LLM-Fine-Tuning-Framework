import os
import pytest
import torch
from utils.device import resolve_device
from utils.precision_manager import PrecisionManager


def test_resolve_device_cli_precedence(monkeypatch):
    monkeypatch.setenv("INFINITUNE_DEVICE", "cpu")
    config = {"model": {"device": "cpu"}}

    # Explicit CLI override wins
    assert resolve_device(config=config, cli_device="cuda") == "cuda"
    assert resolve_device(config=config, cli_device="cpu") == "cpu"


def test_resolve_device_env_precedence(monkeypatch):
    monkeypatch.setenv("INFINITUNE_DEVICE", "cuda")
    config = {"model": {"device": "cpu"}}

    # Env override wins over config when CLI is None or auto
    assert resolve_device(config=config, cli_device=None) == "cuda"
    assert resolve_device(config=config, cli_device="auto") == "cuda"


def test_resolve_device_config_precedence(monkeypatch):
    monkeypatch.delenv("INFINITUNE_DEVICE", raising=False)
    config = {"model": {"device": "mps"}}

    # Config wins over auto-detect when CLI and env are absent/auto
    assert resolve_device(config=config, cli_device=None) == "mps"
    assert resolve_device(config=config, cli_device="auto") == "mps"


def test_resolve_device_autodetect_cpu(monkeypatch):
    monkeypatch.delenv("INFINITUNE_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert resolve_device(config=None, cli_device="auto") == "cpu"
    assert resolve_device(config={}, cli_device=None) == "cpu"


def test_resolve_device_autodetect_cuda(monkeypatch):
    monkeypatch.delenv("INFINITUNE_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_device(config=None, cli_device=None) == "cuda"


def test_resolve_device_autodetect_mps(monkeypatch):
    monkeypatch.delenv("INFINITUNE_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert resolve_device(config=None, cli_device=None) == "mps"


def test_resolve_device_xpu_unavailable_raises():
    if not (getattr(torch, "xpu", None) and torch.xpu.is_available()):
        with pytest.raises(RuntimeError, match="XPU requested but not available"):
            resolve_device(cli_device="xpu")


def test_resolve_device_xpu_mocked_available(monkeypatch):
    class MockXPU:
        @staticmethod
        def is_available():
            return True

    monkeypatch.setattr(torch, "xpu", MockXPU, raising=False)
    assert resolve_device(cli_device="xpu") == "xpu"


def test_precision_manager_mps_half_flag():
    # Without allow_half_on_mps, MPS falls back to fp32
    pm_refused = PrecisionManager(precision="fp16", device="mps", allow_half_on_mps=False)
    assert pm_refused.active_precision == "fp32"
    assert pm_refused.torch_dtype == torch.float32

    # With allow_half_on_mps=True, fp16 is permitted
    pm_allowed = PrecisionManager(precision="fp16", device="mps", allow_half_on_mps=True)
    assert pm_allowed.active_precision == "fp16"
    assert pm_allowed.torch_dtype == torch.float16


def test_precision_manager_cpu_always_fp32():
    pm_cpu_fp16 = PrecisionManager(precision="fp16", device="cpu", allow_half_on_mps=True)
    assert pm_cpu_fp16.active_precision == "fp32"

    pm_cpu_bf16 = PrecisionManager(precision="bf16", device="cpu", allow_half_on_mps=True)
    assert pm_cpu_bf16.active_precision == "fp32"
