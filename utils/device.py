"""
utils/device.py
─────────────────────────────────────────────────────────────────────────────
Unified compute device resolution hierarchy for InfiniTune.

Precedence Order:
  1. CLI parameter (--device) if specified and not 'auto'
  2. Environment variable INFINITUNE_DEVICE if specified and not 'auto'
  3. YAML configuration model.device if specified and not 'auto'
  4. Auto-detection: CUDA > Apple Silicon MPS > CPU fallback

Hardware Gate:
  'xpu' is returned ONLY if explicitly requested AND torch.xpu.is_available() is True.
  Otherwise raises RuntimeError("XPU requested but not available; see section 7 of plan 04").
"""

import os
import torch
from typing import Optional, Dict, Any


def resolve_device(config: Optional[Dict[str, Any]] = None, cli_device: Optional[str] = None) -> str:
    """Resolve target execution device following strict precedence hierarchy."""
    # 1. CLI parameter override
    if cli_device is not None and str(cli_device).lower().strip() not in ("", "auto", "none"):
        target = str(cli_device).lower().strip()
        return _validate_device(target)

    # 2. Environment variable
    env_dev = os.environ.get("INFINITUNE_DEVICE", "").lower().strip()
    if env_dev and env_dev not in ("auto", "none"):
        return _validate_device(env_dev)

    # 3. YAML configuration
    if config:
        cfg_dev = config.get("model", {}).get("device")
        if cfg_dev and str(cfg_dev).lower().strip() not in ("auto", "none"):
            return _validate_device(str(cfg_dev).lower().strip())

    # 4. Auto-detect
    return _autodetect_device()


def _validate_device(device_name: str) -> str:
    """Validate hardware availability for explicitly requested devices."""
    d = device_name.lower().strip()
    if d == "xpu":
        if getattr(torch, "xpu", None) and torch.xpu.is_available():
            return "xpu"
        raise RuntimeError("XPU requested but not available; see section 7 of plan 04")

    if d.startswith("cuda"):
        # Allow cuda or cuda:N
        return d

    if d == "mps":
        return "mps"

    if d == "cpu":
        return "cpu"

    return d


def _autodetect_device() -> str:
    """Auto-detect most capable available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and torch.backends.mps.is_available():
        return "mps"

    return "cpu"
