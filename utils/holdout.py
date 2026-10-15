"""
utils/holdout.py
─────────────────────────────────────────────────────────────────────────────
Deterministic hash-based train/eval holdout partitioning for dataset streams.
"""

import hashlib
from typing import Dict, Any, Optional


def is_holdout_example(
    example: Dict[str, Any],
    holdout_cfg: Optional[Dict[str, Any]],
    seed: Optional[int] = None,
    input_col: str = "instruction",
) -> bool:
    """Check if an example belongs to the holdout partition using deterministic SHA-256 hash.

    Formula:
      int(sha256(f"{seed}:{instruction}\n{input}").hexdigest(), 16) % 10_000 < fraction * 10_000
    """
    if not holdout_cfg:
        return False

    method = holdout_cfg.get("method", "hash")
    if method != "hash":
        return False

    fraction = float(holdout_cfg.get("fraction", 0.02))
    cfg_seed = holdout_cfg.get("seed", 42) if seed is None else seed

    # Extract instruction / input context from example
    if "instruction" in example:
        instruction = str(example.get("instruction", "") or "")
        inp = str(example.get("input", "") or "")
    elif input_col in example:
        instruction = str(example.get(input_col, "") or "")
        inp = str(example.get("input", "") or "") if input_col != "input" else ""
    else:
        instruction = str(example.get("input", "") or "")
        inp = ""

    key_text = f"{cfg_seed}:{instruction}\n{inp}"
    h = int(hashlib.sha256(key_text.encode("utf-8")).hexdigest(), 16)
    threshold = int(fraction * 10_000)
    return (h % 10_000) < threshold
