"""
utils/config_overlay.py
─────────────────────────────────────────────────────────────────────────────
Configuration overlay and deep-merge engine for InfiniTune.

Allows runtime environment customization (e.g. managed Kafka SASL_SSL overlays,
cloud storage directory paths, device overrides) on top of base benchmark configs
without modifying canonical version-controlled YAML files.
"""

import copy
import os
from typing import Any, Dict, Optional
import yaml


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively deep-merge overlay dictionary into a copy of base dictionary.
    Sub-dictionaries are merged recursively. Non-dictionary values in overlay
    override corresponding keys in base.
    """
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config_with_overlay(base_path: str, overlay_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load base configuration YAML file, optionally applying a deep-merged overlay YAML.

    Args:
        base_path: Path to the base YAML configuration.
        overlay_path: Optional path to an overlay YAML configuration.

    Returns:
        Merged configuration dictionary.

    Raises:
        FileNotFoundError: If base_path or overlay_path do not exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    if not os.path.isfile(base_path):
        raise FileNotFoundError(f"Base configuration file not found: {base_path}")

    with open(base_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}

    if not overlay_path:
        return base_cfg

    if not os.path.isfile(overlay_path):
        raise FileNotFoundError(f"Configuration overlay file not found: {overlay_path}")

    with open(overlay_path, "r", encoding="utf-8") as f:
        overlay_cfg = yaml.safe_load(f) or {}

    return deep_merge(base_cfg, overlay_cfg)
