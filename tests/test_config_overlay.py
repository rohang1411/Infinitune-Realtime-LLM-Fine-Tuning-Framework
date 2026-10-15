"""
Unit tests for utils/config_overlay.py: deep_merge and load_config_with_overlay.
"""

import pytest
from pathlib import Path
from utils.config_overlay import deep_merge, load_config_with_overlay


def test_deep_merge_basic():
    base = {"a": 1, "b": {"x": 10, "y": 20}}
    overlay = {"b": {"y": 99, "z": 30}, "c": 3}
    merged = deep_merge(base, overlay)

    assert merged["a"] == 1
    assert merged["b"]["x"] == 10
    assert merged["b"]["y"] == 99
    assert merged["b"]["z"] == 30
    assert merged["c"] == 3

    # Base must not be mutated
    assert base["b"]["y"] == 20


def test_deep_merge_non_dict_override():
    base = {"model": {"name": "gpt2", "layers": [1, 2]}}
    overlay = {"model": {"name": "qwen", "layers": [3]}}
    merged = deep_merge(base, overlay)

    assert merged["model"]["name"] == "qwen"
    assert merged["model"]["layers"] == [3]


def test_load_config_with_overlay_no_overlay(tmp_path: Path):
    base_file = tmp_path / "base.yaml"
    base_file.write_text("model:\n  name: distilgpt2\n  precision: fp32\n", encoding="utf-8")

    cfg = load_config_with_overlay(str(base_file))
    assert cfg["model"]["name"] == "distilgpt2"
    assert cfg["model"]["precision"] == "fp32"


def test_load_config_with_overlay_active(tmp_path: Path):
    base_file = tmp_path / "base.yaml"
    base_file.write_text("model:\n  name: distilgpt2\nkafka:\n  bootstrap: localhost:9092\n", encoding="utf-8")

    overlay_file = tmp_path / "overlay.yaml"
    overlay_file.write_text("kafka:\n  client:\n    security_protocol: SASL_SSL\n  integrity: hmac\n", encoding="utf-8")

    cfg = load_config_with_overlay(str(base_file), str(overlay_file))
    assert cfg["model"]["name"] == "distilgpt2"
    assert cfg["kafka"]["bootstrap"] == "localhost:9092"
    assert cfg["kafka"]["client"]["security_protocol"] == "SASL_SSL"
    assert cfg["kafka"]["integrity"] == "hmac"


def test_load_config_with_overlay_missing_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config_with_overlay(str(tmp_path / "nonexistent.yaml"))

    base_file = tmp_path / "base.yaml"
    base_file.write_text("a: 1\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_config_with_overlay(str(base_file), str(tmp_path / "nonexistent_overlay.yaml"))
