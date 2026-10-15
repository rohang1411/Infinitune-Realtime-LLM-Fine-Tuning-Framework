"""Unit tests validating Colab v2 environment configuration, scripts, and notebook structure."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_requirements_colab_structure():
    req_eval = (ROOT / "requirements-eval.txt").read_text(encoding="utf-8")
    req_colab = (ROOT / "requirements-colab.txt").read_text(encoding="utf-8")

    # Colab must NOT pin torch, numpy, or scikit-learn
    assert "torch==" not in req_colab
    assert "numpy==" not in req_colab
    assert "scikit-learn==" not in req_colab

    # Colab MUST contain core libraries
    assert "transformers==5.5.3" in req_colab
    assert "peft==0.18.1" in req_colab
    assert "datasets==4.5.0" in req_colab
    assert "kafka-python==2.0.2" in req_colab
    assert "accelerate==1.15.0" in req_colab


def test_start_kafka_script_content():
    script_path = ROOT / "scripts" / "colab" / "start_kafka.sh"
    assert script_path.exists(), "start_kafka.sh must exist"

    content = script_path.read_text(encoding="utf-8")
    assert content.startswith("#!/usr/bin/env bash")
    assert "3.9" in content
    assert "openjdk-17-jre-headless" in content
    assert "kafka-storage.sh" in content
    assert "kafka-server-start.sh" in content
    assert "kafka-topics.sh" in content
    assert "9092" in content


def test_colab_v2_notebook_validity():
    nb_path = ROOT / "notebooks" / "infinitune_colab_v2.ipynb"
    assert nb_path.exists(), "infinitune_colab_v2.ipynb must exist"

    nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb_data.get("nbformat") == 4
    cells = nb_data.get("cells", [])
    assert len(cells) >= 9, "Notebook must contain at least 9 distinct markdown and code sections"

    all_code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")

    # Cell 1: GPU check
    assert "nvidia-smi" in all_code
    assert "torch.cuda.is_available()" in all_code

    # Cell 2: Drive mount
    assert "google.colab" in all_code
    assert "drive.mount" in all_code

    # Cell 3: Git clone
    assert "git clone" in all_code

    # Cell 4: Requirements install & env check
    assert "requirements-colab.txt" in all_code
    assert "benchmarks.env_check" in all_code

    # Cell 5: Start Kafka
    assert "start_kafka.sh" in all_code

    # Cell 6: HuggingFace home & token
    assert "HF_HOME" in all_code

    # Cell 7 & 8: benchmarks.run and resume
    assert "benchmarks.run" in all_code
    assert "--kafka-bootstrap" in all_code

    # Cell 9: Zip export
    assert "make_archive" in all_code
