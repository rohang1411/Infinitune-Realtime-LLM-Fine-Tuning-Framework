#!/usr/bin/env python3
"""InfiniTune environment and dependency census probe.

Collects system, hardware, PyTorch device capabilities, and package versions
into reports/env/<hostname>_<YYYYmmdd-HHMMSS>.json for provenance auditing.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import psutil


def get_cpu_model() -> str:
    system = platform.system()
    if system == "Windows":
        return platform.processor() or "Unknown Windows CPU"
    if system == "Darwin":
        try:
            res = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            return res.stdout.strip()
        except Exception as e:
            return f"Darwin (error reading brand_string: {e})"
    if system == "Linux":
        try:
            cpuinfo = Path("/proc/cpuinfo")
            if cpuinfo.exists():
                for line in cpuinfo.read_text(encoding="utf-8").splitlines():
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
        return platform.processor() or "Unknown Linux CPU"
    return platform.processor() or "Unknown CPU"


def get_java_version() -> str | None:
    java_bin = shutil.which("java")
    if not java_bin:
        return None
    try:
        proc = subprocess.run(
            [java_bin, "-version"],
            capture_output=True,
            text=True,
            check=False,
        )
        output = (proc.stderr or proc.stdout or "").strip()
        first_line = output.splitlines()[0] if output else "java found, empty output"
        return first_line
    except Exception as e:
        return f"Error executing java: {e}"


def collect_env(kafka_bootstrap: str | None = None) -> dict:
    # Basic system info
    data: dict = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "platform_machine": platform.machine(),
        "platform_node": platform.node(),
        "cpu_model": get_cpu_model(),
        "total_ram_bytes": psutil.virtual_memory().total,
    }

    # PyTorch and device capabilities
    try:
        import torch

        torch_info: dict = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": None,
            "cuda_capability": None,
            "mps_available": False,
            "xpu_available": False,
        }
        if torch.cuda.is_available():
            torch_info["cuda_device_name"] = torch.cuda.get_device_name(0)
            torch_info["cuda_capability"] = torch.cuda.get_device_capability(0)

        if hasattr(torch.backends, "mps") and hasattr(torch.backends.mps, "is_available"):
            torch_info["mps_available"] = torch.backends.mps.is_available()

        if hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available"):
            try:
                torch_info["xpu_available"] = torch.xpu.is_available()
            except Exception as e:
                torch_info["xpu_available"] = f"Error checking XPU: {e}"

        data["torch"] = torch_info
    except Exception as e:
        data["torch"] = {"error": str(e)}

    # Package versions
    packages = [
        "transformers",
        "peft",
        "datasets",
        "safetensors",
        "kafka-python",
        "numpy",
    ]
    pkg_versions = {}
    for pkg in packages:
        try:
            pkg_versions[pkg] = importlib.metadata.version(pkg)
        except Exception as e:
            pkg_versions[pkg] = f"Not installed or error: {e}"
    data["package_versions"] = pkg_versions

    # import kafka check
    try:
        import kafka

        data["kafka_import_success"] = True
    except Exception as e:
        data["kafka_import_success"] = str(e)

    # java version
    data["java_version"] = get_java_version()

    # Optional kafka bootstrap check
    if kafka_bootstrap:
        kafka_check: dict = {"bootstrap": kafka_bootstrap}
        try:
            from kafka import KafkaAdminClient

            admin = KafkaAdminClient(
                bootstrap_servers=kafka_bootstrap,
                request_timeout_ms=5000,
                api_version_auto_timeout_ms=5000,
            )
            try:
                topics = admin.list_topics()
                kafka_check["topics"] = list(topics)
            except Exception as te:
                kafka_check["topics_error"] = str(te)

            try:
                api_ver = admin._client.check_version()
                kafka_check["api_version"] = api_ver
            except Exception as ve:
                kafka_check["api_version_error"] = str(ve)

            admin.close()
            kafka_check["connected"] = True
        except Exception as e:
            kafka_check["connected"] = False
            kafka_check["error"] = str(e)
        data["kafka_bootstrap_check"] = kafka_check

    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="InfiniTune environment census")
    parser.add_argument(
        "--kafka-bootstrap",
        type=str,
        default=None,
        help="Optional HOST:PORT of a running Kafka broker to test connectivity.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/env"),
        help="Directory where environment JSON will be written (default: reports/env)",
    )
    args = parser.parse_args()

    env_data = collect_env(args.kafka_bootstrap)

    # Format output filename: <hostname>_<YYYYmmdd-HHMMSS>.json
    hostname = platform.node() or "unknown_host"
    timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{hostname}_{timestamp_str}.json"

    json_str = json.dumps(env_data, indent=2, ensure_ascii=False)
    out_path.write_text(json_str, encoding="utf-8")

    print(json_str)
    print(f"\nSaved environment census to: {out_path}")


if __name__ == "__main__":
    main()
