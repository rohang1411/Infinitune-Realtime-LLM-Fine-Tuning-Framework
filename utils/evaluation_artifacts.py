"""
Shared evaluation artifact orchestrator for InfiniTune.

This module centralizes all artifact generation so trainer.py, evaluate.py,
Colab flows, and standalone utility usage all produce the same versioned,
non-overwriting bundle layout.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import time
import uuid
from typing import Any, Dict, List, Optional


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _safe_json_dump(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def _make_bundle_dir(run_root: str) -> str:
    base_dir = os.path.join(run_root, "evaluation_artifacts")
    os.makedirs(base_dir, exist_ok=True)
    bundle_name = f"artifact_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
    bundle_dir = os.path.join(base_dir, bundle_name)
    os.makedirs(bundle_dir, exist_ok=False)
    return bundle_dir


def _load_index(index_path: str) -> Dict[str, Any]:
    if not os.path.isfile(index_path):
        return {"version": 1, "updated_at": _ts(), "bundles": []}
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            bundles = payload.get("bundles", [])
            if isinstance(bundles, list):
                payload["version"] = payload.get("version", 1)
                payload["updated_at"] = payload.get("updated_at", _ts())
                payload["bundles"] = bundles
                return payload
        if isinstance(payload, list):
            return {"version": 1, "updated_at": _ts(), "bundles": payload}
    except Exception:
        pass
    return {"version": 1, "updated_at": _ts(), "bundles": []}


def _copy_file(src: str, dst: str) -> Optional[str]:
    if not src or not os.path.isfile(src):
        return None
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _read_rows(csv_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not csv_path or not os.path.isfile(csv_path):
        return rows
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _write_rows_csv(rows: List[Dict[str, Any]], csv_path: str) -> str:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return csv_path

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return csv_path


def generate_evaluation_artifacts(
    metrics_csv_path: str,
    run_root: str,
    config: Optional[Dict[str, Any]] = None,
    context: str = "standalone",
) -> Dict[str, Any]:
    """
    Generate a versioned evaluation artifact bundle under:

      <run_root>/evaluation_artifacts/artifact_<timestamp>_<uid>/

    Returns a manifest dict. Artifact generation is best-effort by default:
    individual steps record warnings/errors without aborting later steps.
    """
    from utils.plot_metrics import read_metrics_csv, render_plot_artifacts
    from utils.report_html import render_html_report

    run_root = os.path.abspath(run_root or os.path.dirname(os.path.abspath(metrics_csv_path)))
    bundle_dir = _make_bundle_dir(run_root)

    manifest: Dict[str, Any] = {
        "version": 1,
        "created_at": _ts(),
        "context": context,
        "run_root": run_root,
        "artifact_bundle": bundle_dir,
        "metrics_csv_path": os.path.abspath(metrics_csv_path) if metrics_csv_path else "",
        "steps": {},
        "generated_files": {
            "metrics": {},
            "dashboards": {},
            "insights": {"dark": [], "light": []},
            "plots": {"dark": [], "light": []},
            "report": "",
        },
        "warnings": [],
        "errors": [],
    }
    generation_log: List[Dict[str, Any]] = []

    def _record(level: str, step: str, message: str, detail: Optional[str] = None) -> None:
        entry = {
            "timestamp": _ts(),
            "level": level,
            "step": step,
            "message": message,
        }
        if detail:
            entry["detail"] = detail
        generation_log.append(entry)
        if level == "warning":
            manifest["warnings"].append(message if not detail else f"{message}: {detail}")
        if level == "error":
            manifest["errors"].append(message if not detail else f"{message}: {detail}")

    def _run_step(step: str, fn):
        try:
            result = fn()
            manifest["steps"][step] = {"ok": True}
            return result
        except Exception as exc:
            manifest["steps"][step] = {"ok": False, "error": str(exc)}
            _record("error", step, f"{step} failed", str(exc))
            return None

    metrics_dir = os.path.join(bundle_dir, "metrics")
    dashboards_dir = os.path.join(bundle_dir, "dashboards")
    insights_dir = os.path.join(bundle_dir, "insights")
    plots_dir = os.path.join(bundle_dir, "plots")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(dashboards_dir, exist_ok=True)
    os.makedirs(insights_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    def _normalize_metrics():
        nonlocal rows
        rows = read_metrics_csv(metrics_csv_path)
        source_copy = _copy_file(metrics_csv_path, os.path.join(metrics_dir, "source_metrics.csv"))
        resolved_csv = _write_rows_csv(rows, os.path.join(metrics_dir, "resolved_metrics.csv"))
        manifest["generated_files"]["metrics"] = {
            "source_metrics_csv": source_copy or "",
            "resolved_metrics_csv": resolved_csv,
        }
        if not rows:
            _record("warning", "metrics_normalization", "No rows found in metrics CSV")
        return rows

    _run_step("metrics_normalization", _normalize_metrics)

    def _render_plots():
        plot_payload = render_plot_artifacts(
            rows=rows,
            artifact_root=bundle_dir,
            config=config,
        )
        if isinstance(plot_payload, dict):
            for category in ("dashboards", "insights", "plots"):
                if category in plot_payload:
                    manifest["generated_files"][category] = plot_payload[category]
            if "presentation" in plot_payload:
                manifest["presentation"] = plot_payload["presentation"]
            for warning in plot_payload.get("warnings", []):
                _record("warning", "plot_render", warning)
            for error in plot_payload.get("errors", []):
                _record("error", "plot_render", error)
        return plot_payload

    _run_step("plot_render", _render_plots)

    def _render_report():
        report_path = render_html_report(
            rows=rows,
            artifact_root=bundle_dir,
            config=config,
            manifest=manifest,
        )
        if report_path:
            manifest["generated_files"]["report"] = report_path
        else:
            _record("warning", "report_render", "HTML report was not generated")
        return report_path

    _run_step("report_render", _render_report)

    def _write_metadata():
        manifest_path = os.path.join(bundle_dir, "manifest.json")
        generation_log_path = os.path.join(bundle_dir, "generation_log.json")
        _safe_json_dump(manifest_path, manifest)
        _safe_json_dump(generation_log_path, generation_log)
        index_path = os.path.join(run_root, "evaluation_artifacts", "index.json")
        index_payload = _load_index(index_path)
        index_payload["updated_at"] = _ts()
        index_payload["bundles"].append(
            {
                "created_at": manifest["created_at"],
                "context": context,
                "artifact_bundle": bundle_dir,
                "metrics_csv_path": manifest["metrics_csv_path"],
                "report": manifest["generated_files"].get("report", ""),
                "warnings": len(manifest["warnings"]),
                "errors": len(manifest["errors"]),
            }
        )
        _safe_json_dump(index_path, index_payload)
        return manifest_path

    _run_step("metadata_write", _write_metadata)
    return manifest
