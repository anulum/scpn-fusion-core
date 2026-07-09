#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Fail-closed benchmark regression guard for tracked CI artefacts."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLDS = REPO_ROOT / "tools" / "benchmark_regression_thresholds.json"
DEFAULT_SUMMARY = REPO_ROOT / "artifacts" / "benchmark_regression_guard_summary.json"
EXPECTED_THRESHOLD_SCHEMA = "benchmark-regression-thresholds.v2"


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required benchmark artefact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object payload")
    return payload


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return path.as_posix()


def _lookup(payload: Any, metric_path: str) -> Any:
    current = payload
    for part in metric_path.split("."):
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(metric_path)
            current = current[part]
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError) as exc:
                raise KeyError(metric_path) from exc
        else:
            raise KeyError(metric_path)
    return current


def _finite_number(value: Any, metric_path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{metric_path}: expected numeric metric, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{metric_path}: metric is not finite")
    return number


def _require_non_empty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: expected non-empty string")
    return value


def _validate_metric_config(metric: dict[str, Any], *, report_id: str, index: int) -> None:
    metric_path = _require_non_empty_string(
        metric.get("path"),
        label=f"{report_id}.metrics[{index}].path",
    )
    has_equals = "equals" in metric
    has_min = "min" in metric
    has_max = "max" in metric
    if has_equals and (has_min or has_max):
        raise ValueError(f"{metric_path}: equals cannot be combined with min/max")
    if not has_equals and not has_min and not has_max:
        raise ValueError(f"{metric_path}: metric requires equals, min, or max")
    if has_min:
        _finite_number(metric["min"], f"{metric_path}.min")
    if has_max:
        _finite_number(metric["max"], f"{metric_path}.max")
    if has_min and has_max and float(metric["min"]) > float(metric["max"]):
        raise ValueError(f"{metric_path}: min cannot be greater than max")


def _validate_report_config(report_cfg: dict[str, Any], *, index: int) -> None:
    report_id = _require_non_empty_string(report_cfg.get("id"), label=f"reports[{index}].id")
    _require_non_empty_string(report_cfg.get("path"), label=f"{report_id}.path")
    if "expected_schema" in report_cfg:
        _require_non_empty_string(
            report_cfg["expected_schema"], label=f"{report_id}.expected_schema"
        )
    if "expected_benchmark_id" in report_cfg:
        _require_non_empty_string(
            report_cfg["expected_benchmark_id"],
            label=f"{report_id}.expected_benchmark_id",
        )
    metric_cfgs = report_cfg.get("metrics")
    if not isinstance(metric_cfgs, list) or not metric_cfgs:
        raise ValueError(f"{report_id}: report requires at least one metric")
    seen_paths: set[str] = set()
    for metric_index, metric_any in enumerate(metric_cfgs):
        if not isinstance(metric_any, dict):
            raise ValueError(f"{report_id}.metrics[{metric_index}]: expected object")
        metric = dict(metric_any)
        _validate_metric_config(metric, report_id=report_id, index=metric_index)
        metric_path = str(metric["path"])
        if metric_path in seen_paths:
            raise ValueError(f"{report_id}: duplicate metric path {metric_path}")
        seen_paths.add(metric_path)
    if "max_age_seconds" in report_cfg:
        max_age = _finite_number(report_cfg["max_age_seconds"], f"{report_id}.max_age_seconds")
        if max_age <= 0.0:
            raise ValueError(f"{report_id}.max_age_seconds must be > 0")


def _validate_thresholds(thresholds: dict[str, Any]) -> list[dict[str, Any]]:
    schema = thresholds.get("schema")
    if schema != EXPECTED_THRESHOLD_SCHEMA:
        raise ValueError(
            "benchmark regression thresholds schema mismatch: "
            f"expected {EXPECTED_THRESHOLD_SCHEMA!r}, got {schema!r}"
        )
    reports_cfg = thresholds.get("reports", [])
    if not isinstance(reports_cfg, list) or not reports_cfg:
        raise ValueError("benchmark regression thresholds must define at least one report")
    seen_ids: set[str] = set()
    reports: list[dict[str, Any]] = []
    for index, report_cfg_any in enumerate(reports_cfg):
        if not isinstance(report_cfg_any, dict):
            raise ValueError(f"reports[{index}]: expected object")
        report_cfg = dict(report_cfg_any)
        _validate_report_config(report_cfg, index=index)
        report_id = str(report_cfg["id"])
        if report_id in seen_ids:
            raise ValueError(f"duplicate benchmark report id: {report_id}")
        seen_ids.add(report_id)
        reports.append(report_cfg)
    return reports


def _freshness_row(report_path: Path, max_age_seconds: float | None) -> dict[str, Any] | None:
    if max_age_seconds is None:
        return None
    age_seconds = max(0.0, time.time() - report_path.stat().st_mtime)
    passes = age_seconds <= max_age_seconds
    return {
        "path": "__report_age_seconds__",
        "description": "benchmark report freshness",
        "value": age_seconds,
        "max": max_age_seconds,
        "passes": passes,
        "failure_reason": None if passes else f"{age_seconds:.12g} > max {max_age_seconds:.12g}",
    }


def _identity_rows(payload: dict[str, Any], report_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "expected_schema" in report_cfg:
        observed = payload.get("schema")
        expected = str(report_cfg["expected_schema"])
        passes = observed == expected
        rows.append(
            {
                "path": "__report_schema__",
                "description": "benchmark report schema identity",
                "value": observed,
                "expected": expected,
                "passes": passes,
                "failure_reason": None if passes else f"expected {expected!r}, got {observed!r}",
            }
        )
    if "expected_benchmark_id" in report_cfg:
        observed = payload.get("benchmark_id")
        expected = str(report_cfg["expected_benchmark_id"])
        passes = observed == expected
        rows.append(
            {
                "path": "__benchmark_id__",
                "description": "benchmark report id identity",
                "value": observed,
                "expected": expected,
                "passes": passes,
                "failure_reason": None if passes else f"expected {expected!r}, got {observed!r}",
            }
        )
    return rows


def _evaluate_metric(payload: dict[str, Any], metric: dict[str, Any]) -> dict[str, Any]:
    metric_path = str(metric["path"])
    value = _lookup(payload, metric_path)
    row: dict[str, Any] = {
        "path": metric_path,
        "description": str(metric.get("description", metric_path)),
        "value": value,
        "passes": True,
        "failure_reason": None,
    }

    if "equals" in metric:
        expected = metric["equals"]
        row["expected"] = expected
        row["passes"] = bool(value == expected)
        if not row["passes"]:
            row["failure_reason"] = f"expected {expected!r}, got {value!r}"
        return row

    number = _finite_number(value, metric_path)
    if "max" in metric:
        limit = _finite_number(metric["max"], f"{metric_path}.max")
        row["max"] = limit
        if number > limit:
            row["passes"] = False
            row["failure_reason"] = f"{number:.12g} > max {limit:.12g}"
    if "min" in metric:
        limit = _finite_number(metric["min"], f"{metric_path}.min")
        row["min"] = limit
        if number < limit:
            row["passes"] = False
            row["failure_reason"] = f"{number:.12g} < min {limit:.12g}"
    return row


def evaluate(thresholds: dict[str, Any]) -> dict[str, Any]:
    """Evaluate configured benchmark metrics against regression thresholds.

    Args:
        thresholds: Configuration containing report paths and metric bounds.

    Returns:
        Structured summary with per-report rows and aggregate pass state.
    """
    reports: list[dict[str, Any]] = []
    for report_cfg in _validate_thresholds(thresholds):
        report_id = str(report_cfg["id"])
        report_path = _resolve(str(report_cfg["path"]))
        payload = _load_json(report_path)
        metric_cfgs = report_cfg["metrics"]
        metrics = _identity_rows(payload, report_cfg)
        metrics.extend(_evaluate_metric(payload, dict(metric_cfg)) for metric_cfg in metric_cfgs)
        freshness = _freshness_row(
            report_path,
            float(report_cfg["max_age_seconds"]) if "max_age_seconds" in report_cfg else None,
        )
        if freshness is not None:
            metrics.append(freshness)
        reports.append(
            {
                "id": report_id,
                "path": _display_path(report_path),
                "max_age_seconds": report_cfg.get("max_age_seconds"),
                "expected_schema": report_cfg.get("expected_schema"),
                "expected_benchmark_id": report_cfg.get("expected_benchmark_id"),
                "metric_count": len(metrics),
                "failed_metric_count": sum(1 for row in metrics if not bool(row["passes"])),
                "passes": all(bool(row["passes"]) for row in metrics),
                "metrics": metrics,
            }
        )

    return {
        "schema": "benchmark-regression-guard.v1",
        "report_count": len(reports),
        "failed_report_count": sum(1 for report in reports if not bool(report["passes"])),
        "metric_count": sum(int(report["metric_count"]) for report in reports),
        "failed_metric_count": sum(int(report["failed_metric_count"]) for report in reports),
        "reports": reports,
        "overall_pass": all(bool(report["passes"]) for report in reports),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark regression guard.

    Args:
        argv: Optional CLI argument vector.

    Returns:
        ``0`` when all configured regression gates pass, otherwise ``1``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args(argv)

    thresholds_path = _resolve(args.thresholds)
    summary_path = _resolve(args.summary_json)
    try:
        summary = evaluate(_load_json(thresholds_path))
    except (FileNotFoundError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"Benchmark regression guard failed: {exc}")
        return 1

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "Benchmark regression guard: "
        f"reports={summary['report_count']} "
        f"metrics={summary['metric_count']} "
        f"failed_metrics={summary['failed_metric_count']}"
    )
    if not bool(summary["overall_pass"]):
        print("Benchmark regression guard failed.")
        return 1
    print("Benchmark regression guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
