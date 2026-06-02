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
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLDS = REPO_ROOT / "tools" / "benchmark_regression_thresholds.json"
DEFAULT_SUMMARY = REPO_ROOT / "artifacts" / "benchmark_regression_guard_summary.json"


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
    if "max" not in metric and "min" not in metric:
        raise ValueError(f"{metric_path}: metric requires equals, min, or max")
    return row


def evaluate(thresholds: dict[str, Any]) -> dict[str, Any]:
    """Evaluate configured benchmark metrics against regression thresholds.

    Args:
        thresholds: Configuration containing report paths and metric bounds.

    Returns:
        Structured summary with per-report rows and aggregate pass state.
    """
    reports_cfg = thresholds.get("reports", [])
    if not isinstance(reports_cfg, list) or not reports_cfg:
        raise ValueError("benchmark regression thresholds must define at least one report")

    reports: list[dict[str, Any]] = []
    for report_cfg_any in reports_cfg:
        report_cfg = dict(report_cfg_any)
        report_id = str(report_cfg["id"])
        report_path = _resolve(str(report_cfg["path"]))
        payload = _load_json(report_path)
        metric_cfgs = report_cfg.get("metrics", [])
        if not isinstance(metric_cfgs, list) or not metric_cfgs:
            raise ValueError(f"{report_id}: report requires at least one metric")
        metrics = [_evaluate_metric(payload, dict(metric_cfg)) for metric_cfg in metric_cfgs]
        reports.append(
            {
                "id": report_id,
                "path": _display_path(report_path),
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
    summary = evaluate(_load_json(thresholds_path))

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
