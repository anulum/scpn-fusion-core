#!/usr/bin/env python
"""Fail when real-data roadmap progress regresses below pinned baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROGRESS = REPO_ROOT / "artifacts" / "real_data_roadmap_progress.json"
DEFAULT_BASELINE = REPO_ROOT / "tools" / "real_data_roadmap_baseline.json"
DEFAULT_SUMMARY = (
    REPO_ROOT / "artifacts" / "real_data_roadmap_non_regression_summary.json"
)


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object payload")
    return payload


def evaluate(*, progress: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    metric_rows = progress.get("metrics", [])
    if not isinstance(metric_rows, list):
        metric_rows = []
    observed = {
        str(row.get("metric")): int(row.get("current", 0))
        for row in metric_rows
        if isinstance(row, dict) and row.get("metric") is not None
    }

    baseline_metrics = baseline.get("metrics", {})
    if not isinstance(baseline_metrics, dict):
        raise ValueError("Baseline missing 'metrics' object")

    metric_checks: dict[str, dict[str, Any]] = {}
    regressions: list[str] = []
    for metric, baseline_value in baseline_metrics.items():
        key = str(metric)
        base = int(baseline_value)
        current = int(observed.get(key, 0))
        passes = current >= base
        metric_checks[key] = {
            "current": current,
            "baseline": base,
            "passes": passes,
        }
        if not passes:
            regressions.append(key)

    baseline_raw = bool(baseline.get("d3d_raw_ingestion_ready", False))
    current_raw = bool(progress.get("d3d_raw_ingestion_ready", False))
    d3d_raw_pass = (not baseline_raw) or current_raw
    if not d3d_raw_pass:
        regressions.append("d3d_raw_ingestion_ready")

    return {
        "roadmap_version": progress.get("roadmap_version"),
        "metric_checks": metric_checks,
        "d3d_raw_ingestion": {
            "current": current_raw,
            "baseline": baseline_raw,
            "passes": d3d_raw_pass,
        },
        "regressions": sorted(regressions),
        "overall_pass": len(regressions) == 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-json", default=str(DEFAULT_PROGRESS))
    parser.add_argument("--baseline-json", default=str(DEFAULT_BASELINE))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args(argv)

    summary = evaluate(
        progress=_load_json(_resolve(args.progress_json)),
        baseline=_load_json(_resolve(args.baseline_json)),
    )

    out_path = _resolve(args.summary_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        "Real-data roadmap non-regression: "
        f"pass={summary['overall_pass']} regressions={len(summary['regressions'])}"
    )
    if not bool(summary["overall_pass"]):
        print("Regressed metrics:", ", ".join(summary["regressions"]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
