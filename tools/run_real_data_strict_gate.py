#!/usr/bin/env python
"""Run strict real-data validation gate focused on DIII-D raw-ingestion readiness."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIMEOUT_SECONDS = 1800.0
_RAW_SOURCE_TYPE_TOKENS = ("raw", "mdsplus", "omas")


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


def _has_raw_source_type(progress_payload: dict[str, Any]) -> bool:
    explicit = progress_payload.get("d3d_raw_source_type_present")
    if isinstance(explicit, bool):
        return explicit

    source_types_raw = progress_payload.get("d3d_disruption_source_types", [])
    if not isinstance(source_types_raw, list):
        return False
    for value in source_types_raw:
        lowered = str(value).strip().lower()
        if lowered and any(token in lowered for token in _RAW_SOURCE_TYPE_TOKENS):
            return True
    return False


def _run_step(name: str, cmd: list[str], *, timeout_seconds: float) -> None:
    rendered = " ".join(shlex.quote(part) for part in cmd)
    print(f"[real-data-strict] {name}: {rendered}")
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=False,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-json",
        default="artifacts/real_shot_validation.json",
        help="Output/input real-shot validation JSON artifact path.",
    )
    parser.add_argument(
        "--report-md",
        default="artifacts/real_shot_validation.md",
        help="Output real-shot validation Markdown path.",
    )
    parser.add_argument(
        "--guard-summary-json",
        default="artifacts/real_shot_validation_guard_summary.json",
        help="Output path for real-shot guard summary JSON.",
    )
    parser.add_argument(
        "--progress-json",
        default="artifacts/real_data_roadmap_progress.json",
        help="Output path for roadmap progress JSON.",
    )
    parser.add_argument(
        "--progress-md",
        default="artifacts/real_data_roadmap_progress.md",
        help="Output path for roadmap progress Markdown.",
    )
    parser.add_argument(
        "--non-regression-summary-json",
        default="artifacts/real_data_roadmap_non_regression_summary.json",
        help="Output path for roadmap non-regression summary JSON.",
    )
    parser.add_argument(
        "--thresholds",
        default="tools/real_shot_validation_thresholds_raw_ready.json",
        help="Threshold profile for tools/real_shot_validation_guard.py.",
    )
    parser.add_argument(
        "--targets",
        default="tools/real_data_roadmap_targets.json",
        help="Target profile for tools/real_data_roadmap_progress.py.",
    )
    parser.add_argument(
        "--baseline-json",
        default="tools/real_data_roadmap_baseline.json",
        help="Baseline file for tools/real_data_roadmap_non_regression_guard.py.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation/validate_real_shots.py and use pre-existing artifacts.",
    )
    parser.add_argument(
        "--allow-missing-raw-ingestion",
        action="store_true",
        help="Do not fail if roadmap progress reports d3d_raw_ingestion_ready=false.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-step subprocess timeout in seconds.",
    )
    args = parser.parse_args(argv)

    report_json = _resolve(args.report_json)
    report_md = _resolve(args.report_md)
    guard_summary_json = _resolve(args.guard_summary_json)
    progress_json = _resolve(args.progress_json)
    progress_md = _resolve(args.progress_md)
    non_regression_summary_json = _resolve(args.non_regression_summary_json)
    thresholds_path = _resolve(args.thresholds)
    targets_path = _resolve(args.targets)
    baseline_path = _resolve(args.baseline_json)
    guard_thresholds_path = thresholds_path

    if args.allow_missing_raw_ingestion:
        thresholds_payload = _load_json(thresholds_path)
        if bool(thresholds_payload.get("require_disruption_raw_ingestion_ready", False)):
            thresholds_payload["require_disruption_raw_ingestion_ready"] = False
            dry_run_thresholds = REPO_ROOT / "artifacts" / "_tmp_real_shot_validation_thresholds_dry_run.json"
            dry_run_thresholds.parent.mkdir(parents=True, exist_ok=True)
            dry_run_thresholds.write_text(
                json.dumps(thresholds_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            guard_thresholds_path = dry_run_thresholds
            print(
                "[real-data-strict] Dry-run enabled: using guard thresholds override "
                f"at {guard_thresholds_path}"
            )

    if not args.skip_validation:
        _run_step(
            "real-shot validation",
            [
                sys.executable,
                "validation/validate_real_shots.py",
                "--strict-coverage",
                "--output-json",
                str(report_json),
                "--output-md",
                str(report_md),
            ],
            timeout_seconds=float(args.timeout_seconds),
        )
    elif not report_json.exists():
        raise FileNotFoundError(
            f"--skip-validation was set but report JSON was not found: {report_json}"
        )

    _run_step(
        "real-shot guard",
        [
            sys.executable,
            "tools/real_shot_validation_guard.py",
            "--report",
            str(report_json),
            "--thresholds",
            str(guard_thresholds_path),
            "--summary-json",
            str(guard_summary_json),
        ],
        timeout_seconds=float(args.timeout_seconds),
    )
    _run_step(
        "real-data roadmap progress",
        [
            sys.executable,
            "tools/real_data_roadmap_progress.py",
            "--report",
            str(report_json),
            "--targets",
            str(targets_path),
            "--output-json",
            str(progress_json),
            "--output-md",
            str(progress_md),
        ],
        timeout_seconds=float(args.timeout_seconds),
    )

    progress_payload = _load_json(progress_json)
    raw_ready = bool(progress_payload.get("d3d_raw_ingestion_ready", False))
    raw_source_contract = _has_raw_source_type(progress_payload)
    print(
        "[real-data-strict] "
        f"d3d_raw_ingestion_ready={raw_ready} "
        f"d3d_raw_source_contract={raw_source_contract}"
    )
    if raw_ready and not raw_source_contract and not args.allow_missing_raw_ingestion:
        print(
            "Strict real-data gate failed: d3d_raw_ingestion_ready is true but "
            "d3d_raw_source_type_present/source_types contract is missing."
        )
        return 1
    if not args.allow_missing_raw_ingestion and not raw_ready:
        print(
            "Strict real-data gate failed: d3d_raw_ingestion_ready is false. "
            "Use --allow-missing-raw-ingestion only for dry-runs."
        )
        return 1

    _run_step(
        "real-data roadmap non-regression guard",
        [
            sys.executable,
            "tools/real_data_roadmap_non_regression_guard.py",
            "--progress-json",
            str(progress_json),
            "--baseline-json",
            str(baseline_path),
            "--summary-json",
            str(non_regression_summary_json),
        ],
        timeout_seconds=float(args.timeout_seconds),
    )

    print("[real-data-strict] Strict real-data gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
