#!/usr/bin/env python
"""Validate FreeGS strict-backend artifact contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPO_ROOT / "artifacts" / "freegs_benchmark.json"
DEFAULT_SUMMARY = REPO_ROOT / "artifacts" / "freegs_strict_guard_summary.json"


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


def evaluate(report: dict[str, Any]) -> dict[str, Any]:
    cases_raw = report.get("cases", [])
    cases = [dict(case) for case in cases_raw if isinstance(case, dict)]
    if not cases:
        raise ValueError("freegs benchmark artifact has zero cases")

    checks = {
        "strict_requested": bool(report.get("require_freegs_backend", False)),
        "mode_is_freegs": str(report.get("mode", "")).strip() == "freegs",
        "runtime_fallback_disallowed": not bool(report.get("runtime_fallback_allowed", True)),
        "runtime_fallback_case_count_zero": int(report.get("freegs_runtime_fallback_cases", 0))
        == 0,
        "all_cases_pass": all(bool(case.get("passes", False)) for case in cases),
        "all_reference_backends_freegs": all(
            str(case.get("reference_backend", "")).strip() == "freegs" for case in cases
        ),
        "no_case_level_fallback": all(
            not bool(case.get("freegs_fallback", False)) for case in cases
        ),
        "no_case_level_errors": all("error" not in case for case in cases),
    }

    failed_checks = [key for key, value in checks.items() if not bool(value)]
    return {
        "overall_pass": len(failed_checks) == 0,
        "failed_checks": failed_checks,
        "checks": checks,
        "case_count": len(cases),
        "report_mode": report.get("mode"),
        "require_freegs_backend": bool(report.get("require_freegs_backend", False)),
        "runtime_fallback_allowed": bool(report.get("runtime_fallback_allowed", True)),
        "freegs_runtime_fallback_cases": int(report.get("freegs_runtime_fallback_cases", 0)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args(argv)

    report = _load_json(_resolve(args.report))
    summary = evaluate(report)

    summary_path = _resolve(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "FreeGS strict artifact guard: "
        f"pass={summary['overall_pass']} "
        f"failed_checks={len(summary['failed_checks'])}"
    )
    if not bool(summary["overall_pass"]):
        print("Failed checks:", ", ".join(summary["failed_checks"]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
