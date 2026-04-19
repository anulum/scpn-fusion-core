#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Build a coverage delta report from guard summary + threshold config."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = REPO_ROOT / "artifacts" / "coverage_guard_summary.json"
DEFAULT_THRESHOLDS = REPO_ROOT / "tools" / "coverage_guard_thresholds.json"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "artifacts" / "coverage_delta_report.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "artifacts" / "coverage_delta_report.md"


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _to_float_map(payload: Any, *, label: str) -> dict[str, float]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object.")
    out: dict[str, float] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise ValueError(f"{label} keys must be strings.")
        if not isinstance(value, (int, float)):
            raise ValueError(f"{label}[{key}] must be numeric.")
        out[key] = float(value)
    return out


def _delta_row(
    *,
    name: str,
    target: float,
    observed: float | None,
) -> dict[str, Any]:
    if observed is None:
        return {
            "name": name,
            "target": target,
            "observed": None,
            "delta": None,
            "passes": False,
            "missing_observation": True,
        }
    delta = observed - target
    return {
        "name": name,
        "target": target,
        "observed": observed,
        "delta": delta,
        "passes": delta >= 0.0,
        "missing_observation": False,
    }


def build_report(
    *,
    summary: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    line_rate = summary.get("line_rate_pct")
    branch_rate = summary.get("branch_rate_pct")
    domain_line = _to_float_map(summary.get("domain_line_rate_pct"), label="domain_line_rate_pct")
    domain_branch = _to_float_map(
        summary.get("domain_branch_rate_pct"), label="domain_branch_rate_pct"
    )
    file_line = _to_float_map(summary.get("file_line_rate_pct"), label="file_line_rate_pct")
    file_branch = _to_float_map(summary.get("file_branch_rate_pct"), label="file_branch_rate_pct")

    rows_global: list[dict[str, Any]] = []
    rows_domain_line: list[dict[str, Any]] = []
    rows_domain_branch: list[dict[str, Any]] = []
    rows_file_line: list[dict[str, Any]] = []
    rows_file_branch: list[dict[str, Any]] = []

    global_line_target = float(thresholds.get("global_min_line_rate", 0.0))
    rows_global.append(
        _delta_row(
            name="global_line_rate_pct",
            target=global_line_target,
            observed=float(line_rate) if isinstance(line_rate, (int, float)) else None,
        )
    )
    if "global_min_branch_rate" in thresholds:
        rows_global.append(
            _delta_row(
                name="global_branch_rate_pct",
                target=float(thresholds["global_min_branch_rate"]),
                observed=float(branch_rate) if isinstance(branch_rate, (int, float)) else None,
            )
        )

    for name, target in sorted(
        _to_float_map(thresholds.get("domain_min_line_rate"), label="domain_min_line_rate").items()
    ):
        rows_domain_line.append(
            _delta_row(name=name, target=target, observed=domain_line.get(name))
        )
    for name, target in sorted(
        _to_float_map(
            thresholds.get("domain_min_branch_rate"), label="domain_min_branch_rate"
        ).items()
    ):
        rows_domain_branch.append(
            _delta_row(name=name, target=target, observed=domain_branch.get(name))
        )
    for name, target in sorted(
        _to_float_map(thresholds.get("file_min_line_rate"), label="file_min_line_rate").items()
    ):
        rows_file_line.append(_delta_row(name=name, target=target, observed=file_line.get(name)))
    for name, target in sorted(
        _to_float_map(thresholds.get("file_min_branch_rate"), label="file_min_branch_rate").items()
    ):
        rows_file_branch.append(
            _delta_row(name=name, target=target, observed=file_branch.get(name))
        )

    all_rows = [
        *rows_global,
        *rows_domain_line,
        *rows_domain_branch,
        *rows_file_line,
        *rows_file_branch,
    ]
    observed_rows = [row for row in all_rows if row["delta"] is not None]
    missing_rows = [row for row in all_rows if row["missing_observation"]]
    failing_rows = [row for row in observed_rows if not row["passes"]]

    worst_row: dict[str, Any] | None = None
    if observed_rows:
        worst_row = min(observed_rows, key=lambda row: float(row["delta"]))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_pass": len(failing_rows) == 0 and len(missing_rows) == 0,
        "global": rows_global,
        "domain_line": rows_domain_line,
        "domain_branch": rows_domain_branch,
        "file_line": rows_file_line,
        "file_branch": rows_file_branch,
        "failing_count": len(failing_rows),
        "missing_count": len(missing_rows),
        "failing_checks": failing_rows,
        "missing_checks": missing_rows,
        "worst_delta_check": worst_row,
    }


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}%"


def _fmt_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}pp"


def _render_section(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Check | Target | Observed | Delta | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        status = "PASS" if row["passes"] else "FAIL"
        if row["missing_observation"]:
            status = "MISSING"
        lines.append(
            f"| `{row['name']}` | {_fmt_pct(row['target'])} | {_fmt_pct(row['observed'])} | "
            f"{_fmt_delta(row['delta'])} | {status} |"
        )
    lines.append("")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Coverage Delta Report",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Overall status: `{'PASS' if report['overall_pass'] else 'FAIL'}`",
        f"- Failing checks: `{report['failing_count']}`",
        f"- Missing observations: `{report['missing_count']}`",
        "",
    ]
    worst = report.get("worst_delta_check")
    if isinstance(worst, dict):
        lines.extend(
            [
                "## Worst Delta",
                "",
                f"- Check: `{worst.get('name', '?')}`",
                f"- Delta: `{_fmt_delta(worst.get('delta'))}`",
                "",
            ]
        )

    lines.extend(_render_section("Global Coverage Deltas", list(report["global"])))
    lines.extend(_render_section("Domain Line Coverage Deltas", list(report["domain_line"])))
    lines.extend(_render_section("Domain Branch Coverage Deltas", list(report["domain_branch"])))
    lines.extend(_render_section("File Line Coverage Deltas", list(report["file_line"])))
    lines.extend(_render_section("File Branch Coverage Deltas", list(report["file_branch"])))
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage-summary",
        default=str(DEFAULT_SUMMARY),
        help="Coverage summary JSON from tools/coverage_guard.py --summary-json.",
    )
    parser.add_argument(
        "--thresholds",
        default=str(DEFAULT_THRESHOLDS),
        help="Coverage threshold config JSON.",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_JSON),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--output-md",
        default=str(DEFAULT_OUTPUT_MD),
        help="Output Markdown report path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if any threshold delta is negative or missing.",
    )
    args = parser.parse_args(argv)

    summary_path = _resolve(args.coverage_summary)
    thresholds_path = _resolve(args.thresholds)
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)

    summary = _load_json(summary_path, label="coverage summary")
    thresholds = _load_json(thresholds_path, label="coverage thresholds")
    report = build_report(summary=summary, thresholds=thresholds)
    markdown = render_markdown(report)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")

    print(
        "Coverage delta report: "
        f"overall_pass={report['overall_pass']} "
        f"failing={report['failing_count']} "
        f"missing={report['missing_count']}"
    )
    print(f"- {_display_path(output_json)}")
    print(f"- {_display_path(output_md)}")

    if args.strict and not bool(report["overall_pass"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
