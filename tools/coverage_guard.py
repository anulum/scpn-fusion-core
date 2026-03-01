#!/usr/bin/env python
"""Coverage regression guard for release-lane Python coverage artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COVERAGE_XML = REPO_ROOT / "coverage-python.xml"
DEFAULT_THRESHOLDS = REPO_ROOT / "tools" / "coverage_guard_thresholds.json"


@dataclass(frozen=True)
class CoverageSummary:
    line_rate_pct: float
    branch_rate_pct: float | None
    file_line_rate_pct: dict[str, float]
    file_branch_rate_pct: dict[str, float]
    domain_line_rate_pct: dict[str, float]
    domain_branch_rate_pct: dict[str, float]
    lines_covered: int
    lines_valid: int
    branches_covered: int
    branches_valid: int


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _validate_percent(value: float, *, label: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite.")
    if numeric < 0.0 or numeric > 100.0:
        raise ValueError(f"{label} must be in [0, 100].")
    return numeric


def _domain_for(filename: str) -> str:
    normalized = filename.replace("\\", "/")
    parts = normalized.split("/")
    if len(parts) >= 3 and parts[0] == "src" and parts[1] == "scpn_fusion":
        return parts[2]
    return "other"


_CONDITION_COVERAGE_PATTERN = re.compile(r"\((\d+)\s*/\s*(\d+)\)")


def _parse_condition_coverage(value: str) -> tuple[int, int] | None:
    match = _CONDITION_COVERAGE_PATTERN.search(value)
    if match is None:
        return None
    covered = int(match.group(1))
    valid = int(match.group(2))
    if valid <= 0 or covered < 0 or covered > valid:
        return None
    return covered, valid


def load_coverage(path: Path) -> CoverageSummary:
    if not path.exists():
        raise FileNotFoundError(f"Coverage XML not found: {path}")
    root = ET.parse(path).getroot()

    line_rate = _validate_percent(float(root.get("line-rate", "0.0")) * 100.0, label="line_rate")
    lines_covered = int(root.get("lines-covered", "0"))
    lines_valid = int(root.get("lines-valid", "0"))
    branches_covered = int(root.get("branches-covered", "0"))
    branches_valid = int(root.get("branches-valid", "0"))
    branch_rate_attr = root.get("branch-rate")

    file_line_rate_pct: dict[str, float] = {}
    file_branch_rate_pct: dict[str, float] = {}
    domain_hits: dict[str, tuple[int, int]] = {}
    domain_branches: dict[str, tuple[int, int]] = {}
    for cls in root.findall(".//class"):
        filename = cls.get("filename", "").replace("\\", "/")
        if not filename:
            continue
        class_rate = _validate_percent(
            float(cls.get("line-rate", "0.0")) * 100.0,
            label=f"line_rate[{filename}]",
        )
        file_line_rate_pct[filename] = class_rate
        class_branch_rate = cls.get("branch-rate")
        if class_branch_rate is not None:
            file_branch_rate_pct[filename] = _validate_percent(
                float(class_branch_rate) * 100.0,
                label=f"branch_rate[{filename}]",
            )

        covered = 0
        total = 0
        branch_covered = 0
        branch_total = 0
        for line in cls.findall("./lines/line"):
            total += 1
            if int(line.get("hits", "0")) > 0:
                covered += 1
            if str(line.get("branch", "")).lower() == "true":
                condition_coverage = line.get("condition-coverage", "")
                parsed = _parse_condition_coverage(condition_coverage)
                if parsed is not None:
                    branch_covered += parsed[0]
                    branch_total += parsed[1]
        domain = _domain_for(filename)
        existing = domain_hits.get(domain, (0, 0))
        domain_hits[domain] = (existing[0] + covered, existing[1] + total)
        if branch_total > 0:
            file_branch_rate_pct[filename] = _validate_percent(
                100.0 * branch_covered / branch_total,
                label=f"branch_rate[{filename}]",
            )
            existing_branch = domain_branches.get(domain, (0, 0))
            domain_branches[domain] = (
                existing_branch[0] + branch_covered,
                existing_branch[1] + branch_total,
            )

    domain_line_rate_pct: dict[str, float] = {}
    for domain, (covered, total) in domain_hits.items():
        pct = 100.0 * covered / total if total > 0 else 0.0
        domain_line_rate_pct[domain] = _validate_percent(pct, label=f"domain_line_rate[{domain}]")
    domain_branch_rate_pct: dict[str, float] = {}
    for domain, (covered, total) in domain_branches.items():
        pct = 100.0 * covered / total if total > 0 else 0.0
        domain_branch_rate_pct[domain] = _validate_percent(
            pct,
            label=f"domain_branch_rate[{domain}]",
        )

    # Prefer explicit coverage XML counters.
    branch_rate_pct: float | None = None
    if branches_valid > 0:
        branch_rate_pct = _validate_percent(
            100.0 * branches_covered / branches_valid,
            label="branch_rate",
        )
    elif branch_rate_attr is not None:
        # Cobertura emits branch-rate even when no branches are instrumented.
        # Treat this as missing branch signal when branches_valid == 0.
        branch_rate_pct = None

    return CoverageSummary(
        line_rate_pct=line_rate,
        branch_rate_pct=branch_rate_pct,
        file_line_rate_pct=file_line_rate_pct,
        file_branch_rate_pct=file_branch_rate_pct,
        domain_line_rate_pct=domain_line_rate_pct,
        domain_branch_rate_pct=domain_branch_rate_pct,
        lines_covered=lines_covered,
        lines_valid=lines_valid,
        branches_covered=branches_covered,
        branches_valid=branches_valid,
    )


def load_thresholds(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Coverage threshold config not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Threshold config must be a JSON object.")
    if "global_min_line_rate" not in data:
        raise ValueError("Threshold config must define global_min_line_rate.")
    _validate_percent(float(data["global_min_line_rate"]), label="global_min_line_rate")
    optional_scalar_keys = ("global_min_branch_rate",)
    for key in optional_scalar_keys:
        if key in data:
            _validate_percent(float(data[key]), label=key)

    for key in (
        "domain_min_line_rate",
        "file_min_line_rate",
        "domain_min_branch_rate",
        "file_min_branch_rate",
    ):
        value = data.get(key, {})
        if not isinstance(value, dict):
            raise ValueError(f"{key} must be a JSON object when provided.")
        for sub_key, sub_value in value.items():
            _validate_percent(float(sub_value), label=f"{key}[{sub_key}]")
    return data


def evaluate(summary: CoverageSummary, thresholds: dict[str, object]) -> list[str]:
    failures: list[str] = []

    global_min = float(thresholds["global_min_line_rate"])
    if summary.line_rate_pct < global_min:
        failures.append(
            f"Global line coverage {summary.line_rate_pct:.2f}% < threshold {global_min:.2f}%."
        )
    if "global_min_branch_rate" in thresholds:
        branch_target = float(thresholds["global_min_branch_rate"])
        if summary.branch_rate_pct is None:
            failures.append(
                "Global branch coverage missing from report (branches-valid=0); "
                "run coverage with branch collection (e.g. pytest --cov-branch)."
            )
        elif summary.branch_rate_pct < branch_target:
            failures.append(
                f"Global branch coverage {summary.branch_rate_pct:.2f}% < "
                f"threshold {branch_target:.2f}%."
            )

    domain_min = thresholds.get("domain_min_line_rate", {})
    if isinstance(domain_min, dict):
        for domain, threshold in domain_min.items():
            target = float(threshold)
            observed = summary.domain_line_rate_pct.get(domain)
            if observed is None:
                failures.append(f"Domain '{domain}' missing from coverage report.")
                continue
            if observed < target:
                failures.append(
                    f"Domain '{domain}' coverage {observed:.2f}% < threshold {target:.2f}%."
                )

    file_min = thresholds.get("file_min_line_rate", {})
    if isinstance(file_min, dict):
        for filename, threshold in file_min.items():
            target = float(threshold)
            observed = summary.file_line_rate_pct.get(filename)
            if observed is None:
                failures.append(f"File '{filename}' missing from coverage report.")
                continue
            if observed < target:
                failures.append(
                    f"File '{filename}' coverage {observed:.2f}% < threshold {target:.2f}%."
                )

    domain_min_branch = thresholds.get("domain_min_branch_rate", {})
    if isinstance(domain_min_branch, dict):
        for domain, threshold in domain_min_branch.items():
            target = float(threshold)
            observed = summary.domain_branch_rate_pct.get(domain)
            if observed is None:
                failures.append(f"Domain '{domain}' missing branch coverage data.")
                continue
            if observed < target:
                failures.append(
                    f"Domain '{domain}' branch coverage {observed:.2f}% < "
                    f"threshold {target:.2f}%."
                )

    file_min_branch = thresholds.get("file_min_branch_rate", {})
    if isinstance(file_min_branch, dict):
        for filename, threshold in file_min_branch.items():
            target = float(threshold)
            observed = summary.file_branch_rate_pct.get(filename)
            if observed is None:
                failures.append(f"File '{filename}' missing branch coverage data.")
                continue
            if observed < target:
                failures.append(
                    f"File '{filename}' branch coverage {observed:.2f}% < "
                    f"threshold {target:.2f}%."
                )

    return failures


def _write_summary(path: Path, summary: CoverageSummary) -> None:
    payload = {
        "line_rate_pct": summary.line_rate_pct,
        "branch_rate_pct": summary.branch_rate_pct,
        "lines_covered": summary.lines_covered,
        "lines_valid": summary.lines_valid,
        "branches_covered": summary.branches_covered,
        "branches_valid": summary.branches_valid,
        "domain_line_rate_pct": summary.domain_line_rate_pct,
        "domain_branch_rate_pct": summary.domain_branch_rate_pct,
        "file_line_rate_pct": summary.file_line_rate_pct,
        "file_branch_rate_pct": summary.file_branch_rate_pct,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage-xml",
        default=str(DEFAULT_COVERAGE_XML),
        help="Path to coverage XML report (Cobertura format).",
    )
    parser.add_argument(
        "--thresholds",
        default=str(DEFAULT_THRESHOLDS),
        help="JSON file containing coverage thresholds.",
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional path to write coverage summary JSON.",
    )
    args = parser.parse_args(argv)

    coverage_path = _resolve(args.coverage_xml)
    thresholds_path = _resolve(args.thresholds)
    summary = load_coverage(coverage_path)
    thresholds = load_thresholds(thresholds_path)

    print(
        f"Coverage line rate: {summary.line_rate_pct:.2f}% "
        f"({summary.lines_covered}/{summary.lines_valid})"
    )
    if summary.branch_rate_pct is None:
        print("Coverage branch rate: n/a (no branch counters in coverage XML)")
    else:
        print(
            f"Coverage branch rate: {summary.branch_rate_pct:.2f}% "
            f"({summary.branches_covered}/{summary.branches_valid})"
        )
    for domain, pct in sorted(summary.domain_line_rate_pct.items()):
        print(f"Domain {domain:12s}: {pct:6.2f}%")
    for domain, pct in sorted(summary.domain_branch_rate_pct.items()):
        print(f"Domain {domain:12s} (branch): {pct:6.2f}%")

    if args.summary_json:
        _write_summary(_resolve(args.summary_json), summary)

    failures = evaluate(summary, thresholds)
    if failures:
        print("Coverage guard FAILED:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Coverage guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
