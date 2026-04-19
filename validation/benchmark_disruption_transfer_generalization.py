#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Disruption transfer-generalization benchmark (source-scenario -> target-scenario)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from validate_real_shots import validate_disruption


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DISRUPTION_DIR = ROOT / "validation" / "reference_data" / "diiid" / "disruption_shots"
DEFAULT_OUTPUT_JSON = ROOT / "artifacts" / "disruption_transfer_generalization.json"
DEFAULT_OUTPUT_MD = ROOT / "artifacts" / "disruption_transfer_generalization.md"
DEFAULT_SOURCE_SCENARIOS = (
    "hmode",
    "hmode_safe",
    "hybrid",
    "hybrid_safe",
    "tearing",
)

THRESHOLDS = {
    "source_recall_min": 0.60,
    "target_recall_min": 0.60,
    "target_fpr_max": 0.40,
    "transfer_efficiency_min": 0.70,
    "min_source_disruptions": 1,
    "min_target_disruptions": 1,
}


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _scenario_from_filename(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) >= 3:
        return "_".join(parts[2:])
    return stem


def _empty_group() -> dict[str, Any]:
    return {
        "n_shots": 0,
        "n_disruptions": 0,
        "n_safe": 0,
        "true_positives": 0,
        "false_negatives": 0,
        "false_positives": 0,
        "true_negatives": 0,
        "recall": 0.0,
        "false_positive_rate": 0.0,
    }


def _add_observation(group: dict[str, Any], *, is_disruption: bool, detected: bool) -> None:
    group["n_shots"] += 1
    if is_disruption:
        group["n_disruptions"] += 1
        if detected:
            group["true_positives"] += 1
        else:
            group["false_negatives"] += 1
    else:
        group["n_safe"] += 1
        if detected:
            group["false_positives"] += 1
        else:
            group["true_negatives"] += 1


def _finalize_group(group: dict[str, Any]) -> None:
    disruptions = max(int(group["n_disruptions"]), 1)
    safe = max(int(group["n_safe"]), 1)
    group["recall"] = round(float(group["true_positives"]) / disruptions, 3)
    group["false_positive_rate"] = round(float(group["false_positives"]) / safe, 3)


def run_benchmark(
    *,
    disruption_dir: Path,
    source_scenarios: tuple[str, ...],
) -> dict[str, Any]:
    disruption = validate_disruption(disruption_dir)
    shots = disruption.get("shots", [])
    if not isinstance(shots, list):
        raise ValueError("validate_disruption payload is missing shots list.")

    source_set = set(source_scenarios)
    source = _empty_group()
    target = _empty_group()
    scenario_counts: dict[str, int] = {}

    for row in shots:
        if not isinstance(row, dict):
            continue
        filename = str(row.get("file", ""))
        scenario = _scenario_from_filename(filename)
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        is_disruption = bool(row.get("is_disruption", False))
        detected = bool(row.get("detected", False))
        if scenario in source_set:
            _add_observation(source, is_disruption=is_disruption, detected=detected)
        else:
            _add_observation(target, is_disruption=is_disruption, detected=detected)

    _finalize_group(source)
    _finalize_group(target)

    source_recall = float(source["recall"])
    target_recall = float(target["recall"])
    transfer_eff = 1.0
    if source_recall > 0.0:
        transfer_eff = target_recall / source_recall
    transfer_eff = float(round(transfer_eff, 3))

    source_cov_ok = int(source["n_disruptions"]) >= THRESHOLDS["min_source_disruptions"]
    target_cov_ok = int(target["n_disruptions"]) >= THRESHOLDS["min_target_disruptions"]
    source_recall_ok = source_recall >= THRESHOLDS["source_recall_min"]
    target_recall_ok = target_recall >= THRESHOLDS["target_recall_min"]
    target_fpr_ok = float(target["false_positive_rate"]) <= THRESHOLDS["target_fpr_max"]
    transfer_eff_ok = transfer_eff >= THRESHOLDS["transfer_efficiency_min"]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_scenarios": sorted(source_set),
        "thresholds": dict(THRESHOLDS),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "source_group": source,
        "target_group": target,
        "transfer_efficiency": transfer_eff,
        "checks": {
            "source_coverage_ok": source_cov_ok,
            "target_coverage_ok": target_cov_ok,
            "source_recall_ok": source_recall_ok,
            "target_recall_ok": target_recall_ok,
            "target_fpr_ok": target_fpr_ok,
            "transfer_efficiency_ok": transfer_eff_ok,
        },
        "passes": bool(
            source_cov_ok
            and target_cov_ok
            and source_recall_ok
            and target_recall_ok
            and target_fpr_ok
            and transfer_eff_ok
        ),
    }


def _render_md(report: dict[str, Any], *, disruption_dir: Path) -> str:
    src = report["source_group"]
    tgt = report["target_group"]
    lines = [
        "# Disruption Transfer-Generalization Benchmark",
        "",
        f"- Generated at: `{report['generated_at_utc']}`",
        f"- Disruption shots: `{disruption_dir.as_posix()}`",
        f"- Source scenarios: `{', '.join(report['source_scenarios'])}`",
        f"- Overall: `{'PASS' if report['passes'] else 'FAIL'}`",
        "",
        "| Group | Shots | Disruptions | Safe | Recall | FPR |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Source | {src['n_shots']} | {src['n_disruptions']} | {src['n_safe']} | "
            f"{src['recall']:.3f} | {src['false_positive_rate']:.3f} |"
        ),
        (
            f"| Target | {tgt['n_shots']} | {tgt['n_disruptions']} | {tgt['n_safe']} | "
            f"{tgt['recall']:.3f} | {tgt['false_positive_rate']:.3f} |"
        ),
        "",
        f"- Transfer efficiency (target/source recall): `{report['transfer_efficiency']:.3f}`",
        "",
        "## Checks",
        "",
    ]
    for key, value in report["checks"].items():
        lines.append(f"- {key}: `{'PASS' if value else 'FAIL'}`")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--disruption-dir",
        default=str(DEFAULT_DISRUPTION_DIR),
        help="Directory containing disruption-shot NPZ files.",
    )
    parser.add_argument(
        "--source-scenarios",
        default=",".join(DEFAULT_SOURCE_SCENARIOS),
        help="Comma-separated scenarios treated as source-domain calibration cohort.",
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
        help="Return non-zero when benchmark checks fail.",
    )
    args = parser.parse_args(argv)

    disruption_dir = _resolve(args.disruption_dir)
    source_scenarios = tuple(
        scenario.strip() for scenario in str(args.source_scenarios).split(",") if scenario.strip()
    )
    if not source_scenarios:
        raise ValueError("source_scenarios must not be empty.")

    report = run_benchmark(disruption_dir=disruption_dir, source_scenarios=source_scenarios)

    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_render_md(report, disruption_dir=disruption_dir), encoding="utf-8")

    print(
        "Transfer benchmark "
        f"{'PASS' if report['passes'] else 'FAIL'}: "
        f"source_recall={report['source_group']['recall']:.3f} "
        f"target_recall={report['target_group']['recall']:.3f} "
        f"target_fpr={report['target_group']['false_positive_rate']:.3f} "
        f"transfer_eff={report['transfer_efficiency']:.3f}"
    )
    print(f"- {_display_path(output_json)}")
    print(f"- {_display_path(output_md)}")

    if args.strict and not bool(report["passes"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
