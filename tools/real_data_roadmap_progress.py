#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Summarize roadmap progress for real-data validation expansion goals."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPO_ROOT / "artifacts" / "real_shot_validation.json"
DEFAULT_TARGETS = REPO_ROOT / "tools" / "real_data_roadmap_targets.json"
DEFAULT_JSON = REPO_ROOT / "artifacts" / "real_data_roadmap_progress.json"
DEFAULT_MD = REPO_ROOT / "artifacts" / "real_data_roadmap_progress.md"
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


def _safe_ratio(current: int, target: int) -> float:
    if target <= 0:
        return 1.0
    return float(min(1.0, max(0.0, current / target)))


def _is_raw_source_type(value: str) -> bool:
    lowered = str(value).strip().lower()
    return bool(lowered) and any(token in lowered for token in _RAW_SOURCE_TYPE_TOKENS)


def evaluate_progress(*, report: dict[str, Any], targets: dict[str, Any]) -> dict[str, Any]:
    target_cfg = targets.get("targets", {})
    if not isinstance(target_cfg, dict):
        raise ValueError("targets payload missing 'targets' object")

    eq = report.get("equilibrium", {})
    tr = report.get("transport", {})
    dis = report.get("disruption", {})
    eq_rows = eq.get("results", [])
    tr_rows = tr.get("shots", [])
    if not isinstance(eq_rows, list):
        eq_rows = []
    if not isinstance(tr_rows, list):
        tr_rows = []

    equilibrium_total = int(eq.get("n_files", 0) or 0)
    sparc_equilibria = sum(
        1
        for row in eq_rows
        if isinstance(row, dict) and str(row.get("machine", "")).upper() == "SPARC"
    )
    transport_total = int(tr.get("n_shots", 0) or 0)
    transport_machines = sorted(
        {
            str(row.get("machine", "")).strip()
            for row in tr_rows
            if isinstance(row, dict) and str(row.get("machine", "")).strip()
        }
    )
    disruption_total = int(dis.get("n_shots", 0) or 0)
    jet_dt_equilibria = sum(
        1
        for row in eq_rows
        if isinstance(row, dict)
        and str(row.get("machine", "")).upper() == "JET"
        and "dt" in str(row.get("file", "")).lower()
    )
    calibration = dis.get("calibration", {})
    calibration_source = ""
    if isinstance(calibration, dict):
        calibration_source = str(calibration.get("source", "")).strip()
    disruption_source = dis.get("data_source", {})
    d3d_disruption_source_types: list[str] = []
    d3d_raw_ingestion_ready = False
    if isinstance(disruption_source, dict):
        source_types_raw = disruption_source.get("source_types", [])
        if isinstance(source_types_raw, list):
            d3d_disruption_source_types = sorted(
                {str(value).strip() for value in source_types_raw if str(value).strip()}
            )
        if bool(disruption_source.get("raw_ingestion_ready", False)):
            # Keep readiness tied to provenance-contract semantics instead of
            # trusting the boolean in isolation.
            d3d_raw_ingestion_ready = any(
                _is_raw_source_type(source_type) for source_type in d3d_disruption_source_types
            )
    if not d3d_raw_ingestion_ready:
        # Backward-compatible fallback for historical artifacts lacking data_source contract.
        d3d_raw_ingestion_ready = bool(
            calibration_source
            and any(token in calibration_source.lower() for token in ("raw", "mdsplus"))
        )
    d3d_raw_source_type_present = any(
        _is_raw_source_type(source_type) for source_type in d3d_disruption_source_types
    ) or bool(
        d3d_raw_ingestion_ready
        and calibration_source
        and any(token in calibration_source.lower() for token in ("raw", "mdsplus"))
    )

    observed = {
        "equilibrium_files_total": equilibrium_total,
        "sparc_equilibria": sparc_equilibria,
        "transport_shots_total": transport_total,
        "transport_machines_total": len(transport_machines),
        "disruption_shots_total": disruption_total,
        "jet_dt_equilibria": jet_dt_equilibria,
    }

    rows: list[dict[str, Any]] = []
    overall_pass = True
    completion_ratios: list[float] = []
    for key, target_raw in target_cfg.items():
        target = int(target_raw)
        current = int(observed.get(key, 0))
        passed = current >= target
        progress_ratio = _safe_ratio(current, target)
        overall_pass = overall_pass and passed
        completion_ratios.append(progress_ratio)
        rows.append(
            {
                "metric": key,
                "current": current,
                "target": target,
                "progress_ratio": progress_ratio,
                "remaining_to_target": max(0, target - current),
                "passes": passed,
            }
        )

    overall_progress_ratio = (
        float(sum(completion_ratios) / len(completion_ratios)) if completion_ratios else 1.0
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "roadmap_version": str(targets.get("roadmap_version", "v4.0")),
        "overall_pass": overall_pass,
        "overall_progress_ratio": overall_progress_ratio,
        "d3d_raw_ingestion_ready": d3d_raw_ingestion_ready,
        "d3d_raw_source_type_present": d3d_raw_source_type_present,
        "d3d_disruption_source_types": d3d_disruption_source_types,
        "d3d_calibration_source": calibration_source or None,
        "transport_machines": transport_machines,
        "metrics": rows,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Real-Data Roadmap Progress",
        "",
        f"- Generated: `{summary['generated_at_utc']}`",
        f"- Roadmap: `{summary['roadmap_version']}`",
        f"- Overall target pass: `{'YES' if summary['overall_pass'] else 'NO'}`",
        f"- Overall target completion: `{float(summary.get('overall_progress_ratio', 0.0)) * 100.0:.1f}%`",
        f"- DIII-D raw-ingestion readiness: `{'YES' if summary['d3d_raw_ingestion_ready'] else 'NO'}`",
        f"- DIII-D raw source-type contract: `{'YES' if summary.get('d3d_raw_source_type_present', False) else 'NO'}`",
    ]
    source = summary.get("d3d_calibration_source")
    if source:
        lines.append(f"- DIII-D calibration source: `{source}`")
    source_types = summary.get("d3d_disruption_source_types", [])
    if isinstance(source_types, list) and source_types:
        lines.append(
            "- DIII-D disruption source types: "
            + ", ".join(f"`{str(source_type)}`" for source_type in source_types)
        )
    lines.extend(
        [
            "",
            "| Metric | Current | Target | Gap | Progress | Pass |",
            "|---|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in summary.get("metrics", []):
        metric = str(row.get("metric", "unknown"))
        current = int(row.get("current", 0))
        target = int(row.get("target", 0))
        remaining = int(row.get("remaining_to_target", max(0, target - current)))
        progress = float(row.get("progress_ratio", 0.0)) * 100.0
        passed = bool(row.get("passes", False))
        lines.append(
            f"| `{metric}` | {current} | {target} | {remaining} | {progress:.1f}% | {'YES' if passed else 'NO'} |"
        )
    machines = summary.get("transport_machines", [])
    if isinstance(machines, list) and machines:
        lines.extend(
            [
                "",
                "## Transport Machine Coverage",
                "",
                ", ".join(f"`{str(machine)}`" for machine in machines),
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--targets", default=str(DEFAULT_TARGETS))
    parser.add_argument("--output-json", default=str(DEFAULT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_MD))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = _load_json(_resolve(args.report))
    targets = _load_json(_resolve(args.targets))
    summary = evaluate_progress(report=report, targets=targets)

    out_json = _resolve(args.output_json)
    out_md = _resolve(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(summary), encoding="utf-8")

    print(
        "Real-data roadmap progress: "
        f"overall_pass={summary['overall_pass']} "
        f"d3d_raw_ingestion_ready={summary['d3d_raw_ingestion_ready']}"
    )
    if args.strict and not bool(summary["overall_pass"]):
        print("Roadmap progress strict gate failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
