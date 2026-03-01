#!/usr/bin/env python
"""Guard real-shot validation artifact minima and machine diversity contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPO_ROOT / "artifacts" / "real_shot_validation.json"
DEFAULT_THRESHOLDS = REPO_ROOT / "tools" / "real_shot_validation_thresholds.json"
DEFAULT_SUMMARY = REPO_ROOT / "artifacts" / "real_shot_validation_guard_summary.json"


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


def evaluate(*, report: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    coverage_cfg = dict(thresholds.get("coverage_minima", {}))
    required_machines = [str(v) for v in thresholds.get("required_equilibrium_machines", [])]

    eq = dict(report.get("equilibrium", {}))
    tr = dict(report.get("transport", {}))
    dis = dict(report.get("disruption", {}))
    coverage = dict(report.get("dataset_coverage", {}))
    eq_results = [dict(r) for r in eq.get("results", []) if isinstance(r, dict)]
    observed_machines = sorted({str(r.get("machine", "unknown")) for r in eq_results if "machine" in r})

    min_eq = int(coverage_cfg.get("equilibrium_files", 0))
    min_tr = int(coverage_cfg.get("transport_shots", 0))
    min_dis = int(coverage_cfg.get("disruption_shots", 0))

    eq_count = int(eq.get("n_files", 0))
    tr_count = int(tr.get("n_shots", 0))
    dis_count = int(dis.get("n_shots", 0))

    counts_pass = bool(eq_count >= min_eq and tr_count >= min_tr and dis_count >= min_dis)
    machines_pass = all(machine in observed_machines for machine in required_machines)
    artifact_pass = bool(report.get("overall_pass", False)) and bool(coverage.get("passes", False))

    return {
        "overall_artifact_pass": artifact_pass,
        "coverage_counts": {
            "equilibrium_files": {"observed": eq_count, "required_min": min_eq, "passes": eq_count >= min_eq},
            "transport_shots": {"observed": tr_count, "required_min": min_tr, "passes": tr_count >= min_tr},
            "disruption_shots": {"observed": dis_count, "required_min": min_dis, "passes": dis_count >= min_dis},
            "passes": counts_pass,
        },
        "equilibrium_machine_diversity": {
            "observed": observed_machines,
            "required": required_machines,
            "passes": machines_pass,
        },
        "overall_pass": bool(artifact_pass and counts_pass and machines_pass),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args(argv)

    report_path = _resolve(args.report)
    thresholds_path = _resolve(args.thresholds)
    summary_path = _resolve(args.summary_json)

    summary = evaluate(report=_load_json(report_path), thresholds=_load_json(thresholds_path))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "Real-shot guard summary: "
        f"artifact_pass={summary['overall_artifact_pass']} "
        f"coverage_pass={summary['coverage_counts']['passes']} "
        f"machines_pass={summary['equilibrium_machine_diversity']['passes']}"
    )
    if not bool(summary["overall_pass"]):
        print("Real-shot validation guard failed.")
        return 1
    print("Real-shot validation guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
