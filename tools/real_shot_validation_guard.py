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
    required_transport_machines = [
        str(v) for v in thresholds.get("required_transport_machines", [])
    ]
    min_transport_machine_count = int(thresholds.get("min_transport_machine_count", 0))
    min_disruptions = int(thresholds.get("min_disruption_events", 0))
    min_safe = int(thresholds.get("min_safe_events", 0))
    require_calibration_gate = bool(
        thresholds.get("require_disruption_calibration_gate_pass", False)
    )
    require_minima_match = bool(
        thresholds.get("require_dataset_minima_match_thresholds", False)
    )
    require_disruption_source_contract = bool(
        thresholds.get("require_disruption_source_contract", False)
    )
    require_disruption_raw_ingestion_ready = bool(
        thresholds.get("require_disruption_raw_ingestion_ready", False)
    )

    eq = dict(report.get("equilibrium", {}))
    tr = dict(report.get("transport", {}))
    dis = dict(report.get("disruption", {}))
    coverage = dict(report.get("dataset_coverage", {}))
    dataset_minima = dict(report.get("dataset_minima", {}))
    eq_results = [dict(r) for r in eq.get("results", []) if isinstance(r, dict)]
    tr_results = [dict(r) for r in tr.get("shots", []) if isinstance(r, dict)]
    observed_machines = sorted({str(r.get("machine", "unknown")) for r in eq_results if "machine" in r})
    observed_transport_machines = sorted(
        {str(r.get("machine", "unknown")) for r in tr_results if "machine" in r}
    )

    min_eq = int(coverage_cfg.get("equilibrium_files", 0))
    min_tr = int(coverage_cfg.get("transport_shots", 0))
    min_dis = int(coverage_cfg.get("disruption_shots", 0))

    eq_count = int(eq.get("n_files", 0))
    tr_count = int(tr.get("n_shots", 0))
    dis_count = int(dis.get("n_shots", 0))
    disruption_count = int(dis.get("n_disruptions", 0))
    safe_count = int(dis.get("n_safe", 0))
    calibration = dict(dis.get("calibration", {}))
    calibration_gate = calibration.get("gates_overall_pass")
    disruption_data_source = dis.get("data_source", {})
    if not isinstance(disruption_data_source, dict):
        disruption_data_source = {}
    source_types_raw = disruption_data_source.get("source_types", [])
    if not isinstance(source_types_raw, list):
        source_types_raw = []
    source_types = sorted(
        {str(v).strip() for v in source_types_raw if str(v).strip()}
    )
    source_contract_present = bool(
        disruption_data_source.get("manifest_found", False) and source_types
    )
    raw_ingestion_ready = bool(disruption_data_source.get("raw_ingestion_ready", False))

    counts_pass = bool(eq_count >= min_eq and tr_count >= min_tr and dis_count >= min_dis)
    machines_pass = all(machine in observed_machines for machine in required_machines)
    transport_machine_pass = (
        all(machine in observed_transport_machines for machine in required_transport_machines)
        and len(observed_transport_machines) >= min_transport_machine_count
    )
    disruption_balance_pass = bool(disruption_count >= min_disruptions and safe_count >= min_safe)
    calibration_pass = (
        bool(calibration_gate) if require_calibration_gate else True
    )
    minima_alignment_pass = True
    if require_minima_match:
        minima_alignment_pass = bool(
            int(dataset_minima.get("equilibrium_files", 0)) >= min_eq
            and int(dataset_minima.get("transport_shots", 0)) >= min_tr
            and int(dataset_minima.get("disruption_shots", 0)) >= min_dis
        )
    disruption_source_pass = bool(
        (source_contract_present if require_disruption_source_contract else True)
        and (raw_ingestion_ready if require_disruption_raw_ingestion_ready else True)
    )
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
        "transport_machine_diversity": {
            "observed": observed_transport_machines,
            "required": required_transport_machines,
            "required_min_count": min_transport_machine_count,
            "passes": transport_machine_pass,
        },
        "disruption_event_balance": {
            "observed_disruptions": disruption_count,
            "observed_safe": safe_count,
            "required_disruptions": min_disruptions,
            "required_safe": min_safe,
            "passes": disruption_balance_pass,
        },
        "calibration_gate": {
            "required": require_calibration_gate,
            "observed_overall_pass": calibration_gate,
            "passes": calibration_pass,
        },
        "disruption_data_source": {
            "require_source_contract": require_disruption_source_contract,
            "require_raw_ingestion_ready": require_disruption_raw_ingestion_ready,
            "source_contract_present": source_contract_present,
            "raw_ingestion_ready": raw_ingestion_ready,
            "observed_source_types": source_types,
            "passes": disruption_source_pass,
        },
        "dataset_minima_alignment": {
            "required": require_minima_match,
            "observed": {
                "equilibrium_files": int(dataset_minima.get("equilibrium_files", 0)),
                "transport_shots": int(dataset_minima.get("transport_shots", 0)),
                "disruption_shots": int(dataset_minima.get("disruption_shots", 0)),
            },
            "threshold_minima": {
                "equilibrium_files": min_eq,
                "transport_shots": min_tr,
                "disruption_shots": min_dis,
            },
            "passes": minima_alignment_pass,
        },
        "overall_pass": bool(
            artifact_pass
            and counts_pass
            and machines_pass
            and transport_machine_pass
            and disruption_balance_pass
            and calibration_pass
            and disruption_source_pass
            and minima_alignment_pass
        ),
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
        f"machines_pass={summary['equilibrium_machine_diversity']['passes']} "
        f"transport_machine_pass={summary['transport_machine_diversity']['passes']} "
        f"disruption_balance_pass={summary['disruption_event_balance']['passes']} "
        f"calibration_pass={summary['calibration_gate']['passes']} "
        f"disruption_source_pass={summary['disruption_data_source']['passes']}"
    )
    if not bool(summary["overall_pass"]):
        print("Real-shot validation guard failed.")
        return 1
    print("Real-shot validation guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
