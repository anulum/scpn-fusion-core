# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Real-Shot Validation Guard Tests
# ──────────────────────────────────────────────────────────────────────

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "real_shot_validation_guard.py"
SPEC = importlib.util.spec_from_file_location("real_shot_validation_guard", MODULE_PATH)
assert SPEC and SPEC.loader
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)


def _base_report() -> dict:
    return {
        "overall_pass": True,
        "dataset_coverage": {"passes": True},
        "dataset_minima": {
            "equilibrium_files": 12,
            "transport_shots": 52,
            "disruption_shots": 12,
        },
        "equilibrium": {
            "n_files": 18,
            "results": [
                {"machine": "SPARC"},
                {"machine": "DIII-D"},
                {"machine": "JET"},
            ],
        },
        "transport": {
            "n_shots": 53,
            "shots": [
                {"machine": "SPARC"},
                {"machine": "DIII-D"},
                {"machine": "JET"},
                {"machine": "ITER"},
                {"machine": "EAST"},
                {"machine": "KSTAR"},
                {"machine": "C-Mod"},
                {"machine": "ASDEX-U"},
                {"machine": "MAST-U"},
                {"machine": "NSTX-U"},
                {"machine": "TCV"},
                {"machine": "WEST"},
            ],
        },
        "disruption": {
            "n_shots": 16,
            "n_disruptions": 6,
            "n_safe": 10,
            "calibration": {"gates_overall_pass": True},
            "data_source": {
                "manifest_found": True,
                "source_types": ["synthetic_diiid_like"],
                "raw_ingestion_ready": False,
            },
        },
    }


def _thresholds() -> dict:
    return {
        "coverage_minima": {"equilibrium_files": 12, "transport_shots": 52, "disruption_shots": 12},
        "required_equilibrium_machines": ["SPARC", "DIII-D", "JET"],
        "required_transport_machines": ["SPARC", "DIII-D", "JET", "ITER"],
        "min_transport_machine_count": 12,
        "min_disruption_events": 4,
        "min_safe_events": 8,
        "require_disruption_calibration_gate_pass": True,
        "require_disruption_source_contract": True,
        "require_disruption_raw_ingestion_ready": False,
        "require_dataset_minima_match_thresholds": True,
    }


def test_evaluate_passes_with_valid_report() -> None:
    summary = guard.evaluate(report=_base_report(), thresholds=_thresholds())
    assert summary["overall_pass"] is True


def test_evaluate_fails_on_missing_machine() -> None:
    report = _base_report()
    report["equilibrium"]["results"] = [{"machine": "SPARC"}, {"machine": "DIII-D"}]
    summary = guard.evaluate(report=report, thresholds=_thresholds())
    assert summary["equilibrium_machine_diversity"]["passes"] is False
    assert summary["overall_pass"] is False


def test_evaluate_fails_on_missing_transport_machine() -> None:
    report = _base_report()
    report["transport"]["shots"] = [{"machine": "SPARC"}, {"machine": "DIII-D"}]
    summary = guard.evaluate(report=report, thresholds=_thresholds())
    assert summary["transport_machine_diversity"]["passes"] is False
    assert summary["overall_pass"] is False


def test_evaluate_fails_when_disruption_balance_is_below_thresholds() -> None:
    report = _base_report()
    report["disruption"]["n_disruptions"] = 2
    summary = guard.evaluate(report=report, thresholds=_thresholds())
    assert summary["disruption_event_balance"]["passes"] is False
    assert summary["overall_pass"] is False


def test_evaluate_fails_when_disruption_source_contract_missing() -> None:
    report = _base_report()
    report["disruption"]["data_source"] = {}
    summary = guard.evaluate(report=report, thresholds=_thresholds())
    assert summary["disruption_data_source"]["passes"] is False
    assert summary["overall_pass"] is False


def test_evaluate_fails_when_raw_ingestion_is_required() -> None:
    report = _base_report()
    thresholds = _thresholds()
    thresholds["require_disruption_raw_ingestion_ready"] = True
    summary = guard.evaluate(report=report, thresholds=thresholds)
    assert summary["disruption_data_source"]["passes"] is False
    assert summary["overall_pass"] is False


def test_main_writes_summary(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    thresholds_path = tmp_path / "thresholds.json"
    summary_path = tmp_path / "summary.json"
    report_path.write_text(json.dumps(_base_report()), encoding="utf-8")
    thresholds_path.write_text(json.dumps(_thresholds()), encoding="utf-8")
    rc = guard.main(
        [
            "--report",
            str(report_path),
            "--thresholds",
            str(thresholds_path),
            "--summary-json",
            str(summary_path),
        ]
    )
    assert rc == 0
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["overall_pass"] is True
