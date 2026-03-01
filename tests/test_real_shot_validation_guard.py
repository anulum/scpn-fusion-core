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
        "equilibrium": {
            "n_files": 18,
            "results": [
                {"machine": "SPARC"},
                {"machine": "DIII-D"},
                {"machine": "JET"},
            ],
        },
        "transport": {"n_shots": 53},
        "disruption": {"n_shots": 16},
    }


def _thresholds() -> dict:
    return {
        "coverage_minima": {"equilibrium_files": 12, "transport_shots": 52, "disruption_shots": 12},
        "required_equilibrium_machines": ["SPARC", "DIII-D", "JET"],
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
