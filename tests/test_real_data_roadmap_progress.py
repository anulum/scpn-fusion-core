"""Regression tests for tools/real_data_roadmap_progress.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "real_data_roadmap_progress.py"
SPEC = importlib.util.spec_from_file_location("real_data_roadmap_progress", MODULE_PATH)
assert SPEC and SPEC.loader
real_data_roadmap_progress = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(real_data_roadmap_progress)


def test_progress_summary_contains_expected_metrics() -> None:
    report = {
        "equilibrium": {
            "n_files": 18,
            "results": [
                {"machine": "SPARC", "file": "sparc_1300.eqdsk"},
                {"machine": "JET", "file": "jet_dt_3p5MA.geqdsk"},
                {"machine": "DIII-D", "file": "diiid_hmode.geqdsk"},
            ],
        },
        "transport": {
            "n_shots": 53,
            "shots": [
                {"machine": "JET"},
                {"machine": "DIII-D"},
                {"machine": "SPARC"},
                {"machine": "ITER"},
            ],
        },
        "disruption": {
            "n_shots": 16,
            "calibration": {"source": "diiid-disruption-risk-calibration-v1"},
        },
    }
    targets = {
        "roadmap_version": "v4.0",
        "targets": {
            "equilibrium_files_total": 20,
            "sparc_equilibria": 20,
            "transport_shots_total": 100,
            "transport_machines_total": 30,
            "disruption_shots_total": 16,
            "jet_dt_equilibria": 5,
        },
    }
    summary = real_data_roadmap_progress.evaluate_progress(
        report=report, targets=targets
    )
    assert summary["roadmap_version"] == "v4.0"
    assert summary["overall_pass"] is False
    assert summary["d3d_raw_ingestion_ready"] is False
    metrics = {row["metric"]: row for row in summary["metrics"]}
    assert metrics["disruption_shots_total"]["passes"] is True
    assert metrics["transport_shots_total"]["passes"] is False
