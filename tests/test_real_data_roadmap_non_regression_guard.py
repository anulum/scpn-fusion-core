# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Regression tests for tools/real_data_roadmap_non_regression_guard.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "real_data_roadmap_non_regression_guard.py"
SPEC = importlib.util.spec_from_file_location(
    "real_data_roadmap_non_regression_guard",
    MODULE_PATH,
)
assert SPEC and SPEC.loader
real_data_roadmap_non_regression_guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(real_data_roadmap_non_regression_guard)


def test_non_regression_passes_when_metrics_hold_or_improve() -> None:
    progress = {
        "roadmap_version": "v4.0",
        "metrics": [
            {"metric": "equilibrium_files_total", "current": 19},
            {"metric": "transport_shots_total", "current": 60},
        ],
        "d3d_raw_ingestion_ready": False,
    }
    baseline = {
        "metrics": {
            "equilibrium_files_total": 18,
            "transport_shots_total": 53,
        },
        "d3d_raw_ingestion_ready": False,
    }
    summary = real_data_roadmap_non_regression_guard.evaluate(
        progress=progress,
        baseline=baseline,
    )
    assert summary["overall_pass"] is True
    assert summary["regressions"] == []


def test_non_regression_fails_on_metric_drop() -> None:
    progress = {
        "roadmap_version": "v4.0",
        "metrics": [
            {"metric": "equilibrium_files_total", "current": 17},
        ],
        "d3d_raw_ingestion_ready": False,
    }
    baseline = {
        "metrics": {"equilibrium_files_total": 18},
        "d3d_raw_ingestion_ready": False,
    }
    summary = real_data_roadmap_non_regression_guard.evaluate(
        progress=progress,
        baseline=baseline,
    )
    assert summary["overall_pass"] is False
    assert "equilibrium_files_total" in summary["regressions"]
