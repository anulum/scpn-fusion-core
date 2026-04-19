# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Task 8 Free-Boundary Supervisory Control Tests
"""Tests for validation/task8_free_boundary_supervisory_control.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "task8_free_boundary_supervisory_control.py"
SPEC = importlib.util.spec_from_file_location(
    "task8_free_boundary_supervisory_control", MODULE_PATH
)
assert SPEC and SPEC.loader
task8_free_boundary_supervisory_control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(task8_free_boundary_supervisory_control)


def test_task8_campaign_passes_thresholds_smoke() -> None:
    report = task8_free_boundary_supervisory_control.run_campaign(
        seed=42,
        shot_length=72,
        control_dt_s=0.05,
    )
    g = report["task8_free_boundary_supervisory_control"]
    c = g["closed_loop"]
    assert g["passes_thresholds"] is True
    assert c["p95_axis_error_m"] <= g["thresholds"]["max_p95_axis_error_m"]
    assert c["p95_xpoint_error_m"] <= g["thresholds"]["max_p95_xpoint_error_m"]
    assert c["stabilization_rate"] >= g["thresholds"]["min_stabilization_rate"]
    assert c["supervisor_intervention_count"] >= g["thresholds"]["min_supervisor_interventions"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"shot_length": 0}, "shot_length"),
        ({"control_dt_s": 0.0}, "control_dt_s"),
    ],
)
def test_task8_campaign_rejects_invalid_inputs(kwargs: dict[str, int | float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        task8_free_boundary_supervisory_control.run_campaign(**kwargs)


def test_task8_markdown_contains_required_sections() -> None:
    report = task8_free_boundary_supervisory_control.generate_report(
        seed=7,
        shot_length=68,
        control_dt_s=0.05,
    )
    text = task8_free_boundary_supervisory_control.render_markdown(report)
    assert "# Task 8 Free-Boundary Supervisory Control" in text
    assert "Scenario" in text
    assert "Closed-Loop Acceptance" in text
    assert "Safety Supervisor" in text
