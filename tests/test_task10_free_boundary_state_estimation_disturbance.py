# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Task 10 State Estimation And Disturbance Rejection Tests
"""Tests for validation/task10_free_boundary_state_estimation_disturbance.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "task10_free_boundary_state_estimation_disturbance.py"
SPEC = importlib.util.spec_from_file_location(
    "task10_free_boundary_state_estimation_disturbance",
    MODULE_PATH,
)
assert SPEC and SPEC.loader
task10_free_boundary_state_estimation_disturbance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(task10_free_boundary_state_estimation_disturbance)


def test_task10_campaign_passes_thresholds_smoke() -> None:
    report = task10_free_boundary_state_estimation_disturbance.run_campaign(
        seed=42,
        shot_length=84,
        control_dt_s=0.05,
    )
    g = report["task10_free_boundary_state_estimation_disturbance"]
    faulted = g["faulted"]
    thresholds = g["thresholds"]
    assert g["passes_thresholds"] is True
    assert (
        faulted["summary"]["mean_actuator_bias_estimation_error"]
        <= thresholds["max_fault_mean_actuator_bias_estimation_error"]
    )
    assert faulted["sensor_bias_recovery_steps"] <= thresholds["max_sensor_bias_recovery_steps"]
    assert faulted["actuator_bias_recovery_steps"] <= thresholds["max_actuator_bias_recovery_steps"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"shot_length": 0}, "shot_length"),
        ({"control_dt_s": 0.0}, "control_dt_s"),
    ],
)
def test_task10_campaign_rejects_invalid_inputs(kwargs: dict[str, int | float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        task10_free_boundary_state_estimation_disturbance.run_campaign(**kwargs)


def test_task10_markdown_contains_required_sections() -> None:
    report = task10_free_boundary_state_estimation_disturbance.generate_report(
        seed=7,
        shot_length=84,
        control_dt_s=0.05,
    )
    text = task10_free_boundary_state_estimation_disturbance.render_markdown(report)
    assert "# Task 10 Free-Boundary State Estimation And Disturbance Rejection" in text
    assert "Nominal Baseline" in text
    assert "Observer Performance" in text
    assert "Disturbance Rejection" in text
