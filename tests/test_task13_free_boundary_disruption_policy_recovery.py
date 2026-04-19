# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Task 13 Disruption Policy Recovery Tests
"""Tests for validation/task13_free_boundary_disruption_policy_recovery.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "task13_free_boundary_disruption_policy_recovery.py"
SPEC = importlib.util.spec_from_file_location(
    "task13_free_boundary_disruption_policy_recovery",
    MODULE_PATH,
)
assert SPEC and SPEC.loader
task13_free_boundary_disruption_policy_recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(task13_free_boundary_disruption_policy_recovery)


def test_task13_campaign_passes_thresholds_smoke() -> None:
    report = task13_free_boundary_disruption_policy_recovery.run_campaign(
        seed=42,
        shot_length=104,
        control_dt_s=0.05,
    )
    g = report["task13_free_boundary_disruption_policy_recovery"]
    c = g["policy_closed_loop"]
    r = g["recovery_window"]
    th = g["thresholds"]
    assert g["passes_thresholds"] is True
    assert c["peak_alert_level"] == th["required_peak_alert_level"]
    assert c["recovery_transition_count"] >= th["min_recovery_transition_count"]
    assert r["late_max_alert_level"] <= th["max_late_max_alert_level"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"shot_length": 0}, "shot_length"),
        ({"control_dt_s": 0.0}, "control_dt_s"),
    ],
)
def test_task13_campaign_rejects_invalid_inputs(kwargs: dict[str, int | float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        task13_free_boundary_disruption_policy_recovery.run_campaign(**kwargs)


def test_task13_markdown_contains_required_sections() -> None:
    report = task13_free_boundary_disruption_policy_recovery.generate_report(
        seed=7,
        shot_length=104,
        control_dt_s=0.05,
    )
    text = task13_free_boundary_disruption_policy_recovery.render_markdown(report)
    assert "# Task 13 Free-Boundary Disruption Policy Recovery" in text
    assert "Alert Regimes" in text
    assert "Recovery Window" in text
    assert "Closed-Loop Outcome" in text
