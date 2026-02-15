# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Task 5 Disruption + Mitigation Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for validation/task5_disruption_mitigation_integration.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "task5_disruption_mitigation_integration.py"
SPEC = importlib.util.spec_from_file_location(
    "task5_disruption_mitigation_integration",
    MODULE_PATH,
)
assert SPEC and SPEC.loader
task5_disruption_mitigation_integration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(task5_disruption_mitigation_integration)


def test_task5_campaign_passes_thresholds_smoke() -> None:
    report = task5_disruption_mitigation_integration.run_campaign(
        seed=42,
        ensemble_runs=24,
        mpc_steps_per_episode=160,
    )
    g = report["task5_disruption_mitigation"]
    assert g["passes_thresholds"] is True
    assert g["physics_mitigation"]["disruption_prevention_rate"] >= 0.90
    assert g["mpc_elm_lane"]["elm_rejection_rate"] >= 0.90
    assert g["rl_multiobjective"]["multiobjective_success_rate"] >= 0.75


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"ensemble_runs": 0}, "ensemble_runs"),
        ({"mpc_steps_per_episode": 8}, "mpc_steps_per_episode"),
    ],
)
def test_task5_campaign_rejects_invalid_inputs(
    kwargs: dict[str, int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        task5_disruption_mitigation_integration.run_campaign(**kwargs)


def test_task5_markdown_contains_required_sections() -> None:
    report = task5_disruption_mitigation_integration.generate_report(
        seed=7,
        ensemble_runs=16,
        mpc_steps_per_episode=128,
    )
    text = task5_disruption_mitigation_integration.render_markdown(report)
    assert "# Task 5 Disruption + Mitigation Integration" in text
    assert "SPI / Impurity + Post-Disruption Physics" in text
    assert "MPC ELM Disturbance Rejection" in text
    assert "RL Multi-Objective Optimization" in text
