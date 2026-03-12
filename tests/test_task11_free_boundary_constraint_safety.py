# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Task 11 Constraint And Safety Gate Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for validation/task11_free_boundary_constraint_safety.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "task11_free_boundary_constraint_safety.py"
SPEC = importlib.util.spec_from_file_location("task11_free_boundary_constraint_safety", MODULE_PATH)
assert SPEC and SPEC.loader
task11_free_boundary_constraint_safety = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(task11_free_boundary_constraint_safety)


def test_task11_campaign_passes_thresholds_smoke() -> None:
    report = task11_free_boundary_constraint_safety.run_campaign(
        seed=42,
        shot_length=84,
        control_dt_s=0.05,
    )
    g = report["task11_free_boundary_constraint_safety"]
    c = g["constrained_closed_loop"]
    th = g["thresholds"]
    assert g["passes_thresholds"] is True
    assert c["fallback_mode_count"] >= th["min_fallback_mode_count"]
    assert c["invariant_violation_count"] <= th["max_invariant_violation_count"]
    assert c["max_action_l1"] <= th["max_action_l1"] + 1e-9


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"shot_length": 0}, "shot_length"),
        ({"control_dt_s": 0.0}, "control_dt_s"),
    ],
)
def test_task11_campaign_rejects_invalid_inputs(kwargs: dict[str, int | float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        task11_free_boundary_constraint_safety.run_campaign(**kwargs)


def test_task11_markdown_contains_required_sections() -> None:
    report = task11_free_boundary_constraint_safety.generate_report(
        seed=7,
        shot_length=84,
        control_dt_s=0.05,
    )
    text = task11_free_boundary_constraint_safety.render_markdown(report)
    assert "# Task 11 Free-Boundary Constraint And Safety Gate" in text
    assert "Constraint Envelope" in text
    assert "Constraint-Aware Control" in text
    assert "Supervisory Safety" in text
