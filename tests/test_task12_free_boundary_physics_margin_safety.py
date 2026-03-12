# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Task 12 Physics Margin Safety Gate Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for validation/task12_free_boundary_physics_margin_safety.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "task12_free_boundary_physics_margin_safety.py"
SPEC = importlib.util.spec_from_file_location(
    "task12_free_boundary_physics_margin_safety",
    MODULE_PATH,
)
assert SPEC and SPEC.loader
task12_free_boundary_physics_margin_safety = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(task12_free_boundary_physics_margin_safety)


def test_task12_campaign_passes_thresholds_smoke() -> None:
    report = task12_free_boundary_physics_margin_safety.run_campaign(
        seed=42,
        shot_length=84,
        control_dt_s=0.05,
    )
    g = report["task12_free_boundary_physics_margin_safety"]
    c = g["physics_margin_closed_loop"]
    th = g["thresholds"]
    assert g["passes_thresholds"] is True
    assert c["physics_guard_count"] >= th["min_physics_guard_count"]
    assert c["q95_guard_count"] >= th["min_q95_guard_count"]
    assert c["beta_guard_count"] >= th["min_beta_guard_count"]
    assert c["risk_guard_count"] >= th["min_risk_guard_count"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"shot_length": 0}, "shot_length"),
        ({"control_dt_s": 0.0}, "control_dt_s"),
    ],
)
def test_task12_campaign_rejects_invalid_inputs(kwargs: dict[str, int | float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        task12_free_boundary_physics_margin_safety.run_campaign(**kwargs)


def test_task12_markdown_contains_required_sections() -> None:
    report = task12_free_boundary_physics_margin_safety.generate_report(
        seed=7,
        shot_length=84,
        control_dt_s=0.05,
    )
    text = task12_free_boundary_physics_margin_safety.render_markdown(report)
    assert "# Task 12 Free-Boundary Physics Margin Safety Gate" in text
    assert "Physics Margins" in text
    assert "Supervisory Response" in text
    assert "Closed-Loop Outcome" in text
