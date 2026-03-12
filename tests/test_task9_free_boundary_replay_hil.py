# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Task 9 Free-Boundary Replay And HIL Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for validation/task9_free_boundary_replay_hil.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "task9_free_boundary_replay_hil.py"
SPEC = importlib.util.spec_from_file_location("task9_free_boundary_replay_hil", MODULE_PATH)
assert SPEC and SPEC.loader
task9_free_boundary_replay_hil = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(task9_free_boundary_replay_hil)


def test_task9_campaign_passes_thresholds_smoke() -> None:
    report = task9_free_boundary_replay_hil.run_campaign(
        seed=42,
        shot_length=72,
        control_dt_s=0.05,
        hil_steps=256,
    )
    g = report["task9_free_boundary_replay_hil"]
    assert g["passes_thresholds"] is True
    assert g["nominal_replay"]["replay_deterministic"] is True
    assert g["faulted_watchdog"]["watchdog_trip"] is True
    assert (
        g["faulted_watchdog"]["summary"]["failsafe_trip_count"]
        >= g["thresholds"]["min_fault_failsafe_trip_count"]
    )
    assert (
        g["faulted_watchdog"]["summary"]["max_risk_score"]
        >= g["thresholds"]["min_fault_max_risk_score"]
    )
    assert g["hil_compatibility"]["p95_us"] <= g["thresholds"]["max_hil_p95_us"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"shot_length": 0}, "shot_length"),
        ({"control_dt_s": 0.0}, "control_dt_s"),
        ({"hil_steps": 0}, "hil_steps"),
    ],
)
def test_task9_campaign_rejects_invalid_inputs(kwargs: dict[str, int | float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        task9_free_boundary_replay_hil.run_campaign(**kwargs)


def test_task9_markdown_contains_required_sections() -> None:
    report = task9_free_boundary_replay_hil.generate_report(
        seed=7,
        shot_length=68,
        control_dt_s=0.05,
        hil_steps=256,
    )
    text = task9_free_boundary_replay_hil.render_markdown(report)
    assert "# Task 9 Free-Boundary Replay And HIL Gate" in text
    assert "Replay Determinism" in text
    assert "Watchdog And Fail-Safe" in text
    assert "HIL Compatibility" in text
    assert "Faulted failsafe trip count" in text
