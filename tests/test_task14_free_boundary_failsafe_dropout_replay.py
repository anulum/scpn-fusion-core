# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Task 14 Fail-Safe Dropout Replay Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for validation/task14_free_boundary_failsafe_dropout_replay.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "task14_free_boundary_failsafe_dropout_replay.py"
SPEC = importlib.util.spec_from_file_location(
    "task14_free_boundary_failsafe_dropout_replay",
    MODULE_PATH,
)
assert SPEC and SPEC.loader
task14_free_boundary_failsafe_dropout_replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(task14_free_boundary_failsafe_dropout_replay)


def test_task14_campaign_passes_thresholds_smoke() -> None:
    report = task14_free_boundary_failsafe_dropout_replay.run_campaign(
        seed=42,
        shot_length=96,
        control_dt_s=0.05,
    )
    g = report["task14_free_boundary_failsafe_dropout_replay"]
    c = g["faulted_replay"]["summary"]
    r = g["recovery_window"]
    th = g["thresholds"]
    assert g["passes_thresholds"] is True
    assert g["faulted_replay"]["replay_deterministic"] is True
    assert c["degraded_mode_count"] >= th["min_degraded_mode_count"]
    assert c["fallback_mode_count"] >= th["min_fallback_mode_count"]
    assert r["late_max_alert_level"] <= th["max_late_alert_level"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"shot_length": 0}, "shot_length"),
        ({"control_dt_s": 0.0}, "control_dt_s"),
    ],
)
def test_task14_campaign_rejects_invalid_inputs(kwargs: dict[str, int | float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        task14_free_boundary_failsafe_dropout_replay.run_campaign(**kwargs)


def test_task14_markdown_contains_required_sections() -> None:
    report = task14_free_boundary_failsafe_dropout_replay.generate_report(
        seed=7,
        shot_length=96,
        control_dt_s=0.05,
    )
    text = task14_free_boundary_failsafe_dropout_replay.render_markdown(report)
    assert "# Task 14 Free-Boundary Fail-Safe Dropout Replay" in text
    assert "Fault Envelope" in text
    assert "Fail-Safe Degradation" in text
    assert "Recovery Window" in text
