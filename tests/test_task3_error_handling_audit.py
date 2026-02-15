# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Task 3 Error Handling Audit Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for validation/task3_error_handling_audit.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "task3_error_handling_audit.py"
SPEC = importlib.util.spec_from_file_location("task3_error_handling_audit", MODULE_PATH)
assert SPEC and SPEC.loader
task3_error_handling_audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(task3_error_handling_audit)


def test_task3_campaign_passes_thresholds_smoke() -> None:
    out = task3_error_handling_audit.run_campaign(
        seed=42,
        episodes=4,
        sim_seconds=1200,
        dt_s=1.0,
        injected_error_rate=0.20,
        error_check_interval=10,
        flux_scenarios=12,
    )
    assert out["passes_thresholds"] is True
    assert out["soc_monte_carlo"]["mean_uptime_rate"] >= 0.99
    assert out["flux_liveness"]["liveness_pass_rate"] >= 0.99
    assert out["flux_liveness"]["petri_activity_rate"] >= 0.99


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"episodes": 0}, "episodes"),
        ({"sim_seconds": 10}, "sim_seconds"),
        ({"dt_s": 0.0}, "dt_s"),
        ({"injected_error_rate": 1.2}, "injected_error_rate"),
        ({"error_check_interval": 0}, "error_check_interval"),
        ({"bitflip_fraction": -0.1}, "bitflip_fraction"),
        ({"overheat_downtime_steps": 0}, "overheat_downtime_steps"),
        ({"noise_probability": 1.1}, "noise_probability"),
        ({"flux_scenarios": 1}, "flux_scenarios"),
    ],
)
def test_task3_campaign_rejects_invalid_inputs(
    kwargs: dict[str, float | int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        task3_error_handling_audit.run_campaign(**kwargs)


def test_task3_render_markdown_contains_required_sections() -> None:
    report = task3_error_handling_audit.generate_report(
        seed=7,
        episodes=2,
        sim_seconds=600,
        dt_s=1.0,
        injected_error_rate=0.20,
        error_check_interval=10,
        flux_scenarios=8,
    )
    text = task3_error_handling_audit.render_markdown(report)
    assert "# Task 3 Error Handling Audit" in text
    assert "SOC Monte Carlo (RL Agent)" in text
    assert "3D Flux + Petri Liveness" in text
