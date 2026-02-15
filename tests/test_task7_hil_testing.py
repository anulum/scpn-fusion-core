# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Task 7 HIL Testing Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for validation/task7_hil_testing.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "task7_hil_testing.py"
SPEC = importlib.util.spec_from_file_location("task7_hil_testing", MODULE_PATH)
assert SPEC and SPEC.loader
task7_hil_testing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(task7_hil_testing)


def test_task7_campaign_passes_thresholds_smoke() -> None:
    report = task7_hil_testing.run_campaign(
        seed=42,
        hil_steps=220,
        control_dt_s=8e-4,
    )
    g = report["task7_hil_testing"]
    assert g["passes_thresholds"] is True
    assert g["hil_closed_loop"]["p95_latency_ms"] <= 1.0
    assert g["hil_closed_loop"]["stabilization_rate"] >= 0.95
    assert g["determinism"]["bitstream"]["all_pass"] is True
    assert g["determinism"]["replay_deterministic"] is True


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"hil_steps": 0}, "hil_steps"),
        ({"control_dt_s": 0.0}, "control_dt_s"),
    ],
)
def test_task7_campaign_rejects_invalid_inputs(
    kwargs: dict[str, int | float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        task7_hil_testing.run_campaign(**kwargs)


def test_task7_markdown_contains_required_sections() -> None:
    report = task7_hil_testing.generate_report(
        seed=7,
        hil_steps=180,
        control_dt_s=7e-4,
    )
    text = task7_hil_testing.render_markdown(report)
    assert "# Task 7 Hardware-In-The-Loop Testing" in text
    assert "Hardware Profile" in text
    assert "Synthetic HIL Closed Loop" in text
    assert "Determinism Gate" in text
