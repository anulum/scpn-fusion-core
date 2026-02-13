# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GNEU-03 Fueling Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for ice-pellet fueling mode."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from scpn_fusion.control.fueling_mode import simulate_iter_density_control


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "gneu_03_fueling_mode.py"
SPEC = importlib.util.spec_from_file_location("gneu_03_fueling_mode", MODULE_PATH)
assert SPEC and SPEC.loader
gneu_03_fueling_mode = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gneu_03_fueling_mode)


def test_iter_density_control_reaches_1e_minus_3_target() -> None:
    result = simulate_iter_density_control(
        target_density=1.0, initial_density=0.82, steps=3000, dt_s=1e-3
    )
    assert result.final_abs_error <= 1e-3


def test_iter_density_control_is_deterministic() -> None:
    a = simulate_iter_density_control(target_density=1.0, initial_density=0.84, steps=1200)
    b = simulate_iter_density_control(target_density=1.0, initial_density=0.84, steps=1200)
    assert a.final_density == b.final_density
    assert a.final_abs_error == b.final_abs_error
    assert a.rmse == b.rmse


def test_validation_report_marks_threshold_pass() -> None:
    report = gneu_03_fueling_mode.generate_report(
        target_density=1.0, initial_density=0.82, steps=3000, dt_s=1e-3
    )
    g = report["gneu_03"]
    assert g["final_abs_error"] <= 1e-3
    assert g["passes_thresholds"] is True
