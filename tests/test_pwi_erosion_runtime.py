# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — PWI Erosion Runtime Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Runtime summary tests for pwi_erosion demo entrypoint."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.nuclear.pwi_erosion import run_pwi_demo


def test_run_pwi_demo_returns_finite_summary_without_plot() -> None:
    summary = run_pwi_demo(
        material="Tungsten",
        redeposition_factor=0.95,
        flux_particles_m2_s=8e23,
        temp_min_eV=10.0,
        temp_max_eV=80.0,
        num_points=24,
        save_plot=False,
        verbose=False,
    )
    for key in (
        "material",
        "redep_factor",
        "flux_particles_m2_s",
        "points",
        "min_yield",
        "max_yield",
        "min_erosion_mm_year",
        "max_erosion_mm_year",
        "plot_saved",
    ):
        assert key in summary
    assert summary["plot_saved"] is False
    assert summary["plot_error"] is None
    assert summary["points"] == 24
    assert np.isfinite(summary["max_erosion_mm_year"])
    assert summary["max_yield"] >= summary["min_yield"] >= 0.0
    assert summary["max_erosion_mm_year"] >= summary["min_erosion_mm_year"] >= 0.0


def test_run_pwi_demo_is_deterministic_for_fixed_inputs() -> None:
    kwargs = dict(
        material="Tungsten",
        redeposition_factor=0.93,
        flux_particles_m2_s=9e23,
        temp_min_eV=12.0,
        temp_max_eV=90.0,
        num_points=20,
        angle_deg=50.0,
        save_plot=False,
        verbose=False,
    )
    a = run_pwi_demo(**kwargs)
    b = run_pwi_demo(**kwargs)
    for key in (
        "min_yield",
        "max_yield",
        "min_erosion_mm_year",
        "max_erosion_mm_year",
    ):
        assert a[key] == pytest.approx(b[key], rel=0.0, abs=0.0)


def test_run_pwi_demo_rejects_invalid_temperature_range() -> None:
    with pytest.raises(ValueError, match="temp_min_eV"):
        run_pwi_demo(temp_min_eV=40.0, temp_max_eV=40.0, save_plot=False, verbose=False)


def test_run_pwi_demo_rejects_invalid_num_points() -> None:
    with pytest.raises(ValueError, match="num_points"):
        run_pwi_demo(num_points=2, save_plot=False, verbose=False)
