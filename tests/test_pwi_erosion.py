# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — PWI Erosion Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Unit tests for sputtering yield and erosion-rate behavior."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.nuclear.pwi_erosion import SputteringPhysics


def test_yield_is_zero_below_threshold() -> None:
    pwi = SputteringPhysics("Tungsten")
    assert pwi.calculate_yield(100.0) == 0.0


def test_yield_increases_with_energy_above_threshold() -> None:
    pwi = SputteringPhysics("Tungsten")
    y1 = pwi.calculate_yield(250.0)
    y2 = pwi.calculate_yield(500.0)
    y3 = pwi.calculate_yield(1200.0)
    assert 0.0 <= y1 <= y2 <= y3


def test_yield_increases_toward_grazing_angles() -> None:
    pwi = SputteringPhysics("Tungsten")
    base_energy = 900.0
    y0 = pwi.calculate_yield(base_energy, angle_deg=0.0)
    y45 = pwi.calculate_yield(base_energy, angle_deg=45.0)
    y80 = pwi.calculate_yield(base_energy, angle_deg=80.0)
    assert 0.0 <= y0 <= y45 <= y80


def test_yield_angle_clamping_is_stable() -> None:
    pwi = SputteringPhysics("Tungsten")
    energy = 900.0
    assert pwi.calculate_yield(energy, angle_deg=-15.0) == pwi.calculate_yield(
        energy, angle_deg=0.0
    )
    assert pwi.calculate_yield(energy, angle_deg=95.0) == pwi.calculate_yield(
        energy, angle_deg=89.0
    )


def test_erosion_reduces_with_higher_redeposition() -> None:
    low_redep = SputteringPhysics("Tungsten", redeposition_factor=0.8)
    high_redep = SputteringPhysics("Tungsten", redeposition_factor=0.98)

    flux = 1e24
    ion_temp = 60.0
    r1 = low_redep.calculate_erosion_rate(flux, ion_temp)
    r2 = high_redep.calculate_erosion_rate(flux, ion_temp)
    assert r1["Erosion_mm_year"] > r2["Erosion_mm_year"]
    assert r1["Net_Flux"] > r2["Net_Flux"]


def test_redeposition_factor_is_bounded_and_controls_net_flux() -> None:
    low = SputteringPhysics("Tungsten", redeposition_factor=-1.0)
    high = SputteringPhysics("Tungsten", redeposition_factor=3.0)
    assert low.redeposition_factor == 0.0
    assert abs(high.redeposition_factor - 0.999) < 1e-12

    out_low = low.calculate_erosion_rate(1e24, 60.0)
    out_high = high.calculate_erosion_rate(1e24, 60.0)
    assert out_low["Net_Flux"] >= out_high["Net_Flux"]
    assert out_low["Impurity_Source"] == out_low["Net_Flux"]
    assert out_high["Impurity_Source"] == out_high["Net_Flux"]


def test_erosion_outputs_finite_and_nonnegative() -> None:
    pwi = SputteringPhysics("Tungsten", redeposition_factor=0.95)
    out = pwi.calculate_erosion_rate(flux_particles_m2_s=5e23, T_ion_eV=40.0)
    for key in ("Yield", "E_impact", "Net_Flux", "Erosion_mm_year", "Impurity_Source", "Redeposition"):
        assert key in out
        assert np.isfinite(out[key])
    assert out["Yield"] >= 0.0
    assert out["Net_Flux"] >= 0.0
    assert out["Erosion_mm_year"] >= 0.0


# S2-005: PWI erosion angle-energy invariants and redeposition bounds


def test_f_alpha_capped_at_5() -> None:
    pwi = SputteringPhysics("Tungsten")
    y_normal = pwi.calculate_yield(900.0, angle_deg=0.0)
    y_grazing = pwi.calculate_yield(900.0, angle_deg=89.0)
    assert y_normal > 0.0
    assert y_grazing / y_normal <= 5.0 + 1e-9


def test_yield_zero_at_threshold_energy() -> None:
    pwi = SputteringPhysics("Tungsten")
    assert pwi.calculate_yield(200.0) == 0.0


def test_redeposition_zero_identity() -> None:
    pwi = SputteringPhysics("Tungsten", redeposition_factor=0.0)
    flux = 1e24
    out = pwi.calculate_erosion_rate(flux, T_ion_eV=60.0)
    expected_net = out["Yield"] * flux
    assert abs(out["Net_Flux"] - expected_net) < 1e-6


def test_yield_rejects_nonfinite_energy() -> None:
    pwi = SputteringPhysics("Tungsten")
    with pytest.raises(ValueError, match="E_ion_eV must be finite"):
        pwi.calculate_yield(np.nan)


def test_yield_rejects_nonfinite_angle() -> None:
    pwi = SputteringPhysics("Tungsten")
    with pytest.raises(ValueError, match="angle_deg must be finite"):
        pwi.calculate_yield(500.0, angle_deg=np.inf)
