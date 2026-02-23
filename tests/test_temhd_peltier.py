# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — TEMHD Peltier Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Unit tests for TEMHD tridiagonal solve and stability."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.nuclear.temhd_peltier import TEMHD_Stabilizer


def test_solve_tridiagonal_matches_dense_reference() -> None:
    sim = TEMHD_Stabilizer()
    n = 7
    a = np.array([-0.3, -0.25, -0.2, -0.15, -0.1, -0.05], dtype=float)
    b = np.array([2.0, 2.1, 2.2, 2.0, 2.3, 2.15, 2.05], dtype=float)
    c = np.array([-0.2, -0.18, -0.16, -0.14, -0.12, -0.1], dtype=float)
    d = np.array([1.0, 0.5, 0.2, 0.0, 0.3, 0.7, 1.1], dtype=float)

    A = np.diag(b) + np.diag(a, k=-1) + np.diag(c, k=1)
    x_ref = np.linalg.solve(A, d)
    x = sim.solve_tridiagonal(a, b, c, d)

    assert np.allclose(x, x_ref, rtol=1e-10, atol=1e-10)


def test_solve_tridiagonal_rejects_inconsistent_sizes() -> None:
    sim = TEMHD_Stabilizer()
    with pytest.raises(ValueError, match="Invalid tridiagonal sizes"):
        sim.solve_tridiagonal(
            a=np.array([1.0, 2.0]),
            b=np.array([4.0, 4.0, 4.0]),
            c=np.array([1.0]),
            d=np.array([1.0, 2.0, 3.0]),
        )


def test_solve_tridiagonal_rejects_singular_system() -> None:
    sim = TEMHD_Stabilizer()
    with pytest.raises(ValueError, match="Singular diagonal"):
        sim.solve_tridiagonal(
            a=np.array([1.0, 1.0], dtype=float),
            b=np.array([0.0, 1.0, 1.0], dtype=float),
            c=np.array([1.0, 1.0], dtype=float),
            d=np.array([1.0, 2.0, 3.0], dtype=float),
        )


def test_step_higher_flux_increases_surface_temperature() -> None:
    low = TEMHD_Stabilizer(B_field=10.0)
    high = TEMHD_Stabilizer(B_field=10.0)

    low_t = 0.0
    high_t = 0.0
    for _ in range(12):
        low_t, _ = low.step(heat_flux_MW_m2=5.0, dt=0.2)
        high_t, _ = high.step(heat_flux_MW_m2=40.0, dt=0.2)

    assert np.isfinite(low_t)
    assert np.isfinite(high_t)
    assert high_t > low_t


def test_step_remains_finite_over_flux_ramp() -> None:
    sim = TEMHD_Stabilizer(B_field=8.0)
    for q in np.linspace(0.0, 80.0, 20):
        t_surface, k_eff_max = sim.step(heat_flux_MW_m2=float(q), dt=0.25)
        assert np.isfinite(t_surface)
        assert np.isfinite(k_eff_max)
        assert k_eff_max > 0.0


def test_step_rejects_invalid_time_step_and_flux() -> None:
    sim = TEMHD_Stabilizer()
    with pytest.raises(ValueError, match="dt must be a finite positive value"):
        sim.step(heat_flux_MW_m2=5.0, dt=0.0)
    with pytest.raises(ValueError, match="finite non-negative"):
        sim.step(heat_flux_MW_m2=-1.0, dt=0.1)


def test_step_rejects_nonfinite_temperature_state() -> None:
    sim = TEMHD_Stabilizer()
    sim.T[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        sim.step(heat_flux_MW_m2=10.0, dt=0.1)


def test_step_low_viscosity_path_remains_finite() -> None:
    sim = TEMHD_Stabilizer(B_field=10.0)
    sim.viscosity = 1e-12
    t_surface, k_eff_max = sim.step(heat_flux_MW_m2=25.0, dt=0.1)
    assert np.isfinite(t_surface)
    assert np.isfinite(k_eff_max)
    assert k_eff_max > 0.0


# --- S2-006: TEMHD solver pathological edge cases ---


def test_tridiagonal_empty_system() -> None:
    sim = TEMHD_Stabilizer()
    result = sim.solve_tridiagonal(
        a=np.array([], dtype=float),
        b=np.array([], dtype=float),
        c=np.array([], dtype=float),
        d=np.array([], dtype=float),
    )
    assert result.size == 0


def test_tridiagonal_scalar_system() -> None:
    sim = TEMHD_Stabilizer()
    result = sim.solve_tridiagonal(
        a=np.array([], dtype=float),
        b=np.array([3.0]),
        c=np.array([], dtype=float),
        d=np.array([6.0]),
    )
    assert result.shape == (1,)
    assert abs(result[0] - 2.0) < 1e-12


def test_B_field_zero_gives_base_conductivity() -> None:
    sim = TEMHD_Stabilizer(B_field=0.0)
    _, k_eff_max = sim.step(heat_flux_MW_m2=10.0, dt=0.1)
    assert abs(k_eff_max - sim.k_thermal) < 1e-6


def test_zero_flux_temperature_near_wall() -> None:
    sim = TEMHD_Stabilizer(B_field=10.0)
    t_surface = 0.0
    for _ in range(20):
        t_surface, _ = sim.step(heat_flux_MW_m2=0.0, dt=0.1)
    assert abs(t_surface - sim.T_wall) < 1.0


def test_large_dt_remains_finite() -> None:
    sim = TEMHD_Stabilizer(B_field=10.0)
    t_surface, k_eff_max = sim.step(heat_flux_MW_m2=20.0, dt=10.0)
    assert np.isfinite(t_surface)
    assert np.isfinite(k_eff_max)
