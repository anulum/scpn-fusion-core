# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Linear GK Eigenvalue Solver Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.gk_eigenvalue import (
    EigenMode,
    LinearGKResult,
    solve_eigenvalue_single_ky,
    solve_linear_gk,
)
from scpn_fusion.core.gk_geometry import circular_geometry
from scpn_fusion.core.gk_species import VelocityGrid, deuterium_ion, electron


@pytest.fixture
def cyclone_geometry():
    return circular_geometry(
        R0=2.78, a=1.0, rho=0.5, q=1.4, s_hat=0.78, B0=2.0, n_theta=32, n_period=1
    )


@pytest.fixture
def cyclone_species():
    return [deuterium_ion(R_L_T=6.9, R_L_n=2.2), electron(R_L_T=6.9, R_L_n=2.2)]


@pytest.fixture
def small_vgrid():
    return VelocityGrid(n_energy=4, n_lambda=6)


def test_eigenmode_dataclass():
    m = EigenMode(k_y_rho_s=0.3, omega_r=-0.5, gamma=0.2, mode_type="ITG")
    assert m.gamma == 0.2
    assert m.phi_theta is None


def test_single_ky_returns_eigenmode(cyclone_geometry, cyclone_species, small_vgrid):
    mode = solve_eigenvalue_single_ky(
        k_y_rho_s=0.3,
        species_list=cyclone_species,
        geom=cyclone_geometry,
        vgrid=small_vgrid,
        R0=2.78,
        a=1.0,
        B0=2.0,
    )
    assert isinstance(mode, EigenMode)
    assert mode.k_y_rho_s == 0.3
    assert mode.gamma >= 0
    assert mode.mode_type in ("ITG", "TEM", "ETG", "stable")


def test_zero_gradient_stable(cyclone_geometry, small_vgrid):
    species = [deuterium_ion(R_L_T=0.0, R_L_n=0.0), electron(R_L_T=0.0, R_L_n=0.0)]
    mode = solve_eigenvalue_single_ky(
        k_y_rho_s=0.3,
        species_list=species,
        geom=cyclone_geometry,
        vgrid=small_vgrid,
        R0=2.78,
        a=1.0,
        B0=2.0,
    )
    # With zero gradients, no drive → should be stable or near-zero growth
    assert mode.gamma < 0.5  # not strongly unstable


def test_strong_gradient_unstable(cyclone_geometry, small_vgrid):
    species = [deuterium_ion(R_L_T=15.0, R_L_n=2.2), electron(R_L_T=15.0, R_L_n=2.2)]
    mode = solve_eigenvalue_single_ky(
        k_y_rho_s=0.3,
        species_list=species,
        geom=cyclone_geometry,
        vgrid=small_vgrid,
        R0=2.78,
        a=1.0,
        B0=2.0,
    )
    # Strong gradient should produce instability
    assert mode.gamma > 0


def test_solve_linear_gk_spectrum():
    result = solve_linear_gk(
        R0=2.78,
        a=1.0,
        B0=2.0,
        q=1.4,
        s_hat=0.78,
        n_ky_ion=4,
        n_ky_etg=0,
        n_theta=16,
        n_period=1,
    )
    assert isinstance(result, LinearGKResult)
    assert len(result.k_y) == 4
    assert len(result.gamma) == 4
    assert len(result.omega_r) == 4
    assert len(result.mode_type) == 4
    assert len(result.modes) == 4


def test_solve_linear_gk_gamma_max():
    result = solve_linear_gk(
        R0=2.78,
        a=1.0,
        B0=2.0,
        q=1.4,
        s_hat=0.78,
        n_ky_ion=4,
        n_ky_etg=0,
        n_theta=16,
        n_period=1,
    )
    assert result.gamma_max >= 0
    assert np.isfinite(result.gamma_max)
    assert np.isfinite(result.k_y_max)


def test_linear_gk_all_finite():
    result = solve_linear_gk(
        R0=2.78,
        a=1.0,
        B0=2.0,
        q=1.4,
        s_hat=0.78,
        n_ky_ion=4,
        n_theta=16,
        n_period=1,
    )
    assert np.all(np.isfinite(result.gamma))
    assert np.all(np.isfinite(result.omega_r))


def test_linear_gk_empty_etg():
    result = solve_linear_gk(n_ky_ion=4, n_ky_etg=0, n_theta=16, n_period=1)
    assert len(result.k_y) == 4


def test_linear_gk_with_etg():
    result = solve_linear_gk(n_ky_ion=2, n_ky_etg=2, n_theta=16, n_period=1)
    assert len(result.k_y) == 4
    # Last 2 k_y should be > 2.0 (ETG scale)
    assert result.k_y[-1] > 2.0


def test_eigenfunction_shape(cyclone_geometry, cyclone_species, small_vgrid):
    mode = solve_eigenvalue_single_ky(
        k_y_rho_s=0.3,
        species_list=cyclone_species,
        geom=cyclone_geometry,
        vgrid=small_vgrid,
        R0=2.78,
        a=1.0,
        B0=2.0,
    )
    if mode.phi_theta is not None:
        assert len(mode.phi_theta) == len(cyclone_geometry.theta)


def test_custom_species_list():
    ion = deuterium_ion(T_keV=2.0, R_L_T=6.9)
    e = electron(T_keV=2.0, R_L_T=6.9)
    result = solve_linear_gk(
        species_list=[ion, e],
        n_ky_ion=2,
        n_theta=16,
        n_period=1,
    )
    assert len(result.modes) == 2
