# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Stellarator Geometry Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.stellarator_geometry import (
    StellaratorConfig,
    effective_ripple,
    iota_profile,
    iss04_scaling,
    stellarator_flux_surface,
    stellarator_neoclassical_chi,
    w7x_config,
)


# ── W7-X preset ──────────────────────────────────────────────────────


def test_w7x_config_parameters():
    cfg = w7x_config()
    assert cfg.N_fp == 5
    assert pytest.approx(5.5) == cfg.R0
    assert cfg.a == pytest.approx(0.53)
    assert pytest.approx(2.5) == cfg.B0
    assert cfg.iota_0 == pytest.approx(0.87)
    assert cfg.iota_a == pytest.approx(1.0)
    assert cfg.name == "W7-X"


# ── Flux surface shape ───────────────────────────────────────────────


def test_flux_surface_shapes():
    cfg = w7x_config()
    R, Z, B = stellarator_flux_surface(cfg, s=0.25, n_theta=32, n_phi=24)
    assert R.shape == (32, 24)
    assert Z.shape == (32, 24)
    assert B.shape == (32, 24)


def test_flux_surface_B_varies_with_phi():
    """Stellarator |B| must vary toroidally (not axisymmetric)."""
    cfg = w7x_config()
    _, _, B = stellarator_flux_surface(cfg, s=0.5, n_theta=64, n_phi=64)
    # Variation along phi at a fixed theta index
    B_phi_slice = B[0, :]
    assert np.std(B_phi_slice) > 1e-4


def test_flux_surface_B_varies_with_theta():
    """Stellarator |B| varies poloidally as well."""
    cfg = w7x_config()
    _, _, B = stellarator_flux_surface(cfg, s=0.5, n_theta=64, n_phi=64)
    B_theta_slice = B[:, 0]
    assert np.std(B_theta_slice) > 1e-4


def test_flux_surface_R_positive():
    cfg = w7x_config()
    R, _, _ = stellarator_flux_surface(cfg, s=0.9, n_theta=64, n_phi=64)
    assert np.all(R > 0)


# ── Iota profile ─────────────────────────────────────────────────────


def test_iota_profile_range():
    cfg = w7x_config()
    s_arr = np.linspace(0, 1, 50)
    iota = iota_profile(cfg, s_arr)
    assert np.all(iota >= 0.8)
    assert np.all(iota <= 1.1)


def test_iota_profile_monotonic():
    cfg = w7x_config()
    s_arr = np.linspace(0, 1, 100)
    iota = iota_profile(cfg, s_arr)
    assert np.all(np.diff(iota) >= 0)


# ── Effective ripple ─────────────────────────────────────────────────


def test_effective_ripple_positive_bounded():
    cfg = w7x_config()
    for s_val in [0.1, 0.25, 0.5, 0.75, 1.0]:
        eps = effective_ripple(cfg, s_val)
        assert 0 < eps < 1


def test_effective_ripple_increases_with_s():
    cfg = w7x_config()
    eps_inner = effective_ripple(cfg, 0.1)
    eps_outer = effective_ripple(cfg, 0.9)
    assert eps_outer > eps_inner


def test_axisymmetric_limit_ripple_vanishes():
    """Large N_fp with zero mirror ratio -> epsilon_eff = 0."""
    cfg = StellaratorConfig(N_fp=100, mirror_ratio=0.0, R0=6.0, a=2.0, B0=5.0)
    eps = effective_ripple(cfg, 0.5)
    assert eps == pytest.approx(0.0, abs=1e-15)


# ── ISS04 scaling ────────────────────────────────────────────────────


def test_iss04_positive():
    cfg = w7x_config()
    tau = iss04_scaling(cfg, n_e=5.0, P_heat=5.0)
    assert tau > 0
    assert np.isfinite(tau)


def test_iss04_scales_with_density():
    """tau_E ~ n_e^0.54: doubling density should increase tau."""
    cfg = w7x_config()
    tau_lo = iss04_scaling(cfg, n_e=3.0, P_heat=5.0)
    tau_hi = iss04_scaling(cfg, n_e=6.0, P_heat=5.0)
    assert tau_hi > tau_lo


def test_iss04_scales_with_power():
    """tau_E ~ P^-0.61: more power -> lower confinement time (degradation)."""
    cfg = w7x_config()
    tau_lo_p = iss04_scaling(cfg, n_e=5.0, P_heat=3.0)
    tau_hi_p = iss04_scaling(cfg, n_e=5.0, P_heat=10.0)
    assert tau_lo_p > tau_hi_p


def test_iss04_w7x_order_of_magnitude():
    """W7-X at 5 MW, 5e19 m^-3 should give tau_E ~ 0.01-1 s."""
    cfg = w7x_config()
    tau = iss04_scaling(cfg, n_e=5.0, P_heat=5.0)
    assert 0.01 < tau < 1.0


# ── Neoclassical chi ─────────────────────────────────────────────────


def test_neoclassical_chi_positive_finite():
    cfg = w7x_config()
    chi = stellarator_neoclassical_chi(cfg, s=0.5, T_keV=2.0, n_e19=5.0)
    assert chi > 0
    assert np.isfinite(chi)


def test_neoclassical_chi_increases_with_temperature():
    """1/nu regime: chi grows with T (v_th^2 / nu ~ T^(7/2))."""
    cfg = w7x_config()
    chi_cold = stellarator_neoclassical_chi(cfg, s=0.5, T_keV=0.5, n_e19=5.0)
    chi_hot = stellarator_neoclassical_chi(cfg, s=0.5, T_keV=5.0, n_e19=5.0)
    assert chi_hot > chi_cold
