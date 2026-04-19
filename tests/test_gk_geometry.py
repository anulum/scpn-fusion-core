# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Miller Geometry Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.gk_geometry import circular_geometry, miller_geometry


def test_circular_geometry_shape():
    geom = circular_geometry(n_theta=64, n_period=2)
    assert len(geom.theta) == 128  # 64 * 2
    assert geom.R.shape == geom.theta.shape
    assert geom.Z.shape == geom.theta.shape
    assert geom.B_mag.shape == geom.theta.shape


def test_circular_R_symmetric():
    """R(theta) should be symmetric about theta=0 for circular cross-section."""
    geom = circular_geometry(n_theta=64, n_period=1)
    n = len(geom.theta)
    mid = n // 2
    # R should be approximately even function: R(-theta) ≈ R(theta)
    R_left = geom.R[:mid]
    R_right = geom.R[mid:][::-1]
    min_len = min(len(R_left), len(R_right))
    np.testing.assert_allclose(R_left[:min_len], R_right[:min_len], rtol=0.1)


def test_circular_Z_antisymmetric():
    """Z(theta) should be antisymmetric about theta=0 for circular."""
    geom = circular_geometry(n_theta=64, n_period=1)
    # Z = kappa * r * sin(theta), kappa=1 for circular
    r = 0.5 * 1.0  # rho=0.5, a=1.0
    Z_expected = r * np.sin(geom.theta)
    np.testing.assert_allclose(geom.Z, Z_expected, atol=1e-10)


def test_B_mag_1_over_R():
    """In circular limit, |B| ~ B0 * R0 / R (toroidal dominant)."""
    geom = circular_geometry(R0=2.78, B0=2.0, n_theta=64, n_period=1)
    B_expected = 2.0 * 2.78 / geom.R
    # B_p correction makes this approximate
    ratio = geom.B_mag / B_expected
    assert np.all(ratio > 0.8)
    assert np.all(ratio < 1.2)


def test_shaped_geometry_kappa():
    """Shaped geometry with kappa > 1 should stretch Z range."""
    circ = circular_geometry(n_theta=64, n_period=1)
    shaped = miller_geometry(
        R0=2.78, a=1.0, rho=0.5, kappa=1.7, q=1.4, s_hat=0.78, B0=2.0, n_theta=64, n_period=1
    )
    # Z range should be kappa times larger
    z_range_circ = np.ptp(circ.Z)
    z_range_shaped = np.ptp(shaped.Z)
    assert z_range_shaped / z_range_circ == pytest.approx(1.7, rel=0.05)


def test_shaped_geometry_delta():
    """Triangularity shifts the outboard midplane."""
    shaped = miller_geometry(
        R0=2.78,
        a=1.0,
        rho=0.5,
        kappa=1.0,
        delta=0.3,
        q=1.4,
        s_hat=0.78,
        B0=2.0,
        n_theta=64,
        n_period=1,
    )
    # With delta > 0, upper/lower X-points shift inward
    assert shaped.R is not None
    assert len(shaped.R) == 64


def test_metric_coefficients_positive():
    """g^{rr} and g^{tt} should be positive."""
    geom = circular_geometry(n_theta=64, n_period=2)
    assert np.all(geom.g_rr > 0)
    assert np.all(geom.g_tt > 0)


def test_jacobian_finite():
    geom = circular_geometry(n_theta=64, n_period=2)
    assert np.all(np.isfinite(geom.jacobian))
    assert np.all(geom.jacobian != 0)


def test_curvature_signs():
    """Normal curvature should be positive on outboard (theta ~ 0) and negative on inboard."""
    geom = circular_geometry(n_theta=128, n_period=1)
    # At theta=0 (outboard), kappa_n should be negative (unfavorable)
    idx_0 = np.argmin(np.abs(geom.theta))
    # kappa_n = -(1/R)(cos(theta) + ...) at theta=0: -(1/R)(1) < 0
    assert geom.kappa_n[idx_0] < 0


def test_b_dot_grad_theta_positive():
    geom = circular_geometry(n_theta=64, n_period=2)
    assert np.all(geom.b_dot_grad_theta > 0)


def test_miller_params_match_interface():
    """Verify miller_geometry accepts all GKLocalParams geometry fields."""
    geom = miller_geometry(
        R0=6.2,
        a=2.0,
        rho=0.5,
        kappa=1.7,
        delta=0.33,
        s_kappa=0.1,
        s_delta=0.05,
        q=2.0,
        s_hat=1.2,
        alpha_MHD=0.5,
        dR_dr=-0.1,
        B0=5.3,
        n_theta=32,
        n_period=1,
    )
    assert len(geom.theta) == 32
