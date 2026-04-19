# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Neoclassical Module Tests
"""
Tests for neoclassical transport models and bootstrap current.
"""

from __future__ import annotations

import numpy as np

from scpn_fusion.core.neoclassical import (
    chang_hinton_chi,
    collisionality,
    sauter_bootstrap,
)


def test_collisionality_trends():
    """Verify that collisionality scales correctly with density and temperature."""
    # Baseline
    nu_base = collisionality(n_e_19=5.0, T_kev=2.0, q=2.0, R=6.2, epsilon=0.1)

    # nu_star ~ n / T^2
    nu_high_n = collisionality(n_e_19=10.0, T_kev=2.0, q=2.0, R=6.2, epsilon=0.1)
    assert nu_high_n > nu_base

    nu_high_t = collisionality(n_e_19=5.0, T_kev=5.0, q=2.0, R=6.2, epsilon=0.1)
    assert nu_high_t < nu_base


def test_chang_hinton_regimes():
    """Verify Chang-Hinton behavior in banana and Pfirsch-Schlüter regimes."""
    # Banana regime (nu_star << 1)
    # chi ~ q^2 * epsilon^-1.5 * rho_i^2 * nu_ii
    chi_banana = chang_hinton_chi(q=2.0, epsilon=0.1, nu_star=0.01, rho_i=0.01, nu_ii=100.0)

    # Pfirsch-Schlüter regime (nu_star >> 1)
    # chi is reduced by nu_star^(2/3) in Chang-Hinton formula
    chi_ps = chang_hinton_chi(q=2.0, epsilon=0.1, nu_star=100.0, rho_i=0.01, nu_ii=100.0)

    assert chi_banana > chi_ps


def test_bootstrap_flat_profiles():
    """Verify that bootstrap current vanishes when gradients are zero."""
    rho = np.linspace(0, 1, 50)
    Te = np.full(50, 5.0)
    Ti = np.full(50, 5.0)
    ne = np.full(50, 5.0)
    q = np.linspace(1.0, 3.0, 50)

    j_bs = sauter_bootstrap(rho, Te, Ti, ne, q, R0=6.2, a=2.0)

    # Inner points should be zero
    assert np.allclose(j_bs[1:-1], 0.0, atol=1e-10)


def test_bootstrap_gradient_direction():
    """Verify bootstrap current is positive for standard peaked profiles."""
    rho = np.linspace(0, 1, 50)
    Te = 5.0 * (1 - rho**2)
    Ti = 5.0 * (1 - rho**2)
    ne = 5.0 * (1 - rho**2)
    q = 1.0 + 2.0 * rho**2

    j_bs = sauter_bootstrap(rho, Te, Ti, ne, q, R0=6.2, a=2.0)

    # Peaked profiles should drive positive bootstrap current
    assert np.max(j_bs) > 0.0
    # Core should be zero due to zero gradients at rho=0
    assert j_bs[0] == 0.0
