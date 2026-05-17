# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — VMEC-lite Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.vmec_lite import (
    AxisymmetricTokamakBoundary,
    SpectralBasis,
    StellaratorBoundary,
    VMECLiteSolver,
)


def test_axisymmetric_tokamak_convergence():
    solver = VMECLiteSolver(n_s=11, m_pol=2, n_tor=0, n_fp=1)

    b_R, b_Z = AxisymmetricTokamakBoundary.from_parameters(R0=6.2, a=2.0, kappa=1.7, delta=0.33)
    solver.set_boundary(b_R, b_Z)

    p_prof = np.linspace(1e5, 0.0, 11)
    iota_prof = np.linspace(1.0, 0.3, 11)  # q = 1 to 3.3

    solver.set_profiles(p_prof, iota_prof)

    res = solver.solve(max_iter=50, tol=1e-3)

    assert res.iterations > 0
    # Just verify output structure
    assert res.R_mn.shape == (11, solver.basis.n_modes)
    assert np.all(np.isfinite(res.residual_history))
    assert res.residual_history[-1] <= res.residual_history[0]


def test_stellarator_convergence():
    solver = VMECLiteSolver(n_s=11, m_pol=2, n_tor=1, n_fp=5)

    b_R, b_Z = StellaratorBoundary.w7x_standard()
    solver.set_boundary(b_R, b_Z)

    p_prof = np.linspace(5e4, 0.0, 11)
    iota_prof = np.ones(11) * 0.9  # W7-X is roughly shearless

    solver.set_profiles(p_prof, iota_prof)

    res = solver.solve(max_iter=50, tol=1e-3)

    assert res.iterations > 0
    assert res.Z_mn.shape == (11, solver.basis.n_modes)


def test_basis_evaluation():
    solver = VMECLiteSolver(m_pol=1, n_tor=0, n_fp=1)
    # Modes: (0,0), (1,0)
    coeffs = np.array([6.2, 2.0])

    theta = np.array([0.0, np.pi / 2, np.pi])
    zeta = np.zeros(3)

    R_val = solver.basis.evaluate(coeffs, theta, zeta, is_sin=False)

    assert np.isclose(R_val[0], 8.2)  # 6.2 + 2.0 * cos(0)
    assert np.isclose(R_val[1], 6.2)  # 6.2 + 2.0 * cos(pi/2)
    assert np.isclose(R_val[2], 4.2)  # 6.2 + 2.0 * cos(pi)


def test_spectral_basis_rejects_invalid_resolution():
    with pytest.raises(ValueError, match="m_pol"):
        SpectralBasis(m_pol=-1, n_tor=0, n_fp=1)
    with pytest.raises(ValueError, match="n_tor"):
        SpectralBasis(m_pol=1, n_tor=-1, n_fp=1)
    with pytest.raises(ValueError, match="n_fp"):
        SpectralBasis(m_pol=1, n_tor=0, n_fp=0)


def test_solver_rejects_invalid_resolution():
    with pytest.raises(ValueError, match="n_s"):
        VMECLiteSolver(n_s=2)


def test_boundary_rejects_unknown_or_nonfinite_modes():
    solver = VMECLiteSolver(n_s=7, m_pol=1, n_tor=0, n_fp=1)

    with pytest.raises(ValueError, match="mode"):
        solver.set_boundary({(9, 0): 1.0}, {})
    with pytest.raises(ValueError, match="finite"):
        solver.set_boundary({(0, 0): float("nan")}, {})


def test_profiles_reject_invalid_inputs():
    solver = VMECLiteSolver(n_s=7, m_pol=1, n_tor=0, n_fp=1)

    with pytest.raises(ValueError, match="pressure"):
        solver.set_profiles(np.array([1.0, np.nan]), np.array([1.0, 0.8]))
    with pytest.raises(ValueError, match="iota"):
        solver.set_profiles(np.array([1.0, 0.0]), np.array([1.0, 0.0]))


def test_solve_requires_configured_boundary_and_profiles():
    solver = VMECLiteSolver(n_s=7, m_pol=2, n_tor=0, n_fp=1)

    with pytest.raises(ValueError, match="boundary"):
        solver.solve()

    b_R, b_Z = AxisymmetricTokamakBoundary.from_parameters(R0=6.2, a=1.0, kappa=1.5, delta=0.2)
    solver.set_boundary(b_R, b_Z)
    with pytest.raises(ValueError, match="profiles"):
        solver.solve()


def test_solve_preserves_fixed_boundary_coefficients():
    solver = VMECLiteSolver(n_s=9, m_pol=2, n_tor=1, n_fp=5)
    b_R, b_Z = StellaratorBoundary.w7x_standard()
    solver.set_boundary(b_R, b_Z)
    solver.set_profiles(np.linspace(2e4, 0.0, 9), np.full(9, 0.9))

    expected_R = solver.R_mn[-1].copy()
    expected_Z = solver.Z_mn[-1].copy()
    res = solver.solve(max_iter=30, tol=1e-4)

    np.testing.assert_allclose(res.R_mn[-1], expected_R, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(res.Z_mn[-1], expected_Z, rtol=0.0, atol=0.0)
