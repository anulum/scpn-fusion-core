# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
from __future__ import annotations

import numpy as np

from scpn_fusion.core.integrated_transport_solver_runtime_utils import (
    build_cn_tridiag,
    explicit_diffusion_rhs,
    sanitize_with_fallback,
    thomas_solve,
)


def test_thomas_solve_matches_numpy_solve_for_tridiagonal() -> None:
    a = np.array([1.0, 1.0], dtype=np.float64)
    b = np.array([4.0, 4.0, 4.0], dtype=np.float64)
    c = np.array([1.0, 1.0], dtype=np.float64)
    d = np.array([7.0, 8.0, 7.0], dtype=np.float64)

    x = thomas_solve(a, b, c, d)
    A = np.array(
        [
            [4.0, 1.0, 0.0],
            [1.0, 4.0, 1.0],
            [0.0, 1.0, 4.0],
        ],
        dtype=np.float64,
    )
    ref = np.linalg.solve(A, d)
    assert np.allclose(x, ref)


def test_explicit_diffusion_rhs_zero_for_constant_profile() -> None:
    rho = np.linspace(0.1, 1.0, 8, dtype=np.float64)
    T = np.full_like(rho, 2.0)
    chi = np.full_like(rho, 1.5)
    rhs = explicit_diffusion_rhs(rho=rho, drho=float(rho[1] - rho[0]), T=T, chi=chi)
    assert rhs.shape == T.shape
    assert np.allclose(rhs[1:-1], 0.0, atol=1e-12)


def test_build_cn_tridiag_shapes_and_diagonal_signs() -> None:
    rho = np.linspace(0.1, 1.0, 6, dtype=np.float64)
    chi = np.linspace(0.2, 1.2, 6, dtype=np.float64)
    a, b, c = build_cn_tridiag(
        rho=rho,
        drho=float(rho[1] - rho[0]),
        chi=chi,
        dt=1e-3,
    )
    assert a.shape == (5,)
    assert b.shape == (6,)
    assert c.shape == (5,)
    assert np.all(b > 0.0)


def test_sanitize_with_fallback_recovers_non_finite_and_applies_bounds() -> None:
    arr = np.array([1.0, np.nan, np.inf, -2.0], dtype=np.float64)
    ref = np.array([1.0, 3.0, 4.0, 5.0], dtype=np.float64)
    out, recovered = sanitize_with_fallback(arr, ref, floor=0.0, ceil=4.5)
    assert recovered == 2
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)
    assert np.all(out <= 4.5)
