# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Geometric Multigrid Solver Tests
"""Tests for the free-function geometric multigrid solver."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.multigrid_solve import (
    mg_residual,
    multigrid_solve,
    prolongate_bilinear,
    residual_linf,
    restrict_full_weight,
    validate_sor_omega,
)


def _solovev_problem(nr: int = 65, nz: int = 65) -> tuple:
    r_min, r_max, z_min, z_max = 1.2, 2.2, -0.5, 0.5
    r = np.linspace(r_min, r_max, nr)
    z = np.linspace(z_min, z_max, nz)
    rr, zz = np.meshgrid(r, z)
    source = -rr * np.exp(-((rr - 1.7) ** 2 + zz**2) / 0.05)
    return source, r_min, r_max, z_min, z_max, nr, nz


def test_multigrid_solve_converges_and_reports_metadata() -> None:
    source, r_min, r_max, z_min, z_max, nr, nz = _solovev_problem()
    psi_bc = np.zeros((nz, nr))
    psi, residual, n_cycles, converged = multigrid_solve(
        source, psi_bc, r_min, r_max, z_min, z_max, nr, nz, tol=1e-6, max_cycles=200
    )
    assert converged is True
    assert residual < 1e-6
    assert 0 < n_cycles < 50
    assert np.all(np.isfinite(psi))


def test_multigrid_solve_preserves_nonzero_dirichlet_boundary() -> None:
    source, r_min, r_max, z_min, z_max, nr, nz = _solovev_problem()
    psi_bc = np.zeros((nz, nr))
    psi_bc[0, :] = 0.4
    psi_bc[-1, :] = 0.4
    psi_bc[:, 0] = 0.4
    psi_bc[:, -1] = 0.4
    psi, _residual, _n_cycles, _converged = multigrid_solve(
        source, psi_bc, r_min, r_max, z_min, z_max, nr, nz, tol=1e-6, max_cycles=200
    )
    assert float(psi[0, 0]) == 0.4
    assert float(psi[-1, -1]) == 0.4
    np.testing.assert_array_equal(psi[0, :], 0.4)
    np.testing.assert_array_equal(psi[:, -1], 0.4)


def test_multigrid_solve_residual_is_linf() -> None:
    source, r_min, r_max, z_min, z_max, nr, nz = _solovev_problem()
    psi_bc = np.zeros((nz, nr))
    psi, residual, _n_cycles, _converged = multigrid_solve(
        source, psi_bc, r_min, r_max, z_min, z_max, nr, nz, tol=1e-6, max_cycles=200
    )
    r_grid, _ = np.meshgrid(np.linspace(r_min, r_max, nr), np.linspace(z_min, z_max, nz))
    dr = (r_max - r_min) / (nr - 1)
    dz = (z_max - z_min) / (nz - 1)
    assert residual == pytest.approx(residual_linf(psi, source, r_grid, dr, dz), rel=1e-12)


@pytest.mark.parametrize(
    ("source_shape", "psi_shape"),
    [((10, 12), (65, 65)), ((65, 65), (10, 12))],
)
def test_multigrid_solve_rejects_shape_mismatch(
    source_shape: tuple[int, int], psi_shape: tuple[int, int]
) -> None:
    with pytest.raises(ValueError, match="shape"):
        multigrid_solve(np.zeros(source_shape), np.zeros(psi_shape), 1.0, 2.0, -1.0, 1.0, 65, 65)


@pytest.mark.parametrize("tol", [0.0, -1e-6, np.inf, np.nan])
def test_multigrid_solve_rejects_bad_tol(tol: float) -> None:
    with pytest.raises(ValueError, match="tol"):
        multigrid_solve(
            np.zeros((33, 33)), np.zeros((33, 33)), 1.0, 9.0, -5.0, 5.0, 33, 33, tol=tol
        )


def test_multigrid_solve_rejects_bad_max_cycles() -> None:
    with pytest.raises(ValueError, match="max_cycles"):
        multigrid_solve(
            np.zeros((33, 33)), np.zeros((33, 33)), 1.0, 9.0, -5.0, 5.0, 33, 33, max_cycles=0
        )


def test_residual_linf_zero_on_exact_discrete_solution() -> None:
    nr = nz = 17
    r = np.linspace(1.0, 2.0, nr)
    z = np.linspace(-0.5, 0.5, nz)
    rr, zz = np.meshgrid(r, z)
    dr = float(r[1] - r[0])
    dz = float(z[1] - z[0])
    psi = 0.03125 * rr**4 - 0.125 * zz**2 + 0.05 * rr**2 * zz**2
    source = mg_residual(psi, np.zeros_like(psi), rr, dr, dz)  # = L*[psi]
    assert residual_linf(psi, source, rr, dr, dz) < 1e-12


def test_restrict_and_prolongate_shape_contract() -> None:
    fine = np.random.default_rng(0).standard_normal((9, 9))
    coarse = restrict_full_weight(fine)
    assert coarse.shape == (5, 5)
    back = prolongate_bilinear(coarse, 9, 9)
    assert back.shape == (9, 9)


@pytest.mark.parametrize("omega", [0.0, 0.999999, 2.0, np.inf, np.nan])
def test_validate_sor_omega_rejects_out_of_range(omega: float) -> None:
    with pytest.raises(ValueError, match="omega"):
        validate_sor_omega(omega)


def test_residual_linf_is_zero_when_interior_is_empty() -> None:
    """A grid with no interior points has no residual to measure."""
    psi = np.zeros((2, 2), dtype=np.float64)
    r_grid = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float64)
    assert residual_linf(psi, psi, r_grid, 1.0, 1.0) == 0.0
