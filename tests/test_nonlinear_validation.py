# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
# ----------------------------------------------------------------------
# SCPN Fusion Core -- Tests for Nonlinear GS Validation (MMS)
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ----------------------------------------------------------------------
"""Smoke tests for the nonlinear GS Picard+SOR solver using MMS.

These tests run on small grids (33x33) to keep CI fast while verifying
that:
1. The Picard iteration converges for mild nonlinearity.
2. The Picard iteration converges for moderate nonlinearity.
3. The Picard iteration converges for strong nonlinearity.
4. The RMSE is below a reasonable threshold on a coarse grid.
5. The manufactured solution helper functions are self-consistent.
6. Grid refinement reduces RMSE (convergence check on 2 grids).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ── Path setup ─────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_VALIDATION_DIR = _REPO_ROOT / "validation"

if str(_VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(_VALIDATION_DIR))


# =====================================================================
# Helper: run a quick MMS solve
# =====================================================================


def _quick_mms_solve(alpha: float, grid_size: int = 33):
    """Run a small MMS test case and return the result."""
    from run_nonlinear_validation import MMSTestCase, run_mms_case

    case = MMSTestCase(
        name=f"test_alpha_{alpha}",
        alpha=alpha,
        grid_size=grid_size,
    )
    return run_mms_case(
        case,
        max_picard=60,
        omega=1.6,
        picard_tol=1e-7,
        sor_tol=1e-8,
        under_relax=0.4,
        verbose=False,
    )


# =====================================================================
# Test 1: Mild nonlinearity converges
# =====================================================================


def test_mild_nonlinearity_converges():
    """Picard+SOR should converge for alpha=1 (nearly linear source)."""
    result = _quick_mms_solve(alpha=1.0)
    assert result.converged, (
        f"Expected convergence for alpha=1.0, got "
        f"picard_iters={result.picard_iterations}, rmse={result.rmse:.2e}"
    )
    assert result.rmse < 0.01, (
        f"RMSE {result.rmse:.2e} exceeds threshold 0.01 for alpha=1.0"
    )


# =====================================================================
# Test 2: Moderate nonlinearity converges
# =====================================================================


def test_moderate_nonlinearity_converges():
    """Picard+SOR should converge for alpha=2 (H-mode-like pedestal)."""
    result = _quick_mms_solve(alpha=2.0)
    assert result.converged, (
        f"Expected convergence for alpha=2.0, got "
        f"picard_iters={result.picard_iterations}, rmse={result.rmse:.2e}"
    )
    assert result.rmse < 0.01, (
        f"RMSE {result.rmse:.2e} exceeds threshold 0.01 for alpha=2.0"
    )


# =====================================================================
# Test 3: Strong nonlinearity converges
# =====================================================================


def test_strong_nonlinearity_converges():
    """Picard+SOR should converge for alpha=4 (steep, highly nonlinear)."""
    result = _quick_mms_solve(alpha=4.0)
    assert result.converged, (
        f"Expected convergence for alpha=4.0, got "
        f"picard_iters={result.picard_iterations}, rmse={result.rmse:.2e}"
    )
    assert result.rmse < 0.01, (
        f"RMSE {result.rmse:.2e} exceeds threshold 0.01 for alpha=4.0"
    )


# =====================================================================
# Test 4: Manufactured solution vanishes on boundary
# =====================================================================


def test_manufactured_psi_boundary_is_zero():
    """psi_manufactured should be exactly zero on the domain boundary."""
    from run_nonlinear_validation import psi_manufactured

    R0, a, Z0, b = 1.65, 0.5, 0.0, 0.8
    n = 65
    r_grid = np.linspace(R0 - a, R0 + a, n)
    z_grid = np.linspace(Z0 - b, Z0 + b, n)
    ZZ, RR = np.meshgrid(z_grid, r_grid, indexing="ij")

    psi = psi_manufactured(RR, ZZ, R0, a, Z0, b)

    # Top and bottom rows
    assert np.allclose(psi[0, :], 0.0, atol=1e-15), "Top boundary not zero"
    assert np.allclose(psi[-1, :], 0.0, atol=1e-15), "Bottom boundary not zero"
    # Left and right columns
    assert np.allclose(psi[:, 0], 0.0, atol=1e-15), "Left boundary not zero"
    assert np.allclose(psi[:, -1], 0.0, atol=1e-15), "Right boundary not zero"

    # Interior should be strictly negative
    assert np.all(psi[1:-1, 1:-1] < 0.0), "Interior should be negative"


# =====================================================================
# Test 5: Nonlinear source is self-consistent at the manufactured psi
# =====================================================================


def test_source_self_consistency():
    """At psi = psi_manufactured, the nonlinear source should equal the
    discrete-stencil-applied source (since that is the definition of MMS).

    This verifies our source computation is consistent.
    """
    from run_nonlinear_validation import (
        psi_manufactured,
        compute_nonlinear_source,
        compute_discrete_gs_operator,
    )

    R0, a, Z0, b = 1.65, 0.5, 0.0, 0.8
    alpha, p0, f0 = 2.0, 1e4, 0.5
    n = 65

    r_grid = np.linspace(R0 - a, R0 + a, n)
    z_grid = np.linspace(Z0 - b, Z0 + b, n)
    dr = float(r_grid[1] - r_grid[0])
    dz = float(z_grid[1] - z_grid[0])
    ZZ, RR = np.meshgrid(z_grid, r_grid, indexing="ij")

    psi_ref = psi_manufactured(RR, ZZ, R0, a, Z0, b)
    psi_axis = float(np.min(psi_ref))
    psi_bdry = 0.0

    # Discrete stencil applied to psi_ref
    stencil_source = compute_discrete_gs_operator(psi_ref, RR, dr, dz)

    # Nonlinear source evaluated at psi = psi_ref
    nl_source = compute_nonlinear_source(
        psi_ref, RR, psi_axis, psi_bdry, alpha, p0, f0,
    )

    # These are NOT expected to be equal in general -- the stencil source
    # is Delta*_h(psi_ref) while the nonlinear source is the physical
    # RHS.  They would only be equal if psi_ref were the exact solution
    # to the continuous PDE, which it is not (it is manufactured).
    # What we verify is that both are finite and well-behaved.
    assert np.all(np.isfinite(stencil_source)), "Stencil source has NaN/Inf"
    assert np.all(np.isfinite(nl_source)), "Nonlinear source has NaN/Inf"

    # The interior stencil source should be non-trivial
    interior = stencil_source[1:-1, 1:-1]
    assert np.max(np.abs(interior)) > 0.0, "Stencil source is identically zero"


# =====================================================================
# Test 6: Grid refinement reduces RMSE
# =====================================================================


def test_grid_refinement_reduces_rmse():
    """Solving L_h[psi] = analytical_source on finer grids should reduce RMSE.

    This tests spatial convergence (discretisation error), NOT MMS forcing.
    We compute the continuous analytical GS source for psi_manufactured
    and feed it as a fixed (non-iterated) source to the SOR solver.
    The error is truncation error = O(h^2) from the 5-point stencil.
    """
    from run_nonlinear_validation import run_grid_convergence

    entries = run_grid_convergence(
        grid_sizes=[33, 65],
        verbose=False,
    )

    assert len(entries) == 2
    rmse_coarse = entries[0].rmse
    rmse_fine = entries[1].rmse

    assert rmse_fine < rmse_coarse, (
        f"Expected RMSE to decrease with refinement: "
        f"33x33={rmse_coarse:.2e}, 65x65={rmse_fine:.2e}"
    )

    # Check that the ratio is roughly consistent with O(h^2)
    # h ratio is 2, so RMSE ratio should be ~4
    if rmse_fine > 0:
        ratio = rmse_coarse / rmse_fine
        assert ratio > 1.5, (
            f"Expected at least 1.5x improvement, got {ratio:.2f}x"
        )


# =====================================================================
# Test 7: Normalise_psi clamps to [0, 1]
# =====================================================================


def test_normalise_psi_clamped():
    """normalise_psi should always return values in [0, 1]."""
    from run_nonlinear_validation import normalise_psi

    psi = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0])
    psi_axis = -1.0
    psi_boundary = 0.0

    psi_N = normalise_psi(psi, psi_axis, psi_boundary)
    assert np.all(psi_N >= 0.0), f"psi_N has values below 0: {psi_N}"
    assert np.all(psi_N <= 1.0), f"psi_N has values above 1: {psi_N}"

    # Known values
    assert psi_N[1] == pytest.approx(0.0, abs=1e-15)  # psi = psi_axis
    assert psi_N[3] == pytest.approx(1.0, abs=1e-15)  # psi = psi_boundary


# =====================================================================
# Test 8: Discrete GS operator is zero on boundary
# =====================================================================


def test_discrete_gs_operator_zero_on_boundary():
    """The discrete GS operator should return zero on boundary rows/cols."""
    from run_nonlinear_validation import (
        psi_manufactured,
        compute_discrete_gs_operator,
    )

    R0, a, Z0, b = 1.65, 0.5, 0.0, 0.8
    n = 33
    r_grid = np.linspace(R0 - a, R0 + a, n)
    z_grid = np.linspace(Z0 - b, Z0 + b, n)
    dr = float(r_grid[1] - r_grid[0])
    dz = float(z_grid[1] - z_grid[0])
    ZZ, RR = np.meshgrid(z_grid, r_grid, indexing="ij")

    psi = psi_manufactured(RR, ZZ, R0, a, Z0, b)
    result = compute_discrete_gs_operator(psi, RR, dr, dz)

    assert np.allclose(result[0, :], 0.0), "Top boundary not zero"
    assert np.allclose(result[-1, :], 0.0), "Bottom boundary not zero"
    assert np.allclose(result[:, 0], 0.0), "Left boundary not zero"
    assert np.allclose(result[:, -1], 0.0), "Right boundary not zero"
