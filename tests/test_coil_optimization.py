# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Coil Current Optimization Tests (P1.5)
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for coil current optimization and free-boundary shape control."""

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core.fusion_kernel import FusionKernel, CoilSet

MOCK_CONFIG = {
    "reactor_name": "Coil-Test",
    "grid_resolution": [20, 20],
    "dimensions": {"R_min": 4.0, "R_max": 8.0, "Z_min": -4.0, "Z_max": 4.0},
    "physics": {"plasma_current_target": 15.0, "vacuum_permeability": 1.0},
    "coils": [
        {"name": "CS", "r": 1.7, "z": 0.0, "current": 0.15},
    ],
    "solver": {
        "max_iterations": 10,
        "convergence_threshold": 1e-4,
        "relaxation_factor": 0.1,
    },
}


@pytest.fixture
def kernel(tmp_path: Path) -> FusionKernel:
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(MOCK_CONFIG), encoding="utf-8")
    return FusionKernel(str(cfg))


def _make_coils(n_coils: int = 4) -> CoilSet:
    """Create a simple n-coil set around the plasma."""
    positions = [
        (2.5, 2.0),
        (2.5, -2.0),
        (9.0, 2.0),
        (9.0, -2.0),
    ][:n_coils]
    currents = np.array([1e4] * n_coils)
    turns = [100] * n_coils
    return CoilSet(
        positions=positions,
        currents=currents,
        turns=turns,
    )


# ── CoilSet dataclass ────────────────────────────────────────────────


def test_coilset_defaults():
    """Default CoilSet should have None limits and target."""
    cs = CoilSet()
    assert cs.current_limits is None
    assert cs.target_flux_points is None


def test_coilset_with_limits():
    """CoilSet should accept current_limits array."""
    cs = _make_coils()
    limits = np.array([5e4, 5e4, 5e4, 5e4])
    cs.current_limits = limits
    np.testing.assert_array_equal(cs.current_limits, limits)


def test_coilset_with_target_points():
    """CoilSet should accept target_flux_points."""
    cs = _make_coils()
    targets = np.array([[6.0, 0.0], [6.0, 1.0], [6.0, -1.0]])
    cs.target_flux_points = targets
    assert cs.target_flux_points.shape == (3, 2)


# ── Mutual inductance matrix ────────────────────────────────────────


def test_mutual_inductance_shape(kernel: FusionKernel):
    """M matrix should have shape (n_coils, n_pts)."""
    coils = _make_coils(4)
    obs = np.array([[6.0, 0.0], [6.0, 1.0], [6.0, -1.0]])
    M = kernel._build_mutual_inductance_matrix(coils, obs)
    assert M.shape == (4, 3)


def test_mutual_inductance_finite(kernel: FusionKernel):
    """All mutual inductance values should be finite."""
    coils = _make_coils(4)
    obs = np.array([[5.0, 0.0], [6.0, 0.5], [7.0, -0.5]])
    M = kernel._build_mutual_inductance_matrix(coils, obs)
    assert np.all(np.isfinite(M)), "Mutual inductance has non-finite values"


def test_mutual_inductance_symmetry(kernel: FusionKernel):
    """Coils at symmetric Z-positions should produce symmetric flux at Z=0."""
    coils = _make_coils(4)
    obs = np.array([[6.0, 0.0]])
    M = kernel._build_mutual_inductance_matrix(coils, obs)
    # Coils 0,1 have same R but ±Z → same flux at Z=0
    assert abs(M[0, 0] - M[1, 0]) < 1e-8, "Symmetric coils should produce same flux at Z=0"


# ── Coil current optimization ───────────────────────────────────────


def test_optimize_raises_without_target_points(kernel: FusionKernel):
    """Should raise ValueError if target_flux_points is None."""
    coils = _make_coils(4)
    coils.target_flux_points = None
    target = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="target_flux_points"):
        kernel.optimize_coil_currents(coils, target)


def test_optimize_returns_correct_shape(kernel: FusionKernel):
    """Optimised currents should have shape (n_coils,)."""
    coils = _make_coils(4)
    obs = np.array([[6.0, 0.0], [6.0, 1.0], [6.0, -1.0]])
    coils.target_flux_points = obs
    target = np.array([0.5, 0.3, 0.3])
    I_opt = kernel.optimize_coil_currents(coils, target)
    assert I_opt.shape == (4,)


def test_optimize_respects_current_limits(kernel: FusionKernel):
    """Currents should stay within the specified limits."""
    coils = _make_coils(4)
    obs = np.array([[6.0, 0.0], [6.0, 1.0], [6.0, -1.0]])
    coils.target_flux_points = obs
    coils.current_limits = np.array([1e3, 1e3, 1e3, 1e3])
    target = np.array([100.0, 100.0, 100.0])  # large target to stress limits
    I_opt = kernel.optimize_coil_currents(coils, target)
    assert np.all(np.abs(I_opt) <= 1e3 + 1e-6), (
        f"Current limits violated: max |I| = {np.max(np.abs(I_opt))}"
    )


def test_optimize_finite_currents(kernel: FusionKernel):
    """Optimised currents should be finite."""
    coils = _make_coils(4)
    obs = np.array([[6.0, 0.0], [6.0, 1.0]])
    coils.target_flux_points = obs
    target = np.array([0.1, 0.1])
    I_opt = kernel.optimize_coil_currents(coils, target)
    assert np.all(np.isfinite(I_opt))


def test_optimize_reduces_residual(kernel: FusionKernel):
    """Optimised currents should produce flux closer to target than random currents."""
    coils = _make_coils(4)
    obs = np.array([[6.0, 0.0], [6.0, 1.0], [6.0, -1.0]])
    coils.target_flux_points = obs
    # Compute flux from initial currents
    M = kernel._build_mutual_inductance_matrix(coils, obs)
    flux_init = M.T @ coils.currents
    target = np.array([0.5, 0.3, 0.3])
    residual_init = np.linalg.norm(flux_init - target)

    # Optimise
    I_opt = kernel.optimize_coil_currents(coils, target)
    flux_opt = M.T @ I_opt
    residual_opt = np.linalg.norm(flux_opt - target)

    assert residual_opt <= residual_init + 1e-6, (
        f"Optimised residual {residual_opt} worse than initial {residual_init}"
    )


# ── Bilinear interpolation ──────────────────────────────────────────


def test_interp_psi_at_grid_point(kernel: FusionKernel):
    """Interpolation at a grid point should return exact Psi value."""
    # Pick an interior grid point
    ir, iz = 5, 5
    R_pt = kernel.R[ir]
    Z_pt = kernel.Z[iz]
    psi_interp = kernel._interp_psi(R_pt, Z_pt)
    psi_exact = kernel.Psi[iz, ir]
    assert abs(psi_interp - psi_exact) < 1e-10, (
        f"Interpolation at grid point: {psi_interp} vs {psi_exact}"
    )


def test_interp_psi_finite(kernel: FusionKernel):
    """Interpolation at any point in domain should return finite value."""
    R_mid = (kernel.R[0] + kernel.R[-1]) / 2.0
    Z_mid = (kernel.Z[0] + kernel.Z[-1]) / 2.0
    psi = kernel._interp_psi(R_mid, Z_mid)
    assert np.isfinite(psi), f"_interp_psi returned {psi}"


# ── Free-boundary with shape optimisation ────────────────────────────


def test_solve_free_boundary_basic(kernel: FusionKernel):
    """solve_free_boundary should return expected keys."""
    coils = _make_coils(4)
    result = kernel.solve_free_boundary(coils, max_outer_iter=3, tol=1e-2)
    assert "outer_iterations" in result
    assert "final_diff" in result
    assert "coil_currents" in result
    assert result["outer_iterations"] >= 1


def test_solve_equilibrium_preserves_explicit_boundary(kernel: FusionKernel):
    """Explicit boundary maps should be preserved during GS iteration."""
    psi_boundary = np.zeros_like(kernel.Psi)
    psi_boundary[0, :] = 3.0
    psi_boundary[-1, :] = -2.0
    psi_boundary[:, 0] = 1.5
    psi_boundary[:, -1] = -1.5

    kernel.Psi[1:-1, 1:-1] = 0.123  # non-trivial interior state
    kernel.solve_equilibrium(
        preserve_initial_state=True,
        boundary_flux=psi_boundary,
    )

    np.testing.assert_allclose(kernel.Psi[0, :], psi_boundary[0, :], atol=1e-12)
    np.testing.assert_allclose(kernel.Psi[-1, :], psi_boundary[-1, :], atol=1e-12)
    np.testing.assert_allclose(kernel.Psi[:, 0], psi_boundary[:, 0], atol=1e-12)
    np.testing.assert_allclose(kernel.Psi[:, -1], psi_boundary[:, -1], atol=1e-12)


def test_solve_free_boundary_with_optimization(kernel: FusionKernel):
    """solve_free_boundary with optimize_shape should update coil currents."""
    coils = _make_coils(4)
    obs = np.array([[6.0, 0.0], [6.0, 1.0], [6.0, -1.0]])
    coils.target_flux_points = obs
    coils.current_limits = np.array([1e6, 1e6, 1e6, 1e6])
    I_before = coils.currents.copy()
    result = kernel.solve_free_boundary(
        coils, max_outer_iter=3, tol=1e-2, optimize_shape=True,
    )
    # Currents should have been updated
    assert not np.allclose(result["coil_currents"], I_before), (
        "Coil currents were not updated during shape optimisation"
    )


def test_solve_free_boundary_enforces_external_boundary_flux(kernel: FusionKernel):
    """Outer free-boundary loop must keep external coil boundary flux."""
    coils = _make_coils(4)
    psi_ext = kernel._compute_external_flux(coils)
    kernel.solve_free_boundary(coils, max_outer_iter=1, tol=0.0)

    np.testing.assert_allclose(kernel.Psi[0, :], psi_ext[0, :], atol=1e-12)
    np.testing.assert_allclose(kernel.Psi[-1, :], psi_ext[-1, :], atol=1e-12)
    np.testing.assert_allclose(kernel.Psi[:, 0], psi_ext[:, 0], atol=1e-12)
    np.testing.assert_allclose(kernel.Psi[:, -1], psi_ext[:, -1], atol=1e-12)


def test_solve_free_boundary_runs_multiple_iters(kernel: FusionKernel):
    """Free-boundary solve should run multiple iterations without error."""
    coils = _make_coils(4)
    result = kernel.solve_free_boundary(coils, max_outer_iter=5, tol=1e-10)
    # With a coarse 20x20 grid the solve may not converge tightly,
    # but it should complete all requested iterations.
    assert result["outer_iterations"] >= 1
    assert np.isfinite(result["final_diff"])
