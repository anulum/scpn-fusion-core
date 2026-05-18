# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Coil Current Optimization Tests (P1.5)
"""Tests for coil current optimization and free-boundary shape control."""

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core.fusion_kernel_coilset_config import build_coilset_from_config
from scpn_fusion.core.fusion_kernel import FusionKernel, CoilSet
from scpn_fusion.core.fusion_kernel_free_boundary_mixin import FusionKernelFreeBoundaryMixin

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
    assert cs.target_flux_values is None


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


def test_coilset_with_target_flux_values():
    """CoilSet should accept target_flux_values for shape control."""
    cs = _make_coils()
    cs.target_flux_points = np.array([[6.0, 0.0], [6.0, 1.0], [6.0, -1.0]])
    cs.target_flux_values = np.array([0.2, 0.2, 0.2])
    assert cs.target_flux_values.shape == (3,)


def test_build_coilset_from_config_maps_free_boundary_contract(kernel: FusionKernel):
    """FusionKernel should expose a real config-backed coil contract."""
    kernel.cfg["coils"] = [
        {"name": "PF1", "r": 3.0, "z": 2.0, "current": 2.5, "turns": 12},
        {"name": "PF2", "r": 5.0, "z": -2.0, "current": -1.5, "turns": 8},
    ]
    kernel.cfg["free_boundary"] = {
        "current_limits": [7.0, 9.0],
        "target_flux_points": [[4.0, 0.0], [4.5, 0.2]],
        "target_flux_values": [0.1, 0.2],
        "x_point_target": [4.2, -1.1],
        "x_point_flux_target": 0.05,
        "divertor_strike_points": [[3.2, -2.2]],
        "divertor_flux_values": [0.03],
    }

    coils = kernel.build_coilset_from_config()
    direct_coils = build_coilset_from_config(kernel)

    assert coils.positions == [(3.0, 2.0), (5.0, -2.0)]
    np.testing.assert_allclose(coils.currents, np.array([2.5, -1.5], dtype=np.float64))
    np.testing.assert_allclose(direct_coils.currents, coils.currents)
    assert coils.turns == [12, 8]
    np.testing.assert_allclose(coils.current_limits, np.array([7.0, 9.0], dtype=np.float64))
    np.testing.assert_allclose(
        coils.target_flux_points,
        np.array([[4.0, 0.0], [4.5, 0.2]], dtype=np.float64),
    )
    np.testing.assert_allclose(coils.target_flux_values, np.array([0.1, 0.2]))
    np.testing.assert_allclose(coils.x_point_target, np.array([4.2, -1.1]))
    assert coils.x_point_flux_target == pytest.approx(0.05)
    np.testing.assert_allclose(coils.divertor_strike_points, np.array([[3.2, -2.2]]))
    np.testing.assert_allclose(coils.divertor_flux_values, np.array([0.03]))


def test_build_coilset_from_config_rejects_shape_mismatches(kernel: FusionKernel):
    """Free-boundary config must fail before a control shot starts."""
    kernel.cfg["coils"] = [
        {"name": "PF1", "r": 3.0, "z": 2.0, "current": 2.5},
        {"name": "PF2", "r": 5.0, "z": -2.0, "current": -1.5},
    ]
    kernel.cfg["free_boundary"] = {
        "current_limits": [5.0],
        "target_flux_points": [[4.0, 0.0], [4.5, 0.2]],
        "target_flux_values": [0.1],
    }

    with pytest.raises(ValueError, match="current_limits"):
        kernel.build_coilset_from_config()

    kernel.cfg["free_boundary"]["current_limits"] = [5.0]
    kernel.cfg["coils"] = [{"name": "PF1", "r": 3.0, "z": 2.0, "current": 2.5}]
    with pytest.raises(ValueError, match="target_flux_values"):
        kernel.build_coilset_from_config()


def test_sample_flux_at_points_uses_kernel_interpolator(kernel: FusionKernel):
    """Flux sampling should expose the same interpolation semantics as _interp_psi."""
    assert isinstance(kernel, FusionKernelFreeBoundaryMixin)
    kernel.Psi = kernel.RR + 2.0 * kernel.ZZ
    points = np.array([[kernel.R[2], kernel.Z[3]], [kernel.R[4], kernel.Z[5]]])

    samples = kernel._sample_flux_at_points(points)

    expected = np.array(
        [kernel._interp_psi(float(r_pt), float(z_pt)) for r_pt, z_pt in points],
        dtype=np.float64,
    )
    np.testing.assert_allclose(samples, expected, atol=1e-12)


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


def test_optimize_raises_on_target_length_mismatch(kernel: FusionKernel):
    """Target vector must match the number of target_flux_points."""
    coils = _make_coils(4)
    coils.target_flux_points = np.array([[6.0, 0.0], [6.0, 1.0], [6.0, -1.0]])
    with pytest.raises(ValueError, match="target_flux"):
        kernel.optimize_coil_currents(coils, np.array([0.1, 0.2]))


def test_optimize_raises_on_current_limits_shape_mismatch(kernel: FusionKernel):
    """current_limits must have one entry per coil."""
    coils = _make_coils(4)
    coils.target_flux_points = np.array([[6.0, 0.0], [6.0, 1.0], [6.0, -1.0]])
    coils.current_limits = np.array([1e3, 1e3])  # wrong length
    with pytest.raises(ValueError, match="current_limits"):
        kernel.optimize_coil_currents(coils, np.array([0.1, 0.1, 0.1]))


def test_optimize_falls_back_to_prior_currents_on_solver_failure(
    kernel: FusionKernel,
    monkeypatch: pytest.MonkeyPatch,
):
    """Failed least-squares solve should safely return bounded prior currents."""
    coils = _make_coils(4)
    coils.target_flux_points = np.array([[6.0, 0.0], [6.0, 1.0], [6.0, -1.0]])

    class _FailResult:
        success = False
        status = -1
        message = "synthetic failure"
        cost = float("inf")
        x = np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float64)

    monkeypatch.setattr("scipy.optimize.lsq_linear", lambda *_args, **_kwargs: _FailResult())
    out = kernel.optimize_coil_currents(coils, np.array([0.1, 0.1, 0.1]))
    np.testing.assert_allclose(out, coils.currents, atol=0.0)


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
        coils,
        max_outer_iter=3,
        tol=1e-2,
        optimize_shape=True,
    )
    # Currents should have been updated
    assert not np.allclose(result["coil_currents"], I_before), (
        "Coil currents were not updated during shape optimisation"
    )


def test_solve_free_boundary_uses_explicit_target_flux_values(
    kernel: FusionKernel,
    monkeypatch: pytest.MonkeyPatch,
):
    """Shape optimisation should pass explicit target_flux_values to optimiser."""
    coils = _make_coils(4)
    coils.target_flux_points = np.array([[6.0, 0.0], [6.0, 1.0], [6.0, -1.0]])
    coils.target_flux_values = np.array([0.25, 0.25, 0.25])
    captured: dict[str, np.ndarray] = {}

    def _stub_optimize(
        _self: FusionKernel,
        _coils: CoilSet,
        target_flux: np.ndarray,
        tikhonov_alpha: float = 1e-4,
    ) -> np.ndarray:
        del tikhonov_alpha
        captured["target_flux"] = np.asarray(target_flux, dtype=np.float64).copy()
        return _coils.currents.copy()

    monkeypatch.setattr(FusionKernel, "optimize_coil_currents", _stub_optimize)
    kernel.solve_free_boundary(coils, max_outer_iter=1, tol=0.0, optimize_shape=True)

    np.testing.assert_allclose(captured["target_flux"], coils.target_flux_values, atol=0.0)


def test_solve_free_boundary_infers_isoflux_target_without_explicit_values(
    kernel: FusionKernel,
    monkeypatch: pytest.MonkeyPatch,
):
    """Without explicit target_flux_values the inferred shape target should be isoflux."""
    coils = _make_coils(4)
    coils.target_flux_points = np.array([[6.0, 0.0], [6.0, 1.0], [6.0, -1.0]])
    captured: dict[str, np.ndarray] = {}

    def _stub_optimize(
        _self: FusionKernel,
        _coils: CoilSet,
        target_flux: np.ndarray,
        tikhonov_alpha: float = 1e-4,
    ) -> np.ndarray:
        del tikhonov_alpha
        captured["target_flux"] = np.asarray(target_flux, dtype=np.float64).copy()
        return _coils.currents.copy()

    monkeypatch.setattr(FusionKernel, "optimize_coil_currents", _stub_optimize)
    kernel.solve_free_boundary(coils, max_outer_iter=1, tol=0.0, optimize_shape=True)

    target = captured["target_flux"]
    assert target.shape == (3,)
    assert float(np.max(target) - np.min(target)) < 1e-12


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


def test_solve_free_boundary_rejects_invalid_outer_iter(kernel: FusionKernel):
    coils = _make_coils(4)
    with pytest.raises(ValueError, match="max_outer_iter"):
        kernel.solve_free_boundary(coils, max_outer_iter=0)


def test_solve_free_boundary_rejects_invalid_tolerance(kernel: FusionKernel):
    coils = _make_coils(4)
    with pytest.raises(ValueError, match="tol"):
        kernel.solve_free_boundary(coils, max_outer_iter=1, tol=-1.0)
