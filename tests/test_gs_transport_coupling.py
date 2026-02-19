# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GS ↔ Transport Self-Consistency Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for the self-consistent GS <-> transport coupling loop.

Verifies that:
  - run_self_consistent() produces decreasing psi residuals
  - map_profiles_to_2d() is invoked and updates J_phi
  - run_to_steady_state(self_consistent=False) remains backward-compatible
  - run_to_steady_state(self_consistent=True) delegates correctly
  - Output dict has the expected keys and sensible values
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from scpn_fusion.core.integrated_transport_solver import TransportSolver

MOCK_CONFIG = {
    "reactor_name": "GS-Transport-Test",
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
def solver(tmp_path: Path) -> TransportSolver:
    """Create a TransportSolver with reasonable initial profiles."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(MOCK_CONFIG), encoding="utf-8")
    ts = TransportSolver(str(cfg))
    ts.Ti = 5.0 * (1 - ts.rho**2)
    ts.Te = ts.Ti.copy()
    ts.ne = 5.0 * (1 - ts.rho**2) ** 0.5
    ts.update_transport_model(50.0)
    return ts


# ── run_self_consistent() tests ─────────────────────────────────────


def test_self_consistent_returns_expected_keys(solver: TransportSolver):
    """run_self_consistent() must return all documented keys."""
    result = solver.run_self_consistent(
        P_aux=30.0, n_inner=5, n_outer=2, dt=0.01, psi_tol=1e-3,
    )
    expected_keys = {
        "T_avg", "T_core", "tau_e",
        "n_outer_converged", "psi_residuals",
        "Ti_profile", "ne_profile", "converged",
    }
    assert expected_keys == set(result.keys()), (
        f"Missing keys: {expected_keys - set(result.keys())}"
    )


def test_self_consistent_psi_residuals_recorded(solver: TransportSolver):
    """Each outer iteration should record a psi residual.

    The number of recorded residuals equals n_outer_converged (the loop
    may converge and break before n_outer iterations are exhausted).
    """
    n_outer = 5
    result = solver.run_self_consistent(
        P_aux=30.0, n_inner=5, n_outer=n_outer, dt=0.01, psi_tol=1e-10,
    )
    n_done = result["n_outer_converged"]
    assert len(result["psi_residuals"]) == n_done, (
        f"Expected {n_done} residuals, got {len(result['psi_residuals'])}"
    )
    assert n_done >= 1, "Should have done at least 1 outer iteration"
    assert n_done <= n_outer, "Should not exceed n_outer"


def test_self_consistent_residuals_finite(solver: TransportSolver):
    """All psi residuals should be finite and non-negative."""
    result = solver.run_self_consistent(
        P_aux=30.0, n_inner=10, n_outer=3, dt=0.01, psi_tol=1e-10,
    )
    for r in result["psi_residuals"]:
        assert np.isfinite(r), f"Non-finite psi residual: {r}"
        assert r >= 0, f"Negative psi residual: {r}"


def test_self_consistent_profiles_finite(solver: TransportSolver):
    """Output Ti and ne profiles must contain no NaN/Inf."""
    result = solver.run_self_consistent(
        P_aux=30.0, n_inner=10, n_outer=2, dt=0.01,
    )
    assert np.all(np.isfinite(result["Ti_profile"])), "Ti has NaN/Inf"
    assert np.all(np.isfinite(result["ne_profile"])), "ne has NaN/Inf"


def test_self_consistent_positive_temperatures(solver: TransportSolver):
    """Core temperature and average must be positive after coupling."""
    result = solver.run_self_consistent(
        P_aux=30.0, n_inner=10, n_outer=2, dt=0.01,
    )
    assert result["T_avg"] > 0, f"T_avg={result['T_avg']}"
    assert result["T_core"] > 0, f"T_core={result['T_core']}"


def test_self_consistent_convergence_flag(solver: TransportSolver):
    """With a very loose tolerance the loop should converge."""
    result = solver.run_self_consistent(
        P_aux=30.0, n_inner=10, n_outer=10, dt=0.01, psi_tol=1e2,
    )
    assert result["converged"] is True, "Should converge with very loose tol"
    assert result["n_outer_converged"] == 1, "Should converge on first iter"


def test_self_consistent_psi_residual_trend(solver: TransportSolver):
    """Psi residuals should generally decrease (or at least not blow up).

    We check that the last residual is no larger than 10x the first.
    The loop may converge early on a small test grid.
    """
    result = solver.run_self_consistent(
        P_aux=30.0, n_inner=20, n_outer=5, dt=0.01, psi_tol=1e-10,
    )
    residuals = result["psi_residuals"]
    assert len(residuals) >= 1, "Should have at least one residual"
    if len(residuals) >= 2:
        # The residuals should not blow up
        assert residuals[-1] <= residuals[0] * 10.0, (
            f"Residuals blew up: first={residuals[0]:.4e}, "
            f"last={residuals[-1]:.4e}"
        )


# ── map_profiles_to_2d() is actually called ─────────────────────────


def test_map_profiles_called_during_self_consistent(solver: TransportSolver):
    """map_profiles_to_2d() must be called once per outer iteration.

    The count should equal n_outer_converged (the loop may converge early).
    """
    with patch.object(
        solver, "map_profiles_to_2d", wraps=solver.map_profiles_to_2d
    ) as mock_map:
        result = solver.run_self_consistent(
            P_aux=30.0, n_inner=5, n_outer=5, dt=0.01, psi_tol=1e-10,
        )
        expected = result["n_outer_converged"]
        assert mock_map.call_count == expected, (
            f"Expected {expected} calls to map_profiles_to_2d, "
            f"got {mock_map.call_count}"
        )
        assert mock_map.call_count >= 1, "Should call map_profiles_to_2d at least once"


def test_map_profiles_updates_jphi(solver: TransportSolver):
    """map_profiles_to_2d() should change J_phi from its initial state."""
    J_phi_before = solver.J_phi.copy()
    # Run some transport to change profiles
    for _ in range(10):
        solver.update_transport_model(30.0)
        solver.evolve_profiles(0.01, 30.0)
    solver.map_profiles_to_2d()
    J_phi_after = solver.J_phi.copy()
    # J_phi should have been updated
    assert not np.allclose(J_phi_before, J_phi_after), (
        "J_phi unchanged after map_profiles_to_2d()"
    )


# ── solve_equilibrium() is called ──────────────────────────────────


def test_solve_equilibrium_called_during_self_consistent(
    solver: TransportSolver,
):
    """solve_equilibrium() must be called once per outer iteration."""
    with patch.object(
        solver, "solve_equilibrium", wraps=solver.solve_equilibrium
    ) as mock_eq:
        solver.run_self_consistent(
            P_aux=30.0, n_inner=5, n_outer=2, dt=0.01, psi_tol=1e-10,
        )
        assert mock_eq.call_count == 2, (
            f"Expected 2 calls to solve_equilibrium, got {mock_eq.call_count}"
        )


# ── Backward compatibility ──────────────────────────────────────────


def test_run_to_steady_state_default_unchanged(solver: TransportSolver):
    """run_to_steady_state() without self_consistent keeps old behaviour."""
    result = solver.run_to_steady_state(P_aux=30.0, n_steps=20, dt=0.1)
    assert "T_avg" in result
    assert "T_core" in result
    assert "n_steps" in result
    assert result["T_avg"] > 0
    assert np.all(np.isfinite(result["Ti_profile"]))
    # Should NOT have self-consistent keys
    assert "psi_residuals" not in result
    assert "converged" not in result
    assert "n_outer_converged" not in result


def test_run_to_steady_state_self_consistent_delegates(
    solver: TransportSolver,
):
    """self_consistent=True should delegate to run_self_consistent()."""
    with patch.object(
        solver, "run_self_consistent", wraps=solver.run_self_consistent
    ) as mock_sc:
        result = solver.run_to_steady_state(
            P_aux=30.0,
            dt=0.01,
            self_consistent=True,
            sc_n_inner=5,
            sc_n_outer=2,
            sc_psi_tol=1e-3,
        )
        mock_sc.assert_called_once()
        _, kwargs = mock_sc.call_args
        assert kwargs["P_aux"] == 30.0
        assert kwargs["n_inner"] == 5
        assert kwargs["n_outer"] == 2
        assert kwargs["dt"] == 0.01
        assert kwargs["psi_tol"] == 1e-3
        assert kwargs["enforce_numerical_recovery"] is False
        assert kwargs["max_numerical_recoveries"] is None
    # Result should have self-consistent keys
    assert "psi_residuals" in result
    assert "converged" in result


def test_run_to_steady_state_adaptive_still_works(solver: TransportSolver):
    """adaptive=True should still work (not broken by new parameter)."""
    result = solver.run_to_steady_state(
        P_aux=30.0, n_steps=10, dt=0.01, adaptive=True, tol=1e-2,
    )
    assert "dt_history" in result
    assert "error_history" in result
    assert result["T_avg"] > 0


# ── external_profile_mode correctness ───────────────────────────────


def test_external_profile_mode_is_true(solver: TransportSolver):
    """TransportSolver must have external_profile_mode=True so that
    solve_equilibrium uses the J_phi set by map_profiles_to_2d()."""
    assert solver.external_profile_mode is True
