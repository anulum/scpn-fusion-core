# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Crank-Nicolson Transport Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Phase 1 verification: implicit Crank-Nicolson transport solver."""

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core.integrated_transport_solver import TransportSolver

MOCK_CONFIG = {
    "reactor_name": "CN-Test",
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
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(MOCK_CONFIG), encoding="utf-8")
    ts = TransportSolver(str(cfg))
    # Give it a reasonable parabolic initial profile
    ts.Ti = 5.0 * (1 - ts.rho**2)
    ts.Te = ts.Ti.copy()
    ts.ne = 5.0 * (1 - ts.rho**2) ** 0.5
    ts.update_transport_model(50.0)
    return ts


# ── Thomas solver unit tests ──────────────────────────────────────

def test_thomas_solve_identity():
    """Identity matrix should return RHS unchanged."""
    n = 10
    a = np.zeros(n - 1)
    b = np.ones(n)
    c = np.zeros(n - 1)
    d = np.arange(n, dtype=float)
    x = TransportSolver._thomas_solve(a, b, c, d)
    np.testing.assert_allclose(x, d, atol=1e-12)


def test_thomas_solve_tridiag():
    """Standard [-1, 2, -1] tridiagonal system against known solution."""
    n = 5
    a = -1.0 * np.ones(n - 1)
    b = 2.0 * np.ones(n)
    c = -1.0 * np.ones(n - 1)
    d = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
    x = TransportSolver._thomas_solve(a, b, c, d)
    # Verify A @ x = d
    Ax = np.zeros(n)
    Ax[0] = b[0] * x[0] + c[0] * x[1]
    for i in range(1, n - 1):
        Ax[i] = a[i - 1] * x[i - 1] + b[i] * x[i] + c[i] * x[i + 1]
    Ax[-1] = a[-1] * x[-2] + b[-1] * x[-1]
    np.testing.assert_allclose(Ax, d, atol=1e-10)


# ── CN transport tests ────────────────────────────────────────────

def test_cn_large_dt_no_nan(solver: TransportSolver):
    """dt=1.0 must produce no NaN — the whole point of CN."""
    for _ in range(10):
        T_avg, T_core = solver.evolve_profiles(dt=1.0, P_aux=50.0)
    assert np.all(np.isfinite(solver.Ti)), "CN produced NaN at dt=1.0"
    assert T_core > 0, "Core temperature should be positive"


def test_cn_energy_decreases_no_heating(solver: TransportSolver):
    """Without heating, stored energy should decrease (diffusion only)."""
    W_before = float(np.sum(solver.Ti))
    for _ in range(20):
        solver.evolve_profiles(dt=0.1, P_aux=0.0)
    W_after = float(np.sum(solver.Ti))
    assert W_after < W_before, "Energy should decrease with no heating"


def test_cn_heats_to_steady_state(solver: TransportSolver):
    """50 steps at dt=0.5 with P_aux=50 MW should heat the core above 1 keV."""
    for _ in range(50):
        solver.update_transport_model(50.0)
        solver.evolve_profiles(dt=0.5, P_aux=50.0)
    assert solver.Ti[0] > 1.0, f"Core T = {solver.Ti[0]:.3f} keV, expected > 1"


def test_cn_matches_euler_small_dt(tmp_path: Path):
    """At tiny dt, CN and forward-Euler should agree within 5%."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(MOCK_CONFIG), encoding="utf-8")

    dt = 0.0001
    P_aux = 20.0

    # CN solver
    ts_cn = TransportSolver(str(cfg))
    ts_cn.Ti = 5.0 * (1 - ts_cn.rho**2)
    ts_cn.Te = ts_cn.Ti.copy()
    ts_cn.ne = 5.0 * (1 - ts_cn.rho**2) ** 0.5
    ts_cn.update_transport_model(P_aux)
    ts_cn.evolve_profiles(dt, P_aux)

    # Manual forward-Euler for comparison
    ts_fe = TransportSolver(str(cfg))
    ts_fe.Ti = 5.0 * (1 - ts_fe.rho**2)
    ts_fe.Te = ts_fe.Ti.copy()
    ts_fe.ne = 5.0 * (1 - ts_fe.rho**2) ** 0.5
    ts_fe.update_transport_model(P_aux)

    # Compute Euler step manually
    Lh = ts_fe._explicit_diffusion_rhs(ts_fe.Ti, ts_fe.chi_i)
    S_heat, _ = ts_fe._compute_aux_heating_sources(P_aux)
    S_rad = 5.0 * ts_fe.ne * ts_fe.n_impurity * np.sqrt(ts_fe.Te + 0.1)
    euler_Ti = ts_fe.Ti + dt * (Lh + S_heat - S_rad)
    euler_Ti[0] = euler_Ti[1]
    euler_Ti[-1] = 0.1
    euler_Ti = np.maximum(0.01, euler_Ti)

    # Interior agreement (exclude boundaries and edge region where
    # the CN half-grid stencil differs from np.gradient)
    core = slice(2, len(ts_cn.rho) * 3 // 4)
    rel_diff = np.abs(ts_cn.Ti[core] - euler_Ti[core]) / (euler_Ti[core] + 1e-10)
    assert np.max(rel_diff) < 0.05, f"Max relative diff {np.max(rel_diff):.4f} > 5%"


def test_cn_backward_compatible(solver: TransportSolver):
    """run_to_steady_state() should still work with CN under the hood."""
    result = solver.run_to_steady_state(P_aux=30.0, n_steps=20, dt=0.1)
    assert "T_avg" in result
    assert "T_core" in result
    assert result["T_avg"] > 0
    assert np.all(np.isfinite(result["Ti_profile"]))


def test_cn_boundary_conditions(solver: TransportSolver):
    """Core Neumann and edge Dirichlet must be enforced after each step."""
    for _ in range(5):
        solver.evolve_profiles(dt=0.5, P_aux=40.0)
    # Neumann: T[0] == T[1]
    assert abs(solver.Ti[0] - solver.Ti[1]) < 1e-12, "Core Neumann BC violated"
    # Dirichlet: T[-1] == 0.1
    assert abs(solver.Ti[-1] - 0.1) < 1e-12, "Edge Dirichlet BC violated"
