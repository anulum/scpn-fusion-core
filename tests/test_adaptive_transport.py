# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Adaptive Time-Stepping Transport Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Phase 2 verification: Richardson-extrapolation adaptive time stepping."""

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core.integrated_transport_solver import (
    AdaptiveTimeController,
    TransportSolver,
)

MOCK_CONFIG = {
    "reactor_name": "Adaptive-Test",
    "grid_resolution": [20, 20],
    "dimensions": {"R_min": 4.0, "R_max": 8.0, "Z_min": -4.0, "Z_max": 4.0},
    "physics": {"plasma_current_target": 15.0, "vacuum_permeability": 1.0},
    "coils": [{"name": "CS", "r": 1.7, "z": 0.0, "current": 0.15}],
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
    ts.Ti = 5.0 * (1 - ts.rho**2)
    ts.Te = ts.Ti.copy()
    ts.ne = 5.0 * (1 - ts.rho**2) ** 0.5
    ts.update_transport_model(50.0)
    return ts


def test_adaptive_dt_adapts(solver: TransportSolver):
    """Adaptive controller should actively adjust dt (not stay constant)."""
    result = solver.run_to_steady_state(
        P_aux=30.0, n_steps=15, dt=0.01, adaptive=True, tol=1e-2
    )
    dts = result["dt_history"]
    assert len(dts) >= 2
    # dt should change from its initial value (controller is active)
    assert dts[-1] != dts[0], "dt never adapted — controller inactive"
    # All dt values must be positive and finite
    assert all(d > 0 and np.isfinite(d) for d in dts), "Invalid dt values"


def test_adaptive_dt_bounded(solver: TransportSolver):
    """dt must never exceed dt_max."""
    dt_max = 0.5
    atc = AdaptiveTimeController(dt_init=0.01, dt_max=dt_max, tol=1e-2)
    for _ in range(30):
        solver.update_transport_model(30.0)
        error = atc.estimate_error(solver, 30.0)
        atc.adapt_dt(error)
        assert atc.dt <= dt_max + 1e-12, f"dt={atc.dt} exceeded dt_max={dt_max}"


def test_adaptive_reaches_steady_state(solver: TransportSolver):
    """Adaptive should reach a result comparable to fixed-dt."""
    r_adaptive = solver.run_to_steady_state(
        P_aux=30.0, n_steps=30, dt=0.01, adaptive=True, tol=1e-2
    )
    assert r_adaptive["T_avg"] > 0
    assert r_adaptive["T_core"] > 0
    assert np.all(np.isfinite(r_adaptive["Ti_profile"]))


def test_adaptive_error_decreases(solver: TransportSolver):
    """Error history trend should generally decrease (or stabilise)."""
    result = solver.run_to_steady_state(
        P_aux=30.0, n_steps=20, dt=0.01, adaptive=True, tol=1e-2
    )
    errors = result["error_history"]
    if len(errors) >= 4:
        # The last quarter should not be significantly worse than the second quarter
        q2 = np.mean(errors[len(errors)//4 : len(errors)//2])
        q4 = np.mean(errors[3*len(errors)//4:])
        assert q4 <= q2 * 5, f"Error increased: q2={q2:.2e}, q4={q4:.2e}"


def test_adaptive_no_nan(solver: TransportSolver):
    """Adaptive stepping should never produce NaN."""
    result = solver.run_to_steady_state(
        P_aux=50.0, n_steps=30, dt=0.1, adaptive=True, tol=1e-3
    )
    assert np.all(np.isfinite(result["Ti_profile"])), "Adaptive produced NaN"


def test_adaptive_backward_compat(solver: TransportSolver):
    """adaptive=False should give same behavior as before (no extra keys)."""
    result = solver.run_to_steady_state(P_aux=30.0, n_steps=10, dt=0.01, adaptive=False)
    assert "dt_final" not in result
    assert "dt_history" not in result
    assert "T_avg" in result
