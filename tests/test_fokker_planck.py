# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Fokker-Planck RE Solver Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────

import numpy as np
import pytest
from scpn_fusion.control.fokker_planck_re import FokkerPlanckSolver

def test_fokker_planck_initialization():
    solver = FokkerPlanckSolver(np_grid=50, p_max=10.0)
    assert solver.p.shape == (50,)
    assert solver.f.shape == (50,)
    assert solver.time == 0.0

def test_fokker_planck_coefficients_finite():
    solver = FokkerPlanckSolver()
    A, D, Fc = solver.compute_coefficients(E_field=1.0, n_e=5e19, Z_eff=1.5, T_e_eV=5000.0)
    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(D))
    assert Fc > 0.0

def test_fokker_planck_step_conserves_positivity():
    solver = FokkerPlanckSolver()
    # Inject some population
    solver.f[10] = 1.0e10
    
    # Evolve
    # High E field to drive acceleration
    state = solver.step(dt=1e-5, E_field=10.0, n_e=5e19, T_e_eV=5000.0, Z_eff=1.0)
    
    assert np.all(state.f >= 0.0)
    assert state.n_re > 0.0
    assert state.current_re > 0.0


# S2-001: Fokker-Planck 1/p² divergence guard


def test_drag_finite_near_zero_momentum():
    """With thermal regularization, drag should be finite even near p → 0."""
    solver = FokkerPlanckSolver(np_grid=100, p_max=10.0)
    A, D, Fc = solver.compute_coefficients(
        E_field=1.0, n_e=5e19, Z_eff=1.5, T_e_eV=5000.0
    )
    assert np.all(np.isfinite(A)), "Advection has non-finite values near p=0"
    assert np.all(np.isfinite(D)), "Diffusion has non-finite values near p=0"


def test_drag_regularized_at_thermal_speed():
    """Drag at the smallest momentum should be bounded (not divergent)."""
    solver = FokkerPlanckSolver(np_grid=200, p_max=10.0)
    A, D, Fc = solver.compute_coefficients(
        E_field=1.0, n_e=5e19, Z_eff=1.5, T_e_eV=100.0  # lower Te
    )
    # At the first grid point (smallest p), drag should be large but finite
    assert np.isfinite(A[0])
    # The total advection (drag + synchrotron) should be bounded, not divergent
    # With regularization the value is O(10^5) rather than Inf
    assert abs(A[0]) < 1e8, f"A[0]={A[0]:.2e} is unreasonably large"


def test_fokker_planck_rejects_invalid_constructor_inputs():
    with pytest.raises(ValueError, match="np_grid"):
        FokkerPlanckSolver(np_grid=4, p_max=10.0)
    with pytest.raises(ValueError, match="p_max"):
        FokkerPlanckSolver(np_grid=32, p_max=0.0)


def test_fokker_planck_step_rejects_invalid_dt():
    solver = FokkerPlanckSolver()
    with pytest.raises(ValueError, match="dt"):
        solver.step(dt=0.0, E_field=10.0, n_e=5e19, T_e_eV=5000.0, Z_eff=1.0)
