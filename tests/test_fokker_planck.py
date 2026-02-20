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
