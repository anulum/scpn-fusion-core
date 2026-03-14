# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GK Interface Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.gk_interface import GKLocalParams, GKOutput, GKSolverBase


def test_gk_local_params_defaults():
    p = GKLocalParams(R_L_Ti=6.9, R_L_Te=6.9, R_L_ne=2.2, q=1.4, s_hat=0.78)
    assert p.R_L_Ti == 6.9
    assert p.kappa == 1.0
    assert p.delta == 0.0
    assert p.R0 == 6.2
    assert p.epsilon == 0.1


def test_gk_local_params_all_fields():
    p = GKLocalParams(
        R_L_Ti=8.0,
        R_L_Te=7.0,
        R_L_ne=3.0,
        q=2.0,
        s_hat=1.2,
        alpha_MHD=0.5,
        Te_Ti=1.1,
        Z_eff=2.0,
        nu_star=0.5,
        beta_e=0.03,
        epsilon=0.15,
        kappa=1.7,
        delta=0.33,
        rho=0.5,
        R0=6.2,
        a=2.0,
        B0=5.3,
        n_e=10.0,
        T_e_keV=8.0,
        T_i_keV=7.3,
    )
    assert p.kappa == 1.7
    assert p.T_i_keV == 7.3


def test_gk_output_defaults():
    o = GKOutput(chi_i=1.0, chi_e=0.8, D_e=0.3)
    assert o.D_i == 0.0
    assert o.dominant_mode == "stable"
    assert o.converged is True
    assert len(o.gamma) == 0


def test_gk_output_with_spectrum():
    ky = np.linspace(0.1, 2.0, 16)
    gamma = np.random.default_rng(0).random(16)
    omega = np.random.default_rng(1).random(16) - 0.5
    o = GKOutput(
        chi_i=2.5,
        chi_e=1.8,
        D_e=0.5,
        gamma=gamma,
        omega_r=omega,
        k_y=ky,
        dominant_mode="ITG",
        converged=True,
    )
    assert o.dominant_mode == "ITG"
    assert len(o.gamma) == 16
    assert len(o.k_y) == 16


def test_gk_solver_base_is_abstract():
    with pytest.raises(TypeError):
        GKSolverBase()  # type: ignore[abstract]


class _MockSolver(GKSolverBase):
    """Minimal concrete solver for testing the base class contract."""

    def __init__(self, chi_i: float = 1.0):
        self._chi_i = chi_i

    def prepare_input(self, params):
        from pathlib import Path

        return Path("/tmp/mock")

    def run(self, input_path, *, timeout_s=30.0):
        return GKOutput(chi_i=self._chi_i, chi_e=self._chi_i * 0.8, D_e=0.1)

    def is_available(self):
        return True


def test_solver_base_run_from_params():
    solver = _MockSolver(chi_i=3.0)
    params = GKLocalParams(R_L_Ti=6.9, R_L_Te=6.9, R_L_ne=2.2, q=1.4, s_hat=0.78)
    result = solver.run_from_params(params)
    assert result.chi_i == 3.0
    assert result.chi_e == pytest.approx(2.4)


def test_solver_is_available():
    solver = _MockSolver()
    assert solver.is_available() is True
