# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Analytic Solver Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Deterministic tests for analytic_solver runtime and solve paths."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control.analytic_solver import (
    AnalyticEquilibriumSolver,
    run_analytic_solver,
)


class _DummyKernel:
    """Minimal kernel exposing config/grid/vacuum field for analytic solver tests."""

    def __init__(self, _config_path: str) -> None:
        self.cfg = {
            "coils": [
                {"name": "PF1", "r": 5.9, "z": -0.2, "current": 0.0},
                {"name": "PF2", "r": 6.0, "z": 0.2, "current": 0.0},
                {"name": "PF3", "r": 6.3, "z": -0.2, "current": 0.0},
                {"name": "PF4", "r": 6.4, "z": 0.2, "current": 0.0},
            ]
        }
        self.R = np.linspace(5.6, 6.7, 51)
        self.Z = np.linspace(-0.6, 0.6, 31)
        self.dR = float(self.R[1] - self.R[0])
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)

    def calculate_vacuum_field(self) -> np.ndarray:
        psi = np.zeros_like(self.RR, dtype=np.float64)
        for coil in self.cfg["coils"]:
            cur = float(coil["current"])
            r0 = float(coil["r"])
            z0 = float(coil["z"])
            psi += cur * np.exp(
                -((self.RR - r0) ** 2 + (self.ZZ - z0) ** 2) / 0.08
            )
        return psi


def test_calculate_required_bv_returns_finite_expected_sign() -> None:
    solver = AnalyticEquilibriumSolver(
        "dummy.json", kernel_factory=_DummyKernel, verbose=False
    )
    bv = solver.calculate_required_Bv(6.2, 2.0, 15.0, beta_p=0.5, li=0.8)
    assert np.isfinite(bv)
    assert bv < 0.0


def test_solve_coil_currents_hits_target_bv_projection() -> None:
    solver = AnalyticEquilibriumSolver(
        "dummy.json", kernel_factory=_DummyKernel, verbose=False
    )
    target_bv = -0.02
    currents = solver.solve_coil_currents(target_bv, 6.2, target_Z=0.0)
    efficiencies = solver.compute_coil_efficiencies(6.2, target_Z=0.0)
    projected_bv = float(np.dot(efficiencies, currents))
    assert projected_bv == pytest.approx(target_bv, rel=1e-7, abs=1e-9)

    solver.apply_currents(currents)
    applied = np.asarray(
        [float(c["current"]) for c in solver.kernel.cfg["coils"]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(applied, currents, rtol=0.0, atol=0.0)


def test_run_analytic_solver_returns_deterministic_summary_without_write() -> None:
    kwargs = dict(
        config_path="dummy.json",
        target_r=6.2,
        target_z=0.0,
        a_minor=2.0,
        ip_target_ma=15.0,
        ridge_lambda=0.01,
        save_config=False,
        verbose=False,
        kernel_factory=_DummyKernel,
    )
    a = run_analytic_solver(**kwargs)
    b = run_analytic_solver(**kwargs)
    for key in (
        "config_path",
        "target_r_m",
        "target_z_m",
        "a_minor_m",
        "ip_target_ma",
        "required_bv_t",
        "coil_current_l2_norm",
        "max_abs_coil_current_ma",
    ):
        assert key in a
    assert a["config_path"] == "dummy.json"
    assert a["output_config_path"] is None
    assert set(a["coil_currents_ma"].keys()) == {"PF1", "PF2", "PF3", "PF4"}
    for key in (
        "required_bv_t",
        "coil_current_l2_norm",
        "max_abs_coil_current_ma",
    ):
        assert a[key] == pytest.approx(b[key], rel=0.0, abs=0.0)
    for k, v in a["coil_currents_ma"].items():
        assert float(v) == pytest.approx(float(b["coil_currents_ma"][k]), rel=0.0, abs=0.0)
