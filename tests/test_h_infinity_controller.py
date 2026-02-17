"""Regression tests for Riccati-based H-infinity controller synthesis."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control.h_infinity_controller import (
    HInfinityController,
    get_radial_robust_controller,
)


def _reference_plant() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the canonical 2-state vertical-stability test plant."""
    A = np.array(
        [
            [0.0, 1.0],
            [100.0**2, -10.0],
        ],
        dtype=float,
    )
    B1 = np.array([[0.0], [0.5]], dtype=float)
    B2 = np.array([[0.0], [1.0]], dtype=float)
    C1 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=float)
    C2 = np.array([[1.0, 0.0]], dtype=float)
    return A, B1, B2, C1, C2


def test_h_infinity_synthesis_produces_stable_controller() -> None:
    ctrl = get_radial_robust_controller(gamma_growth=100.0)
    assert ctrl.is_stable
    assert np.isfinite(ctrl.gamma)
    assert ctrl.gamma > 1.0
    assert ctrl.F.shape == (1, 2)
    assert ctrl.L_gain.shape == (2, 1)


def test_h_infinity_riccati_residuals_are_small() -> None:
    ctrl = get_radial_robust_controller(gamma_growth=100.0)
    res_x, res_y = ctrl.riccati_residual_norms()
    assert res_x < 1.0e-3
    assert res_y < 5.0e-2


def test_h_infinity_step_stays_finite() -> None:
    ctrl = get_radial_robust_controller(gamma_growth=100.0)
    for k in range(200):
        error = 0.2 * np.exp(-0.02 * k)
        u = ctrl.step(float(error), dt=1.0e-3)
        assert np.isfinite(u)
    assert np.all(np.isfinite(ctrl.state))


def test_h_infinity_exposes_feasibility_diagnostics() -> None:
    ctrl = get_radial_robust_controller(gamma_growth=100.0)
    assert isinstance(ctrl.robust_feasible, bool)
    assert np.isfinite(ctrl.spectral_radius_xy)
    assert ctrl.spectral_radius_xy >= 0.0


def test_h_infinity_rejects_invalid_gamma() -> None:
    A, B1, B2, C1, C2 = _reference_plant()
    with pytest.raises(ValueError, match="gamma must be finite and > 1.0"):
        HInfinityController(A, B1, B2, C1, C2, gamma=1.0)
    with pytest.raises(ValueError, match="gamma must be finite and > 1.0"):
        HInfinityController(A, B1, B2, C1, C2, gamma=float("nan"))


def test_h_infinity_rejects_nonfinite_runtime_inputs() -> None:
    ctrl = get_radial_robust_controller(gamma_growth=100.0)
    with pytest.raises(ValueError, match="error must be finite"):
        ctrl.step(float("nan"), dt=1.0e-3)
    with pytest.raises(ValueError, match="dt must be finite and > 0"):
        ctrl.step(0.1, dt=0.0)


def test_h_infinity_strict_mode_rejects_infeasible_solution() -> None:
    A, B1, B2, C1, C2 = _reference_plant()
    with pytest.raises(ValueError, match="spectral feasibility condition failed"):
        HInfinityController(
            A,
            B1,
            B2,
            C1,
            C2,
            enforce_robust_feasibility=True,
        )


def test_h_infinity_strict_mode_accepts_feasible_synthesis(monkeypatch: pytest.MonkeyPatch) -> None:
    def _feasible_synthesize(
        self: HInfinityController, gamma: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = np.zeros((self.n, self.n), dtype=float)
        y = np.zeros((self.n, self.n), dtype=float)
        f = np.zeros((self.m, self.n), dtype=float)
        l_gain = np.zeros((self.n, self.p), dtype=float)
        return x, y, f, l_gain

    monkeypatch.setattr(HInfinityController, "_synthesize", _feasible_synthesize)
    A, B1, B2, C1, C2 = _reference_plant()
    ctrl = HInfinityController(
        A,
        B1,
        B2,
        C1,
        C2,
        enforce_robust_feasibility=True,
    )
    assert ctrl.robust_feasible is True
    assert ctrl.spectral_radius_xy == 0.0
