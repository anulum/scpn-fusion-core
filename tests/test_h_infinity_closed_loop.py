# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — H-Infinity Controller Closed-Loop Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ──────────────────────────────────────────────────────────────────────
"""
Closed-loop integration tests for the H-infinity robust controller.

Verifies that the Tustin-discretised observer + state-feedback loop
stabilises the vertical instability plant model and rejects step
disturbances.

The plant simulation uses exact ZOH discretisation (matrix exponential)
to avoid forward-Euler instability on the stiff plant model.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from scpn_fusion.control.h_infinity_controller import (
    HInfinityController,
    get_radial_robust_controller,
)


def _zoh_discretize(A: np.ndarray, B: np.ndarray, dt: float):
    """Exact zero-order-hold discretisation via matrix exponential.

    Returns (Ad, Bd) such that  x_{k+1} = Ad @ x_k + Bd @ u_k.
    """
    n = A.shape[0]
    m = B.shape[1]
    # Augmented matrix method: [A B; 0 0] * dt → expm → extract blocks
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A * dt
    M[:n, n:] = B * dt
    eM = expm(M)
    return eM[:n, :n], eM[:n, n:]


# ── Factory tests ────────────────────────────────────────────────────

class TestFactory:
    """Smoke tests for get_radial_robust_controller."""

    def test_synthesis_succeeds(self):
        ctrl = get_radial_robust_controller()
        assert ctrl.gamma > 1.0
        assert ctrl.is_stable

    def test_robust_feasibility(self):
        ctrl = get_radial_robust_controller()
        assert ctrl.robust_feasibility_margin() > 0

    def test_riccati_residuals_small(self):
        ctrl = get_radial_robust_controller()
        res_x, res_y = ctrl.riccati_residual_norms()
        # Plant eigenvalue ~100 produces O(1e-2) residuals due to scaling
        assert res_x < 0.1, f"X Riccati residual too large: {res_x}"
        assert res_y < 0.1, f"Y Riccati residual too large: {res_y}"

    def test_gain_margin_positive(self):
        ctrl = get_radial_robust_controller()
        assert ctrl.gain_margin_db > 0, "Closed-loop gain margin must be positive"


# ── Closed-loop stability tests ─────────────────────────────────────

class TestClosedLoop:
    """Integration tests simulating the full observer-feedback loop.

    The plant is discretised with exact ZOH (matrix exponential) to
    avoid forward-Euler instability on the stiff vertical stability
    model (eigenvalue ~+100).
    """

    @pytest.fixture()
    def ctrl(self):
        return get_radial_robust_controller()

    def test_step_response_converges(self, ctrl):
        """A constant error should be driven to zero by the controller."""
        dt = 0.05
        n_steps = 600  # 30 seconds
        A, B2, C2 = ctrl.A, ctrl.B2, ctrl.C2
        Ad, Bd = _zoh_discretize(A, B2, dt)

        x_plant = np.array([0.1, 0.0])  # initial offset
        errors = []

        for _ in range(n_steps):
            y = (C2 @ x_plant).item()
            u = ctrl.step(y, dt)
            x_plant = Ad @ x_plant + Bd.ravel() * u
            errors.append(abs(y))

        # Must converge: final error < 1% of initial
        assert errors[-1] < 0.01 * errors[0], (
            f"Controller did not converge: final={errors[-1]:.4e}, "
            f"initial={errors[0]:.4e}"
        )

    def test_large_dt_stable(self, ctrl):
        """At dt = 0.07 (40% above the flight-sim default), must not diverge.

        The open-loop plant amplifies ~780x per step at dt=0.07, making
        this a demanding test of the DARE discrete-time synthesis.
        dt=0.075+ overflows float64 during the observer transient.
        """
        dt = 0.07
        A, B2, C2 = ctrl.A, ctrl.B2, ctrl.C2
        Ad, Bd = _zoh_discretize(A, B2, dt)

        x_plant = np.array([0.01, 0.0])

        for _ in range(500):  # 35 seconds
            y = (C2 @ x_plant).item()
            u = ctrl.step(y, dt)
            x_plant = Ad @ x_plant + Bd.ravel() * u

        final_err = abs((C2 @ x_plant).item())
        assert final_err < 0.1, f"Diverged at large dt: final_err={final_err:.4e}"

    def test_small_dt_stable(self, ctrl):
        """dt = 0.001 must converge quickly."""
        dt = 0.001
        A, B2, C2 = ctrl.A, ctrl.B2, ctrl.C2
        Ad, Bd = _zoh_discretize(A, B2, dt)

        x_plant = np.array([0.1, 0.0])

        for _ in range(5000):  # 5 seconds at 1 kHz
            y = (C2 @ x_plant).item()
            u = ctrl.step(y, dt)
            x_plant = Ad @ x_plant + Bd.ravel() * u

        final_err = abs((C2 @ x_plant).item())
        assert final_err < 1e-3, f"Did not converge at small dt: {final_err:.4e}"

    def test_disturbance_rejection(self, ctrl):
        """A kick disturbance at t=5s should be rejected."""
        dt = 0.05
        A, B2, C2, B1 = ctrl.A, ctrl.B2, ctrl.C2, ctrl.B1
        Ad_u, Bd_u = _zoh_discretize(A, B2, dt)
        _, Bd_w = _zoh_discretize(A, B1, dt)

        x_plant = np.array([0.0, 0.0])

        for step_i in range(600):
            t = step_i * dt
            y = (C2 @ x_plant).item()
            u = ctrl.step(y, dt)
            w = 10.0 if abs(t - 5.0) < dt else 0.0
            x_plant = Ad_u @ x_plant + Bd_u.ravel() * u + Bd_w.ravel() * w

        final_err = abs((C2 @ x_plant).item())
        assert final_err < 0.05, (
            f"Disturbance not rejected: final={final_err:.4e}"
        )

    def test_output_saturation(self):
        """Output clipping should limit the control signal."""
        ctrl = get_radial_robust_controller()
        ctrl.u_max = 50.0  # tight saturation
        ctrl.state = np.array([1e3, 1e3])  # large state → large u_raw

        u = ctrl.step(0.0, 0.05)
        assert abs(u) <= 50.0 + 1e-12, f"Saturation violated: u={u}"

    def test_reset_clears_state(self, ctrl):
        """reset() must zero the observer state."""
        ctrl.step(1.0, 0.05)
        ctrl.step(1.0, 0.05)
        ctrl.reset()
        assert np.allclose(ctrl.state, 0.0)

    def test_independent_instances(self):
        """Two controllers must maintain independent observer states."""
        c1 = get_radial_robust_controller()
        c2 = get_radial_robust_controller()

        # Drive c1 with error, leave c2 idle
        for _ in range(10):
            c1.step(1.0, 0.05)
            c2.step(0.0, 0.05)

        # c1 should have non-zero state, c2 should be near-zero
        assert np.linalg.norm(c1.state) > 1e-6
        assert np.linalg.norm(c2.state) < 1e-6


# ── Edge case tests ──────────────────────────────────────────────────

class TestEdgeCases:
    def test_nan_error_raises(self):
        ctrl = get_radial_robust_controller()
        with pytest.raises(ValueError, match="finite"):
            ctrl.step(float("nan"), 0.05)

    def test_negative_dt_raises(self):
        ctrl = get_radial_robust_controller()
        with pytest.raises(ValueError, match="finite and > 0"):
            ctrl.step(0.0, -0.01)

    def test_zero_dt_raises(self):
        ctrl = get_radial_robust_controller()
        with pytest.raises(ValueError, match="finite and > 0"):
            ctrl.step(0.0, 0.0)

    def test_dt_change_updates_cache(self):
        ctrl = get_radial_robust_controller()
        ctrl.step(1.0, 0.05)
        assert ctrl._cached_dt == 0.05
        ctrl.step(1.0, 0.01)
        assert ctrl._cached_dt == 0.01

    def test_custom_gamma(self):
        """Explicit gamma should be used directly."""
        ctrl = HInfinityController(
            A=np.array([[0, 1], [100, -10]]),
            B1=np.array([[0], [0.5]]),
            B2=np.array([[0], [1]]),
            C1=np.array([[1, 0], [0, 0]]),
            C2=np.array([[1, 0]]),
            gamma=50.0,
        )
        assert ctrl.gamma == pytest.approx(50.0)
        assert ctrl.is_stable
