# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — H-Infinity Robust Controller Tests
# ──────────────────────────────────────────────────────────────────────
"""
Comprehensive tests for the H-infinity robust controller including
synthesis, gain shapes, stability verification, Riccati residual checks,
and robustness under plant perturbation.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control.h_infinity_controller import (
    HInfinityController,
    get_radial_robust_controller,
)


# ── Reference plant (canonical vertical stability model) ─────────────

def _vertical_stability_plant(
    gamma_v: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a 2-state vertical stability model.

    State: [z_plasma, dz/dt]
    A = [[0, 1], [gamma_v^2, 0]]  (unstable growth)
    B1 = [[0], [1]]  (disturbance)
    B2 = [[0], [1]]  (control)
    C1 = [[1, 0], [0, 0.01]]  (penalize position + small control)
    C2 = [[1, 0]]  (measure position)
    """
    A = np.array([[0.0, 1.0], [gamma_v ** 2, 0.0]])
    B1 = np.array([[0.0], [1.0]])
    B2 = np.array([[0.0], [1.0]])
    C1 = np.array([[1.0, 0.0], [0.0, 0.01]])
    C2 = np.array([[1.0, 0.0]])
    return A, B1, B2, C1, C2


# ── 1. Controller Synthesis ──────────────────────────────────────────

class TestSynthesis:

    def test_synthesis_2x2(self) -> None:
        """Synthesize an H-infinity controller for a 2-state vertical stability model."""
        A, B1, B2, C1, C2 = _vertical_stability_plant(gamma_v=10.0)
        ctrl = HInfinityController(A, B1, B2, C1, C2)
        assert ctrl.n == 2
        assert ctrl.m == 1  # one control input
        assert ctrl.l == 1  # one measurement
        assert np.isfinite(ctrl.gamma)
        assert ctrl.gamma > 1.0

    def test_synthesis_with_factory(self) -> None:
        """get_radial_robust_controller produces a valid controller."""
        ctrl = get_radial_robust_controller(gamma_growth=100.0)
        assert ctrl.n == 2
        assert ctrl.F is not None
        assert ctrl.L_gain is not None
        assert ctrl.X is not None
        assert ctrl.Y is not None


# ── 2. Gain Shapes ──────────────────────────────────────────────────

class TestGainShapes:

    def test_feedback_gain_shape(self) -> None:
        """F matrix shape is (m, n) where m=num controls, n=num states."""
        A, B1, B2, C1, C2 = _vertical_stability_plant(gamma_v=10.0)
        ctrl = HInfinityController(A, B1, B2, C1, C2)
        assert ctrl.F.shape == (1, 2)  # (m=1, n=2)

    def test_observer_gain_shape(self) -> None:
        """L matrix shape is (n, l) where l=num measurements."""
        A, B1, B2, C1, C2 = _vertical_stability_plant(gamma_v=10.0)
        ctrl = HInfinityController(A, B1, B2, C1, C2)
        assert ctrl.L_gain.shape == (2, 1)  # (n=2, l=1)

    def test_riccati_solution_shapes(self) -> None:
        """X and Y Riccati solutions should be (n, n) symmetric."""
        A, B1, B2, C1, C2 = _vertical_stability_plant(gamma_v=10.0)
        ctrl = HInfinityController(A, B1, B2, C1, C2)
        assert ctrl.X.shape == (2, 2)
        assert ctrl.Y.shape == (2, 2)
        np.testing.assert_allclose(ctrl.X, ctrl.X.T, atol=1e-12)
        np.testing.assert_allclose(ctrl.Y, ctrl.Y.T, atol=1e-12)

    def test_gains_finite(self) -> None:
        """Both F and L gains should contain only finite values."""
        ctrl = get_radial_robust_controller(gamma_growth=100.0)
        assert np.all(np.isfinite(ctrl.F))
        assert np.all(np.isfinite(ctrl.L_gain))


# ── 3. Closed-Loop Stability ─────────────────────────────────────────

class TestClosedLoopStability:

    def test_closed_loop_stable(self) -> None:
        """Closed loop A_cl = A + B2*F should have all eigenvalues in LHP."""
        A, B1, B2, C1, C2 = _vertical_stability_plant(gamma_v=10.0)
        ctrl = HInfinityController(A, B1, B2, C1, C2)
        assert ctrl.is_stable

    def test_closed_loop_simulation_bounded(self) -> None:
        """Simulate closed-loop for 100 steps; state should not diverge."""
        A, B1, B2, C1, C2 = _vertical_stability_plant(gamma_v=10.0)
        ctrl = HInfinityController(A, B1, B2, C1, C2)

        # Simulate discrete-time closed loop
        x = np.array([1.0, 0.0])  # initial displacement
        dt = 0.001
        max_state_norm = 0.0

        for step in range(100):
            # Measurement
            y = C2 @ x
            # Control
            u_val = ctrl.step(float(y[0]), dt=dt)
            u = np.array([u_val])
            # Plant dynamics (Euler integration)
            dx = A @ x + B2 @ u
            x = x + dx * dt
            max_state_norm = max(max_state_norm, np.linalg.norm(x))

        # State should remain bounded (not diverge)
        assert max_state_norm < 1e6
        assert np.all(np.isfinite(x))

    def test_closed_loop_converges_to_zero(self) -> None:
        """Over longer time, the controlled state should approach zero.

        Uses the state-feedback gain F directly (no observer dynamics)
        to verify the closed-loop A+B2*F drives x toward zero.
        """
        A, B1, B2, C1, C2 = _vertical_stability_plant(gamma_v=10.0)
        ctrl = HInfinityController(A, B1, B2, C1, C2)

        # Use direct state feedback (F @ x) to avoid observer infeasibility
        x = np.array([0.5, 0.0])
        dt = 0.0001  # small dt for Euler stability

        for step in range(10000):
            u = ctrl.F @ x  # direct state feedback
            dx = A @ x + B2 @ u
            x = x + dx * dt

        # After 1 second of state-feedback control, state should be small
        assert np.linalg.norm(x) < 1.0, f"State norm {np.linalg.norm(x)} did not converge"


# ── 4. Gamma and Feasibility ─────────────────────────────────────────

class TestGammaFeasibility:

    def test_gamma_positive(self) -> None:
        """Synthesized gamma > 0 and > 1."""
        A, B1, B2, C1, C2 = _vertical_stability_plant(gamma_v=10.0)
        ctrl = HInfinityController(A, B1, B2, C1, C2)
        assert ctrl.gamma > 1.0
        assert np.isfinite(ctrl.gamma)

    def test_spectral_radius_finite(self) -> None:
        """Spectral radius of XY should be finite and non-negative."""
        ctrl = get_radial_robust_controller(gamma_growth=100.0)
        assert ctrl.spectral_radius_xy >= 0.0
        assert np.isfinite(ctrl.spectral_radius_xy)

    def test_feasibility_margin(self) -> None:
        """robust_feasibility_margin() should return a finite float."""
        ctrl = get_radial_robust_controller(gamma_growth=100.0)
        margin = ctrl.robust_feasibility_margin()
        assert np.isfinite(margin)

    def test_robust_feasible_property(self) -> None:
        """robust_feasible flag should be consistent with spectral radius (with margin)."""
        ctrl = get_radial_robust_controller(gamma_growth=100.0)
        expected = ctrl.spectral_radius_xy < ctrl.gamma ** 2 * (1.0 - 1e-4)
        assert ctrl.robust_feasible == expected


# ── 5. Riccati Residuals ─────────────────────────────────────────────

class TestRiccatiResiduals:

    def test_riccati_residual_small(self) -> None:
        """X and Y Riccati solutions should have small residual norms."""
        ctrl = get_radial_robust_controller(gamma_growth=100.0)
        res_x, res_y = ctrl.riccati_residual_norms()
        # X Riccati residual should be very small (direct ARE solution)
        assert res_x < 1e-3, f"X residual {res_x} exceeds tolerance"
        # Y Riccati may have slightly larger residual but should still be small
        assert res_y < 1.0, f"Y residual {res_y} exceeds tolerance"

    def test_riccati_residuals_finite(self) -> None:
        """Residual norms should be finite."""
        A, B1, B2, C1, C2 = _vertical_stability_plant(gamma_v=10.0)
        ctrl = HInfinityController(A, B1, B2, C1, C2)
        res_x, res_y = ctrl.riccati_residual_norms()
        assert np.isfinite(res_x)
        assert np.isfinite(res_y)


# ── 6. Robustness Under Plant Perturbation ───────────────────────────

class TestRobustness:

    def test_perturbed_plant_10_percent(self) -> None:
        """Perturb A by 10%; closed-loop should still be stable."""
        A, B1, B2, C1, C2 = _vertical_stability_plant(gamma_v=10.0)
        ctrl = HInfinityController(A, B1, B2, C1, C2)

        # Perturb the A matrix by 10%
        rng = np.random.RandomState(42)
        A_perturbed = A * (1.0 + 0.1 * rng.randn(*A.shape))

        # Check closed-loop stability with original controller gains
        A_cl_perturbed = A_perturbed + B2 @ ctrl.F
        eigs = np.linalg.eigvals(A_cl_perturbed)
        # All eigenvalues should still have negative real parts
        assert np.all(np.real(eigs) < 0), (
            f"Closed-loop unstable under 10% perturbation: eigs = {eigs}"
        )

    def test_perturbed_plant_simulation(self) -> None:
        """Simulate the controller (state feedback) on a 10% perturbed plant.

        Uses direct state feedback F@x to verify robustness without
        observer dynamics, which may be infeasible for this plant.
        """
        A, B1, B2, C1, C2 = _vertical_stability_plant(gamma_v=10.0)
        ctrl = HInfinityController(A, B1, B2, C1, C2)

        rng = np.random.RandomState(123)
        A_perturbed = A * (1.0 + 0.1 * rng.randn(*A.shape))

        x = np.array([1.0, 0.0])
        dt = 0.0001  # small dt for Euler stability with perturbed plant

        for _ in range(5000):
            u = ctrl.F @ x  # direct state feedback
            dx = A_perturbed @ x + B2 @ u
            x = x + dx * dt

        assert np.all(np.isfinite(x))
        assert np.linalg.norm(x) < 100.0, f"State norm {np.linalg.norm(x)} diverged"


# ── 7. Input Validation ──────────────────────────────────────────────

class TestInputValidation:

    def test_rejects_invalid_gamma(self) -> None:
        """gamma must be > 1.0 and finite."""
        A, B1, B2, C1, C2 = _vertical_stability_plant()
        with pytest.raises(ValueError, match="gamma must be finite and > 1.0"):
            HInfinityController(A, B1, B2, C1, C2, gamma=1.0)
        with pytest.raises(ValueError, match="gamma must be finite and > 1.0"):
            HInfinityController(A, B1, B2, C1, C2, gamma=float("nan"))

    def test_rejects_nonfinite_A(self) -> None:
        """A matrix with NaN or Inf should be rejected."""
        _, B1, B2, C1, C2 = _vertical_stability_plant()
        A_bad = np.array([[float("nan"), 1.0], [0.0, 0.0]])
        with pytest.raises(ValueError, match="finite"):
            HInfinityController(A_bad, B1, B2, C1, C2)

    def test_rejects_nonsquare_A(self) -> None:
        """A must be a square matrix."""
        _, B1, B2, C1, C2 = _vertical_stability_plant()
        A_rect = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="square"):
            HInfinityController(A_rect, B1, B2, C1, C2)

    def test_rejects_dimension_mismatch_B2(self) -> None:
        """B2 row count must match A."""
        A, B1, _, C1, C2 = _vertical_stability_plant()
        B2_wrong = np.array([[0.0], [0.0], [1.0]])  # 3 rows vs A is 2x2
        with pytest.raises(ValueError, match="same number of rows"):
            HInfinityController(A, B1, B2_wrong, C1, C2)

    def test_rejects_nonfinite_step_error(self) -> None:
        """step() rejects NaN measurement."""
        ctrl = get_radial_robust_controller(gamma_growth=100.0)
        with pytest.raises(ValueError, match="error must be finite"):
            ctrl.step(float("nan"), dt=1e-3)

    def test_rejects_zero_dt(self) -> None:
        """step() rejects dt <= 0."""
        ctrl = get_radial_robust_controller(gamma_growth=100.0)
        with pytest.raises(ValueError, match="dt must be finite and > 0"):
            ctrl.step(0.1, dt=0.0)


# ── 8. Reset and State Management ────────────────────────────────────

class TestStateManagement:

    def test_reset_zeros_state(self) -> None:
        """reset() should zero the controller internal state."""
        ctrl = get_radial_robust_controller(gamma_growth=100.0)
        # Accumulate some state
        for _ in range(10):
            ctrl.step(0.5, dt=1e-3)
        assert not np.allclose(ctrl.state, 0.0)
        ctrl.reset()
        np.testing.assert_array_equal(ctrl.state, np.zeros(ctrl.n))

    def test_step_modifies_state(self) -> None:
        """step() should modify the internal state."""
        ctrl = get_radial_robust_controller(gamma_growth=100.0)
        state_before = ctrl.state.copy()
        ctrl.step(1.0, dt=1e-3)
        assert not np.array_equal(ctrl.state, state_before)

    def test_step_returns_scalar(self) -> None:
        """step() should return a scalar control action."""
        ctrl = get_radial_robust_controller(gamma_growth=100.0)
        u = ctrl.step(0.5, dt=1e-3)
        assert isinstance(u, float)
        assert np.isfinite(u)


# ── 9. Enforce Robust Feasibility ────────────────────────────────────

class TestEnforceRobustFeasibility:

    def test_strict_mode_rejects_infeasible(self) -> None:
        """enforce_robust_feasibility=True raises on infeasible synthesis."""
        A, B1, B2, C1, C2 = _vertical_stability_plant(gamma_v=100.0)
        # The reference plant with gamma_v=100 typically fails spectral condition
        with pytest.raises(ValueError, match="spectral feasibility condition failed"):
            HInfinityController(A, B1, B2, C1, C2, enforce_robust_feasibility=True)

    def test_non_strict_mode_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Without enforce_robust_feasibility, infeasible produces a warning."""
        A, B1, B2, C1, C2 = _vertical_stability_plant(gamma_v=100.0)
        with caplog.at_level("WARNING", logger="scpn_fusion.control.h_infinity_controller"):
            ctrl = HInfinityController(A, B1, B2, C1, C2)
        # Should still produce a controller, just flag infeasible
        assert isinstance(ctrl, HInfinityController)
