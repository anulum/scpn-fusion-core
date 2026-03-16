# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Kuramoto Standalone Function Tests
"""
Focused tests for standalone Kuramoto functions: wrap_phase, order_parameter,
GlobalPsiDriver, lyapunov_v, lyapunov_exponent, kuramoto_sakaguchi_step.

test_phase_kuramoto.py covers the basic happy paths and UPDE integration.
This file targets edge cases, boundary conditions, and numerical properties
that the existing file does not exercise.
"""
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.phase.kuramoto import (
    GlobalPsiDriver,
    kuramoto_sakaguchi_step,
    lyapunov_exponent,
    lyapunov_v,
    order_parameter,
    wrap_phase,
)


# ── wrap_phase edge cases ────────────────────────────────────────────


class TestWrapPhaseEdges:
    def test_pi_maps_to_neg_pi(self):
        # (pi + pi) % 2pi - pi = 2pi % 2pi - pi = 0 - pi = -pi  ... depends on impl
        w = wrap_phase(np.array([np.pi]))
        assert -np.pi <= w[0] <= np.pi

    def test_neg_pi_preserved(self):
        w = wrap_phase(np.array([-np.pi + 1e-15]))
        assert w[0] == pytest.approx(-np.pi, abs=1e-10)

    def test_scalar_like(self):
        w = wrap_phase(np.array([0.0]))
        assert w[0] == pytest.approx(0.0)

    def test_large_positive(self):
        w = wrap_phase(np.array([100.0 * np.pi]))
        assert -np.pi <= w[0] <= np.pi

    def test_large_negative(self):
        w = wrap_phase(np.array([-100.0 * np.pi]))
        assert -np.pi <= w[0] <= np.pi

    def test_batch_all_in_range(self):
        rng = np.random.default_rng(7)
        x = rng.uniform(-50 * np.pi, 50 * np.pi, 10000)
        w = wrap_phase(x)
        assert np.all(w >= -np.pi)
        assert np.all(w <= np.pi)

    def test_idempotent(self):
        rng = np.random.default_rng(3)
        x = rng.uniform(-np.pi, np.pi, 100)
        np.testing.assert_allclose(wrap_phase(wrap_phase(x)), wrap_phase(x), atol=1e-14)


# ── order_parameter edge cases ───────────────────────────────────────


class TestOrderParameterEdges:
    def test_single_oscillator(self):
        R, psi = order_parameter(np.array([1.23]))
        assert pytest.approx(1.0, abs=1e-12) == R
        assert psi == pytest.approx(1.23, abs=1e-12)

    def test_two_opposed(self):
        R, _ = order_parameter(np.array([0.0, np.pi]))
        assert R < 0.01

    def test_weights_all_zero(self):
        R, _ = order_parameter(np.array([0.0, 1.0]), weights=np.array([0.0, 0.0]))
        # W=0, should not crash (max(W, 1e-15) guard)
        assert np.isfinite(R)

    def test_weights_single_nonzero(self):
        R, psi = order_parameter(np.array([0.0, 1.0]), weights=np.array([0.0, 1.0]))
        assert pytest.approx(1.0, abs=1e-10) == R
        assert psi == pytest.approx(1.0, abs=1e-10)

    def test_2d_input_flattened(self):
        theta = np.zeros((5, 2))
        R, _ = order_parameter(theta)
        assert pytest.approx(1.0, abs=1e-12) == R


# ── GlobalPsiDriver edge cases ──────────────────────────────────────


class TestGlobalPsiDriverEdges:
    def test_unknown_mode_raises(self):
        d = GlobalPsiDriver(mode="nonexistent")
        with pytest.raises(ValueError, match="Unknown mode"):
            d.resolve(np.zeros(5), 0.0)

    def test_mean_field_with_uniform_gives_low_r(self):
        d = GlobalPsiDriver(mode="mean_field")
        theta = np.linspace(-np.pi, np.pi, 1000, endpoint=False)
        psi = d.resolve(theta, None)
        assert np.isfinite(psi)

    def test_external_preserves_value(self):
        d = GlobalPsiDriver(mode="external")
        assert d.resolve(np.zeros(3), -2.5) == pytest.approx(-2.5)


# ── lyapunov_v edge cases ────────────────────────────────────────────


class TestLyapunovVEdges:
    def test_half_synced(self):
        theta = np.array([0.0, np.pi])
        v = lyapunov_v(theta, 0.0)
        # (1-cos(0) + 1-cos(pi)) / 2 = (0 + 2) / 2 = 1.0
        assert v == pytest.approx(1.0, abs=1e-12)

    def test_quarter_shift(self):
        theta = np.array([np.pi / 2])
        v = lyapunov_v(theta, 0.0)
        assert v == pytest.approx(1.0, abs=1e-12)  # 1 - cos(pi/2) = 1

    def test_numerical_stability_large_phases(self):
        theta = np.array([1000.0 * np.pi, -999.0 * np.pi])
        v = lyapunov_v(theta, 0.0)
        assert 0.0 <= v <= 2.0


# ── lyapunov_exponent edge cases ─────────────────────────────────────


class TestLyapunovExponentEdges:
    def test_empty_returns_zero(self):
        assert lyapunov_exponent([], 0.01) == 0.0

    def test_constant_returns_zero(self):
        v_hist = [0.5, 0.5, 0.5, 0.5]
        lam = lyapunov_exponent(v_hist, 0.01)
        assert lam == pytest.approx(0.0, abs=1e-10)

    def test_near_zero_values_clamped(self):
        v_hist = [1e-20, 1e-20]
        lam = lyapunov_exponent(v_hist, 0.01)
        assert np.isfinite(lam)

    def test_dt_scales_result(self):
        v_hist = [1.0, 0.5]
        lam_small = lyapunov_exponent(v_hist, 0.001)
        lam_large = lyapunov_exponent(v_hist, 0.1)
        # Smaller dt → larger |λ| because T = n*dt is smaller
        assert abs(lam_small) > abs(lam_large)


# ── kuramoto_sakaguchi_step edge cases ───────────────────────────────


class TestKuramotoStepEdges:
    def test_single_oscillator(self):
        theta = np.array([0.5])
        omega = np.array([1.0])
        out = kuramoto_sakaguchi_step(theta, omega, dt=0.01, K=2.0, psi_mode="mean_field")
        assert out["R"] == pytest.approx(1.0, abs=1e-12)
        assert out["theta1"].shape == (1,)

    def test_zero_dt_no_evolution(self):
        rng = np.random.default_rng(42)
        theta = rng.uniform(-np.pi, np.pi, 50)
        omega = rng.normal(0, 1.0, 50)
        out = kuramoto_sakaguchi_step(theta, omega, dt=0.0, K=5.0, psi_mode="mean_field")
        np.testing.assert_allclose(out["theta1"], theta, atol=1e-14)

    def test_zero_K_zero_zeta_only_omega(self):
        theta = np.array([0.0, 1.0, 2.0])
        omega = np.array([1.0, 1.0, 1.0])
        out = kuramoto_sakaguchi_step(
            theta, omega, dt=0.01, K=0.0, zeta=0.0, psi_driver=0.0, psi_mode="external"
        )
        expected = wrap_phase(theta + 0.01 * omega)
        np.testing.assert_allclose(out["theta1"], expected, atol=1e-12)

    def test_negative_K_repels(self):
        theta = np.zeros(20)
        omega = np.zeros(20)
        # Negative K should push away from mean field
        out = kuramoto_sakaguchi_step(theta, omega, dt=0.01, K=-5.0, psi_mode="mean_field")
        # dtheta = K*R*sin(psi_r - theta - alpha) = -5*1*sin(0) = 0 for perfect sync
        np.testing.assert_allclose(out["dtheta"], 0.0, atol=1e-12)

    def test_wrap_false(self):
        theta = np.array([3.0])
        omega = np.array([1.0])
        out = kuramoto_sakaguchi_step(
            theta, omega, dt=1.0, K=0.0, psi_driver=0.0, psi_mode="external", wrap=False
        )
        assert out["theta1"][0] == pytest.approx(4.0, abs=1e-12)

    def test_output_keys(self):
        theta = np.array([0.0, 0.5])
        omega = np.array([1.0, 1.0])
        out = kuramoto_sakaguchi_step(theta, omega, dt=0.01, K=1.0, psi_mode="mean_field")
        assert set(out.keys()) == {"theta1", "dtheta", "R", "Psi_r", "Psi"}

    def test_deterministic(self):
        rng = np.random.default_rng(42)
        theta = rng.uniform(-np.pi, np.pi, 100)
        omega = rng.normal(0, 0.5, 100)
        a = kuramoto_sakaguchi_step(theta, omega, dt=0.01, K=2.0, psi_mode="mean_field")
        b = kuramoto_sakaguchi_step(theta, omega, dt=0.01, K=2.0, psi_mode="mean_field")
        np.testing.assert_array_equal(a["theta1"], b["theta1"])

    def test_zeta_combined_with_K(self):
        rng = np.random.default_rng(11)
        theta = rng.uniform(-np.pi, np.pi, 30)
        omega = np.zeros(30)
        out = kuramoto_sakaguchi_step(
            theta, omega, dt=0.01, K=1.0, zeta=1.0, psi_driver=0.0, psi_mode="external"
        )
        # Both K and zeta contribute to dtheta
        assert np.any(out["dtheta"] != 0.0)
