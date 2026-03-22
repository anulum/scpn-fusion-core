# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# SCPN Fusion Core — Fusion Control Room Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control.fusion_control_room import (
    DiagnosticSystem,
    KalmanObserver,
    NeuralController,
    TokamakPhysicsEngine,
)


class TestTokamakPhysicsEngine:
    def test_init_default(self):
        engine = TokamakPhysicsEngine(size=32, seed=42)
        assert engine.size == 32
        assert engine.density.shape == (32, 32)

    def test_rejects_small_size(self):
        with pytest.raises(ValueError, match="16"):
            TokamakPhysicsEngine(size=8)

    def test_solve_flux_surfaces(self):
        engine = TokamakPhysicsEngine(size=32, seed=42)
        density, psi = engine.solve_flux_surfaces()
        assert density.shape == (32, 32)
        assert psi.shape == (32, 32)
        assert np.all(np.isfinite(density))

    def test_solve_with_kernel_psi(self):
        class MockKernel:
            Psi = np.ones((32, 32)) * 0.5

        engine = TokamakPhysicsEngine(size=32, seed=42, kernel=MockKernel())
        density, psi = engine.solve_flux_surfaces()
        assert np.all(np.isfinite(psi))


class TestDiagnosticSystem:
    def test_measure_adds_noise(self):
        rng = np.random.default_rng(42)
        diag = DiagnosticSystem(rng)
        m = diag.measure_position(0.0)
        assert m != 0.0
        assert abs(m) < 1.0

    def test_deterministic_with_rng(self):
        m1 = DiagnosticSystem(np.random.default_rng(42)).measure_position(1.0)
        m2 = DiagnosticSystem(np.random.default_rng(42)).measure_position(1.0)
        assert m1 == m2


class TestKalmanObserver:
    def test_init(self):
        kf = KalmanObserver(dt=0.1)
        assert kf.x[0] == 0.0
        assert kf.x[1] == 0.0

    def test_update_converges(self):
        kf = KalmanObserver(dt=0.1)
        for _ in range(20):
            z_est = kf.update(measured_z=1.0)
        assert abs(z_est - 1.0) < 0.1

    def test_dropout_increases_uncertainty(self):
        kf = KalmanObserver(dt=0.1)
        kf.update(measured_z=1.0)
        P_before = kf.P.copy()
        kf.update(measured_z=1.0, dropout=True)
        assert np.trace(kf.P) > np.trace(P_before)

    def test_nan_measurement_treated_as_dropout(self):
        kf = KalmanObserver(dt=0.1)
        kf.update(measured_z=1.0)
        P_before = kf.P.copy()
        kf.update(measured_z=float("nan"))
        assert np.trace(kf.P) > np.trace(P_before)


class TestNeuralController:
    def test_init(self):
        ctrl = NeuralController(dt=0.1)
        assert ctrl.kp > 0

    def test_compute_action_returns_tuple(self):
        ctrl = NeuralController(dt=0.1)
        result = ctrl.compute_action(0.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_compute_action_zero_input(self):
        ctrl = NeuralController(dt=0.1)
        up, down = ctrl.compute_action(0.0)
        assert up + down < 0.1

    def test_compute_action_positive_input(self):
        ctrl = NeuralController(dt=0.1)
        up, down = ctrl.compute_action(1.0)
        assert up + down > 0

    def test_integral_accumulates(self):
        ctrl = NeuralController(dt=0.1)
        # error = -measured_z, so positive input gives negative integral
        ctrl.compute_action(1.0)
        ctrl.compute_action(1.0)
        assert ctrl.integral_error != 0.0

    def test_action_bounded(self):
        ctrl = NeuralController(dt=0.1)
        for _ in range(100):
            up, down = ctrl.compute_action(10.0)
        assert np.isfinite(up)
        assert np.isfinite(down)
        assert 0.0 <= up <= 1.0
        assert 0.0 <= down <= 1.0
