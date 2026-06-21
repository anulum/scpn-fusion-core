# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Fusion Control Room Tests
"""Tests for the vertical-stability control room: physics, Kalman observer, and demo."""

from __future__ import annotations

from typing import Any

import matplotlib.figure
import numpy as np
import pytest

import scpn_fusion.control.fusion_control_room as fcr
from scpn_fusion.control.fusion_control_room import (
    DiagnosticSystem,
    KalmanObserver,
    NeuralController,
    TokamakPhysicsEngine,
    run_control_room,
)


class TestTokamakPhysicsEngine:
    """Reduced Grad-Shafranov plasma engine and vertical dynamics."""

    def test_init_default(self) -> None:
        """Construction allocates the requested square density grid."""
        engine = TokamakPhysicsEngine(size=32, seed=42)
        assert engine.size == 32
        assert engine.density.shape == (32, 32)

    def test_rejects_small_size(self) -> None:
        """A grid below the minimum size is rejected."""
        with pytest.raises(ValueError, match="16"):
            TokamakPhysicsEngine(size=8)

    def test_solve_flux_surfaces(self) -> None:
        """Flux-surface solve returns finite density and flux grids."""
        engine = TokamakPhysicsEngine(size=32, seed=42)
        density, psi = engine.solve_flux_surfaces()
        assert density.shape == (32, 32)
        assert psi.shape == (32, 32)
        assert np.all(np.isfinite(density))

    def test_solve_with_kernel_psi(self) -> None:
        """A kernel-supplied flux map is accepted and stays finite."""

        class MockKernel:
            Psi = np.ones((32, 32)) * 0.5

        engine = TokamakPhysicsEngine(size=32, seed=42, kernel=MockKernel())
        _density, psi = engine.solve_flux_surfaces()
        assert np.all(np.isfinite(psi))

    def test_kernel_psi_wrong_shape_falls_back(self) -> None:
        """A kernel flux map of the wrong shape is ignored in favour of the analytic field."""

        class TinyKernel:
            Psi = np.ones((4, 4))

        engine = TokamakPhysicsEngine(size=32, seed=42, kernel=TinyKernel())
        _density, psi = engine.solve_flux_surfaces()
        assert psi.shape == (32, 32)
        assert np.all(np.isfinite(psi))

    def test_step_dynamics_advances_position(self) -> None:
        """Stepping the vertical dynamics returns a finite displacement."""
        engine = TokamakPhysicsEngine(size=32, seed=42)
        z = engine.step_dynamics(coil_action_top=0.1, coil_action_bottom=0.2)
        assert np.isfinite(z)


class TestDiagnosticSystem:
    """Noisy vertical-position diagnostic."""

    def test_measure_adds_noise(self) -> None:
        """A measurement perturbs the true position within sensor bounds."""
        diag = DiagnosticSystem(np.random.default_rng(42))
        m = diag.measure_position(0.0)
        assert m != 0.0
        assert abs(m) < 1.0

    def test_deterministic_with_rng(self) -> None:
        """A fixed RNG seed reproduces the measurement."""
        m1 = DiagnosticSystem(np.random.default_rng(42)).measure_position(1.0)
        m2 = DiagnosticSystem(np.random.default_rng(42)).measure_position(1.0)
        assert m1 == m2


class TestKalmanObserver:
    """Constant-velocity Kalman state estimator with dropout handling."""

    def test_init(self) -> None:
        """The observer starts at rest."""
        kf = KalmanObserver(dt=0.1)
        assert kf.x[0] == 0.0
        assert kf.x[1] == 0.0

    def test_update_converges(self) -> None:
        """Repeated measurements converge the estimate to the truth."""
        kf = KalmanObserver(dt=0.1)
        z_est = 0.0
        for _ in range(20):
            z_est = kf.update(measured_z=1.0)
        assert abs(z_est - 1.0) < 0.1

    def test_dropout_increases_uncertainty(self) -> None:
        """A sensor dropout grows the state covariance."""
        kf = KalmanObserver(dt=0.1)
        kf.update(measured_z=1.0)
        p_before = kf.P.copy()
        kf.update(measured_z=1.0, dropout=True)
        assert np.trace(kf.P) > np.trace(p_before)

    def test_nan_measurement_treated_as_dropout(self) -> None:
        """A NaN measurement is handled as a dropout, growing uncertainty."""
        kf = KalmanObserver(dt=0.1)
        kf.update(measured_z=1.0)
        p_before = kf.P.copy()
        kf.update(measured_z=float("nan"))
        assert np.trace(kf.P) > np.trace(p_before)


class TestNeuralController:
    """PID-style vertical-stability controller with actuator saturation."""

    def test_init(self) -> None:
        """The controller starts with a positive proportional gain."""
        assert NeuralController(dt=0.1).kp > 0

    def test_compute_action_returns_tuple(self) -> None:
        """An action is a two-element coil command tuple."""
        result = NeuralController(dt=0.1).compute_action(0.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_compute_action_zero_input(self) -> None:
        """A centred plasma needs negligible coil action."""
        up, down = NeuralController(dt=0.1).compute_action(0.0)
        assert up + down < 0.1

    def test_compute_action_positive_input(self) -> None:
        """A positive displacement drives the lower coil branch."""
        up, down = NeuralController(dt=0.1).compute_action(1.0)
        assert up + down > 0

    def test_compute_action_negative_input(self) -> None:
        """A negative displacement drives the upper coil branch."""
        ctrl = NeuralController(dt=0.1)
        up, down = ctrl.compute_action(-1.0)
        assert up + down > 0
        assert up >= 0.0
        assert down >= 0.0

    def test_integral_accumulates(self) -> None:
        """The integral term accumulates over successive steps."""
        ctrl = NeuralController(dt=0.1)
        ctrl.compute_action(1.0)
        ctrl.compute_action(1.0)
        assert ctrl.integral_error != 0.0

    def test_action_bounded(self) -> None:
        """The actuator saturation keeps both coil commands within [0, 1]."""
        ctrl = NeuralController(dt=0.1)
        up, down = 0.0, 0.0
        for _ in range(100):
            up, down = ctrl.compute_action(10.0)
        assert 0.0 <= up <= 1.0
        assert 0.0 <= down <= 1.0


class _GoodKernel:
    """Kernel stub with a valid coil config and a no-op equilibrium solve."""

    def __init__(self, _config: str) -> None:
        self.Psi = np.ones((32, 32)) * 0.5
        self.cfg: dict[str, Any] = {"coils": [{"current": 0.0} for _ in range(5)]}

    def solve_equilibrium(self) -> None:
        """No-op equilibrium solve."""
        return None


class _CoilFailKernel(_GoodKernel):
    """Kernel whose coil config raises when indexed, to exercise the coil-update guard."""

    def __init__(self, _config: str) -> None:
        super().__init__(_config)
        self.cfg = {"coils": [None, None, None, None, None]}


class _SolveFailKernel(_GoodKernel):
    """Kernel whose equilibrium solve always fails."""

    def solve_equilibrium(self) -> None:
        """Raise to exercise the solve-failure guard."""
        raise RuntimeError("solve diverged")


def test_run_control_room_rejects_bad_duration() -> None:
    """A sub-unit simulation duration is rejected."""
    with pytest.raises(ValueError, match="sim_duration"):
        run_control_room(sim_duration=0)


def test_run_control_room_analytic_no_render() -> None:
    """The analytic-source loop runs and returns deterministic metrics without rendering."""
    result = run_control_room(sim_duration=5, save_animation=False, save_report=False, verbose=True)
    assert result["psi_source"] == "analytic"
    assert result["steps"] == 5
    assert np.isfinite(result["final_z"])
    assert result["kernel_fallback_used"] is False


def test_run_control_room_with_kernel() -> None:
    """A healthy kernel factory drives the kernel-backed flux source."""
    result = run_control_room(
        sim_duration=5,
        save_animation=False,
        save_report=False,
        verbose=True,
        kernel_factory=_GoodKernel,
    )
    assert result["psi_source"] == "kernel"
    assert result["kernel_coil_update_failures"] == 0
    assert result["kernel_solve_failures"] == 0


def test_run_control_room_kernel_init_failure_falls_back() -> None:
    """A failing kernel constructor falls back to the analytic source when allowed."""

    def _boom(_config: str) -> Any:
        raise RuntimeError("kernel init failed")

    result = run_control_room(
        sim_duration=3,
        save_animation=False,
        save_report=False,
        verbose=True,
        kernel_factory=_boom,
    )
    assert result["psi_source"] == "analytic"
    assert result["kernel_error"] is not None
    assert result["kernel_fallback_used"] is True


def test_run_control_room_kernel_init_failure_strict_raises() -> None:
    """A failing kernel constructor raises when fallback is disallowed."""

    def _boom(_config: str) -> Any:
        raise RuntimeError("kernel init failed")

    with pytest.raises(RuntimeError, match="allow_kernel_fallback=False"):
        run_control_room(
            sim_duration=3,
            save_animation=False,
            save_report=False,
            kernel_factory=_boom,
            allow_kernel_fallback=False,
        )


def test_run_control_room_coil_update_failure_falls_back() -> None:
    """A coil-update fault is counted and tolerated when fallback is allowed."""
    result = run_control_room(
        sim_duration=3,
        save_animation=False,
        save_report=False,
        kernel_factory=_CoilFailKernel,
    )
    assert result["kernel_coil_update_failures"] > 0
    assert result["kernel_coil_update_error"] is not None


def test_run_control_room_coil_update_failure_strict_raises() -> None:
    """A coil-update fault raises when fallback is disallowed."""
    with pytest.raises(RuntimeError, match="Kernel coil update"):
        run_control_room(
            sim_duration=3,
            save_animation=False,
            save_report=False,
            kernel_factory=_CoilFailKernel,
            allow_kernel_fallback=False,
        )


def test_run_control_room_solve_failure_falls_back() -> None:
    """An equilibrium-solve fault is counted and tolerated when fallback is allowed."""
    result = run_control_room(
        sim_duration=3,
        save_animation=False,
        save_report=False,
        kernel_factory=_SolveFailKernel,
    )
    assert result["kernel_solve_failures"] > 0
    assert result["kernel_solve_error"] is not None


def test_run_control_room_solve_failure_strict_raises() -> None:
    """An equilibrium-solve fault raises when fallback is disallowed."""
    with pytest.raises(RuntimeError, match="Kernel solve"):
        run_control_room(
            sim_duration=3,
            save_animation=False,
            save_report=False,
            kernel_factory=_SolveFailKernel,
            allow_kernel_fallback=False,
        )


class _StubAnimation:
    """FuncAnimation stub that records whether the animation save was attempted."""

    saved = False

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def save(self, *args: object, **kwargs: object) -> None:
        """Record a successful save."""
        _StubAnimation.saved = True


def test_run_control_room_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    """The render path saves both the animation and the status report."""
    _StubAnimation.saved = False
    saved_png: list[bool] = []
    monkeypatch.setattr(fcr, "FuncAnimation", _StubAnimation)
    monkeypatch.setattr(
        matplotlib.figure.Figure, "savefig", lambda self, *a, **k: saved_png.append(True)
    )

    result = run_control_room(sim_duration=5, save_animation=True, save_report=True, verbose=False)

    assert _StubAnimation.saved is True
    assert result["animation_saved"] is True
    assert result["report_saved"] is True
    assert saved_png


def test_run_control_room_render_failures_are_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Animation and report rendering faults are caught and reported, not raised."""

    class _FailingAnimation:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def save(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("gif writer unavailable")

    def _bad_savefig(self: Any, *a: object, **k: object) -> None:
        raise RuntimeError("png backend unavailable")

    monkeypatch.setattr(fcr, "FuncAnimation", _FailingAnimation)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", _bad_savefig)

    result = run_control_room(sim_duration=4, save_animation=True, save_report=True, verbose=False)

    assert result["animation_saved"] is False
    assert result["animation_error"] is not None
    assert result["report_saved"] is False
    assert result["report_error"] is not None


def test_step_dynamics_random_disturbance_spike() -> None:
    """Across many steps the stochastic disturbance-spike branch is exercised."""
    engine = TokamakPhysicsEngine(size=32, seed=7)
    positions = [engine.step_dynamics(0.0, 0.0) for _ in range(300)]
    assert all(np.isfinite(z) for z in positions)


def test_run_control_room_config_file_uses_default_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """A config file with no explicit factory routes through the default kernel constructor."""
    monkeypatch.setattr(fcr, "FusionKernel", _GoodKernel)
    result = run_control_room(
        sim_duration=4,
        save_animation=False,
        save_report=False,
        verbose=False,
        config_file="some_config.json",
    )
    assert result["psi_source"] == "kernel"
