# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Neural-Surrogate MPC Tests
"""Deterministic tests for neural_surrogate_mpc runtime/controller paths."""

from __future__ import annotations

import numpy as np
import pytest
from typing import Any, cast

from scpn_fusion.control.neural_surrogate_mpc import (
    FloatArray,
    ModelPredictiveController,
    MpcKernel,
    NeuralSurrogate,
    create_mpc_controller,
    run_mpc_simulation,
)


class _DummyKernel:
    """Deterministic stand-in for FusionKernel used by the neural-surrogate MPC tests."""

    def __init__(self, _config_file: str) -> None:
        self.cfg: dict[str, Any] = {
            "physics": {"plasma_current_target": 7.0},
            "coils": [
                {"name": "PF1", "current": 0.0},
                {"name": "PF2", "current": 0.0},
                {"name": "PF3", "current": 0.0},
                {"name": "PF4", "current": 0.0},
            ],
        }
        self.R = np.linspace(5.8, 6.3, 25)
        self.Z = np.linspace(-0.4, 0.4, 25)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.zeros((len(self.Z), len(self.R)), dtype=np.float64)
        self._xp = (5.0, -3.5)
        self.solve_equilibrium()

    def solve_equilibrium(self) -> None:
        coils = cast(list[dict[str, Any]], self.cfg["coils"])
        i = [float(c["current"]) for c in coils]
        radial_drive = 0.8 * i[2] - 0.4 * i[1] + 0.1 * i[3]
        vertical_drive = 0.7 * i[3] - 0.6 * i[0] + 0.1 * i[2]

        center_r = 6.0 + 0.08 * np.tanh(radial_drive / 7.0)
        center_z = 0.0 + 0.06 * np.tanh(vertical_drive / 7.0)

        ir = int(np.argmin(np.abs(self.R - center_r)))
        iz = int(np.argmin(np.abs(self.Z - center_z)))
        self.Psi.fill(-1.0)
        physics = cast(dict[str, Any], self.cfg["physics"])
        self.Psi[iz, ir] = 1.0 + 0.001 * float(physics["plasma_current_target"])

        xr = 5.0 + 0.03 * np.tanh((i[1] - i[0]) / 6.0)
        xz = -3.5 + 0.03 * np.tanh((i[3] - i[2]) / 6.0)
        self._xp = (float(xr), float(xz))

    def find_x_point(self, _psi: np.ndarray) -> tuple[tuple[float, float], float]:
        return self._xp, 0.0


def test_run_mpc_simulation_returns_finite_bounded_summary() -> None:
    summary = run_mpc_simulation(
        config_file="dummy.json",
        shot_length=22,
        prediction_horizon=6,
        disturbance_start_step=3,
        disturbance_per_step_ma=0.7,
        current_target_bounds=(7.0, 9.0),
        action_limit=0.4,
        coil_current_limits=(-1.25, 1.25),
        save_plot=False,
        verbose=False,
        kernel_factory=_DummyKernel,
    )
    for key in (
        "config_path",
        "steps",
        "prediction_horizon",
        "runtime_seconds",
        "final_target_ip_ma",
        "final_r_axis",
        "final_z_axis",
        "final_xpoint_r",
        "final_xpoint_z",
        "mean_tracking_error",
        "max_abs_action",
        "max_abs_coil_current",
        "plot_saved",
    ):
        assert key in summary
    assert summary["config_path"] == "dummy.json"
    assert summary["steps"] == 22
    assert summary["prediction_horizon"] == 6
    assert summary["plot_saved"] is False
    assert summary["plot_error"] is None
    assert np.isfinite(summary["runtime_seconds"])
    assert np.isfinite(summary["mean_tracking_error"])
    assert summary["max_abs_action"] <= 0.4 + 1e-9
    assert summary["max_abs_coil_current"] <= 1.25 + 1e-9
    assert 7.0 <= summary["final_target_ip_ma"] <= 9.0


def test_run_mpc_simulation_is_deterministic_for_fixed_inputs() -> None:
    kwargs = dict(
        config_file="dummy.json",
        shot_length=18,
        prediction_horizon=5,
        disturbance_start_step=4,
        disturbance_per_step_ma=0.5,
        current_target_bounds=(6.5, 8.0),
        action_limit=0.35,
        coil_current_limits=(-1.0, 1.0),
        save_plot=False,
        verbose=False,
        kernel_factory=_DummyKernel,
    )
    a = run_mpc_simulation(**kwargs)
    b = run_mpc_simulation(**kwargs)
    for key in (
        "final_target_ip_ma",
        "final_r_axis",
        "final_z_axis",
        "final_xpoint_r",
        "final_xpoint_z",
        "mean_tracking_error",
        "max_abs_action",
        "max_abs_coil_current",
    ):
        assert a[key] == pytest.approx(b[key], rel=0.0, abs=0.0)


def test_mpc_plan_is_clipped_to_action_limit() -> None:
    surrogate = NeuralSurrogate(n_coils=3, n_state=4, verbose=False)
    surrogate.B[:] = 10.0

    mpc = ModelPredictiveController(
        surrogate=surrogate,
        target_state=np.zeros(4, dtype=np.float64),
        prediction_horizon=4,
        learning_rate=1.0,
        iterations=5,
        action_limit=0.3,
        action_regularization=0.0,
    )
    action = mpc.plan_trajectory(np.full(4, 5.0, dtype=np.float64))
    assert float(np.max(np.abs(action))) <= 0.3 + 1e-12


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"shot_length": 0}, "shot_length"),
        ({"disturbance_start_step": -1}, "disturbance_start_step"),
    ],
)
def test_run_mpc_simulation_rejects_invalid_runtime_inputs(
    kwargs: dict[str, int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        run_mpc_simulation(
            config_file="dummy.json",
            save_plot=False,
            verbose=False,
            kernel_factory=_DummyKernel,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"prediction_horizon": 0}, "prediction_horizon"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"iterations": 0}, "iterations"),
        ({"action_limit": 0.0}, "action_limit"),
        ({"action_regularization": -1.0}, "action_regularization"),
    ],
)
def test_mpc_controller_rejects_invalid_constructor_inputs(
    kwargs: dict[str, float | int], match: str
) -> None:
    surrogate = NeuralSurrogate(n_coils=3, n_state=4, verbose=False)
    with pytest.raises(ValueError, match=match):
        ModelPredictiveController(
            surrogate=surrogate,
            target_state=np.zeros(4, dtype=np.float64),
            **kwargs,
        )


def test_surrogate_rejects_invalid_perturbation() -> None:
    surrogate = NeuralSurrogate(n_coils=4, n_state=4, verbose=False)
    kernel = _DummyKernel("dummy.json")
    with pytest.raises(ValueError, match="perturbation"):
        surrogate.train_on_kernel(kernel, perturbation=0.0)


# ── Multi-backend dispatch (Rust <-> NumPy surrogate MPC) ────────────


def _mpc_case() -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return a deterministic ``(B, target, state)`` MPC planning case."""
    b_matrix = np.array(
        [[0.1, 0.0], [0.0, 0.1], [0.05, 0.05], [0.02, -0.02]],
        dtype=np.float64,
    )
    target = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    state = np.array([0.5, -0.3, 0.1, 0.0], dtype=np.float64)
    return b_matrix, target, state


def test_mpc_dispatch_registers_both_tiers() -> None:
    """The class-kernel registry carries RUST and NUMPY surrogate-MPC tiers."""
    from scpn_fusion.core import _multi_compat as multi

    kernels = multi.registered_kernel_classes()
    assert "neural_surrogate_mpc" in kernels
    tiers = [tier.rstrip("*") for tier in kernels["neural_surrogate_mpc"]]
    assert "rust" in tiers
    assert "numpy" in tiers


def test_mpc_numpy_floor_without_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    """The factory resolves to the NumPy MPC kernel when Rust is unavailable."""
    from scpn_fusion.core import _multi_compat as multi

    b_matrix, target, state = _mpc_case()
    multi._ensure_probed()
    monkeypatch.setitem(multi._availability, multi.BackendTier.RUST, False)
    monkeypatch.delitem(multi._class_dispatch_cache, "neural_surrogate_mpc", raising=False)
    try:
        controller = create_mpc_controller(b_matrix, target)
        assert isinstance(controller, MpcKernel)
        action = controller.plan(state)
        assert action.shape == (2,)
        assert bool(np.all(np.isfinite(action)))
    finally:
        multi._class_dispatch_cache.pop("neural_surrogate_mpc", None)


def test_create_mpc_controller_returns_plan_surface() -> None:
    """The dispatched controller exposes the ``plan`` protocol surface."""
    b_matrix, target, state = _mpc_case()
    controller = create_mpc_controller(b_matrix, target)
    assert callable(controller.plan)
    action = controller.plan(state)
    assert action.shape == (2,)
    assert bool(np.all(np.abs(action) <= 2.0 + 1e-9))


def test_mpc_kernel_matches_controller() -> None:
    """The NumPy adapter reproduces the wrapped controller exactly."""
    b_matrix, target, state = _mpc_case()
    kernel = MpcKernel(b_matrix, target)
    surrogate = NeuralSurrogate(n_coils=2, n_state=4, verbose=False)
    surrogate.B = b_matrix.copy()
    controller = ModelPredictiveController(surrogate, target)
    np.testing.assert_array_equal(kernel.plan(state), controller.plan_trajectory(state))


def test_mpc_kernel_rejects_malformed_inputs() -> None:
    """The adapter fails closed on non-2D B, target/state mismatch, and non-finite."""
    b_matrix, target, _ = _mpc_case()
    with pytest.raises(ValueError, match="2D"):
        MpcKernel(np.array([1.0, 2.0]), target)
    with pytest.raises(ValueError, match="target length"):
        MpcKernel(b_matrix, np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="finite"):
        MpcKernel(b_matrix, np.array([0.0, 0.0, np.nan, 0.0]))
    kernel = MpcKernel(b_matrix, target)
    with pytest.raises(ValueError, match="finite"):
        kernel.plan(np.array([np.nan, 0.0, 0.0, 0.0]))


def test_mpc_rust_numpy_plan_parity() -> None:
    """Rust and NumPy tiers plan the identical action to round-off.

    Both tiers run the identical gradient-descent planner over the linear
    surrogate at the canonical configuration, so the planned action is
    bit-exact up to floating-point round-off.
    """
    pytest.importorskip("scpn_fusion_rs")
    from scpn_fusion.core import _multi_compat_providers as providers

    rng = np.random.default_rng(2026)
    numpy_cls = providers._load_numpy_mpc_controller()
    rust_cls = providers._load_rust_mpc_controller()
    for _ in range(5):
        n_state, n_coils = 4, 3
        b_matrix = rng.normal(size=(n_state, n_coils)) * 0.1
        target = rng.normal(size=n_state)
        state = rng.normal(size=n_state)
        action_numpy = np.asarray(numpy_cls(b_matrix, target).plan(state), dtype=np.float64)
        action_rust = np.asarray(rust_cls(b_matrix, target).plan(state), dtype=np.float64)
        np.testing.assert_allclose(action_numpy, action_rust, rtol=1e-9, atol=1e-12)
