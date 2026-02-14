# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Fusion SOTA MPC Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Deterministic tests for fusion_sota_mpc runtime/controller paths."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control.fusion_sota_mpc import (
    ModelPredictiveController,
    NeuralSurrogate,
    run_sota_simulation,
)


class _DummyKernel:
    """Deterministic stand-in for FusionKernel used by SOTA MPC tests."""

    def __init__(self, _config_file: str) -> None:
        self.cfg = {
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
        i = [float(c["current"]) for c in self.cfg["coils"]]
        radial_drive = 0.8 * i[2] - 0.4 * i[1] + 0.1 * i[3]
        vertical_drive = 0.7 * i[3] - 0.6 * i[0] + 0.1 * i[2]

        center_r = 6.0 + 0.08 * np.tanh(radial_drive / 7.0)
        center_z = 0.0 + 0.06 * np.tanh(vertical_drive / 7.0)

        ir = int(np.argmin(np.abs(self.R - center_r)))
        iz = int(np.argmin(np.abs(self.Z - center_z)))
        self.Psi.fill(-1.0)
        self.Psi[iz, ir] = 1.0 + 0.001 * float(
            self.cfg["physics"]["plasma_current_target"]
        )

        xr = 5.0 + 0.03 * np.tanh((i[1] - i[0]) / 6.0)
        xz = -3.5 + 0.03 * np.tanh((i[3] - i[2]) / 6.0)
        self._xp = (float(xr), float(xz))

    def find_x_point(self, _psi: np.ndarray) -> tuple[tuple[float, float], float]:
        return self._xp, 0.0


def test_run_sota_simulation_returns_finite_bounded_summary() -> None:
    summary = run_sota_simulation(
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


def test_run_sota_simulation_is_deterministic_for_fixed_inputs() -> None:
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
    a = run_sota_simulation(**kwargs)
    b = run_sota_simulation(**kwargs)
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
