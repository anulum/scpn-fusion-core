# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Fusion Optimal Control Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Deterministic tests for fusion_optimal_control runtime/controller paths."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control.fusion_optimal_control import OptimalController, run_optimal_control


class _DummyKernel:
    """Small deterministic kernel stand-in with R/Z/Psi fields."""

    def __init__(self, _config_file: str) -> None:
        self.cfg = {
            "physics": {"plasma_current_target": 8.0},
            "coils": [
                {"name": "PF1", "r": 5.7, "z": -0.25, "current": 0.0},
                {"name": "PF2", "r": 5.9, "z": 0.25, "current": 0.0},
                {"name": "PF3", "r": 6.1, "z": -0.25, "current": 0.0},
                {"name": "PF4", "r": 6.3, "z": 0.25, "current": 0.0},
            ],
        }
        self.R = np.linspace(5.8, 6.3, 21)
        self.Z = np.linspace(-0.4, 0.4, 21)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.zeros((len(self.Z), len(self.R)), dtype=np.float64)
        self.J_phi = np.zeros_like(self.Psi)
        self.solve_equilibrium()

    def solve_equilibrium(self) -> None:
        i = [float(c["current"]) for c in self.cfg["coils"]]
        radial_drive = 0.7 * i[2] - 0.5 * i[1] + 0.2 * i[3]
        vertical_drive = 0.6 * i[3] - 0.6 * i[0] + 0.2 * i[2]
        center_r = 6.02 + 0.09 * np.tanh(radial_drive / 8.0)
        center_z = 0.00 + 0.11 * np.tanh(vertical_drive / 8.0)

        ir = int(np.argmin(np.abs(self.R - center_r)))
        iz = int(np.argmin(np.abs(self.Z - center_z)))
        self.Psi.fill(-1.0)
        self.Psi[iz, ir] = 1.0 + 0.001 * float(
            self.cfg["physics"]["plasma_current_target"]
        )
        self.J_phi = np.exp(
            -((self.RR - center_r) ** 2 + ((self.ZZ - center_z) / 1.5) ** 2)
        )


def test_run_optimal_control_returns_finite_bounded_summary() -> None:
    summary = run_optimal_control(
        config_file="dummy.json",
        shot_steps=16,
        target_r=6.02,
        target_z=0.01,
        seed=123,
        save_plot=False,
        verbose=False,
        kernel_factory=_DummyKernel,
        coil_current_limits=(-2.0, 2.0),
        current_target_limits=(7.0, 9.0),
    )
    for key in (
        "seed",
        "config_path",
        "steps",
        "final_target_ip_ma",
        "final_axis_r",
        "final_axis_z",
        "mean_abs_r_error",
        "mean_abs_z_error",
        "mean_error_norm",
        "max_abs_delta_i",
        "max_abs_coil_current",
        "plot_saved",
    ):
        assert key in summary
    assert summary["config_path"] == "dummy.json"
    assert summary["seed"] == 123
    assert summary["steps"] == 16
    assert summary["plot_saved"] is False
    assert summary["plot_error"] is None
    assert np.isfinite(summary["mean_error_norm"])
    assert summary["max_abs_coil_current"] <= 2.0 + 1e-9
    assert 7.0 <= summary["final_target_ip_ma"] <= 9.0


def test_run_optimal_control_is_deterministic_for_fixed_seed() -> None:
    kwargs = dict(
        config_file="dummy.json",
        shot_steps=14,
        target_r=6.01,
        target_z=-0.02,
        seed=77,
        save_plot=False,
        verbose=False,
        kernel_factory=_DummyKernel,
        coil_current_limits=(-2.0, 2.0),
        current_target_limits=(6.5, 8.5),
    )
    a = run_optimal_control(**kwargs)
    b = run_optimal_control(**kwargs)
    for key in (
        "final_target_ip_ma",
        "final_axis_r",
        "final_axis_z",
        "mean_abs_r_error",
        "mean_abs_z_error",
        "mean_error_norm",
        "max_abs_delta_i",
        "max_abs_coil_current",
    ):
        assert a[key] == pytest.approx(b[key], rel=0.0, abs=0.0)


def test_optimal_controller_enforces_correction_and_coil_limits() -> None:
    pilot = OptimalController(
        "dummy.json",
        kernel_factory=_DummyKernel,
        verbose=False,
        correction_limit=0.25,
        coil_current_limits=(-1.0, 1.0),
        current_target_limits=(8.0, 9.0),
    )
    pilot.identify_system(perturbation=0.4)
    curr_pos = pilot.get_plasma_pos()
    delta = pilot.compute_optimal_correction(curr_pos, np.array([6.3, 0.3]))
    assert float(np.max(np.abs(delta))) <= 0.25 + 1e-12

    summary = pilot.run_optimal_shot(
        shot_steps=10,
        target_r=6.3,
        target_z=0.3,
        gain=1.5,
        ip_start_ma=4.0,
        ip_span_ma=50.0,
        identify_first=False,
        save_plot=False,
    )
    assert summary["max_abs_coil_current"] <= 1.0 + 1e-9
    assert 8.0 <= summary["final_target_ip_ma"] <= 9.0


def test_run_optimal_control_does_not_mutate_global_numpy_rng_state() -> None:
    np.random.seed(1357)
    state = np.random.get_state()

    run_optimal_control(
        config_file="dummy.json",
        shot_steps=8,
        target_r=6.02,
        target_z=0.01,
        seed=55,
        save_plot=False,
        verbose=False,
        kernel_factory=_DummyKernel,
        coil_current_limits=(-2.0, 2.0),
        current_target_limits=(7.0, 9.0),
    )

    observed = float(np.random.random())
    np.random.set_state(state)
    expected = float(np.random.random())
    assert observed == expected
