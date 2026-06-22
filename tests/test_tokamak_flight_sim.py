# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tokamak Flight Sim Tests
"""Deterministic smoke tests for tokamak_flight_sim runtime entry points."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_fusion.control.tokamak_flight_sim import (
    FirstOrderActuator,
    IsoFluxController,
    run_flight_sim,
)


class _DummyKernel:
    """Lightweight deterministic stand-in for FusionKernel in CI tests."""

    def __init__(self, _config_file: str) -> None:
        self.cfg: dict[str, Any] = {
            "physics": {"plasma_current_target": 5.0},
            "coils": [{"current": 0.0} for _ in range(5)],
        }
        self.R = np.linspace(5.8, 6.4, 13)
        self.Z = np.linspace(-0.3, 0.3, 13)
        self.NR = len(self.R)
        self.NZ = len(self.Z)
        self.dR = float(self.R[1] - self.R[0])
        self.dZ = float(self.Z[1] - self.Z[0])
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.zeros((len(self.Z), len(self.R)), dtype=np.float64)
        self._ticks = 0
        self.solve_equilibrium()

    def solve_equilibrium(self) -> None:
        self._ticks += 1
        radial_drive = float(self.cfg["coils"][2]["current"])
        vertical_drive = float(self.cfg["coils"][4]["current"]) - float(
            self.cfg["coils"][0]["current"]
        )
        center_r = 6.1 + 0.05 * np.tanh(radial_drive / 10.0)
        center_z = 0.0 + 0.04 * np.tanh(vertical_drive / 10.0)
        ir = int(np.argmin(np.abs(self.R - center_r)))
        iz = int(np.argmin(np.abs(self.Z - center_z)))
        self.Psi.fill(-1.0)
        self.Psi[iz, ir] = 1.0 + 0.001 * float(self.cfg["physics"]["plasma_current_target"])

    def find_x_point(self, _psi: np.ndarray[Any, Any]) -> tuple[tuple[float, float], float]:
        return (float(self.R[-2]), float(self.Z[1])), 0.0


def test_run_flight_sim_returns_finite_summary_without_plot() -> None:
    """The flight sim returns a finite, complete summary without plotting."""
    summary = run_flight_sim(
        config_file="dummy.json",
        shot_duration=18,
        seed=123,
        save_plot=False,
        verbose=False,
        kernel_factory=_DummyKernel,
    )
    for key in (
        "seed",
        "config_path",
        "steps",
        "final_ip_ma",
        "final_axis_r",
        "final_axis_z",
        "final_beta_scale",
        "mean_abs_r_error",
        "mean_abs_z_error",
        "mean_abs_radial_actuator_lag",
        "mean_abs_vertical_actuator_lag",
        "mean_abs_heating_actuator_lag",
        "plot_saved",
    ):
        assert key in summary
    assert summary["config_path"] == "dummy.json"
    assert summary["steps"] == 18
    assert summary["plot_saved"] is False
    assert summary["plot_error"] is None
    assert np.isfinite(summary["final_ip_ma"])
    assert np.isfinite(summary["final_axis_r"])
    assert np.isfinite(summary["final_axis_z"])
    assert np.isfinite(summary["final_beta_scale"])
    assert np.isfinite(summary["mean_abs_r_error"])
    assert np.isfinite(summary["mean_abs_z_error"])
    assert np.isfinite(summary["mean_abs_radial_actuator_lag"])
    assert np.isfinite(summary["mean_abs_vertical_actuator_lag"])
    assert np.isfinite(summary["mean_abs_heating_actuator_lag"])


def test_run_flight_sim_is_deterministic_for_fixed_seed() -> None:
    """A fixed seed reproduces the flight-sim summary."""
    kwargs: dict[str, Any] = dict(
        config_file="dummy.json",
        shot_duration=14,
        seed=77,
        save_plot=False,
        verbose=False,
        kernel_factory=_DummyKernel,
    )
    a = run_flight_sim(**kwargs)
    b = run_flight_sim(**kwargs)
    assert a["final_ip_ma"] == b["final_ip_ma"]
    assert a["final_axis_r"] == b["final_axis_r"]
    assert a["final_axis_z"] == b["final_axis_z"]
    assert a["mean_abs_r_error"] == b["mean_abs_r_error"]
    assert a["mean_abs_z_error"] == b["mean_abs_z_error"]
    assert a["mean_abs_radial_actuator_lag"] == b["mean_abs_radial_actuator_lag"]
    assert a["mean_abs_vertical_actuator_lag"] == b["mean_abs_vertical_actuator_lag"]
    assert a["final_beta_scale"] == b["final_beta_scale"]
    assert a["mean_abs_heating_actuator_lag"] == b["mean_abs_heating_actuator_lag"]


def test_run_flight_sim_does_not_mutate_global_numpy_rng_state() -> None:
    """The run leaves the global numpy RNG state untouched."""
    np.random.seed(2468)
    state = np.random.get_state()

    run_flight_sim(
        config_file="dummy.json",
        shot_duration=10,
        seed=55,
        save_plot=False,
        verbose=False,
        kernel_factory=_DummyKernel,
    )

    observed = float(np.random.random())
    np.random.set_state(state)
    expected = float(np.random.random())
    assert observed == expected


def test_run_flight_sim_rejects_invalid_shot_duration() -> None:
    """A non-positive shot duration is rejected."""
    with pytest.raises(ValueError, match="shot_duration"):
        run_flight_sim(
            config_file="dummy.json",
            shot_duration=0,
            seed=1,
            save_plot=False,
            verbose=False,
            kernel_factory=_DummyKernel,
        )


def test_first_order_actuator_rejects_invalid_params() -> None:
    """Invalid actuator constants are rejected."""
    with pytest.raises(ValueError, match="tau_s"):
        FirstOrderActuator(tau_s=0.0, dt_s=0.05)
    with pytest.raises(ValueError, match="dt_s"):
        FirstOrderActuator(tau_s=0.05, dt_s=0.0)


def test_isoflux_controller_rejects_invalid_control_dt() -> None:
    """A non-positive control timestep is rejected."""
    with pytest.raises(ValueError, match="control_dt_s"):
        IsoFluxController(
            config_file="dummy.json",
            kernel_factory=_DummyKernel,
            verbose=False,
            control_dt_s=0.0,
        )


def test_isoflux_controller_rejects_invalid_heating_and_limit_controls() -> None:
    """Invalid heating-lag or current-limit controls are rejected."""
    with pytest.raises(ValueError, match="heating_actuator_tau_s"):
        IsoFluxController(
            config_file="dummy.json",
            kernel_factory=_DummyKernel,
            verbose=False,
            heating_actuator_tau_s=0.0,
        )
    with pytest.raises(ValueError, match="actuator_current_delta_limit"):
        IsoFluxController(
            config_file="dummy.json",
            kernel_factory=_DummyKernel,
            verbose=False,
            actuator_current_delta_limit=0.0,
        )
    with pytest.raises(ValueError, match="heating_beta_max"):
        IsoFluxController(
            config_file="dummy.json",
            kernel_factory=_DummyKernel,
            verbose=False,
            heating_beta_max=1.0,
        )


def test_run_flight_sim_heating_tau_controls_actuator_lag() -> None:
    """A larger heating time constant increases actuator lag."""
    fast = run_flight_sim(
        config_file="dummy.json",
        shot_duration=18,
        seed=10,
        save_plot=False,
        verbose=False,
        heating_actuator_tau_s=0.002,
        kernel_factory=_DummyKernel,
    )
    slow = run_flight_sim(
        config_file="dummy.json",
        shot_duration=18,
        seed=10,
        save_plot=False,
        verbose=False,
        heating_actuator_tau_s=0.5,
        kernel_factory=_DummyKernel,
    )
    assert fast["mean_abs_heating_actuator_lag"] < slow["mean_abs_heating_actuator_lag"]


def test_first_order_actuator_measurement_includes_noise() -> None:
    """A non-zero sensor-noise actuator returns a finite noisy measurement."""
    act = FirstOrderActuator(
        tau_s=0.05,
        dt_s=0.05,
        u_min=-1.0,
        u_max=1.0,
        rate_limit=10.0,
        sensor_noise_std=0.1,
        delay_steps=1,
    )
    act.step(0.5)
    measurement = act.get_measurement()
    assert np.isfinite(measurement)


def test_run_flight_sim_renders_plot(monkeypatch: pytest.MonkeyPatch) -> None:
    """The verbose plotting path renders the flight report and marks it saved."""
    import matplotlib.pyplot as plt

    saved: list[str] = []
    monkeypatch.setattr(plt, "savefig", lambda path, *a, **k: saved.append(str(path)))
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)

    summary = run_flight_sim(
        config_file="dummy.json",
        shot_duration=24,
        seed=5,
        save_plot=True,
        verbose=True,
        kernel_factory=_DummyKernel,
    )
    assert summary["plot_saved"] is True
    assert saved


def test_run_flight_sim_records_plot_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A flight-report render failure is caught and reported, not raised."""
    import matplotlib.pyplot as plt

    def _boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("backend down")

    monkeypatch.setattr(plt, "savefig", _boom)
    summary = run_flight_sim(
        config_file="dummy.json",
        shot_duration=24,
        seed=5,
        save_plot=True,
        verbose=True,
        kernel_factory=_DummyKernel,
    )
    assert summary["plot_saved"] is False


def test_first_order_actuator_measurement_without_noise() -> None:
    """A noiseless actuator returns its delayed measurement unchanged."""
    act = FirstOrderActuator(
        tau_s=0.05, dt_s=0.05, u_min=-1.0, u_max=1.0, rate_limit=10.0, sensor_noise_std=0.0
    )
    act.step(0.5)
    assert np.isfinite(act.get_measurement())


def test_run_flight_sim_resolves_default_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """A null config path is resolved to the default ITER configuration."""
    seen: list[str] = []

    def _factory(config_path: str) -> Any:
        seen.append(config_path)
        return _DummyKernel(config_path)

    run_flight_sim(
        config_file=None,
        shot_duration=12,
        seed=1,
        save_plot=False,
        verbose=False,
        kernel_factory=_factory,
    )
    assert seen and "iter_config" in seen[0]


class _LimiterKernel(_DummyKernel):
    """Kernel stub that never forms a divertor X-point, to exercise the limited-plasma panel."""

    def find_x_point(self, _psi: np.ndarray[Any, Any]) -> tuple[tuple[float, float], float]:
        """Report a limiter (origin) X-point, i.e. no divertor."""
        return (0.0, 0.0), 0.0


def test_run_flight_sim_renders_limited_plasma_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    """A never-diverted plasma renders the limited-plasma fallback panel."""
    import matplotlib.pyplot as plt

    saved: list[str] = []
    monkeypatch.setattr(plt, "savefig", lambda path, *a, **k: saved.append(str(path)))
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)
    summary = run_flight_sim(
        config_file="dummy.json",
        shot_duration=18,
        seed=2,
        save_plot=True,
        verbose=False,
        kernel_factory=_LimiterKernel,
    )
    assert summary["plot_saved"] is True
