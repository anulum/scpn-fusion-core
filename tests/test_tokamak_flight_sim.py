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
    CoilCurrentOffsetCommand,
    ControlObservation,
    FirstOrderActuator,
    IsoFluxController,
    map_axis_commands_to_coil_offsets,
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
        "disrupted",
        "t_disruption_s",
        "simulated_duration_s",
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
    assert summary["t_disruption_s"] <= summary["simulated_duration_s"]


def test_flight_sim_reports_first_disruption_time_from_trajectory() -> None:
    """Disruption timing is derived from the first threshold crossing, not a placeholder."""

    class OffsetTargetKernel(_DummyKernel):
        """Place the requested axis outside the dummy plant's reachable range."""

        def __init__(self, config_file: str) -> None:
            super().__init__(config_file)
            self.cfg["target"] = {"R_axis": 7.0, "Z_axis": 0.0}

    summary = run_flight_sim(
        config_file="dummy.json",
        shot_duration=8,
        seed=9,
        save_plot=False,
        verbose=False,
        control_dt_s=0.05,
        kernel_factory=OffsetTargetKernel,
    )

    assert summary["disrupted"] is True
    assert summary["t_disruption_s"] == pytest.approx(0.0)


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
    with pytest.raises(ValueError, match="actuator_current_offset_limit_ma"):
        IsoFluxController(
            config_file="dummy.json",
            kernel_factory=_DummyKernel,
            verbose=False,
            actuator_current_offset_limit_ma=0.0,
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


def test_first_order_actuator_applies_exact_command_transport_delay() -> None:
    """A command reaches the lag element only after the configured delay."""
    actuator = FirstOrderActuator(
        tau_s=1.0e-9,
        dt_s=1.0,
        u_min=-1.0,
        u_max=1.0,
        rate_limit=10.0,
        command_delay_steps=2,
    )

    assert actuator.step(1.0) == pytest.approx(0.0)
    assert actuator.step(1.0) == pytest.approx(0.0)
    assert actuator.step(1.0) == pytest.approx(1.0)


def test_isoflux_stress_scenario_is_seeded_and_reported() -> None:
    """The public runtime applies reproducible noise and reports exact delay units."""
    common: dict[str, Any] = {
        "config_file": "dummy.json",
        "shot_duration": 12,
        "save_plot": False,
        "verbose": False,
        "kernel_factory": _DummyKernel,
        "measurement_noise_std_m": 0.03,
        "actuator_delay_s": 0.10,
        "control_dt_s": 0.05,
    }
    first = run_flight_sim(seed=71, **common)
    replay = run_flight_sim(seed=71, **common)
    different_seed = run_flight_sim(seed=72, **common)

    assert first["measurement_noise_std_m"] == pytest.approx(0.03)
    assert first["actuator_delay_s"] == pytest.approx(0.10)
    assert first["actuator_delay_steps"] == 2
    assert first["realized_measurement_noise_rms_m"] > 0.0
    assert first["realized_measurement_noise_rms_m"] == replay["realized_measurement_noise_rms_m"]
    assert first["mean_abs_radial_actuator_lag"] == replay["mean_abs_radial_actuator_lag"]
    assert first["mean_abs_vertical_actuator_lag"] == replay["mean_abs_vertical_actuator_lag"]
    assert (first["mean_abs_radial_actuator_lag"], first["mean_abs_vertical_actuator_lag"]) != (
        different_seed["mean_abs_radial_actuator_lag"],
        different_seed["mean_abs_vertical_actuator_lag"],
    )


def test_stress_inputs_materially_change_consumed_noise_and_actuator_trajectory() -> None:
    """Noise amplitude and command delay each change their public runtime surface."""
    noiseless = run_flight_sim(
        "dummy.json",
        shot_duration=6,
        save_plot=False,
        verbose=False,
        kernel_factory=_DummyKernel,
        measurement_noise_std_m=0.0,
        seed=23,
    )
    noisy = run_flight_sim(
        "dummy.json",
        shot_duration=6,
        save_plot=False,
        verbose=False,
        kernel_factory=_DummyKernel,
        measurement_noise_std_m=0.03,
        seed=23,
    )
    assert noiseless["realized_measurement_noise_rms_m"] == 0.0
    assert noisy["realized_measurement_noise_rms_m"] > 0.0
    assert noiseless["disturbance_trace_digest"] != noisy["disturbance_trace_digest"]

    class ConstantPolicy:
        """Command one observable full-coil offset setpoint."""

        def step(self, _observation: ControlObservation) -> CoilCurrentOffsetCommand:
            return CoilCurrentOffsetCommand((0.0, 0.0, 0.04, 0.0, 0.0))

    controllers = [
        IsoFluxController(
            "dummy.json",
            kernel_factory=_DummyKernel,
            verbose=False,
            actuator_tau_s=1.0e-12,
            actuator_delay_s=delay,
            control_dt_s=0.05,
        )
        for delay in (0.0, 0.10)
    ]
    for controller in controllers:
        controller.run_shot(shot_duration=3, save_plot=False, control_policy=ConstantPolicy())
    immediate = np.asarray(controllers[0].history["coil_current_offset_applied_ma"])
    delayed = np.asarray(controllers[1].history["coil_current_offset_applied_ma"])
    assert immediate[:, 2] == pytest.approx((0.04, 0.04, 0.04))
    assert delayed[:, 2] == pytest.approx((0.0, 0.0, 0.04))


def test_isoflux_rejects_fractional_command_delay_step() -> None:
    """The runtime never silently rounds a requested physical delay."""
    with pytest.raises(ValueError, match="integer multiple"):
        IsoFluxController(
            "dummy.json",
            kernel_factory=_DummyKernel,
            control_dt_s=0.05,
            actuator_delay_s=0.075,
        )


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


def test_first_order_actuator_enforces_rate_limit() -> None:
    """A command exceeding the per-step slew budget is rate-limited to max_du."""
    act = FirstOrderActuator(tau_s=0.001, dt_s=1.0, u_min=-10.0, u_max=10.0, rate_limit=0.1)
    out = act.step(10.0)
    assert out == pytest.approx(0.1)  # max_du = rate_limit * dt_s


def test_first_order_actuator_set_delay_buffer_is_bounded() -> None:
    """set_delay_buffer keeps the delay line bounded regardless of input length."""
    act = FirstOrderActuator(
        tau_s=0.05, dt_s=0.05, u_min=-1.0, u_max=1.0, rate_limit=10.0, delay_steps=2
    )
    act.set_delay_buffer([1.0] * 50)
    assert len(act._delay_buffer) == act.delay_steps + 1
    assert act.get_measurement() == 1.0


def test_first_order_actuator_delay_buffer_stays_bounded() -> None:
    """The actuator delay buffer stays length-bounded across a long shot."""
    act = FirstOrderActuator(
        tau_s=0.05, dt_s=0.05, u_min=-1.0, u_max=1.0, rate_limit=10.0, delay_steps=3
    )
    for _ in range(5000):
        act.step(0.5)
    assert len(act._delay_buffer) <= act.delay_steps + 1
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


def test_first_order_actuator_holds_on_nonfinite_command() -> None:
    """A NaN command is held, not latched — one bad sample can't poison the actuator."""
    act = FirstOrderActuator(tau_s=0.05, dt_s=0.05, u_min=-10.0, u_max=10.0, rate_limit=100.0)
    act.step(2.0)
    held = act.state
    out = act.step(float("nan"))
    assert out == pytest.approx(held)  # last valid state held
    assert np.isfinite(act.state)
    assert act.faults == 1
    # Not poisoned: a subsequent finite command advances the actuator again.
    nxt = act.step(2.0)
    assert np.isfinite(nxt)
    assert act.faults == 1  # only the NaN sample counted


def test_first_order_actuator_holds_on_inf_command() -> None:
    act = FirstOrderActuator(tau_s=0.05, dt_s=0.05, u_min=-10.0, u_max=10.0, rate_limit=100.0)
    act.step(1.0)
    held = act.state
    out = act.step(float("inf"))
    assert out == pytest.approx(held)
    assert np.isfinite(out)
    assert act.faults == 1


def test_first_order_actuator_default_limits_are_physical() -> None:
    """The default offset saturation is 0.05 MA (50 kA), not a unit error."""
    act = FirstOrderActuator(tau_s=0.05, dt_s=0.05)
    assert act.u_max == pytest.approx(0.05)
    assert act.u_min == pytest.approx(-0.05)
    assert act.rate_limit == pytest.approx(1.0)


@pytest.mark.parametrize("lower", [None, -0.5])
@pytest.mark.parametrize("upper", [None, 0.5])
@pytest.mark.parametrize("rate", [None, 0.25])
@pytest.mark.parametrize("command_delay,measurement_delay", [(0, 0), (2, 3)])
def test_first_order_actuator_optional_limits_preserve_delayed_trajectory(
    lower: float | None,
    upper: float | None,
    rate: float | None,
    command_delay: int,
    measurement_delay: int,
) -> None:
    """Optional bounds preserve lag, slew and both delays against a scalar recurrence."""
    actuator = FirstOrderActuator(
        tau_s=0.05,
        dt_s=0.05,
        u_min=lower,
        u_max=upper,
        rate_limit=rate,
        command_delay_steps=command_delay,
        delay_steps=measurement_delay,
    )
    commands = [2.0] * 6 + [-3.0] * 8 + [0.0] * 5
    queued = [0.0] * command_delay
    states = [0.0] * measurement_delay
    state = 0.0
    for command in commands:
        bounded = command
        if lower is not None:
            bounded = max(lower, bounded)
        if upper is not None:
            bounded = min(upper, bounded)
        queued.append(bounded)
        delta = (queued.pop(0) - state) / 2.0
        if rate is not None:
            delta = max(-rate * 0.05, min(rate * 0.05, delta))
        state += delta
        states.append(state)
        assert actuator.step(command) == pytest.approx(state, abs=1e-15)
        assert actuator.get_measurement() == pytest.approx(states[-1 - measurement_delay])
    assert actuator.faults == 0


@pytest.mark.parametrize("parameter", ["u_min", "u_max", "rate_limit"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_first_order_actuator_rejects_nonfinite_numeric_limits(
    parameter: str, invalid: float
) -> None:
    """None is the only unbounded sentinel; numeric NaN and infinities still reject."""
    limits: dict[str, float | None] = {"u_min": None, "u_max": None, "rate_limit": None}
    limits[parameter] = invalid
    with pytest.raises(ValueError, match=parameter):
        FirstOrderActuator(
            tau_s=0.05,
            dt_s=0.05,
            u_min=limits["u_min"],
            u_max=limits["u_max"],
            rate_limit=limits["rate_limit"],
        )


@pytest.mark.parametrize(
    "lower,upper,rate", [(1.0, 1.0, None), (2.0, 1.0, None), (None, None, 0.0), (None, None, -1.0)]
)
def test_first_order_actuator_rejects_invalid_finite_limits(
    lower: float | None, upper: float | None, rate: float | None
) -> None:
    """Optional bounds do not admit reversed intervals or nonpositive slew rates."""
    with pytest.raises(ValueError):
        FirstOrderActuator(tau_s=0.05, dt_s=0.05, u_min=lower, u_max=upper, rate_limit=rate)


@pytest.mark.parametrize("command", [float("nan"), float("inf"), -float("inf")])
def test_first_order_actuator_unbounded_fault_holds_and_recovers(command: float) -> None:
    """An unbounded channel holds finite state on invalid input and resumes its lag."""
    actuator = FirstOrderActuator(
        tau_s=0.05, dt_s=0.05, u_min=None, u_max=None, rate_limit=None, delay_steps=1
    )
    assert actuator.step(4.0) == 2.0
    assert actuator.get_measurement() == 0.0
    assert actuator.step(command) == 2.0
    assert actuator.get_measurement() == 2.0
    assert actuator.faults == 1
    assert actuator.step(-4.0) == -1.0
    assert actuator.get_measurement() == 2.0
    assert actuator.faults == 1


@pytest.mark.parametrize(
    "noise,measurement_delay,command_delay,seed",
    [
        (float("nan"), 0, 0, None),
        (-0.1, 0, 0, None),
        (0.0, -1, 0, None),
        (0.0, True, 0, None),
        (0.0, 0, -1, None),
        (0.0, 0, True, None),
        (0.0, 0, 0, -1),
        (0.0, 0, 0, 2**64),
        (0.0, 0, 0, True),
    ],
)
def test_first_order_actuator_unbounded_limits_retain_configuration_guards(
    noise: float, measurement_delay: int, command_delay: int, seed: int | None
) -> None:
    """Unbounded channels still reject invalid noise, delays and random seeds."""
    with pytest.raises(ValueError):
        FirstOrderActuator(
            tau_s=0.05,
            dt_s=0.05,
            u_min=None,
            u_max=None,
            rate_limit=None,
            sensor_noise_std=noise,
            delay_steps=measurement_delay,
            command_delay_steps=command_delay,
            rng_seed=seed,
        )


def test_pid_step_ignores_nonfinite_error() -> None:
    """A non-finite error returns a safe zero command and never latches err_sum."""
    ctrl = IsoFluxController(config_file="dummy.json", kernel_factory=_DummyKernel, verbose=False)
    pid = {"Kp": 2.0, "Ki": 0.1, "Kd": 0.5, "err_sum": 3.0, "last_err": 0.5}
    out = ctrl.pid_step(pid, float("nan"))
    assert out == 0.0
    assert pid["err_sum"] == 3.0  # integrator untouched (not poisoned)
    assert pid["last_err"] == 0.5
    # A following finite error still accumulates normally.
    finite_out = ctrl.pid_step(pid, 1.0)
    assert np.isfinite(finite_out)
    assert pid["err_sum"] == 4.0


def test_isoflux_default_current_offset_limit_is_physical() -> None:
    """The controller's default current-offset saturation is 0.05 MA."""
    ctrl = IsoFluxController(config_file="dummy.json", kernel_factory=_DummyKernel, verbose=False)
    assert ctrl._act_radial.u_max == pytest.approx(0.05)


def test_axis_mapping_uses_pf3_and_differential_pf1_pf5_offsets() -> None:
    """The public axis adapter preserves the declared ITER PF-coil convention."""
    command = map_axis_commands_to_coil_offsets(7, 0.02, -0.01)
    assert command.coil_current_offsets_ma == pytest.approx((0.01, 0.0, 0.02, 0.0, -0.01, 0.0, 0.0))


def test_full_coil_policy_runs_once_per_step_from_shared_observation() -> None:
    """One policy call drives every coil through the same actuator transfer layer."""

    class RecordingPolicy:
        """Record observations and command fixed full-coil offsets."""

        def __init__(self) -> None:
            self.observations: list[ControlObservation] = []

        def step(self, observation: ControlObservation) -> CoilCurrentOffsetCommand:
            self.observations.append(observation)
            return CoilCurrentOffsetCommand((0.01, 0.02, 0.03, 0.04, 0.05))

    policy = RecordingPolicy()
    ctrl = IsoFluxController(
        config_file="dummy.json",
        kernel_factory=_DummyKernel,
        verbose=False,
        actuator_tau_s=1.0e-9,
        actuator_current_offset_limit_ma=0.05,
        control_dt_s=0.05,
        measurement_noise_std_m=0.01,
        rng_seed=17,
    )
    summary = ctrl.run_shot(shot_duration=3, save_plot=False, control_policy=policy)

    assert [item.step_index for item in policy.observations] == [0, 1, 2]
    assert all(item.control_dt_s == pytest.approx(0.05) for item in policy.observations)
    assert len(summary["control_policy_latency_us"]) == 3
    assert summary["mean_control_policy_latency_us"] >= 0.0
    assert summary["simulation_wall_time_us"] >= sum(summary["control_policy_latency_us"])
    assert ctrl.history["coil_current_offset_cmd_ma"][0] == pytest.approx(
        (0.01, 0.02, 0.03, 0.04, 0.05)
    )
    assert [coil["current"] for coil in ctrl.kernel.cfg["coils"]] == pytest.approx(
        (0.01, 0.02, 0.03, 0.04, 0.05), rel=1.0e-6, abs=1.0e-9
    )


def test_policy_offsets_are_relative_to_immutable_initial_currents() -> None:
    """A constant actuator offset is not accumulated into coil current every step."""

    class ConstantPolicy:
        """Return one fixed radial offset setpoint."""

        def step(self, _observation: ControlObservation) -> CoilCurrentOffsetCommand:
            return CoilCurrentOffsetCommand((0.0, 0.0, 0.04, 0.0, 0.0))

    ctrl = IsoFluxController(
        config_file="dummy.json",
        kernel_factory=_DummyKernel,
        verbose=False,
        actuator_tau_s=1.0e-9,
        actuator_current_offset_limit_ma=0.05,
        control_dt_s=0.05,
    )
    ctrl.run_shot(shot_duration=5, save_plot=False, control_policy=ConstantPolicy())

    assert ctrl.kernel.cfg["coils"][2]["current"] == pytest.approx(0.04, rel=1.0e-6)


def test_magnetic_actuator_metrics_use_ma_and_seconds() -> None:
    """Full-coil offset effort and command tracking retain physical dimensions."""

    class ConstantPolicy:
        """Return known full-coil offset setpoints."""

        def step(self, _observation: ControlObservation) -> CoilCurrentOffsetCommand:
            return CoilCurrentOffsetCommand((0.01, -0.02, 0.03, -0.04, 0.05))

    ctrl = IsoFluxController(
        config_file="dummy.json",
        kernel_factory=_DummyKernel,
        verbose=False,
        actuator_tau_s=1.0e-12,
        actuator_current_offset_limit_ma=0.05,
        control_dt_s=0.05,
    )
    summary = ctrl.run_shot(shot_duration=2, save_plot=False, control_policy=ConstantPolicy())

    expected_integral_ma_s = 2 * 0.05 * (0.01 + 0.02 + 0.03 + 0.04 + 0.05)
    assert summary["magnetic_actuator_absolute_current_offset_integral_ma_s"] == pytest.approx(
        expected_integral_ma_s, rel=1.0e-8
    )
    assert summary["mean_abs_coil_current_offset_tracking_error_ma"] == pytest.approx(
        0.0, abs=1.0e-10
    )


def test_disturbance_trace_digest_identifies_consumed_two_channel_trace() -> None:
    """The summary binds the exact materialized R/Z noise samples to a digest."""
    common: dict[str, Any] = {
        "config_file": "dummy.json",
        "shot_duration": 6,
        "save_plot": False,
        "verbose": False,
        "kernel_factory": _DummyKernel,
        "measurement_noise_std_m": 0.02,
    }
    first = run_flight_sim(seed=101, **common)
    replay = run_flight_sim(seed=101, **common)
    different = run_flight_sim(seed=102, **common)

    assert first["disturbance_trace_sample_count"] == 6
    assert len(first["disturbance_trace_digest"]) == 64
    assert first["disturbance_trace_digest"] == replay["disturbance_trace_digest"]
    assert first["disturbance_trace_digest"] != different["disturbance_trace_digest"]


def test_disruption_detects_final_post_actuation_state_at_shot_end() -> None:
    """A threshold crossing caused by the last command is timed at the shot endpoint."""

    class NearThresholdKernel(_DummyKernel):
        """Set a target that crosses the disruption threshold only after inward motion."""

        def __init__(self, config_file: str) -> None:
            super().__init__(config_file)
            self.cfg["target"] = {"R_axis": 6.58, "Z_axis": 0.0}

        def solve_equilibrium(self) -> None:
            """Resolve the dummy equilibrium with MA-scale PF3 sensitivity."""
            self._ticks += 1
            radial_drive_ma = float(self.cfg["coils"][2]["current"])
            center_r = 6.1 + 0.05 * np.tanh(radial_drive_ma / 0.005)
            ir = int(np.argmin(np.abs(self.R - center_r)))
            iz = int(np.argmin(np.abs(self.Z)))
            self.Psi.fill(-1.0)
            self.Psi[iz, ir] = 1.0

    class InwardPolicy:
        """Apply the maximum inward PF3 offset."""

        def step(self, observation: ControlObservation) -> CoilCurrentOffsetCommand:
            return map_axis_commands_to_coil_offsets(len(observation.coil_currents_ma), -0.05, 0.0)

    ctrl = IsoFluxController(
        config_file="dummy.json",
        kernel_factory=NearThresholdKernel,
        verbose=False,
        actuator_tau_s=1.0e-12,
        control_dt_s=0.05,
    )
    summary = ctrl.run_shot(shot_duration=1, save_plot=False, control_policy=InwardPolicy())

    assert summary["disrupted"] is True
    assert summary["t_disruption_s"] == pytest.approx(0.05)


@pytest.mark.parametrize(
    "command",
    [
        CoilCurrentOffsetCommand((0.0, 0.0)),
        CoilCurrentOffsetCommand((0.0, 0.0, float("nan"), 0.0, 0.0)),
    ],
)
def test_policy_rejects_incomplete_or_nonfinite_full_coil_commands(
    command: CoilCurrentOffsetCommand,
) -> None:
    """The public policy boundary fails closed on invalid full-coil vectors."""

    class InvalidPolicy:
        """Return the supplied invalid command through the public policy surface."""

        def step(self, _observation: ControlObservation) -> CoilCurrentOffsetCommand:
            return command

    ctrl = IsoFluxController(config_file="dummy.json", kernel_factory=_DummyKernel, verbose=False)
    with pytest.raises(ValueError, match="control policy"):
        ctrl.run_shot(shot_duration=1, save_plot=False, control_policy=InvalidPolicy())
