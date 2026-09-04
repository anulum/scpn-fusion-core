# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tokamak Flight Sim
"""Tokamak flight simulator with actuator dynamics and isoflux feedback."""

from __future__ import annotations

import hashlib
import logging
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_fusion._data_paths import default_iter_config_path

logger = logging.getLogger(__name__)

from scpn_fusion.core.fusion_kernel import FusionKernel

SHOT_DURATION = 50
DEFAULT_TARGET_R = 6.2
DEFAULT_TARGET_Z = 0.0
TARGET_ELONGATION = 1.7
FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class ControlObservation:
    """Measured plant state supplied to one control-policy evaluation.

    Attributes
    ----------
    step_index : int
        Zero-based control-step index.
    time_s, control_dt_s : float
        Shot time and control period in seconds.
    measured_axis_r_m, measured_axis_z_m : float
        Noisy magnetic-axis observations in metres.
    target_axis_r_m, target_axis_z_m : float
        Magnetic-axis targets in metres.
    x_point_r_m, x_point_z_m : float
        X-point coordinates in metres from the current equilibrium solve.
    coil_currents_ma : tuple of float
        Coil currents before the command for this step, in mega-amperes.
    """

    step_index: int
    time_s: float
    control_dt_s: float
    measured_axis_r_m: float
    measured_axis_z_m: float
    target_axis_r_m: float
    target_axis_z_m: float
    x_point_r_m: float
    x_point_z_m: float
    coil_currents_ma: tuple[float, ...]

    @property
    def radial_error_m(self) -> float:
        """Return target-minus-measurement radial error in metres."""
        return self.target_axis_r_m - self.measured_axis_r_m

    @property
    def vertical_error_m(self) -> float:
        """Return target-minus-measurement vertical error in metres."""
        return self.target_axis_z_m - self.measured_axis_z_m


@dataclass(frozen=True)
class CoilCurrentOffsetCommand:
    """Full-coil current-offset command produced once per control step.

    ``coil_current_offsets_ma`` must contain one finite offset, in mega-amperes,
    for every coil in the plant configuration. Each offset is relative to the
    immutable initial coil-current vector. The simulator applies command delay,
    lag, slew limits, and saturation independently to every element.
    """

    coil_current_offsets_ma: tuple[float, ...]


class ControlPolicy(Protocol):
    """Stateful controller contract shared by all Python flight-sim lanes."""

    def step(self, observation: ControlObservation) -> CoilCurrentOffsetCommand:
        """Return one full-coil command for one measured plant observation."""
        ...


def map_axis_commands_to_coil_offsets(
    n_coils: int,
    radial_command_ma: float,
    vertical_command_ma: float,
) -> CoilCurrentOffsetCommand:
    """Map axis commands onto the flight simulator's ITER PF-coil convention.

    Radial control acts through PF3 (index 2). Vertical control is a
    differential pair: ``-vertical`` on the top PF coil (index 0) and
    ``+vertical`` on the bottom PF coil (index 4). Other coils receive a zero
    offset. At least five coils are therefore required. Inputs and outputs are
    in mega-amperes.
    """
    if isinstance(n_coils, bool) or not isinstance(n_coils, int) or n_coils < 5:
        raise ValueError("n_coils must be an integer >= 5 for the ITER PF mapping.")
    radial = float(radial_command_ma)
    vertical = float(vertical_command_ma)
    if not np.isfinite(radial) or not np.isfinite(vertical):
        raise ValueError("axis commands must be finite.")
    offsets = np.zeros(n_coils, dtype=np.float64)
    offsets[2] = radial
    offsets[0] = -vertical
    offsets[4] = vertical
    return CoilCurrentOffsetCommand(tuple(float(value) for value in offsets))


class _PidAxisPolicy:
    """Adapt the built-in radial/vertical PID state to the full-coil contract."""

    def __init__(self, controller: IsoFluxController) -> None:
        self._controller = controller

    def step(self, observation: ControlObservation) -> CoilCurrentOffsetCommand:
        radial = self._controller.pid_step(self._controller.pid_R, observation.radial_error_m)
        vertical = self._controller.pid_step(self._controller.pid_Z, observation.vertical_error_m)
        return map_axis_commands_to_coil_offsets(
            len(observation.coil_currents_ma), radial, vertical
        )


class FirstOrderActuator:
    """Discrete first-order actuator with rate limits, noise, and delay.

    Models a realistic coil power supply for tokamak control:
    - First-order lag: u_applied(s) = 1/(tau*s+1) * u_cmd
    - Rate limit: abs(du/dt) <= rate_limit in command units per second
    - Sensor noise: additive Gaussian on measurement
    - Measurement delay: pure transport delay on feedback signal

    Parameters
    ----------
    tau_s : float
        Actuator time constant [s].
    dt_s : float
        Simulation timestep [s].
    u_min, u_max : float or None
        Finite saturation limits in command units; ``None`` disables that side
        for an unbounded simulation. Defaults are +/-0.05 MA (50 kA) for
        flight-simulator current offsets. Numeric NaN and infinity are invalid.
    rate_limit : float or None
        Positive finite maximum change in command units per second, or ``None``
        for unbounded simulated slew. Default 1.0 MA/s for flight offsets.
    sensor_noise_std : float
        Standard deviation of additive sensor noise. Default 0.0 (disabled).
    delay_steps : int
        Number of timesteps of measurement delay. Default 0.
    command_delay_steps : int
        Number of timesteps of pure command transport delay before lag.
        Default 0.
    rng_seed : int or None
        Random seed for reproducible noise (None = random).

    Notes
    -----
    Commands, state, saturation and sensor noise must use the same units.
    The flight simulator uses MA offsets, free-boundary tracking uses A, and
    the heating channel uses dimensionless beta. No unit conversion occurs
    here. Unbounded simulation channels do not define physical device limits.

    Raises
    ------
    ValueError
        If a supplied numeric limit is nonfinite, finite bounds are unordered,
        the slew rate is nonpositive, or timing/noise configuration is invalid.

    """

    def __init__(
        self,
        *,
        tau_s: float,
        dt_s: float,
        u_min: float | None = -0.05,
        u_max: float | None = 0.05,
        rate_limit: float | None = 1.0,
        sensor_noise_std: float = 0.0,
        delay_steps: int = 0,
        command_delay_steps: int = 0,
        rng_seed: Optional[int] = None,
    ) -> None:
        """Validate the actuator constants and initialise the delay buffer."""
        tau_s = float(tau_s)
        dt_s = float(dt_s)
        if not np.isfinite(tau_s) or tau_s <= 0.0:
            raise ValueError("tau_s must be finite and > 0.")
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and > 0.")
        self.tau_s = tau_s
        self.dt_s = dt_s
        self.u_min = None if u_min is None else float(u_min)
        self.u_max = None if u_max is None else float(u_max)
        if self.u_min is not None and not np.isfinite(self.u_min):
            raise ValueError("u_min must be finite or None.")
        if self.u_max is not None and not np.isfinite(self.u_max):
            raise ValueError("u_max must be finite or None.")
        if self.u_min is not None and self.u_max is not None and self.u_min >= self.u_max:
            raise ValueError("u_min must be less than u_max when both are supplied.")
        self.rate_limit = None if rate_limit is None else float(rate_limit)
        if self.rate_limit is not None and (
            not np.isfinite(self.rate_limit) or self.rate_limit <= 0.0
        ):
            raise ValueError("rate_limit must be finite and > 0, or None.")
        self.sensor_noise_std = float(sensor_noise_std)
        if not np.isfinite(self.sensor_noise_std) or self.sensor_noise_std < 0.0:
            raise ValueError("sensor_noise_std must be finite and >= 0.")
        if isinstance(delay_steps, bool) or not isinstance(delay_steps, int) or delay_steps < 0:
            raise ValueError("delay_steps must be an integer >= 0.")
        self.delay_steps = delay_steps
        if (
            isinstance(command_delay_steps, bool)
            or not isinstance(command_delay_steps, int)
            or command_delay_steps < 0
        ):
            raise ValueError("command_delay_steps must be an integer >= 0.")
        self.command_delay_steps = command_delay_steps
        if rng_seed is not None and (
            isinstance(rng_seed, bool) or not isinstance(rng_seed, int) or not 0 <= rng_seed < 2**64
        ):
            raise ValueError("rng_seed must be an unsigned 64-bit integer or None.")
        self._rng = np.random.default_rng(rng_seed)
        self.state = 0.0
        self.faults = 0
        # Bounded ring buffer: holding delay_steps + 1 samples is sufficient for
        # the tail-indexed delayed read and keeps memory flat across a long shot.
        self._delay_buffer: deque[float] = deque(
            [0.0] * max(self.delay_steps, 1), maxlen=self.delay_steps + 1
        )
        self._command_buffer: deque[float] = deque()

    def step(self, command: float) -> float:
        """Apply command through actuator dynamics with rate limiting.

        A non-finite command (NaN/inf) is a fault the actuator cannot realise;
        the last valid state is held (fail-safe hold) and counted in ``faults``
        rather than being latched into ``self.state`` — one bad sample can never
        poison the actuator. The delay line still advances so measurement timing
        stays consistent.
        """
        if not np.isfinite(command):
            self.faults += 1
            self._delay_buffer.append(self.state)
            return self.state

        bounded_command = (
            float(np.clip(command, self.u_min, self.u_max))
            if self.u_min is not None or self.u_max is not None
            else float(command)
        )
        self._command_buffer.append(bounded_command)
        u_cmd = (
            self._command_buffer.popleft()
            if len(self._command_buffer) > self.command_delay_steps
            else 0.0
        )
        alpha = self.dt_s / (self.tau_s + self.dt_s)
        u_new = self.state + alpha * (u_cmd - self.state)

        # Rate limiting (coil current slew rate)
        du = u_new - self.state
        if self.rate_limit is not None:
            max_du = self.rate_limit * self.dt_s
            if abs(du) > max_du:
                du = np.sign(du) * max_du
                u_new = self.state + du

        self.state = (
            float(np.clip(u_new, self.u_min, self.u_max))
            if self.u_min is not None or self.u_max is not None
            else float(u_new)
        )

        # Update delay buffer
        self._delay_buffer.append(self.state)

        return self.state

    def get_measurement(self) -> float:
        """Return delayed, noisy measurement of actuator output."""
        idx = max(0, len(self._delay_buffer) - 1 - self.delay_steps)
        delayed = self._delay_buffer[idx]

        if self.sensor_noise_std > 0:
            noise = float(self._rng.normal(0.0, self.sensor_noise_std))
            return delayed + noise
        return delayed

    def set_delay_buffer(self, values: Iterable[float]) -> None:
        """Replace the delay line with *values*, keeping it length-bounded."""
        self._delay_buffer = deque(values, maxlen=self.delay_steps + 1)


class IsoFluxController:
    """Simulate a tokamak plasma-position control system.

    Position estimates may include deterministic seeded measurement noise.
    PID or substituted controller outputs pass through saturation, rate limits,
    a pure command delay, and first-order power-supply lag before they modify
    coil currents.
    """

    def __init__(
        self,
        config_file: str,
        kernel_factory: Callable[[str], Any] = FusionKernel,
        verbose: bool = True,
        actuator_tau_s: float = 0.06,
        heating_actuator_tau_s: Optional[float] = None,
        actuator_current_offset_limit_ma: float = 0.05,
        heating_beta_max: float = 5.0,
        control_dt_s: float = 0.05,
        measurement_noise_std_m: float = 0.0,
        actuator_delay_s: float = 0.0,
        rng_seed: Optional[int] = None,
    ) -> None:
        """Build the kernel, control gains, actuators, and telemetry history.

        Parameters
        ----------
        config_file : str
            Fusion-kernel machine configuration path.
        kernel_factory : callable
            Callable constructing the equilibrium kernel from ``config_file``.
        verbose : bool
            Emit per-step simulator logging when true.
        actuator_tau_s, heating_actuator_tau_s : float
            Magnetic and heating actuator lag constants, in seconds.
        actuator_current_offset_limit_ma : float
            Absolute magnetic-coil setpoint-offset bound, in mega-amperes.
        heating_beta_max : float
            Upper bound for the normalized heating scale.
        control_dt_s : float
            Control period in seconds.
        measurement_noise_std_m : float
            Gaussian standard deviation applied independently to R/Z position
            measurements before control, in metres.
        actuator_delay_s : float
            Pure command transport delay in seconds; it must be an integer
            multiple of ``control_dt_s``.
        rng_seed : int or None
            Master seed for independent radial and vertical noise streams.
        """
        self.kernel = kernel_factory(config_file)
        self.verbose = bool(verbose)
        self.history: dict[str, list[Any]] = {
            "t": [],
            "Ip": [],
            "R_axis": [],
            "Z_axis": [],
            "R_axis_measured": [],
            "Z_axis_measured": [],
            "X_point": [],
            "ctrl_R_cmd": [],
            "ctrl_R_applied": [],
            "ctrl_Z_cmd": [],
            "ctrl_Z_applied": [],
            "coil_current_offset_cmd_ma": [],
            "coil_current_offset_applied_ma": [],
            "control_policy_latency_us": [],
            "beta_cmd": [],
            "beta_applied": [],
        }
        control_dt_s = float(control_dt_s)
        if not np.isfinite(control_dt_s) or control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be finite and > 0.")
        self.control_dt_s = control_dt_s
        measurement_noise_std_m = float(measurement_noise_std_m)
        if not np.isfinite(measurement_noise_std_m) or measurement_noise_std_m < 0.0:
            raise ValueError("measurement_noise_std_m must be finite and >= 0.")
        actuator_delay_s = float(actuator_delay_s)
        if not np.isfinite(actuator_delay_s) or actuator_delay_s < 0.0:
            raise ValueError("actuator_delay_s must be finite and >= 0.")
        delay_steps_float = actuator_delay_s / control_dt_s
        delay_steps = int(round(delay_steps_float))
        if not np.isclose(delay_steps_float, delay_steps, rtol=0.0, atol=1.0e-12):
            raise ValueError("actuator_delay_s must be an integer multiple of control_dt_s.")
        self.measurement_noise_std_m = measurement_noise_std_m
        self.actuator_delay_s = actuator_delay_s
        self.actuator_delay_steps = delay_steps
        if rng_seed is not None and (
            isinstance(rng_seed, bool) or not isinstance(rng_seed, int) or not 0 <= rng_seed < 2**64
        ):
            raise ValueError("rng_seed must be an unsigned 64-bit integer or None.")
        self.rng_seed = rng_seed
        seed_sequence = np.random.SeedSequence(self.rng_seed)
        radial_seed, vertical_seed = seed_sequence.spawn(2)
        self._radial_measurement_rng = np.random.default_rng(radial_seed)
        self._vertical_measurement_rng = np.random.default_rng(vertical_seed)
        self.measured_R = float("nan")
        self.measured_Z = float("nan")
        actuator_current_offset_limit_ma = float(actuator_current_offset_limit_ma)
        if (
            not np.isfinite(actuator_current_offset_limit_ma)
            or actuator_current_offset_limit_ma <= 0.0
        ):
            raise ValueError("actuator_current_offset_limit_ma must be finite and > 0.")
        heating_beta_max = float(heating_beta_max)
        if not np.isfinite(heating_beta_max) or heating_beta_max <= 1.0:
            raise ValueError("heating_beta_max must be finite and > 1.0.")
        if heating_actuator_tau_s is None:
            heating_actuator_tau_s = float(actuator_tau_s)
        heating_actuator_tau_s = float(heating_actuator_tau_s)
        if not np.isfinite(heating_actuator_tau_s) or heating_actuator_tau_s <= 0.0:
            raise ValueError("heating_actuator_tau_s must be finite and > 0.")

        # PID Gains for Position Control
        # Radial Control (Horizontal) -> Controlled by Outer Coils (PF2, PF3, PF4)
        self.pid_R = {"Kp": 2.0, "Ki": 0.1, "Kd": 0.5, "err_sum": 0, "last_err": 0}

        # Vertical Control (Z-pos) -> Controlled by Top/Bottom diff (PF1 vs PF5)
        self.pid_Z = {"Kp": 5.0, "Ki": 0.2, "Kd": 2.0, "err_sum": 0, "last_err": 0}

        coils = self.kernel.cfg.get("coils", [])
        if not isinstance(coils, list) or len(coils) < 5:
            raise ValueError("flight-sim control requires at least five configured coils.")
        self._initial_coil_currents_ma = tuple(float(coil.get("current", 0.0)) for coil in coils)
        if not np.all(np.isfinite(self._initial_coil_currents_ma)):
            raise ValueError("configured coil currents must be finite mega-ampere values.")
        self._coil_actuators = [
            FirstOrderActuator(
                tau_s=actuator_tau_s,
                dt_s=self.control_dt_s,
                u_min=-actuator_current_offset_limit_ma,
                u_max=actuator_current_offset_limit_ma,
                command_delay_steps=delay_steps,
            )
            for _ in coils
        ]
        self._act_top = self._coil_actuators[0]
        self._act_radial = self._coil_actuators[2]
        self._act_bottom = self._coil_actuators[4]
        self._act_heating = FirstOrderActuator(
            tau_s=heating_actuator_tau_s,
            dt_s=self.control_dt_s,
            u_min=1.0,
            u_max=heating_beta_max,
            command_delay_steps=delay_steps,
        )

        target = self.kernel.cfg.get("target", {})
        self.target_R = float(target.get("R_axis", DEFAULT_TARGET_R))
        self.target_Z = float(target.get("Z_axis", DEFAULT_TARGET_Z))

    def _log(self, message: str) -> None:
        if self.verbose:
            logger.info(message)

    def pid_step(self, pid: Dict[str, float], error: float) -> float:
        """Update one PID state dictionary and return its control command.

        A non-finite error is a sensor/estimate fault: the integrator is NOT
        accumulated (so one NaN can never latch ``err_sum``) and a zero command
        is returned — a fail-safe hold rather than a poisoned controller.
        """
        if not np.isfinite(error):
            return 0.0
        pid["err_sum"] += error
        d_err = error - pid["last_err"]
        pid["last_err"] = error
        return (pid["Kp"] * error) + (pid["Ki"] * pid["err_sum"]) + (pid["Kd"] * d_err)

    def _set_coil_current_offset(self, coil_idx: int, offset_ma: float) -> None:
        """Set one coil to its immutable initial current plus an offset."""
        coils = self.kernel.cfg.get("coils", [])
        if 0 <= coil_idx < len(coils):
            coils[coil_idx]["current"] = self._initial_coil_currents_ma[coil_idx] + float(offset_ma)

    def _axis_position(self) -> tuple[float, float]:
        """Return the magnetic-axis position with sub-grid interpolation."""
        idx_max = int(np.argmax(self.kernel.Psi))
        iz, ir = np.unravel_index(idx_max, self.kernel.Psi.shape)
        curr_R = float(self.kernel.R[ir])
        curr_Z = float(self.kernel.Z[iz])
        psi = self.kernel.Psi
        if 1 <= ir <= self.kernel.NR - 2:
            a, b, c = psi[iz, ir - 1], psi[iz, ir], psi[iz, ir + 1]
            denom = 2.0 * (a - 2.0 * b + c)
            if abs(denom) > 1e-30:
                curr_R += float(np.clip(-(c - a) / denom, -0.5, 0.5)) * self.kernel.dR
        if 1 <= iz <= self.kernel.NZ - 2:
            a, b, c = psi[iz - 1, ir], psi[iz, ir], psi[iz + 1, ir]
            denom = 2.0 * (a - 2.0 * b + c)
            if abs(denom) > 1e-30:
                curr_Z += float(np.clip(-(c - a) / denom, -0.5, 0.5)) * self.kernel.dZ
        return curr_R, curr_Z

    def _validate_policy_command(
        self,
        command: CoilCurrentOffsetCommand,
    ) -> FloatArray:
        if not isinstance(command, CoilCurrentOffsetCommand):
            raise TypeError("control policy must return CoilCurrentOffsetCommand.")
        offsets = np.asarray(command.coil_current_offsets_ma, dtype=np.float64)
        expected_shape = (len(self._coil_actuators),)
        if offsets.shape != expected_shape:
            raise ValueError(
                "control policy must return exactly one coil-current offset "
                f"per configured coil; expected {expected_shape[0]}, got {offsets.size}."
            )
        if not np.all(np.isfinite(offsets)):
            raise ValueError("control policy coil-current offsets must be finite.")
        return offsets

    def _materialize_measurement_noise(self, steps: int) -> tuple[FloatArray, str]:
        """Create and hash the exact two-channel R/Z disturbance trace."""
        trace = np.empty((steps, 2), dtype=np.float64)
        trace[:, 0] = self._radial_measurement_rng.normal(
            0.0, self.measurement_noise_std_m, size=steps
        )
        trace[:, 1] = self._vertical_measurement_rng.normal(
            0.0, self.measurement_noise_std_m, size=steps
        )
        canonical = np.ascontiguousarray(trace, dtype="<f8")
        header = (
            "scpn-fusion-position-noise-v1\n"
            f"shape={steps},2\n"
            f"sample_period_s={self.control_dt_s.hex()}\n"
            "channels=R_axis_m,Z_axis_m\n"
            "dtype=<f8\n"
        ).encode("ascii")
        digest = hashlib.sha256(header + canonical.tobytes(order="C")).hexdigest()
        return trace, digest

    def run_shot(
        self,
        shot_duration: int = 30,
        save_plot: bool = True,
        output_path: str = "Tokamak_Flight_Report.png",
        control_policy: Optional[ControlPolicy] = None,
    ) -> Dict[str, Any]:
        """Run a simulated tokamak shot.

        Parameters
        ----------
        shot_duration : int
            Number of simulation steps. Default 30.
        save_plot : bool
            Whether to generate a summary plot.
        output_path : str
            Filename for the plot.
        control_policy : ControlPolicy or None
            Stateful controller evaluated exactly once per step. It receives
            the common measured observation and must return one current-offset
            command per configured coil. ``None`` selects the built-in
            two-axis PID mapped onto the ITER PF3/PF1/PF5 convention.

        """
        steps = int(shot_duration)
        if steps < 1:
            raise ValueError("shot_duration must be >= 1.")
        self._log(f"--- INITIATING TOKAMAK FLIGHT SIMULATOR ({steps} steps) ---")
        self._log(f"Scenario: Current Ramp-Up & Divertor Formation (dt={self.control_dt_s}s)")
        active_policy = control_policy if control_policy is not None else _PidAxisPolicy(self)
        measurement_noise, disturbance_trace_digest = self._materialize_measurement_noise(steps)

        # Initial Solve
        simulation_start_ns = time.perf_counter_ns()
        self.kernel.solve_equilibrium()

        Ip_cfg = float(self.kernel.cfg["physics"]["plasma_current_target"])

        # Physics Evolution Loop
        for t in range(steps):
            time_s = t * self.control_dt_s
            target_Ip = Ip_cfg * (0.98 + 0.02 * t / steps)
            physics_cfg = self.kernel.cfg.setdefault("physics", {})
            physics_cfg["plasma_current_target"] = target_Ip

            # Heating ramp — drives outward Shafranov shift
            beta_cmd = 1.0 + (0.002 * t)
            beta_applied = self._act_heating.step(beta_cmd)

            physics_cfg["beta_scale"] = beta_applied

            curr_R, curr_Z = self._axis_position()

            xp_pos, _ = self.kernel.find_x_point(self.kernel.Psi)

            measured_R = curr_R + float(measurement_noise[t, 0])
            measured_Z = curr_Z + float(measurement_noise[t, 1])
            self.measured_R = measured_R
            self.measured_Z = measured_Z
            observation = ControlObservation(
                step_index=t,
                time_s=time_s,
                control_dt_s=self.control_dt_s,
                measured_axis_r_m=measured_R,
                measured_axis_z_m=measured_Z,
                target_axis_r_m=self.target_R,
                target_axis_z_m=self.target_Z,
                x_point_r_m=float(xp_pos[0]),
                x_point_z_m=float(xp_pos[1]),
                coil_currents_ma=tuple(
                    float(coil.get("current", 0.0)) for coil in self.kernel.cfg["coils"]
                ),
            )
            policy_start_ns = time.perf_counter_ns()
            policy_command = active_policy.step(observation)
            policy_latency_us = (time.perf_counter_ns() - policy_start_ns) / 1.0e3
            coil_offset_cmd_ma = self._validate_policy_command(policy_command)
            ctrl_radial_cmd = float(coil_offset_cmd_ma[2])
            ctrl_vertical_cmd = 0.5 * float(coil_offset_cmd_ma[4] - coil_offset_cmd_ma[0])

            applied_offsets_ma: list[float] = []
            for actuator, offset_command_ma in zip(
                self._coil_actuators, coil_offset_cmd_ma, strict=True
            ):
                applied_offsets_ma.append(actuator.step(float(offset_command_ma)))
            coil_offset_applied_ma = np.asarray(applied_offsets_ma, dtype=np.float64)
            ctrl_radial = float(coil_offset_applied_ma[2])
            ctrl_vertical_applied = 0.5 * float(
                coil_offset_applied_ma[4] - coil_offset_applied_ma[0]
            )
            for coil_idx, offset_ma in enumerate(coil_offset_applied_ma):
                self._set_coil_current_offset(coil_idx, float(offset_ma))

            self.kernel.solve_equilibrium()

            self.history["t"].append(t)
            self.history["Ip"].append(target_Ip)
            self.history["R_axis"].append(curr_R)
            self.history["Z_axis"].append(curr_Z)
            self.history["R_axis_measured"].append(measured_R)
            self.history["Z_axis_measured"].append(measured_Z)
            self.history["X_point"].append(xp_pos)
            self.history["ctrl_R_cmd"].append(ctrl_radial_cmd)
            self.history["ctrl_R_applied"].append(ctrl_radial)
            self.history["ctrl_Z_cmd"].append(ctrl_vertical_cmd)
            self.history["ctrl_Z_applied"].append(ctrl_vertical_applied)
            self.history["coil_current_offset_cmd_ma"].append(coil_offset_cmd_ma.copy())
            self.history["coil_current_offset_applied_ma"].append(coil_offset_applied_ma.copy())
            self.history["control_policy_latency_us"].append(policy_latency_us)
            self.history["beta_cmd"].append(beta_cmd)
            self.history["beta_applied"].append(beta_applied)

            self._log(
                f"Time {time_s:.2f}s (Step {t}): Ip={target_Ip:.1f}MA | "
                f"Axis=({curr_R:.2f}, {curr_Z:.2f}) | XP=({xp_pos[0]:.2f}, {xp_pos[1]:.2f}) | Ctrl_R={ctrl_radial:.2f} | Psi_max={np.max(self.kernel.Psi):.2f}"
            )

        simulation_wall_time_us = (time.perf_counter_ns() - simulation_start_ns) / 1.0e3

        plot_saved = False
        plot_error = None
        if save_plot:
            plot_saved, plot_error = self.visualize_flight(output_path=output_path)

        final_axis_r, final_axis_z = self._axis_position()
        final_ip_ma = float(self.history["Ip"][-1]) if self.history["Ip"] else 0.0
        radial_errors = np.abs(
            np.append(np.asarray(self.history["R_axis"], dtype=np.float64), final_axis_r)
            - self.target_R
        )
        vertical_errors = np.abs(
            np.append(np.asarray(self.history["Z_axis"], dtype=np.float64), final_axis_z)
            - self.target_Z
        )
        mean_abs_r_error = float(
            (0.5 * radial_errors[0] + np.sum(radial_errors[1:-1]) + 0.5 * radial_errors[-1]) / steps
        )
        mean_abs_z_error = float(
            (0.5 * vertical_errors[0] + np.sum(vertical_errors[1:-1]) + 0.5 * vertical_errors[-1])
            / steps
        )
        disruption_samples = np.flatnonzero((radial_errors > 0.5) | (vertical_errors > 0.5))
        disrupted = bool(disruption_samples.size)
        t_disruption_s = (
            int(disruption_samples[0]) * self.control_dt_s
            if disrupted
            else steps * self.control_dt_s
        )
        mean_abs_radial_actuator_lag = (
            float(
                np.mean(
                    np.abs(
                        np.asarray(self.history["ctrl_R_cmd"], dtype=np.float64)
                        - np.asarray(self.history["ctrl_R_applied"], dtype=np.float64)
                    )
                )
            )
            if self.history["ctrl_R_cmd"]
            else 0.0
        )
        mean_abs_vertical_actuator_lag = (
            float(
                np.mean(
                    np.abs(
                        np.asarray(self.history["ctrl_Z_cmd"], dtype=np.float64)
                        - np.asarray(self.history["ctrl_Z_applied"], dtype=np.float64)
                    )
                )
            )
            if self.history["ctrl_Z_cmd"]
            else 0.0
        )
        mean_abs_heating_actuator_lag = (
            float(
                np.mean(
                    np.abs(
                        np.asarray(self.history["beta_cmd"], dtype=np.float64)
                        - np.asarray(self.history["beta_applied"], dtype=np.float64)
                    )
                )
            )
            if self.history["beta_cmd"]
            else 0.0
        )
        coil_commands = np.asarray(self.history["coil_current_offset_cmd_ma"], dtype=np.float64)
        coil_applied = np.asarray(self.history["coil_current_offset_applied_ma"], dtype=np.float64)
        mean_abs_coil_current_offset_tracking_error_ma = float(
            np.mean(np.abs(coil_commands - coil_applied))
        )
        magnetic_actuator_absolute_current_offset_integral_ma_s = float(
            self.control_dt_s * np.sum(np.abs(coil_applied))
        )
        control_policy_latency_us = [
            float(value) for value in self.history["control_policy_latency_us"]
        ]
        final_beta_scale = (
            float(self.history["beta_applied"][-1]) if self.history["beta_applied"] else 1.0
        )
        radial_noise = np.asarray(self.history["R_axis_measured"]) - np.asarray(
            self.history["R_axis"]
        )
        vertical_noise = np.asarray(self.history["Z_axis_measured"]) - np.asarray(
            self.history["Z_axis"]
        )
        realized_measurement_noise_rms_m = float(
            np.sqrt(np.mean(np.concatenate((radial_noise**2, vertical_noise**2))))
        )
        return {
            "steps": int(steps),
            "final_ip_ma": final_ip_ma,
            "final_axis_r": final_axis_r,
            "final_axis_z": final_axis_z,
            "final_beta_scale": final_beta_scale,
            "mean_abs_r_error": mean_abs_r_error,
            "mean_abs_z_error": mean_abs_z_error,
            "mean_abs_r_error_m": mean_abs_r_error,
            "mean_abs_z_error_m": mean_abs_z_error,
            "disrupted": disrupted,
            "t_disruption_s": t_disruption_s,
            "simulated_duration_s": steps * self.control_dt_s,
            "mean_abs_radial_actuator_lag": mean_abs_radial_actuator_lag,
            "mean_abs_vertical_actuator_lag": mean_abs_vertical_actuator_lag,
            "mean_abs_coil_current_offset_tracking_error_ma": (
                mean_abs_coil_current_offset_tracking_error_ma
            ),
            "magnetic_actuator_absolute_current_offset_integral_ma_s": (
                magnetic_actuator_absolute_current_offset_integral_ma_s
            ),
            "control_policy_latency_us": control_policy_latency_us,
            "mean_control_policy_latency_us": float(np.mean(control_policy_latency_us)),
            "simulation_wall_time_us": simulation_wall_time_us,
            "mean_abs_heating_actuator_lag": mean_abs_heating_actuator_lag,
            "plot_saved": bool(plot_saved),
            "plot_error": plot_error,
            "measurement_noise_std_m": self.measurement_noise_std_m,
            "actuator_delay_s": self.actuator_delay_s,
            "actuator_delay_steps": self.actuator_delay_steps,
            "rng_seed": self.rng_seed,
            "realized_measurement_noise_rms_m": realized_measurement_noise_rms_m,
            "disturbance_trace_digest": disturbance_trace_digest,
            "disturbance_trace_sample_count": steps,
        }

    def visualize_flight(
        self,
        output_path: str = "Tokamak_Flight_Report.png",
    ) -> Tuple[bool, Optional[str]]:
        """Render the flight trajectory report plot when plotting is available."""
        try:
            import matplotlib.pyplot as plt
        except (
            Exception
        ) as exc:  # pragma: no cover - defensive: matplotlib is present in the runtime
            return False, f"matplotlib unavailable: {exc}"
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            ax1.set_title("Plasma Trajectory Control")
            ax1.plot(self.history["t"], self.history["R_axis"], "b-", label="R Axis (Radial)")
            ax1.plot(self.history["t"], self.history["Z_axis"], "r-", label="Z Axis (Vertical)")

            ax1.axhline(self.target_R, color="b", linestyle="--", alpha=0.5, label="Target R")
            ax1.axhline(self.target_Z, color="r", linestyle="--", alpha=0.5, label="Target Z")

            ax1.set_xlabel("Shot Time (a.u.)")
            ax1.set_ylabel("Position (m)")
            ax1.legend()
            ax1.grid(True)

            rx = [p[0] for p in self.history["X_point"]]
            rz = [p[1] for p in self.history["X_point"]]

            # Filter out 0,0 (Limiter phase)
            valid_idx = [i for i, x in enumerate(rx) if x > 1.0]
            if valid_idx:
                ax2.plot(
                    [rx[i] for i in valid_idx], [rz[i] for i in valid_idx], "g-o", markersize=4
                )
                ax2.set_title("Divertor X-Point Movement")
                ax2.set_xlabel("R (m)")
                ax2.set_ylabel("Z (m)")
                ax2.grid(True)

                # Draw final shape if available from kernel implementation.
                if hasattr(self.kernel, "RR") and hasattr(self.kernel, "ZZ"):
                    ax2.contour(
                        self.kernel.RR,
                        self.kernel.ZZ,
                        self.kernel.Psi,
                        levels=10,
                        colors="k",
                        alpha=0.2,
                    )
            else:
                ax2.text(0.5, 0.5, "Plasma Remained Limited (No Divertor)", ha="center")

            plt.tight_layout()
            plt.savefig(output_path)
            plt.close(fig)
            self._log(f"Flight Sim Complete. Report: {output_path}")
            return True, None
        except Exception as exc:
            return False, str(exc)


def run_flight_sim(
    config_file: Optional[str] = None,
    shot_duration: int = SHOT_DURATION,
    seed: int = 42,
    save_plot: bool = True,
    output_path: str = "Tokamak_Flight_Report.png",
    verbose: bool = True,
    actuator_tau_s: float = 0.06,
    heating_actuator_tau_s: Optional[float] = None,
    actuator_current_offset_limit_ma: float = 0.05,
    heating_beta_max: float = 5.0,
    control_dt_s: float = 0.05,
    measurement_noise_std_m: float = 0.0,
    actuator_delay_s: float = 0.0,
    kernel_factory: Callable[[str], Any] = FusionKernel,
) -> Dict[str, Any]:
    """Run the public tokamak flight simulator and return a scenario-bound summary.

    ``measurement_noise_std_m`` acts on R/Z observations before the controller;
    ``actuator_delay_s`` acts on commands before actuator lag. ``seed`` controls
    only local simulator streams and never mutates NumPy's global RNG.
    """
    seed_int = int(seed)
    if config_file is None:
        config_file = str(default_iter_config_path())

    sim = IsoFluxController(
        config_file=str(config_file),
        kernel_factory=kernel_factory,
        verbose=verbose,
        actuator_tau_s=actuator_tau_s,
        heating_actuator_tau_s=heating_actuator_tau_s,
        actuator_current_offset_limit_ma=actuator_current_offset_limit_ma,
        heating_beta_max=heating_beta_max,
        control_dt_s=control_dt_s,
        measurement_noise_std_m=measurement_noise_std_m,
        actuator_delay_s=actuator_delay_s,
        rng_seed=seed_int,
    )
    summary = sim.run_shot(
        shot_duration=shot_duration,
        save_plot=save_plot,
        output_path=output_path,
    )
    summary["seed"] = seed_int
    summary["config_path"] = str(config_file)
    return summary


if __name__ == "__main__":
    run_flight_sim()
