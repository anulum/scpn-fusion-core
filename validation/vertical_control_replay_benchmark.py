#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Vertical Control Replay Benchmark
"""Deterministic vertical-control replay benchmark scaffold."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Protocol

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpn_fusion.control.sliding_mode_vertical import SuperTwistingSMC  # noqa: E402
from scpn_fusion.control.rzip_model import RZIPModel  # noqa: E402
from scpn_fusion.core.vessel_model import VesselElement, VesselModel  # noqa: E402

SCHEMA_PATH = "schemas/vertical_control_replay_benchmark.schema.json"
SCHEMA_VERSION = "1.0.0"
PRIMARY_CONTROLLER_IDS = ("pid", "super_twisting", "sliding_mode_vertical")
DIAGNOSTIC_CONTROLLER_IDS = ("no_control",)
CONTROLLER_IDS = PRIMARY_CONTROLLER_IDS + DIAGNOSTIC_CONTROLLER_IDS


@dataclass(frozen=True)
class ReplayScenario:
    """Deterministic vertical-axis replay scenario definition."""

    n_steps: int = 240
    dt_s: float = 0.001
    initial_z_m: float = 0.006
    initial_dz_dt_m_per_s: float = 0.0
    vertical_growth_rate_s_inv: float = 18.0
    damping_s_inv: float = 7.5
    actuator_gain_m_per_s2: float = 95.0
    disturbance_start_step: int = 60
    disturbance_stop_step: int = 130
    disturbance_accel_m_per_s2: float = 0.22
    uncertainty_scale: float = 0.12


@dataclass(frozen=True)
class MachineProfile:
    """Geometry-neutral tokamak profile bounds for replay parametrisation."""

    profile_id: str = "iter_like"
    provenance: str = "public design-scale profile, reduced-order replay fixture"
    plasma_current_MA: float = 15.0
    elongation: float = 1.7
    vertical_growth_rate_s_inv: float = 18.0
    actuator_gain_m_per_s2: float = 95.0
    damping_s_inv: float = 7.5
    wall_time_s: float = 0.08


@dataclass(frozen=True)
class ActuatorLimits:
    """Vertical actuator clipping and slew-rate contract."""

    max_abs_command: float = 0.45
    max_slew_per_step: float = 0.035


@dataclass(frozen=True)
class BenchmarkThresholds:
    """Acceptance thresholds for the scaffold replay contract."""

    max_p95_abs_z_m: float = 0.0085
    max_final_abs_z_m: float = 0.004
    max_abs_command: float = 0.45
    max_abs_slew: float = 0.035
    max_post_disturbance_decay_ratio: float = 0.75


class Controller(Protocol):
    """Controller protocol for deterministic replay lanes."""

    controller_id: str

    def step(self, *, z_m: float, dz_dt_m_per_s: float, dt_s: float) -> float:
        """Return the next unconstrained actuator command."""


class VerticalPlantContract(Protocol):
    """Plant contract used by replay lanes to advance vertical state."""

    @property
    def contract_id(self) -> str:
        """Identifier of the concrete plant contract."""

    def step_state(
        self,
        *,
        z_m: float,
        dz_dt_m_per_s: float,
        command: float,
        disturbance_accel_m_per_s2: float,
        growth_scale: float,
        damping_scale: float,
        actuator_scale: float,
        dt_s: float,
    ) -> tuple[float, float]:
        """Return the next vertical state `(z_m, dz_dt_m_per_s)`."""

    def report(self) -> dict[str, Any]:
        """Return machine-checkable plant-contract metadata."""


@dataclass
class PIDVerticalController:
    """Small deterministic PID controller used as a baseline replay lane."""

    kp: float = 42.0
    kd: float = 2.6
    ki: float = 0.0
    controller_id: str = "pid"
    _integral: float = 0.0

    def step(self, *, z_m: float, dz_dt_m_per_s: float, dt_s: float) -> float:
        """Return PID control command from position and velocity error."""
        self._integral += z_m * dt_s
        return -(self.kp * z_m + self.kd * dz_dt_m_per_s + self.ki * self._integral)


@dataclass
class SuperTwistingVerticalController:
    """Deterministic super-twisting sliding-mode replay lane."""

    alpha: float = 0.95
    beta: float = 28.0
    surface_velocity_weight: float = 0.12
    controller_id: str = "super_twisting"
    _v: float = 0.0

    def step(self, *, z_m: float, dz_dt_m_per_s: float, dt_s: float) -> float:
        """Return super-twisting action on the sliding manifold."""
        s = z_m + self.surface_velocity_weight * dz_dt_m_per_s
        sign = float(np.sign(s))
        self._v -= self.beta * sign * dt_s
        return -self.alpha * float(np.sqrt(abs(s))) * sign + self._v


@dataclass
class SlidingModeVerticalController:
    """Adapter around the repository sliding-mode vertical stabilisation path."""

    controller_id: str = "sliding_mode_vertical"
    _smc: SuperTwistingSMC | None = None

    def __post_init__(self) -> None:
        if self._smc is None:
            self._smc = SuperTwistingSMC(alpha=0.95, beta=28.0, c=0.12, u_max=0.45)

    def step(self, *, z_m: float, dz_dt_m_per_s: float, dt_s: float) -> float:
        """Delegate the step command to repository SMC implementation."""
        assert self._smc is not None
        return self._smc.step(z_m, dz_dt_m_per_s, dt_s)


@dataclass
class NoControlController:
    """Diagnostic-only sentinel lane used to calibrate failure-mode sensitivity."""

    controller_id: str = "no_control"

    def step(self, *, z_m: float, dz_dt_m_per_s: float, dt_s: float) -> float:
        """Return zero control for the diagnostics lane."""
        return 0.0


@dataclass(frozen=True)
class RZIPVerticalPlantContract:
    """RZIP-backed vertical replay plant contract."""

    machine_profile: MachineProfile
    scenario: ReplayScenario
    _rzip_model: RZIPModel = field(init=False, repr=False)
    _state_space_checksum: str = field(init=False, repr=False)
    _open_loop_growth_rate: float = field(init=False, repr=False)
    contract_id: str = "rzip_vertical_state_space_v1"
    source_module: str = "scpn_fusion.control.rzip_model"

    def __post_init__(self) -> None:
        model = self._build_model()
        object.__setattr__(self, "_rzip_model", model)
        matrices = model.build_state_space()
        object.__setattr__(
            self, "_state_space_checksum", _sha256_json([m.tolist() for m in matrices])
        )
        object.__setattr__(
            self, "_open_loop_growth_rate", max(float(model.vertical_growth_rate()), 0.0)
        )

    def _geometry(self) -> dict[str, float]:
        if self.machine_profile.profile_id == "diii_d_like":
            return {"R0": 1.67, "a": 0.67, "B0": 2.1}
        if self.machine_profile.profile_id == "compact_tokamak":
            return {"R0": 1.85, "a": 0.45, "B0": 5.3}
        return {"R0": 6.2, "a": 2.0, "B0": 5.3}

    def _build_model(self) -> RZIPModel:
        geom = self._geometry()
        kappa = float(self.machine_profile.elongation)
        a = float(geom["a"])
        z_wall = max(0.1, a * kappa)
        inductance = 1.0e-5
        resistance = inductance / float(self.machine_profile.wall_time_s)
        vessel = VesselModel(
            [
                VesselElement(
                    R=float(geom["R0"]),
                    Z=z_wall,
                    resistance=resistance,
                    cross_section=0.1,
                    inductance=inductance,
                ),
                VesselElement(
                    R=float(geom["R0"]),
                    Z=-z_wall,
                    resistance=resistance,
                    cross_section=0.1,
                    inductance=inductance,
                ),
            ]
        )
        active_coils = [
            VesselElement(
                R=float(geom["R0"]) + 0.25 * a,
                Z=1.15 * z_wall,
                resistance=1.0e-2,
                cross_section=0.05,
                inductance=5.0e-5,
            ),
            VesselElement(
                R=float(geom["R0"]) + 0.25 * a,
                Z=-1.15 * z_wall,
                resistance=1.0e-2,
                cross_section=0.05,
                inductance=5.0e-5,
            ),
        ]
        return RZIPModel(
            R0=float(geom["R0"]),
            a=a,
            kappa=kappa,
            Ip_MA=float(self.machine_profile.plasma_current_MA),
            B0=float(geom["B0"]),
            n_index=-1.0,
            vessel=vessel,
            active_coils=active_coils,
        )

    def step_state(
        self,
        *,
        z_m: float,
        dz_dt_m_per_s: float,
        command: float,
        disturbance_accel_m_per_s2: float,
        growth_scale: float,
        damping_scale: float,
        actuator_scale: float,
        dt_s: float,
    ) -> tuple[float, float]:
        """Advance one simulation step and return next `(z_m, dz_dt_m_per_s)`."""
        accel = (
            self.scenario.vertical_growth_rate_s_inv * growth_scale * z_m
            - self.scenario.damping_s_inv * damping_scale * dz_dt_m_per_s
            + self.scenario.actuator_gain_m_per_s2 * actuator_scale * command
            + disturbance_accel_m_per_s2
        )
        next_dz = dz_dt_m_per_s + accel * dt_s
        next_z = z_m + next_dz * dt_s
        return float(next_z), float(next_dz)

    def _trajectory_checksum(self) -> str:
        z = float(self.scenario.initial_z_m)
        dz = float(self.scenario.initial_dz_dt_m_per_s)
        trajectory: list[tuple[float, float]] = []
        for idx in range(int(self.scenario.n_steps)):
            command = 0.1 * float(np.sin(0.1 * idx))
            disturbance = _disturbance_for_step(idx, self.scenario)
            z, dz = self.step_state(
                z_m=z,
                dz_dt_m_per_s=dz,
                command=command,
                disturbance_accel_m_per_s2=disturbance,
                growth_scale=1.0,
                damping_scale=1.0,
                actuator_scale=1.0,
                dt_s=float(self.scenario.dt_s),
            )
            trajectory.append((z, dz))
        return _sha256_json(trajectory)

    def report(self) -> dict[str, Any]:
        """Return reproducible plant report including trajectory checksums."""
        first = self._trajectory_checksum()
        second = self._trajectory_checksum()
        return {
            "contract_id": self.contract_id,
            "source_module": self.source_module,
            "state_variables": ["z_m", "dz_dt_m_per_s"],
            "input_variables": ["normalised_vertical_actuator_command"],
            "state_space_checksum_sha256": str(self._state_space_checksum),
            "open_loop_growth_rate_s_inv": float(self._open_loop_growth_rate),
            "effective_growth_rate_s_inv": float(self.scenario.vertical_growth_rate_s_inv),
            "effective_damping_s_inv": float(self.scenario.damping_s_inv),
            "effective_actuator_gain_m_per_s2": float(self.scenario.actuator_gain_m_per_s2),
            "deterministic_state_trajectory_checksum_sha256": first,
            "repeat_state_trajectory_checksum_sha256": second,
            "deterministic_state_trajectory_pass": first == second,
        }


def machine_profiles() -> dict[str, MachineProfile]:
    """Return committed geometry-neutral profile contracts for replay coverage."""

    return {
        "iter_like": MachineProfile(),
        "diii_d_like": MachineProfile(
            profile_id="diii_d_like",
            provenance="public DIII-D-scale profile, reduced-order replay fixture",
            plasma_current_MA=1.8,
            elongation=1.9,
            vertical_growth_rate_s_inv=22.0,
            actuator_gain_m_per_s2=120.0,
            damping_s_inv=8.5,
            wall_time_s=0.025,
        ),
        "compact_tokamak": MachineProfile(
            profile_id="compact_tokamak",
            provenance="public compact-tokamak profile, reduced-order replay fixture",
            plasma_current_MA=8.0,
            elongation=2.0,
            vertical_growth_rate_s_inv=28.0,
            actuator_gain_m_per_s2=150.0,
            damping_s_inv=9.5,
            wall_time_s=0.015,
        ),
    }


def _require_finite(name: str, value: float, *, minimum: float | None = None) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and out < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return out


def _validate_scenario(scenario: ReplayScenario) -> None:
    if isinstance(scenario.n_steps, bool) or int(scenario.n_steps) < 24:
        raise ValueError("n_steps must be an integer >= 24.")
    _require_finite("dt_s", scenario.dt_s, minimum=1.0e-6)
    _require_finite("initial_z_m", scenario.initial_z_m)
    _require_finite("initial_dz_dt_m_per_s", scenario.initial_dz_dt_m_per_s)
    _require_finite("vertical_growth_rate_s_inv", scenario.vertical_growth_rate_s_inv)
    _require_finite("damping_s_inv", scenario.damping_s_inv, minimum=0.0)
    _require_finite("actuator_gain_m_per_s2", scenario.actuator_gain_m_per_s2, minimum=1.0e-12)
    if (
        not 0
        <= scenario.disturbance_start_step
        < scenario.disturbance_stop_step
        <= scenario.n_steps
    ):
        raise ValueError("disturbance steps must satisfy 0 <= start < stop <= n_steps.")
    _require_finite("disturbance_accel_m_per_s2", scenario.disturbance_accel_m_per_s2)
    _require_finite("uncertainty_scale", scenario.uncertainty_scale, minimum=0.0)


def _validate_machine_profile(profile: MachineProfile) -> None:
    if not str(profile.profile_id).strip():
        raise ValueError("profile_id is required.")
    if not str(profile.provenance).strip():
        raise ValueError("machine profile provenance is required.")
    _require_finite("plasma_current_MA", profile.plasma_current_MA, minimum=1.0e-12)
    _require_finite("elongation", profile.elongation, minimum=1.0)
    _require_finite(
        "vertical_growth_rate_s_inv", profile.vertical_growth_rate_s_inv, minimum=1.0e-12
    )
    _require_finite("actuator_gain_m_per_s2", profile.actuator_gain_m_per_s2, minimum=1.0e-12)
    _require_finite("damping_s_inv", profile.damping_s_inv, minimum=0.0)
    _require_finite("wall_time_s", profile.wall_time_s, minimum=1.0e-12)


def _scenario_from_profile(profile: MachineProfile) -> ReplayScenario:
    return ReplayScenario(
        vertical_growth_rate_s_inv=profile.vertical_growth_rate_s_inv,
        damping_s_inv=profile.damping_s_inv,
        actuator_gain_m_per_s2=profile.actuator_gain_m_per_s2,
    )


def _validate_limits(limits: ActuatorLimits) -> None:
    _require_finite("max_abs_command", limits.max_abs_command, minimum=1.0e-12)
    _require_finite("max_slew_per_step", limits.max_slew_per_step, minimum=1.0e-12)


def _apply_actuator_limits(raw: float, previous: float, limits: ActuatorLimits) -> float:
    clipped_raw = float(np.clip(raw, -limits.max_abs_command, limits.max_abs_command))
    delta = float(
        np.clip(clipped_raw - previous, -limits.max_slew_per_step, limits.max_slew_per_step)
    )
    return float(np.clip(previous + delta, -limits.max_abs_command, limits.max_abs_command))


def _disturbance_for_step(step: int, scenario: ReplayScenario) -> float:
    if scenario.disturbance_start_step <= step < scenario.disturbance_stop_step:
        return scenario.disturbance_accel_m_per_s2
    return 0.0


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return proc.stdout.strip() or "unknown"


def _provenance(
    scenario: ReplayScenario, limits: ActuatorLimits, machine_profile: MachineProfile
) -> dict[str, Any]:
    return {
        "source_commit": _source_commit(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "schema": SCHEMA_PATH,
        "schema_version": SCHEMA_VERSION,
        "scenario_checksum_sha256": _sha256_json(asdict(scenario)),
        "machine_profile_checksum_sha256": _sha256_json(asdict(machine_profile)),
        "actuator_limits_checksum_sha256": _sha256_json(asdict(limits)),
    }


def _simulate(
    controller: Controller,
    *,
    scenario: ReplayScenario,
    limits: ActuatorLimits,
    thresholds: BenchmarkThresholds,
    plant_contract: VerticalPlantContract,
    growth_scale: float = 1.0,
    damping_scale: float = 1.0,
    actuator_scale: float = 1.0,
    sensor_bias_m: float = 0.0,
    latency_steps: int = 0,
) -> dict[str, Any]:
    z = float(scenario.initial_z_m)
    dz = float(scenario.initial_dz_dt_m_per_s)
    command = 0.0
    z_trace: list[float] = []
    raw_command_trace: list[float] = []
    command_trace: list[float] = []
    slew_trace: list[float] = []
    disturbance_trace: list[float] = []
    measurement_history: list[tuple[float, float]] = [(z, dz)]

    for step in range(int(scenario.n_steps)):
        measurement_index = max(0, len(measurement_history) - 1 - int(latency_steps))
        measured_z, measured_dz = measurement_history[measurement_index]
        raw_command = controller.step(
            z_m=measured_z + sensor_bias_m,
            dz_dt_m_per_s=measured_dz,
            dt_s=scenario.dt_s,
        )
        raw_command_trace.append(float(raw_command))
        bounded_command = _apply_actuator_limits(raw_command, command, limits)
        slew_trace.append(abs(bounded_command - command))
        command = bounded_command
        disturbance = _disturbance_for_step(step, scenario)
        disturbance_trace.append(float(disturbance))
        z, dz = plant_contract.step_state(
            z_m=z,
            dz_dt_m_per_s=dz,
            command=command,
            disturbance_accel_m_per_s2=disturbance,
            growth_scale=growth_scale,
            damping_scale=damping_scale,
            actuator_scale=actuator_scale,
            dt_s=float(scenario.dt_s),
        )
        measurement_history.append((float(z), float(dz)))
        z_trace.append(float(z))
        command_trace.append(float(command))

    abs_z = np.abs(np.asarray(z_trace, dtype=np.float64))
    raw_commands = np.asarray(raw_command_trace, dtype=np.float64)
    commands = np.asarray(command_trace, dtype=np.float64)
    slews = np.asarray(slew_trace, dtype=np.float64)
    raw_exceeds_abs = np.abs(raw_commands) > limits.max_abs_command + 1.0e-12
    raw_slew = np.diff(np.asarray([0.0, *raw_command_trace], dtype=np.float64))
    raw_exceeds_slew = np.abs(raw_slew) > limits.max_slew_per_step + 1.0e-12
    relaxation_start_index = min(
        max(0, int(scenario.disturbance_stop_step)), max(0, len(z_trace) - 1)
    )
    relaxation_start_abs = float(abs_z[relaxation_start_index])
    relaxation_end_abs = float(abs_z[-1])
    relaxation_ratio = relaxation_end_abs / max(relaxation_start_abs, 1.0e-12)
    return {
        "n_steps": int(scenario.n_steps),
        "final_abs_z_m": float(abs_z[-1]),
        "p95_abs_z_m": float(np.percentile(abs_z, 95.0)),
        "max_abs_z_m": float(np.max(abs_z)),
        "mean_abs_z_m": float(np.mean(abs_z)),
        "max_abs_command": float(np.max(np.abs(commands))),
        "max_abs_slew": float(np.max(slews)),
        "final_command": float(commands[-1]),
        "actuator_limit_application": {
            "applied_after_controller_output": True,
            "max_raw_abs_command": float(np.max(np.abs(raw_commands))),
            "max_bounded_abs_command": float(np.max(np.abs(commands))),
            "max_bounded_abs_slew": float(np.max(slews)),
            "abs_command_clipped_samples": int(np.count_nonzero(raw_exceeds_abs)),
            "slew_limited_samples": int(np.count_nonzero(raw_exceeds_slew)),
        },
        "post_disturbance_relaxation": {
            "start_step": int(scenario.disturbance_stop_step),
            "start_abs_z_m": relaxation_start_abs,
            "end_abs_z_m": relaxation_end_abs,
            "displacement_decay_ratio": float(relaxation_ratio),
            "max_decay_ratio": float(thresholds.max_post_disturbance_decay_ratio),
            "passes": bool(
                relaxation_ratio <= thresholds.max_post_disturbance_decay_ratio + 1.0e-12
            ),
        },
        "trace_checksums": {
            "state_z_m_sha256": _sha256_json(z_trace),
            "raw_command_sha256": _sha256_json(raw_command_trace),
            "command_sha256": _sha256_json(command_trace),
            "disturbance_sha256": _sha256_json(disturbance_trace),
        },
    }


def _uncertainty_cases(scenario: ReplayScenario) -> list[dict[str, Any]]:
    scale = float(scenario.uncertainty_scale)
    sensor_bias = max(1.0e-6, abs(float(scenario.initial_z_m)) * scale)
    cases: list[dict[str, Any]] = []
    for growth_scale in (1.0, 1.0 + scale):
        for damping_scale in (1.0, max(0.0, 1.0 - scale)):
            for actuator_scale in (1.0, max(0.0, 1.0 - scale)):
                for sensor_bias_m in (0.0, sensor_bias):
                    for latency_steps in (0, 1):
                        cases.append(
                            {
                                "case_id": (
                                    f"g{growth_scale:.3f}_d{damping_scale:.3f}_"
                                    f"a{actuator_scale:.3f}_b{sensor_bias_m:.6f}_"
                                    f"l{latency_steps}"
                                ),
                                "growth_scale": float(growth_scale),
                                "damping_scale": float(damping_scale),
                                "actuator_scale": float(actuator_scale),
                                "sensor_bias_m": float(sensor_bias_m),
                                "latency_steps": int(latency_steps),
                            }
                        )
    return cases


def _controller_factory(controller_id: str) -> Controller:
    if controller_id == "pid":
        return PIDVerticalController()
    if controller_id == "super_twisting":
        return SuperTwistingVerticalController()
    if controller_id == "sliding_mode_vertical":
        return SlidingModeVerticalController()
    if controller_id == "no_control":
        return NoControlController()
    raise ValueError(f"unknown controller_id: {controller_id}")


def _run_controller(
    controller_id: str,
    *,
    scenario: ReplayScenario,
    limits: ActuatorLimits,
    thresholds: BenchmarkThresholds,
    plant_contract: VerticalPlantContract,
) -> dict[str, Any]:
    nominal = _simulate(
        _controller_factory(controller_id),
        scenario=scenario,
        limits=limits,
        thresholds=thresholds,
        plant_contract=plant_contract,
    )
    high_growth = _simulate(
        _controller_factory(controller_id),
        scenario=scenario,
        limits=limits,
        thresholds=thresholds,
        plant_contract=plant_contract,
        growth_scale=1.0 + scenario.uncertainty_scale,
    )
    low_actuator = _simulate(
        _controller_factory(controller_id),
        scenario=scenario,
        limits=limits,
        thresholds=thresholds,
        plant_contract=plant_contract,
        actuator_scale=1.0 - scenario.uncertainty_scale,
    )
    grid_results = [
        {
            "case_id": str(case["case_id"]),
            **_simulate(
                _controller_factory(controller_id),
                scenario=scenario,
                limits=limits,
                thresholds=thresholds,
                plant_contract=plant_contract,
                growth_scale=float(case["growth_scale"]),
                damping_scale=float(case["damping_scale"]),
                actuator_scale=float(case["actuator_scale"]),
                sensor_bias_m=float(case["sensor_bias_m"]),
                latency_steps=int(case["latency_steps"]),
            ),
        }
        for case in _uncertainty_cases(scenario)
    ]
    max_abs_z = max(float(result["max_abs_z_m"]) for result in grid_results)
    max_p95_abs_z = max(float(result["p95_abs_z_m"]) for result in grid_results)
    p95_abs_z_values = np.asarray(
        [float(result["p95_abs_z_m"]) for result in grid_results], dtype=np.float64
    )
    worst_case = max(grid_results, key=lambda result: float(result["max_abs_z_m"]))
    return {
        "controller_id": controller_id,
        **nominal,
        "uncertainty": {
            "cases": [str(result["case_id"]) for result in grid_results],
            "n_cases": len(grid_results),
            "nominal_abs_z_m": float(nominal["max_abs_z_m"]),
            "high_growth_abs_z_m": float(high_growth["max_abs_z_m"]),
            "low_actuator_abs_z_m": float(low_actuator["max_abs_z_m"]),
            "max_abs_z_m": max_abs_z,
            "max_p95_abs_z_m": max_p95_abs_z,
            "p95_abs_z_m_p95": float(np.percentile(p95_abs_z_values, 95.0)),
            "worst_case": {
                "case_id": str(worst_case["case_id"]),
                "max_abs_z_m": float(worst_case["max_abs_z_m"]),
                "p95_abs_z_m": float(worst_case["p95_abs_z_m"]),
            },
        },
    }


def _passes(
    result: dict[str, Any], limits: ActuatorLimits, thresholds: BenchmarkThresholds
) -> bool:
    return bool(
        result["p95_abs_z_m"] <= thresholds.max_p95_abs_z_m
        and result["final_abs_z_m"] <= thresholds.max_final_abs_z_m
        and result["max_abs_command"]
        <= min(thresholds.max_abs_command, limits.max_abs_command) + 1.0e-12
        and result["max_abs_slew"]
        <= min(thresholds.max_abs_slew, limits.max_slew_per_step) + 1.0e-12
        and result["post_disturbance_relaxation"]["passes"]
    )


def _build_fairness_report(
    first: dict[str, dict[str, Any]],
    second: dict[str, dict[str, Any]],
    *,
    scenario: ReplayScenario,
) -> dict[str, Any]:
    disturbance_checksums = {
        cid: res["trace_checksums"]["disturbance_sha256"] for cid, res in first.items()
    }
    actuator_contract_ok = all(
        bool(res["actuator_limit_application"]["applied_after_controller_output"])
        for res in first.values()
    )
    return {
        "passes_fairness_checks": bool(
            len(set(disturbance_checksums.values())) == 1
            and first == second
            and actuator_contract_ok
        ),
        "shared_initial_measurement": {
            "z_m": float(scenario.initial_z_m),
            "dz_dt_m_per_s": float(scenario.initial_dz_dt_m_per_s),
        },
        "disturbance_trace_checksums": disturbance_checksums,
        "all_controllers_share_disturbance_trace": len(set(disturbance_checksums.values())) == 1,
        "controller_state_reset_pass": first == second,
        "actuator_limits_applied_after_controller_output": actuator_contract_ok,
        "controller_trace_roles": {
            **{cid: "primary" for cid in PRIMARY_CONTROLLER_IDS},
            **{cid: "diagnostic_only" for cid in DIAGNOSTIC_CONTROLLER_IDS},
        },
    }


def run_benchmark(
    *,
    scenario: ReplayScenario | None = None,
    machine_profile: MachineProfile | None = None,
    actuator_limits: ActuatorLimits | None = None,
    thresholds: BenchmarkThresholds | None = None,
) -> dict[str, Any]:
    """Run the deterministic vertical-control replay benchmark."""

    machine_profile = machine_profile or machine_profiles()["iter_like"]
    _validate_machine_profile(machine_profile)
    scenario = scenario or _scenario_from_profile(machine_profile)
    actuator_limits = actuator_limits or ActuatorLimits()
    thresholds = thresholds or BenchmarkThresholds()
    _validate_scenario(scenario)
    _validate_limits(actuator_limits)
    plant_contract = RZIPVerticalPlantContract(machine_profile=machine_profile, scenario=scenario)

    first = {
        cid: _run_controller(
            cid,
            scenario=scenario,
            limits=actuator_limits,
            thresholds=thresholds,
            plant_contract=plant_contract,
        )
        for cid in CONTROLLER_IDS
    }
    second = {
        cid: _run_controller(
            cid,
            scenario=scenario,
            limits=actuator_limits,
            thresholds=thresholds,
            plant_contract=plant_contract,
        )
        for cid in CONTROLLER_IDS
    }
    deterministic = first == second
    controller_passes = {
        cid: _passes(res, actuator_limits, thresholds) for cid, res in first.items()
    }
    controller_roles = {
        **{cid: "primary" for cid in PRIMARY_CONTROLLER_IDS},
        **{cid: "diagnostic_only" for cid in DIAGNOSTIC_CONTROLLER_IDS},
    }
    max_p95 = max(float(first[cid]["p95_abs_z_m"]) for cid in PRIMARY_CONTROLLER_IDS)
    max_uncertain = max(
        float(first[cid]["uncertainty"]["max_abs_z_m"]) for cid in PRIMARY_CONTROLLER_IDS
    )
    primary_uncertainty = [first[cid]["uncertainty"] for cid in PRIMARY_CONTROLLER_IDS]
    uncertainty_p95_values = np.asarray(
        [float(result["p95_abs_z_m_p95"]) for result in primary_uncertainty],
        dtype=np.float64,
    )
    worst_uncertainty = max(
        (
            {
                "controller_id": cid,
                **first[cid]["uncertainty"]["worst_case"],
            }
            for cid in PRIMARY_CONTROLLER_IDS
        ),
        key=lambda result: float(result["max_abs_z_m"]),
    )
    trace_integrity = {
        "state_trace_checksum_sha256": _sha256_json(
            {cid: res["trace_checksums"]["state_z_m_sha256"] for cid, res in first.items()}
        ),
        "command_trace_checksum_sha256": _sha256_json(
            {cid: res["trace_checksums"]["command_sha256"] for cid, res in first.items()}
        ),
        "disturbance_trace_checksum_sha256": _sha256_json(
            {cid: res["trace_checksums"]["disturbance_sha256"] for cid, res in first.items()}
        ),
        "controller_trace_count": len(first),
        "samples_per_trace": int(scenario.n_steps),
    }
    fairness_report = _build_fairness_report(first, second, scenario=scenario)

    return {
        "SPDX-License-Identifier": "AGPL-3.0-or-later",
        "vertical_control_replay_benchmark": {
            "schema_version": SCHEMA_VERSION,
            "scenario": asdict(scenario),
            "machine_profile": asdict(machine_profile),
            "plant_contract": plant_contract.report(),
            "actuator_limits": asdict(actuator_limits),
            "thresholds": asdict(thresholds),
            "deterministic_replay_pass": deterministic,
            "controllers": first,
            "controller_roles": controller_roles,
            "controller_passes": controller_passes,
            "fairness_report": fairness_report,
            "trace_integrity": trace_integrity,
            "uncertainty_report": {
                "n_scenarios": int(primary_uncertainty[0]["n_cases"]),
                "grid_axes": [
                    "growth_scale",
                    "damping_scale",
                    "actuator_scale",
                    "sensor_bias_m",
                    "latency_steps",
                ],
                "max_p95_abs_z_m": max(
                    max_p95,
                    max(float(result["max_p95_abs_z_m"]) for result in primary_uncertainty),
                ),
                "p95_abs_z_m_p95": float(np.percentile(uncertainty_p95_values, 95.0)),
                "max_uncertain_abs_z_m": max_uncertain,
                "worst_case": worst_uncertainty,
                "uncertainty_scale": float(scenario.uncertainty_scale),
            },
            "provenance": _provenance(scenario, actuator_limits, machine_profile),
            "passes_thresholds": bool(
                deterministic
                and fairness_report["passes_fairness_checks"]
                and all(controller_passes[cid] for cid in PRIMARY_CONTROLLER_IDS)
            ),
        },
    }


def run_profile_suite() -> dict[str, Any]:
    """Run the vertical-control replay benchmark across committed profiles."""

    profiles = machine_profiles()
    profile_ids = sorted(profiles)
    reports = {
        profile_id: run_benchmark(machine_profile=profiles[profile_id])[
            "vertical_control_replay_benchmark"
        ]
        for profile_id in profile_ids
    }
    return {
        "SPDX-License-Identifier": "AGPL-3.0-or-later",
        "vertical_control_replay_profile_suite": {
            "schema_version": SCHEMA_VERSION,
            "profile_ids": profile_ids,
            "reports": reports,
            "all_profiles_pass": all(
                bool(report["passes_thresholds"]) for report in reports.values()
            ),
            "trace_integrity": {
                "profile_trace_checksum_sha256": _sha256_json(
                    {
                        profile_id: report["trace_integrity"]["state_trace_checksum_sha256"]
                        for profile_id, report in reports.items()
                    }
                ),
                "profile_count": len(profile_ids),
            },
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact benchmark report for operator review."""

    bench = report["vertical_control_replay_benchmark"]
    lines = [
        "# Vertical Control Replay Benchmark",
        "",
        f"- Schema version: `{bench['schema_version']}`",
        f"- Deterministic replay pass: `{'YES' if bench['deterministic_replay_pass'] else 'NO'}`",
        f"- Overall pass: `{'YES' if bench['passes_thresholds'] else 'NO'}`",
        f"- Steps: `{bench['scenario']['n_steps']}`",
        f"- dt: `{bench['scenario']['dt_s']:.6f} s`",
        f"- State trace checksum: `{bench['trace_integrity']['state_trace_checksum_sha256']}`",
        "",
        "| Controller | P95 |z| (m) | Final |z| (m) | Max |u| | Max slew | Pass |",
        "|------------|-------------|---------------|---------|----------|------|",
    ]
    for cid, result in bench["controllers"].items():
        lines.append(
            f"| {cid} | {result['p95_abs_z_m']:.6f} | {result['final_abs_z_m']:.6f} | "
            f"{result['max_abs_command']:.6f} | {result['max_abs_slew']:.6f} | "
            f"{'YES' if bench['controller_passes'][cid] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "## Post-disturbance relaxation",
            "",
            "- Primary controllers must reduce vertical displacement after the "
            "disturbance window ends.",
            f"- Maximum accepted final/start ratio: "
            f"`{bench['thresholds']['max_post_disturbance_decay_ratio']:.3f}`",
            "",
            "| Controller | Start |z| (m) | Final/start ratio | Pass |",
            "|------------|----------------|-------------------|------|",
        ]
    )
    for cid, result in bench["controllers"].items():
        relaxation = result["post_disturbance_relaxation"]
        lines.append(
            f"| {cid} | {relaxation['start_abs_z_m']:.6f} | "
            f"{relaxation['displacement_decay_ratio']:.6f} | "
            f"{'YES' if relaxation['passes'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "## Uncertainty",
            "",
            f"- Cases: `{bench['uncertainty_report']['n_scenarios']}`",
            f"- Max uncertain |z|: `{bench['uncertainty_report']['max_uncertain_abs_z_m']:.6f} m`",
            "",
        ]
    )
    return "\n".join(lines)


def render_profile_suite_markdown(report: dict[str, Any]) -> str:
    """Render a compact multi-profile benchmark report."""

    suite = report["vertical_control_replay_profile_suite"]
    lines = [
        "# Vertical Control Replay Benchmark",
        "",
        "## Profile suite",
        "",
        f"- Schema version: `{suite['schema_version']}`",
        f"- Profiles: `{', '.join(suite['profile_ids'])}`",
        f"- Overall pass: `{'YES' if suite['all_profiles_pass'] else 'NO'}`",
        f"- Profile trace checksum: `{suite['trace_integrity']['profile_trace_checksum_sha256']}`",
        "",
        "| Profile | Max P95 |z| (m) | Max uncertain |z| (m) | Pass |",
        "|---------|----------------|-----------------------|------|",
    ]
    for profile_id in suite["profile_ids"]:
        bench = suite["reports"][profile_id]
        lines.append(
            f"| {profile_id} | {bench['uncertainty_report']['max_p95_abs_z_m']:.6f} | "
            f"{bench['uncertainty_report']['max_uncertain_abs_z_m']:.6f} | "
            f"{'YES' if bench['passes_thresholds'] else 'NO'} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the vertical control replay benchmark from CLI arguments."""
    """CLI entrypoint for vertical control replay benchmark with optional profile suite."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "vertical_control_replay_benchmark.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "vertical_control_replay_benchmark.md"),
    )
    parser.add_argument(
        "--all-profiles",
        action="store_true",
        help="Run all committed machine profiles and write a profile-suite report.",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = run_profile_suite() if args.all_profiles else run_benchmark()
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(
        render_profile_suite_markdown(report) if args.all_profiles else render_markdown(report),
        encoding="utf-8",
    )

    if args.all_profiles:
        suite = report["vertical_control_replay_profile_suite"]
        print("Vertical control replay profile suite complete.")
        print(
            "profiles={profiles}, pass={passed}".format(
                profiles=",".join(suite["profile_ids"]),
                passed=suite["all_profiles_pass"],
            )
        )
        if args.strict and not suite["all_profiles_pass"]:
            return 2
        return 0

    bench = report["vertical_control_replay_benchmark"]
    print("Vertical control replay benchmark complete.")
    print(
        "deterministic={det}, controllers={controllers}, pass={passed}".format(
            det=bench["deterministic_replay_pass"],
            controllers=",".join(sorted(bench["controllers"])),
            passed=bench["passes_thresholds"],
        )
    )
    if args.strict and not bench["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
