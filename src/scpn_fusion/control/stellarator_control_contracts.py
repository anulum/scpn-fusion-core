# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Geometry-Neutral Stellarator Control Contracts
"""Geometry-neutral contracts for stellarator control replay benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


def _require_finite(name: str, value: float) -> float:
    value_f = float(value)
    if not math.isfinite(value_f):
        raise ValueError(f"{name} must be finite.")
    return value_f


def _require_non_empty(name: str, value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


@dataclass(frozen=True)
class MagneticConfiguration:
    """Device geometry metadata without tokamak-specific field assumptions."""

    name: str
    device_class: str
    field_periods: int
    coordinate_system: str
    reference: str

    def __post_init__(self) -> None:
        """Validate device geometry metadata after construction."""
        _require_non_empty("name", self.name)
        _require_non_empty("device_class", self.device_class)
        _require_non_empty("coordinate_system", self.coordinate_system)
        _require_non_empty("reference", self.reference)
        if int(self.field_periods) < 1:
            raise ValueError("field_periods must be >= 1.")

    def to_dict(self) -> dict[str, object]:
        """Serialise magnetic configuration metadata for benchmark reports."""
        return {
            "name": self.name,
            "device_class": self.device_class,
            "field_periods": int(self.field_periods),
            "coordinate_system": self.coordinate_system,
            "reference": self.reference,
        }


@dataclass(frozen=True)
class ActuatorChannel:
    """A bounded actuator channel with slew and replay latency metadata."""

    name: str
    unit: str
    min_value: float
    max_value: float
    slew_rate_per_s: float
    latency_steps: int = 0
    failure_mode: str = "none"

    def __post_init__(self) -> None:
        """Validate actuator bounds, slew rate, and latency after construction."""
        _require_non_empty("name", self.name)
        _require_non_empty("unit", self.unit)
        min_value = _require_finite("min_value", self.min_value)
        max_value = _require_finite("max_value", self.max_value)
        if max_value <= min_value:
            raise ValueError("max_value must be greater than min_value.")
        slew = _require_finite("slew_rate_per_s", self.slew_rate_per_s)
        if slew <= 0.0:
            raise ValueError("slew_rate_per_s must be > 0.")
        if int(self.latency_steps) < 0:
            raise ValueError("latency_steps must be >= 0.")

    def clamp(self, value: float) -> float:
        """Clamp a requested actuator value to the configured finite range."""
        value_f = _require_finite("actuator value", value)
        return min(max(value_f, float(self.min_value)), float(self.max_value))

    def apply_slew(self, previous: float, requested: float, dt_s: float) -> float:
        """Apply actuator slew-rate limiting between previous and requested values."""
        previous_f = _require_finite("previous actuator value", previous)
        requested_f = self.clamp(requested)
        dt = _require_finite("dt_s", dt_s)
        if dt <= 0.0:
            raise ValueError("dt_s must be > 0.")
        max_delta = float(self.slew_rate_per_s) * dt
        delta = min(max(requested_f - previous_f, -max_delta), max_delta)
        return self.clamp(previous_f + delta)

    def to_dict(self) -> dict[str, object]:
        """Serialise actuator limits, slew rate, latency, and failure metadata."""
        return {
            "name": self.name,
            "unit": self.unit,
            "min_value": float(self.min_value),
            "max_value": float(self.max_value),
            "slew_rate_per_s": float(self.slew_rate_per_s),
            "latency_steps": int(self.latency_steps),
            "failure_mode": self.failure_mode,
        }


@dataclass(frozen=True)
class ActuatorSet:
    """Named actuator collection with uniqueness validation."""

    channels: tuple[ActuatorChannel, ...]

    def __post_init__(self) -> None:
        """Validate that actuator channel names are present and unique."""
        if not self.channels:
            raise ValueError("ActuatorSet requires at least one channel.")
        names = [channel.name for channel in self.channels]
        if len(set(names)) != len(names):
            raise ValueError("Actuator channel names must be unique.")

    def by_name(self, name: str) -> ActuatorChannel:
        """Return the actuator channel with the requested name."""
        for channel in self.channels:
            if channel.name == name:
                return channel
        raise KeyError(f"unknown actuator channel: {name}")

    def to_dict(self) -> dict[str, object]:
        """Serialise all actuator channels in deterministic order."""
        return {"channels": [channel.to_dict() for channel in self.channels]}


@dataclass(frozen=True)
class DiagnosticChannel:
    """One replay diagnostic value with uncertainty and provenance."""

    name: str
    value: float
    unit: str
    sigma: float
    provenance: str

    def __post_init__(self) -> None:
        """Validate diagnostic value, unit, sigma, and provenance after construction."""
        _require_non_empty("name", self.name)
        _require_non_empty("unit", self.unit)
        _require_non_empty("provenance", self.provenance)
        _require_finite("value", self.value)
        sigma = _require_finite("sigma", self.sigma)
        if sigma < 0.0:
            raise ValueError("sigma must be >= 0.")

    def to_dict(self) -> dict[str, object]:
        """Serialise one diagnostic channel value and uncertainty."""
        return {
            "name": self.name,
            "value": float(self.value),
            "unit": self.unit,
            "sigma": float(self.sigma),
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class DiagnosticFrame:
    """A geometry-neutral diagnostic frame at one replay tick."""

    step: int
    time_s: float
    channels: tuple[DiagnosticChannel, ...]

    def __post_init__(self) -> None:
        """Validate frame step, time, and unique channel names after construction."""
        if int(self.step) < 0:
            raise ValueError("step must be >= 0.")
        time_s = _require_finite("time_s", self.time_s)
        if time_s < 0.0:
            raise ValueError("time_s must be >= 0.")
        if not self.channels:
            raise ValueError("DiagnosticFrame requires at least one channel.")
        names = [channel.name for channel in self.channels]
        if len(set(names)) != len(names):
            raise ValueError("Diagnostic channel names must be unique.")

    def as_mapping(self) -> dict[str, float]:
        """Return diagnostic values keyed by channel name."""
        return {channel.name: float(channel.value) for channel in self.channels}

    def to_dict(self) -> dict[str, object]:
        """Serialise a replay diagnostic frame."""
        return {
            "step": int(self.step),
            "time_s": float(self.time_s),
            "channels": [channel.to_dict() for channel in self.channels],
        }


@dataclass(frozen=True)
class ControlObjective:
    """Control targets, weights, and hard constraints by metric name."""

    target_metrics: Mapping[str, float]
    weights: Mapping[str, float]
    constraints: Mapping[str, float]

    def __post_init__(self) -> None:
        """Validate objective targets, weights, and constraints after construction."""
        if not self.target_metrics:
            raise ValueError("target_metrics must not be empty.")
        for group_name, group in (
            ("target_metrics", self.target_metrics),
            ("weights", self.weights),
            ("constraints", self.constraints),
        ):
            for key, value in group.items():
                _require_non_empty(f"{group_name} key", key)
                _require_finite(f"{group_name}.{key}", value)

    def to_dict(self) -> dict[str, object]:
        """Serialise objective targets, weights, and constraints."""
        return {
            "target_metrics": {key: float(value) for key, value in self.target_metrics.items()},
            "weights": {key: float(value) for key, value in self.weights.items()},
            "constraints": {key: float(value) for key, value in self.constraints.items()},
        }


@dataclass(frozen=True)
class ReplayScenario:
    """Complete deterministic replay scenario description."""

    name: str
    seed: int
    steps: int
    dt_s: float
    magnetic_configuration: MagneticConfiguration
    actuator_set: ActuatorSet
    objective: ControlObjective
    initial_frame: DiagnosticFrame
    fault_schedule: Mapping[int, Mapping[str, str]]

    def __post_init__(self) -> None:
        """Validate replay scenario timing, faults, and actuator references."""
        _require_non_empty("name", self.name)
        if int(self.steps) < 2:
            raise ValueError("steps must be >= 2.")
        dt = _require_finite("dt_s", self.dt_s)
        if dt <= 0.0:
            raise ValueError("dt_s must be > 0.")
        for step, faults in self.fault_schedule.items():
            if int(step) < 0 or int(step) >= int(self.steps):
                raise ValueError("fault schedule step out of replay bounds.")
            for channel_name in faults:
                self.actuator_set.by_name(channel_name)

    def to_dict(self) -> dict[str, object]:
        """Serialise the complete replay scenario contract."""
        return {
            "name": self.name,
            "seed": int(self.seed),
            "steps": int(self.steps),
            "dt_s": float(self.dt_s),
            "magnetic_configuration": self.magnetic_configuration.to_dict(),
            "actuator_set": self.actuator_set.to_dict(),
            "objective": self.objective.to_dict(),
            "initial_frame": self.initial_frame.to_dict(),
            "fault_schedule": {
                str(int(step)): dict(faults) for step, faults in self.fault_schedule.items()
            },
        }


__all__ = [
    "ActuatorChannel",
    "ActuatorSet",
    "ControlObjective",
    "DiagnosticChannel",
    "DiagnosticFrame",
    "MagneticConfiguration",
    "ReplayScenario",
]
