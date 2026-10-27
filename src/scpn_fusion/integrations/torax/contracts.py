# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TORAX Runtime Contracts
"""Immutable, versioned contracts for process-isolated TORAX execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence, cast

from .serialization import canonical_sha256

TORAX_REQUEST_SCHEMA = "scpn-fusion-core.torax-runtime-request.v1"
TORAX_OUTCOME_SCHEMA = "scpn-fusion-core.torax-runtime-outcome.v1"
TORAX_VERSION = "1.4.3"
SIMULATION_CLOCK_DOMAIN = "simulation_monotonic"


class ToraxFailureCode(str, Enum):
    """Stable failure codes exposed independently of TORAX implementation types."""

    INVALID_REQUEST = "INVALID_REQUEST"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    BACKEND_VERSION_MISMATCH = "BACKEND_VERSION_MISMATCH"
    CONFIGURATION_REJECTED = "CONFIGURATION_REJECTED"
    TIMEOUT = "TIMEOUT"
    PROCESS_FAILURE = "PROCESS_FAILURE"
    OUTPUT_SCHEMA_MISMATCH = "OUTPUT_SCHEMA_MISMATCH"
    PROVENANCE_FAILURE = "PROVENANCE_FAILURE"
    NAN_DETECTED = "NAN_DETECTED"
    QUASINEUTRALITY_BROKEN = "QUASINEUTRALITY_BROKEN"
    NEGATIVE_CORE_PROFILES = "NEGATIVE_CORE_PROFILES"
    REACHED_MIN_DT = "REACHED_MIN_DT"
    LOW_TEMPERATURE_COLLAPSE = "LOW_TEMPERATURE_COLLAPSE"
    DID_NOT_REACH_T_FINAL = "DID_NOT_REACH_T_FINAL"

    @classmethod
    def from_sim_error(cls, name: str) -> ToraxFailureCode | None:
        """Map every TORAX 1.4.3 ``SimError`` name one-to-one."""
        if name == "NO_ERROR":
            return None
        try:
            return cls[name]
        except KeyError as error:
            raise ValueError(f"unknown TORAX SimError: {name}") from error


class ToraxRuntimeError(RuntimeError):
    """Raised by :meth:`ToraxRunOutcome.require_success` for typed failures."""

    def __init__(self, code: ToraxFailureCode, message: str) -> None:
        super().__init__(f"{code.value}: {message}")
        self.code = code


@dataclass(frozen=True)
class ToraxClock:
    """Exact simulation-monotonic clock and bounded execution policy."""

    domain: str
    epoch: str
    initial_ns: int
    final_ns: int
    timeout_s: float
    max_steps: int
    reset_policy: str

    def __post_init__(self) -> None:
        """Reject mixed clock domains and unbounded execution policies."""
        if self.domain != SIMULATION_CLOCK_DOMAIN:
            raise ValueError(f"clock.domain must be {SIMULATION_CLOCK_DOMAIN!r}")
        _nonempty(self.epoch, "clock.epoch")
        if self.initial_ns < 0 or self.final_ns <= self.initial_ns:
            raise ValueError("clock bounds must satisfy 0 <= initial_ns < final_ns")
        _positive_finite(self.timeout_s, "clock.timeout_s")
        if self.max_steps <= 0:
            raise ValueError("clock.max_steps must be > 0")
        if self.reset_policy != "fresh_process_no_hidden_state":
            raise ValueError("v1 reset_policy must require a fresh process with no hidden state")

    def to_dict(self) -> dict[str, object]:
        """Serialize the clock without converting integer nanoseconds to floats."""
        return {
            "domain": self.domain,
            "epoch": self.epoch,
            "initial_ns": self.initial_ns,
            "final_ns": self.final_ns,
            "timeout_s": self.timeout_s,
            "max_steps": self.max_steps,
            "reset_policy": self.reset_policy,
        }

    @classmethod
    def from_dict(cls, value: object) -> ToraxClock:
        """Parse a strict clock object."""
        raw = _strict_object(
            value,
            "clock",
            {
                "domain",
                "epoch",
                "initial_ns",
                "final_ns",
                "timeout_s",
                "max_steps",
                "reset_policy",
            },
        )
        return cls(
            domain=_string(raw["domain"], "clock.domain"),
            epoch=_string(raw["epoch"], "clock.epoch"),
            initial_ns=_integer(raw["initial_ns"], "clock.initial_ns"),
            final_ns=_integer(raw["final_ns"], "clock.final_ns"),
            timeout_s=_number(raw["timeout_s"], "clock.timeout_s"),
            max_steps=_integer(raw["max_steps"], "clock.max_steps"),
            reset_policy=_string(raw["reset_policy"], "clock.reset_policy"),
        )


@dataclass(frozen=True)
class ToraxGeometry:
    """Typed geometry identity shared with external consumers."""

    kind: str
    frame: str
    major_radius_m: float
    minor_radius_m: float
    magnetic_field_t: float
    radial_coordinate: str = "rho_norm"
    radial_coordinate_unit: str = "1"

    def __post_init__(self) -> None:
        """Reject invalid geometry values or a noncanonical radial frame."""
        _nonempty(self.kind, "geometry.kind")
        _nonempty(self.frame, "geometry.frame")
        _positive_finite(self.major_radius_m, "geometry.major_radius_m")
        _positive_finite(self.minor_radius_m, "geometry.minor_radius_m")
        _positive_finite(self.magnetic_field_t, "geometry.magnetic_field_t")
        if self.radial_coordinate != "rho_norm" or self.radial_coordinate_unit != "1":
            raise ValueError("v1 geometry requires dimensionless rho_norm")

    def to_dict(self) -> dict[str, object]:
        """Serialize the geometry vocabulary and SI values."""
        return dict(vars(self))

    @classmethod
    def from_dict(cls, value: object) -> ToraxGeometry:
        """Parse a strict geometry object."""
        keys = {
            "kind",
            "frame",
            "major_radius_m",
            "minor_radius_m",
            "magnetic_field_t",
            "radial_coordinate",
            "radial_coordinate_unit",
        }
        raw = _strict_object(value, "geometry", keys)
        return cls(
            kind=_string(raw["kind"], "geometry.kind"),
            frame=_string(raw["frame"], "geometry.frame"),
            major_radius_m=_number(raw["major_radius_m"], "geometry.major_radius_m"),
            minor_radius_m=_number(raw["minor_radius_m"], "geometry.minor_radius_m"),
            magnetic_field_t=_number(raw["magnetic_field_t"], "geometry.magnetic_field_t"),
            radial_coordinate=_string(raw["radial_coordinate"], "geometry.radial_coordinate"),
            radial_coordinate_unit=_string(
                raw["radial_coordinate_unit"], "geometry.radial_coordinate_unit"
            ),
        )


@dataclass(frozen=True)
class ToraxSignal:
    """Unit-bearing profile or scalar trajectory bound to one configuration value."""

    name: str
    role: str
    unit: str
    frame: str
    time_ns: tuple[int, ...]
    coordinate_name: str
    coordinate_unit: str
    coordinate: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    binding_name: str
    calibration: str
    provenance: str
    uncertainty_kind: str
    uncertainty: tuple[tuple[float, ...], ...] | None
    application_semantics: str
    model_delay_ns: int
    saturation_minimum: float | None
    saturation_maximum: float | None
    maximum_slew_per_s: float | None
    hardware_limits_status: str
    validity: str = "VALID"

    def __post_init__(self) -> None:
        """Reject malformed units, clocks, shapes, values, and uncertainty."""
        for name, value in (
            ("name", self.name),
            ("role", self.role),
            ("unit", self.unit),
            ("frame", self.frame),
            ("binding_name", self.binding_name),
            ("calibration", self.calibration),
            ("provenance", self.provenance),
            ("uncertainty_kind", self.uncertainty_kind),
        ):
            _nonempty(value, f"signal.{name}")
        if self.role not in {"initial_state", "control", "disturbance", "parameter"}:
            raise ValueError(f"unsupported signal role: {self.role}")
        if self.validity != "VALID":
            raise ValueError("runtime requests admit only VALID input signals")
        expected_semantics = {
            "initial_state": "initial_condition",
            "control": "prescribed_source",
            "disturbance": "prescribed_source",
            "parameter": "model_parameter",
        }[self.role]
        if self.application_semantics != expected_semantics:
            raise ValueError(f"signal {self.name} has inconsistent application semantics")
        if self.model_delay_ns < 0:
            raise ValueError(f"signal {self.name} model_delay_ns must be >= 0")
        limits = (
            self.saturation_minimum,
            self.saturation_maximum,
            self.maximum_slew_per_s,
        )
        if self.role in {"control", "disturbance"}:
            if self.hardware_limits_status != "not_declared_no_actuation_authority":
                raise ValueError(f"signal {self.name} cannot claim hardware limits in v1")
            if self.model_delay_ns != 0 or any(value is not None for value in limits):
                raise ValueError(
                    f"signal {self.name} must expose absent actuator dynamics explicitly"
                )
        elif self.hardware_limits_status != "not_applicable" or any(
            value is not None for value in limits
        ):
            raise ValueError(f"signal {self.name} non-actuator limit fields must be not_applicable")
        _strictly_increasing(self.time_ns, f"signal {self.name} time_ns")
        width = len(self.coordinate) if self.coordinate else 1
        if self.coordinate:
            _strictly_increasing_floats(self.coordinate, f"signal {self.name} coordinate")
            _nonempty(self.coordinate_name, f"signal {self.name} coordinate_name")
            _nonempty(self.coordinate_unit, f"signal {self.name} coordinate_unit")
        elif self.coordinate_name or self.coordinate_unit:
            raise ValueError(f"scalar signal {self.name} must not declare a coordinate")
        if len(self.values) != len(self.time_ns) or any(len(row) != width for row in self.values):
            raise ValueError(f"signal {self.name} values shape does not match time/coordinate")
        _finite_matrix(self.values, f"signal {self.name} values")
        if self.uncertainty is not None:
            if len(self.uncertainty) != len(self.values) or any(
                len(row) != width for row in self.uncertainty
            ):
                raise ValueError(f"signal {self.name} uncertainty shape mismatch")
            _finite_matrix(self.uncertainty, f"signal {self.name} uncertainty")
            if any(item < 0.0 for row in self.uncertainty for item in row):
                raise ValueError(f"signal {self.name} uncertainty must be non-negative")

    def to_dict(self) -> dict[str, object]:
        """Serialize the signal using arrays rather than implementation tuples."""
        return {
            "name": self.name,
            "role": self.role,
            "unit": self.unit,
            "frame": self.frame,
            "time_ns": list(self.time_ns),
            "coordinate_name": self.coordinate_name,
            "coordinate_unit": self.coordinate_unit,
            "coordinate": list(self.coordinate),
            "values": [list(row) for row in self.values],
            "binding_name": self.binding_name,
            "calibration": self.calibration,
            "provenance": self.provenance,
            "uncertainty_kind": self.uncertainty_kind,
            "uncertainty": None
            if self.uncertainty is None
            else [list(row) for row in self.uncertainty],
            "application_semantics": self.application_semantics,
            "model_delay_ns": self.model_delay_ns,
            "saturation_minimum": self.saturation_minimum,
            "saturation_maximum": self.saturation_maximum,
            "maximum_slew_per_s": self.maximum_slew_per_s,
            "hardware_limits_status": self.hardware_limits_status,
            "validity": self.validity,
        }

    @classmethod
    def from_dict(cls, value: object, *, label: str = "signal") -> ToraxSignal:
        """Parse one strict typed signal."""
        keys = {
            "name",
            "role",
            "unit",
            "frame",
            "time_ns",
            "coordinate_name",
            "coordinate_unit",
            "coordinate",
            "values",
            "binding_name",
            "calibration",
            "provenance",
            "uncertainty_kind",
            "uncertainty",
            "application_semantics",
            "model_delay_ns",
            "saturation_minimum",
            "saturation_maximum",
            "maximum_slew_per_s",
            "hardware_limits_status",
            "validity",
        }
        raw = _strict_object(value, label, keys)
        uncertainty_raw = raw["uncertainty"]
        return cls(
            name=_string(raw["name"], f"{label}.name"),
            role=_string(raw["role"], f"{label}.role"),
            unit=_string(raw["unit"], f"{label}.unit"),
            frame=_string(raw["frame"], f"{label}.frame"),
            time_ns=_integer_tuple(raw["time_ns"], f"{label}.time_ns"),
            coordinate_name=_string(raw["coordinate_name"], f"{label}.coordinate_name", empty=True),
            coordinate_unit=_string(raw["coordinate_unit"], f"{label}.coordinate_unit", empty=True),
            coordinate=_number_tuple(raw["coordinate"], f"{label}.coordinate"),
            values=_number_matrix(raw["values"], f"{label}.values"),
            binding_name=_string(raw["binding_name"], f"{label}.binding_name"),
            calibration=_string(raw["calibration"], f"{label}.calibration"),
            provenance=_string(raw["provenance"], f"{label}.provenance"),
            uncertainty_kind=_string(raw["uncertainty_kind"], f"{label}.uncertainty_kind"),
            uncertainty=None
            if uncertainty_raw is None
            else _number_matrix(uncertainty_raw, f"{label}.uncertainty"),
            application_semantics=_string(
                raw["application_semantics"], f"{label}.application_semantics"
            ),
            model_delay_ns=_integer(raw["model_delay_ns"], f"{label}.model_delay_ns"),
            saturation_minimum=_optional_number(
                raw["saturation_minimum"], f"{label}.saturation_minimum"
            ),
            saturation_maximum=_optional_number(
                raw["saturation_maximum"], f"{label}.saturation_maximum"
            ),
            maximum_slew_per_s=_optional_number(
                raw["maximum_slew_per_s"], f"{label}.maximum_slew_per_s"
            ),
            hardware_limits_status=_string(
                raw["hardware_limits_status"], f"{label}.hardware_limits_status"
            ),
            validity=_string(raw["validity"], f"{label}.validity"),
        )


@dataclass(frozen=True)
class ToraxConfigBinding:
    """Exact typed-to-backend overlap assertion using a JSON object path."""

    name: str
    config_path: tuple[str, ...]
    unit: str
    interpretation: str
    value: object

    def __post_init__(self) -> None:
        """Validate the binding identity, path, unit, and JSON value."""
        _nonempty(self.name, "binding.name")
        _nonempty(self.unit, f"binding {self.name} unit")
        if self.interpretation not in {"scalar", "time_scalar_map", "time_radial_map"}:
            raise ValueError(f"binding {self.name} has an unsupported interpretation")
        if not self.config_path or any(not part for part in self.config_path):
            raise ValueError(f"binding {self.name} requires a non-empty config path")
        canonical_sha256(self.value)

    def to_dict(self) -> dict[str, object]:
        """Serialize one exact configuration binding."""
        return {
            "name": self.name,
            "config_path": list(self.config_path),
            "unit": self.unit,
            "interpretation": self.interpretation,
            "value": _thaw_json(self.value),
        }

    @classmethod
    def from_dict(cls, value: object, *, label: str = "binding") -> ToraxConfigBinding:
        """Parse one strict configuration binding."""
        raw = _strict_object(
            value,
            label,
            {"name", "config_path", "unit", "interpretation", "value"},
        )
        return cls(
            name=_string(raw["name"], f"{label}.name"),
            config_path=_string_tuple(raw["config_path"], f"{label}.config_path"),
            unit=_string(raw["unit"], f"{label}.unit"),
            interpretation=_string(raw["interpretation"], f"{label}.interpretation"),
            value=_freeze_json(raw["value"]),
        )


@dataclass(frozen=True)
class ToraxRunRequest:
    """Complete TORAX configuration plus a checked typed consumer view."""

    request_id: str
    event_id: str
    model_id: str
    scenario_id: str
    reactor_family: str
    reactor_id: str
    configuration_id: str
    clock: ToraxClock
    geometry: ToraxGeometry
    initial_state: tuple[ToraxSignal, ...]
    controls: tuple[ToraxSignal, ...]
    models: Mapping[str, object]
    torax_config: Mapping[str, object]
    bindings: tuple[ToraxConfigBinding, ...]
    custody: Mapping[str, object]
    expected_torax_version: str = TORAX_VERSION
    schema: str = TORAX_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        """Validate the full request and every typed-to-config overlap."""
        if self.schema != TORAX_REQUEST_SCHEMA:
            raise ValueError(f"request schema must be {TORAX_REQUEST_SCHEMA!r}")
        for name, value in (
            ("request_id", self.request_id),
            ("event_id", self.event_id),
            ("model_id", self.model_id),
            ("scenario_id", self.scenario_id),
            ("reactor_family", self.reactor_family),
            ("reactor_id", self.reactor_id),
            ("configuration_id", self.configuration_id),
        ):
            _nonempty(value, name)
        if self.expected_torax_version != TORAX_VERSION:
            raise ValueError(f"v1 request requires TORAX {TORAX_VERSION}")
        if not self.initial_state or not self.controls:
            raise ValueError("initial_state and controls must both be non-empty")
        all_signals = self.initial_state + self.controls
        if any(signal.role != "initial_state" for signal in self.initial_state):
            raise ValueError("initial_state contains a non-state signal")
        if any(
            signal.role not in {"control", "disturbance", "parameter"} for signal in self.controls
        ):
            raise ValueError("controls contains an initial-state signal")
        if len({signal.name for signal in all_signals}) != len(all_signals):
            raise ValueError("signal names must be unique")
        if any(
            time_ns < self.clock.initial_ns or time_ns > self.clock.final_ns
            for signal in all_signals
            for time_ns in signal.time_ns
        ):
            raise ValueError("signal time lies outside the request clock")
        binding_by_name = {binding.name: binding for binding in self.bindings}
        if len(binding_by_name) != len(self.bindings):
            raise ValueError("binding names must be unique")
        for signal in all_signals:
            if signal.binding_name not in binding_by_name:
                raise ValueError(f"signal {signal.name} references a missing binding")
            bound_time, bound_coordinate, bound_values = _interpret_binding(
                binding_by_name[signal.binding_name],
                initial_ns=self.clock.initial_ns,
            )
            if (
                signal.time_ns != bound_time
                or signal.coordinate != bound_coordinate
                or signal.values != bound_values
            ):
                raise ValueError(f"signal {signal.name} disagrees with its interpreted binding")
        for binding in self.bindings:
            actual = _resolve_path(self.torax_config, binding.config_path)
            if canonical_sha256(actual) != canonical_sha256(binding.value):
                raise ValueError(f"binding {binding.name} does not match torax_config")
        required = {
            "clock.initial_s": self.clock.initial_ns / 1_000_000_000,
            "clock.final_s": self.clock.final_ns / 1_000_000_000,
            "geometry.major_radius_m": self.geometry.major_radius_m,
            "geometry.minor_radius_m": self.geometry.minor_radius_m,
            "geometry.magnetic_field_t": self.geometry.magnetic_field_t,
        }
        for name, expected in required.items():
            required_binding = binding_by_name.get(name)
            if required_binding is None or canonical_sha256(
                required_binding.value
            ) != canonical_sha256(expected):
                raise ValueError(f"missing or inconsistent required binding: {name}")
        custody = dict(self.custody)
        required_custody = {
            "caller",
            "created_at_utc",
            "source_repo_commit",
            "config_sha256",
            "deck_path",
            "deck_sha256",
        }
        if set(custody) != required_custody:
            raise ValueError("custody fields do not match the v1 contract")
        if custody["config_sha256"] != canonical_sha256(self.torax_config):
            raise ValueError("custody.config_sha256 does not match torax_config")
        for key in required_custody - {"config_sha256"}:
            _nonempty(_string(custody[key], f"custody.{key}"), f"custody.{key}")

    def to_dict(self) -> dict[str, object]:
        """Serialize the complete request without losing backend configuration fields."""
        return {
            "schema": self.schema,
            "request_id": self.request_id,
            "event_id": self.event_id,
            "model_id": self.model_id,
            "scenario_id": self.scenario_id,
            "reactor_family": self.reactor_family,
            "reactor_id": self.reactor_id,
            "configuration_id": self.configuration_id,
            "clock": self.clock.to_dict(),
            "geometry": self.geometry.to_dict(),
            "initial_state": [signal.to_dict() for signal in self.initial_state],
            "controls": [signal.to_dict() for signal in self.controls],
            "models": _thaw_json(self.models),
            "torax_config": _thaw_json(self.torax_config),
            "bindings": [binding.to_dict() for binding in self.bindings],
            "custody": _thaw_json(self.custody),
            "expected_torax_version": self.expected_torax_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> ToraxRunRequest:
        """Parse and validate a v1 request with unknown-field refusal."""
        keys = {
            "schema",
            "request_id",
            "event_id",
            "model_id",
            "scenario_id",
            "reactor_family",
            "reactor_id",
            "configuration_id",
            "clock",
            "geometry",
            "initial_state",
            "controls",
            "models",
            "torax_config",
            "bindings",
            "custody",
            "expected_torax_version",
        }
        raw = _strict_object(value, "request", keys)
        return cls(
            schema=_string(raw["schema"], "request.schema"),
            request_id=_string(raw["request_id"], "request.request_id"),
            event_id=_string(raw["event_id"], "request.event_id"),
            model_id=_string(raw["model_id"], "request.model_id"),
            scenario_id=_string(raw["scenario_id"], "request.scenario_id"),
            reactor_family=_string(raw["reactor_family"], "request.reactor_family"),
            reactor_id=_string(raw["reactor_id"], "request.reactor_id"),
            configuration_id=_string(raw["configuration_id"], "request.configuration_id"),
            clock=ToraxClock.from_dict(raw["clock"]),
            geometry=ToraxGeometry.from_dict(raw["geometry"]),
            initial_state=tuple(
                ToraxSignal.from_dict(item, label=f"initial_state[{index}]")
                for index, item in enumerate(_array(raw["initial_state"], "request.initial_state"))
            ),
            controls=tuple(
                ToraxSignal.from_dict(item, label=f"controls[{index}]")
                for index, item in enumerate(_array(raw["controls"], "request.controls"))
            ),
            models=_freeze_object(raw["models"], "request.models"),
            torax_config=_freeze_object(raw["torax_config"], "request.torax_config"),
            bindings=tuple(
                ToraxConfigBinding.from_dict(item, label=f"bindings[{index}]")
                for index, item in enumerate(_array(raw["bindings"], "request.bindings"))
            ),
            custody=_freeze_object(raw["custody"], "request.custody"),
            expected_torax_version=_string(
                raw["expected_torax_version"], "request.expected_torax_version"
            ),
        )


@dataclass(frozen=True)
class ToraxArtifact:
    """Checksummed complete TORAX output and its inventory manifest."""

    sidecar_path: str
    sidecar_sha256: str
    sidecar_bytes: int
    manifest_path: str
    manifest_sha256: str
    format: str = "NetCDF-DataTree"

    def __post_init__(self) -> None:
        """Validate exact sidecar and manifest custody fields."""
        _nonempty(self.sidecar_path, "artifact.sidecar_path")
        _digest(self.sidecar_sha256, "artifact.sidecar_sha256")
        if self.sidecar_bytes <= 0:
            raise ValueError("artifact.sidecar_bytes must be > 0")
        _nonempty(self.manifest_path, "artifact.manifest_path")
        _digest(self.manifest_sha256, "artifact.manifest_sha256")
        if self.format != "NetCDF-DataTree":
            raise ValueError("unsupported TORAX sidecar format")

    def to_dict(self) -> dict[str, object]:
        """Serialize artifact custody fields."""
        return dict(vars(self))

    @classmethod
    def from_dict(cls, value: object) -> ToraxArtifact:
        """Parse strict artifact custody fields."""
        keys = {
            "sidecar_path",
            "sidecar_sha256",
            "sidecar_bytes",
            "manifest_path",
            "manifest_sha256",
            "format",
        }
        raw = _strict_object(value, "artifact", keys)
        return cls(
            sidecar_path=_string(raw["sidecar_path"], "artifact.sidecar_path"),
            sidecar_sha256=_string(raw["sidecar_sha256"], "artifact.sidecar_sha256"),
            sidecar_bytes=_integer(raw["sidecar_bytes"], "artifact.sidecar_bytes"),
            manifest_path=_string(raw["manifest_path"], "artifact.manifest_path"),
            manifest_sha256=_string(raw["manifest_sha256"], "artifact.manifest_sha256"),
            format=_string(raw["format"], "artifact.format"),
        )


@dataclass(frozen=True)
class ToraxProjection:
    """Consumer-critical, unit-bearing projection of one complete TORAX run."""

    time_ns: tuple[int, ...]
    rho_norm: tuple[float, ...]
    profiles: Mapping[str, tuple[tuple[float, ...], ...]]
    profile_units: Mapping[str, str]
    source_totals: Mapping[str, tuple[float, ...]]
    source_units: Mapping[str, str]
    state_budgets: tuple[Mapping[str, float], ...]
    budget_units: Mapping[str, str]
    numerics: Mapping[str, object]
    uncertainty: Mapping[str, object]
    scientific_sha256: str

    def __post_init__(self) -> None:
        """Validate projection clocks, shapes, values, units, and uncertainty."""
        _strictly_increasing(self.time_ns, "projection.time_ns")
        _strictly_increasing_floats(self.rho_norm, "projection.rho_norm")
        required_profiles = {
            "ion_temperature",
            "electron_temperature",
            "electron_density",
            "poloidal_flux",
        }
        if set(self.profiles) != required_profiles or set(self.profile_units) != required_profiles:
            raise ValueError("projection must contain exactly Ti, Te, ne, and poloidal flux")
        for name, snapshots in self.profiles.items():
            if len(snapshots) != len(self.time_ns) or any(
                len(snapshot) != len(self.rho_norm) for snapshot in snapshots
            ):
                raise ValueError(f"profile {name} shape mismatch")
            _finite_matrix(snapshots, f"projection profile {name}")
        if len(self.state_budgets) != len(self.time_ns):
            raise ValueError("state budget length must match time")
        _digest(self.scientific_sha256, "projection.scientific_sha256")
        if self.uncertainty.get("kind") not in {"not_evaluated", "numerical_refinement"}:
            raise ValueError(
                "result uncertainty must be numerical-refinement-only or not evaluated"
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize the critical projection."""
        return {
            "time_ns": list(self.time_ns),
            "rho_norm": list(self.rho_norm),
            "profiles": {name: [list(row) for row in rows] for name, rows in self.profiles.items()},
            "profile_units": dict(self.profile_units),
            "source_totals": {name: list(values) for name, values in self.source_totals.items()},
            "source_units": dict(self.source_units),
            "state_budgets": [dict(row) for row in self.state_budgets],
            "budget_units": dict(self.budget_units),
            "numerics": _thaw_json(self.numerics),
            "uncertainty": _thaw_json(self.uncertainty),
            "scientific_sha256": self.scientific_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> ToraxProjection:
        """Parse a strict critical projection."""
        keys = {
            "time_ns",
            "rho_norm",
            "profiles",
            "profile_units",
            "source_totals",
            "source_units",
            "state_budgets",
            "budget_units",
            "numerics",
            "uncertainty",
            "scientific_sha256",
        }
        raw = _strict_object(value, "projection", keys)
        profiles_raw = _object(raw["profiles"], "projection.profiles")
        totals_raw = _object(raw["source_totals"], "projection.source_totals")
        budgets_raw = _array(raw["state_budgets"], "projection.state_budgets")
        return cls(
            time_ns=_integer_tuple(raw["time_ns"], "projection.time_ns"),
            rho_norm=_number_tuple(raw["rho_norm"], "projection.rho_norm"),
            profiles=MappingProxyType(
                {
                    name: _number_matrix(item, f"projection.profiles.{name}")
                    for name, item in profiles_raw.items()
                }
            ),
            profile_units=_string_mapping(raw["profile_units"], "projection.profile_units"),
            source_totals=MappingProxyType(
                {
                    name: _number_tuple(item, f"projection.source_totals.{name}")
                    for name, item in totals_raw.items()
                }
            ),
            source_units=_string_mapping(raw["source_units"], "projection.source_units"),
            state_budgets=tuple(
                MappingProxyType(
                    {
                        name: _number(item, f"projection.state_budgets[{index}].{name}")
                        for name, item in _object(row, f"projection.state_budgets[{index}]").items()
                    }
                )
                for index, row in enumerate(budgets_raw)
            ),
            budget_units=_string_mapping(raw["budget_units"], "projection.budget_units"),
            numerics=_freeze_object(raw["numerics"], "projection.numerics"),
            uncertainty=_freeze_object(raw["uncertainty"], "projection.uncertainty"),
            scientific_sha256=_string(raw["scientific_sha256"], "projection.scientific_sha256"),
        )


@dataclass(frozen=True)
class ToraxProvenance:
    """Runtime and source identity needed to reproduce one TORAX outcome."""

    torax_version: str
    torax_license: str
    source_repo_commit: str
    python_version: str
    platform: str
    runtime_backend: str
    precision: str
    request_sha256: str
    config_sha256: str
    deck_sha256: str
    runner_sha256: str
    started_at_utc: str
    finished_at_utc: str

    def __post_init__(self) -> None:
        """Validate complete runtime and source provenance."""
        for key, value in vars(self).items():
            _nonempty(str(value), f"provenance.{key}")
        for key in ("request_sha256", "config_sha256", "deck_sha256", "runner_sha256"):
            _digest(str(getattr(self, key)), f"provenance.{key}")

    def to_dict(self) -> dict[str, object]:
        """Serialize runtime provenance."""
        return dict(vars(self))

    @classmethod
    def from_dict(cls, value: object) -> ToraxProvenance:
        """Parse strict runtime provenance."""
        keys = set(cls.__dataclass_fields__)
        raw = _strict_object(value, "provenance", keys)
        return cls(**{key: _string(raw[key], f"provenance.{key}") for key in keys})


@dataclass(frozen=True)
class ToraxRunOutcome:
    """Discriminated success/failure result for one TORAX request."""

    request_id: str
    event_id: str
    complete: bool
    reached_time_ns: int
    sim_error: str
    provenance: ToraxProvenance
    projection: ToraxProjection | None
    artifact: ToraxArtifact | None
    failure_code: ToraxFailureCode | None
    failure_message: str | None
    schema: str = TORAX_OUTCOME_SCHEMA

    def __post_init__(self) -> None:
        """Enforce the success/failure discriminated-union invariants."""
        if self.schema != TORAX_OUTCOME_SCHEMA:
            raise ValueError(f"outcome schema must be {TORAX_OUTCOME_SCHEMA!r}")
        _nonempty(self.request_id, "outcome.request_id")
        _nonempty(self.event_id, "outcome.event_id")
        _nonempty(self.sim_error, "outcome.sim_error")
        if self.reached_time_ns < 0:
            raise ValueError("outcome.reached_time_ns must be >= 0")
        success = self.failure_code is None
        if success != self.complete:
            raise ValueError("only complete outcomes can be successful")
        if success and (
            self.projection is None or self.artifact is None or self.failure_message is not None
        ):
            raise ValueError("success requires projection and artifact and no failure message")
        if not success and (self.failure_message is None or not self.failure_message.strip()):
            raise ValueError("failure requires a non-empty message")

    @property
    def success(self) -> bool:
        """Return whether the solver and all output/custody gates passed."""
        return self.failure_code is None

    def require_success(self) -> ToraxRunOutcome:
        """Return this outcome or raise a stable typed runtime exception."""
        if self.failure_code is not None:
            raise ToraxRuntimeError(self.failure_code, cast(str, self.failure_message))
        return self

    def to_dict(self) -> dict[str, object]:
        """Serialize the discriminated outcome."""
        return {
            "schema": self.schema,
            "request_id": self.request_id,
            "event_id": self.event_id,
            "success": self.success,
            "complete": self.complete,
            "reached_time_ns": self.reached_time_ns,
            "sim_error": self.sim_error,
            "provenance": self.provenance.to_dict(),
            "projection": None if self.projection is None else self.projection.to_dict(),
            "artifact": None if self.artifact is None else self.artifact.to_dict(),
            "failure_code": None if self.failure_code is None else self.failure_code.value,
            "failure_message": self.failure_message,
        }

    @classmethod
    def from_dict(cls, value: object) -> ToraxRunOutcome:
        """Parse and validate a v1 outcome with unknown-field refusal."""
        keys = {
            "schema",
            "request_id",
            "event_id",
            "success",
            "complete",
            "reached_time_ns",
            "sim_error",
            "provenance",
            "projection",
            "artifact",
            "failure_code",
            "failure_message",
        }
        raw = _strict_object(value, "outcome", keys)
        success = _boolean(raw["success"], "outcome.success")
        failure_raw = raw["failure_code"]
        code = (
            None
            if failure_raw is None
            else ToraxFailureCode(_string(failure_raw, "outcome.failure_code"))
        )
        if success != (code is None):
            raise ValueError("outcome.success disagrees with failure_code")
        projection_raw = raw["projection"]
        artifact_raw = raw["artifact"]
        message_raw = raw["failure_message"]
        return cls(
            schema=_string(raw["schema"], "outcome.schema"),
            request_id=_string(raw["request_id"], "outcome.request_id"),
            event_id=_string(raw["event_id"], "outcome.event_id"),
            complete=_boolean(raw["complete"], "outcome.complete"),
            reached_time_ns=_integer(raw["reached_time_ns"], "outcome.reached_time_ns"),
            sim_error=_string(raw["sim_error"], "outcome.sim_error"),
            provenance=ToraxProvenance.from_dict(raw["provenance"]),
            projection=None
            if projection_raw is None
            else ToraxProjection.from_dict(projection_raw),
            artifact=None if artifact_raw is None else ToraxArtifact.from_dict(artifact_raw),
            failure_code=code,
            failure_message=None
            if message_raw is None
            else _string(message_raw, "outcome.failure_message"),
        )


def _strict_object(value: object, label: str, keys: set[str]) -> dict[str, object]:
    raw = _object(value, label)
    if set(raw) != keys:
        missing = sorted(keys - set(raw))
        unknown = sorted(set(raw) - keys)
        raise ValueError(f"{label} fields differ; missing={missing}, unknown={unknown}")
    return raw


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return {cast(str, key): item for key, item in value.items()}


def _array(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return list(value)


def _string(value: object, label: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(
            f"{label} must be a non-empty string" if not empty else f"{label} must be a string"
        )
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{label} must be finite")
    return result


def _optional_number(value: object, label: str) -> float | None:
    return None if value is None else _number(value, label)


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    return tuple(
        _string(item, f"{label}[{index}]") for index, item in enumerate(_array(value, label))
    )


def _integer_tuple(value: object, label: str) -> tuple[int, ...]:
    return tuple(
        _integer(item, f"{label}[{index}]") for index, item in enumerate(_array(value, label))
    )


def _number_tuple(value: object, label: str) -> tuple[float, ...]:
    return tuple(
        _number(item, f"{label}[{index}]") for index, item in enumerate(_array(value, label))
    )


def _number_matrix(value: object, label: str) -> tuple[tuple[float, ...], ...]:
    return tuple(
        _number_tuple(row, f"{label}[{index}]") for index, row in enumerate(_array(value, label))
    )


def _string_mapping(value: object, label: str) -> Mapping[str, str]:
    raw = _object(value, label)
    return MappingProxyType({name: _string(item, f"{label}.{name}") for name, item in raw.items()})


def _nonempty(value: str, label: str) -> None:
    if not value.strip():
        raise ValueError(f"{label} must be non-empty")


def _positive_finite(value: float, label: str) -> None:
    if value <= 0.0 or value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{label} must be finite and > 0")


def _strictly_increasing(values: Sequence[int], label: str) -> None:
    if not values or any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(f"{label} must be non-empty and strictly increasing")


def _strictly_increasing_floats(values: Sequence[float], label: str) -> None:
    if not values or any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(f"{label} must be non-empty and strictly increasing")
    _finite_matrix((tuple(values),), label)


def _finite_matrix(values: Sequence[Sequence[float]], label: str) -> None:
    if any(item != item or item in (float("inf"), float("-inf")) for row in values for item in row):
        raise ValueError(f"{label} must contain only finite values")


def _digest(value: str, label: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _resolve_path(root: Mapping[str, object], path: tuple[str, ...]) -> object:
    current: object = root
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"configuration path does not exist: {'/'.join(path)}")
        current = current[part]
    return current


def _interpret_binding(
    binding: ToraxConfigBinding,
    *,
    initial_ns: int,
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[tuple[float, ...], ...]]:
    if binding.interpretation == "scalar":
        return (initial_ns,), (), ((_number(binding.value, f"binding {binding.name} value"),),)
    time_map = _object(binding.value, f"binding {binding.name} value")
    parsed_times = sorted(
        (
            _seconds_key_to_ns(key, f"binding {binding.name} time"),
            item,
        )
        for key, item in time_map.items()
    )
    times = tuple(time_ns for time_ns, _ in parsed_times)
    if binding.interpretation == "time_scalar_map":
        values = tuple(
            (_number(item, f"binding {binding.name} value"),) for _, item in parsed_times
        )
        return times, (), values
    coordinate: tuple[float, ...] | None = None
    rows: list[tuple[float, ...]] = []
    for _, item in parsed_times:
        radial_map = _object(item, f"binding {binding.name} radial map")
        parsed_radial = sorted(
            (_numeric_key(key, f"binding {binding.name} coordinate"), value)
            for key, value in radial_map.items()
        )
        row_coordinate = tuple(position for position, _ in parsed_radial)
        if coordinate is None:
            coordinate = row_coordinate
        elif coordinate != row_coordinate:
            raise ValueError(f"binding {binding.name} radial coordinates change over time")
        rows.append(
            tuple(
                _number(value, f"binding {binding.name} radial value") for _, value in parsed_radial
            )
        )
    if coordinate is None:
        raise ValueError(f"binding {binding.name} time-radial map is empty")
    return times, coordinate, tuple(rows)


def _seconds_key_to_ns(value: str, label: str) -> int:
    seconds = _numeric_key(value, label)
    scaled = seconds * 1_000_000_000.0
    rounded = round(scaled)
    if abs(scaled - rounded) > 1e-3:
        raise ValueError(f"{label} is not exactly representable in integer nanoseconds")
    return rounded


def _numeric_key(value: str, label: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a numeric object key") from error
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{label} must be finite")
    return result


def _freeze_object(value: object, label: str) -> Mapping[str, object]:
    return cast(Mapping[str, object], _freeze_json(_object(value, label)))


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    canonical_sha256(value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


__all__ = [
    "SIMULATION_CLOCK_DOMAIN",
    "TORAX_OUTCOME_SCHEMA",
    "TORAX_REQUEST_SCHEMA",
    "TORAX_VERSION",
    "ToraxArtifact",
    "ToraxClock",
    "ToraxConfigBinding",
    "ToraxFailureCode",
    "ToraxGeometry",
    "ToraxProjection",
    "ToraxProvenance",
    "ToraxRunOutcome",
    "ToraxRunRequest",
    "ToraxRuntimeError",
    "ToraxSignal",
]
