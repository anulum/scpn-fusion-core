# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TORAX Deterministic Review Envelope
"""Deterministic, non-actuating TORAX evidence envelope for SPO consumers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from .contracts import TORAX_OUTCOME_SCHEMA, ToraxRunOutcome, ToraxRunRequest
from .serialization import canonical_json_bytes, canonical_sha256

TORAX_REVIEW_SCHEMA = "scpn-fusion-core.torax-runtime-review-envelope.v1"
COUPLED_TRANSPORT_SOURCE_SCHEMA = "scpn-fusion-core.coupled-transport-model-intersection.v1"
MAX_REVIEW_ENVELOPE_BYTES = 64 * 1024 * 1024
_U0_REACTOR_KEYS = {
    "cadence",
    "configuration",
    "configuration_version",
    "confinement_family",
    "context_id",
    "conversion",
    "coordinate_frame",
    "drivers",
    "evidence_class",
    "event_id",
    "facility",
    "operating_point",
    "reaction",
    "registry_digest",
    "registry_version",
    "schema_version",
    "topology",
}
_CLOCK_KEYS = {
    "domain",
    "epoch",
    "kind",
    "latency_s",
    "picosecond_offset",
    "requested_final_ns",
    "reset_policy",
    "sample_ns",
    "sample_rate_hz",
    "synchronized_to",
    "timestamp_ns",
}
_CALIBRATION_KEYS = {"calibrated_at_ns", "calibration_id", "transfer_function_id"}
_FORBIDDEN_TYPED_KEYS = {"q95", "li3", "beta_N", "W_thermal_total", "regime", "phase"}
_NONDETERMINISTIC_KEYS = {
    "started_at_utc",
    "finished_at_utc",
    "platform",
    "sidecar_path",
    "manifest_path",
}


@dataclass(frozen=True)
class ToraxReviewEnvelope:
    """Canonical review-only projection with numerical-refinement uncertainty."""

    source_schema: str
    source_revision: str
    model_intersection_schema: str
    event_id: str
    payload: Mapping[str, object]
    provenance: Mapping[str, object]
    payload_sha256: str
    schema: str = TORAX_REVIEW_SCHEMA

    def __post_init__(self) -> None:
        """Reject incomplete, nondeterministic, inferred, or actuating payloads."""
        if self.schema != TORAX_REVIEW_SCHEMA:
            raise ValueError(f"review schema must be {TORAX_REVIEW_SCHEMA!r}")
        if self.source_schema != TORAX_OUTCOME_SCHEMA:
            raise ValueError(f"source_schema must be {TORAX_OUTCOME_SCHEMA!r}")
        _commit(self.source_revision, "source_revision")
        if self.model_intersection_schema != COUPLED_TRANSPORT_SOURCE_SCHEMA:
            raise ValueError(
                f"model_intersection_schema must be {COUPLED_TRANSPORT_SOURCE_SCHEMA!r}"
            )
        if not self.event_id.strip():
            raise ValueError("event_id must be non-empty")
        _exact_keys(
            self.payload,
            "payload",
            {"clock", "reactor", "observables", "completion", "uncertainty", "validity"},
        )
        _validate_clock(_mapping(self.payload["clock"], "payload.clock"))
        _validate_reactor(
            _mapping(self.payload["reactor"], "payload.reactor"), event_id=self.event_id
        )
        observables = _mapping(self.payload["observables"], "payload.observables")
        _validate_observables(observables)
        _validate_uncertainty(
            _mapping(self.payload["uncertainty"], "payload.uncertainty"), observables
        )
        validity = _mapping(self.payload["validity"], "payload.validity")
        _exact_keys(validity, "payload.validity", {"authority", "ood", "quality", "state"})
        if validity["state"] != "VALID" or validity["authority"] != "review_only_non_actuating":
            raise ValueError("review envelope must be VALID and review-only/non-actuating")
        if validity["ood"] is not False:
            raise ValueError("canonical review evidence must explicitly be in distribution")
        completion = _mapping(self.payload["completion"], "payload.completion")
        _exact_keys(
            completion,
            "payload.completion",
            {"complete", "reached_final_ns", "sim_error"},
        )
        if completion["complete"] is not True or completion["sim_error"] != "NO_ERROR":
            raise ValueError("review envelope requires a complete NO_ERROR source run")
        _validate_provenance(self.provenance)
        forbidden = _find_keys(self.to_dict_without_digest(), _FORBIDDEN_TYPED_KEYS)
        if forbidden:
            raise ValueError(
                f"review envelope contains forbidden inferred keys: {sorted(forbidden)}"
            )
        nondeterministic = _find_keys(self.to_dict_without_digest(), _NONDETERMINISTIC_KEYS)
        if nondeterministic:
            raise ValueError(
                f"review envelope contains nondeterministic custody keys: {sorted(nondeterministic)}"
            )
        _digest(self.payload_sha256, "payload_sha256")
        if canonical_sha256(self.payload) != self.payload_sha256:
            raise ValueError("payload_sha256 does not match the deterministic payload")

    def to_dict_without_digest(self) -> dict[str, object]:
        """Return deterministic fields other than the derived payload digest."""
        return {
            "schema": self.schema,
            "source_schema": self.source_schema,
            "source_revision": self.source_revision,
            "model_intersection_schema": self.model_intersection_schema,
            "event_id": self.event_id,
            "payload": _plain(self.payload),
            "provenance": _plain(self.provenance),
        }

    def to_dict(self) -> dict[str, object]:
        """Serialize the canonical review envelope."""
        return {**self.to_dict_without_digest(), "payload_sha256": self.payload_sha256}

    @classmethod
    def from_dict(cls, value: object) -> ToraxReviewEnvelope:
        """Parse and verify a deterministic review envelope."""
        raw = _mapping(value, "review_envelope")
        _exact_keys(
            raw,
            "review_envelope",
            {
                "schema",
                "source_schema",
                "source_revision",
                "model_intersection_schema",
                "event_id",
                "payload",
                "provenance",
                "payload_sha256",
            },
        )
        return cls(
            schema=_text(raw["schema"], "schema"),
            source_schema=_text(raw["source_schema"], "source_schema"),
            source_revision=_text(raw["source_revision"], "source_revision"),
            model_intersection_schema=_text(
                raw["model_intersection_schema"], "model_intersection_schema"
            ),
            event_id=_text(raw["event_id"], "event_id"),
            payload=_freeze_mapping(raw["payload"], "payload"),
            provenance=_freeze_mapping(raw["provenance"], "provenance"),
            payload_sha256=_text(raw["payload_sha256"], "payload_sha256"),
        )


def build_review_envelope(
    *,
    request: ToraxRunRequest,
    refined_request: ToraxRunRequest,
    primary: ToraxRunOutcome,
    refined: ToraxRunOutcome,
    refinement_metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    primary_dt_ns: int,
    refined_dt_ns: int,
    source_revision: str,
    runtime_source_sha256: str,
    artifact_content_sha256: str,
    manifest_inventory_sha256: str,
) -> ToraxReviewEnvelope:
    """Project real primary/refined outcomes into deterministic review bytes."""
    primary.require_success()
    refined.require_success()
    if primary.projection is None or primary.artifact is None:
        raise ValueError("primary outcome lacks projection or artifact")
    if refined.projection is None:
        raise ValueError("refined outcome lacks projection")
    _commit(source_revision, "source_revision")
    if (
        primary.request_id != request.request_id
        or primary.provenance.request_sha256 != canonical_sha256(request.to_dict())
    ):
        raise ValueError("primary outcome does not bind the supplied primary request")
    if (
        refined.request_id != refined_request.request_id
        or refined.provenance.request_sha256 != canonical_sha256(refined_request.to_dict())
    ):
        raise ValueError("refined outcome does not bind the supplied refined request")
    projection = primary.projection
    calibration = {
        "calibrated_at_ns": request.clock.initial_ns,
        "calibration_id": "fusion.torax.direct_model_output.v1",
        "transfer_function_id": "fusion.torax.identity_projection.v1",
    }

    def observable(unit: str, samples: object) -> dict[str, object]:
        return {"calibration": dict(calibration), "samples": samples, "unit": unit}

    plasma = _mapping(request.torax_config["plasma_composition"], "plasma_composition")
    profile_conditions = _mapping(request.torax_config["profile_conditions"], "profile_conditions")
    sample_interval_ns = projection.time_ns[1] - projection.time_ns[0]
    payload: dict[str, object] = {
        "clock": {
            "domain": request.clock.domain,
            "epoch": request.clock.epoch,
            "kind": "simulation_monotonic",
            "latency_s": 0.0,
            "picosecond_offset": 0,
            "requested_final_ns": request.clock.final_ns,
            "reset_policy": request.clock.reset_policy,
            "sample_ns": list(projection.time_ns),
            "sample_rate_hz": 1_000_000_000.0 / sample_interval_ns,
            "synchronized_to": None,
            "timestamp_ns": projection.time_ns[-1],
        },
        "reactor": {
            "cadence": "single_experiment",
            "configuration": "conventional_tokamak",
            "configuration_version": "1.0.0",
            "confinement_family": "magnetic_closed",
            "context_id": "fusion.torax.circular_iter_scale_comparison",
            "conversion": "experimental_no_power_conversion",
            "coordinate_frame": request.geometry.frame,
            "drivers": ["external_magnetic_coils", "plasma_current", "combined"],
            "evidence_class": "S",
            "event_id": request.event_id,
            "facility": "simulation_only_no_facility",
            "operating_point": {
                "effective_charge": plasma["Z_eff"],
                "impurity": plasma["impurity"],
                "magnetic_field_t": request.geometry.magnetic_field_t,
                "main_ion": plasma["main_ion"],
                "major_radius_m": request.geometry.major_radius_m,
                "minor_radius_m": request.geometry.minor_radius_m,
                "plasma_current_a": profile_conditions["Ip"],
            },
            "reaction": "deuterium_deuterium",
            "registry_digest": "786d9542ce76c56dd7748fa948b17efed6c073525e527ce90e6d5e29a2d00090",
            "registry_version": "1.0.0",
            "schema_version": "1.0.0",
            "topology": "axisymmetric torus",
        },
        "observables": {
            "rho": {
                "frame": request.geometry.frame,
                "name": request.geometry.radial_coordinate,
                "samples": list(projection.rho_norm),
                "unit": request.geometry.radial_coordinate_unit,
            },
            "profiles": {
                name: observable(projection.profile_units[name], [list(row) for row in rows])
                for name, rows in projection.profiles.items()
            },
            "source_totals": {
                name: observable(projection.source_units[name], list(values))
                for name, values in projection.source_totals.items()
            },
            "state_budgets": {
                name: observable(
                    projection.budget_units[name],
                    [row[name] for row in projection.state_budgets],
                )
                for name in sorted(projection.budget_units)
            },
            "numerics": _plain(projection.numerics),
        },
        "completion": {
            "complete": primary.complete,
            "sim_error": primary.sim_error,
            "reached_final_ns": primary.reached_time_ns,
        },
        "uncertainty": {
            "kind": "numerical_refinement",
            "primary_dt_ns": primary_dt_ns,
            "refined_dt_ns": refined_dt_ns,
            "observables": _plain(refinement_metrics),
        },
        "validity": {
            "state": "VALID",
            "quality": "frozen_model_intersection_reference",
            "authority": "review_only_non_actuating",
            "ood": False,
        },
    }
    custody = request.custody
    provenance: dict[str, object] = {
        "model_intersection_revision": custody["source_repo_commit"],
        "runtime_source_sha256": runtime_source_sha256,
        "request_sha256": primary.provenance.request_sha256,
        "refined_request_sha256": refined.provenance.request_sha256,
        "deck_sha256": primary.provenance.deck_sha256,
        "runner_sha256": primary.provenance.runner_sha256,
        "artifact_content_sha256": artifact_content_sha256,
        "manifest_inventory_sha256": manifest_inventory_sha256,
        "primary_projection_sha256": projection.scientific_sha256,
        "refined_projection_sha256": refined.projection.scientific_sha256,
    }
    return ToraxReviewEnvelope(
        source_schema=TORAX_OUTCOME_SCHEMA,
        source_revision=source_revision,
        model_intersection_schema=COUPLED_TRANSPORT_SOURCE_SCHEMA,
        event_id=request.event_id,
        payload=MappingProxyType(payload),
        provenance=MappingProxyType(provenance),
        payload_sha256=canonical_sha256(payload),
    )


def review_envelope_to_bytes(envelope: ToraxReviewEnvelope) -> bytes:
    """Return the unique canonical UTF-8 representation of an envelope."""
    if not isinstance(envelope, ToraxReviewEnvelope):
        raise TypeError("envelope must be a ToraxReviewEnvelope")
    payload = canonical_json_bytes(envelope.to_dict())
    if len(payload) > MAX_REVIEW_ENVELOPE_BYTES:
        raise ValueError("review envelope exceeds the maximum byte size")
    return payload


def review_envelope_from_bytes(
    payload: bytes,
    *,
    expected_sha256: str | None = None,
) -> ToraxReviewEnvelope:
    """Decode canonical bytes and refuse size, digest, duplicate, or encoding drift."""
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("review envelope must be non-empty bytes")
    if len(payload) > MAX_REVIEW_ENVELOPE_BYTES:
        raise ValueError("review envelope exceeds the maximum byte size")
    if expected_sha256 is not None:
        _digest(expected_sha256, "expected_sha256")
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError("review envelope byte digest mismatch")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("review envelope bytes must be strict UTF-8") from error
    try:
        raw = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise ValueError("review envelope JSON is invalid") from error
    envelope = ToraxReviewEnvelope.from_dict(raw)
    if review_envelope_to_bytes(envelope) != payload:
        raise ValueError("review envelope bytes must use canonical JSON")
    return envelope


def review_envelope_sha256(envelope: ToraxReviewEnvelope) -> str:
    """Hash the exact canonical review-envelope bytes."""
    return hashlib.sha256(review_envelope_to_bytes(envelope)).hexdigest()


def _validate_clock(clock: Mapping[str, object]) -> None:
    _exact_keys(clock, "payload.clock", _CLOCK_KEYS)
    if clock["domain"] != "simulation_monotonic" or clock["kind"] != "simulation_monotonic":
        raise ValueError("review clock must be simulation_monotonic")
    if clock["latency_s"] != 0.0:
        raise ValueError("direct simulation projection latency must be declared as 0.0 s")
    if clock["reset_policy"] != "fresh_process_no_hidden_state":
        raise ValueError("review clock must retain the fresh-process reset policy")
    sample_ns = _integer_sequence(clock["sample_ns"], "payload.clock.sample_ns")
    if any(right <= left for left, right in zip(sample_ns, sample_ns[1:])):
        raise ValueError("review sample_ns must be strictly increasing")
    if clock["timestamp_ns"] != sample_ns[-1]:
        raise ValueError("review timestamp_ns must equal the final sample")
    if clock["requested_final_ns"] != sample_ns[-1]:
        raise ValueError("review clock must reach the requested final time")
    _nonnegative_finite(clock["latency_s"], "payload.clock.latency_s")
    _positive_finite(clock["sample_rate_hz"], "payload.clock.sample_rate_hz")


def _validate_reactor(reactor: Mapping[str, object], *, event_id: str) -> None:
    _exact_keys(reactor, "payload.reactor", _U0_REACTOR_KEYS)
    if reactor["schema_version"] != "1.0.0" or reactor["registry_version"] != "1.0.0":
        raise ValueError("review reactor must declare U0 and registry version 1.0.0")
    _digest(_text(reactor["registry_digest"], "payload.reactor.registry_digest"), "registry_digest")
    if reactor["event_id"] != event_id:
        raise ValueError("review reactor event_id must match the envelope")
    drivers = reactor["drivers"]
    if not isinstance(drivers, (list, tuple)) or not drivers or len(set(drivers)) != len(drivers):
        raise ValueError("review reactor drivers must be a non-empty unique sequence")
    operating_point = _mapping(reactor["operating_point"], "payload.reactor.operating_point")
    _exact_keys(
        operating_point,
        "payload.reactor.operating_point",
        {
            "effective_charge",
            "impurity",
            "magnetic_field_t",
            "main_ion",
            "major_radius_m",
            "minor_radius_m",
            "plasma_current_a",
        },
    )


def _validate_observables(observables: Mapping[str, object]) -> None:
    _exact_keys(
        observables,
        "payload.observables",
        {"numerics", "profiles", "rho", "source_totals", "state_budgets"},
    )
    rho = _mapping(observables["rho"], "payload.observables.rho")
    _exact_keys(rho, "payload.observables.rho", {"frame", "name", "samples", "unit"})
    for category in ("profiles", "source_totals", "state_budgets"):
        entries = _mapping(observables[category], f"payload.observables.{category}")
        if not entries:
            raise ValueError(f"payload.observables.{category} must not be empty")
        for name, value in entries.items():
            label = f"payload.observables.{category}.{name}"
            item = _mapping(value, label)
            _exact_keys(item, label, {"calibration", "samples", "unit"})
            _text(item["unit"], f"{label}.unit")
            calibration = _mapping(item["calibration"], f"{label}.calibration")
            _exact_keys(calibration, f"{label}.calibration", _CALIBRATION_KEYS)
            if calibration["calibrated_at_ns"] != 0:
                raise ValueError(
                    "direct model-output calibration must be anchored at scenario start"
                )
            if calibration["transfer_function_id"] != "fusion.torax.identity_projection.v1":
                raise ValueError("review observables require the declared identity projection")


def _validate_uncertainty(
    uncertainty: Mapping[str, object], observables: Mapping[str, object]
) -> None:
    _exact_keys(
        uncertainty,
        "payload.uncertainty",
        {"kind", "observables", "primary_dt_ns", "refined_dt_ns"},
    )
    if uncertainty["kind"] != "numerical_refinement":
        raise ValueError("review uncertainty must be numerical_refinement")
    primary_dt = _integer(uncertainty["primary_dt_ns"], "primary_dt_ns")
    refined_dt = _integer(uncertainty["refined_dt_ns"], "refined_dt_ns")
    if primary_dt <= refined_dt or refined_dt <= 0:
        raise ValueError("refined_dt_ns must be positive and smaller than primary_dt_ns")
    metrics = _mapping(uncertainty["observables"], "payload.uncertainty.observables")
    _exact_keys(
        metrics, "payload.uncertainty.observables", {"profiles", "source_totals", "state_budgets"}
    )
    for category in ("profiles", "source_totals", "state_budgets"):
        expected = _mapping(observables[category], f"payload.observables.{category}")
        actual = _mapping(metrics[category], f"payload.uncertainty.observables.{category}")
        _exact_keys(actual, f"payload.uncertainty.observables.{category}", set(expected))
        for name, raw in actual.items():
            metric = _mapping(raw, f"uncertainty.{category}.{name}")
            _exact_keys(
                metric, f"uncertainty.{category}.{name}", {"absolute_rms", "relative_l2", "unit"}
            )
            expected_item = _mapping(expected[name], f"observables.{category}.{name}")
            if metric["unit"] != expected_item["unit"]:
                raise ValueError(f"uncertainty unit mismatch for {category}.{name}")
            _nonnegative_finite(metric["absolute_rms"], f"{category}.{name}.absolute_rms")
            _nonnegative_finite(metric["relative_l2"], f"{category}.{name}.relative_l2")


def _validate_provenance(provenance: Mapping[str, object]) -> None:
    _exact_keys(
        provenance,
        "provenance",
        {
            "artifact_content_sha256",
            "deck_sha256",
            "manifest_inventory_sha256",
            "model_intersection_revision",
            "primary_projection_sha256",
            "refined_projection_sha256",
            "refined_request_sha256",
            "request_sha256",
            "runner_sha256",
            "runtime_source_sha256",
        },
    )
    for name, digest in provenance.items():
        if name == "model_intersection_revision":
            _commit(str(digest), f"provenance.{name}")
        else:
            _digest(str(digest), f"provenance.{name}")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _exact_keys(value: Mapping[str, object], label: str, expected: set[str]) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} fields differ; missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


def _find_keys(value: object, targets: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in targets:
                found.add(str(key))
            found.update(_find_keys(item, targets))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_find_keys(item, targets))
    return found


def _freeze_mapping(value: object, label: str) -> Mapping[str, object]:
    return cast(Mapping[str, object], _freeze(_mapping(value, label)))


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    canonical_sha256(value)
    return value


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _integer_sequence(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a non-empty integer sequence")
    return tuple(_integer(item, label) for item in value)


def _nonnegative_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite non-negative number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return parsed


def _positive_finite(value: object, label: str) -> float:
    parsed = _nonnegative_finite(value, label)
    if parsed <= 0.0:
        raise ValueError(f"{label} must be > 0")
    return parsed


def _digest(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _commit(value: str, label: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase Git commit")


def _unique_object(pairs: Iterable[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


__all__ = [
    "COUPLED_TRANSPORT_SOURCE_SCHEMA",
    "MAX_REVIEW_ENVELOPE_BYTES",
    "TORAX_REVIEW_SCHEMA",
    "ToraxReviewEnvelope",
    "build_review_envelope",
    "review_envelope_from_bytes",
    "review_envelope_sha256",
    "review_envelope_to_bytes",
]
