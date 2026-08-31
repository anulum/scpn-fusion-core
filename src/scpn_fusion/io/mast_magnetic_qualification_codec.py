# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FAIR-MAST magnetic diagnostic qualification codec
"""Canonical codec for FAIR-MAST magnetic diagnostic qualification evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import cast

from .mast_magnetic_archive_codec import JsonObject, JsonValue

MAST_MAGNETIC_DIAGNOSTIC_QUALIFICATION_SCHEMA = (
    "scpn-fusion-core.mast-magnetic-diagnostic-qualification.v1"
)
MAST_MAGNETIC_DIAGNOSTIC_QUALIFICATION_SCHEMA_VERSION = "1.0.0"
MAX_MAST_MAGNETIC_DIAGNOSTIC_QUALIFICATION_BYTES = 8 * 1024 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_PAYLOAD_KEYS = {
    "archive_envelope_sha256",
    "archive_observation_id",
    "array_inventory",
    "authority",
    "channel_geometry_evidence",
    "clock_evidence",
    "completeness",
    "event_identity",
    "external_limitations",
    "facility",
    "ingestion_mapping",
    "measurement_evidence",
    "producer_project",
    "qualification_summary",
    "reactor_configuration",
    "shot_id",
    "source_archive",
}
_MEASUREMENT_NAMES = {
    "b_field_pol_probe_cc_field",
    "b_field_pol_probe_ccbv_field",
    "b_field_pol_probe_obr_field",
    "b_field_pol_probe_obv_field",
    "b_field_pol_probe_omv_voltage",
    "b_field_tor_probe_cc_field",
    "b_field_tor_probe_omaha_voltage",
    "b_field_tor_probe_saddle_field",
    "b_field_tor_probe_saddle_voltage",
    "flux_loop_flux",
    "ip",
}
_AUTHORITY: JsonObject = {
    "actionable": False,
    "classification_performed": False,
    "direct_actuation": False,
    "execution_permitted": False,
    "phase_inference_performed": False,
    "review_only": True,
}
_QUALIFICATION_SUMMARY: JsonObject = {
    "calibration_state": "applied_transforms_recorded_lineage_unavailable",
    "channel_geometry_mapping_state": "identifier_correspondence_only",
    "event_identity_state": "shot_only_event_unresolved",
    "observation_operator_state": "quantity_paths_only_transfer_functions_unavailable",
    "provider_quality_state": "not_supplied",
    "source_clock_relationship_state": "derived_archive_grids_no_instrument_clock_relation",
    "uncertainty_state": "not_supplied",
    "validity_state": "source_shot_ranges_only",
}


class MastMagneticDiagnosticQualificationError(ValueError):
    """Raised when diagnostic qualification evidence is incomplete or noncanonical."""


@dataclass(frozen=True)
class MastMagneticDiagnosticQualification:
    """Validated immutable qualification evidence for one complete archive envelope."""

    _canonical_bytes: bytes

    def to_bytes(self) -> bytes:
        """Return the exact canonical UTF-8 bytes."""
        return self._canonical_bytes

    @property
    def sha256(self) -> str:
        """Return the SHA-256 identity of the canonical document."""
        return hashlib.sha256(self._canonical_bytes).hexdigest()

    @property
    def document(self) -> JsonObject:
        """Return a defensive copy of the qualification document."""
        return deepcopy(_parse_json_object(self._canonical_bytes))

    @property
    def payload(self) -> JsonObject:
        """Return a defensive copy of the validated payload."""
        return deepcopy(_as_object(self.document["payload"], "payload"))


def encode_mast_magnetic_diagnostic_qualification(
    payload: Mapping[str, JsonValue],
) -> MastMagneticDiagnosticQualification:
    """Validate qualification evidence and bind it to canonical transport bytes."""
    payload_copy = deepcopy(dict(payload))
    validate_mast_magnetic_diagnostic_qualification_payload(payload_copy)
    payload_bytes = _canonical_json_bytes(payload_copy)
    document: JsonObject = {
        "payload": payload_copy,
        "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "schema": MAST_MAGNETIC_DIAGNOSTIC_QUALIFICATION_SCHEMA,
        "schema_version": MAST_MAGNETIC_DIAGNOSTIC_QUALIFICATION_SCHEMA_VERSION,
    }
    encoded = _canonical_json_bytes(document)
    if len(encoded) > MAX_MAST_MAGNETIC_DIAGNOSTIC_QUALIFICATION_BYTES:
        raise MastMagneticDiagnosticQualificationError("qualification document is too large")
    return MastMagneticDiagnosticQualification(encoded)


def decode_mast_magnetic_diagnostic_qualification(
    data: bytes,
) -> MastMagneticDiagnosticQualification:
    """Decode canonical qualification bytes and reject structural or semantic drift."""
    if len(data) > MAX_MAST_MAGNETIC_DIAGNOSTIC_QUALIFICATION_BYTES:
        raise MastMagneticDiagnosticQualificationError("qualification document is too large")
    document = _parse_json_object(data)
    _require_exact_keys(
        document,
        {"payload", "payload_sha256", "schema", "schema_version"},
        "document",
    )
    _require_equal(
        document["schema"],
        MAST_MAGNETIC_DIAGNOSTIC_QUALIFICATION_SCHEMA,
        "schema",
    )
    _require_equal(
        document["schema_version"],
        MAST_MAGNETIC_DIAGNOSTIC_QUALIFICATION_SCHEMA_VERSION,
        "schema_version",
    )
    payload = _as_object(document["payload"], "payload")
    validate_mast_magnetic_diagnostic_qualification_payload(payload)
    _require_equal(
        document["payload_sha256"],
        hashlib.sha256(_canonical_json_bytes(payload)).hexdigest(),
        "payload_sha256",
    )
    if _canonical_json_bytes(document) != data:
        raise MastMagneticDiagnosticQualificationError("document is not canonical JSON")
    return MastMagneticDiagnosticQualification(data)


def validate_mast_magnetic_diagnostic_qualification_payload(payload: JsonObject) -> None:
    """Validate exact source binding, evidence completeness and non-actuating authority."""
    _require_exact_keys(payload, _PAYLOAD_KEYS, "payload")
    _require_equal(payload["producer_project"], "SCPN-FUSION-CORE", "producer_project")
    _require_equal(payload["source_archive"], "FAIR-MAST", "source_archive")
    _require_equal(payload["facility"], "MAST", "facility")
    _require_equal(payload["reactor_configuration"], "spherical_tokamak", "configuration")
    shot_id = _as_positive_integer(payload["shot_id"], "shot_id")
    _require_sha256(payload["archive_envelope_sha256"], "archive_envelope_sha256")
    observation_id = _as_nonempty_string(payload["archive_observation_id"], "observation_id")
    if not observation_id.startswith(f"mast-{shot_id}-complete-magnetics-"):
        raise MastMagneticDiagnosticQualificationError("observation_id does not bind shot_id")
    _require_equal(payload["authority"], _AUTHORITY, "authority")
    _require_equal(payload["qualification_summary"], _QUALIFICATION_SUMMARY, "summary")

    mapping = _as_object(payload["ingestion_mapping"], "ingestion_mapping")
    _require_exact_keys(
        mapping,
        {
            "dataset_license_name",
            "dataset_license_url",
            "mapping_path",
            "mapping_sha256",
            "mapping_url",
            "source_revision",
            "source_tree_state",
        },
        "ingestion_mapping",
    )
    _require_sha256(mapping["mapping_sha256"], "mapping_sha256")
    _require_matching_string(mapping["source_revision"], _REVISION_RE, "source_revision")
    if mapping["source_tree_state"] not in {"clean", "dirty"}:
        raise MastMagneticDiagnosticQualificationError("source_tree_state is unsupported")
    _require_equal(mapping["mapping_path"], "mappings/level2/mast.yml", "mapping_path")
    revision = cast(str, mapping["source_revision"])
    _require_equal(
        mapping["mapping_url"],
        "https://raw.githubusercontent.com/ukaea/fair-mast-ingestion/"
        f"{revision}/mappings/level2/mast.yml",
        "mapping URL",
    )
    _require_equal(mapping["dataset_license_name"], "Creative Commons 4.0 BY-SA", "license")
    _require_https_url(mapping["dataset_license_url"], "dataset_license_url")

    arrays = _validate_array_inventory(_as_list(payload["array_inventory"], "array_inventory"))
    clocks = _validate_clocks(_as_list(payload["clock_evidence"], "clock_evidence"), arrays)
    measurements = _validate_measurements(
        _as_list(payload["measurement_evidence"], "measurement_evidence"), arrays, clocks, shot_id
    )
    mappings = _validate_channel_mappings(
        _as_list(payload["channel_geometry_evidence"], "channel_geometry_evidence"),
        measurements,
        arrays,
    )
    _validate_event_identity(_as_object(payload["event_identity"], "event_identity"), shot_id)
    _validate_external_limitations(
        _as_list(payload["external_limitations"], "external_limitations")
    )
    _validate_completeness(
        _as_object(payload["completeness"], "completeness"),
        arrays,
        clocks,
        measurements,
        mappings,
    )


def _validate_array_inventory(raw_arrays: list[JsonValue]) -> dict[str, JsonObject]:
    if len(raw_arrays) != 72:
        raise MastMagneticDiagnosticQualificationError("array inventory must contain 72 arrays")
    arrays: dict[str, JsonObject] = {}
    allowed_roles = {"channel_coordinate", "clock", "geometry", "measurement", "shot_identity"}
    for index, raw in enumerate(raw_arrays):
        item = _as_object(raw, f"array_inventory[{index}]")
        _require_exact_keys(
            item,
            {"clock_dimensions", "dimension_names", "name", "role", "shape"},
            f"array_inventory[{index}]",
        )
        name = _as_nonempty_string(item["name"], f"array_inventory[{index}].name")
        if name in arrays or (arrays and name <= next(reversed(arrays))):
            raise MastMagneticDiagnosticQualificationError(
                "array inventory is not unique and sorted"
            )
        if item["role"] not in allowed_roles:
            raise MastMagneticDiagnosticQualificationError(f"array {name} has unsupported role")
        shape = _as_nonnegative_integer_list(item["shape"], f"array {name} shape")
        dimensions = _as_string_list(item["dimension_names"], f"array {name} dimensions")
        clock_dimensions = _as_string_list(item["clock_dimensions"], f"array {name} clocks")
        if len(shape) != len(dimensions) or not set(clock_dimensions).issubset(dimensions):
            raise MastMagneticDiagnosticQualificationError(
                f"array {name} shape or clock dimensions differ"
            )
        arrays[name] = item
    return arrays


def _validate_clocks(
    raw_clocks: list[JsonValue], arrays: Mapping[str, JsonObject]
) -> dict[str, JsonObject]:
    if len(raw_clocks) != 4:
        raise MastMagneticDiagnosticQualificationError("clock evidence must contain four grids")
    clocks: dict[str, JsonObject] = {}
    for index, raw in enumerate(raw_clocks):
        item = _as_object(raw, f"clock_evidence[{index}]")
        _require_exact_keys(
            item,
            {
                "archive_grid_reproduced",
                "dropna",
                "first_value_s",
                "grid_origin",
                "interpolation_method",
                "last_value_s",
                "name",
                "sample_count",
                "source_clock_relation_claimed",
                "start_s",
                "step_s",
            },
            f"clock_evidence[{index}]",
        )
        name = _as_nonempty_string(item["name"], f"clock_evidence[{index}].name")
        if name in clocks or (clocks and name <= next(reversed(clocks))):
            raise MastMagneticDiagnosticQualificationError(
                "clock evidence is not unique and sorted"
            )
        if name not in arrays or arrays[name]["role"] != "clock":
            raise MastMagneticDiagnosticQualificationError(f"clock {name} is not source-bound")
        _require_equal(item["grid_origin"], "level2_interpolation", f"clock {name} origin")
        _require_equal(item["interpolation_method"], "zero", f"clock {name} interpolation")
        _require_equal(item["dropna"], True, f"clock {name} dropna")
        _require_equal(item["archive_grid_reproduced"], True, f"clock {name} reproduction")
        _require_equal(
            item["source_clock_relation_claimed"], False, f"clock {name} source relation"
        )
        sample_count = _as_positive_integer(item["sample_count"], f"clock {name} samples")
        shape = _as_nonnegative_integer_list(arrays[name]["shape"], f"clock {name} shape")
        if shape != [sample_count]:
            raise MastMagneticDiagnosticQualificationError(f"clock {name} sample count mismatch")
        for field in ("first_value_s", "last_value_s", "start_s", "step_s"):
            _as_finite_number(item[field], f"clock {name} {field}")
        start = cast(float | int, item["start_s"])
        step = cast(float | int, item["step_s"])
        first = cast(float | int, item["first_value_s"])
        last = cast(float | int, item["last_value_s"])
        if step <= 0:
            raise MastMagneticDiagnosticQualificationError(f"clock {name} step is not positive")
        tolerance = max(1e-15, float(step) * 1e-9)
        expected_last = float(start) + (sample_count - 1) * float(step)
        if not math.isclose(
            float(first), float(start), rel_tol=0.0, abs_tol=tolerance
        ) or not math.isclose(float(last), expected_last, rel_tol=0.0, abs_tol=tolerance):
            raise MastMagneticDiagnosticQualificationError(
                f"clock {name} bounds do not reproduce its grid"
            )
        clocks[name] = item
    return clocks


def _validate_measurements(
    raw_measurements: list[JsonValue],
    arrays: Mapping[str, JsonObject],
    clocks: Mapping[str, JsonObject],
    shot_id: int,
) -> dict[str, JsonObject]:
    measurements: dict[str, JsonObject] = {}
    for index, raw in enumerate(raw_measurements):
        item = _as_object(raw, f"measurement_evidence[{index}]")
        _require_exact_keys(
            item,
            {
                "applied_background_sample_range",
                "applied_scale",
                "archive_channel_ids",
                "array_name",
                "calibration_lineage_state",
                "channel_quality",
                "clock_name",
                "configured_source_channels",
                "empirical_quality",
                "imas_quantity_path",
                "observation_operator_state",
                "provider_quality_flags_supplied",
                "source_name",
                "source_shot_max",
                "source_shot_min",
                "source_valid_for_shot",
                "target_units",
                "uncertainty_supplied",
                "units",
            },
            f"measurement_evidence[{index}]",
        )
        name = _as_nonempty_string(item["array_name"], f"measurement_evidence[{index}].name")
        if name in measurements or (measurements and name <= next(reversed(measurements))):
            raise MastMagneticDiagnosticQualificationError(
                "measurement evidence is not unique and sorted"
            )
        if name not in arrays or arrays[name]["role"] != "measurement":
            raise MastMagneticDiagnosticQualificationError(
                f"measurement {name} is not source-bound"
            )
        clock_name = _as_nonempty_string(item["clock_name"], f"measurement {name} clock")
        if clock_name not in clocks:
            raise MastMagneticDiagnosticQualificationError(f"measurement {name} clock is unknown")
        archive_channels = _as_string_list(item["archive_channel_ids"], f"{name} channels")
        configured_channels = _as_string_list(
            item["configured_source_channels"], f"{name} source channels"
        )
        if len(archive_channels) != len(configured_channels):
            raise MastMagneticDiagnosticQualificationError(f"measurement {name} channel mismatch")
        if len(set(archive_channels)) != len(archive_channels) or len(
            set(configured_channels)
        ) != len(configured_channels):
            raise MastMagneticDiagnosticQualificationError(
                f"measurement {name} channels are not unique"
            )
        channel_quality = _as_list(item["channel_quality"], f"{name} channel quality")
        if len(channel_quality) != len(archive_channels):
            raise MastMagneticDiagnosticQualificationError(
                f"measurement {name} channel quality count mismatch"
            )
        for channel_index, (raw_quality, channel_id) in enumerate(
            zip(channel_quality, archive_channels, strict=True)
        ):
            record = _as_object(raw_quality, f"{name} channel_quality[{channel_index}]")
            _require_exact_keys(
                record,
                {"archive_channel_id", "quality"},
                f"{name} channel_quality[{channel_index}]",
            )
            _require_equal(record["archive_channel_id"], channel_id, f"{name} channel id")
            _validate_quality(
                _as_object(record["quality"], f"{name} channel {channel_id} quality"),
                f"{name}/{channel_id}",
            )
        _require_equal(item["source_valid_for_shot"], True, f"measurement {name} validity")
        for field in ("source_shot_min", "source_shot_max"):
            value = item[field]
            if value is not None:
                _as_positive_integer(value, f"measurement {name} {field}")
        minimum = cast(int | None, item["source_shot_min"])
        maximum = cast(int | None, item["source_shot_max"])
        if minimum is not None and maximum is not None and minimum > maximum:
            raise MastMagneticDiagnosticQualificationError(
                f"measurement {name} source shot range is invalid"
            )
        if (minimum is not None and shot_id < minimum) or (
            maximum is not None and shot_id > maximum
        ):
            raise MastMagneticDiagnosticQualificationError(
                f"measurement {name} shot range mismatch"
            )
        if _as_finite_number(item["applied_scale"], f"measurement {name} scale") <= 0:
            raise MastMagneticDiagnosticQualificationError(
                f"measurement {name} scale is not positive"
            )
        background = item["applied_background_sample_range"]
        if background is not None:
            bounds = _as_nonnegative_integer_list(background, f"measurement {name} background")
            if len(bounds) != 2 or bounds[1] <= bounds[0]:
                raise MastMagneticDiagnosticQualificationError(
                    f"measurement {name} background range is invalid"
                )
        _require_equal(
            item["calibration_lineage_state"],
            "not_supplied",
            f"measurement {name} calibration",
        )
        _require_equal(
            item["observation_operator_state"],
            "imas_quantity_path_only_transfer_function_not_supplied",
            f"measurement {name} operator",
        )
        _require_equal(item["provider_quality_flags_supplied"], False, f"{name} quality flags")
        _require_equal(item["uncertainty_supplied"], False, f"{name} uncertainty")
        _as_nonempty_string(item["source_name"], f"measurement {name} source_name")
        _as_nonempty_string(item["imas_quantity_path"], f"measurement {name} IMAS path")
        for field in ("target_units", "units"):
            if item[field] is not None:
                _as_nonempty_string(item[field], f"measurement {name} {field}")
        aggregate_quality = _as_object(item["empirical_quality"], f"{name} quality")
        _validate_quality(aggregate_quality, name)
        if channel_quality:
            for field in (
                "finite_count",
                "infinite_count",
                "nan_count",
                "sample_count",
                "zero_count",
            ):
                channel_total = sum(
                    cast(
                        int,
                        _as_object(
                            _as_object(record, "channel quality")["quality"],
                            "quality record",
                        )[field],
                    )
                    for record in channel_quality
                )
                if aggregate_quality[field] != channel_total:
                    raise MastMagneticDiagnosticQualificationError(
                        f"measurement {name} aggregate {field} differs from channels"
                    )
        measurements[name] = item
    if set(measurements) != _MEASUREMENT_NAMES:
        raise MastMagneticDiagnosticQualificationError(
            "measurement evidence does not cover the complete magnetic group"
        )
    return measurements


def _validate_quality(quality: JsonObject, name: str) -> None:
    _require_exact_keys(
        quality,
        {
            "finite_count",
            "infinite_count",
            "minimum_positive_level_spacing_hex",
            "nan_count",
            "nan_fraction",
            "sample_count",
            "unique_finite_value_count",
            "zero_count",
        },
        f"measurement {name} quality",
    )
    sample_count = _as_positive_integer(quality["sample_count"], f"{name} sample_count")
    finite = _as_nonnegative_integer(quality["finite_count"], f"{name} finite_count")
    nan = _as_nonnegative_integer(quality["nan_count"], f"{name} nan_count")
    infinite = _as_nonnegative_integer(quality["infinite_count"], f"{name} infinite_count")
    if finite + nan + infinite != sample_count:
        raise MastMagneticDiagnosticQualificationError(f"measurement {name} quality count mismatch")
    fraction = _as_finite_number(quality["nan_fraction"], f"{name} nan_fraction")
    if not 0.0 <= fraction <= 1.0 or not math.isclose(fraction, nan / sample_count, abs_tol=1e-15):
        raise MastMagneticDiagnosticQualificationError(f"measurement {name} NaN fraction mismatch")
    unique = _as_nonnegative_integer(quality["unique_finite_value_count"], f"{name} unique values")
    zero = _as_nonnegative_integer(quality["zero_count"], f"{name} zero count")
    if unique > finite or zero > finite:
        raise MastMagneticDiagnosticQualificationError(f"measurement {name} quality bounds fail")
    spacing = quality["minimum_positive_level_spacing_hex"]
    if spacing is not None:
        value = _as_nonempty_string(spacing, f"{name} level spacing")
        try:
            parsed = float.fromhex(value)
        except ValueError as exc:
            raise MastMagneticDiagnosticQualificationError(
                f"measurement {name} level spacing is invalid"
            ) from exc
        if not math.isfinite(parsed) or parsed <= 0:
            raise MastMagneticDiagnosticQualificationError(
                f"measurement {name} level spacing is not positive"
            )


def _validate_channel_mappings(
    raw_mappings: list[JsonValue],
    measurements: Mapping[str, JsonObject],
    arrays: Mapping[str, JsonObject],
) -> list[JsonObject]:
    mappings: list[JsonObject] = []
    previous: tuple[str, str] | None = None
    for index, raw in enumerate(raw_mappings):
        item = _as_object(raw, f"channel_geometry_evidence[{index}]")
        _require_exact_keys(
            item,
            {
                "archive_channel_id",
                "geometry_channel_id",
                "geometry_coordinate",
                "identifier_match_method",
                "measurement_array",
                "physical_mapping_claimed",
            },
            f"channel_geometry_evidence[{index}]",
        )
        measurement = _as_nonempty_string(item["measurement_array"], "mapping measurement")
        channel = _as_nonempty_string(item["archive_channel_id"], "mapping channel")
        key = (measurement, channel)
        if measurement not in measurements or previous is not None and key <= previous:
            raise MastMagneticDiagnosticQualificationError("channel mappings are not source-bound")
        geometry = item["geometry_coordinate"]
        geometry_channel = item["geometry_channel_id"]
        method = item["identifier_match_method"]
        if geometry is None:
            _require_equal(geometry_channel, None, "unavailable geometry channel")
            _require_equal(method, "unavailable_in_archive", "unavailable geometry method")
        else:
            geometry_name = _as_nonempty_string(geometry, "geometry coordinate")
            if geometry_name not in arrays or arrays[geometry_name]["role"] != "channel_coordinate":
                raise MastMagneticDiagnosticQualificationError("geometry coordinate is not bound")
            _as_nonempty_string(geometry_channel, "geometry channel")
            if method not in {"casefold_exact", "prefix_normalised", "numeric_suffix_normalised"}:
                raise MastMagneticDiagnosticQualificationError("mapping method is unsupported")
        _require_equal(item["physical_mapping_claimed"], False, "physical mapping authority")
        mappings.append(item)
        previous = key
    if not mappings:
        raise MastMagneticDiagnosticQualificationError("channel geometry evidence is empty")
    expected = {
        (name, channel)
        for name, measurement in measurements.items()
        for channel in _as_string_list(measurement["archive_channel_ids"], f"{name} channels")
    }
    actual = {
        (
            cast(str, mapping["measurement_array"]),
            cast(str, mapping["archive_channel_id"]),
        )
        for mapping in mappings
    }
    if actual != expected:
        raise MastMagneticDiagnosticQualificationError(
            "channel geometry evidence does not cover every measurement channel"
        )
    return mappings


def _validate_event_identity(event: JsonObject, shot_id: int) -> None:
    _require_exact_keys(
        event,
        {"event_id", "event_time_epoch", "shot_id", "state"},
        "event_identity",
    )
    _require_equal(event["shot_id"], shot_id, "event_identity.shot_id")
    _require_equal(event["event_id"], None, "event_identity.event_id")
    _require_equal(event["event_time_epoch"], None, "event_identity.event_time_epoch")
    _require_equal(event["state"], "shot_only_event_unresolved", "event_identity.state")


def _validate_external_limitations(raw_limitations: list[JsonValue]) -> None:
    if len(raw_limitations) != 1:
        raise MastMagneticDiagnosticQualificationError("one scoped external limitation is required")
    item = _as_object(raw_limitations[0], "external_limitations[0]")
    _require_exact_keys(
        item,
        {"applicability_to_shot", "issue", "reported_shot_id", "scope", "url"},
        "external limitation",
    )
    _require_equal(item["issue"], 211, "external limitation issue")
    _require_equal(item["reported_shot_id"], 29980, "external limitation shot")
    _require_equal(item["applicability_to_shot"], "not_assumed", "external applicability")
    _require_equal(item["scope"], "reported_level2_numeric_resolution_and_saddle_nan", "scope")
    _require_equal(
        item["url"],
        "https://github.com/ukaea/fair-mast/issues/211",
        "external limitation URL",
    )


def _validate_completeness(
    completeness: JsonObject,
    arrays: Mapping[str, JsonObject],
    clocks: Mapping[str, JsonObject],
    measurements: Mapping[str, JsonObject],
    mappings: list[JsonObject],
) -> None:
    _require_exact_keys(
        completeness,
        {
            "archive_array_count",
            "archive_arrays_classified",
            "channel_record_count",
            "clock_count",
            "measurement_count",
            "measurements_analysed",
        },
        "completeness",
    )
    _require_equal(completeness["archive_array_count"], len(arrays), "archive_array_count")
    _require_equal(completeness["archive_arrays_classified"], True, "arrays classified")
    _require_equal(completeness["clock_count"], len(clocks), "clock_count")
    _require_equal(completeness["measurement_count"], len(measurements), "measurement_count")
    _require_equal(completeness["measurements_analysed"], True, "measurements analysed")
    _require_equal(completeness["channel_record_count"], len(mappings), "channel_record_count")


def _canonical_json_bytes(value: JsonValue) -> bytes:
    _reject_nonfinite_numbers(value, "document")
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - JsonValue is prevalidated
        raise MastMagneticDiagnosticQualificationError(f"JSON encoding failed: {exc}") from exc
    return (encoded + "\n").encode()


def _parse_json_object(data: bytes) -> JsonObject:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MastMagneticDiagnosticQualificationError(
            "qualification document is not valid UTF-8"
        ) from exc
    try:
        decoded = json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
    except (json.JSONDecodeError, MastMagneticDiagnosticQualificationError) as exc:
        raise MastMagneticDiagnosticQualificationError("qualification JSON is invalid") from exc
    return _as_object(cast(JsonValue, decoded), "document")


def _reject_duplicate_pairs(pairs: list[tuple[str, JsonValue]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise MastMagneticDiagnosticQualificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_numbers(value: JsonValue, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):  # pragma: no cover - fields validate
        raise MastMagneticDiagnosticQualificationError(f"{path} contains a nonfinite number")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite_numbers(item, f"{path}[{index}]")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite_numbers(item, f"{path}.{key}")


def _as_object(value: JsonValue, label: str) -> JsonObject:
    if not isinstance(value, dict):
        raise MastMagneticDiagnosticQualificationError(f"{label} is not an object")
    return value


def _as_list(value: JsonValue, label: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise MastMagneticDiagnosticQualificationError(f"{label} is not an array")
    return value


def _as_nonempty_string(value: JsonValue, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise MastMagneticDiagnosticQualificationError(f"{label} is not a nonempty string")
    return value


def _as_string_list(value: JsonValue, label: str) -> list[str]:
    raw = _as_list(value, label)
    if not all(isinstance(item, str) and item for item in raw):
        raise MastMagneticDiagnosticQualificationError(f"{label} is not a string array")
    return cast(list[str], raw)


def _as_positive_integer(value: JsonValue, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MastMagneticDiagnosticQualificationError(f"{label} is not a positive integer")
    return value


def _as_nonnegative_integer(value: JsonValue, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MastMagneticDiagnosticQualificationError(f"{label} is not a nonnegative integer")
    return value


def _as_nonnegative_integer_list(value: JsonValue, label: str) -> list[int]:
    raw = _as_list(value, label)
    return [_as_nonnegative_integer(item, f"{label}[{index}]") for index, item in enumerate(raw)]


def _as_finite_number(value: JsonValue, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise MastMagneticDiagnosticQualificationError(f"{label} is not a finite number")
    return float(value)


def _require_exact_keys(value: JsonObject, expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise MastMagneticDiagnosticQualificationError(f"{label} keys differ from contract")


def _require_equal(actual: JsonValue, expected: JsonValue, label: str) -> None:
    if actual != expected:
        raise MastMagneticDiagnosticQualificationError(f"{label} differs from contract")


def _require_matching_string(value: JsonValue, pattern: re.Pattern[str], label: str) -> None:
    text = _as_nonempty_string(value, label)
    if pattern.fullmatch(text) is None:
        raise MastMagneticDiagnosticQualificationError(f"{label} has invalid syntax")


def _require_sha256(value: JsonValue, label: str) -> None:
    _require_matching_string(value, _SHA256_RE, label)


def _require_https_url(value: JsonValue, label: str) -> None:
    if not _as_nonempty_string(value, label).startswith("https://"):
        raise MastMagneticDiagnosticQualificationError(f"{label} is not HTTPS")
