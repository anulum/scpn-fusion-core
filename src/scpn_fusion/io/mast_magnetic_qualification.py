# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FAIR-MAST magnetic diagnostic qualification
"""Derive source-bound qualification evidence without inventing diagnostic authority."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .mast_magnetic_archive import MastMagneticArchiveDependencyError
from .mast_magnetic_archive_codec import (
    JsonObject,
    JsonValue,
    MastCompleteMagneticArchiveEnvelope,
    decode_mast_complete_magnetic_archive_envelope,
)
from .mast_magnetic_qualification_codec import (
    MastMagneticDiagnosticQualification,
    MastMagneticDiagnosticQualificationError,
    encode_mast_magnetic_diagnostic_qualification,
)

_MAPPING_PATH = "mappings/level2/mast.yml"
_MAPPING_URL = (
    "https://raw.githubusercontent.com/ukaea/fair-mast-ingestion/{revision}/" + _MAPPING_PATH
)
_ISSUE_URL = "https://github.com/ukaea/fair-mast/issues/211"

_GEOMETRY_BINDINGS: dict[str, tuple[str | None, str]] = {
    "b_field_pol_probe_cc_field": ("b_field_pol_probe_cc_geometry_channel", "CCMV"),
    "b_field_pol_probe_ccbv_field": ("b_field_pol_probe_ccbv_geometry_channel", ""),
    "b_field_pol_probe_obr_field": ("b_field_pol_probe_obr_geometry_channel", ""),
    "b_field_pol_probe_obv_field": ("b_field_pol_probe_obv_geometry_channel", ""),
    "b_field_pol_probe_omv_voltage": ("b_field_pol_probe_omv_geometry_channel", "OMV"),
    "b_field_tor_probe_cc_field": ("b_field_tor_probe_cc_geometry_channel", "CCMT"),
    "b_field_tor_probe_omaha_voltage": (None, ""),
    "b_field_tor_probe_saddle_field": (
        "b_field_tor_probe_saddle_m_geometry_channel",
        "SADOUT",
    ),
    "b_field_tor_probe_saddle_voltage": (
        "b_field_tor_probe_saddle_m_geometry_channel",
        "SADOUT",
    ),
    "flux_loop_flux": ("flux_loop_geometry_channel", "FL"),
    "ip": (None, ""),
}


def build_mast_magnetic_diagnostic_qualification(
    archive_envelope: MastCompleteMagneticArchiveEnvelope | bytes,
    shot_archive_root: Path,
    ingestion_mapping_path: Path,
) -> MastMagneticDiagnosticQualification:
    """Build diagnostic qualification evidence for one complete FAIR-MAST archive.

    Parameters
    ----------
    archive_envelope:
        Validated complete magnetic archive envelope or its canonical bytes.
    shot_archive_root:
        Fully materialised ``<shot>.zarr`` root used to measure data quality.
    ingestion_mapping_path:
        Exact ``mappings/level2/mast.yml`` from the envelope's source revision.

    Returns
    -------
    MastMagneticDiagnosticQualification
        Canonical review-only evidence binding every archive array and measurement.

    Raises
    ------
    MastMagneticDiagnosticQualificationError
        If mapping, channel, geometry, clock or evidence completeness checks fail.
    MastMagneticArchiveDependencyError
        If the Zarr-v3 qualification dependency profile is unavailable.
    """
    envelope = (
        decode_mast_complete_magnetic_archive_envelope(archive_envelope)
        if isinstance(archive_envelope, bytes)
        else decode_mast_complete_magnetic_archive_envelope(archive_envelope.to_bytes())
    )
    archive_payload = envelope.payload
    shot_id = _positive_integer(archive_payload["shot_id"], "shot_id")
    revision = _nonempty_string(archive_payload["source_ingestion_revision"], "revision")
    tree_state = _nonempty_string(archive_payload["source_ingestion_tree_state"], "tree state")
    mapping_bytes = _read_regular_file(ingestion_mapping_path, "ingestion mapping")
    mapping = _parse_mapping(mapping_bytes)
    _require_equal(mapping.get("facility"), "MAST", "mapping facility")
    dataset = _object(_object(mapping.get("datasets"), "datasets").get("magnetics"), "magnetics")
    profiles = _object(dataset.get("profiles"), "magnetics.profiles")
    interpolation = _object(dataset.get("interpolate"), "magnetics.interpolate")
    licence = _object(mapping.get("license"), "mapping.license")

    zarr_group = _open_zarr_group(shot_archive_root / "magnetics")
    array_inventory = _build_array_inventory(archive_payload, profiles)
    clock_evidence = _build_clock_evidence(archive_payload, interpolation, zarr_group)
    measurement_evidence = _build_measurement_evidence(
        archive_payload,
        profiles,
        zarr_group,
        shot_id,
    )
    channel_geometry_evidence = _build_channel_geometry_evidence(
        measurement_evidence,
        zarr_group,
    )
    payload: JsonObject = {
        "archive_envelope_sha256": envelope.sha256,
        "archive_observation_id": archive_payload["observation_id"],
        "array_inventory": cast(JsonValue, array_inventory),
        "authority": {
            "actionable": False,
            "classification_performed": False,
            "direct_actuation": False,
            "execution_permitted": False,
            "phase_inference_performed": False,
            "review_only": True,
        },
        "channel_geometry_evidence": cast(JsonValue, channel_geometry_evidence),
        "clock_evidence": cast(JsonValue, clock_evidence),
        "completeness": {
            "archive_array_count": len(array_inventory),
            "archive_arrays_classified": True,
            "channel_record_count": len(channel_geometry_evidence),
            "clock_count": len(clock_evidence),
            "measurement_count": len(measurement_evidence),
            "measurements_analysed": True,
        },
        "event_identity": {
            "event_id": None,
            "event_time_epoch": None,
            "shot_id": shot_id,
            "state": "shot_only_event_unresolved",
        },
        "external_limitations": [
            {
                "applicability_to_shot": "not_assumed",
                "issue": 211,
                "reported_shot_id": 29980,
                "scope": "reported_level2_numeric_resolution_and_saddle_nan",
                "url": _ISSUE_URL,
            }
        ],
        "facility": "MAST",
        "ingestion_mapping": {
            "dataset_license_name": licence.get("name"),
            "dataset_license_url": licence.get("url"),
            "mapping_path": _MAPPING_PATH,
            "mapping_sha256": hashlib.sha256(mapping_bytes).hexdigest(),
            "mapping_url": _MAPPING_URL.format(revision=revision),
            "source_revision": revision,
            "source_tree_state": tree_state,
        },
        "measurement_evidence": cast(JsonValue, measurement_evidence),
        "producer_project": "SCPN-FUSION-CORE",
        "qualification_summary": {
            "calibration_state": "applied_transforms_recorded_lineage_unavailable",
            "channel_geometry_mapping_state": "identifier_correspondence_only",
            "event_identity_state": "shot_only_event_unresolved",
            "observation_operator_state": ("quantity_paths_only_transfer_functions_unavailable"),
            "provider_quality_state": "not_supplied",
            "source_clock_relationship_state": (
                "derived_archive_grids_no_instrument_clock_relation"
            ),
            "uncertainty_state": "not_supplied",
            "validity_state": "source_shot_ranges_only",
        },
        "reactor_configuration": "spherical_tokamak",
        "shot_id": shot_id,
        "source_archive": "FAIR-MAST",
    }
    return encode_mast_magnetic_diagnostic_qualification(payload)


def verify_mast_magnetic_diagnostic_qualification(
    expected: MastMagneticDiagnosticQualification | bytes,
    archive_envelope: MastCompleteMagneticArchiveEnvelope | bytes,
    shot_archive_root: Path,
    ingestion_mapping_path: Path,
) -> None:
    """Rebuild qualification evidence and require byte-identical canonical output."""
    expected_bytes = expected if isinstance(expected, bytes) else expected.to_bytes()
    actual = build_mast_magnetic_diagnostic_qualification(
        archive_envelope,
        shot_archive_root,
        ingestion_mapping_path,
    )
    if actual.to_bytes() != expected_bytes:
        raise MastMagneticDiagnosticQualificationError(
            "source data and mapping do not reproduce tracked qualification evidence"
        )


def _build_array_inventory(
    archive_payload: JsonObject,
    profiles: JsonObject,
) -> list[JsonObject]:
    clocks = {
        _nonempty_string(_object(item, "clock")["name"], "clock name")
        for item in _array(archive_payload["clocks"], "clocks")
    }
    inventory: list[JsonObject] = []
    for raw in _array(archive_payload["arrays"], "arrays"):
        item = _object(raw, "archive array")
        name = _nonempty_string(item["name"], "archive array name")
        profile = profiles.get(name)
        if name in clocks:
            role = "clock"
        elif name == "shot_id":
            role = "shot_identity"
        elif name.endswith("_channel") or name == "coordinate":
            role = "channel_coordinate"
        elif isinstance(profile, dict) and profile.get("geometry") is not None:
            role = "geometry"
        elif isinstance(profile, dict):
            source = profile.get("source")
            if source is not None and source != "":
                role = "measurement"
            else:
                raise MastMagneticDiagnosticQualificationError(
                    f"archive array {name} has no qualification role"
                )
        else:
            raise MastMagneticDiagnosticQualificationError(
                f"archive array {name} has no qualification role"
            )
        inventory.append(
            {
                "clock_dimensions": item["clock_dimensions"],
                "dimension_names": item["dimension_names"],
                "name": name,
                "role": role,
                "shape": item["shape"],
            }
        )
    return sorted(inventory, key=lambda record: cast(str, record["name"]))


def _build_clock_evidence(
    archive_payload: JsonObject,
    interpolation: JsonObject,
    zarr_group: Any,
) -> list[JsonObject]:
    result: list[JsonObject] = []
    for raw in _array(archive_payload["clocks"], "clocks"):
        archive_clock = _object(raw, "archive clock")
        name = _nonempty_string(archive_clock["name"], "clock name")
        settings = _object(interpolation.get(name), f"interpolation.{name}")
        start = _finite_number(settings.get("start"), f"{name} start")
        step = _finite_number(settings.get("step"), f"{name} step")
        values = np.asarray(cast(Any, zarr_group)[name].values, dtype=np.float64)
        expected = start + np.arange(values.size, dtype=np.float64) * step
        tolerance = max(1e-15, step * 1e-9)
        if values.ndim != 1 or not np.allclose(values, expected, rtol=0.0, atol=tolerance):
            raise MastMagneticDiagnosticQualificationError(
                f"clock {name} does not reproduce its Level-2 interpolation grid"
            )
        result.append(
            {
                "archive_grid_reproduced": True,
                "dropna": settings.get("dropna"),
                "first_value_s": float(values[0]),
                "grid_origin": "level2_interpolation",
                "interpolation_method": settings.get("method"),
                "last_value_s": float(values[-1]),
                "name": name,
                "sample_count": int(values.size),
                "source_clock_relation_claimed": False,
                "start_s": start,
                "step_s": step,
            }
        )
    return sorted(result, key=lambda record: cast(str, record["name"]))


def _build_measurement_evidence(
    archive_payload: JsonObject,
    profiles: JsonObject,
    zarr_group: Any,
    shot_id: int,
) -> list[JsonObject]:
    archive_arrays = {
        _nonempty_string(_object(item, "archive array")["name"], "array name"): _object(
            item, "archive array"
        )
        for item in _array(archive_payload["arrays"], "arrays")
    }
    result: list[JsonObject] = []
    for name in sorted(_GEOMETRY_BINDINGS):
        profile = _object(profiles.get(name), f"profile {name}")
        source = _select_source(profile.get("source"), shot_id, name)
        configured_channels = _string_sequence(source.get("channels", []), f"{name} channels")
        data_array = cast(Any, zarr_group)[name]
        archive_channels, channel_axis = _archive_channels(data_array)
        if len(configured_channels) != len(archive_channels):
            raise MastMagneticDiagnosticQualificationError(
                f"{name} configured and archived channel counts differ"
            )
        for configured, archived in zip(configured_channels, archive_channels, strict=True):
            if not _configured_channel_matches(configured, archived):
                raise MastMagneticDiagnosticQualificationError(
                    f"{name} channel {archived} does not match source {configured}"
                )
        dimensions = _string_sequence(archive_arrays[name]["dimension_names"], f"{name} dims")
        clock_names = [dimension for dimension in dimensions if dimension.startswith("time")]
        if len(clock_names) != 1:
            raise MastMagneticDiagnosticQualificationError(f"{name} has no unique archive clock")
        values = np.asarray(data_array.values)
        channel_quality = [
            {
                "archive_channel_id": channel_id,
                "quality": _quality_record(np.take(values, index, axis=cast(int, channel_axis))),
            }
            for index, channel_id in enumerate(archive_channels)
        ]
        source_range = _source_range(source)
        background = source.get("background_correction")
        background_range: JsonValue = None
        if background is not None:
            background_object = _object(cast(JsonValue, background), f"{name} background")
            background_range = [
                _nonnegative_integer(background_object.get("tmin"), f"{name} background start"),
                _nonnegative_integer(background_object.get("tmax"), f"{name} background end"),
            ]
        actual_units = cast(Any, data_array).attrs.get("units")
        result.append(
            {
                "applied_background_sample_range": background_range,
                "applied_scale": _finite_number(profile.get("scale", 1.0), f"{name} scale"),
                "archive_channel_ids": cast(JsonValue, archive_channels),
                "array_name": name,
                "calibration_lineage_state": "not_supplied",
                "channel_quality": cast(JsonValue, channel_quality),
                "clock_name": clock_names[0],
                "configured_source_channels": cast(JsonValue, configured_channels),
                "empirical_quality": _quality_record(values),
                "imas_quantity_path": profile.get("imas"),
                "observation_operator_state": (
                    "imas_quantity_path_only_transfer_function_not_supplied"
                ),
                "provider_quality_flags_supplied": False,
                "source_name": source.get("name"),
                "source_shot_max": source_range[1],
                "source_shot_min": source_range[0],
                "source_valid_for_shot": True,
                "target_units": profile.get("target_units"),
                "uncertainty_supplied": False,
                "units": actual_units,
            }
        )
    return result


def _build_channel_geometry_evidence(
    measurements: Sequence[JsonObject],
    zarr_group: Any,
) -> list[JsonObject]:
    records: list[JsonObject] = []
    for measurement in measurements:
        name = _nonempty_string(measurement["array_name"], "measurement name")
        geometry_name, geometry_prefix = _GEOMETRY_BINDINGS[name]
        channels = _string_sequence(measurement["archive_channel_ids"], f"{name} channels")
        if not channels:
            continue
        if geometry_name is None:
            records.extend(
                {
                    "archive_channel_id": channel,
                    "geometry_channel_id": None,
                    "geometry_coordinate": None,
                    "identifier_match_method": "unavailable_in_archive",
                    "measurement_array": name,
                    "physical_mapping_claimed": False,
                }
                for channel in channels
            )
            continue
        geometry_channels = [
            str(value) for value in np.asarray(cast(Any, zarr_group)[geometry_name].values)
        ]
        for channel in channels:
            matches = [
                (geometry_channel, method)
                for geometry_channel in geometry_channels
                if (method := _identifier_match_method(channel, geometry_channel, geometry_prefix))
                is not None
            ]
            if len(matches) != 1:
                raise MastMagneticDiagnosticQualificationError(
                    f"{name} channel {channel} has {len(matches)} geometry identifier matches"
                )
            geometry_channel, method = matches[0]
            records.append(
                {
                    "archive_channel_id": channel,
                    "geometry_channel_id": geometry_channel,
                    "geometry_coordinate": geometry_name,
                    "identifier_match_method": method,
                    "measurement_array": name,
                    "physical_mapping_claimed": False,
                }
            )
    return sorted(
        records,
        key=lambda record: (
            cast(str, record["measurement_array"]),
            cast(str, record["archive_channel_id"]),
        ),
    )


def _quality_record(values: NDArray[Any]) -> JsonObject:
    numeric = np.asarray(values)
    sample_count = int(numeric.size)
    finite_mask = np.isfinite(numeric)
    finite_values = np.asarray(numeric[finite_mask], dtype=np.float64)
    nan_count = int(np.isnan(numeric).sum())
    infinite_count = int(np.isinf(numeric).sum())
    unique = np.unique(finite_values)
    positive_spacing = np.diff(unique)
    positive_spacing = positive_spacing[positive_spacing > 0]
    minimum_spacing: JsonValue = (
        None if positive_spacing.size == 0 else float(positive_spacing.min()).hex()
    )
    return {
        "finite_count": int(finite_values.size),
        "infinite_count": infinite_count,
        "minimum_positive_level_spacing_hex": minimum_spacing,
        "nan_count": nan_count,
        "nan_fraction": nan_count / sample_count,
        "sample_count": sample_count,
        "unique_finite_value_count": int(unique.size),
        "zero_count": int(np.count_nonzero(finite_values == 0.0)),
    }


def _archive_channels(data_array: Any) -> tuple[list[str], int | None]:
    dimensions = [str(dimension) for dimension in data_array.dims]
    channel_dimensions = [dimension for dimension in dimensions if dimension.endswith("_channel")]
    if not channel_dimensions:
        return [], None
    if len(channel_dimensions) != 1:
        raise MastMagneticDiagnosticQualificationError(
            f"measurement {data_array.name} has multiple channel dimensions"
        )
    dimension = channel_dimensions[0]
    values = np.asarray(data_array.coords[dimension].values)
    return [str(value) for value in values], dimensions.index(dimension)


def _select_source(raw_source: JsonValue | None, shot_id: int, name: str) -> JsonObject:
    if isinstance(raw_source, str):
        if not raw_source:
            raise MastMagneticDiagnosticQualificationError(f"{name} source is empty")
        return {"name": raw_source}
    candidates = _array(raw_source, f"{name} sources")
    selected: list[JsonObject] = []
    for raw_candidate in candidates:
        candidate = _object(raw_candidate, f"{name} source")
        minimum, maximum = _source_range(candidate)
        if (minimum is None or shot_id >= minimum) and (maximum is None or shot_id <= maximum):
            selected.append(candidate)
    if len(selected) != 1:
        raise MastMagneticDiagnosticQualificationError(
            f"{name} has {len(selected)} source mappings for shot {shot_id}"
        )
    return selected[0]


def _source_range(source: JsonObject) -> tuple[int | None, int | None]:
    raw_range = source.get("shot_range")
    if raw_range is None:
        return None, None
    shot_range = _object(raw_range, "shot range")
    return (
        _positive_integer(shot_range.get("shot_min"), "shot_min"),
        _positive_integer(shot_range.get("shot_max"), "shot_max"),
    )


def _configured_channel_matches(configured: str, archived: str) -> bool:
    return _alphanumeric(configured).endswith(_alphanumeric(archived))


def _identifier_match_method(signal: str, geometry: str, prefix: str) -> str | None:
    signal_raw = _alphanumeric(signal)
    geometry_raw = _alphanumeric(geometry)
    if signal_raw == geometry_raw:
        return "casefold_exact"
    prefix_raw = _alphanumeric(prefix)
    stripped = geometry_raw.removeprefix(prefix_raw)
    if stripped == signal_raw:
        return "prefix_normalised"
    if _normalise_numeric_suffix(stripped) == _normalise_numeric_suffix(signal_raw):
        return "numeric_suffix_normalised"
    return None


def _alphanumeric(value: str) -> str:
    return "".join(character for character in value.upper() if character.isalnum())


def _normalise_numeric_suffix(value: str) -> str:
    match = re.fullmatch(r"([A-Z]*)([0-9]+)", value)
    if match is None:
        return value
    return f"{match.group(1)}{int(match.group(2))}"


def _open_zarr_group(path: Path) -> Any:
    try:
        import xarray as xr
    except ImportError as exc:
        raise MastMagneticArchiveDependencyError(
            "install the hash-locked mast optional dependency profile"
        ) from exc
    try:
        return xr.open_zarr(path, consolidated=False, zarr_format=3)
    except (OSError, TypeError, ValueError) as exc:
        raise MastMagneticDiagnosticQualificationError(
            "magnetic Zarr group cannot be opened"
        ) from exc


def _parse_mapping(data: bytes) -> JsonObject:
    try:
        import yaml  # type: ignore[import-untyped]  # PyYAML does not ship inline types
    except ImportError as exc:
        raise MastMagneticArchiveDependencyError(
            "MAST qualification requires Python >=3.11 and scpn-fusion[mast]"
        ) from exc
    try:
        decoded = yaml.safe_load(data)
    except yaml.YAMLError as exc:
        raise MastMagneticDiagnosticQualificationError("ingestion mapping YAML is invalid") from exc
    if not isinstance(decoded, dict) or not all(isinstance(key, str) for key in decoded):
        raise MastMagneticDiagnosticQualificationError("ingestion mapping is not an object")
    return cast(JsonObject, decoded)


def _read_regular_file(path: Path, label: str) -> bytes:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise MastMagneticDiagnosticQualificationError(f"{label} is missing") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise MastMagneticDiagnosticQualificationError(f"{label} is not a regular file")
    return resolved.read_bytes()


def _object(value: JsonValue | None, label: str) -> JsonObject:
    if not isinstance(value, dict):
        raise MastMagneticDiagnosticQualificationError(f"{label} is not an object")
    return value


def _array(value: JsonValue, label: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise MastMagneticDiagnosticQualificationError(f"{label} is not an array")
    return value


def _string_sequence(value: JsonValue, label: str) -> list[str]:
    raw = _array(value, label)
    if not all(isinstance(item, str) and item for item in raw):
        raise MastMagneticDiagnosticQualificationError(f"{label} is not a string array")
    return cast(list[str], raw)


def _nonempty_string(value: JsonValue, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise MastMagneticDiagnosticQualificationError(f"{label} is not a nonempty string")
    return value


def _positive_integer(value: JsonValue, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MastMagneticDiagnosticQualificationError(f"{label} is not a positive integer")
    return value


def _nonnegative_integer(value: JsonValue, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MastMagneticDiagnosticQualificationError(f"{label} is not a nonnegative integer")
    return value


def _finite_number(value: JsonValue, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise MastMagneticDiagnosticQualificationError(f"{label} is not a finite number")
    try:
        result = float(value)
    except ValueError as exc:
        raise MastMagneticDiagnosticQualificationError(f"{label} is not a finite number") from exc
    if not math.isfinite(result):
        raise MastMagneticDiagnosticQualificationError(f"{label} is not a finite number")
    return result


def _require_equal(actual: JsonValue | None, expected: JsonValue, label: str) -> None:
    if actual != expected:
        raise MastMagneticDiagnosticQualificationError(f"{label} differs from contract")
