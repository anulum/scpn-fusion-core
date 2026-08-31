# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — complete FAIR-MAST magnetic archive envelope codec
"""Strict canonical codec for complete FAIR-MAST magnetic archive envelopes."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TypeAlias, cast

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

MAST_COMPLETE_MAGNETIC_ARCHIVE_SCHEMA = (
    "scpn-fusion-core.mast-complete-magnetic-archive-envelope.v1"
)
MAST_COMPLETE_MAGNETIC_ARCHIVE_SCHEMA_VERSION = "1.0.0"
MAX_MAST_COMPLETE_MAGNETIC_ARCHIVE_BYTES = 8 * 1024 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_OBSERVATION_ID_RE = re.compile(r"^mast-[1-9][0-9]*-complete-magnetics-[0-9a-f]{16}$")

_PAYLOAD_KEYS = {
    "archive",
    "arrays",
    "authority",
    "clocks",
    "completeness",
    "event_id",
    "event_identity_state",
    "facility",
    "observation_id",
    "producer_artifacts",
    "provenance",
    "qualification",
    "reactor_configuration",
    "shot_id",
    "producer_project",
    "source_archive",
    "source_ingestion_revision",
    "source_ingestion_tree_state",
}
_OBJECT_KEYS = {"path", "sha256", "size_bytes", "source_url"}
_ARRAY_KEYS = {
    "archive_path",
    "attributes",
    "clock_dimensions",
    "data_object_paths",
    "data_type",
    "decoded_content_sha256",
    "decoded_nonfinite_count",
    "decoded_value_count",
    "dimension_names",
    "metadata_object_path",
    "metadata_object_sha256",
    "name",
    "shape",
    "zarr_metadata",
}
_CLOCK_KEYS = {
    "clock_kind_candidate",
    "clock_qualification",
    "finite",
    "first_value_s",
    "last_value_s",
    "mapping_evidence_claimed",
    "maximum_interval_s",
    "mean_interval_s",
    "minimum_interval_s",
    "name",
    "sample_count",
    "strictly_increasing",
    "units",
}
_QUALIFICATION = {
    "calibration_state": "unresolved",
    "channel_geometry_mapping_state": "unresolved",
    "classification_eligible": False,
    "diagnostic_semantics_state": "source_preserved_unqualified",
    "event_clock_state": "unresolved",
    "observation_operator_state": "not_supplied",
    "phase_eligible": False,
    "quality_state": "unknown",
    "raw_samples_present": True,
    "semantic_ingress_eligible": False,
    "source_kind": "physical_archive",
    "synthetic": False,
    "source_clock_relationship_state": "unresolved",
    "uncertainty_state": "unknown",
    "validity_state": "unknown",
}
_AUTHORITY = {
    "actionable": False,
    "classification_performed": False,
    "direct_actuation": False,
    "execution_permitted": False,
    "review_only": True,
}


class MastMagneticArchiveValidationError(ValueError):
    """Raised when a magnetic archive envelope is incomplete or noncanonical."""


@dataclass(frozen=True)
class MastCompleteMagneticArchiveEnvelope:
    """Validated immutable transport bytes for one complete magnetic archive group."""

    _canonical_bytes: bytes

    def to_bytes(self) -> bytes:
        """Return the exact canonical UTF-8 transport bytes."""
        return self._canonical_bytes

    @property
    def sha256(self) -> str:
        """Return the SHA-256 identity of the complete envelope bytes."""
        return hashlib.sha256(self._canonical_bytes).hexdigest()

    @property
    def document(self) -> JsonObject:
        """Return a defensive copy of the validated JSON document."""
        return deepcopy(_parse_json_object(self._canonical_bytes))

    @property
    def payload(self) -> JsonObject:
        """Return a defensive copy of the validated payload."""
        document = self.document
        return deepcopy(_as_object(document["payload"], "payload"))


def mast_complete_magnetic_archive_sha256(data: bytes) -> str:
    """Return the SHA-256 digest of transport or source bytes."""
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: JsonValue) -> bytes:
    """Encode finite JSON with sorted keys, compact separators and one newline."""
    _reject_nonfinite_numbers(value, "document")
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise MastMagneticArchiveValidationError(f"JSON encoding failed: {exc}") from exc
    return (encoded + "\n").encode("utf-8")


def encode_mast_complete_magnetic_archive_envelope(
    payload: Mapping[str, JsonValue],
) -> MastCompleteMagneticArchiveEnvelope:
    """Validate a complete payload and bind it to canonical transport bytes."""
    payload_copy = deepcopy(dict(payload))
    validate_mast_complete_magnetic_archive_payload(payload_copy)
    payload_bytes = canonical_json_bytes(payload_copy)
    document: JsonObject = {
        "payload": payload_copy,
        "payload_sha256": mast_complete_magnetic_archive_sha256(payload_bytes),
        "schema": MAST_COMPLETE_MAGNETIC_ARCHIVE_SCHEMA,
        "schema_version": MAST_COMPLETE_MAGNETIC_ARCHIVE_SCHEMA_VERSION,
    }
    encoded = canonical_json_bytes(document)
    if len(encoded) > MAX_MAST_COMPLETE_MAGNETIC_ARCHIVE_BYTES:
        raise MastMagneticArchiveValidationError(
            f"envelope exceeds {MAX_MAST_COMPLETE_MAGNETIC_ARCHIVE_BYTES} bytes"
        )
    return MastCompleteMagneticArchiveEnvelope(encoded)


def decode_mast_complete_magnetic_archive_envelope(
    data: bytes,
) -> MastCompleteMagneticArchiveEnvelope:
    """Decode canonical bytes and reject any structural or semantic drift."""
    if len(data) > MAX_MAST_COMPLETE_MAGNETIC_ARCHIVE_BYTES:
        raise MastMagneticArchiveValidationError(
            f"envelope exceeds {MAX_MAST_COMPLETE_MAGNETIC_ARCHIVE_BYTES} bytes"
        )
    document = _parse_json_object(data)
    _require_exact_keys(
        document,
        {"schema", "schema_version", "payload", "payload_sha256"},
        "document",
    )
    _require_equal(document["schema"], MAST_COMPLETE_MAGNETIC_ARCHIVE_SCHEMA, "schema")
    _require_equal(
        document["schema_version"],
        MAST_COMPLETE_MAGNETIC_ARCHIVE_SCHEMA_VERSION,
        "schema_version",
    )
    payload = _as_object(document["payload"], "payload")
    validate_mast_complete_magnetic_archive_payload(payload)
    expected_payload_digest = mast_complete_magnetic_archive_sha256(canonical_json_bytes(payload))
    _require_equal(document["payload_sha256"], expected_payload_digest, "payload_sha256")
    if canonical_json_bytes(document) != data:
        raise MastMagneticArchiveValidationError("envelope is not canonical JSON")
    return MastCompleteMagneticArchiveEnvelope(data)


def validate_mast_complete_magnetic_archive_payload(payload: JsonObject) -> None:
    """Validate completeness, source fidelity and permanent non-actuating authority."""
    _require_exact_keys(payload, _PAYLOAD_KEYS, "payload")
    _require_equal(payload["producer_project"], "SCPN-FUSION-CORE", "producer_project")
    _require_equal(payload["source_archive"], "FAIR-MAST", "source_archive")
    _require_matching_string(
        payload["source_ingestion_revision"],
        _REVISION_RE,
        "source_ingestion_revision",
    )
    tree_state = _as_nonempty_string(
        payload["source_ingestion_tree_state"], "source_ingestion_tree_state"
    )
    if tree_state not in {"clean", "dirty"}:
        raise MastMagneticArchiveValidationError("source_ingestion_tree_state is unsupported")
    _require_equal(payload["facility"], "MAST", "facility")
    _require_equal(payload["reactor_configuration"], "spherical_tokamak", "configuration")
    shot_id = _as_positive_integer(payload["shot_id"], "shot_id")
    _require_matching_string(payload["observation_id"], _OBSERVATION_ID_RE, "observation_id")
    if not cast(str, payload["observation_id"]).startswith(f"mast-{shot_id}-"):
        raise MastMagneticArchiveValidationError("observation_id does not bind shot_id")
    _require_equal(payload["event_id"], None, "event_id")
    _require_equal(
        payload["event_identity_state"],
        "unresolved_facility_mapping",
        "event_identity_state",
    )
    _validate_archive(_as_object(payload["archive"], "archive"))
    objects = _validate_provenance(_as_object(payload["provenance"], "provenance"), shot_id)
    arrays = _validate_arrays(_as_list(payload["arrays"], "arrays"), objects)
    clocks = _validate_clocks(_as_list(payload["clocks"], "clocks"), arrays)
    _validate_dimension_and_clock_bindings(arrays, clocks)
    _validate_completeness(
        _as_object(payload["completeness"], "completeness"), objects, arrays, clocks
    )
    _validate_producer_artifacts(_as_list(payload["producer_artifacts"], "producer_artifacts"))
    _require_equal(payload["qualification"], _QUALIFICATION, "qualification")
    _require_equal(payload["authority"], _AUTHORITY, "authority")
    group_metadata_path = cast(str, _as_object(payload["archive"], "archive")["metadata_path"])
    used_paths = {group_metadata_path}
    for array in arrays.values():
        used_paths.add(cast(str, array["metadata_object_path"]))
        used_paths.update(_as_string_list(array["data_object_paths"], "data_object_paths"))
    if used_paths != set(objects):
        missing = sorted(set(objects) - used_paths)
        extra = sorted(used_paths - set(objects))
        raise MastMagneticArchiveValidationError(
            f"archive object coverage mismatch: unbound={missing}, undeclared={extra}"
        )


def _validate_archive(archive: JsonObject) -> None:
    _require_exact_keys(
        archive,
        {
            "group",
            "group_metadata",
            "group_metadata_sha256",
            "metadata_path",
            "root_url",
            "zarr_format",
        },
        "archive",
    )
    _require_equal(archive["group"], "magnetics", "archive.group")
    _require_equal(archive["zarr_format"], 3, "archive.zarr_format")
    _require_sha256(archive["group_metadata_sha256"], "archive.group_metadata_sha256")
    metadata_path = _as_nonempty_string(archive["metadata_path"], "archive.metadata_path")
    if not metadata_path.endswith("/magnetics/zarr.json"):
        raise MastMagneticArchiveValidationError("archive metadata path is not magnetics/zarr.json")
    root_url = _as_nonempty_string(archive["root_url"], "archive.root_url")
    if not root_url.startswith("https://s3.echo.stfc.ac.uk/mast/level2/shots/"):
        raise MastMagneticArchiveValidationError("archive root URL is not the FAIR-MAST endpoint")
    metadata = _as_object(archive["group_metadata"], "archive.group_metadata")
    _require_equal(metadata.get("zarr_format"), 3, "archive.group_metadata.zarr_format")
    _require_equal(metadata.get("node_type"), "group", "archive.group_metadata.node_type")


def _validate_provenance(provenance: JsonObject, shot_id: int) -> dict[str, JsonObject]:
    _require_exact_keys(
        provenance,
        {
            "license",
            "limitations",
            "magnetic_group_manifest_sha256",
            "object_count",
            "objects",
            "parent_download_manifest_sha256",
            "parent_object_count",
            "parent_total_bytes",
            "retrieved_at_utc",
            "schema",
            "sha256",
            "total_bytes",
        },
        "provenance",
    )
    _require_equal(
        provenance["schema"], "scpn-fusion-open-disruption-data-provenance.v1", "provenance.schema"
    )
    _require_sha256(provenance["sha256"], "provenance.sha256")
    _require_sha256(
        provenance["parent_download_manifest_sha256"],
        "provenance.parent_download_manifest_sha256",
    )
    _require_sha256(
        provenance["magnetic_group_manifest_sha256"],
        "provenance.magnetic_group_manifest_sha256",
    )
    _as_nonempty_string(provenance["retrieved_at_utc"], "provenance.retrieved_at_utc")
    limitations = _as_string_list(provenance["limitations"], "provenance.limitations")
    if not limitations:
        raise MastMagneticArchiveValidationError("provenance limitations are empty")
    license_record = _as_object(provenance["license"], "provenance.license")
    _require_equal(license_record.get("spdx"), "CC-BY-SA-4.0", "provenance.license.spdx")
    for key in ("attribution", "name", "statement_scope", "url"):
        _as_nonempty_string(license_record.get(key), f"provenance.license.{key}")
    entries = _as_list(provenance["objects"], "provenance.objects")
    if not entries:
        raise MastMagneticArchiveValidationError("provenance objects are empty")
    objects: dict[str, JsonObject] = {}
    previous = ""
    total_bytes = 0
    manifest_lines: list[str] = []
    expected_prefix = f"raw/{shot_id}.zarr/magnetics/"
    for index, raw_entry in enumerate(entries):
        entry = _as_object(raw_entry, f"provenance.objects[{index}]")
        _require_exact_keys(entry, _OBJECT_KEYS, f"provenance.objects[{index}]")
        path = _as_nonempty_string(entry["path"], f"objects[{index}].path")
        pure_path = PurePosixPath(path)
        if (
            not path.startswith(expected_prefix)
            or path != pure_path.as_posix()
            or ".." in pure_path.parts
            or path <= previous
        ):
            raise MastMagneticArchiveValidationError(
                "provenance object paths are not sorted/complete"
            )
        previous = path
        digest = _require_sha256(entry["sha256"], f"objects[{index}].sha256")
        size = _as_nonnegative_integer(entry["size_bytes"], f"objects[{index}].size_bytes")
        source_url = _as_nonempty_string(entry["source_url"], f"objects[{index}].source_url")
        expected_url = "https://s3.echo.stfc.ac.uk/mast/level2/shots/" + path.removeprefix("raw/")
        if source_url != expected_url:
            raise MastMagneticArchiveValidationError(f"source URL does not bind object path {path}")
        objects[path] = entry
        total_bytes += size
        manifest_lines.append(f"{digest}:{size}:{path}\n")
    _require_equal(provenance["object_count"], len(objects), "provenance.object_count")
    _require_equal(provenance["total_bytes"], total_bytes, "provenance.total_bytes")
    parent_object_count = _as_positive_integer(
        provenance["parent_object_count"], "provenance.parent_object_count"
    )
    parent_total_bytes = _as_positive_integer(
        provenance["parent_total_bytes"], "provenance.parent_total_bytes"
    )
    if parent_object_count < len(objects) or parent_total_bytes < total_bytes:
        raise MastMagneticArchiveValidationError(
            "parent provenance is smaller than magnetics group"
        )
    manifest_digest = mast_complete_magnetic_archive_sha256("".join(manifest_lines).encode())
    _require_equal(
        provenance["magnetic_group_manifest_sha256"],
        manifest_digest,
        "magnetic_group_manifest_sha256",
    )
    return objects


def _validate_arrays(
    raw_arrays: list[JsonValue], objects: Mapping[str, JsonObject]
) -> dict[str, JsonObject]:
    if not raw_arrays:
        raise MastMagneticArchiveValidationError("arrays are empty")
    arrays: dict[str, JsonObject] = {}
    previous = ""
    for index, raw_array in enumerate(raw_arrays):
        array = _as_object(raw_array, f"arrays[{index}]")
        _require_exact_keys(array, _ARRAY_KEYS, f"arrays[{index}]")
        name = _as_nonempty_string(array["name"], f"arrays[{index}].name")
        if name <= previous:
            raise MastMagneticArchiveValidationError("array names are not sorted and unique")
        previous = name
        shape = _as_nonnegative_integer_list(array["shape"], f"arrays[{index}].shape")
        dimensions = _as_string_list(array["dimension_names"], f"arrays[{index}].dimension_names")
        if len(shape) != len(dimensions) or len(set(dimensions)) != len(dimensions):
            raise MastMagneticArchiveValidationError(f"array {name} has invalid dimensions")
        value_count = math.prod(shape) if shape else 1
        _require_equal(array["decoded_value_count"], value_count, f"arrays[{index}].value_count")
        nonfinite = _as_nonnegative_integer(
            array["decoded_nonfinite_count"], f"arrays[{index}].nonfinite_count"
        )
        if nonfinite > value_count:
            raise MastMagneticArchiveValidationError(f"array {name} nonfinite count exceeds size")
        _require_sha256(array["decoded_content_sha256"], f"arrays[{index}].decoded_digest")
        metadata_path = _as_nonempty_string(
            array["metadata_object_path"], f"arrays[{index}].metadata_path"
        )
        if metadata_path not in objects or not metadata_path.endswith(f"/{name}/zarr.json"):
            raise MastMagneticArchiveValidationError(f"array {name} metadata object is unbound")
        _require_equal(
            array["metadata_object_sha256"],
            objects[metadata_path]["sha256"],
            f"arrays[{index}].metadata_digest",
        )
        data_paths = _as_string_list(array["data_object_paths"], f"arrays[{index}].data_paths")
        if data_paths != sorted(set(data_paths)):
            raise MastMagneticArchiveValidationError(f"array {name} data objects are incomplete")
        if any(path not in objects for path in data_paths):
            raise MastMagneticArchiveValidationError(f"array {name} references unknown data object")
        _as_string_list(array["clock_dimensions"], f"arrays[{index}].clock_dimensions")
        metadata = _as_object(array["zarr_metadata"], f"arrays[{index}].zarr_metadata")
        _require_equal(metadata.get("node_type"), "array", f"arrays[{index}].node_type")
        _require_equal(metadata.get("zarr_format"), 3, f"arrays[{index}].zarr_format")
        _require_equal(metadata.get("shape"), shape, f"arrays[{index}].metadata.shape")
        expected_metadata_dimensions = cast(JsonValue, dimensions) if shape else None
        _require_equal(
            metadata.get("dimension_names"),
            expected_metadata_dimensions,
            f"arrays[{index}].metadata.dimensions",
        )
        _require_equal(metadata.get("attributes"), array["attributes"], f"arrays[{index}].attrs")
        arrays[name] = array
    return arrays


def _validate_clocks(
    raw_clocks: list[JsonValue], arrays: Mapping[str, JsonObject]
) -> dict[str, JsonObject]:
    if not raw_clocks:
        raise MastMagneticArchiveValidationError("no native archive clocks declared")
    clocks: dict[str, JsonObject] = {}
    previous = ""
    for index, raw_clock in enumerate(raw_clocks):
        clock = _as_object(raw_clock, f"clocks[{index}]")
        _require_exact_keys(clock, _CLOCK_KEYS, f"clocks[{index}]")
        name = _as_nonempty_string(clock["name"], f"clocks[{index}].name")
        if name <= previous or name not in arrays:
            raise MastMagneticArchiveValidationError("clock names are not sorted or source-bound")
        previous = name
        source_array = arrays[name]
        shape = _as_nonnegative_integer_list(source_array["shape"], f"clock {name} shape")
        dimensions = _as_string_list(source_array["dimension_names"], f"clock {name} dimensions")
        if shape != [_as_positive_integer(clock["sample_count"], f"clock {name} samples")]:
            raise MastMagneticArchiveValidationError(f"clock {name} sample count mismatch")
        if dimensions != [name]:
            raise MastMagneticArchiveValidationError(f"clock {name} is not its own coordinate")
        _require_equal(clock["units"], "s", f"clock {name} units")
        _require_equal(clock["finite"], True, f"clock {name} finite")
        _require_equal(clock["strictly_increasing"], True, f"clock {name} increasing")
        _require_equal(clock["clock_kind_candidate"], "shot_relative", f"clock {name} kind")
        _require_equal(clock["clock_qualification"], "unresolved", f"clock {name} qualification")
        _require_equal(clock["mapping_evidence_claimed"], False, f"clock {name} mapping")
        first = _as_finite_number(clock["first_value_s"], f"clock {name} first")
        last = _as_finite_number(clock["last_value_s"], f"clock {name} last")
        if last <= first:
            raise MastMagneticArchiveValidationError(f"clock {name} bounds are invalid")
        for field in ("minimum_interval_s", "mean_interval_s", "maximum_interval_s"):
            if _as_finite_number(clock[field], f"clock {name} {field}") <= 0.0:
                raise MastMagneticArchiveValidationError(f"clock {name} interval is not positive")
        clocks[name] = clock
    return clocks


def _validate_dimension_and_clock_bindings(
    arrays: Mapping[str, JsonObject], clocks: Mapping[str, JsonObject]
) -> None:
    clock_names = set(clocks)
    for name, array in arrays.items():
        dimensions = _as_string_list(array["dimension_names"], f"array {name} dimensions")
        shape = _as_nonnegative_integer_list(array["shape"], f"array {name} shape")
        for dimension, size in zip(dimensions, shape, strict=True):
            if dimension not in arrays:
                raise MastMagneticArchiveValidationError(
                    f"array {name} dimension {dimension} has no source coordinate"
                )
            coordinate_shape = _as_nonnegative_integer_list(
                arrays[dimension]["shape"], f"coordinate {dimension} shape"
            )
            if coordinate_shape != [size]:
                raise MastMagneticArchiveValidationError(
                    f"array {name} dimension {dimension} cardinality mismatch"
                )
        expected_clocks = [dimension for dimension in dimensions if dimension in clock_names]
        declared_clocks = _as_string_list(array["clock_dimensions"], f"array {name} clocks")
        if declared_clocks != expected_clocks:
            raise MastMagneticArchiveValidationError(f"array {name} clock binding mismatch")


def _validate_completeness(
    record: JsonObject,
    objects: Mapping[str, JsonObject],
    arrays: Mapping[str, JsonObject],
    clocks: Mapping[str, JsonObject],
) -> None:
    _require_exact_keys(
        record,
        {"array_count", "arrays_complete", "clock_count", "objects_complete", "source_decoded"},
        "completeness",
    )
    _require_equal(record["array_count"], len(arrays), "completeness.array_count")
    _require_equal(record["clock_count"], len(clocks), "completeness.clock_count")
    _require_equal(record["arrays_complete"], True, "completeness.arrays_complete")
    _require_equal(record["objects_complete"], True, "completeness.objects_complete")
    _require_equal(record["source_decoded"], True, "completeness.source_decoded")


def _validate_producer_artifacts(raw_artifacts: list[JsonValue]) -> None:
    if not raw_artifacts:
        raise MastMagneticArchiveValidationError("producer artifact inventory is empty")
    previous = ""
    for index, raw_artifact in enumerate(raw_artifacts):
        artifact = _as_object(raw_artifact, f"producer_artifacts[{index}]")
        _require_exact_keys(artifact, {"path", "sha256"}, f"producer_artifacts[{index}]")
        path = _as_nonempty_string(artifact["path"], f"producer_artifacts[{index}].path")
        if path <= previous:
            raise MastMagneticArchiveValidationError("producer artifacts are not sorted/unique")
        previous = path
        _require_sha256(artifact["sha256"], f"producer_artifacts[{index}].sha256")


def _parse_json_object(data: bytes) -> JsonObject:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MastMagneticArchiveValidationError("envelope is not valid UTF-8") from exc
    try:
        value = json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
    except (json.JSONDecodeError, MastMagneticArchiveValidationError) as exc:
        raise MastMagneticArchiveValidationError(f"invalid envelope JSON: {exc}") from exc
    return _as_object(cast(JsonValue, value), "document")


def _reject_duplicate_pairs(pairs: list[tuple[str, JsonValue]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise MastMagneticArchiveValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_numbers(value: JsonValue, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise MastMagneticArchiveValidationError(f"{path} contains a nonfinite number")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite_numbers(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite_numbers(item, f"{path}.{key}")


def _require_exact_keys(record: Mapping[str, JsonValue], expected: set[str], path: str) -> None:
    actual = set(record)
    if actual != expected:
        raise MastMagneticArchiveValidationError(
            f"{path} keys differ: missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


def _require_equal(actual: object, expected: object, path: str) -> None:
    if actual != expected or type(actual) is not type(expected):
        raise MastMagneticArchiveValidationError(f"{path} must equal {expected!r}")


def _as_object(value: JsonValue | object, path: str) -> JsonObject:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise MastMagneticArchiveValidationError(f"{path} must be a JSON object")
    return cast(JsonObject, value)


def _as_list(value: JsonValue, path: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise MastMagneticArchiveValidationError(f"{path} must be a JSON array")
    return value


def _as_nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MastMagneticArchiveValidationError(f"{path} must be a non-empty string")
    return value


def _as_string_list(value: JsonValue, path: str) -> list[str]:
    values = _as_list(value, path)
    if any(not isinstance(item, str) or not item for item in values):
        raise MastMagneticArchiveValidationError(f"{path} must contain non-empty strings")
    return cast(list[str], values)


def _as_nonnegative_integer_list(value: JsonValue, path: str) -> list[int]:
    values = _as_list(value, path)
    return [_as_nonnegative_integer(item, f"{path}[{index}]") for index, item in enumerate(values)]


def _as_nonnegative_integer(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise MastMagneticArchiveValidationError(f"{path} must be a non-negative integer")
    return value


def _as_positive_integer(value: object, path: str) -> int:
    result = _as_nonnegative_integer(value, path)
    if result == 0:
        raise MastMagneticArchiveValidationError(f"{path} must be positive")
    return result


def _as_finite_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MastMagneticArchiveValidationError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise MastMagneticArchiveValidationError(f"{path} must be finite")
    return result


def _require_sha256(value: object, path: str) -> str:
    return _require_matching_string(value, _SHA256_RE, path)


def _require_matching_string(value: object, pattern: re.Pattern[str], path: str) -> str:
    result = _as_nonempty_string(value, path)
    if pattern.fullmatch(result) is None:
        raise MastMagneticArchiveValidationError(f"{path} has invalid format")
    return result


__all__ = [
    "MAST_COMPLETE_MAGNETIC_ARCHIVE_SCHEMA",
    "MAST_COMPLETE_MAGNETIC_ARCHIVE_SCHEMA_VERSION",
    "MAX_MAST_COMPLETE_MAGNETIC_ARCHIVE_BYTES",
    "MastCompleteMagneticArchiveEnvelope",
    "MastMagneticArchiveValidationError",
    "canonical_json_bytes",
    "decode_mast_complete_magnetic_archive_envelope",
    "encode_mast_complete_magnetic_archive_envelope",
    "mast_complete_magnetic_archive_sha256",
    "validate_mast_complete_magnetic_archive_payload",
]
