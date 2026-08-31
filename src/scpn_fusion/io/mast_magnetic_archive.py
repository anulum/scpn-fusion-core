# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — complete FAIR-MAST magnetic archive capture
"""Capture every object and decoded array in a FAIR-MAST ``magnetics`` group."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .mast_magnetic_archive_codec import (
    JsonObject,
    JsonValue,
    MastCompleteMagneticArchiveEnvelope,
    MastMagneticArchiveValidationError,
    canonical_json_bytes,
    decode_mast_complete_magnetic_archive_envelope,
    encode_mast_complete_magnetic_archive_envelope,
    mast_complete_magnetic_archive_sha256,
)

_PROVENANCE_SCHEMA = "scpn-fusion-open-disruption-data-provenance.v1"
_PRODUCER_PROJECT = "SCPN-FUSION-CORE"
_FAIR_MAST_ROOT = "https://s3.echo.stfc.ac.uk/mast/level2/shots"


class MastMagneticArchiveDependencyError(ImportError):
    """Raised when the safe Zarr-v3 MAST dependency profile is unavailable."""


def build_mast_complete_magnetic_archive_envelope(
    provenance_path: Path,
    shot_archive_root: Path,
) -> MastCompleteMagneticArchiveEnvelope:
    """Verify, decode and describe one complete authentic magnetic archive group.

    Parameters
    ----------
    provenance_path:
        Exact tracked parent provenance manifest containing every source object.
    shot_archive_root:
        Local root of the fully materialised ``<shot>.zarr`` archive.

    Returns
    -------
    MastCompleteMagneticArchiveEnvelope
        Canonical review-only envelope binding all archive objects and arrays.

    Raises
    ------
    MastMagneticArchiveValidationError
        If any object, array, clock, source identity or completeness invariant fails.
    MastMagneticArchiveDependencyError
        If Python or the installed optional dependency profile cannot decode Zarr v3.
    """
    provenance_bytes = _read_regular_file(provenance_path, "provenance manifest")
    provenance = _parse_json_object(provenance_bytes, "provenance manifest")
    _require_equal(provenance.get("schema"), _PROVENANCE_SCHEMA, "provenance schema")
    dataset = _as_object(provenance.get("dataset"), "provenance.dataset")
    shot_id = _as_positive_integer(dataset.get("shot_id"), "provenance.dataset.shot_id")
    _require_equal(dataset.get("device"), "MAST", "provenance.dataset.device")
    groups = _as_string_list(dataset.get("downloaded_groups"), "downloaded_groups")
    if "magnetics" not in groups:
        raise MastMagneticArchiveValidationError("parent provenance omits magnetics group")

    all_objects = _validate_parent_object_manifest(dataset, shot_id)
    magnetic_prefix = f"raw/{shot_id}.zarr/magnetics/"
    magnetic_objects = {
        path: record for path, record in all_objects.items() if path.startswith(magnetic_prefix)
    }
    if not magnetic_objects:
        raise MastMagneticArchiveValidationError("parent provenance contains no magnetic objects")

    try:
        archive_root = shot_archive_root.resolve(strict=True)
    except OSError as exc:
        raise MastMagneticArchiveValidationError("shot archive root is missing") from exc
    if not archive_root.is_dir():
        raise MastMagneticArchiveValidationError("shot archive root is not a directory")
    magnetic_root = archive_root / "magnetics"
    if not magnetic_root.is_dir() or magnetic_root.is_symlink():
        raise MastMagneticArchiveValidationError("magnetics group is missing or symlinked")
    _verify_complete_local_object_set(archive_root, magnetic_objects, shot_id)

    verified_bytes: dict[str, bytes] = {}
    for path, record in magnetic_objects.items():
        local_path = archive_root / path.removeprefix(f"raw/{shot_id}.zarr/")
        data = _read_regular_file(local_path, path)
        _require_equal(len(data), record["size_bytes"], f"size for {path}")
        _require_equal(
            mast_complete_magnetic_archive_sha256(data), record["sha256"], f"digest for {path}"
        )
        verified_bytes[path] = data

    group_metadata_path = f"{magnetic_prefix}zarr.json"
    if group_metadata_path not in verified_bytes:
        raise MastMagneticArchiveValidationError("magnetics group metadata object is missing")
    group_metadata = _parse_json_object(
        verified_bytes[group_metadata_path], "magnetics group metadata"
    )
    _require_equal(group_metadata.get("zarr_format"), 3, "group zarr format")
    _require_equal(group_metadata.get("node_type"), "group", "group node type")
    group_attributes = _as_object(group_metadata.get("attributes"), "group attributes")
    _validate_group_identity_and_license(group_attributes)
    source_ingestion_revision, source_ingestion_tree_state = _source_ingestion_identity(
        group_attributes
    )

    zarr_group = _open_zarr_v3_group(magnetic_root)
    array_names = tuple(sorted(cast(Any, zarr_group).array_keys()))
    metadata_names = tuple(
        sorted(
            path.removeprefix(magnetic_prefix).removesuffix("/zarr.json")
            for path in magnetic_objects
            if path != group_metadata_path and path.endswith("/zarr.json")
        )
    )
    if array_names != metadata_names:
        raise MastMagneticArchiveValidationError(
            f"decoded arrays differ from source metadata: decoded={array_names}, metadata={metadata_names}"
        )

    metadata_by_name = {
        name: _parse_json_object(
            verified_bytes[f"{magnetic_prefix}{name}/zarr.json"], f"array metadata {name}"
        )
        for name in array_names
    }
    clock_names = _discover_native_clocks(metadata_by_name)
    arrays = [
        _build_array_record(
            name,
            metadata_by_name[name],
            cast(Any, zarr_group)[name],
            magnetic_prefix,
            magnetic_objects,
            clock_names,
        )
        for name in array_names
    ]
    clocks = [
        _build_clock_record(name, cast(Any, zarr_group)[name], metadata_by_name[name])
        for name in clock_names
    ]

    magnetic_manifest_digest = _object_manifest_digest(magnetic_objects.values())
    parent_files = _as_list(dataset["files"], "provenance.dataset.files")
    parent_total_bytes = sum(
        _as_nonnegative_integer(
            _as_object(item, "provenance.dataset.files entry").get("size_bytes"),
            "parent size_bytes",
        )
        for item in parent_files
    )
    root_url = f"{_FAIR_MAST_ROOT}/{shot_id}.zarr/magnetics/"
    payload: JsonObject = {
        "archive": {
            "group": "magnetics",
            "group_metadata": group_metadata,
            "group_metadata_sha256": cast(str, magnetic_objects[group_metadata_path]["sha256"]),
            "metadata_path": group_metadata_path,
            "root_url": root_url,
            "zarr_format": 3,
        },
        "arrays": cast(JsonValue, arrays),
        "authority": {
            "actionable": False,
            "classification_performed": False,
            "direct_actuation": False,
            "execution_permitted": False,
            "review_only": True,
        },
        "clocks": cast(JsonValue, clocks),
        "completeness": {
            "array_count": len(arrays),
            "arrays_complete": True,
            "clock_count": len(clocks),
            "objects_complete": True,
            "source_decoded": True,
        },
        "event_id": None,
        "event_identity_state": "unresolved_facility_mapping",
        "facility": "MAST",
        "observation_id": f"mast-{shot_id}-complete-magnetics-{magnetic_manifest_digest[:16]}",
        "producer_artifacts": cast(JsonValue, _producer_artifact_records()),
        "provenance": {
            "license": provenance.get("license"),
            "limitations": provenance.get("limitations"),
            "magnetic_group_manifest_sha256": magnetic_manifest_digest,
            "object_count": len(magnetic_objects),
            "objects": cast(JsonValue, list(magnetic_objects.values())),
            "parent_download_manifest_sha256": dataset["download_manifest_sha256"],
            "parent_object_count": len(parent_files),
            "parent_total_bytes": parent_total_bytes,
            "retrieved_at_utc": provenance.get("retrieved_at_utc"),
            "schema": provenance.get("schema"),
            "sha256": mast_complete_magnetic_archive_sha256(provenance_bytes),
            "total_bytes": sum(
                cast(int, record["size_bytes"]) for record in magnetic_objects.values()
            ),
        },
        "qualification": {
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
        },
        "reactor_configuration": "spherical_tokamak",
        "shot_id": shot_id,
        "producer_project": _PRODUCER_PROJECT,
        "source_archive": "FAIR-MAST",
        "source_ingestion_revision": source_ingestion_revision,
        "source_ingestion_tree_state": source_ingestion_tree_state,
    }
    return encode_mast_complete_magnetic_archive_envelope(payload)


def verify_mast_complete_magnetic_archive_source(
    envelope: MastCompleteMagneticArchiveEnvelope | bytes,
    shot_archive_root: Path,
) -> None:
    """Reverify every declared object and reject undeclared local archive objects."""
    validated = (
        decode_mast_complete_magnetic_archive_envelope(envelope)
        if isinstance(envelope, bytes)
        else decode_mast_complete_magnetic_archive_envelope(envelope.to_bytes())
    )
    payload = validated.payload
    shot_id = _as_positive_integer(payload["shot_id"], "shot_id")
    provenance = _as_object(payload["provenance"], "provenance")
    objects = {
        cast(str, record["path"]): record
        for record in (
            _as_object(item, "provenance object")
            for item in _as_list(provenance["objects"], "provenance.objects")
        )
    }
    try:
        archive_root = shot_archive_root.resolve(strict=True)
    except OSError as exc:
        raise MastMagneticArchiveValidationError("shot archive root is missing") from exc
    _verify_complete_local_object_set(archive_root, objects, shot_id)
    for path, record in objects.items():
        local_path = archive_root / path.removeprefix(f"raw/{shot_id}.zarr/")
        data = _read_regular_file(local_path, path)
        _require_equal(len(data), record["size_bytes"], f"size for {path}")
        _require_equal(
            mast_complete_magnetic_archive_sha256(data), record["sha256"], f"digest for {path}"
        )


def _validate_parent_object_manifest(dataset: JsonObject, shot_id: int) -> dict[str, JsonObject]:
    raw_files = _as_list(dataset.get("files"), "provenance.dataset.files")
    if not raw_files:
        raise MastMagneticArchiveValidationError("parent provenance file inventory is empty")
    objects: dict[str, JsonObject] = {}
    for index, raw_file in enumerate(raw_files):
        record = _as_object(raw_file, f"provenance.dataset.files[{index}]")
        if set(record) != {"path", "sha256", "size_bytes", "source_url"}:
            raise MastMagneticArchiveValidationError("parent provenance object keys differ")
        path = _as_nonempty_string(record.get("path"), f"files[{index}].path")
        expected_prefix = f"raw/{shot_id}.zarr/"
        pure_path = PurePosixPath(path)
        if (
            not path.startswith(expected_prefix)
            or path != pure_path.as_posix()
            or ".." in pure_path.parts
            or path in objects
        ):
            raise MastMagneticArchiveValidationError(
                "parent object path is duplicate or cross-shot"
            )
        digest = _as_nonempty_string(record.get("sha256"), f"files[{index}].sha256")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise MastMagneticArchiveValidationError(
                "parent object digest is not lowercase SHA-256"
            )
        size = _as_nonnegative_integer(record.get("size_bytes"), f"files[{index}].size_bytes")
        source_url = _as_nonempty_string(record.get("source_url"), f"files[{index}].source_url")
        expected_url = f"{_FAIR_MAST_ROOT}/" + path.removeprefix("raw/")
        if source_url != expected_url:
            raise MastMagneticArchiveValidationError(f"source URL does not bind {path}")
        objects[path] = {
            "path": path,
            "sha256": digest,
            "size_bytes": size,
            "source_url": source_url,
        }
    ordered = dict(sorted(objects.items()))
    digest = _object_manifest_digest(ordered.values())
    _require_equal(
        dataset.get("download_manifest_sha256"), digest, "parent download manifest digest"
    )
    return ordered


def _verify_complete_local_object_set(
    archive_root: Path, objects: Mapping[str, JsonObject], shot_id: int
) -> None:
    magnetic_root = archive_root / "magnetics"
    actual_paths: set[str] = set()
    for candidate in magnetic_root.rglob("*"):
        if candidate.is_symlink():
            raise MastMagneticArchiveValidationError(f"archive contains symlink: {candidate}")
        if candidate.is_file():
            relative = candidate.relative_to(archive_root).as_posix()
            actual_paths.add(f"raw/{shot_id}.zarr/{relative}")
        elif not candidate.is_dir():
            raise MastMagneticArchiveValidationError(
                f"archive contains non-regular object: {candidate}"
            )
    expected_paths = set(objects)
    if actual_paths != expected_paths:
        raise MastMagneticArchiveValidationError(
            "local archive inventory differs: "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"undeclared={sorted(actual_paths - expected_paths)}"
        )


def _build_array_record(
    name: str,
    metadata: JsonObject,
    zarr_array: Any,
    magnetic_prefix: str,
    objects: Mapping[str, JsonObject],
    clock_names: tuple[str, ...],
) -> JsonObject:
    _require_equal(metadata.get("zarr_format"), 3, f"{name} zarr format")
    _require_equal(metadata.get("node_type"), "array", f"{name} node type")
    shape = _as_nonnegative_integer_list(metadata.get("shape"), f"{name} shape")
    dimension_names = _dimension_names(metadata, shape, name)
    if len(shape) != len(dimension_names):
        raise MastMagneticArchiveValidationError(f"{name} shape/dimension rank differs")
    decoded = np.asarray(zarr_array[...])
    if list(decoded.shape) != shape:
        raise MastMagneticArchiveValidationError(f"{name} decoded shape differs from metadata")
    value_count = int(decoded.size) if decoded.ndim else 1
    decoded_digest = _decoded_array_digest(decoded)
    nonfinite_count = _nonfinite_count(decoded)
    metadata_path = f"{magnetic_prefix}{name}/zarr.json"
    data_prefix = f"{magnetic_prefix}{name}/"
    data_paths = sorted(
        path for path in objects if path.startswith(data_prefix) and path != metadata_path
    )
    attributes_value = metadata.get("attributes", {})
    attributes = _as_object(attributes_value, f"{name} attributes")
    return {
        "archive_path": f"magnetics/{name}",
        "attributes": attributes,
        "clock_dimensions": cast(
            JsonValue,
            [dimension for dimension in dimension_names if dimension in clock_names],
        ),
        "data_object_paths": cast(JsonValue, data_paths),
        "data_type": metadata.get("data_type"),
        "decoded_content_sha256": decoded_digest,
        "decoded_nonfinite_count": nonfinite_count,
        "decoded_value_count": value_count,
        "dimension_names": cast(JsonValue, dimension_names),
        "metadata_object_path": metadata_path,
        "metadata_object_sha256": cast(str, objects[metadata_path]["sha256"]),
        "name": name,
        "shape": cast(JsonValue, shape),
        "zarr_metadata": metadata,
    }


def _discover_native_clocks(metadata_by_name: Mapping[str, JsonObject]) -> tuple[str, ...]:
    clocks: list[str] = []
    for name, metadata in metadata_by_name.items():
        shape = _as_nonnegative_integer_list(metadata.get("shape"), f"{name} shape")
        dimensions = _dimension_names(metadata, shape, name)
        attributes = _as_object(metadata.get("attributes", {}), f"{name} attributes")
        if len(shape) == 1 and dimensions == [name] and attributes.get("units") == "s":
            clocks.append(name)
    if not clocks:
        raise MastMagneticArchiveValidationError("magnetics group declares no native clocks")
    return tuple(sorted(clocks))


def _build_clock_record(name: str, zarr_array: Any, metadata: JsonObject) -> JsonObject:
    values = np.asarray(zarr_array[...], dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise MastMagneticArchiveValidationError(f"clock {name} is not a nontrivial vector")
    finite = bool(np.all(np.isfinite(values)))
    intervals = np.diff(values)
    increasing = bool(np.all(intervals > 0.0))
    if not finite or not increasing:
        raise MastMagneticArchiveValidationError(f"clock {name} is nonfinite or not increasing")
    attributes = _as_object(metadata.get("attributes", {}), f"clock {name} attributes")
    _require_equal(attributes.get("units"), "s", f"clock {name} units")
    return {
        "clock_kind_candidate": "shot_relative",
        "clock_qualification": "unresolved",
        "finite": True,
        "first_value_s": float(values[0]),
        "last_value_s": float(values[-1]),
        "mapping_evidence_claimed": False,
        "maximum_interval_s": float(np.max(intervals)),
        "mean_interval_s": float(np.mean(intervals)),
        "minimum_interval_s": float(np.min(intervals)),
        "name": name,
        "sample_count": int(values.size),
        "strictly_increasing": True,
        "units": "s",
    }


def _dimension_names(metadata: JsonObject, shape: list[int], name: str) -> list[str]:
    raw_dimensions = metadata.get("dimension_names")
    if raw_dimensions is None and not shape:
        return []
    return _as_string_list(raw_dimensions, f"{name} dimensions")


def _decoded_array_digest(values: NDArray[Any]) -> str:
    shape = list(values.shape)
    if values.dtype.kind in {"O", "S", "U", "T"}:
        content: JsonObject = {
            "dtype": str(values.dtype),
            "shape": cast(JsonValue, shape),
            "values": cast(JsonValue, values.tolist()),
        }
        return mast_complete_magnetic_archive_sha256(canonical_json_bytes(content))
    canonical_dtype = values.dtype.newbyteorder("<")
    contiguous = np.ascontiguousarray(values.astype(canonical_dtype, copy=False))
    header: JsonObject = {"dtype": contiguous.dtype.str, "shape": cast(JsonValue, shape)}
    digest = hashlib.sha256(canonical_json_bytes(header))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _nonfinite_count(values: NDArray[Any]) -> int:
    if values.dtype.kind not in {"f", "c"}:
        return 0
    return int(values.size - np.count_nonzero(np.isfinite(values)))


def _open_zarr_v3_group(magnetic_root: Path) -> Any:
    try:
        zarr = import_module("zarr")
    except ImportError as exc:
        raise MastMagneticArchiveDependencyError(
            "complete FAIR-MAST ingestion requires Python >=3.11 and scpn-fusion[mast]"
        ) from exc
    version_parts = tuple(int(part) for part in zarr.__version__.split(".")[:2])
    if version_parts < (3, 1):
        raise MastMagneticArchiveDependencyError(
            f"complete FAIR-MAST ingestion requires Zarr >=3.1,<3.2; found {zarr.__version__}"
        )
    if version_parts >= (3, 2):
        raise MastMagneticArchiveDependencyError(
            f"unreviewed Zarr version {zarr.__version__}; require the hash-locked MAST profile"
        )
    try:
        return zarr.open_group(magnetic_root, mode="r")
    except Exception as exc:
        raise MastMagneticArchiveValidationError(
            f"cannot open authentic Zarr v3 group: {exc}"
        ) from exc


def _validate_group_identity_and_license(attributes: JsonObject) -> None:
    _require_equal(attributes.get("name"), "magnetics", "group name")
    _require_equal(attributes.get("imas"), "magnetics", "group IMAS identity")
    _require_equal(
        attributes.get("license_url"),
        "https://creativecommons.org/licenses/by-sa/4.0/",
        "group license URL",
    )
    _require_equal(
        attributes.get("license_name"),
        "Creative Commons 4.0 BY-SA",
        "group license name",
    )
    for key in ("description", "ingested_at", "commit_url"):
        _as_nonempty_string(attributes.get(key), f"group {key}")


def _source_ingestion_identity(attributes: JsonObject) -> tuple[str, str]:
    commit_url = _as_nonempty_string(attributes.get("commit_url"), "group commit_url")
    expected_prefix = "https://github.com/ukaea/fair-mast-ingestion/tree/"
    if not commit_url.startswith(expected_prefix):
        raise MastMagneticArchiveValidationError("group commit URL is not FAIR-MAST ingestion")
    identity = commit_url.removeprefix(expected_prefix)
    tree_state = "dirty" if identity.endswith(" (dirty)") else "clean"
    revision = identity.removesuffix(" (dirty)")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise MastMagneticArchiveValidationError("group ingestion revision is not a Git SHA-1")
    return revision, tree_state


def _producer_artifact_records() -> list[JsonObject]:
    module_dir = Path(__file__).resolve().parent
    paths = [
        module_dir / "mast_magnetic_archive.py",
        module_dir / "mast_magnetic_archive_acquisition.py",
        module_dir / "mast_magnetic_archive_codec.py",
    ]
    return [
        {
            "path": f"src/scpn_fusion/io/{path.name}",
            "sha256": mast_complete_magnetic_archive_sha256(_read_regular_file(path, path.name)),
        }
        for path in sorted(paths)
    ]


def _object_manifest_digest(records: Any) -> str:
    lines: list[str] = []
    for record in sorted(records, key=lambda item: cast(str, item["path"])):
        lines.append(f"{record['sha256']}:{record['size_bytes']}:{record['path']}\n")
    return mast_complete_magnetic_archive_sha256("".join(lines).encode("utf-8"))


def _parse_json_object(data: bytes, path: str) -> JsonObject:
    def reject_duplicates(pairs: list[tuple[str, JsonValue]]) -> JsonObject:
        result: JsonObject = {}
        for key, value in pairs:
            if key in result:
                raise MastMagneticArchiveValidationError(f"{path} has duplicate key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MastMagneticArchiveValidationError(f"{path} is not valid UTF-8 JSON") from exc
    return _as_object(cast(JsonValue, value), path)


def _read_regular_file(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise MastMagneticArchiveValidationError(f"{label} is missing, symlinked or non-regular")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise MastMagneticArchiveValidationError(f"cannot read {label}: {exc}") from exc


def _as_object(value: object, path: str) -> JsonObject:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise MastMagneticArchiveValidationError(f"{path} must be an object")
    return cast(JsonObject, value)


def _as_list(value: object, path: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise MastMagneticArchiveValidationError(f"{path} must be an array")
    return cast(list[JsonValue], value)


def _as_string_list(value: object, path: str) -> list[str]:
    result = _as_list(value, path)
    if any(not isinstance(item, str) or not item for item in result):
        raise MastMagneticArchiveValidationError(f"{path} must contain non-empty strings")
    return cast(list[str], result)


def _as_nonnegative_integer_list(value: object, path: str) -> list[int]:
    return [
        _as_nonnegative_integer(item, f"{path}[{index}]")
        for index, item in enumerate(_as_list(value, path))
    ]


def _as_nonnegative_integer(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise MastMagneticArchiveValidationError(f"{path} must be a non-negative integer")
    return value


def _as_positive_integer(value: object, path: str) -> int:
    result = _as_nonnegative_integer(value, path)
    if result == 0:
        raise MastMagneticArchiveValidationError(f"{path} must be positive")
    return result


def _as_nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MastMagneticArchiveValidationError(f"{path} must be a non-empty string")
    return value


def _require_equal(actual: object, expected: object, path: str) -> None:
    if actual != expected or type(actual) is not type(expected):
        raise MastMagneticArchiveValidationError(f"{path} must equal {expected!r}")


__all__ = [
    "MastMagneticArchiveDependencyError",
    "build_mast_complete_magnetic_archive_envelope",
    "verify_mast_complete_magnetic_archive_source",
]
