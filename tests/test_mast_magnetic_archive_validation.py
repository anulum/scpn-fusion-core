# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — complete FAIR-MAST magnetic semantic validation tests
"""Named semantic refusal tests over the complete authentic witness."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import cast

import pytest

from scpn_fusion.io import (
    MastMagneticArchiveValidationError,
    decode_mast_complete_magnetic_archive_envelope,
    encode_mast_complete_magnetic_archive_envelope,
)
from scpn_fusion.io.mast_magnetic_archive_codec import JsonObject, JsonValue
from scpn_fusion.io.mast_magnetic_archive_codec import canonical_json_bytes

_REFERENCE = Path("validation/reference_data/mast/disruption_shots/MAGNETIC_ARCHIVE_ENVELOPE.json")


def _payload() -> JsonObject:
    return decode_mast_complete_magnetic_archive_envelope(_REFERENCE.read_bytes()).payload


def _object(record: JsonObject, key: str) -> JsonObject:
    return cast(JsonObject, record[key])


def _array(record: JsonObject, key: str) -> list[JsonObject]:
    return cast(list[JsonObject], record[key])


def _assert_rejected(payload: JsonObject, message: str) -> None:
    with pytest.raises(MastMagneticArchiveValidationError, match=message):
        encode_mast_complete_magnetic_archive_envelope(payload)


def test_envelope_value_object_is_immutable_and_digest_bound() -> None:
    """Public copies cannot mutate canonical bytes or their SHA-256 identity."""
    source = _REFERENCE.read_bytes()
    envelope = decode_mast_complete_magnetic_archive_envelope(source)
    expected_digest = envelope.sha256
    document = envelope.document
    document["schema"] = "changed"
    payload = envelope.payload
    payload["facility"] = "changed"
    assert envelope.to_bytes() == source
    assert envelope.document["schema"] != "changed"
    assert envelope.payload["facility"] == "MAST"
    assert len(expected_digest) == 64


def test_transport_rejects_invalid_utf8_duplicate_keys_and_nonobjects() -> None:
    """Ambiguous or non-object JSON never reaches semantic validation."""
    with pytest.raises(MastMagneticArchiveValidationError, match="UTF-8"):
        decode_mast_complete_magnetic_archive_envelope(b"\xff")
    with pytest.raises(MastMagneticArchiveValidationError, match="duplicate JSON key"):
        decode_mast_complete_magnetic_archive_envelope(b'{"schema":1,"schema":2}\n')
    with pytest.raises(MastMagneticArchiveValidationError, match="must be a JSON object"):
        decode_mast_complete_magnetic_archive_envelope(b"[]\n")


def test_transport_rejects_nonfinite_and_oversized_documents() -> None:
    """Canonical transport is finite and DoS-bounded."""
    payload = _payload()
    archive = _object(payload, "archive")
    group_metadata = _object(archive, "group_metadata")
    attributes = _object(group_metadata, "attributes")
    attributes["invalid_number"] = math.nan
    _assert_rejected(payload, "nonfinite")

    with pytest.raises(MastMagneticArchiveValidationError, match="exceeds"):
        decode_mast_complete_magnetic_archive_envelope(b" " * (8 * 1024 * 1024 + 1))

    payload = _payload()
    _object(payload, "provenance")["limitations"] = cast(JsonValue, ["x" * (8 * 1024 * 1024)])
    _assert_rejected(payload, "exceeds")

    with pytest.raises(MastMagneticArchiveValidationError, match="encoding failed"):
        canonical_json_bytes(cast(JsonValue, object()))


def test_document_identity_fields_are_exact() -> None:
    """Schema, version, payload digest, and exact document keys are immutable."""
    document = json.loads(_REFERENCE.read_bytes())
    document["schema"] = "unknown"
    with pytest.raises(MastMagneticArchiveValidationError, match="schema"):
        decode_mast_complete_magnetic_archive_envelope(
            (json.dumps(document, separators=(",", ":"), sort_keys=True) + "\n").encode()
        )

    document = json.loads(_REFERENCE.read_bytes())
    document["payload_sha256"] = "0" * 64
    with pytest.raises(MastMagneticArchiveValidationError, match="payload_sha256"):
        decode_mast_complete_magnetic_archive_envelope(
            (json.dumps(document, separators=(",", ":"), sort_keys=True) + "\n").encode()
        )

    document = json.loads(_REFERENCE.read_bytes())
    document["unknown"] = None
    with pytest.raises(MastMagneticArchiveValidationError, match="keys differ"):
        decode_mast_complete_magnetic_archive_envelope(
            (json.dumps(document, separators=(",", ":"), sort_keys=True) + "\n").encode()
        )


def test_payload_source_and_event_identity_are_exact() -> None:
    """Producer, archive, shot, observation and unresolved-event identities are bound."""
    payload = _payload()
    payload["producer_project"] = "another-project"
    _assert_rejected(payload, "producer_project")

    payload = _payload()
    payload["source_archive"] = "another-archive"
    _assert_rejected(payload, "source_archive")

    payload = _payload()
    payload["source_ingestion_revision"] = "not-a-revision"
    _assert_rejected(payload, "source_ingestion_revision")

    payload = _payload()
    payload["source_ingestion_tree_state"] = "unknown"
    _assert_rejected(payload, "tree_state")

    payload = _payload()
    payload["observation_id"] = "mast-1-complete-magnetics-0000000000000000"
    _assert_rejected(payload, "shot_id")

    payload = _payload()
    payload["event_id"] = "invented"
    _assert_rejected(payload, "event_id")

    payload = _payload()
    payload["shot_id"] = 0
    _assert_rejected(payload, "positive")


def test_archive_identity_and_metadata_are_exact() -> None:
    """The envelope cannot switch group, Zarr version, endpoint, or metadata node."""
    payload = _payload()
    _object(payload, "archive")["group"] = "summary"
    _assert_rejected(payload, "archive.group")

    payload = _payload()
    _object(payload, "archive")["metadata_path"] = "raw/27707.zarr/summary/zarr.json"
    _assert_rejected(payload, "metadata path")

    payload = _payload()
    _object(payload, "archive")["root_url"] = "https://example.invalid/"
    _assert_rejected(payload, "root URL")

    payload = _payload()
    archive = _object(payload, "archive")
    _object(archive, "group_metadata")["node_type"] = "array"
    _assert_rejected(payload, "node_type")


def test_provenance_licence_limitations_and_parent_bounds_are_exact() -> None:
    """Source custody cannot omit its licence, limits, digest, or parent scope."""
    payload = _payload()
    provenance = _object(payload, "provenance")
    _object(provenance, "license")["spdx"] = "UNKNOWN"
    _assert_rejected(payload, "license.spdx")

    payload = _payload()
    _object(payload, "provenance")["limitations"] = []
    _assert_rejected(payload, "limitations")

    payload = _payload()
    provenance = _object(payload, "provenance")
    provenance["parent_object_count"] = 1
    _assert_rejected(payload, "parent provenance")

    payload = _payload()
    provenance = _object(payload, "provenance")
    provenance["magnetic_group_manifest_sha256"] = "0" * 64
    _assert_rejected(payload, "magnetic_group_manifest_sha256")


def test_provenance_objects_require_safe_sorted_bound_paths_and_urls() -> None:
    """Every object remains uniquely ordered inside the exact shot and host."""
    payload = _payload()
    objects = _array(_object(payload, "provenance"), "objects")
    objects[0]["path"] = cast(str, objects[1]["path"])
    objects[0]["source_url"] = cast(str, objects[1]["source_url"])
    _assert_rejected(payload, "paths")

    payload = _payload()
    objects = _array(_object(payload, "provenance"), "objects")
    objects[0]["source_url"] = "https://example.invalid/object"
    _assert_rejected(payload, "source URL")

    payload = _payload()
    provenance = _object(payload, "provenance")
    provenance["object_count"] = 1
    _assert_rejected(payload, "object_count")

    payload = _payload()
    _object(payload, "provenance")["objects"] = []
    _assert_rejected(payload, "objects are empty")


def test_array_shape_dimension_and_decoded_counts_are_exact() -> None:
    """Decoded array facts cannot diverge from the native Zarr metadata."""
    payload = _payload()
    arrays = _array(payload, "arrays")
    arrays[0]["decoded_value_count"] = 1
    _assert_rejected(payload, "value_count")

    payload = _payload()
    arrays = _array(payload, "arrays")
    arrays[0]["decoded_nonfinite_count"] = 10_000_000
    _assert_rejected(payload, "nonfinite count")

    payload = _payload()
    arrays = _array(payload, "arrays")
    arrays[0]["dimension_names"] = cast(JsonValue, ["missing-coordinate"])
    _assert_rejected(payload, "metadata.dimensions")

    payload = _payload()
    arrays = _array(payload, "arrays")
    arrays[1]["name"] = cast(str, arrays[0]["name"])
    _assert_rejected(payload, "sorted and unique")

    payload = _payload()
    payload["arrays"] = []
    _assert_rejected(payload, "arrays are empty")

    payload = _payload()
    arrays = _array(payload, "arrays")
    arrays[0]["dimension_names"] = cast(JsonValue, ["duplicate", "duplicate"])
    _assert_rejected(payload, "invalid dimensions")


def test_array_metadata_and_object_bindings_are_exact() -> None:
    """Array metadata and chunk paths must bind declared source objects."""
    payload = _payload()
    arrays = _array(payload, "arrays")
    arrays[0]["metadata_object_path"] = "raw/27707.zarr/magnetics/missing/zarr.json"
    _assert_rejected(payload, "metadata object")

    payload = _payload()
    arrays = _array(payload, "arrays")
    arrays[0]["metadata_object_sha256"] = "0" * 64
    _assert_rejected(payload, "metadata_digest")

    payload = _payload()
    arrays = _array(payload, "arrays")
    arrays[0]["data_object_paths"] = cast(JsonValue, ["missing"])
    _assert_rejected(payload, "unknown data object")

    payload = _payload()
    arrays = _array(payload, "arrays")
    _object(arrays[0], "zarr_metadata")["zarr_format"] = 2
    _assert_rejected(payload, "zarr_format")

    payload = _payload()
    arrays = _array(payload, "arrays")
    data_paths = cast(list[JsonValue], arrays[0]["data_object_paths"])
    data_paths.append(data_paths[0])
    _assert_rejected(payload, "data objects")


def test_clock_source_bounds_and_qualification_are_exact() -> None:
    """Native clocks must stay monotonic, source-bound and explicitly unresolved."""
    payload = _payload()
    clocks = _array(payload, "clocks")
    clocks[0]["sample_count"] = 1
    _assert_rejected(payload, "sample count")

    payload = _payload()
    clocks = _array(payload, "clocks")
    clocks[0]["last_value_s"] = clocks[0]["first_value_s"]
    _assert_rejected(payload, "bounds")

    payload = _payload()
    clocks = _array(payload, "clocks")
    clocks[0]["minimum_interval_s"] = 0.0
    _assert_rejected(payload, "interval")

    payload = _payload()
    clocks = _array(payload, "clocks")
    clocks[0]["mapping_evidence_claimed"] = True
    _assert_rejected(payload, "mapping")

    payload = _payload()
    payload["clocks"] = []
    _assert_rejected(payload, "no native archive clocks")

    payload = _payload()
    clocks = _array(payload, "clocks")
    clocks[0]["name"] = "ip"
    _assert_rejected(payload, "own coordinate")

    payload = _payload()
    clocks = _array(payload, "clocks")
    clocks[0]["name"] = "missing-clock"
    _assert_rejected(payload, "sorted or source-bound")

    payload = _payload()
    clocks = _array(payload, "clocks")
    clocks[0]["first_value_s"] = "not-a-number"
    _assert_rejected(payload, "finite number")

    payload = _payload()
    clocks = _array(payload, "clocks")
    clocks[0]["first_value_s"] = math.nan
    _assert_rejected(payload, "must be finite")


def test_completeness_qualification_authority_and_object_coverage_are_exact() -> None:
    """No completeness, qualification, authority, or source coverage drift is allowed."""
    payload = _payload()
    _object(payload, "completeness")["arrays_complete"] = False
    _assert_rejected(payload, "arrays_complete")

    payload = _payload()
    _object(payload, "qualification")["calibration_state"] = "calibrated"
    _assert_rejected(payload, "qualification")

    payload = _payload()
    _object(payload, "authority")["execution_permitted"] = True
    _assert_rejected(payload, "authority")

    payload = _payload()
    objects = _array(_object(payload, "provenance"), "objects")
    removed = objects.pop()
    provenance = _object(payload, "provenance")
    provenance["object_count"] = len(objects)
    provenance["total_bytes"] = cast(int, provenance["total_bytes"]) - cast(
        int, removed["size_bytes"]
    )
    manifest = "".join(
        f"{item['sha256']}:{item['size_bytes']}:{item['path']}\n" for item in objects
    ).encode()
    provenance["magnetic_group_manifest_sha256"] = hashlib.sha256(manifest).hexdigest()
    _assert_rejected(payload, "coverage mismatch")


def test_dimension_coordinate_and_clock_bindings_are_exact() -> None:
    """Every dimension and clock binding is validated against its source coordinate."""
    payload = _payload()
    arrays = _array(payload, "arrays")
    arrays[0]["dimension_names"] = cast(JsonValue, ["missing-coordinate"])
    _object(arrays[0], "zarr_metadata")["dimension_names"] = cast(JsonValue, ["missing-coordinate"])
    _assert_rejected(payload, "no source coordinate")

    payload = _payload()
    arrays = _array(payload, "arrays")
    arrays[0]["shape"] = cast(JsonValue, [6])
    arrays[0]["decoded_value_count"] = 6
    _object(arrays[0], "zarr_metadata")["shape"] = cast(JsonValue, [6])
    _assert_rejected(payload, "cardinality mismatch")

    payload = _payload()
    arrays = _array(payload, "arrays")
    ip = next(array for array in arrays if array["name"] == "ip")
    ip["clock_dimensions"] = []
    _assert_rejected(payload, "clock binding")


def test_scalar_type_guards_and_producer_inventory_are_exact() -> None:
    """Primitive JSON types and producer records fail closed without coercion."""
    payload = _payload()
    payload["arrays"] = "not-an-array"
    _assert_rejected(payload, "JSON array")

    payload = _payload()
    payload["observation_id"] = ""
    _assert_rejected(payload, "non-empty string")

    payload = _payload()
    arrays = _array(payload, "arrays")
    arrays[0]["dimension_names"] = cast(JsonValue, [""])
    _assert_rejected(payload, "non-empty strings")

    payload = _payload()
    payload["shot_id"] = -1
    _assert_rejected(payload, "non-negative integer")

    payload = _payload()
    payload["producer_artifacts"] = []
    _assert_rejected(payload, "inventory is empty")

    payload = _payload()
    artifacts = _array(payload, "producer_artifacts")
    artifacts[1]["path"] = cast(str, artifacts[0]["path"])
    _assert_rejected(payload, "sorted/unique")
