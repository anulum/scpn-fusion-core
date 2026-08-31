# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FAIR-MAST magnetic diagnostic qualification validation
"""Independent schema and scientific-fact checks over the authentic witness."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import jsonschema  # type: ignore[import-untyped]
import pytest

from scpn_fusion.io import (
    MastMagneticDiagnosticQualificationError,
    decode_mast_complete_magnetic_archive_envelope,
    decode_mast_magnetic_diagnostic_qualification,
    encode_mast_magnetic_diagnostic_qualification,
)
from scpn_fusion.io.mast_magnetic_archive_codec import JsonObject, JsonValue

_ARCHIVE = Path("validation/reference_data/mast/disruption_shots/MAGNETIC_ARCHIVE_ENVELOPE.json")
_QUALIFICATION = Path(
    "validation/reference_data/mast/disruption_shots/MAGNETIC_DIAGNOSTIC_QUALIFICATION.json"
)
_SCHEMA = Path("schemas/mast_magnetic_diagnostic_qualification.schema.json")


def _payload() -> JsonObject:
    return decode_mast_magnetic_diagnostic_qualification(_QUALIFICATION.read_bytes()).payload


def _objects(payload: JsonObject, key: str) -> list[JsonObject]:
    return cast(list[JsonObject], payload[key])


def _assert_rejected(payload: JsonObject, message: str) -> None:
    with pytest.raises(MastMagneticDiagnosticQualificationError, match=message):
        encode_mast_magnetic_diagnostic_qualification(payload)


def test_real_qualification_validates_against_independent_json_schema() -> None:
    """The exact source-derived witness satisfies its published strict schema."""
    document = json.loads(_QUALIFICATION.read_bytes())
    schema = json.loads(_SCHEMA.read_bytes())
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(
        document
    )


def test_qualification_is_cryptographically_bound_to_complete_archive() -> None:
    """Qualification identity names the exact complete source envelope and observation."""
    archive = decode_mast_complete_magnetic_archive_envelope(_ARCHIVE.read_bytes())
    qualification = decode_mast_magnetic_diagnostic_qualification(_QUALIFICATION.read_bytes())
    assert len(qualification.sha256) == 64
    document = qualification.document
    document["schema"] = "mutated-copy"
    assert qualification.document["schema"] != "mutated-copy"
    assert qualification.payload["archive_envelope_sha256"] == archive.sha256
    assert qualification.payload["archive_observation_id"] == archive.payload["observation_id"]
    assert qualification.payload["shot_id"] == archive.payload["shot_id"] == 27707


def test_level2_grids_and_applied_transforms_are_explicit() -> None:
    """Archive grids and applied transforms are recorded without claiming raw clocks."""
    payload = decode_mast_magnetic_diagnostic_qualification(_QUALIFICATION.read_bytes()).payload
    clocks = {
        cast(str, item["name"]): item for item in cast(list[JsonObject], payload["clock_evidence"])
    }
    assert {
        name: (item["sample_count"], item["start_s"], item["step_s"])
        for name, item in clocks.items()
    } == {
        "time": (2152, -0.1, 0.0002),
        "time_mirnov": (215051, -0.1, 0.000002),
        "time_omaha": (860201, -0.1, 0.0000005),
        "time_saddle": (21506, -0.1, 0.00002),
    }
    assert all(item["grid_origin"] == "level2_interpolation" for item in clocks.values())
    assert all(item["source_clock_relation_claimed"] is False for item in clocks.values())

    measurements = {
        cast(str, item["array_name"]): item
        for item in cast(list[JsonObject], payload["measurement_evidence"])
    }
    assert measurements["b_field_pol_probe_cc_field"]["applied_scale"] == 0.000002
    assert measurements["b_field_tor_probe_saddle_voltage"]["applied_background_sample_range"] == [
        0,
        10,
    ]
    assert all(item["uncertainty_supplied"] is False for item in measurements.values())


def test_every_measurement_and_channel_has_empirical_quality() -> None:
    """No measurement or configured channel is omitted from exact quality accounting."""
    payload = decode_mast_magnetic_diagnostic_qualification(_QUALIFICATION.read_bytes()).payload
    measurements = cast(list[JsonObject], payload["measurement_evidence"])
    for measurement in measurements:
        quality = cast(JsonObject, measurement["empirical_quality"])
        assert (
            cast(int, quality["finite_count"])
            + cast(int, quality["nan_count"])
            + cast(int, quality["infinite_count"])
            == quality["sample_count"]
        )
        channels = cast(list[str], measurement["archive_channel_ids"])
        channel_quality = cast(list[JsonObject], measurement["channel_quality"])
        assert [record["archive_channel_id"] for record in channel_quality] == channels


def test_source_identity_mapping_revision_and_licence_are_exact() -> None:
    """Shot observation, upstream revision, tree state, URL and licence cannot drift."""
    payload = _payload()
    payload["archive_observation_id"] = "mast-1-complete-magnetics-0000000000000000"
    _assert_rejected(payload, "observation_id")

    payload = _payload()
    mapping = cast(JsonObject, payload["ingestion_mapping"])
    mapping["source_tree_state"] = "unknown"
    _assert_rejected(payload, "source_tree_state")

    payload = _payload()
    mapping = cast(JsonObject, payload["ingestion_mapping"])
    mapping["source_revision"] = "not-a-revision"
    _assert_rejected(payload, "source_revision")

    payload = _payload()
    mapping = cast(JsonObject, payload["ingestion_mapping"])
    mapping["mapping_url"] = "https://example.invalid/mast.yml"
    _assert_rejected(payload, "mapping URL")

    payload = _payload()
    cast(JsonObject, payload["ingestion_mapping"])["dataset_license_url"] = "http://invalid"
    _assert_rejected(payload, "HTTPS")


def test_array_inventory_rejects_omissions_order_role_and_dimension_drift() -> None:
    """All 72 source arrays retain unique order, valid roles, shapes and clock dimensions."""
    payload = _payload()
    payload["array_inventory"] = cast(JsonValue, _objects(payload, "array_inventory")[:-1])
    _assert_rejected(payload, "72 arrays")

    payload = _payload()
    arrays = _objects(payload, "array_inventory")
    arrays[1]["name"] = arrays[0]["name"]
    _assert_rejected(payload, "unique and sorted")

    payload = _payload()
    _objects(payload, "array_inventory")[0]["role"] = "invented"
    _assert_rejected(payload, "unsupported role")

    payload = _payload()
    _objects(payload, "array_inventory")[0]["shape"] = cast(JsonValue, [1, 2])
    _assert_rejected(payload, "shape or clock dimensions")

    payload = _payload()
    _objects(payload, "array_inventory")[0]["clock_dimensions"] = cast(JsonValue, ["missing_clock"])
    _assert_rejected(payload, "shape or clock dimensions")


def test_clock_evidence_rejects_missing_duplicate_unbound_and_wrong_shape() -> None:
    """Exactly four ordered grids must bind clock arrays with matching sample counts."""
    payload = _payload()
    payload["clock_evidence"] = cast(JsonValue, _objects(payload, "clock_evidence")[:-1])
    _assert_rejected(payload, "four grids")

    payload = _payload()
    clocks = _objects(payload, "clock_evidence")
    clocks[1]["name"] = clocks[0]["name"]
    _assert_rejected(payload, "unique and sorted")

    payload = _payload()
    clocks = _objects(payload, "clock_evidence")
    clock_array = next(
        item for item in _objects(payload, "array_inventory") if item["name"] == clocks[0]["name"]
    )
    clock_array["role"] = "geometry"
    _assert_rejected(payload, "not source-bound")

    payload = _payload()
    _objects(payload, "clock_evidence")[0]["sample_count"] = 1
    _assert_rejected(payload, "sample count mismatch")


def test_clock_evidence_rejects_nonpositive_steps_and_inconsistent_bounds() -> None:
    """Archive-grid bounds must reproduce start plus the exact Level-2 step sequence."""
    payload = _payload()
    _objects(payload, "clock_evidence")[0]["step_s"] = 0.0
    _assert_rejected(payload, "step is not positive")

    payload = _payload()
    _objects(payload, "clock_evidence")[0]["first_value_s"] = 0.0
    _assert_rejected(payload, "bounds do not reproduce")

    payload = _payload()
    _objects(payload, "clock_evidence")[0]["last_value_s"] = 0.0
    _assert_rejected(payload, "bounds do not reproduce")


def test_measurement_identity_clock_and_channels_are_complete_and_unique() -> None:
    """Every named measurement binds a measurement array, known clock and unique channels."""
    payload = _payload()
    measurements = _objects(payload, "measurement_evidence")
    measurements[1]["array_name"] = measurements[0]["array_name"]
    _assert_rejected(payload, "unique and sorted")

    payload = _payload()
    first = _objects(payload, "measurement_evidence")[0]
    array = next(
        item for item in _objects(payload, "array_inventory") if item["name"] == first["array_name"]
    )
    array["role"] = "geometry"
    _assert_rejected(payload, "not source-bound")

    payload = _payload()
    _objects(payload, "measurement_evidence")[0]["clock_name"] = "unknown"
    _assert_rejected(payload, "clock is unknown")

    payload = _payload()
    first = _objects(payload, "measurement_evidence")[0]
    first["configured_source_channels"] = cast(
        JsonValue, cast(list[str], first["configured_source_channels"])[:-1]
    )
    _assert_rejected(payload, "channel mismatch")

    payload = _payload()
    first = _objects(payload, "measurement_evidence")[0]
    channels = cast(list[str], first["archive_channel_ids"])
    channels[1] = channels[0]
    _assert_rejected(payload, "channels are not unique")

    payload = _payload()
    first = _objects(payload, "measurement_evidence")[0]
    first["channel_quality"] = cast(JsonValue, _objects(first, "channel_quality")[:-1])
    _assert_rejected(payload, "channel quality count mismatch")


def test_measurement_validity_scale_and_background_ranges_fail_closed() -> None:
    """Source shot ranges, applied scales and background sample slices stay physically valid."""
    payload = _payload()
    first = _objects(payload, "measurement_evidence")[0]
    first["source_shot_min"] = 30_000
    first["source_shot_max"] = 20_000
    _assert_rejected(payload, "source shot range is invalid")

    payload = _payload()
    _objects(payload, "measurement_evidence")[0]["source_shot_min"] = 30_000
    _assert_rejected(payload, "shot range mismatch")

    payload = _payload()
    first = _objects(payload, "measurement_evidence")[0]
    first["source_shot_min"] = 1
    first["source_shot_max"] = 20_000
    _assert_rejected(payload, "shot range mismatch")

    payload = _payload()
    _objects(payload, "measurement_evidence")[0]["applied_scale"] = 0.0
    _assert_rejected(payload, "scale is not positive")

    payload = _payload()
    _objects(payload, "measurement_evidence")[0]["applied_background_sample_range"] = cast(
        JsonValue, [10, 0]
    )
    _assert_rejected(payload, "background range is invalid")


def test_measurement_aggregate_quality_must_equal_channel_totals() -> None:
    """Aggregate finite, nonfinite, sample and zero counts reproduce channel evidence."""
    for field in ("finite_count", "infinite_count", "nan_count", "sample_count", "zero_count"):
        payload = _payload()
        first = _objects(payload, "measurement_evidence")[0]
        quality = cast(JsonObject, first["empirical_quality"])
        quality[field] = cast(int, quality[field]) + 1
        if field in {"finite_count", "infinite_count", "nan_count", "sample_count"}:
            message = "quality count|NaN fraction|aggregate"
        else:
            message = "aggregate"
        _assert_rejected(payload, message)


def test_quality_statistics_reject_fraction_bounds_and_level_spacing_drift() -> None:
    """NaN fractions, finite bounds and exact positive level spacing remain self-consistent."""
    payload = _payload()
    quality = cast(JsonObject, _objects(payload, "measurement_evidence")[0]["empirical_quality"])
    quality["nan_fraction"] = 0.5
    _assert_rejected(payload, "NaN fraction")

    payload = _payload()
    quality = cast(JsonObject, _objects(payload, "measurement_evidence")[0]["empirical_quality"])
    quality["unique_finite_value_count"] = cast(int, quality["finite_count"]) + 1
    _assert_rejected(payload, "quality bounds")

    payload = _payload()
    quality = cast(JsonObject, _objects(payload, "measurement_evidence")[0]["empirical_quality"])
    quality["minimum_positive_level_spacing_hex"] = "not-hex"
    _assert_rejected(payload, "level spacing is invalid")

    payload = _payload()
    quality = cast(JsonObject, _objects(payload, "measurement_evidence")[0]["empirical_quality"])
    quality["minimum_positive_level_spacing_hex"] = "0x0.0p+0"
    _assert_rejected(payload, "level spacing is not positive")

    payload = _payload()
    quality = cast(JsonObject, _objects(payload, "measurement_evidence")[0]["empirical_quality"])
    quality["minimum_positive_level_spacing_hex"] = "inf"
    _assert_rejected(payload, "level spacing is not positive")


def test_measurement_set_cannot_omit_a_diagnostic() -> None:
    """The complete magnetic qualification cannot silently omit one measurement family."""
    payload = _payload()
    measurements = _objects(payload, "measurement_evidence")
    omitted = cast(str, measurements[-1]["array_name"])
    payload["measurement_evidence"] = cast(JsonValue, measurements[:-1])
    payload["channel_geometry_evidence"] = cast(
        JsonValue,
        [
            item
            for item in _objects(payload, "channel_geometry_evidence")
            if item["measurement_array"] != omitted
        ],
    )
    _assert_rejected(payload, "complete magnetic group")


def test_channel_geometry_records_reject_order_unbound_geometry_and_methods() -> None:
    """Every channel mapping stays ordered, source-bound and identifier-only."""
    payload = _payload()
    mappings = _objects(payload, "channel_geometry_evidence")
    mappings[1]["archive_channel_id"] = mappings[0]["archive_channel_id"]
    mappings[1]["measurement_array"] = mappings[0]["measurement_array"]
    _assert_rejected(payload, "not source-bound")

    payload = _payload()
    mappings = _objects(payload, "channel_geometry_evidence")
    available = next(item for item in mappings if item["geometry_coordinate"] is not None)
    available["geometry_coordinate"] = "missing_geometry"
    _assert_rejected(payload, "geometry coordinate is not bound")

    payload = _payload()
    mappings = _objects(payload, "channel_geometry_evidence")
    available = next(item for item in mappings if item["geometry_coordinate"] is not None)
    available["identifier_match_method"] = "positional"
    _assert_rejected(payload, "mapping method is unsupported")


def test_channel_geometry_records_cannot_be_empty_or_incomplete() -> None:
    """The identifier ledger covers every configured archived signal channel."""
    payload = _payload()
    payload["channel_geometry_evidence"] = []
    _assert_rejected(payload, "evidence is empty")

    payload = _payload()
    payload["channel_geometry_evidence"] = cast(
        JsonValue, _objects(payload, "channel_geometry_evidence")[:-1]
    )
    _assert_rejected(payload, "does not cover every measurement channel")


def test_external_limitation_and_completeness_are_exact() -> None:
    """The scoped upstream issue and all evidence cardinalities remain explicit."""
    payload = _payload()
    payload["external_limitations"] = []
    _assert_rejected(payload, "one scoped external limitation")

    payload = _payload()
    cast(JsonObject, payload["completeness"])["archive_array_count"] = 1
    _assert_rejected(payload, "archive_array_count")


def test_transport_rejects_invalid_utf8_json_duplicate_keys_and_nonobjects() -> None:
    """Only unambiguous canonical UTF-8 JSON objects can enter semantic validation."""
    with pytest.raises(MastMagneticDiagnosticQualificationError, match="UTF-8"):
        decode_mast_magnetic_diagnostic_qualification(b"\xff")
    with pytest.raises(MastMagneticDiagnosticQualificationError, match="qualification JSON"):
        decode_mast_magnetic_diagnostic_qualification(b"{")
    with pytest.raises(MastMagneticDiagnosticQualificationError, match="qualification JSON"):
        decode_mast_magnetic_diagnostic_qualification(b'{"schema":1,"schema":2}\n')
    with pytest.raises(MastMagneticDiagnosticQualificationError, match="not an object"):
        decode_mast_magnetic_diagnostic_qualification(b"[]\n")


def test_public_encoder_rejects_nonfinite_and_oversized_documents() -> None:
    """Canonical qualification transport is finite and bounded against oversized evidence."""
    payload = _payload()
    _objects(payload, "measurement_evidence")[0]["applied_scale"] = math.nan
    _assert_rejected(payload, "finite number")

    payload = _payload()
    _objects(payload, "measurement_evidence")[0]["source_name"] = "x" * (8 * 1024 * 1024)
    _assert_rejected(payload, "too large")


def test_public_encoder_rejects_malformed_scalar_and_collection_types() -> None:
    """Wrong object, array, string, integer and shape element types fail before use."""
    payload = _payload()
    payload["ingestion_mapping"] = []
    _assert_rejected(payload, "not an object")

    payload = _payload()
    payload["array_inventory"] = cast(JsonValue, {})
    _assert_rejected(payload, "not an array")

    payload = _payload()
    payload["archive_observation_id"] = ""
    _assert_rejected(payload, "nonempty string")

    payload = _payload()
    _objects(payload, "array_inventory")[0]["dimension_names"] = cast(JsonValue, [1])
    _assert_rejected(payload, "string array")

    payload = _payload()
    payload["shot_id"] = 0
    _assert_rejected(payload, "positive integer")

    payload = _payload()
    _objects(payload, "array_inventory")[0]["shape"] = cast(JsonValue, [-1])
    _assert_rejected(payload, "nonnegative integer")
