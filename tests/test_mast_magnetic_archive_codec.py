# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — complete FAIR-MAST magnetic archive codec tests
"""Contract tests for the authentic complete magnetic archive envelope."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import cast

import jsonschema  # type: ignore[import-untyped]
import pytest

from scpn_fusion.io import (
    MastMagneticArchiveValidationError,
    decode_mast_complete_magnetic_archive_envelope,
    encode_mast_complete_magnetic_archive_envelope,
)
from scpn_fusion.io.mast_magnetic_archive_codec import JsonObject, JsonValue

_REFERENCE = Path("validation/reference_data/mast/disruption_shots/MAGNETIC_ARCHIVE_ENVELOPE.json")
_SCHEMA = Path("schemas/mast_complete_magnetic_archive_envelope.schema.json")


def test_complete_reference_envelope_covers_every_source_object_array_and_clock() -> None:
    """The tracked real-shot proof must describe the whole native group."""
    envelope = decode_mast_complete_magnetic_archive_envelope(_REFERENCE.read_bytes())
    document = envelope.document
    jsonschema.Draft202012Validator(json.loads(_SCHEMA.read_text())).validate(document)
    payload = envelope.payload
    provenance = cast(JsonObject, payload["provenance"])
    completeness = cast(JsonObject, payload["completeness"])
    arrays = cast(list[JsonObject], payload["arrays"])
    clocks = cast(list[JsonObject], payload["clocks"])

    assert provenance["object_count"] == 253
    assert provenance["total_bytes"] == 12_916_991
    assert payload["source_ingestion_tree_state"] == "dirty"
    assert completeness == {
        "array_count": 72,
        "arrays_complete": True,
        "clock_count": 4,
        "objects_complete": True,
        "source_decoded": True,
    }
    assert [clock["name"] for clock in clocks] == [
        "time",
        "time_mirnov",
        "time_omaha",
        "time_saddle",
    ]
    assert {array["name"] for array in arrays} >= {
        "b_field_pol_probe_cc_field",
        "b_field_tor_probe_omaha_voltage",
        "flux_loop_flux",
        "ip",
        "shot_id",
    }
    shapes = {cast(str, array["name"]): array["shape"] for array in arrays}
    assert (
        shapes["b_field_pol_probe_cc_channel"],
        shapes["b_field_pol_probe_cc_geometry_channel"],
    ) == ([5], [40])
    assert (
        shapes["b_field_pol_probe_omv_channel"],
        shapes["b_field_pol_probe_omv_geometry_channel"],
    ) == ([3], [21])
    assert (
        shapes["b_field_tor_probe_cc_channel"],
        shapes["b_field_tor_probe_cc_geometry_channel"],
    ) == ([3], [36])
    assert payload["qualification"] == {
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
        "source_clock_relationship_state": "unresolved",
        "synthetic": False,
        "uncertainty_state": "unknown",
        "validity_state": "unknown",
    }


def test_complete_reference_round_trips_exact_canonical_bytes() -> None:
    """Decode and re-encode preserve the byte-level evidence identity."""
    source = _REFERENCE.read_bytes()
    decoded = decode_mast_complete_magnetic_archive_envelope(source)
    assert encode_mast_complete_magnetic_archive_envelope(decoded.payload).to_bytes() == source


def test_codec_rejects_authority_escalation_and_noncanonical_transport() -> None:
    """Neither semantic authority drift nor transport ambiguity is accepted."""
    source = _REFERENCE.read_bytes()
    payload = deepcopy(decode_mast_complete_magnetic_archive_envelope(source).payload)
    authority = cast(JsonObject, payload["authority"])
    authority["actionable"] = cast(JsonValue, True)
    with pytest.raises(MastMagneticArchiveValidationError, match="authority"):
        encode_mast_complete_magnetic_archive_envelope(payload)

    with pytest.raises(MastMagneticArchiveValidationError, match="canonical"):
        decode_mast_complete_magnetic_archive_envelope(source.rstrip() + b"  \n")


def test_codec_rejects_archive_path_traversal() -> None:
    """A signed manifest cannot escape its shot archive root."""
    payload = deepcopy(
        decode_mast_complete_magnetic_archive_envelope(_REFERENCE.read_bytes()).payload
    )
    provenance = cast(JsonObject, payload["provenance"])
    objects = cast(list[JsonObject], provenance["objects"])
    objects[0]["path"] = "raw/27707.zarr/magnetics/../outside"
    objects[0]["source_url"] = (
        "https://s3.echo.stfc.ac.uk/mast/level2/shots/27707.zarr/magnetics/../outside"
    )
    with pytest.raises(MastMagneticArchiveValidationError, match="paths"):
        encode_mast_complete_magnetic_archive_envelope(payload)
