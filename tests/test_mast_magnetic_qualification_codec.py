# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FAIR-MAST magnetic diagnostic qualification codec tests
"""Canonical transport and semantic refusal tests over the real witness."""

from __future__ import annotations

import json
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from scpn_fusion.io import (
    MastMagneticDiagnosticQualificationError,
    decode_mast_magnetic_diagnostic_qualification,
    encode_mast_magnetic_diagnostic_qualification,
)
from scpn_fusion.io.mast_magnetic_archive_codec import JsonObject, JsonValue

_REFERENCE = Path(
    "validation/reference_data/mast/disruption_shots/MAGNETIC_DIAGNOSTIC_QUALIFICATION.json"
)


def _payload() -> JsonObject:
    return decode_mast_magnetic_diagnostic_qualification(_REFERENCE.read_bytes()).payload


def test_qualification_round_trips_exact_canonical_bytes() -> None:
    """The real qualification witness has one byte-level transport identity."""
    source = _REFERENCE.read_bytes()
    decoded = decode_mast_magnetic_diagnostic_qualification(source)
    assert encode_mast_magnetic_diagnostic_qualification(decoded.payload).to_bytes() == source


def test_codec_rejects_phase_actuation_and_physical_mapping_escalation() -> None:
    """Evidence cannot silently become a classifier, actuator or physical geometry join."""
    payload = _payload()
    cast(JsonObject, payload["authority"])["phase_inference_performed"] = True
    with pytest.raises(MastMagneticDiagnosticQualificationError, match="authority"):
        encode_mast_magnetic_diagnostic_qualification(payload)

    payload = _payload()
    mappings = cast(list[JsonObject], payload["channel_geometry_evidence"])
    mappings[0]["physical_mapping_claimed"] = True
    with pytest.raises(MastMagneticDiagnosticQualificationError, match="physical mapping"):
        encode_mast_magnetic_diagnostic_qualification(payload)


def test_codec_rejects_invented_clock_relation_and_quality_drift() -> None:
    """Source-clock claims and measured quality counts remain evidence-bound."""
    payload = _payload()
    clocks = cast(list[JsonObject], payload["clock_evidence"])
    clocks[0]["source_clock_relation_claimed"] = True
    with pytest.raises(MastMagneticDiagnosticQualificationError, match="source relation"):
        encode_mast_magnetic_diagnostic_qualification(payload)

    payload = _payload()
    measurements = cast(list[JsonObject], payload["measurement_evidence"])
    quality = cast(JsonObject, measurements[0]["empirical_quality"])
    quality["nan_count"] = cast(int, quality["nan_count"]) + 1
    with pytest.raises(MastMagneticDiagnosticQualificationError, match="quality count"):
        encode_mast_magnetic_diagnostic_qualification(payload)


def test_codec_rejects_noncanonical_and_oversized_transport() -> None:
    """Ambiguous or unbounded transports are rejected before qualification use."""
    source = _REFERENCE.read_bytes()
    with pytest.raises(MastMagneticDiagnosticQualificationError, match="canonical"):
        decode_mast_magnetic_diagnostic_qualification(source.rstrip() + b"  \n")
    with pytest.raises(MastMagneticDiagnosticQualificationError, match="too large"):
        decode_mast_magnetic_diagnostic_qualification(b" " * (8 * 1024 * 1024 + 1))

    payload = deepcopy(_payload())
    payload["unknown"] = cast(JsonValue, None)
    with pytest.raises(MastMagneticDiagnosticQualificationError, match="keys differ"):
        encode_mast_magnetic_diagnostic_qualification(payload)


Path_ = tuple[str | int, ...]


def _leaf_paths(value: object, path: Path_ = ()) -> Iterator[Path_]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaf_paths(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaf_paths(item, (*path, index))
    else:
        yield path


def _value_at(value: object, path: Path_) -> object:
    for step in path:
        if isinstance(step, int):
            value = cast(list[object], value)[step]
        else:
            value = cast(dict[str, object], value)[step]
    return value


def _replaced(payload: object, path: Path_, value: object) -> JsonObject:
    copy = deepcopy(payload)
    parent = _value_at(copy, path[:-1])
    if isinstance(path[-1], int):
        cast(list[object], parent)[path[-1]] = value
    else:
        cast(dict[str, object], parent)[path[-1]] = value
    return cast(JsonObject, copy)


def _one_path_per_field(payload: object) -> list[Path_]:
    """Keep the first leaf of every field pattern; list items share their field's validator."""
    seen: dict[tuple[str, ...], Path_] = {}
    for path in _leaf_paths(payload):
        pattern = tuple("*" if isinstance(step, int) else step for step in path)
        seen.setdefault(pattern, path)
    return list(seen.values())


def test_validation_refuses_every_nonfinite_or_non_json_field_before_encoding() -> None:
    """The encoder relies on field validation: no field may carry NaN, infinity or a non-JSON value."""
    payload = _payload()
    numeric = 0
    for path in _one_path_per_field(payload):
        candidates: list[object] = [{"not", "json"}]
        leaf = _value_at(payload, path)
        if isinstance(leaf, int | float) and not isinstance(leaf, bool):
            numeric += 1
            candidates += [float("nan"), float("inf"), float("-inf")]
        for bad in candidates:
            with pytest.raises(MastMagneticDiagnosticQualificationError):
                encode_mast_magnetic_diagnostic_qualification(_replaced(payload, path, bad))
    assert numeric >= 20


def test_decoder_refuses_a_nan_that_json_text_can_carry() -> None:
    """``json.loads`` accepts the NaN literal; the decoder must still refuse it."""
    document = cast(dict[str, object], json.loads(_REFERENCE.read_bytes()))
    path = next(
        path
        for path in _leaf_paths(document["payload"])
        if isinstance(_value_at(document["payload"], path), float)
    )
    corrupted = _replaced(document, ("payload", *path), float("nan"))
    data = (json.dumps(corrupted, sort_keys=True, separators=(",", ":")) + "\n").encode()
    assert b"NaN" in data

    with pytest.raises(MastMagneticDiagnosticQualificationError):
        decode_mast_magnetic_diagnostic_qualification(data)
