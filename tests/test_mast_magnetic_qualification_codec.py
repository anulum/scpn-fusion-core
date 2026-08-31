# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FAIR-MAST magnetic diagnostic qualification codec tests
"""Canonical transport and semantic refusal tests over the real witness."""

from __future__ import annotations

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
