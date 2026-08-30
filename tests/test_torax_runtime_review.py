# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TORAX Deterministic Review Envelope Tests
"""Portable consumer tests for the non-actuating TORAX review envelope."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from scpn_fusion.integrations.torax import (
    COUPLED_TRANSPORT_SOURCE_SCHEMA,
    TORAX_OUTCOME_SCHEMA,
    ToraxReviewEnvelope,
    review_envelope_from_bytes,
    review_envelope_sha256,
    review_envelope_to_bytes,
)
from scpn_fusion.integrations.torax import review as review_module
from scpn_fusion.integrations.torax.serialization import load_json_object

ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "validation/reference_data/torax/torax_runtime_review_envelope_v1.json"


def test_tracked_review_envelope_is_deterministic_and_review_only() -> None:
    """SPO receives canonical bytes without runtime paths, wall clocks, or inferred scalars."""
    fixture_bytes = REVIEW_PATH.read_bytes()
    envelope = review_envelope_from_bytes(
        fixture_bytes,
        expected_sha256=hashlib.sha256(fixture_bytes).hexdigest(),
    )
    raw = envelope.to_dict()
    assert envelope.source_schema == TORAX_OUTCOME_SCHEMA
    assert envelope.model_intersection_schema == COUPLED_TRANSPORT_SOURCE_SCHEMA
    assert len(envelope.source_revision) == 40
    assert review_envelope_to_bytes(envelope) == fixture_bytes
    assert review_envelope_sha256(envelope) == hashlib.sha256(fixture_bytes).hexdigest()
    text = fixture_bytes.decode("utf-8")
    for excluded in (
        "q95",
        "li3",
        "beta_N",
        "W_thermal_total",
        '"regime"',
        '"phase"',
        "started_at_utc",
        "finished_at_utc",
        '"platform"',
        "sidecar_path",
        "manifest_path",
    ):
        assert excluded not in text
    payload = raw["payload"]
    assert isinstance(payload, dict)
    uncertainty = payload["uncertainty"]
    validity = payload["validity"]
    assert isinstance(uncertainty, dict)
    assert isinstance(validity, dict)
    assert uncertainty["kind"] == "numerical_refinement"
    uncertainty_observables = uncertainty["observables"]
    observables = payload["observables"]
    assert isinstance(uncertainty_observables, dict)
    assert isinstance(observables, dict)
    for category in ("profiles", "source_totals", "state_budgets"):
        assert set(uncertainty_observables[category]) == set(observables[category])
        for name, metric in uncertainty_observables[category].items():
            assert metric["unit"] == observables[category][name]["unit"]
            assert metric["absolute_rms"] >= 0.0
            assert metric["relative_l2"] >= 0.0
    assert validity == {
        "authority": "review_only_non_actuating",
        "ood": False,
        "quality": "frozen_model_intersection_reference",
        "state": "VALID",
    }


def test_review_envelope_rejects_unknown_fields_tampering_and_nondeterministic_custody() -> None:
    """Schema drift, payload mutation, and runtime-only metadata all fail closed."""
    raw = load_json_object(REVIEW_PATH)
    unknown = copy.deepcopy(raw)
    unknown["extra"] = True
    with pytest.raises(ValueError, match=r"unknown=\['extra'\]"):
        ToraxReviewEnvelope.from_dict(unknown)

    tampered = copy.deepcopy(raw)
    payload = tampered["payload"]
    assert isinstance(payload, dict)
    completion = payload["completion"]
    assert isinstance(completion, dict)
    completion["reached_final_ns"] = 1
    with pytest.raises(ValueError, match="payload_sha256"):
        ToraxReviewEnvelope.from_dict(tampered)

    nondeterministic = copy.deepcopy(raw)
    provenance = nondeterministic["provenance"]
    assert isinstance(provenance, dict)
    provenance["platform"] = "host-specific"
    with pytest.raises(ValueError, match=r"unknown=\['platform'\]"):
        ToraxReviewEnvelope.from_dict(nondeterministic)


def test_review_envelope_byte_codec_refuses_alternate_or_untrusted_encodings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public admission codec refuses duplicates, noncanonical bytes, size, and digest drift."""
    payload = REVIEW_PATH.read_bytes()
    with pytest.raises(ValueError, match="canonical JSON"):
        review_envelope_from_bytes(payload + b"\n")
    duplicate = b'{"schema":"duplicate",' + payload[1:]
    with pytest.raises(ValueError, match="duplicate JSON object key: schema"):
        review_envelope_from_bytes(duplicate)
    with pytest.raises(ValueError, match="byte digest mismatch"):
        review_envelope_from_bytes(payload, expected_sha256="0" * 64)
    monkeypatch.setattr(review_module, "MAX_REVIEW_ENVELOPE_BYTES", len(payload) - 1)
    with pytest.raises(ValueError, match="maximum byte size"):
        review_envelope_from_bytes(payload)
