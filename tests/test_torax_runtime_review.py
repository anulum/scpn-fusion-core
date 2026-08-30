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
import inspect
from pathlib import Path

import pytest

from scpn_fusion.integrations.torax import (
    COUPLED_TRANSPORT_SOURCE_SCHEMA,
    TORAX_OUTCOME_SCHEMA,
    ToraxReviewEnvelope,
    build_review_envelope,
    review_envelope_from_bytes,
    review_envelope_sha256,
    review_envelope_to_bytes,
)
from scpn_fusion.integrations.torax import review as review_module
from scpn_fusion.integrations.torax.serialization import canonical_sha256, load_json_object

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
            assert metric["absolute_rms_difference"] >= 0.0
            assert 0.0 <= metric["relative_l2"] <= 0.02
    calibration = observables["profiles"]["electron_density"]["calibration"]
    assert calibration == {
        "basis": "simulation_declared_units",
        "calibrated_at_ns": 0,
        "calibration_id": "fusion.torax.simulation_declared_units.v1",
        "empirical": False,
        "transfer": "identity",
        "transfer_function_id": "fusion.torax.identity_projection.v1",
    }
    reactor = payload["reactor"]
    assert reactor["drivers"] == ["external_magnetic_coils", "plasma_current"]
    assert reactor["operating_point"]["fuel_class_basis"] == (
        "deuterium_only_input_no_fusion_power_or_burn_model"
    )
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
    observables = payload["observables"]
    assert isinstance(observables, dict)
    observables["source_totals"]["electron_heat"]["samples"][0] += 1.0
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


def test_review_builder_derives_uncertainty_internally() -> None:
    """No caller-controlled refinement metric can enter the public producer API."""
    assert "refinement_metrics" not in inspect.signature(build_review_envelope).parameters


def test_review_decoder_recursively_closes_u1_names_units_shapes_and_semantics() -> None:
    """A recomputed payload digest cannot admit semantic or numeric schema drift."""
    source = load_json_object(REVIEW_PATH)

    def reseal(raw: dict[str, object]) -> dict[str, object]:
        payload = raw["payload"]
        assert isinstance(payload, dict)
        raw["payload_sha256"] = canonical_sha256(payload)
        return raw

    missing = copy.deepcopy(source)
    profiles = missing["payload"]["observables"]["profiles"]
    profiles.pop("poloidal_flux")
    with pytest.raises(ValueError, match="fields differ"):
        ToraxReviewEnvelope.from_dict(reseal(missing))

    unit_drift = copy.deepcopy(source)
    unit_drift["payload"]["observables"]["source_totals"]["particles"]["unit"] = "A"
    with pytest.raises(ValueError, match="unit does not match"):
        ToraxReviewEnvelope.from_dict(reseal(unit_drift))

    profile_shape = copy.deepcopy(source)
    profile_shape["payload"]["observables"]["profiles"]["ion_temperature"]["samples"].pop()
    with pytest.raises(ValueError, match="one radial row per clock sample"):
        ToraxReviewEnvelope.from_dict(reseal(profile_shape))

    scalar_shape = copy.deepcopy(source)
    scalar_shape["payload"]["observables"]["state_budgets"]["thermal_energy"]["samples"].pop()
    with pytest.raises(ValueError, match="one scalar per clock sample"):
        ToraxReviewEnvelope.from_dict(reseal(scalar_shape))

    nonfinite_sample = copy.deepcopy(source)
    nonfinite_sample["payload"]["observables"]["source_totals"]["electron_heat"]["samples"][0] = (
        float("inf")
    )
    with pytest.raises(ValueError, match="non-finite JSON number|finite numbers"):
        ToraxReviewEnvelope.from_dict(nonfinite_sample)

    nonmonotonic_rho = copy.deepcopy(source)
    nonmonotonic_rho["payload"]["observables"]["rho"]["samples"][1] = 0.0
    with pytest.raises(ValueError, match="strictly increase"):
        ToraxReviewEnvelope.from_dict(reseal(nonmonotonic_rho))

    inconsistent_rate = copy.deepcopy(source)
    inconsistent_rate["payload"]["clock"]["sample_rate_hz"] = 99.0
    with pytest.raises(ValueError, match="disagrees"):
        ToraxReviewEnvelope.from_dict(reseal(inconsistent_rate))

    nonuniform_clock = copy.deepcopy(source)
    nonuniform_clock["payload"]["clock"]["sample_ns"][1] += 1
    with pytest.raises(ValueError, match="fixed positive interval"):
        ToraxReviewEnvelope.from_dict(reseal(nonuniform_clock))

    coordinate_frame_drift = copy.deepcopy(source)
    coordinate_frame_drift["payload"]["observables"]["rho"]["frame"] = "unbound_frame"
    with pytest.raises(ValueError, match="match the reactor frame"):
        ToraxReviewEnvelope.from_dict(reseal(coordinate_frame_drift))

    completion_drift = copy.deepcopy(source)
    completion_drift["payload"]["completion"]["reached_final_ns"] = 10_000_000
    with pytest.raises(ValueError, match="completion must equal"):
        ToraxReviewEnvelope.from_dict(reseal(completion_drift))

    negative_uncertainty = copy.deepcopy(source)
    negative_uncertainty["payload"]["uncertainty"]["observables"]["profiles"]["electron_density"][
        "absolute_rms_difference"
    ] = -1.0
    with pytest.raises(ValueError, match="finite non-negative"):
        ToraxReviewEnvelope.from_dict(reseal(negative_uncertainty))

    empirical_claim = copy.deepcopy(source)
    empirical_claim["payload"]["observables"]["profiles"]["electron_density"]["calibration"][
        "empirical"
    ] = True
    with pytest.raises(ValueError, match="identity transfer"):
        ToraxReviewEnvelope.from_dict(reseal(empirical_claim))

    burn_claim = copy.deepcopy(source)
    burn_claim["payload"]["reactor"]["operating_point"]["fuel_class_basis"] = (
        "deuterium_tritium_burn"
    )
    with pytest.raises(ValueError, match="without modeled burn or power"):
        ToraxReviewEnvelope.from_dict(reseal(burn_claim))

    high_refinement_error = copy.deepcopy(source)
    high_refinement_error["payload"]["uncertainty"]["observables"]["state_budgets"][
        "thermal_energy"
    ]["relative_l2"] = 0.0200001
    with pytest.raises(ValueError, match="exceeds the refinement gate"):
        ToraxReviewEnvelope.from_dict(reseal(high_refinement_error))

    uncertainty_unit_drift = copy.deepcopy(source)
    uncertainty_unit_drift["payload"]["uncertainty"]["observables"]["source_totals"]["particles"][
        "unit"
    ] = "A"
    with pytest.raises(ValueError, match="uncertainty unit mismatch"):
        ToraxReviewEnvelope.from_dict(reseal(uncertainty_unit_drift))

    bad_numerics = copy.deepcopy(source)
    bad_numerics["payload"]["observables"]["numerics"]["sim_status"] = "running"
    with pytest.raises(ValueError, match="completed NO_ERROR"):
        ToraxReviewEnvelope.from_dict(reseal(bad_numerics))
