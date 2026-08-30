# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TORAX Runtime Contract Tests
"""Behavioral tests for versioned TORAX request and outcome contracts."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scpn_fusion.integrations.torax.contracts import (
    ToraxFailureCode,
    ToraxRunOutcome,
    ToraxRunRequest,
)
from scpn_fusion.integrations.torax.serialization import load_json_object
from validation.benchmark_torax_runtime_contract import build_request

ROOT = Path(__file__).resolve().parents[1]


def test_request_round_trip_preserves_complete_config_and_typed_bindings() -> None:
    """The portable request round trip retains every backend field and binding."""
    request = build_request(dt_s=0.01, request_id="round-trip", event_id="event-1")
    restored = ToraxRunRequest.from_dict(request.to_dict())
    assert restored.to_dict() == request.to_dict()
    assert len(restored.initial_state) == 3
    assert len(restored.controls) == 19
    assert len(restored.bindings) == 27
    assert restored.clock.final_ns == 20_000_000
    with pytest.raises(TypeError):
        restored.torax_config["solver"] = {}  # type: ignore[index] # immutable contract assertion


def test_request_rejects_unknown_fields_and_typed_config_disagreement() -> None:
    """Schema drift and any disagreement with the complete TORAX config fail closed."""
    payload = build_request(dt_s=0.01, request_id="strict", event_id="event-2").to_dict()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match=r"unknown=\['unexpected'\]"):
        ToraxRunRequest.from_dict(payload)

    payload = build_request(dt_s=0.01, request_id="binding", event_id="event-3").to_dict()
    config = copy.deepcopy(payload["torax_config"])
    assert isinstance(config, dict)
    geometry = config["geometry"]
    assert isinstance(geometry, dict)
    geometry["R_major"] = 6.3
    payload["torax_config"] = config
    with pytest.raises(ValueError, match="does not match torax_config"):
        ToraxRunRequest.from_dict(payload)

    payload = build_request(dt_s=0.01, request_id="typed", event_id="event-typed").to_dict()
    initial_state = payload["initial_state"]
    assert isinstance(initial_state, list)
    ion_temperature = initial_state[0]
    assert isinstance(ion_temperature, dict)
    ion_temperature["values"] = [[99.0, 0.5]]
    with pytest.raises(ValueError, match="disagrees with its interpreted binding"):
        ToraxRunRequest.from_dict(payload)


def test_request_rejects_nonmonotone_clocked_signals_and_wrong_backend_version() -> None:
    """Mixed/stale clock samples and unpinned backend versions are inadmissible."""
    payload = build_request(dt_s=0.01, request_id="clock", event_id="event-4").to_dict()
    controls = payload["controls"]
    assert isinstance(controls, list)
    edge = controls[1]
    assert isinstance(edge, dict)
    edge["time_ns"] = [20_000_000, 0]
    with pytest.raises(ValueError, match="strictly increasing"):
        ToraxRunRequest.from_dict(payload)

    payload = build_request(dt_s=0.01, request_id="version", event_id="event-5").to_dict()
    payload["expected_torax_version"] = "1.4.2"
    with pytest.raises(ValueError, match="requires TORAX 1.4.3"):
        ToraxRunRequest.from_dict(payload)


def test_all_upstream_sim_errors_have_stable_one_to_one_codes() -> None:
    """Every TORAX 1.4.3 SimError maps without collapsing distinct failures."""
    expected = {
        "NO_ERROR": None,
        "NAN_DETECTED": ToraxFailureCode.NAN_DETECTED,
        "QUASINEUTRALITY_BROKEN": ToraxFailureCode.QUASINEUTRALITY_BROKEN,
        "NEGATIVE_CORE_PROFILES": ToraxFailureCode.NEGATIVE_CORE_PROFILES,
        "REACHED_MIN_DT": ToraxFailureCode.REACHED_MIN_DT,
        "LOW_TEMPERATURE_COLLAPSE": ToraxFailureCode.LOW_TEMPERATURE_COLLAPSE,
        "DID_NOT_REACH_T_FINAL": ToraxFailureCode.DID_NOT_REACH_T_FINAL,
    }
    assert {name: ToraxFailureCode.from_sim_error(name) for name in expected} == expected
    with pytest.raises(ValueError, match="unknown TORAX SimError"):
        ToraxFailureCode.from_sim_error("NEW_UNPINNED_ERROR")


def test_tracked_real_outcome_round_trip_is_strict_and_omits_inferred_truth() -> None:
    """The generated portable outcome parses without TORAX and exposes no inferred scalar view."""
    outcome = ToraxRunOutcome.from_dict(
        load_json_object(ROOT / "validation/reference_data/torax/torax_runtime_result_v1.json")
    )
    assert outcome.success
    assert outcome.projection is not None
    assert set(outcome.projection.profiles) == {
        "ion_temperature",
        "electron_temperature",
        "electron_density",
        "poloidal_flux",
    }
    projection_text = str(outcome.projection.to_dict())
    for excluded in ("q95", "li3", "beta_N", "W_thermal_total", "regime", "phase"):
        assert excluded not in projection_text
