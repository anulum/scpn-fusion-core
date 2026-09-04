# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Benchmark Result Collection Tests
"""Artifact-boundary tests for the benchmark result collector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from validation import collect_results
from validation import stress_test_campaign
from validation.stress_campaign_contract import (
    CampaignResults,
    ControllerMetrics,
    StressScenario,
)


def test_real_shot_loader_reads_json_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public real-shot loader preserves a valid object-root artifact."""
    payload = {"overall_pass": False, "disruption": {"recall": 0.75}}
    artifact_path = tmp_path / "real_shot_validation.json"
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(collect_results, "ARTIFACTS", tmp_path)

    assert collect_results.load_real_shot_validation() == payload


def test_real_shot_loader_rejects_non_object_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A list-root artifact fails closed before report rendering."""
    artifact_path = tmp_path / "real_shot_validation.json"
    artifact_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(collect_results, "ARTIFACTS", tmp_path)

    with pytest.raises(ValueError, match="JSON object with string keys"):
        collect_results.load_real_shot_validation()


def test_controller_collector_preserves_lane_status_and_scenario_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compact collector cannot hide unavailable lanes or scenario identity."""
    scenario = StressScenario(master_seed=31)
    lane = ControllerMetrics(
        name="H-infinity",
        requested_episodes=5,
        scenario_digest=scenario.digest,
        evaluation_contract_digest="contract-digest",
        policy_implementation="test.hinf",
        status="unavailable",
        reason="research_gate_disabled",
    )
    results = CampaignResults(
        {"H-infinity": lane},
        scenario=scenario,
        campaign_identity={"digest": "campaign-digest"},
    )
    monkeypatch.setattr(stress_test_campaign, "run_campaign", lambda **kwargs: results)

    compact = collect_results.run_controller_campaign(quick=True)

    assert compact is not None
    assert compact["scenario_digest"] == scenario.digest
    assert compact["campaign_identity_digest"] == "campaign-digest"
    assert compact["controllers"]["H-infinity"]["status"] == "unavailable"
    assert compact["controllers"]["H-infinity"]["policy_implementation"] == "test.hinf"
    assert compact["controllers"]["H-infinity"]["mean_tracking_reward_m"] is None
    assert compact["controllers"]["H-infinity"]["mean_abs_z_error_m"] is None
    assert compact["controllers"]["H-infinity"]["comparable"] is False
