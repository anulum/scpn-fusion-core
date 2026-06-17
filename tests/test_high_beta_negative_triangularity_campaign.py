# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — High-Beta Negative-Triangularity Campaign Tests
"""Tests for validation/high_beta_negative_triangularity_campaign.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "high_beta_negative_triangularity_campaign.py"
SPEC = importlib.util.spec_from_file_location(
    "high_beta_negative_triangularity_campaign",
    MODULE_PATH,
)
assert SPEC and SPEC.loader
high_beta_negative_triangularity_campaign = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = high_beta_negative_triangularity_campaign
SPEC.loader.exec_module(high_beta_negative_triangularity_campaign)


def test_campaign_passes_and_preserves_claim_boundaries() -> None:
    report = high_beta_negative_triangularity_campaign.run_campaign()

    assert report["campaign_id"] == "high_beta_negative_triangularity_campaign"
    assert report["acceptance"]["passes_thresholds"] is True
    assert report["acceptance"]["failure_reasons"] == []
    assert "hardware_beta_40_operation" in report["acceptance"]["blocked_claims"]
    assert (
        "experimental_negative_triangularity_elm_free_operation"
        in report["acceptance"]["blocked_claims"]
    )

    components = report["components"]
    assert components["geometry"]["passes_geometry_contract"] is True
    assert components["geometry"]["aspect_ratio"] == pytest.approx(1.5)
    assert components["geometry"]["elongation_kappa"] > 2.0
    assert components["geometry"]["triangularity_delta"] < 0.0
    assert components["edge_elm"]["passes_edge_contract"] is True
    assert components["edge_elm"]["peeling_ballooning_margin"] > 0.05
    assert components["edge_elm"]["peeling_ballooning_unstable"] is False
    assert components["divertor"]["passes_divertor_contract"] is True
    assert components["vertical_control"]["passes_thresholds"] is True
    assert components["free_boundary_disruption_policy"]["passes_thresholds"] is True


def test_campaign_is_deterministic_for_stable_payload() -> None:
    first = high_beta_negative_triangularity_campaign.run_campaign()
    second = high_beta_negative_triangularity_campaign.run_campaign()

    assert first == second
    assert first["scenario_checksum_sha256"] == second["scenario_checksum_sha256"]
    assert first["component_checksum_sha256"] == second["component_checksum_sha256"]


def _assert_invalid_scenario(scenario_kwargs: dict[str, float], match: str) -> None:
    scenario = high_beta_negative_triangularity_campaign.NegativeTriangularityScenario(
        **scenario_kwargs
    )
    with pytest.raises(ValueError, match=match):
        high_beta_negative_triangularity_campaign.run_campaign(scenario)


def test_campaign_rejects_non_negative_triangularity() -> None:
    _assert_invalid_scenario({"triangularity_delta": 0.1}, "triangularity_delta")


def test_campaign_rejects_zero_major_radius() -> None:
    _assert_invalid_scenario({"major_radius_m": 0.0}, "major_radius_m")


def test_campaign_rejects_low_elongation() -> None:
    _assert_invalid_scenario({"elongation_kappa": 0.5}, "elongation_kappa")


def test_markdown_contains_integrated_campaign_sections() -> None:
    report = high_beta_negative_triangularity_campaign.generate_report()
    text = high_beta_negative_triangularity_campaign.render_markdown(report)

    assert "# High-Beta Negative-Triangularity Integrated Campaign" in text
    assert "Blocked claims" in text
    assert "Geometry" in text
    assert "Edge And Divertor" in text
    assert "Control And Disruption" in text
    assert "hardware_beta_40_operation" in text
