# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Whole-Plant Fault Scenario Tests
"""Tests for validation/whole_plant_fault_tolerant_scenario.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "whole_plant_fault_tolerant_scenario.py"
SPEC = importlib.util.spec_from_file_location(
    "whole_plant_fault_tolerant_scenario",
    MODULE_PATH,
)
assert SPEC and SPEC.loader
whole_plant_fault_tolerant_scenario = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(whole_plant_fault_tolerant_scenario)


def test_campaign_passes_available_reduced_order_evidence() -> None:
    report = whole_plant_fault_tolerant_scenario.run_campaign(seed=42)
    rows = {row["scenario_id"]: row for row in report["scenario_rows"]}

    assert report["schema"] == "scpn-fusion-core.whole_plant_fault_tolerant_scenario.v1"
    assert report["passes_available_evidence"] is True
    assert report["full_whole_plant_claim_ready"] is False
    assert report["measured_lane_count"] == 7
    assert report["blocked_lane_count"] == 2
    assert "Reduced-order software campaign only" in report["claim_boundary"]

    measured_ids = {
        "vertical_excursion_vde",
        "disruption_risk_spike",
        "sensor_dropout_noise",
        "actuator_saturation_dropout",
        "controller_failover",
        "cooling_thermal_limit",
        "shielding_wall_load_warning",
    }
    for scenario_id in measured_ids:
        assert rows[scenario_id]["status"] == "measured_reduced_order"
        assert rows[scenario_id]["passes_thresholds"] is True

    for scenario_id in ("direct_energy_conversion_fault", "rebco_quench_fault"):
        assert rows[scenario_id]["status"] == "blocked_no_subsystem_model"
        assert rows[scenario_id]["passes_thresholds"] is False


def test_campaign_trace_checksum_is_deterministic_for_seed() -> None:
    first = whole_plant_fault_tolerant_scenario.run_campaign(seed=42)
    second = whole_plant_fault_tolerant_scenario.run_campaign(seed=42)

    assert first["trace_checksum"] == second["trace_checksum"]
    assert len(first["trace_checksum"]) == 64


def test_campaign_rejects_invalid_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        whole_plant_fault_tolerant_scenario.run_campaign(seed=-1)


def test_markdown_contains_scenario_and_blocked_subsystems() -> None:
    report = whole_plant_fault_tolerant_scenario.generate_report(seed=42)
    text = whole_plant_fault_tolerant_scenario.render_markdown(report)

    assert "# Whole-Plant Fault-Tolerant Scenario Campaign" in text
    assert "Scenario Matrix" in text
    assert "Fault Controller" in text
    assert "Thermal And Wall Loads" in text
    assert "Blocked Subsystems" in text
    assert "direct_energy_conversion_fault" in text
    assert "rebco_quench_fault" in text


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    out_json = tmp_path / "campaign.json"
    out_md = tmp_path / "campaign.md"
    code = whole_plant_fault_tolerant_scenario.main(
        [
            "--strict",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ]
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    markdown = out_md.read_text(encoding="utf-8")
    assert code == 0
    assert payload["passes_available_evidence"] is True
    assert payload["full_whole_plant_claim_ready"] is False
    assert "Whole-Plant Fault-Tolerant Scenario Campaign" in markdown
