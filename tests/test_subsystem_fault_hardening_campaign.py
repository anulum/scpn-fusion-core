# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Subsystem Fault Campaign Tests
"""Tests for validation/subsystem_fault_hardening_campaign.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "subsystem_fault_hardening_campaign.py"
SPEC = importlib.util.spec_from_file_location(
    "subsystem_fault_hardening_campaign",
    MODULE_PATH,
)
assert SPEC and SPEC.loader
subsystem_fault_hardening_campaign = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subsystem_fault_hardening_campaign
SPEC.loader.exec_module(subsystem_fault_hardening_campaign)


def test_subsystem_campaign_passes_reduced_order_evidence() -> None:
    report = subsystem_fault_hardening_campaign.run_campaign()
    rows = {row["scenario_id"]: row for row in report["scenario_rows"]}

    assert report["schema"] == "scpn-fusion-core.subsystem_fault_hardening.v1"
    assert report["passes_available_evidence"] is True
    assert report["full_fidelity_claim_ready"] is False
    assert report["measured_lane_count"] == 3
    assert set(rows) == {
        "rebco_quench_fault",
        "direct_energy_conversion_fault",
        "disruption_structural_shock_strain",
    }
    assert all(row["status"] == "measured_reduced_order" for row in rows.values())
    assert all(row["passes_thresholds"] is True for row in rows.values())
    assert "not certified quench protection" in report["claim_boundary"]


def test_subsystem_campaign_trace_checksum_is_deterministic() -> None:
    first = subsystem_fault_hardening_campaign.run_campaign()
    second = subsystem_fault_hardening_campaign.run_campaign()

    assert first["trace_checksum"] == second["trace_checksum"]
    assert len(first["trace_checksum"]) == 64


def test_subsystem_campaign_markdown_contains_boundaries() -> None:
    report = subsystem_fault_hardening_campaign.generate_report()
    text = subsystem_fault_hardening_campaign.render_markdown(report)

    assert "# Subsystem Fault Hardening Campaign" in text
    assert "Scenario Matrix" in text
    assert "Diagnostics" in text
    assert "Boundaries" in text
    assert "finite-element analysis" in text


def test_subsystem_campaign_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    out_json = tmp_path / "subsystem.json"
    out_md = tmp_path / "subsystem.md"
    code = subsystem_fault_hardening_campaign.main(
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
    assert payload["full_fidelity_claim_ready"] is False
    assert "Subsystem Fault Hardening Campaign" in markdown
