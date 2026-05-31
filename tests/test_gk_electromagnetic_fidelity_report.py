#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the fail-closed nonlinear GK electromagnetic fidelity gate."""

from __future__ import annotations

import json
from pathlib import Path

from validation.benchmark_gk_electromagnetic_fidelity import run_benchmark, write_reports


def test_gk_electromagnetic_fidelity_report_gates_em_separately() -> None:
    report = run_benchmark()

    assert report["schema"] == "gk-electromagnetic-fidelity.v1"
    assert report["status"] == "blocked_missing_full_vlasov_maxwell_field_solve"
    assert report["electrostatic_gate"]["electromagnetic_enabled"] is False
    assert report["electromagnetic_gate"]["electromagnetic_enabled"] is True
    assert report["electromagnetic_gate"]["compact_closure_ready"] is True
    assert report["electromagnetic_gate"]["full_vlasov_maxwell_parity_ready"] is False
    assert report["external_em_parity_comparison_ready"] is False
    assert report["required_external_solver_families"] == ["GENE", "CGYRO", "GS2"]
    assert {
        "electromagnetic_phi_energy",
        "electromagnetic_apar_energy",
        "electromagnetic_bpar_energy",
    }.issubset(set(report["required_external_observables"]))
    assert "Faraday induction equation for evolving B" in report["omitted_physics"]
    assert "displacement-current Ampere-Maxwell evolution" in report["omitted_physics"]
    assert report["locally_actionable_contract_ready"] is True


def test_gk_electromagnetic_fidelity_report_writes_json_and_markdown(tmp_path: Path) -> None:
    report = run_benchmark()

    write_reports(report, report_dir=tmp_path)

    payload = json.loads((tmp_path / "gk_electromagnetic_fidelity.json").read_text())
    assert payload["schema"] == "gk-electromagnetic-fidelity.v1"
    assert (tmp_path / "gk_electromagnetic_fidelity.md").exists()
