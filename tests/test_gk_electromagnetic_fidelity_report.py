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
    assert report["status"] == "blocked_missing_external_em_parity_outputs"
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
    assert "Faraday induction equation for evolving B" not in report["omitted_physics"]
    assert "displacement-current Ampere-Maxwell evolution" not in report["omitted_physics"]
    assert report["locally_actionable_contract_ready"] is True


def test_gk_electromagnetic_fidelity_report_declares_maxwell_evolution_contract() -> None:
    report = run_benchmark()

    contract = report["maxwell_evolution_contract"]
    assert contract["native_field_evolution_mode"] == "local_spectral_maxwell_evolution"
    assert contract["full_vlasov_maxwell_parity_ready"] is False
    equations = {equation["equation_id"]: equation for equation in contract["equations"]}
    assert equations["faraday_induction"]["implemented_by_native_solver"] is True
    assert equations["ampere_maxwell_displacement_current"]["implemented_by_native_solver"] is True
    assert equations["inductive_parallel_electric_field"]["implemented_by_native_solver"] is True
    assert equations["compact_parallel_ampere_closure"]["compact_closure_ready"] is True
    assert (
        equations["compact_perpendicular_pressure_balance_closure"]["compact_closure_ready"] is True
    )
    assert contract["blocking_equation_ids"] == [
        "self_consistent_kinetic_current_coupling",
        "same_deck_external_em_parity",
    ]


def test_gk_electromagnetic_fidelity_report_records_grid_convergence_evidence() -> None:
    report = run_benchmark()

    evidence = report["electromagnetic_grid_convergence_evidence"]
    assert evidence["schema"] == "gk-electromagnetic-grid-convergence.v1"
    assert report["electromagnetic_grid_convergence_ready"] is True
    assert evidence["grid_case_count"] >= 3
    assert evidence["field_energy_histories_finite"] is True
    assert evidence["compact_closure_residuals_converged"] is True
    assert evidence["field_energy_refinement_comparison_ready"] is True
    assert evidence["max_relative_total_energy_drift"] <= evidence["relative_energy_tolerance"]
    assert all(row["compact_closure_ready"] for row in evidence["grid_rows"])
    assert all(row["field_energy_total_closes"] for row in evidence["grid_rows"])
    assert (
        "grid-convergence evidence for electromagnetic field-energy histories"
        not in report["missing_full_fidelity_requirements"]
    )
    assert (
        "same-deck external electromagnetic grid-convergence evidence"
        in report["missing_full_fidelity_requirements"]
    )


def test_gk_electromagnetic_fidelity_report_records_maxwell_evolution_evidence() -> None:
    report = run_benchmark()

    evidence = report["maxwell_evolution_evidence"]
    assert evidence["schema"] == "gk-maxwell-evolution.v1"
    assert evidence["faraday_induction_supported"] is True
    assert evidence["ampere_maxwell_displacement_current_supported"] is True
    assert evidence["inductive_parallel_electric_field_supported"] is True
    assert evidence["self_consistent_kinetic_current_supported"] is False
    assert (
        evidence["max_relative_total_field_energy_drift"] <= evidence["relative_energy_tolerance"]
    )
    assert evidence["max_faraday_linf_residual"] <= evidence["residual_tolerance"]
    assert evidence["max_ampere_maxwell_linf_residual"] <= evidence["residual_tolerance"]
    assert evidence["max_inductive_e_parallel_linf_residual"] <= evidence["residual_tolerance"]


def test_gk_electromagnetic_fidelity_report_records_native_same_case_thresholds() -> None:
    report = run_benchmark()

    evidence = report["native_em_same_case_threshold_evidence"]
    assert evidence["schema"] == "gk-native-em-same-case-thresholds.v1"
    assert evidence["reference_kind"] == "native_deterministic_replay_not_external_parity"
    assert evidence["same_case_thresholds_ready"] is True
    assert evidence["observable_count"] == 4
    assert {row["observable"] for row in evidence["observable_rows"]} == {
        "electromagnetic_phi_energy",
        "electromagnetic_apar_energy",
        "electromagnetic_bpar_energy",
        "electromagnetic_total_field_energy",
    }
    assert all(row["threshold_pass"] for row in evidence["observable_rows"])
    assert all(
        row["max_absolute_error"] <= row["absolute_tolerance"]
        for row in evidence["observable_rows"]
    )
    assert (
        "native electromagnetic phi/A_parallel/B_parallel same-case parity thresholds"
        not in report["missing_full_fidelity_requirements"]
    )
    assert (
        "external electromagnetic phi/A_parallel/B_parallel same-case parity thresholds"
        in report["missing_full_fidelity_requirements"]
    )


def test_gk_electromagnetic_fidelity_report_writes_json_and_markdown(tmp_path: Path) -> None:
    report = run_benchmark()

    write_reports(report, report_dir=tmp_path)

    payload = json.loads((tmp_path / "gk_electromagnetic_fidelity.json").read_text())
    assert payload["schema"] == "gk-electromagnetic-fidelity.v1"
    assert (tmp_path / "gk_electromagnetic_fidelity.md").exists()
