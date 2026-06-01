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
    assert report["electrostatic_gk_ready"] is True
    assert report["compact_em_ready"] is True
    assert report["source_free_maxwell_ready"] is True
    assert report["sourced_maxwell_ready"] is False
    assert report["external_em_parity_ready"] is False
    assert report["full_vlasov_maxwell_ready"] is False
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

    evidence = report["external_em_parity_evidence"]
    assert evidence["schema"] == "gk-electromagnetic-external-parity-evidence.v1"
    assert evidence["status"] == "blocked_missing_same_deck_external_em_outputs"
    assert evidence["same_deck_group_ready"] is False
    assert evidence["solver_family_completeness_ready"] is False
    rows = {row["solver_family"]: row for row in evidence["solver_family_rows"]}
    assert set(rows) == {"GENE", "CGYRO", "GS2"}
    for row in rows.values():
        assert row["same_deck_reference_output_ready"] is False
        assert row["native_same_case_comparison_ready"] is False
        assert row["complete_required_observables"] is False
        assert set(row["observable_presence"]) == set(report["required_external_observables"])
        assert not any(row["observable_presence"].values())
    matrix = {row["surface"]: row for row in report["electromagnetic_evidence_gate_matrix"]}
    assert matrix["electrostatic_gk_gate_separation"]["ready"] is True
    assert matrix["compact_A_parallel_B_parallel_closure"]["ready"] is True
    assert matrix["source_free_faraday_induction"]["ready"] is True
    assert matrix["source_free_ampere_maxwell_displacement_current"]["ready"] is True
    assert matrix["source_free_inductive_parallel_electric_field"]["ready"] is True
    assert matrix["magnetic_divergence_constraint"]["ready"] is True
    assert matrix["electromagnetic_energy_invariant_diagnostics"]["ready"] is True
    assert matrix["native_em_same_case_thresholds"]["ready"] is True
    assert matrix["sourced_kinetic_current_maxwell_coupling"]["ready"] is False
    assert matrix["external_em_gene_cgyro_gs2_parity"]["ready"] is False
    assert matrix["external_em_grid_convergence"]["ready"] is False


def test_gk_electromagnetic_fidelity_report_declares_maxwell_evolution_contract() -> None:
    report = run_benchmark()

    contract = report["maxwell_evolution_contract"]
    assert contract["native_field_evolution_mode"] == "local_spectral_maxwell_evolution"
    assert contract["full_vlasov_maxwell_parity_ready"] is False
    equations = {equation["equation_id"]: equation for equation in contract["equations"]}
    assert equations["faraday_induction"]["implemented_by_native_solver"] is True
    assert equations["ampere_maxwell_displacement_current"]["implemented_by_native_solver"] is True
    assert equations["inductive_parallel_electric_field"]["implemented_by_native_solver"] is True
    assert equations["magnetic_divergence_constraint"]["implemented_by_native_solver"] is True
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
    assert evidence["max_magnetic_divergence_linf_residual"] <= evidence["residual_tolerance"]


def test_gk_electromagnetic_fidelity_report_records_sourced_maxwell_future_contract() -> None:
    report = run_benchmark()

    contract = report["sourced_maxwell_contract"]
    assert contract["schema"] == "gk-sourced-maxwell-contract.v1"
    assert contract["sourced_maxwell_ready"] is False
    assert contract["status"] == "blocked_sourced_maxwell_requires_self_consistent_field_coupling"
    assert contract["current_moment_ready"] is True
    assert contract["current_moment_history_ready"] is True
    assert contract["continuity_residual_history_ready"] is True
    assert contract["field_particle_exchange_ready"] is True
    assert contract["self_consistent_sourced_field_evolution_ready"] is False
    assert "J_parallel(kx, ky, t)" in contract["required_inputs"]
    assert "rho_charge(kx, ky, t)" in contract["required_inputs"]
    assert "dE_parallel_dt(kx, ky, t)" in contract["required_inputs"]
    assert (
        "J_parallel(kx, ky, t) derived from the evolved 5D distribution"
        in contract["readiness_criteria"]
    )
    field_contract = contract["sourced_field_evolution_contract"]
    assert field_contract["schema"] == "gk-sourced-field-evolution-contract.v1"
    assert field_contract["ready"] is False
    assert field_contract["status"] == "blocked_missing_self_consistent_sourced_field_evolution"
    assert field_contract["blocking_terms"] == [
        "self_consistent_dE_parallel_dt_from_sourced_A_parallel_phi",
        "curl_B_minus_mu0_J_minus_mu0_epsilon0_dE_dt_history",
        "sourced_faraday_curl_E_plus_dB_dt_history",
        "field_particle_energy_balance_residual_history",
    ]
    equation_rows = {row["equation_id"]: row for row in field_contract["equation_rows"]}
    assert set(equation_rows) == {
        "parallel_field_particle_energy_balance",
        "sourced_faraday_induction",
        "sourced_parallel_ampere_maxwell",
    }
    assert not any(row["implemented_by_native_solver"] for row in equation_rows.values())


def test_gk_electromagnetic_fidelity_report_extracts_time_resolved_current_moments() -> None:
    report = run_benchmark()

    evidence = report["sourced_current_moment_evidence"]
    assert evidence["schema"] == "gk-sourced-current-moment-evidence.v1"
    assert evidence["status"] == "accepted_time_resolved_current_and_continuity_proxy_field_coupling_missing"
    assert evidence["current_moment_ready"] is True
    assert evidence["current_moment_source"] == "native_time_resolved_5d_distribution_state"
    assert evidence["phase_space_source_shape"] == [2, 4, 4, 8, 5, 4]
    assert evidence["j_parallel_shape"] == [5, 4, 4]
    assert evidence["j_kx_shape"] == [5, 4, 4]
    assert evidence["j_ky_shape"] == [5, 4, 4]
    assert evidence["charge_density_shape"] == [5, 4, 4]
    assert evidence["e_parallel_shape"] == [5, 4, 4]
    assert evidence["time_resolved_current_history_ready"] is True
    assert evidence["perpendicular_current_history_ready"] is True
    assert evidence["continuity_residual_history_ready"] is True
    assert evidence["continuity_residual_status"] == "accepted_spectral_continuity_proxy_not_sourced_field_coupling"
    assert evidence["d_charge_dt_ready"] is True
    assert evidence["field_particle_exchange_ready"] is True
    assert evidence["field_particle_exchange_status"] == "accepted_native_j_parallel_e_parallel_proxy"
    assert len(evidence["field_particle_exchange_t"]) == 5
    assert evidence["continuity_relative_residual_max"] <= evidence["continuity_relative_residual_tolerance"]
    assert evidence["j_parallel_l2_norm_max"] > 0.0
    assert evidence["charge_density_l2_norm_max"] > 0.0
    residual_rows = evidence["sourced_ampere_maxwell_residual_rows"]
    assert len(residual_rows) == 1
    assert residual_rows[0]["ready"] is False
    assert residual_rows[0]["status"] == "blocked_missing_sourced_field_evolution_terms"
    assert "missing_self_consistent_displacement_current_from_sourced_field_evolution" in residual_rows[0]["blockers"]
    assert "missing_self_consistent_e_parallel_field_evolution" in residual_rows[0]["blockers"]


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
