# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Aurora/STRAHL Impurity Contract Tests

from __future__ import annotations

import numpy as np

from scpn_fusion.core.impurity_transport import AuroraParityCase, AuroraParityImpuritySolver
from tools.run_aurora_reference_artifact import build_aurora_reference_execution_report
from validation.benchmark_impurity_transport_contract import run_benchmark


def test_aurora_parity_solver_exports_separate_native_contract() -> None:
    radius_m = np.linspace(0.0, 0.3, 6, dtype=np.float64)
    time_s = np.array([0.0, 1.0e-6, 2.0e-6], dtype=np.float64)
    charge_state = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    density = np.zeros((radius_m.size, charge_state.size), dtype=np.float64)
    density[:, 1] = 1.0e12 * (1.0 - 0.2 * radius_m / radius_m[-1])
    case = AuroraParityCase(
        element="Ar",
        charge_states=charge_state,
        radius_m=radius_m,
        time_s=time_s,
        ne_t_r=np.full((time_s.size, radius_m.size), 2.0e19, dtype=np.float64),
        Te_t_r=np.full((time_s.size, radius_m.size), 500.0, dtype=np.float64),
        initial_charge_state_density_rz=density,
        diffusion_m2_s_r_z=np.full((radius_m.size, charge_state.size), 0.2, dtype=np.float64),
        convection_m_s_r_z=np.zeros((radius_m.size, charge_state.size), dtype=np.float64),
        major_radius_m=1.7,
    )

    artifact = AuroraParityImpuritySolver(case).solve()
    payload = artifact.to_dict()
    validation = artifact.validate_contract()

    assert payload["provenance"]["implementation"] == "native_aurora_compatibility_parity_solver"
    assert payload["provenance"]["parity_status"] == (
        "native_aurora_compatibility_mode_threshold_gated"
    )
    assert validation["passed"] is True
    assert validation["observable_shapes"]["charge_state_density_r_t"] == [3, 6, 3]
    assert validation["source_sink_conservative"] is True


def test_impurity_benchmark_exports_validated_aurora_strahl_artifact_gate() -> None:
    report = run_benchmark()

    artifact = report["artifact_contract"]
    assert artifact["schema"] == "aurora-strahl-charge-state-artifact.v1"
    assert artifact["contract_validation"]["passed"] is True
    assert artifact["same_case_aurora_strahl_comparison_ready"] is True
    assert artifact["required_aurora_strahl_observables"] == [
        "charge_state_density_r_t",
        "total_impurity_density_r_t",
        "line_radiation_power_t",
        "line_radiation_power_t_r_z",
        "source_sink_matrix_t_r_z_z",
        "total_impurity_inventory_t",
    ]
    assert artifact["observable_shapes"]["source_sink_matrix_t_r_z_z"] == [3, 80, 4, 4]
    assert artifact["observable_shapes"]["line_radiation_power_t_r_z"] == [3, 80, 4]
    assert report["invariants"]["source_sink_matrix_conservative"] is True
    assert report["invariants"]["line_radiation_power_finite"] is True


def test_impurity_benchmark_exports_fail_closed_transport_operator_evidence() -> None:
    report = run_benchmark()

    evidence = report["native_impurity_transport_evidence"]
    assert evidence["schema"] == "native-impurity-transport-operator-evidence.v1"
    assert evidence["operator_evidence_status"] == (
        "blocked_native_charge_state_contract_not_full_aurora_strahl_transport_operator"
    )
    assert evidence["density_axes"] == ["time_s", "radius_m", "charge_state"]
    assert evidence["density_shape"] == [3, 80, 4]
    assert evidence["source_sink_shape"] == [3, 80, 4, 4]
    assert evidence["line_radiation_shape"] == [3, 80, 4]
    assert evidence["native_artifact_ready"] is True
    assert evidence["charge_state_density_closure"] is True
    assert evidence["source_sink_conservative"] is True
    assert evidence["inventory_conserved"] is True
    assert evidence["charge_state_radial_transport_operator_ready"] is False
    assert evidence["aurora_strahl_same_case_comparison_ready"] is True
    assert evidence["aurora_strahl_same_case_threshold_ready"] is True
    assert evidence["aurora_strahl_same_case_threshold_passed"] is False
    assert evidence["operator_terms_present"]["trace_radial_transport"] is True
    assert evidence["operator_terms_present"]["charge_state_source_sink_matrix"] is True
    assert evidence["operator_terms_present"]["line_radiation_power"] is True
    assert evidence["operator_terms_present"]["charge_state_resolved_radial_transport"] is False
    assert evidence["operator_terms_present"]["external_adas_transport_coefficients"] is False
    assert evidence["operator_terms_present"]["same_case_aurora_strahl_transport_output"] is True
    same_case = evidence["same_case_aurora_strahl_comparison"]
    assert same_case["schema"] == "aurora-strahl-native-same-case-comparison.v1"
    assert same_case["status"] == "blocked_native_aurora_same_case_threshold_mismatch"
    assert same_case["comparison_ready"] is True
    assert same_case["threshold_checks_ready"] is True
    assert same_case["thresholds_passed"] is False
    assert same_case["native_coordinate_match"] is True
    assert same_case["native_total_density_closure"] is True
    assert same_case["external_coefficient_tables_ready"] is True
    assert same_case["aurora_case_profiles_ready"] is True
    assert same_case["native_density_shape"] == [4, 65, 19]
    assert same_case["reference_density_shape"] == [4, 65, 19]
    assert {check["threshold"] for check in same_case["checks"]} == {
        "charge_state_density_relative_l2_max",
        "particle_conservation_relative_error_max",
        "radiated_power_relative_l2_max",
        "total_density_relative_l2_max",
    }
    assert all(check["valid"] for check in same_case["checks"])
    assert not any(check["passed"] for check in same_case["checks"])
    budget = evidence["source_sink_budget_evidence"]
    assert budget["schema"] == "native-impurity-source-sink-budget-evidence.v1"
    assert (
        budget["status"]
        == "native_artifact_source_sink_budget_only_not_aurora_strahl_operator_parity"
    )
    assert budget["budget_terms"] == [
        "source_sink_matrix_t_r_z_z",
        "ionisation_source_matrix",
        "recombination_sink_matrix",
        "line_radiation_power_t_r_z",
        "total_impurity_inventory_t",
    ]
    assert budget["time_count"] == 3
    assert budget["radius_count"] == 80
    assert budget["charge_state_count"] == 4
    assert budget["all_budget_terms_finite"] is True
    assert budget["ionisation_recombination_nonnegative"] is True
    assert budget["source_sink_transfer_conservative"] is True
    assert budget["source_sink_diagonal_nonpositive"] is True
    assert budget["source_sink_offdiagonal_nonnegative"] is True
    assert budget["radial_total_density_conserved"] is True
    assert budget["max_radial_total_density_relative_change"] <= 1.0e-12
    assert budget["line_radiation_nonnegative"] is True
    assert budget["inventory_relative_change_max"] <= 1.0e-12
    assert budget["aurora_strahl_same_case_budget_ready"] is False
    assert report["invariants"]["native_impurity_transport_evidence_fail_closed"] is True
    assert report["invariants"]["native_source_sink_budget_evidence_fail_closed"] is True
    assert report["invariants"]["charge_state_radial_density_conservation"] is True
    assert "Aurora source/recycling/effective transport closure" in evidence["blocking_requirements"]
    assert "native same-case Aurora threshold pass" in evidence["blocking_requirements"]


def test_aurora_reference_report_declares_fail_closed_transport_output_contract() -> None:
    report = build_aurora_reference_execution_report(write=False)

    assert report["accepted_full_fidelity_ready"] is False
    assert report["same_case_comparison_ready"] is False
    assert report["required_output_contract"]["schema"] == "aurora-strahl-output-contract.v1"
    assert report["required_output_contract"]["coordinate_axes"] == [
        "time_s",
        "radius_m",
        "charge_state",
    ]
    assert report["required_output_contract"]["observables"] == [
        "charge_state_density_r_t",
        "total_impurity_density_r_t",
        "line_radiation_power_t",
        "line_radiation_power_t_r_z",
        "source_sink_matrix_t_r_z_z",
        "total_impurity_inventory_t",
    ]
