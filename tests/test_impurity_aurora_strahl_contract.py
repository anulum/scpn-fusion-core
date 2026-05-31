# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Aurora/STRAHL Impurity Contract Tests

from __future__ import annotations

from tools.run_aurora_reference_artifact import build_aurora_reference_execution_report
from validation.benchmark_impurity_transport_contract import run_benchmark


def test_impurity_benchmark_exports_validated_aurora_strahl_artifact_gate() -> None:
    report = run_benchmark()

    artifact = report["artifact_contract"]
    assert artifact["schema"] == "aurora-strahl-charge-state-artifact.v1"
    assert artifact["contract_validation"]["passed"] is True
    assert artifact["same_case_aurora_strahl_comparison_ready"] is False
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
    assert evidence["aurora_strahl_same_case_threshold_ready"] is False
    assert evidence["operator_terms_present"]["trace_radial_transport"] is True
    assert evidence["operator_terms_present"]["charge_state_source_sink_matrix"] is True
    assert evidence["operator_terms_present"]["line_radiation_power"] is True
    assert evidence["operator_terms_present"]["charge_state_resolved_radial_transport"] is False
    assert evidence["operator_terms_present"]["external_adas_transport_coefficients"] is False
    assert evidence["operator_terms_present"]["same_case_aurora_strahl_transport_output"] is False
    assert report["invariants"]["native_impurity_transport_evidence_fail_closed"] is True
    assert "public Aurora or STRAHL radial transport output" in evidence["blocking_requirements"]


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
