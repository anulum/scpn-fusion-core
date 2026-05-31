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
