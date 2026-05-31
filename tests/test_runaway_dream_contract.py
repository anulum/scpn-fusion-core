# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — DREAM Runaway Contract Tests

from __future__ import annotations

from tools.run_dream_reference_artifact import build_dream_reference_execution_report
from validation.benchmark_runaway_dream_contract import run_benchmark


def test_runaway_dream_benchmark_exports_native_kinetic_artifact_gate() -> None:
    report = run_benchmark(repeats=2)

    artifact = report["native_kinetic_artifact"]
    assert artifact["schema"] == "dream-kinetic-artifact.v1"
    assert artifact["parity_status"] == "native_contract_only_not_dream_parity"
    assert artifact["contract_validation"]["passed"] is True
    assert artifact["same_case_dream_comparison_ready"] is False
    assert artifact["required_dream_observables"] == [
        "f_p_xi_t",
        "runaway_current_t",
        "avalanche_growth_rate_t",
        "synchrotron_loss_power_t",
        "partial_screening_drag_t",
        "bremsstrahlung_loss_power_t",
    ]
    assert artifact["observable_shapes"]["f_p_xi_t"] == [4, 4, 48, 5]
    assert report["invariants"]["native_kinetic_artifact_contract"] is True


def test_runaway_dream_benchmark_exports_fail_closed_operator_evidence() -> None:
    report = run_benchmark(repeats=2)

    evidence = report["native_kinetic_operator_evidence"]
    assert evidence["schema"] == "native-runaway-kinetic-operator-evidence.v1"
    assert evidence["operator_evidence_status"] == (
        "blocked_native_projection_artifact_not_full_dream_operator"
    )
    assert evidence["distribution_axes"] == [
        "time_s",
        "radius_m",
        "momentum_mec",
        "pitch_cosine",
    ]
    assert evidence["distribution_shape"] == [4, 4, 48, 5]
    assert evidence["native_artifact_ready"] is True
    assert evidence["radius_pitch_are_evolved_operator_axes"] is False
    assert evidence["full_momentum_pitch_radius_operator_ready"] is False
    assert evidence["dream_same_case_threshold_ready"] is False
    assert evidence["operator_terms_present"]["momentum_advection_drag"] is True
    assert evidence["operator_terms_present"]["momentum_diffusion"] is True
    assert evidence["operator_terms_present"]["synchrotron_radiation_reaction"] is True
    assert evidence["operator_terms_present"]["full_pitch_angle_scattering_operator"] is False
    assert evidence["operator_terms_present"]["full_radial_transport_operator"] is False
    assert evidence["operator_terms_present"]["partial_screening_dream_operator"] is False
    assert evidence["operator_terms_present"]["bremsstrahlung_radiation_loss_operator"] is False
    assert all(evidence["observable_finiteness"].values())
    assert all(evidence["observable_nonnegativity"].values())
    assert report["invariants"]["native_kinetic_operator_evidence_fail_closed"] is True
    assert "compiled DREAM iface/dreami same-case output" in evidence["blocking_requirements"]


def test_dream_reference_request_declares_fail_closed_output_contract() -> None:
    report = build_dream_reference_execution_report(write=False, execute_backend=False)

    assert report["accepted_full_fidelity_ready"] is False
    assert report["reference_output_ready"] is False
    assert report["same_case_comparison_ready"] is False
    assert report["required_output_contract"]["schema"] == "dream-output-contract.v1"
    assert report["required_output_contract"]["coordinate_axes"] == [
        "time_s",
        "radius_m",
        "momentum_mec",
        "pitch_cosine",
    ]
    assert report["required_output_contract"]["observables"] == [
        "f_p_xi_t",
        "runaway_current_t",
        "avalanche_growth_rate_t",
        "synchrotron_loss_power_t",
        "partial_screening_drag_t",
        "bremsstrahlung_loss_power_t",
    ]
