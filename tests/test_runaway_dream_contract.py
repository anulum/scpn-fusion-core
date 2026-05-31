# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
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
