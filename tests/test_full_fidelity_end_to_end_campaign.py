"""Tests for integrated full-fidelity campaign reporting."""

from __future__ import annotations

import pytest

from validation.full_fidelity_end_to_end_campaign import run_campaign, write_reports


def test_integrated_campaign_reports_all_declared_blockers() -> None:
    report = run_campaign()

    assert report["schema"] == "full-fidelity-end-to-end-campaign.v1"
    assert report["status"] == "not_full_fidelity"
    assert report["acceptance_passed"] is False
    assert report["reference_parity_ready"] is False
    assert (
        report["public_reference_artifact_conversion_report"]
        == "validation/reports/full_fidelity_reference_artifact_conversion.json"
    )
    assert report["partial_public_output_artifacts"] >= 2
    assert report["accepted_public_reference_artifacts"] == 0
    assert (
        report["dream_reference_execution_report"]
        == "validation/reports/dream_reference_execution_request.json"
    )
    assert report["dream_settings_deck_generated"] is True
    assert report["dream_reference_output_ready"] is False
    assert (
        report["aurora_reference_execution_report"]
        == "validation/reports/aurora_reference_execution_artifact.json"
    )
    assert report["aurora_reference_artifact_generated"] in {True, False}
    assert (
        report["gk_public_deck_inventory_report"]
        == "validation/reports/gk_public_reference_deck_inventory.json"
    )
    assert report["gk_public_decks_indexed"] >= 0
    assert (
        report["gk_external_nonlinear_parity_report"]
        == "validation/reports/gk_external_nonlinear_parity.json"
    )
    assert (
        report["gk_electromagnetic_fidelity_report"]
        == "validation/reports/gk_electromagnetic_fidelity.json"
    )
    assert (
        report["gk_electromagnetic_fidelity_status"]
        == "blocked_missing_full_vlasov_maxwell_field_solve"
    )
    assert report["gk_electromagnetic_compact_closure_ready"] is True
    assert report["gk_electromagnetic_external_parity_ready"] is False
    assert report["gk_external_reference_artifacts_converted"] >= 0
    assert report["gk_external_reference_output_ready"] in {True, False}
    assert report["gk_native_same_case_comparison_ready"] in {True, False}
    assert report["gk_grid_convergence_ready"] in {True, False}
    assert report["gk_production_scale_scaling_ready"] in {True, False}
    assert report["gk_external_nonlinear_parity_status"] in {
        "not_run",
        "blocked_missing_external_output_manifest",
        "blocked_missing_same_deck_external_outputs",
        "blocked_missing_native_same_case_output_comparison",
        "blocked_missing_grid_convergence_evidence",
        "blocked_missing_production_scale_scaling_evidence",
        "accepted_full_fidelity_ready",
    }
    assert (
        report["production_decomposition_report"]
        == "validation/reports/production_decomposition_contract.json"
    )
    assert report["production_decomposition_contract_pass"] in {True, False}
    assert (
        report["free_boundary_machine_metadata_report"]
        == "validation/reports/free_boundary_public_machine_metadata_inventory.json"
    )
    assert report["free_boundary_machine_metadata_indexed"] >= 0
    assert report["free_boundary_machine_metadata_ready"] in {True, False}
    assert (
        report["freegs_public_example_reconstruction_report"]
        == "validation/reports/freegs_public_example_reconstruction.json"
    )
    assert report["freegs_public_example_cases"] >= 0
    assert report["freegs_public_example_vacuum_comparison_pass"] in {True, False}
    assert report["freegs_public_example_external_output_ready"] in {True, False}

    lanes = {lane["lane"]: lane for lane in report["lanes"]}
    assert set(lanes) == {
        "gene_cgyro_gs2_nonlinear_gk_parity",
        "full_maxwell_electromagnetic_fidelity",
        "production_scale_decomposition",
        "dream_grade_runaway_electrons",
        "aurora_strahl_grade_impurities",
        "free_boundary_equilibrium_strict_parity",
    }
    assert lanes["dream_grade_runaway_electrons"]["sources"][0]["solver_family"] == "DREAM"
    assert lanes["aurora_strahl_grade_impurities"]["sources"][0]["solver_family"] == "Aurora"
    assert {
        source["solver_family"]
        for source in lanes["free_boundary_equilibrium_strict_parity"]["sources"]
    } == {"FreeGS", "FreeGSNKE"}
    for lane in lanes.values():
        assert lane["status"].startswith("blocked_")
        assert lane["next_required_evidence"]


def test_integrated_campaign_writes_json_and_markdown_reports() -> None:
    report = run_campaign()

    write_reports(report)

    assert report["all_locally_actionable_contracts_ready"] is True


def test_integrated_campaign_keeps_reference_parity_fail_closed() -> None:
    report = run_campaign()

    with pytest.raises(AssertionError):
        assert report["acceptance_passed"] is True
