# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
"""Tests for integrated full-fidelity campaign reporting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import validation.full_fidelity_end_to_end_campaign as campaign
from validation.full_fidelity_end_to_end_campaign import (
    LEDGER_JSON_REPORT,
    LEDGER_MD_REPORT,
    ROOT,
    build_public_ledger,
    render_public_ledger_markdown,
    run_campaign,
    write_reports,
)


def test_integrated_campaign_reports_all_declared_blockers() -> None:
    report = run_campaign()

    assert report["schema"] == "full-fidelity-end-to-end-campaign.v1"
    assert report["status"] == "not_full_fidelity"
    assert report["acceptance_passed"] is False
    assert report["sas_dataset_readiness_report"] == "validation/reports/sas_dataset_readiness.json"
    assert report["sas_dataset_readiness_status"] in {
        "blocked_missing_required_external_parity_datasets",
        "blocked_missing_sas_dataset_manifest",
        "blocked_incomplete_sas_dataset_manifest",
        "accepted_full_fidelity_dataset_ready",
    }
    assert report["sas_dataset_available_entries"] >= 0
    assert report["sas_dataset_blocked_entries"] >= 0
    assert report["sas_dataset_checksum_rows"] >= 0
    assert report["sas_dataset_external_parity_outputs_ready"] in {True, False}
    assert report["sas_dataset_accepted_full_fidelity_ready"] is False
    assert report["sas_dataset_next_required_evidence"]
    assert report["reference_parity_ready"] is False
    assert (
        report["public_reference_artifact_conversion_report"]
        == "validation/reports/full_fidelity_reference_artifact_conversion.json"
    )
    assert report["partial_public_output_artifacts"] >= 2
    assert report["accepted_public_reference_artifacts"] == 1
    assert (
        report["dream_reference_execution_report"]
        == "validation/reports/dream_reference_execution_request.json"
    )
    assert report["dream_settings_deck_generated"] is True
    assert report["dream_reference_output_ready"] is False
    assert report["runaway_native_kinetic_operator_evidence_ready"] is True
    assert report["runaway_full_momentum_pitch_radius_operator_ready"] is False
    assert report["runaway_dream_same_case_threshold_ready"] is False
    assert report["runaway_source_term_budget_evidence_ready"] is True
    assert report["runaway_source_term_budget_dream_same_case_ready"] is False
    assert (
        report["runaway_kinetic_operator_evidence_status"]
        == "blocked_native_projection_artifact_not_full_dream_operator"
    )
    assert (
        report["aurora_reference_execution_report"]
        == "validation/reports/aurora_reference_execution_artifact.json"
    )
    assert report["aurora_reference_artifact_generated"] in {True, False}
    assert report["impurity_native_transport_evidence_ready"] is True
    assert report["impurity_charge_state_radial_transport_operator_ready"] is True
    assert report["impurity_aurora_strahl_same_case_comparison_ready"] is True
    assert report["impurity_aurora_strahl_same_case_threshold_ready"] is True
    assert report["impurity_aurora_strahl_same_case_threshold_passed"] is True
    assert (
        report["impurity_aurora_strahl_same_case_comparison_status"]
        == "accepted_native_aurora_effective_transport_closure_thresholds"
    )
    assert report["impurity_source_sink_budget_evidence_ready"] is True
    assert report["impurity_source_sink_budget_aurora_strahl_same_case_ready"] is True
    assert (
        report["impurity_transport_operator_evidence_status"]
        == "accepted_native_effective_transport_source_sink_closure"
    )
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
        report["gk_electromagnetic_fidelity_status"] == "blocked_missing_external_em_parity_outputs"
    )
    assert report["gk_electromagnetic_compact_closure_ready"] is True
    assert report["gk_electromagnetic_grid_convergence_ready"] is True
    assert report["gk_electromagnetic_maxwell_evolution_ready"] is True
    assert report["gk_electromagnetic_native_same_case_thresholds_ready"] is True
    assert report["gk_electromagnetic_self_consistent_kinetic_current_ready"] is False
    assert report["gk_electromagnetic_external_parity_ready"] is False
    assert report["gk_external_reference_artifacts_converted"] >= 0
    assert report["gk_external_reference_output_ready"] in {True, False}
    assert report["gk_native_same_case_comparison_ready"] in {True, False}
    assert report["gk_grid_convergence_ready"] in {True, False}
    assert report["gk_production_scale_scaling_ready"] in {True, False}
    assert report["gk_evidence_package_ready"] in {True, False}
    assert report["gk_external_nonlinear_parity_status"] in {
        "not_run",
        "blocked_missing_external_output_manifest",
        "blocked_missing_same_deck_external_outputs",
        "blocked_same_deck_identity_mismatch",
        "blocked_missing_native_same_case_output_comparison",
        "blocked_missing_grid_convergence_evidence",
        "blocked_missing_production_scale_scaling_evidence",
        "blocked_incomplete_evidence_package",
        "accepted_full_fidelity_ready",
    }
    assert (
        report["production_decomposition_report"]
        == "validation/reports/production_decomposition_contract.json"
    )
    assert report["production_decomposition_contract_pass"] in {True, False}
    assert report["production_decomposition_halo_face_integrity_ready"] is True
    assert report["production_decomposition_distributed_halo_exchange_ready"] is False
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
    assert (
        report["free_boundary_strict_parity_report"]
        == "validation/reports/free_boundary_strict_parity_benchmark.json"
    )
    assert report["freegs_public_example_cases"] >= 0
    assert report["freegs_public_example_vacuum_comparison_pass"] in {True, False}
    assert report["freegs_public_example_external_output_ready"] in {True, False}
    assert report["free_boundary_strict_threshold_acceptance_ready"] is True
    assert report["free_boundary_geometry_containment_ready"] is True
    assert report["free_boundary_boundary_containment_metric_ready"] is True
    assert report["free_boundary_grid_convergence_ready"] is True
    assert report["free_boundary_coil_vacuum_sidecar_ready"] is True
    assert report["free_boundary_same_case_public_reference_output_ready"] is True
    assert report["free_boundary_failed_threshold_check_count"] >= 0
    assert report["free_boundary_strict_parity_blockers"] == []
    assert (
        report["free_boundary_strict_parity_status"]
        == "accepted_full_fidelity_free_boundary_parity"
    )

    lanes = {lane["lane"]: lane for lane in report["lanes"]}
    assert set(lanes) == {
        "sas_dataset_readiness",
        "gene_cgyro_gs2_nonlinear_gk_parity",
        "full_maxwell_electromagnetic_fidelity",
        "production_scale_decomposition",
        "dream_grade_runaway_electrons",
        "aurora_strahl_grade_impurities",
        "free_boundary_equilibrium_strict_parity",
    }
    assert lanes["sas_dataset_readiness"]["surface"] == "external_reference_data"
    assert lanes["sas_dataset_readiness"]["status"].startswith("blocked_")
    assert lanes["dream_grade_runaway_electrons"]["sources"][0]["solver_family"] == "DREAM"
    assert lanes["aurora_strahl_grade_impurities"]["sources"][0]["solver_family"] == "Aurora"
    assert lanes["aurora_strahl_grade_impurities"]["reference_cases_ready"] is True
    assert (
        lanes["aurora_strahl_grade_impurities"]["status"]
        == "accepted_native_aurora_effective_transport_closure_thresholds"
    )
    assert {
        source["solver_family"]
        for source in lanes["free_boundary_equilibrium_strict_parity"]["sources"]
    } == {"FreeGS", "FreeGSNKE"}
    assert (
        lanes["free_boundary_equilibrium_strict_parity"]["status"]
        == "accepted_full_fidelity_free_boundary_parity"
    )
    assert lanes["free_boundary_equilibrium_strict_parity"]["reference_cases_ready"] is True
    assert lanes["free_boundary_equilibrium_strict_parity"]["next_required_evidence"] == []
    for lane in lanes.values():
        assert lane["status"].startswith("blocked_") or lane["lane"] in {
            "aurora_strahl_grade_impurities",
            "free_boundary_equilibrium_strict_parity",
        }
        if lane["lane"] != "free_boundary_equilibrium_strict_parity":
            assert lane["next_required_evidence"]


def test_integrated_campaign_writes_json_and_markdown_reports() -> None:
    report = run_campaign()

    write_reports(report)

    assert report["all_locally_actionable_contracts_ready"] is all(
        bool(lane["locally_actionable_contract_ready"]) for lane in report["lanes"]
    )
    assert LEDGER_JSON_REPORT.is_file()
    assert LEDGER_MD_REPORT.is_file()


def test_public_ledger_is_generated_from_campaign_lanes() -> None:
    report = run_campaign()
    write_reports(report)
    ledger = build_public_ledger(report)

    assert ledger["schema"] == "full-fidelity-validation-ledger.v1"
    assert ledger["status"] == "not_full_fidelity"
    assert ledger["acceptance_passed"] is False
    assert ledger["ledger_publication_ready"] is False
    assert ledger["campaign_report"] == "validation/reports/full_fidelity_end_to_end_campaign.json"
    assert ledger["blocked_lane_count"] == 6
    assert ledger["accepted_full_fidelity_lane_count"] == 1

    rows = {row["lane"]: row for row in ledger["lanes"]}
    assert set(rows) == {lane["lane"] for lane in report["lanes"]}
    assert rows["free_boundary_equilibrium_strict_parity"]["accepted_full_fidelity_lane"] is True
    assert (
        rows["free_boundary_equilibrium_strict_parity"]["blocked_for_public_full_fidelity"] is False
    )
    assert rows["gene_cgyro_gs2_nonlinear_gk_parity"]["blocked_for_public_full_fidelity"] is True
    assert rows["gene_cgyro_gs2_nonlinear_gk_parity"]["public_source_licenses_ready"] is False
    assert (
        rows["gene_cgyro_gs2_nonlinear_gk_parity"]["public_sources"][0]["redistribution_license"]
        == "not_declared_in_source_registry"
    )


def test_public_ledger_records_source_report_checksums() -> None:
    report = run_campaign()
    write_reports(report)
    ledger = json.loads(LEDGER_JSON_REPORT.read_text(encoding="utf-8"))
    source_reports = {row["path"]: row for row in ledger["source_reports"]}
    campaign_path = ROOT / "validation/reports/full_fidelity_end_to_end_campaign.json"
    digest = hashlib.sha256(campaign_path.read_bytes()).hexdigest()

    assert (
        source_reports["validation/reports/full_fidelity_end_to_end_campaign.json"]["sha256"]
        == digest
    )
    assert (
        source_reports["validation/reports/full_fidelity_end_to_end_campaign.json"]["schema"]
        == "full-fidelity-end-to-end-campaign.v1"
    )
    assert (
        source_reports["validation/reports/full_fidelity_acceptance_benchmark.json"]["schema"]
        == "full-fidelity-acceptance.v1"
    )
    assert all(row["exists"] for row in source_reports.values())


def test_public_ledger_markdown_names_fail_closed_boundary() -> None:
    report = run_campaign()
    write_reports(report)
    ledger = build_public_ledger(report)
    rendered = render_public_ledger_markdown(ledger)

    assert "# Full-Fidelity Validation Ledger" in rendered
    assert "publication readiness remains false" in rendered
    assert "`blocked_missing_external_output_manifest`" in rendered
    assert "`accepted_full_fidelity_free_boundary_parity`" in rendered


def test_public_ledger_source_report_records_fail_closed_missing_and_invalid_inputs() -> None:
    missing = campaign._source_report_record("validation/reports/not_a_report.json")
    assert missing["exists"] is False
    assert missing["sha256"] is None

    rel_dir = ROOT / "artifacts" / "_tmp_full_fidelity_ledger_tests"
    rel_dir.mkdir(parents=True, exist_ok=True)
    text_report = rel_dir / "report.md"
    text_report.write_text("report", encoding="utf-8")
    invalid = rel_dir / "invalid.json"
    invalid.write_text("{invalid", encoding="utf-8")
    try:
        text_record = campaign._source_report_record(text_report.relative_to(ROOT).as_posix())
        invalid_record = campaign._source_report_record(invalid.relative_to(ROOT).as_posix())
    finally:
        text_report.unlink(missing_ok=True)
        invalid.unlink(missing_ok=True)
        rel_dir.rmdir()

    assert text_record["exists"] is True
    assert text_record["schema"] is None
    assert invalid_record["exists"] is True
    assert invalid_record["status"] == "invalid_json"
    assert invalid_record["json_error"]


def test_loader_defaults_remain_fail_closed_when_optional_reports_are_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"

    monkeypatch.setattr(campaign, "PUBLIC_SOURCE_DOWNLOADS", missing)
    assert campaign._load_downloads()["all_reachable_downloads_completed"] is False

    monkeypatch.setattr(campaign, "REFERENCE_ARTIFACT_CONVERSION", missing)
    assert campaign._load_conversion()["accepted_full_fidelity_artifacts"] == 0

    monkeypatch.setattr(campaign, "DREAM_EXECUTION_REQUEST", missing)
    assert campaign._load_dream_execution()["status"] == "not_run"

    monkeypatch.setattr(campaign, "AURORA_EXECUTION_ARTIFACT", missing)
    assert campaign._load_aurora_execution()["reference_output_ready"] is False

    monkeypatch.setattr(campaign, "GK_DECK_INVENTORY", missing)
    assert campaign._load_gk_deck_inventory()["deck_count"] == 0

    monkeypatch.setattr(campaign, "GK_EXTERNAL_PARITY", missing)
    assert campaign._load_gk_external_parity()["reference_output_ready"] is False

    monkeypatch.setattr(campaign, "PRODUCTION_DECOMPOSITION", missing)
    assert campaign._load_production_decomposition()["production_scale_ready"] is False

    monkeypatch.setattr(campaign, "FREE_BOUNDARY_MACHINE_METADATA", missing)
    assert campaign._load_free_boundary_machine_metadata()["machine_metadata_ready"] is False

    monkeypatch.setattr(campaign, "FREEGS_PUBLIC_RECONSTRUCTION", missing)
    assert campaign._load_freegs_public_reconstruction()["case_count"] == 0

    monkeypatch.setattr(campaign, "FREE_BOUNDARY_STRICT_PARITY", missing)
    assert campaign._load_free_boundary_strict_parity()["accepted_full_fidelity"] is False


def test_loader_schema_guards_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "wrong", "sources": []}), encoding="utf-8")

    loader_cases = [
        ("PUBLIC_SOURCES", campaign._load_sources, "public source registry"),
        ("PUBLIC_SOURCE_DOWNLOADS", campaign._load_downloads, "public source download"),
        ("REFERENCE_ARTIFACT_CONVERSION", campaign._load_conversion, "reference artifact"),
        ("DREAM_EXECUTION_REQUEST", campaign._load_dream_execution, "DREAM"),
        ("AURORA_EXECUTION_ARTIFACT", campaign._load_aurora_execution, "Aurora"),
        ("GK_DECK_INVENTORY", campaign._load_gk_deck_inventory, "GK public"),
        ("GK_EXTERNAL_PARITY", campaign._load_gk_external_parity, "GK external"),
        ("PRODUCTION_DECOMPOSITION", campaign._load_production_decomposition, "production"),
        (
            "FREE_BOUNDARY_MACHINE_METADATA",
            campaign._load_free_boundary_machine_metadata,
            "free-boundary",
        ),
        ("FREEGS_PUBLIC_RECONSTRUCTION", campaign._load_freegs_public_reconstruction, "FreeGS"),
        (
            "FREE_BOUNDARY_STRICT_PARITY",
            campaign._load_free_boundary_strict_parity,
            "strict parity",
        ),
    ]
    for attr, loader, match in loader_cases:
        monkeypatch.setattr(campaign, attr, bad)
        with pytest.raises(ValueError, match=match):
            loader()

    bad.write_text(
        json.dumps({"schema": "full-fidelity-public-source-registry.v1", "sources": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(campaign, "PUBLIC_SOURCES", bad)
    with pytest.raises(ValueError, match="must define sources"):
        campaign._load_sources()


def test_acceptance_surface_and_main_fail_closed_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(KeyError, match="missing"):
        campaign._acceptance_surface({"surfaces": [{"surface": "present"}]}, "missing")

    written: list[dict[str, object]] = []

    def fake_run_campaign() -> dict[str, object]:
        return {"schema": "fixture", "status": "not_full_fidelity"}

    def fake_write_reports(report: dict[str, object]) -> None:
        written.append(report)

    monkeypatch.setattr(campaign, "run_campaign", fake_run_campaign)
    monkeypatch.setattr(campaign, "write_reports", fake_write_reports)

    assert campaign.main() == 0
    assert written == [{"schema": "fixture", "status": "not_full_fidelity"}]


def test_integrated_campaign_keeps_reference_parity_fail_closed() -> None:
    report = run_campaign()

    with pytest.raises(AssertionError):
        assert report["acceptance_passed"] is True
