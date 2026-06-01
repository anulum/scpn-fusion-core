"""Tests for the strict free-boundary parity benchmark."""

from __future__ import annotations

from pathlib import Path

from validation.benchmark_free_boundary_strict_parity import (
    evaluate_strict_parity,
    render_markdown,
    run_benchmark,
)


def _passing_freegs_report() -> dict:
    return {
        "case_count": 1,
        "external_nonlinear_output_ready": True,
        "strict_free_boundary_parity_evidence": {
            "coil_vacuum_sidecar_ready": True,
            "failed_threshold_check_count": 0,
            "grid_convergence_ready": True,
            "native_same_case_profile_source_ready": True,
            "strict_threshold_acceptance_ready": True,
            "geometry_containment_evidence": {
                "boundary_containment_metric_ready": True,
                "strict_geometry_containment_ready": True,
            },
            "grid_convergence_evidence": {
                "required_resolution_count": 3,
                "schema": "strict-free-boundary-grid-convergence-evidence.v1",
                "status": "accepted_public_freegs_grid_convergence_evidence",
                "cases": [
                    {
                        "case_id": "case",
                        "machine_class": "TestTokamak",
                        "observed_resolution_count": 3,
                        "required_resolution_count": 3,
                        "missing_resolution_count": 0,
                        "grid_convergence_case_ready": True,
                        "blocking_reason": "",
                    }
                ],
            },
            "cases": [
                {
                    "case_id": "case",
                    "external_nonlinear_output_ready": True,
                    "native_same_case_profile_source_ready": True,
                    "strict_threshold_acceptance_ready": True,
                    "threshold_checks": [
                        {
                            "metric": "psi_n_rmse",
                            "passed": True,
                            "value": 0.01,
                            "limit": 0.05,
                            "comparator": "<=",
                        }
                    ],
                }
            ],
        },
    }


def _passing_machine_metadata_report() -> dict:
    return {
        "machine_metadata_ready": True,
        "machine_config_count": 1,
        "machines": ["TestTokamak"],
        "reference_output_ready": True,
        "schema": "free-boundary-public-machine-metadata-inventory-report.v1",
        "status": "accepted_public_machine_metadata",
    }


def test_strict_parity_blocks_current_tracked_reports() -> None:
    report = run_benchmark(write=False)

    assert report["accepted_full_fidelity"] is False
    assert report["status"] == "blocked_free_boundary_strict_parity"
    assert "grid_convergence_evidence_missing" in report["blockers"]
    assert "public_external_coil_vacuum_sidecars_missing" in report["blockers"]
    assert report["failed_threshold_check_count"] >= 1


def test_strict_parity_accepts_only_complete_contract() -> None:
    report = evaluate_strict_parity(
        _passing_freegs_report(),
        _passing_machine_metadata_report(),
        freegs_report_path=Path("validation/reports/freegs_public_example_reconstruction.json"),
        machine_metadata_report_path=Path(
            "validation/reports/free_boundary_public_machine_metadata_inventory.json"
        ),
    )

    assert report["accepted_full_fidelity"] is True
    assert report["blockers"] == []
    assert report["checks"]["coil_vacuum_sidecar_ready"] is True
    assert report["checks"]["grid_convergence_ready"] is True


def test_strict_parity_markdown_exposes_blockers() -> None:
    report = evaluate_strict_parity(
        _passing_freegs_report()
        | {
            "strict_free_boundary_parity_evidence": {
                **_passing_freegs_report()["strict_free_boundary_parity_evidence"],
                "coil_vacuum_sidecar_ready": False,
            }
        },
        _passing_machine_metadata_report(),
        freegs_report_path=Path("validation/reports/freegs_public_example_reconstruction.json"),
        machine_metadata_report_path=Path(
            "validation/reports/free_boundary_public_machine_metadata_inventory.json"
        ),
    )

    markdown = render_markdown(report)

    assert "public_external_coil_vacuum_sidecars_missing" in markdown
    assert "Accepted full fidelity: `False`" in markdown
