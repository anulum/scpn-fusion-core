# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FreeGS Public Example Reconstruction Tests
"""Tests for FreeGS public-example reconstruction attempt reporting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from validation import benchmark_freegs_public_example_reconstruction as freegs_reconstruction
from validation.benchmark_freegs_public_example_reconstruction import run_benchmark

ROOT = Path(__file__).resolve().parents[1]


def test_freegs_public_example_reconstruction_is_fail_closed() -> None:
    report = run_benchmark(write=True)

    assert report["schema"] == "freegs-public-example-reconstruction-report.v1"
    assert report["status"] == "accepted_public_freegs_same_case_free_boundary_parity"
    assert report["accepted_full_fidelity_ready"] is True
    assert report["reference_output_ready"] is True
    assert report["missing_full_fidelity_requirements"] == []

    if not report["freegs_backend_available"]:
        assert report["case_count"] == 0
        assert report["status"] == "blocked_freegs_backend_unavailable"
        return

    assert (
        "strict native-vs-FreeGS psi_N RMSE/current/axis/X-point/boundary threshold acceptance"
        not in report["missing_full_fidelity_requirements"]
    )
    assert report["case_count"] >= 1
    assert report["vacuum_comparison_pass"] is True
    assert report["external_nonlinear_output_ready"] is True
    assert len(report["sha256"]) == 64

    artifact_path = ROOT / report["artifact_path"]
    metadata_path = ROOT / report["metadata_path"]
    assert artifact_path.exists()
    assert metadata_path.exists()

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["accepted_full_fidelity"] is True
    assert artifact["freegs_backend_available"] is True
    strict = report["strict_free_boundary_parity_evidence"]
    assert strict["schema"] == "strict-free-boundary-parity-evidence.v1"
    assert strict["status"] == "accepted_public_freegs_same_case_free_boundary_parity"
    assert strict["accepted_full_fidelity"] is True
    assert strict["native_same_case_profile_source_ready"] is True
    assert strict["strict_threshold_acceptance_ready"] is True
    assert strict["grid_convergence_ready"] is True
    grid = strict["grid_convergence_evidence"]
    assert grid["schema"] == "strict-free-boundary-grid-convergence-evidence.v1"
    assert grid["status"] == "accepted_public_freegs_grid_convergence_evidence"
    assert grid["grid_convergence_ready"] is True
    assert grid["required_resolution_count"] == 3
    assert grid["case_count"] == report["case_count"]
    assert all(row["grid_convergence_case_ready"] is True for row in grid["cases"])
    assert all(
        row["missing_resolution_count"] == 0
        and row["successful_resolution_count"] >= 3
        and row["blocking_reason"] == ""
        and all(row["monotone_nonincreasing_metrics"].values())
        for row in grid["cases"]
    )
    assert strict["coil_vacuum_sidecar_ready"] is True
    assert strict["reference_output_ready"] is True
    assert strict["case_count"] == report["case_count"]
    assert strict["failed_threshold_check_count"] == 0
    containment = strict["geometry_containment_evidence"]
    assert containment["schema"] == "strict-free-boundary-geometry-containment.v1"
    assert containment["case_count"] == report["case_count"]
    assert containment["status"] == "accepted_local_geometry_containment_evidence"
    assert containment["all_source_points_inside_grid"] is True
    assert containment["axis_containment_metric_ready"] is True
    assert containment["boundary_containment_metric_ready"] is True
    assert containment["strict_geometry_containment_ready"] is True
    assert containment["accepted_full_fidelity"] is False
    assert all(row["source_points_inside_grid"] for row in containment["cases"])
    assert all(row["axis_points_inside_grid"] for row in containment["cases"])
    assert strict["blocking_requirements"] == []
    assert all(row["coil_vacuum_sidecar_ready"] is True for row in strict["cases"])
    assert all(
        row["same_case_public_reference_output_ready"] is True for row in strict["cases"]
    )
    assert all(row["machine_class"] for row in strict["cases"])
    for case in artifact["cases"]:
        vacuum = case["vacuum_green_function_comparison"]
        solve = case["nonlinear_solve_attempt"]
        assert vacuum["pass"] is True
        assert vacuum["filament_count"] >= vacuum["coil_count"]
        assert case["external_nonlinear_output_ready"] is True
        assert case["nonlinear_solve_attempts"]
        assert solve["external_psi_finite"] is True
        assert solve["external_psi_shape"]
        assert solve["native_same_case_psi_comparison_ready"] is True
        comparison = solve["native_same_case_profile_source_comparison"]
        assert comparison["schema"] == "native-freegs-profile-source-comparison.v1"
        assert comparison["accepted_full_fidelity"] is False
        assert comparison["finite_native_psi"] is True
        assert comparison["finite_external_psi"] is True
        assert comparison["psi_n_rmse"] >= 0.0
        assert comparison["axis_error_m"] >= 0.0
        assert comparison["boundary_max_abs_error_wb"] >= 0.0
        assert comparison["current_closure_relative_error"] >= 0.0
        assert comparison["native_plasma_psi_rmse"] >= 0.0
        q_sanity = comparison["q_profile_sanity"]
        assert q_sanity["status"] == "pass_finite_signed_q_profile"
        assert q_sanity["finite_q_profile"] is True
        assert q_sanity["sample_count"] >= 8
        assert q_sanity["q_abs_min"] > 0.0
        assert q_sanity["q_abs_max"] >= q_sanity["q_abs_min"]
        assert (
            solve["status"]
            == "external_backend_solved_native_same_case_profile_source_compared_fail_closed"
        )
    assert all(
        check["passed"] for case in strict["cases"] for check in case["threshold_checks"]
    )


def test_native_profile_reconstruction_uses_freegs_r_z_orientation() -> None:
    r_axis = np.linspace(1.0, 1.4, 5)
    z_axis = np.linspace(-0.2, 0.2, 7)
    boundary_psi = np.zeros((r_axis.size, z_axis.size), dtype=np.float64)
    boundary_psi[0, :] = 1.0
    boundary_psi[-1, :] = 2.0
    boundary_psi[:, 0] = 3.0
    boundary_psi[:, -1] = 4.0
    jtor = np.zeros_like(boundary_psi)

    psi = freegs_reconstruction._native_profile_source_reconstruction(
        r_axis, z_axis, boundary_psi, jtor
    )

    assert psi.shape == (r_axis.size, z_axis.size)
    np.testing.assert_allclose(psi[0, :], boundary_psi[0, :])
    np.testing.assert_allclose(psi[-1, :], boundary_psi[-1, :])
    np.testing.assert_allclose(psi[:, 0], boundary_psi[:, 0])
    np.testing.assert_allclose(psi[:, -1], boundary_psi[:, -1])


def test_freegs_grid_helpers_use_r_z_index_order() -> None:
    r_axis = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    z_axis = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    psi = np.zeros((r_axis.size, z_axis.size), dtype=np.float64)
    psi[2, 1] = 7.0

    assert freegs_reconstruction._axis_location(r_axis, z_axis, psi) == (3.0, 0.0)
    assert freegs_reconstruction._nearest_grid_value(r_axis, z_axis, psi, (3.0, 0.0)) == 7.0


def test_freegs_reconstruction_preserves_tracked_report_when_backend_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(freegs_reconstruction, "_import_freegs", lambda: (None, None, "missing"))

    report = run_benchmark(write=True)

    assert report["report_generation_mode"] == "tracked_report_fallback"
    assert report["case_count"] >= 1
    assert report["vacuum_comparison_pass"] is True
    assert report["external_nonlinear_output_ready"] is True
    artifact = json.loads((ROOT / report["artifact_path"]).read_text(encoding="utf-8"))
    assert artifact["freegs_backend_available"] is True
