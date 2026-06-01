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

import pytest

from validation import benchmark_freegs_public_example_reconstruction as freegs_reconstruction
from validation.benchmark_freegs_public_example_reconstruction import run_benchmark

ROOT = Path(__file__).resolve().parents[1]


def test_freegs_public_example_reconstruction_is_fail_closed() -> None:
    report = run_benchmark(write=True)

    assert report["schema"] == "freegs-public-example-reconstruction-report.v1"
    assert report["status"].startswith("blocked_")
    assert report["accepted_full_fidelity_ready"] is False
    assert report["reference_output_ready"] is False
    assert report["missing_full_fidelity_requirements"]

    if not report["freegs_backend_available"]:
        assert report["case_count"] == 0
        assert report["status"] == "blocked_freegs_backend_unavailable"
        return

    assert report["case_count"] >= 1
    assert report["vacuum_comparison_pass"] is True
    assert report["external_nonlinear_output_ready"] is True
    assert len(report["sha256"]) == 64

    artifact_path = ROOT / report["artifact_path"]
    metadata_path = ROOT / report["metadata_path"]
    assert artifact_path.exists()
    assert metadata_path.exists()

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["accepted_full_fidelity"] is False
    assert artifact["freegs_backend_available"] is True
    strict = report["strict_free_boundary_parity_evidence"]
    assert strict["schema"] == "strict-free-boundary-parity-evidence.v1"
    assert strict["status"] == "blocked_strict_thresholds_or_grid_convergence_missing"
    assert strict["native_same_case_profile_source_ready"] is True
    assert strict["strict_threshold_acceptance_ready"] is False
    assert strict["grid_convergence_ready"] is False
    assert strict["coil_vacuum_sidecar_ready"] is False
    assert strict["accepted_full_fidelity"] is False
    assert strict["case_count"] == report["case_count"]
    assert strict["failed_threshold_check_count"] >= 1
    assert strict["blocking_requirements"] == [
        "strict native-vs-FreeGS psi_N RMSE/current/axis/X-point/boundary threshold acceptance",
        "grid convergence across public example resolutions",
        "coil/vacuum reconstruction linked to public machine current sidecars",
    ]
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
    assert any(
        check["metric"] in {"psi_n_rmse", "axis_error_m", "xpoint_psi_n_error_max"}
        and not check["passed"]
        for case in strict["cases"]
        for check in case["threshold_checks"]
    )


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
