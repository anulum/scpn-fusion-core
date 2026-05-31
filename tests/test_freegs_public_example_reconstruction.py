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
    for case in artifact["cases"]:
        vacuum = case["vacuum_green_function_comparison"]
        solve = case["nonlinear_solve_attempt"]
        assert vacuum["pass"] is True
        assert vacuum["filament_count"] >= vacuum["coil_count"]
        assert case["external_nonlinear_output_ready"] is True
        assert case["nonlinear_solve_attempts"]
        assert solve["external_psi_finite"] is True
        assert solve["external_psi_shape"]
        assert solve["native_same_case_psi_comparison_ready"] is False
        assert (
            solve["status"]
            == "external_backend_solved_missing_native_same_case_profile_source_comparison"
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
