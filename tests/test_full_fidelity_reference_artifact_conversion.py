"""Tests for public full-fidelity reference artifact conversion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.convert_full_fidelity_reference_artifacts import run_conversion

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CASES = ROOT / "validation" / "reference_data" / "full_fidelity_reference_cases.json"


def test_converter_exports_real_public_output_payloads_with_checksums() -> None:
    report = run_conversion(write=True)

    assert report["schema"] == "full-fidelity-reference-artifact-conversion.v1"
    assert report["accepted_full_fidelity_artifacts"] == 0
    assert report["partial_output_artifacts"] >= 2
    assert report["reference_manifest_updated"] is False

    converted = {artifact["artifact_id"]: artifact for artifact in report["converted_artifacts"]}
    assert "dream_avalanche_public_raw" in converted
    assert "freegsnke_static_inverse_baseline_public" in converted

    for artifact in converted.values():
        path = ROOT / artifact["artifact_path"]
        metadata_path = ROOT / artifact["metadata_path"]
        assert path.exists()
        assert metadata_path.exists()
        assert artifact["redistribution_license"]
        assert artifact["provenance_url"].startswith("https://")
        assert len(artifact["sha256"]) == 64
        assert artifact["finite_numeric_payload"] is True
        with np.load(path, allow_pickle=False) as payload:
            assert payload.files
            for name in payload.files:
                array = np.asarray(payload[name], dtype=float)
                assert array.size > 0
                assert np.all(np.isfinite(array))


def test_converter_keeps_partial_outputs_out_of_acceptance_manifest() -> None:
    report = run_conversion(write=True)
    manifest = json.loads(REFERENCE_CASES.read_text(encoding="utf-8"))

    assert report["accepted_full_fidelity_artifacts"] == 0
    for surface in manifest["surfaces"].values():
        for case in surface["required_cases"]:
            assert case["status"] == "missing_public_artifact"
            assert case["artifact_path"] is None
            assert case["sha256"] is None


def test_converter_reports_missing_solver_output_mappings() -> None:
    report = run_conversion(write=True)

    blockers = {blocker["surface"]: blocker for blocker in report["blocking_sources"]}
    assert "native_nonlinear_gyrokinetics" in blockers
    assert "impurity_transport" in blockers
    assert "free_boundary_equilibrium" in blockers

    assert "nonlinear output" in blockers["native_nonlinear_gyrokinetics"]["reason"]
    assert "Aurora/STRAHL output" in blockers["impurity_transport"]["reason"]
    assert "strict FreeGS" in blockers["free_boundary_equilibrium"]["reason"]
