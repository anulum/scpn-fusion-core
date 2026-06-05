"""Tests for public full-fidelity reference artifact conversion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import convert_full_fidelity_reference_artifacts as conversion
from tools.convert_full_fidelity_reference_artifacts import run_conversion

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CASES = ROOT / "validation" / "reference_data" / "full_fidelity_reference_cases.json"


def test_converter_exports_real_public_output_payloads_with_checksums() -> None:
    report = run_conversion(write=True)

    assert report["schema"] == "full-fidelity-reference-artifact-conversion.v1"
    assert report["accepted_full_fidelity_artifacts"] == 1
    assert report["partial_output_artifacts"] >= 2
    assert report["status"] == "accepted_public_reference_artifact_available"

    converted = {artifact["artifact_id"]: artifact for artifact in report["converted_artifacts"]}
    assert "aurora_argon_transport_public" in converted
    assert "dream_avalanche_public_raw" in converted
    assert "freegsnke_static_inverse_baseline_public" in converted
    assert "freegsnke_mastu_current_sidecars_public" in converted
    assert converted["aurora_argon_transport_public"]["accepted_full_fidelity"] is True
    assert converted["aurora_argon_transport_public"]["missing_required_observables"] == []
    assert converted["aurora_argon_transport_public"]["solver_output_comparison_ready"] is True
    assert {
        "convection_m_s_r_z",
        "diffusion_m2_s_r_z",
        "effective_source_m3_s_t_r_z",
        "electron_density_t_r_m3",
        "electron_temperature_t_r_ev",
        "ionisation_coeff_m3_s_t_r_z",
        "line_radiation_coeff_w_m3_t_r_z",
        "recombination_coeff_m3_s_t_r_z",
        "source_sink_matrix_t_r_z_z",
    }.issubset(set(converted["aurora_argon_transport_public"]["available_observables"]))

    for artifact in converted.values():
        path = ROOT / artifact["artifact_path"]
        metadata_path = ROOT / artifact["metadata_path"]
        assert path.exists()
        assert metadata_path.exists()
        assert artifact["redistribution_license"]
        assert artifact["provenance_url"].startswith("https://")
        assert len(artifact["sha256"]) == 64
        assert artifact["finite_numeric_payload"] is True
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as payload:
                assert payload.files
                for name in payload.files:
                    array = np.asarray(payload[name], dtype=float)
                    assert array.size > 0
                    assert np.all(np.isfinite(array))
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["schema"] == "freegsnke-current-sidecars.v1"
            assert payload["cases"]
            for case in payload["cases"]:
                currents = np.asarray([row["current_a"] for row in case["coil_currents"]])
                assert currents.size > 0
                assert np.all(np.isfinite(currents))


def test_converter_keeps_partial_outputs_out_of_acceptance_manifest() -> None:
    report = run_conversion(write=True)
    manifest = json.loads(REFERENCE_CASES.read_text(encoding="utf-8"))

    assert report["accepted_full_fidelity_artifacts"] == 1
    impurity_case = manifest["surfaces"]["impurity_transport"]["required_cases"][0]
    assert impurity_case["status"] == "available"
    assert impurity_case["artifact_path"].endswith("aurora_argon_transport_public.npz")
    assert len(impurity_case["sha256"]) == 64

    for surface_name, surface in manifest["surfaces"].items():
        if surface_name == "impurity_transport":
            continue
        for case in surface["required_cases"]:
            assert case["status"] == "missing_public_artifact"
            assert case["artifact_path"] is None
            assert case["sha256"] is None


def test_converter_reports_missing_solver_output_mappings() -> None:
    report = run_conversion(write=True)

    blockers = {blocker["surface"]: blocker for blocker in report["blocking_sources"]}
    assert "native_nonlinear_gyrokinetics" in blockers
    assert "impurity_transport" not in blockers
    assert "free_boundary_equilibrium" in blockers

    assert "nonlinear output" in blockers["native_nonlinear_gyrokinetics"]["reason"]
    assert "strict FreeGS" in blockers["free_boundary_equilibrium"]["reason"]


def test_converter_uses_tracked_artifacts_when_external_cache_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(conversion, "CACHE_ROOT", Path("/nonexistent/full_fidelity_public_sources"))

    report = run_conversion(write=False)

    assert report["partial_output_artifacts"] >= 2
    assert report["accepted_full_fidelity_artifacts"] == 1
    assert "tracked_artifact_fallback" in report["conversion_modes"]
