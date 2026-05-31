#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for fail-closed nonlinear GK external output parity artefacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from tools import gk_external_output_parity as parity


def _payload(path: Path, scale: float = 1.0) -> None:
    coordinates = {
        "species_index": [0.0, 1.0],
        "kx_rhos": [0.1, 0.2],
        "ky_rhos": [0.05, 0.15],
        "theta_rad": [-1.0, 1.0],
        "vpar_vth": [-2.0, 2.0],
        "mu_normalized": [0.25, 0.75],
        "time_s": [0.0, 1.0],
    }
    distribution = np.arange(64, dtype=float).reshape(2, 2, 2, 2, 2, 2) * scale + 1.0
    spectrum = np.arange(8, dtype=float).reshape(2, 2, 2) * scale + 1.0
    payload = {
        "schema": "gk-nonlinear-external-output.v1",
        "coordinates": coordinates,
        "observables": {
            "nonlinear_distribution_function": distribution.tolist(),
            "nonlinear_distribution_function_imag": (distribution * 0.01).tolist(),
            "ion_heat_flux_spectrum": spectrum.tolist(),
            "electron_heat_flux_spectrum": (spectrum * 1.2).tolist(),
            "zonal_flow_energy": (np.ones((2, 2)) * scale).tolist(),
            "saturated_phi_rms": (np.ones(2) * scale).tolist(),
            "electromagnetic_apar_energy": (spectrum * 0.25).tolist(),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gk_external_output_parity_blocks_without_manifest(tmp_path: Path) -> None:
    report = parity.build_gk_external_output_parity_report(
        source_root=tmp_path / "missing",
        artifact_dir=tmp_path / "artifacts",
        report_dir=tmp_path / "reports",
        write=True,
    )

    assert report["schema"] == "gk-external-nonlinear-output-parity-report.v1"
    assert report["status"] == "blocked_missing_external_output_manifest"
    assert report["accepted_full_fidelity_ready"] is False
    assert report["reference_output_ready"] is False
    assert report["native_same_case_comparison_ready"] is False
    assert report["grid_convergence_ready"] is False
    assert report["production_scale_scaling_ready"] is False
    assert report["required_solver_families"] == ["GENE", "CGYRO", "GS2"]

    rows = {row["solver_family"]: row for row in report["external_output_rows"]}
    assert set(rows) == {"GENE", "CGYRO", "GS2"}
    for row in rows.values():
        assert row["status"].startswith("blocked_")
        assert row["reference_output_ready"] is False
        assert "same_deck_external_nonlinear_output" in row["missing_requirements"]


def test_gk_external_output_parity_converts_valid_public_output(tmp_path: Path) -> None:
    source_root = tmp_path / "external"
    source_root.mkdir()
    output = source_root / "gene_case.json"
    _payload(output)
    manifest = {
        "schema": "gk-nonlinear-external-output-manifest.v1",
        "cases": [
            {
                "case_id": "gene_itg_public",
                "deck_id": "gene_itg_public_deck",
                "solver_family": "GENE",
                "output_path": output.name,
                "provenance_url": "https://example.invalid/gene/gene_itg_public",
                "redistribution_license": "CC-BY-4.0",
                "sha256": _sha256(output),
            }
        ],
        "grid_convergence_evidence": [
            {
                "case_id": "gene_itg_public",
                "observable": "ion_heat_flux_spectrum",
                "coarse_grid": [2, 2, 2],
                "fine_grid": [4, 4, 4],
                "relative_l2": 0.08,
            }
        ],
        "production_scaling_evidence": [
            {
                "case_id": "gene_itg_public",
                "device": "public-cpu-cluster",
                "grid": [2, 2, 2, 2, 2, 2],
                "ranks": 8,
                "wall_time_s": 12.5,
            }
        ],
    }
    (source_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = parity.build_gk_external_output_parity_report(
        source_root=source_root,
        artifact_dir=tmp_path / "artifacts",
        report_dir=tmp_path / "reports",
        write=True,
    )

    rows = {row["solver_family"]: row for row in report["external_output_rows"]}
    gene = rows["GENE"]
    assert gene["reference_output_ready"] is True
    assert gene["status"] == "blocked_missing_native_same_case_output_comparison"
    assert gene["converted_artifact_path"].endswith("gene_itg_public.npz")
    assert len(gene["sha256"]) == 64
    assert report["converted_reference_artifacts"] == 1
    assert report["accepted_full_fidelity_ready"] is False
    assert report["grid_convergence_ready"] is True
    assert report["production_scale_scaling_ready"] is True

    with np.load(tmp_path / gene["converted_artifact_path"], allow_pickle=False) as payload_npz:
        assert "nonlinear_distribution_function" in payload_npz.files
        assert "nonlinear_distribution_function_imag" in payload_npz.files
        assert "ion_heat_flux_spectrum" in payload_npz.files
        assert "time_s" in payload_npz.files


def test_gk_external_output_parity_compares_native_same_case_output(tmp_path: Path) -> None:
    source_root = tmp_path / "external"
    source_root.mkdir()
    reference = source_root / "gs2_reference.json"
    native = source_root / "gs2_native.json"
    _payload(reference, scale=1.0)
    _payload(native, scale=1.0)
    manifest = {
        "schema": "gk-nonlinear-external-output-manifest.v1",
        "cases": [
            {
                "case_id": "gs2_itg_public",
                "deck_id": "gs2_itg_public_deck",
                "solver_family": "GS2",
                "output_path": reference.name,
                "native_output_path": native.name,
                "provenance_url": "https://example.invalid/gs2/gs2_itg_public",
                "redistribution_license": "MIT",
                "sha256": _sha256(reference),
            }
        ],
        "grid_convergence_evidence": [],
        "production_scaling_evidence": [],
    }
    (source_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = parity.build_gk_external_output_parity_report(
        source_root=source_root,
        artifact_dir=tmp_path / "artifacts",
        report_dir=tmp_path / "reports",
        write=True,
    )

    rows = {row["solver_family"]: row for row in report["external_output_rows"]}
    gs2 = rows["GS2"]
    assert gs2["native_same_case_comparison_ready"] is True
    assert gs2["native_same_case_comparison_passed"] is True
    assert gs2["status"] == "native_same_case_comparison_passed"
    assert gs2["threshold_evaluation"]["passed"] is True
    assert report["native_same_case_comparison_ready"] is False
    assert report["accepted_full_fidelity_ready"] is False
