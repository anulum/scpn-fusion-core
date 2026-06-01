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
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tools import gk_external_output_parity as parity

DECK_PHYSICS_SHA256 = "0" * 64


def _payload_maps(
    scale: float = 1.0,
) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]:
    coordinates = {
        "species_index": np.asarray([0.0, 1.0], dtype=np.float64),
        "kx_rhos": np.asarray([0.1, 0.2], dtype=np.float64),
        "ky_rhos": np.asarray([0.05, 0.15], dtype=np.float64),
        "theta_rad": np.asarray([-1.0, 1.0], dtype=np.float64),
        "vpar_vth": np.asarray([-2.0, 2.0], dtype=np.float64),
        "mu_normalized": np.asarray([0.25, 0.75], dtype=np.float64),
        "time_s": np.asarray([0.0, 1.0], dtype=np.float64),
    }
    distribution = np.arange(64, dtype=np.float64).reshape(2, 2, 2, 2, 2, 2) * scale + 1.0
    spectrum = np.arange(8, dtype=np.float64).reshape(2, 2, 2) * scale + 1.0
    observables = {
        "nonlinear_distribution_function": distribution,
        "nonlinear_distribution_function_imag": distribution * 0.01,
        "ion_heat_flux_spectrum": spectrum,
        "electron_heat_flux_spectrum": spectrum * 1.2,
        "zonal_flow_energy": np.ones((2, 2), dtype=np.float64) * scale,
        "saturated_phi_rms": np.ones(2, dtype=np.float64) * scale,
        "electromagnetic_phi_energy": spectrum * 0.5,
        "electromagnetic_apar_energy": spectrum * 0.25,
        "electromagnetic_bpar_energy": spectrum * 0.125,
    }
    return coordinates, observables


def _payload(path: Path, scale: float = 1.0) -> None:
    coordinates, observables = _payload_maps(scale)
    payload = {
        "schema": "gk-nonlinear-external-output.v1",
        "coordinates": {name: value.tolist() for name, value in coordinates.items()},
        "observables": {name: value.tolist() for name, value in observables.items()},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _payload_npz(path: Path, scale: float = 1.0) -> None:
    coordinates, observables = _payload_maps(scale)
    arrays: dict[str, Any] = {}
    arrays.update(coordinates)
    arrays.update(observables)
    np.savez_compressed(path, **arrays)


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
    assert report["same_deck_group_ready"] is False
    assert report["native_same_case_comparison_ready"] is False
    assert report["grid_convergence_ready"] is False
    assert report["production_scale_scaling_ready"] is False
    assert report["required_solver_families"] == ["GENE", "CGYRO", "GS2"]
    assert report["solver_family_completeness_ready"] is False

    rows = {row["solver_family"]: row for row in report["external_output_rows"]}
    assert set(rows) == {"GENE", "CGYRO", "GS2"}
    for row in rows.values():
        assert row["status"].startswith("blocked_")
        assert row["reference_output_ready"] is False
        assert "same_deck_external_nonlinear_output" in row["missing_requirements"]

    matrix = {row["solver_family"]: row for row in report["solver_family_completeness_matrix"]}
    assert set(matrix) == {"GENE", "CGYRO", "GS2"}
    for row in matrix.values():
        assert row["same_deck_reference_output_ready"] is False
        assert row["native_same_case_comparison_ready"] is False
        assert row["complete_required_observables"] is False
        assert set(row["observable_presence"]) == set(report["required_observables"])
        assert not any(row["observable_presence"].values())


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
                "benchmark_case_id": "public_itg_em_same_deck",
                "deck_physics_sha256": DECK_PHYSICS_SHA256,
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
                "solver_family": "GENE",
                "observable": "ion_heat_flux_spectrum",
                "coarse_grid": [2, 2, 2],
                "fine_grid": [4, 4, 4],
                "relative_l2": 0.08,
            }
        ],
        "production_scaling_evidence": [
            {
                "case_id": "gene_itg_public",
                "solver_family": "GENE",
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
    assert report["same_deck_group_ready"] is False
    assert report["grid_convergence_ready"] is False
    assert report["production_scale_scaling_ready"] is False
    assert report["solver_family_completeness_ready"] is False

    completeness = {
        row["solver_family"]: row for row in report["solver_family_completeness_matrix"]
    }
    assert completeness["GENE"]["same_deck_reference_output_ready"] is True
    assert completeness["GENE"]["native_same_case_comparison_ready"] is False
    assert completeness["GENE"]["complete_required_observables"] is True
    assert all(completeness["GENE"]["observable_presence"].values())
    assert completeness["CGYRO"]["same_deck_reference_output_ready"] is False
    assert completeness["GS2"]["same_deck_reference_output_ready"] is False

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
                "benchmark_case_id": "public_itg_em_same_deck",
                "deck_physics_sha256": DECK_PHYSICS_SHA256,
                "solver_family": "GS2",
                "output_path": reference.name,
                "native_output_path": native.name,
                "native_output_sha256": _sha256(native),
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


def test_gk_external_output_parity_accepts_npz_payload_with_separated_metadata(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "external"
    source_root.mkdir()
    reference = source_root / "gene_reference.npz"
    native = source_root / "gene_native.npz"
    _payload_npz(reference, scale=1.0)
    _payload_npz(native, scale=1.0)
    manifest = {
        "schema": "gk-nonlinear-external-output-manifest.v1",
        "cases": [
            {
                "case_id": "gene_itg_npz_public",
                "deck_id": "gene_itg_npz_public_deck",
                "benchmark_case_id": "public_itg_em_same_deck",
                "deck_physics_sha256": DECK_PHYSICS_SHA256,
                "solver_family": "GENE",
                "output_path": reference.name,
                "native_output_path": native.name,
                "native_output_sha256": _sha256(native),
                "provenance_url": "https://example.invalid/gene/gene_itg_npz_public",
                "redistribution_license": "CC-BY-4.0",
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
    gene = rows["GENE"]
    assert gene["reference_output_ready"] is True
    assert gene["native_same_case_comparison_ready"] is True
    assert gene["native_same_case_comparison_passed"] is True
    metadata = json.loads((tmp_path / gene["metadata_path"]).read_text(encoding="utf-8"))
    assert "species_index" in metadata["available_coordinates"]
    assert "ion_heat_flux_spectrum" not in metadata["available_coordinates"]
    assert "ion_heat_flux_spectrum" in metadata["available_observables"]
    assert "species_index" not in metadata["available_observables"]


def test_gk_external_output_parity_blocks_unchecksummed_native_output(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "external"
    source_root.mkdir()
    reference = source_root / "cgyro_reference.json"
    native = source_root / "cgyro_native.json"
    _payload(reference, scale=1.0)
    _payload(native, scale=1.0)
    manifest = {
        "schema": "gk-nonlinear-external-output-manifest.v1",
        "cases": [
            {
                "case_id": "cgyro_itg_public",
                "deck_id": "cgyro_itg_public_deck",
                "benchmark_case_id": "public_itg_em_same_deck",
                "deck_physics_sha256": DECK_PHYSICS_SHA256,
                "solver_family": "CGYRO",
                "output_path": reference.name,
                "native_output_path": native.name,
                "provenance_url": "https://example.invalid/cgyro/cgyro_itg_public",
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
    cgyro = rows["CGYRO"]
    assert cgyro["reference_output_ready"] is True
    assert cgyro["native_same_case_comparison_ready"] is False
    assert cgyro["status"] == "blocked_native_same_case_output_checksum_missing"
    assert cgyro["threshold_evaluation"]["reason"] == "native_output_sha256_missing"
    assert report["accepted_full_fidelity_ready"] is False


def test_gk_external_output_parity_blocks_cross_solver_deck_mismatch(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "external"
    source_root.mkdir()
    cases = []
    for family, suffix, benchmark_case_id, deck_hash in (
        ("GENE", "gene", "public_itg_em_same_deck", "1" * 64),
        ("CGYRO", "cgyro", "public_itg_em_same_deck", "1" * 64),
        ("GS2", "gs2", "different_public_itg_em_deck", "2" * 64),
    ):
        reference = source_root / f"{suffix}_reference.json"
        native = source_root / f"{suffix}_native.json"
        _payload(reference, scale=1.0)
        _payload(native, scale=1.0)
        cases.append(
            {
                "case_id": f"{suffix}_itg_public",
                "deck_id": f"{suffix}_itg_public_deck",
                "benchmark_case_id": benchmark_case_id,
                "deck_physics_sha256": deck_hash,
                "solver_family": family,
                "output_path": reference.name,
                "native_output_path": native.name,
                "native_output_sha256": _sha256(native),
                "provenance_url": f"https://example.invalid/{suffix}/itg_public",
                "redistribution_license": "CC-BY-4.0",
                "sha256": _sha256(reference),
            }
        )
    manifest = {
        "schema": "gk-nonlinear-external-output-manifest.v1",
        "cases": cases,
        "grid_convergence_evidence": [
            {
                "case_id": case["case_id"],
                "solver_family": case["solver_family"],
                "observable": "ion_heat_flux_spectrum",
                "coarse_grid": [2, 2, 2],
                "fine_grid": [4, 4, 4],
                "relative_l2": 0.08,
            }
            for case in cases
        ],
        "production_scaling_evidence": [
            {
                "case_id": case["case_id"],
                "solver_family": case["solver_family"],
                "device": "public-cpu-cluster",
                "grid": [2, 2, 2, 2, 2, 2],
                "ranks": 8,
                "wall_time_s": 12.5,
            }
            for case in cases
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

    assert report["reference_output_ready"] is True
    assert report["native_same_case_comparison_ready"] is True
    assert report["grid_convergence_ready"] is True
    assert report["production_scale_scaling_ready"] is True
    assert report["same_deck_group_ready"] is False
    assert report["same_deck_group"]["reason"] == "same_deck_identity_mismatch"
    assert report["status"] == "blocked_same_deck_identity_mismatch"
    assert report["accepted_full_fidelity_ready"] is False


def test_gk_external_output_parity_blocks_unlinked_convergence_and_scaling_rows(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "external"
    source_root.mkdir()
    cases = []
    for family, suffix in (("GENE", "gene"), ("CGYRO", "cgyro"), ("GS2", "gs2")):
        reference = source_root / f"{suffix}_reference.json"
        native = source_root / f"{suffix}_native.json"
        _payload(reference, scale=1.0)
        _payload(native, scale=1.0)
        cases.append(
            {
                "case_id": f"{suffix}_itg_public",
                "deck_id": f"{suffix}_itg_public_deck",
                "benchmark_case_id": "public_itg_em_same_deck",
                "deck_physics_sha256": DECK_PHYSICS_SHA256,
                "solver_family": family,
                "output_path": reference.name,
                "native_output_path": native.name,
                "native_output_sha256": _sha256(native),
                "provenance_url": f"https://example.invalid/{suffix}/itg_public",
                "redistribution_license": "CC-BY-4.0",
                "sha256": _sha256(reference),
            }
        )
    manifest = {
        "schema": "gk-nonlinear-external-output-manifest.v1",
        "cases": cases,
        "grid_convergence_evidence": [
            {
                "case_id": f"{case['case_id']}_unlinked",
                "solver_family": case["solver_family"],
                "observable": "ion_heat_flux_spectrum",
                "coarse_grid": [2, 2, 2],
                "fine_grid": [4, 4, 4],
                "relative_l2": 0.08,
            }
            for case in cases
        ],
        "production_scaling_evidence": [
            {
                "case_id": f"{case['case_id']}_unlinked",
                "solver_family": case["solver_family"],
                "device": "public-cpu-cluster",
                "grid": [2, 2, 2, 2, 2, 2],
                "ranks": 8,
                "wall_time_s": 12.5,
            }
            for case in cases
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

    assert report["reference_output_ready"] is True
    assert report["same_deck_group_ready"] is True
    assert report["native_same_case_comparison_ready"] is True
    assert report["grid_convergence_ready"] is False
    assert report["production_scale_scaling_ready"] is False
    assert report["status"] == "blocked_missing_grid_convergence_evidence"
    assert report["accepted_full_fidelity_ready"] is False


def test_gk_external_output_parity_blocks_non_redistributable_output(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "external"
    source_root.mkdir()
    output = source_root / "gene_case.json"
    _payload(output)
    manifest = {
        "schema": "gk-nonlinear-external-output-manifest.v1",
        "cases": [
            {
                "case_id": "gene_restricted_itg",
                "deck_id": "gene_restricted_itg_deck",
                "benchmark_case_id": "restricted_itg_em_same_deck",
                "deck_physics_sha256": DECK_PHYSICS_SHA256,
                "solver_family": "GENE",
                "output_path": output.name,
                "provenance_url": "file:///private/gene/restricted-output",
                "redistribution_license": "all-rights-reserved",
                "sha256": _sha256(output),
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
    gene = rows["GENE"]
    assert gene["reference_output_ready"] is False
    assert gene["status"] == "blocked_external_output_provenance_or_license_invalid"
    assert gene["reason"] == "non_redistributable_license"
    assert report["converted_reference_artifacts"] == 0
    assert report["accepted_full_fidelity_ready"] is False
