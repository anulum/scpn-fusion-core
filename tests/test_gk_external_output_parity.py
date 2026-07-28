#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
"""Tests for fail-closed nonlinear GK external output parity artefacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
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


def _case(
    source_root: Path,
    *,
    family: str = "GENE",
    native: bool = False,
    suffix: str = ".json",
) -> dict[str, Any]:
    slug = family.lower()
    reference = source_root / f"{slug}_reference{suffix}"
    if suffix == ".npz":
        _payload_npz(reference)
    else:
        _payload(reference)
    case: dict[str, Any] = {
        "case_id": f"{slug}_itg_public",
        "deck_id": f"{slug}_itg_public_deck",
        "benchmark_case_id": "public_itg_em_same_deck",
        "deck_physics_sha256": DECK_PHYSICS_SHA256,
        "solver_family": family,
        "output_path": reference.name,
        "provenance_url": f"https://example.invalid/{slug}/itg_public",
        "redistribution_license": "MIT",
        "sha256": _sha256(reference),
    }
    if native:
        native_path = source_root / f"{slug}_native{suffix}"
        if suffix == ".npz":
            _payload_npz(native_path)
        else:
            _payload(native_path)
        case.update(
            native_output_path=native_path.name,
            native_output_sha256=_sha256(native_path),
        )
    return case


def _write_manifest(
    source_root: Path,
    cases: list[Any],
    *,
    grid_rows: Any = None,
    scaling_rows: Any = None,
) -> Path:
    manifest_path = source_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": parity.MANIFEST_SCHEMA,
                "cases": cases,
                "grid_convergence_evidence": [] if grid_rows is None else grid_rows,
                "production_scaling_evidence": [] if scaling_rows is None else scaling_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _build(tmp_path: Path, source_root: Path, *, write: bool = True) -> dict[str, Any]:
    return parity.build_gk_external_output_parity_report(
        source_root=source_root,
        artifact_dir=tmp_path / "artifacts",
        report_dir=tmp_path / "reports",
        write=write,
    )


def _row(report: dict[str, Any], family: str = "GENE") -> dict[str, Any]:
    return next(row for row in report["external_output_rows"] if row["solver_family"] == family)


def test_gk_external_output_parity_blocks_without_manifest(tmp_path: Path) -> None:
    """Emit explicit blocked rows when no external manifest is present."""
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
    assert report["evidence_package_ready"] is False
    assert report["evidence_package_contract"]["contract_id"] == (
        "gk_external_nonlinear_full_fidelity_evidence_package_v1"
    )

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
    evidence = {row["solver_family"]: row for row in report["evidence_package_matrix"]}
    assert set(evidence) == {"GENE", "CGYRO", "GS2"}
    assert not any(row["ready"] for row in evidence.values())
    surfaces = report["roadmap_evidence_surface_matrix"]
    assert report["roadmap_evidence_surfaces_ready"] is False
    assert len(surfaces) == 21
    for row in surfaces:
        assert row["ready"] is False
        assert row["blockers"]
    distribution_rows = [
        row for row in surfaces if row["surface"] == "nonlinear_distribution_output"
    ]
    assert {row["solver_family"] for row in distribution_rows} == {"GENE", "CGYRO", "GS2"}
    assert all(
        set(row["required_observables"])
        == {"nonlinear_distribution_function", "nonlinear_distribution_function_imag"}
        for row in distribution_rows
    )


def test_gk_external_output_parity_converts_valid_public_output(tmp_path: Path) -> None:
    """Convert one licensed public payload without promoting incomplete parity."""
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
    assert report["evidence_package_ready"] is False

    completeness = {
        row["solver_family"]: row for row in report["solver_family_completeness_matrix"]
    }
    assert completeness["GENE"]["same_deck_reference_output_ready"] is True
    assert completeness["GENE"]["native_same_case_comparison_ready"] is False
    assert completeness["GENE"]["complete_required_observables"] is True
    assert all(completeness["GENE"]["observable_presence"].values())
    assert completeness["CGYRO"]["same_deck_reference_output_ready"] is False
    assert completeness["GS2"]["same_deck_reference_output_ready"] is False
    surfaces = {
        (row["solver_family"], row["surface"]): row
        for row in report["roadmap_evidence_surface_matrix"]
    }
    assert surfaces[("GENE", "nonlinear_distribution_output")]["ready"] is True
    assert surfaces[("GENE", "heat_flux_spectra_time_kx_ky_species")]["ready"] is True
    assert surfaces[("GENE", "field_energy_history_phi_apar_bpar")]["ready"] is True
    assert surfaces[("GENE", "zonal_flow_and_saturation_metrics")]["ready"] is True
    assert surfaces[("GENE", "native_same_case_solver_output_comparison")]["ready"] is False
    assert surfaces[("GENE", "grid_convergence_evidence")]["ready"] is True
    assert surfaces[("GENE", "production_scale_scaling_evidence")]["ready"] is True
    assert surfaces[("CGYRO", "nonlinear_distribution_output")]["ready"] is False
    assert report["roadmap_evidence_surfaces_ready"] is False

    with np.load(tmp_path / gene["converted_artifact_path"], allow_pickle=False) as payload_npz:
        assert "nonlinear_distribution_function" in payload_npz.files
        assert "nonlinear_distribution_function_imag" in payload_npz.files
        assert "ion_heat_flux_spectrum" in payload_npz.files
        assert "time_s" in payload_npz.files


def test_gk_external_output_parity_compares_native_same_case_output(tmp_path: Path) -> None:
    """Evaluate native same-case thresholds against a checksummed reference."""
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
    assert report["evidence_package_ready"] is False


def test_gk_external_output_parity_accepts_npz_payload_with_separated_metadata(
    tmp_path: Path,
) -> None:
    """Separate coordinate and observable arrays in a bounded NPZ payload."""
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
    """Reject a native comparison that lacks an exact checksum."""
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


def test_gk_external_output_parity_accepts_complete_same_deck_evidence_package(
    tmp_path: Path,
) -> None:
    """Accept only a complete three-solver same-deck evidence package."""
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

    assert report["status"] == "accepted_full_fidelity_ready"
    assert report["accepted_full_fidelity_ready"] is True
    assert report["reference_output_ready"] is True
    assert report["same_deck_group_ready"] is True
    assert report["native_same_case_comparison_ready"] is True
    assert report["grid_convergence_ready"] is True
    assert report["production_scale_scaling_ready"] is True
    assert report["evidence_package_ready"] is True
    assert report["grid_convergence_contract"]["max_relative_l2"] == 0.15
    assert report["production_scale_scaling_contract"]["min_phase_cells"] == 64
    assert report["production_scale_scaling_contract"]["min_ranks"] == 1
    assert len(report["threshold_contract_matrix"]) == 8
    assert len(report["grid_convergence_evidence_matrix"]) == 3
    assert len(report["production_scale_scaling_evidence_matrix"]) == 3
    for row in report["grid_convergence_evidence_matrix"]:
        assert row["ready"] is True
        assert row["reasons"] == []
        assert row["relative_l2"] <= row["threshold"]
    for row in report["production_scale_scaling_evidence_matrix"]:
        assert row["ready"] is True
        assert row["reasons"] == []
        assert row["phase_cells"] >= 64
        assert row["ranks"] >= 1
    for row in report["evidence_package_matrix"]:
        assert row["ready"] is True
        assert row["converted_artifact_ready"] is True
        assert row["converted_metadata_ready"] is True
        assert len(row["converted_artifact_sha256"]) == 64
        assert len(row["converted_metadata_sha256"]) == 64


def test_gk_external_output_parity_blocks_cross_solver_deck_mismatch(
    tmp_path: Path,
) -> None:
    """Block otherwise complete rows that do not share one physics deck."""
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
    assert report["evidence_package_ready"] is False


def test_gk_external_output_parity_blocks_unlinked_convergence_and_scaling_rows(
    tmp_path: Path,
) -> None:
    """Reject convergence and scaling rows linked to different case identifiers."""
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


def test_gk_external_output_parity_blocks_threshold_failed_grid_and_scaling_evidence(
    tmp_path: Path,
) -> None:
    """Reject convergence and scaling evidence outside published limits."""
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
                "case_id": case["case_id"],
                "solver_family": case["solver_family"],
                "observable": "ion_heat_flux_spectrum",
                "coarse_grid": [2, 2, 2],
                "fine_grid": [4, 4, 4],
                "relative_l2": 0.50,
            }
            for case in cases
        ],
        "production_scaling_evidence": [
            {
                "case_id": case["case_id"],
                "solver_family": case["solver_family"],
                "device": "public-cpu-cluster",
                "grid": [1, 1, 1],
                "ranks": 0,
                "wall_time_s": 100_000.0,
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
    assert all(
        "relative_l2_exceeds_threshold" in row["reasons"]
        for row in report["grid_convergence_evidence_matrix"]
    )
    assert all(
        "wall_time_exceeds_threshold" in row["reasons"]
        for row in report["production_scale_scaling_evidence_matrix"]
    )
    assert all(
        "phase_cells_below_threshold" in row["reasons"]
        for row in report["production_scale_scaling_evidence_matrix"]
    )
    assert all(
        "ranks_below_threshold" in row["reasons"]
        for row in report["production_scale_scaling_evidence_matrix"]
    )


def test_gk_external_output_parity_blocks_non_redistributable_output(
    tmp_path: Path,
) -> None:
    """Block external outputs without a public redistribution grant."""
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


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "manifest must be a JSON object"),
        ({"schema": "wrong", "cases": []}, "schema mismatch"),
        ({"schema": parity.MANIFEST_SCHEMA, "cases": {}}, "cases must be a list"),
    ],
)
def test_gk_external_output_parity_rejects_malformed_manifest_contracts(
    tmp_path: Path, payload: Any, message: str
) -> None:
    """Reject non-object, wrong-schema, and non-list manifest containers."""
    source_root = tmp_path / "external"
    source_root.mkdir()
    (source_root / "manifest.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _build(tmp_path, source_root)


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        ("root", "must be a JSON object"),
        ("schema", "schema mismatch"),
        ("cases", "reference case is missing"),
    ],
)
def test_gk_external_output_parity_rejects_malformed_reference_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    message: str,
) -> None:
    """Fail closed when the tracked reference contract is structurally invalid."""
    reference_path = tmp_path / "reference_cases.json"
    payload: Any = json.loads(parity.REFERENCE_CASES.read_text(encoding="utf-8"))
    if scenario == "root":
        payload = []
    elif scenario == "schema":
        payload["schema"] = "wrong"
    else:
        payload["surfaces"]["native_nonlinear_gyrokinetics"]["required_cases"] = []
    reference_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(parity, "REFERENCE_CASES", reference_path)
    with pytest.raises(ValueError, match=message):
        _build(tmp_path, tmp_path / "missing")


@pytest.mark.parametrize(
    ("scenario", "expected_status", "expected_reason"),
    [
        ("missing_field", "blocked_external_output_manifest_incomplete", None),
        ("invalid_deck_hash", "blocked_external_output_manifest_incomplete", None),
        (
            "private_url",
            "blocked_external_output_provenance_or_license_invalid",
            "non_public_provenance_url",
        ),
        ("escape", "blocked_external_output_path_invalid", "path escapes"),
        ("missing_file", "blocked_external_output_file_missing", None),
        ("checksum", "blocked_external_output_checksum_mismatch", None),
        ("bad_json", "blocked_external_output_payload_invalid", None),
        ("list_root", "blocked_external_output_payload_invalid", "must be an object"),
        ("wrong_schema", "blocked_external_output_payload_invalid", "schema mismatch"),
        ("map_types", "blocked_external_output_payload_invalid", "must define"),
        ("unsupported", "blocked_external_output_payload_invalid", "unsupported"),
    ],
)
def test_gk_external_output_parity_blocks_invalid_external_payloads(
    tmp_path: Path,
    scenario: str,
    expected_status: str,
    expected_reason: str | None,
) -> None:
    """Block invalid manifest fields, paths, checksums, and payload containers."""
    source_root = tmp_path / "external"
    source_root.mkdir()
    case = _case(source_root)
    output = source_root / str(case["output_path"])
    if scenario == "missing_field":
        del case["deck_id"]
    elif scenario == "invalid_deck_hash":
        case["deck_physics_sha256"] = "bad"
    elif scenario == "private_url":
        case["provenance_url"] = "file:///private/output"
    elif scenario == "escape":
        case["output_path"] = "../outside.json"
    elif scenario == "missing_file":
        case["output_path"] = "absent.json"
    elif scenario == "checksum":
        case["sha256"] = "f" * 64
    elif scenario == "bad_json":
        output.write_text("{", encoding="utf-8")
        case["sha256"] = _sha256(output)
    elif scenario == "list_root":
        output.write_text("[]\n", encoding="utf-8")
        case["sha256"] = _sha256(output)
    elif scenario == "wrong_schema":
        output.write_text(json.dumps({"schema": "wrong"}) + "\n", encoding="utf-8")
        case["sha256"] = _sha256(output)
    elif scenario == "map_types":
        output.write_text(
            json.dumps({"schema": parity.OUTPUT_SCHEMA, "coordinates": [], "observables": []})
            + "\n",
            encoding="utf-8",
        )
        case["sha256"] = _sha256(output)
    elif scenario == "unsupported":
        renamed = output.with_suffix(".txt")
        output.rename(renamed)
        case["output_path"] = renamed.name
        case["sha256"] = _sha256(renamed)
    _write_manifest(source_root, [case])

    row = _row(_build(tmp_path, source_root))
    assert row["status"] == expected_status
    if expected_reason is not None:
        assert expected_reason in row.get("reason", "")


@pytest.mark.parametrize(
    ("scenario", "failure_group", "reason"),
    [
        ("coordinate_missing", "coordinate_failures", "missing"),
        ("coordinate_rank", "coordinate_failures", "not_one_dimensional"),
        ("coordinate_short", "coordinate_failures", "below_min_length"),
        ("coordinate_nonfinite", "coordinate_failures", "non_finite"),
        ("coordinate_order", "coordinate_failures", "not_strictly_increasing"),
        ("observable_missing", "observable_failures", "missing"),
        ("observable_empty", "observable_failures", "empty"),
        ("observable_nonfinite", "observable_failures", "non_finite"),
        ("observable_rank", "observable_failures", "rank_below_minimum"),
        ("observable_axis", "observable_failures", "axis_length_mismatch"),
    ],
)
def test_gk_external_output_parity_reports_coordinate_and_observable_failures(
    tmp_path: Path, scenario: str, failure_group: str, reason: str
) -> None:
    """Expose exact coordinate and observable contract failures in blocked rows."""
    source_root = tmp_path / "external"
    source_root.mkdir()
    case = _case(source_root)
    output = source_root / str(case["output_path"])
    payload = json.loads(output.read_text(encoding="utf-8"))
    if scenario == "coordinate_missing":
        del payload["coordinates"]["time_s"]
    elif scenario == "coordinate_rank":
        payload["coordinates"]["time_s"] = [[0.0, 1.0]]
    elif scenario == "coordinate_short":
        payload["coordinates"]["time_s"] = [0.0]
    elif scenario == "coordinate_nonfinite":
        payload["coordinates"]["time_s"] = [0.0, float("nan")]
    elif scenario == "coordinate_order":
        payload["coordinates"]["time_s"] = [1.0, 0.0]
    elif scenario == "observable_missing":
        del payload["observables"]["ion_heat_flux_spectrum"]
    elif scenario == "observable_empty":
        payload["observables"]["ion_heat_flux_spectrum"] = []
    elif scenario == "observable_nonfinite":
        payload["observables"]["ion_heat_flux_spectrum"] = [float("nan")]
    elif scenario == "observable_rank":
        payload["observables"]["nonlinear_distribution_function"] = [1.0, 2.0]
    else:
        payload["observables"]["ion_heat_flux_spectrum"] = [[[1.0]]]
    output.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    case["sha256"] = _sha256(output)
    _write_manifest(source_root, [case])

    row = _row(_build(tmp_path, source_root))
    assert row["status"] == "blocked_external_output_contract_invalid"
    assert any(failure["reason"] == reason for failure in row[failure_group])


def test_gk_external_output_parity_blocks_oversized_npz_member(tmp_path: Path) -> None:
    """Reject a compressed NPZ member above the central expanded-size cap."""
    source_root = tmp_path / "external"
    source_root.mkdir()
    output = source_root / "gene_reference.npz"
    np.savez_compressed(output, oversized=np.zeros(1_400_000, dtype=np.float64))
    case = {
        "case_id": "gene_itg_public",
        "deck_id": "gene_itg_public_deck",
        "benchmark_case_id": "public_itg_em_same_deck",
        "deck_physics_sha256": DECK_PHYSICS_SHA256,
        "solver_family": "GENE",
        "output_path": output.name,
        "provenance_url": "https://example.invalid/gene/itg_public",
        "redistribution_license": "MIT",
        "sha256": _sha256(output),
    }
    _write_manifest(source_root, [case])

    row = _row(_build(tmp_path, source_root))
    assert row["status"] == "blocked_external_output_payload_invalid"
    assert "member too large" in row["reason"]


def test_gk_external_output_parity_reports_malformed_evidence_rows(tmp_path: Path) -> None:
    """Report every malformed convergence and scaling field without promotion."""
    source_root = tmp_path / "external"
    source_root.mkdir()
    case = _case(source_root)
    case_id = str(case["case_id"])
    valid_grid = {
        "case_id": case_id,
        "solver_family": "GENE",
        "observable": "ion_heat_flux_spectrum",
        "coarse_grid": [2, 2],
        "fine_grid": [4, 4],
        "relative_l2": 0.1,
    }
    grid_rows: list[Any] = [
        "not-an-object",
        {**valid_grid, "solver_family": "UNKNOWN"},
        {**valid_grid, "case_id": "wrong"},
        {**valid_grid, "observable": "not_required"},
        {**valid_grid, "coarse_grid": [{"bad": 1}]},
        {**valid_grid, "coarse_grid": [[2, 2]]},
        {**valid_grid, "coarse_grid": [2, float("nan")]},
        {**valid_grid, "coarse_grid": [2, 0]},
        {**valid_grid, "coarse_grid": [2, 1.5]},
        {**valid_grid, "coarse_grid": [2, 2], "fine_grid": [4, 4, 4]},
        {**valid_grid, "coarse_grid": [4, 4], "fine_grid": [2, 8]},
        {**valid_grid, "coarse_grid": [2, 2], "fine_grid": [2, 2]},
        {**valid_grid, "relative_l2": "bad"},
        {**valid_grid, "relative_l2": float("nan")},
        {**valid_grid, "relative_l2": 0.5},
    ]
    valid_scaling = {
        "case_id": case_id,
        "solver_family": "GENE",
        "device": "cpu-cluster",
        "grid": [2, 2, 2, 2, 2, 2],
        "ranks": 2,
        "wall_time_s": 2.0,
    }
    scaling_rows: list[Any] = [
        "not-an-object",
        {**valid_scaling, "solver_family": "UNKNOWN"},
        {**valid_scaling, "case_id": "wrong"},
        {**valid_scaling, "device": ""},
        {**valid_scaling, "grid": [1, 1, 1]},
        {**valid_scaling, "ranks": "bad"},
        {**valid_scaling, "ranks": 1.5},
        {**valid_scaling, "ranks": 0},
        {**valid_scaling, "wall_time_s": "bad"},
        {**valid_scaling, "wall_time_s": 0},
        {**valid_scaling, "wall_time_s": 100_000},
    ]
    _write_manifest(source_root, [case], grid_rows=grid_rows, scaling_rows=scaling_rows)

    report = _build(tmp_path, source_root)
    grid_reasons = {
        reason for row in report["grid_convergence_evidence_matrix"] for reason in row["reasons"]
    }
    scaling_reasons = {
        reason
        for row in report["production_scale_scaling_evidence_matrix"]
        for reason in row["reasons"]
    }
    assert {
        "row_not_object",
        "unknown_solver_family",
        "missing_converted_reference_case",
        "case_id_mismatch",
        "observable_not_required",
        "coarse_grid_non_numeric",
        "coarse_grid_not_one_dimensional",
        "coarse_grid_non_finite",
        "coarse_grid_not_positive",
        "coarse_grid_not_integer",
        "grid_rank_mismatch",
        "fine_grid_not_refinement",
        "fine_grid_not_larger",
        "relative_l2_not_numeric",
        "relative_l2_non_finite",
        "relative_l2_exceeds_threshold",
    } <= grid_reasons
    assert {
        "row_not_object",
        "unknown_solver_family",
        "missing_converted_reference_case",
        "case_id_mismatch",
        "device_missing",
        "phase_cells_below_threshold",
        "ranks_not_numeric",
        "ranks_not_integer",
        "ranks_below_threshold",
        "wall_time_not_numeric",
        "wall_time_not_positive",
        "wall_time_exceeds_threshold",
    } <= scaling_reasons


def test_gk_external_output_parity_blocks_non_list_evidence_containers(
    tmp_path: Path,
) -> None:
    """Treat non-list convergence and scaling containers as absent evidence."""
    source_root = tmp_path / "external"
    source_root.mkdir()
    _write_manifest(source_root, [_case(source_root)], grid_rows={}, scaling_rows={})
    report = _build(tmp_path, source_root)
    assert report["grid_convergence_evidence_matrix"] == []
    assert report["production_scale_scaling_evidence_matrix"] == []
    assert report["grid_convergence_rows"] == []
    assert report["production_scaling_rows"] == []


@pytest.mark.parametrize(
    ("scenario", "expected_status"),
    [
        ("invalid_hash", "blocked_native_same_case_output_checksum_invalid"),
        ("checksum", "blocked_native_same_case_output_checksum_mismatch"),
        ("missing_file", "blocked_native_same_case_output_missing_or_invalid"),
        ("bad_json", "blocked_native_same_case_output_missing_or_invalid"),
        ("wrong_schema", "blocked_native_same_case_output_missing_or_invalid"),
        ("contract", "blocked_native_same_case_output_contract_invalid"),
        ("threshold", "blocked_native_same_case_comparison_failed"),
    ],
)
def test_gk_external_output_parity_blocks_invalid_native_comparisons(
    tmp_path: Path, scenario: str, expected_status: str
) -> None:
    """Block invalid native checksums, payloads, contracts, and threshold failures."""
    source_root = tmp_path / "external"
    source_root.mkdir()
    case = _case(source_root, native=True)
    native_path = source_root / str(case["native_output_path"])
    if scenario == "invalid_hash":
        case["native_output_sha256"] = "bad"
    elif scenario == "checksum":
        case["native_output_sha256"] = "f" * 64
    elif scenario == "missing_file":
        native_path.unlink()
    elif scenario == "bad_json":
        native_path.write_text("{", encoding="utf-8")
        case["native_output_sha256"] = _sha256(native_path)
    elif scenario == "wrong_schema":
        native_path.write_text(json.dumps({"schema": "wrong"}) + "\n", encoding="utf-8")
        case["native_output_sha256"] = _sha256(native_path)
    elif scenario == "contract":
        payload = json.loads(native_path.read_text(encoding="utf-8"))
        payload["coordinates"]["time_s"] = [1.0, 0.0]
        native_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        case["native_output_sha256"] = _sha256(native_path)
    else:
        _payload(native_path, scale=4.0)
        case["native_output_sha256"] = _sha256(native_path)
    _write_manifest(source_root, [case])

    row = _row(_build(tmp_path, source_root))
    assert row["status"] == expected_status
    assert row["native_same_case_comparison_passed"] is False


@pytest.mark.parametrize(
    ("scenario", "reason"),
    [
        ("non_numeric", "non_numeric_contract"),
        ("missing_axis", "axis_contract_missing"),
        ("axis_rank", "rank_below_axis_count"),
    ],
)
def test_gk_external_output_parity_applies_reference_observable_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    reason: str,
) -> None:
    """Apply numeric, axis-presence, and axis-rank rules through report generation."""
    source_root = tmp_path / "external"
    source_root.mkdir()
    case = _case(source_root)
    reference = json.loads(parity.REFERENCE_CASES.read_text(encoding="utf-8"))
    contract = reference["surfaces"]["native_nonlinear_gyrokinetics"]["required_cases"][0]
    if scenario == "non_numeric":
        contract["observable_contracts"]["ion_heat_flux_spectrum"]["numeric"] = False
    elif scenario == "missing_axis":
        contract["observable_contracts"]["ion_heat_flux_spectrum"]["axes"].append("absent")
    else:
        contract["observable_contracts"]["saturated_phi_rms"]["axes"].append("kx_rhos")
    reference_path = tmp_path / "reference_cases.json"
    reference_path.write_text(json.dumps(reference) + "\n", encoding="utf-8")
    monkeypatch.setattr(parity, "REFERENCE_CASES", reference_path)
    _write_manifest(source_root, [case])

    row = _row(_build(tmp_path, source_root))
    assert row["status"] == "blocked_external_output_contract_invalid"
    assert any(failure["reason"] == reason for failure in row["observable_failures"])


def test_gk_external_output_parity_reports_scaling_block_after_valid_grid(
    tmp_path: Path,
) -> None:
    """Report a scaling-only blocker after all earlier gates are complete."""
    source_root = tmp_path / "external"
    source_root.mkdir()
    cases = [
        _case(source_root, family=family, native=True) for family in parity.REQUIRED_SOLVER_FAMILIES
    ]
    grid_rows = [
        {
            "case_id": case["case_id"],
            "solver_family": case["solver_family"],
            "observable": "ion_heat_flux_spectrum",
            "coarse_grid": [2, 2, 2],
            "fine_grid": [4, 4, 4],
            "relative_l2": 0.1,
        }
        for case in cases
    ]
    _write_manifest(source_root, cases, grid_rows=grid_rows, scaling_rows=[])
    report = _build(tmp_path, source_root)
    assert report["grid_convergence_ready"] is True
    assert report["production_scale_scaling_ready"] is False
    assert report["status"] == "blocked_missing_production_scale_scaling_evidence"


def test_gk_external_output_parity_no_write_and_cli_paths(tmp_path: Path) -> None:
    """Exercise the real CLI against temporary roots in write and check modes."""
    missing_root = tmp_path / "missing"
    artifact_dir = tmp_path / "artifacts"
    report_dir = tmp_path / "reports"
    base_command = [
        sys.executable,
        "tools/gk_external_output_parity.py",
        "--source-root",
        str(missing_root),
        "--artifact-dir",
        str(artifact_dir),
        "--report-dir",
        str(report_dir),
        "--no-write",
    ]
    clean = subprocess.run(base_command, cwd=parity.ROOT, check=False)
    blocked = subprocess.run([*base_command, "--check"], cwd=parity.ROOT, check=False)
    assert clean.returncode == 0
    assert blocked.returncode == 1
    assert not artifact_dir.exists()
    assert not report_dir.exists()

    report = parity.build_gk_external_output_parity_report(
        source_root=missing_root,
        artifact_dir=parity.ROOT / "artifacts",
        report_dir=report_dir,
        write=False,
    )
    assert report["status"] == "blocked_missing_external_output_manifest"

    parity.main(
        [
            "--source-root",
            str(missing_root),
            "--artifact-dir",
            str(artifact_dir),
            "--report-dir",
            str(report_dir),
            "--no-write",
        ]
    )
    with pytest.raises(SystemExit) as exc_info:
        parity.main(
            [
                "--source-root",
                str(missing_root),
                "--artifact-dir",
                str(artifact_dir),
                "--report-dir",
                str(report_dir),
                "--no-write",
                "--check",
            ]
        )
    assert exc_info.value.code == 1


def test_gk_external_output_parity_no_write_blocks_incomplete_evidence_package(
    tmp_path: Path,
) -> None:
    """Keep a fully valid semantic package blocked when output custody is unwritten."""
    source_root = tmp_path / "external"
    source_root.mkdir()
    cases = [
        _case(source_root, family=family, native=True) for family in parity.REQUIRED_SOLVER_FAMILIES
    ]
    grid_rows = [
        {
            "case_id": case["case_id"],
            "solver_family": case["solver_family"],
            "observable": "ion_heat_flux_spectrum",
            "coarse_grid": [2, 2, 2],
            "fine_grid": [4, 4, 4],
            "relative_l2": 0.1,
        }
        for case in cases
    ]
    scaling_rows = [
        {
            "case_id": case["case_id"],
            "solver_family": case["solver_family"],
            "device": "cpu-cluster",
            "grid": [2, 2, 2, 2, 2, 2],
            "ranks": 2,
            "wall_time_s": 2.0,
        }
        for case in cases
    ]
    _write_manifest(source_root, cases, grid_rows=grid_rows, scaling_rows=scaling_rows)
    report = parity.build_gk_external_output_parity_report(
        source_root=source_root,
        artifact_dir=parity.ROOT / "artifacts",
        report_dir=tmp_path / "reports",
        write=False,
    )
    assert report["reference_output_ready"] is True
    assert report["native_same_case_comparison_ready"] is True
    assert report["status"] == "blocked_incomplete_evidence_package"
    assert report["evidence_package_ready"] is False


def test_gk_external_output_parity_reports_missing_native_after_complete_references(
    tmp_path: Path,
) -> None:
    """Reach the native-comparison blocker only after all reference rows agree."""
    source_root = tmp_path / "external"
    source_root.mkdir()
    cases = [_case(source_root, family=family) for family in parity.REQUIRED_SOLVER_FAMILIES]
    _write_manifest(source_root, cases)
    report = _build(tmp_path, source_root)
    assert report["reference_output_ready"] is True
    assert report["same_deck_group_ready"] is True
    assert report["status"] == "blocked_missing_native_same_case_output_comparison"


def test_gk_external_output_parity_rejects_invalid_shared_deck_identity(
    tmp_path: Path,
) -> None:
    """Reject complete family rows with missing shared benchmark/deck identity."""
    source_root = tmp_path / "external"
    source_root.mkdir()
    cases = [_case(source_root, family=family) for family in parity.REQUIRED_SOLVER_FAMILIES]
    for case in cases:
        case["benchmark_case_id"] = ""
        case["deck_physics_sha256"] = "bad"
    _write_manifest(source_root, ["ignored", *cases, dict(cases[0])])
    report = _build(tmp_path, source_root)
    assert report["same_deck_group"]["reason"] == "missing_or_invalid_same_deck_identity"
    assert report["same_deck_group_ready"] is False


def test_gk_external_output_parity_accepts_unclassified_npz_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Load NPZ arrays when a reference contract declares no named classifications."""
    source_root = tmp_path / "external"
    source_root.mkdir()
    case = _case(source_root, suffix=".npz")
    reference = json.loads(parity.REFERENCE_CASES.read_text(encoding="utf-8"))
    contract = reference["surfaces"]["native_nonlinear_gyrokinetics"]["required_cases"][0]
    contract["coordinate_contracts"] = {}
    contract["observable_contracts"] = {}
    contract["required_observables"] = []
    contract["thresholds"] = {}
    contract["threshold_contracts"] = {}
    reference_path = tmp_path / "reference_cases.json"
    reference_path.write_text(json.dumps(reference) + "\n", encoding="utf-8")
    monkeypatch.setattr(parity, "REFERENCE_CASES", reference_path)
    _write_manifest(source_root, [case])
    row = _row(_build(tmp_path, source_root))
    assert row["reference_output_ready"] is True
    assert "ion_heat_flux_spectrum" in row["available_observables"]


def test_gk_external_output_parity_ignores_non_list_axes_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Treat a non-list axes field as absent while retaining other validation."""
    source_root = tmp_path / "external"
    source_root.mkdir()
    case = _case(source_root)
    reference = json.loads(parity.REFERENCE_CASES.read_text(encoding="utf-8"))
    contract = reference["surfaces"]["native_nonlinear_gyrokinetics"]["required_cases"][0]
    contract["observable_contracts"]["ion_heat_flux_spectrum"]["axes"] = "not-a-list"
    reference_path = tmp_path / "reference_cases.json"
    reference_path.write_text(json.dumps(reference) + "\n", encoding="utf-8")
    monkeypatch.setattr(parity, "REFERENCE_CASES", reference_path)
    _write_manifest(source_root, [case])
    assert _row(_build(tmp_path, source_root))["reference_output_ready"] is True


@pytest.mark.parametrize(
    ("scenario", "expected_reason", "expected_passed"),
    [
        ("contract", "missing_threshold_contract", False),
        ("limit", "invalid_threshold_value", False),
        ("comparator", "unsupported_comparator", False),
        ("metric", "unsupported_metric", False),
        ("observable", "missing_observable_contract", False),
        ("native", "invalid_native_observable", False),
        ("reference", "invalid_reference_observable", False),
        ("shape", "observable_shape_mismatch", False),
        ("absolute", None, True),
        ("minimum", None, True),
    ],
)
def test_gk_external_output_parity_evaluates_threshold_contract_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    expected_reason: str | None,
    expected_passed: bool,
) -> None:
    """Evaluate malformed and valid threshold contracts through report generation."""
    source_root = tmp_path / "external"
    source_root.mkdir()
    case = _case(source_root, native=True)
    reference_output = source_root / str(case["output_path"])
    native_output = source_root / str(case["native_output_path"])
    reference_payload = json.loads(reference_output.read_text(encoding="utf-8"))
    native_payload = json.loads(native_output.read_text(encoding="utf-8"))
    reference = json.loads(parity.REFERENCE_CASES.read_text(encoding="utf-8"))
    contract = reference["surfaces"]["native_nonlinear_gyrokinetics"]["required_cases"][0]
    threshold_name = "probe_threshold"
    threshold_contract: Any = {
        "comparator": "<=",
        "metric": "relative_l2",
        "observable": "ion_heat_flux_spectrum",
    }
    threshold_limit: Any = 0.2
    if scenario == "contract":
        threshold_contract = []
    elif scenario == "limit":
        threshold_limit = "bad"
    elif scenario == "comparator":
        threshold_contract["comparator"] = "!="
    elif scenario == "metric":
        threshold_contract["metric"] = "unsupported"
    elif scenario == "observable":
        threshold_contract["observable"] = None
    elif scenario in {"native", "reference", "shape"}:
        threshold_contract["observable"] = "extra_probe"
        reference_payload["observables"]["extra_probe"] = [1.0]
        native_payload["observables"]["extra_probe"] = [1.0]
        if scenario == "native":
            native_payload["observables"]["extra_probe"] = [float("nan")]
        elif scenario == "reference":
            reference_payload["observables"]["extra_probe"] = [float("nan")]
        else:
            native_payload["observables"]["extra_probe"] = [1.0, 2.0]
    elif scenario == "absolute":
        threshold_contract.update(metric="absolute_error", observable="zonal_flow_energy")
    else:
        threshold_contract.update(comparator=">=", metric="relative_l2")
        threshold_limit = 0.0
    contract["thresholds"] = {threshold_name: threshold_limit}
    contract["threshold_contracts"] = {threshold_name: threshold_contract}
    reference_output.write_text(json.dumps(reference_payload) + "\n", encoding="utf-8")
    native_output.write_text(json.dumps(native_payload) + "\n", encoding="utf-8")
    case["sha256"] = _sha256(reference_output)
    case["native_output_sha256"] = _sha256(native_output)
    reference_path = tmp_path / "reference_cases.json"
    reference_path.write_text(json.dumps(reference) + "\n", encoding="utf-8")
    monkeypatch.setattr(parity, "REFERENCE_CASES", reference_path)
    _write_manifest(source_root, [case])

    report = _build(tmp_path, source_root)
    evaluation = _row(report)["threshold_evaluation"]
    check = evaluation["checks"][0]
    assert check["passed"] is expected_passed
    if expected_reason is None:
        assert "reason" not in check
    else:
        assert check["reason"] == expected_reason
    if scenario == "limit":
        assert report["threshold_contract_matrix"][0]["limit"] is None
        assert "| - |" in (tmp_path / "reports" / parity.MD_REPORT.name).read_text(encoding="utf-8")


def test_gk_external_output_parity_marks_unknown_roadmap_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep an unrecognized roadmap surface blocked in the public report matrix."""
    unknown_surface = {
        "surface": "unknown_surface",
        "description": "Test-only unknown contract surface.",
        "required_observables": [],
    }
    monkeypatch.setattr(parity, "ROADMAP_EVIDENCE_SURFACES", (unknown_surface,))
    report = _build(tmp_path, tmp_path / "missing")
    assert all(row["ready"] is False for row in report["roadmap_evidence_surface_matrix"])
    assert all(
        row["blockers"] == ["unknown_roadmap_evidence_surface"]
        for row in report["roadmap_evidence_surface_matrix"]
    )
