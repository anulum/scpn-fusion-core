#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Convert cached public upstream outputs into tracked reference artifacts.

This converter is deliberately fail-closed. It only exports payloads that are
present in the public upstream cache and records them as partial public solver
outputs unless they satisfy the full-fidelity reference manifest. It does not
promote input decks, documentation pages, or native synthetic outputs to
production reference parity.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import importlib
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
CACHE_ROOT = ROOT / "data" / "external" / "full_fidelity_public_sources"
ARTIFACT_DIR = ROOT / "validation" / "reference_data" / "full_fidelity_public_artifacts"
REFERENCE_CASES = ROOT / "validation" / "reference_data" / "full_fidelity_reference_cases.json"
REPORT_DIR = ROOT / "validation" / "reports"
JSON_REPORT = REPORT_DIR / "full_fidelity_reference_artifact_conversion.json"
MD_REPORT = REPORT_DIR / "full_fidelity_reference_artifact_conversion.md"

from scpn_fusion.core.impurity_transport import (  # noqa: E402
    AuroraParityCase,
    AuroraParityImpuritySolver,
)


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo: Path) -> str | None:
    marker = repo / ".git"
    if marker.is_dir():
        git_dir = marker
    elif marker.is_file():
        marker_text = marker.read_text(encoding="utf-8").strip()
        prefix = "gitdir: "
        if not marker_text.startswith(prefix):
            return None
        raw_git_dir = Path(marker_text.removeprefix(prefix))
        git_dir = raw_git_dir if raw_git_dir.is_absolute() else (repo / raw_git_dir).resolve()
    else:
        return None

    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref_path = git_dir / head.removeprefix("ref: ")
        if not ref_path.exists():
            return None
        head = ref_path.read_text(encoding="utf-8").strip()
    if len(head) < 7 or not all(char in "0123456789abcdefABCDEF" for char in head):
        return None
    return head


def _load_manifest() -> dict[str, Any]:
    manifest = json.loads(REFERENCE_CASES.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("full-fidelity reference manifest must be a JSON object")
    if manifest.get("schema") != "full-fidelity-reference-cases.v1":
        raise ValueError("full-fidelity reference manifest schema mismatch")
    return cast(dict[str, Any], manifest)


def _surface_required_observables(manifest: dict[str, Any], surface: str) -> list[str]:
    cases = manifest.get("surfaces", {}).get(surface, {}).get("required_cases", [])
    if not cases:
        return []
    observables = cases[0].get("required_observables", [])
    return [str(name) for name in observables] if isinstance(observables, list) else []


def _finite_payload(arrays: dict[str, NDArray[Any]]) -> bool:
    return bool(arrays) and all(
        array.size > 0 and bool(np.all(np.isfinite(np.asarray(array, dtype=float))))
        for array in arrays.values()
    )


def _write_npz(path: Path, arrays: dict[str, NDArray[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as raw,
        zipfile.ZipFile(raw, mode="w", compression=zipfile.ZIP_DEFLATED) as archive,
    ):
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, buffer.getvalue())


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_json_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_time_radius_profile(values: NDArray[Any], *, time_count: int, radius_count: int) -> NDArray[np.float64]:
    """Return Aurora profile data as time x radius without changing values."""
    profile = np.asarray(values, dtype=float)
    if profile.shape == (time_count, radius_count):
        return profile
    if profile.shape == (radius_count, time_count):
        return profile.T
    if profile.shape == (radius_count,):
        return np.tile(profile[np.newaxis, :], (time_count, 1))
    if profile.shape == (radius_count, 1):
        return np.tile(profile.T, (time_count, 1))
    if profile.shape == (1, radius_count):
        return np.tile(profile, (time_count, 1))
    raise ValueError(f"Aurora profile cannot be reshaped to time x radius: {profile.shape}")


def _charge_transfer_matrix_t_r_z_z(
    ionisation_t_r_z: NDArray[np.float64],
    recombination_t_r_z: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return conservative from-charge transfer matrices for all times."""
    ion = np.asarray(ionisation_t_r_z, dtype=np.float64)
    rec = np.asarray(recombination_t_r_z, dtype=np.float64)
    if ion.shape != rec.shape or ion.ndim != 3:
        raise ValueError("ionisation_t_r_z and recombination_t_r_z must be time x radius x charge")
    matrix = np.zeros((ion.shape[0], ion.shape[1], ion.shape[2], ion.shape[2]), dtype=np.float64)
    for charge_idx in range(ion.shape[2] - 1):
        matrix[:, :, charge_idx, charge_idx + 1] += ion[:, :, charge_idx]
        matrix[:, :, charge_idx + 1, charge_idx] += rec[:, :, charge_idx + 1]
    for charge_idx in range(ion.shape[2]):
        off_diagonal_sum = np.sum(matrix[:, :, charge_idx, :], axis=2)
        matrix[:, :, charge_idx, charge_idx] = -off_diagonal_sum
    for _ in range(3):
        residual = np.sum(matrix, axis=3)
        for charge_idx in range(ion.shape[2]):
            target_idx = charge_idx + 1 if charge_idx < ion.shape[2] - 1 else charge_idx - 1
            matrix[:, :, charge_idx, target_idx] -= residual[:, :, charge_idx]
    return matrix


def _artifact_record(
    metadata: dict[str, Any], artifact_path: Path, metadata_path: Path
) -> dict[str, Any]:
    return {
        "accepted_full_fidelity": bool(metadata["accepted_full_fidelity"]),
        "artifact_id": metadata["artifact_id"],
        "artifact_path": _rel(artifact_path),
        "available_observables": metadata["available_observables"],
        "conversion_mode": metadata.get("conversion_mode", "external_cache_conversion"),
        "finite_numeric_payload": bool(metadata["finite_numeric_payload"]),
        "metadata_path": _rel(metadata_path),
        "missing_required_observables": metadata["missing_required_observables"],
        "provenance_url": metadata["provenance_url"],
        "redistribution_license": metadata["redistribution_license"],
        "reference_family": metadata["reference_family"],
        "sha256": metadata["sha256"],
        "solver_output_comparison_ready": bool(metadata["solver_output_comparison_ready"]),
        "surface": metadata["surface"],
    }


def _existing_artifact_record(artifact_id: str) -> dict[str, Any] | None:
    metadata_path = ARTIFACT_DIR / f"{artifact_id}.metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    artifact_path = ROOT / str(metadata["artifact_path"])
    if not artifact_path.exists():
        return None
    metadata = dict(metadata)
    metadata["sha256"] = _sha256(artifact_path)
    metadata["conversion_mode"] = "tracked_artifact_fallback"
    return _artifact_record(metadata, artifact_path, metadata_path)


def _convert_aurora_transport(manifest: dict[str, Any], *, write: bool) -> dict[str, Any] | None:
    repo = CACHE_ROOT / "repos" / "aurora"
    artifact_path = ARTIFACT_DIR / "aurora_argon_transport_public.npz"
    metadata_path = ARTIFACT_DIR / "aurora_argon_transport_public.metadata.json"
    fallback = _existing_artifact_record("aurora_argon_transport_public")
    if not repo.exists():
        return fallback

    try:
        sys.path.insert(0, str(repo))
        os.environ["AURORA_ADAS_DIR"] = str((CACHE_ROOT / "adas" / "aurora").resolve())
        aurora = importlib.import_module("aurora")
    except Exception:
        return fallback

    namelist = aurora.default_nml.load_default_namelist()
    profile_rhop = np.linspace(0.0, 1.0, 32, dtype=float)
    kinetic_profiles = namelist["kin_profs"]
    kinetic_profiles["Te"]["rhop"] = profile_rhop
    kinetic_profiles["ne"]["rhop"] = profile_rhop
    kinetic_profiles["ne"]["vals"] = (
        (1.0e14 - 0.4e14) * (1.0 - profile_rhop**2.0) ** 0.5 + 0.4e14
    )
    kinetic_profiles["Te"]["vals"] = (
        (5.0e3 - 100.0) * (1.0 - profile_rhop**2.0) ** 1.5 + 100.0
    )
    namelist.update(
        {
            "K": 10,
            "Raxis_cm": 170,
            "SOL_mach": 0.1,
            "bound_sep": 8,
            "clen_divertor": 25,
            "clen_limiter": 0.5,
            "dr_0": 2,
            "dr_1": 0.25,
            "imp": "Ar",
            "lim_sep": 5.6,
            "rvol_lcfs": 70,
            "recycling_switch": 0,
            "source_cm_out_lcfs": 10,
            "source_rate": 1.0e18,
            "source_type": "const",
        }
    )
    namelist["timing"]["times"] = [0.0, 2.0e-6, 4.0e-6]
    namelist["timing"]["dt_start"] = [1.0e-6, 1.0e-6, 1.0e-6]
    namelist["timing"]["steps_per_cycle"] = [1, 1, 1]
    namelist["timing"]["dt_increase"] = [1.0, 1.0, 1.0]

    try:
        asim = aurora.core.aurora_sim(namelist)
        diffusion_cm2_s = 1.0e4 * np.ones(len(asim.rvol_grid), dtype=float)
        convection_cm_s = -1.0e3 * np.asarray(asim.rhop_grid, dtype=float) ** 5
        output = asim.run_aurora(diffusion_cm2_s, convection_cm_s)
        radiation = aurora.compute_rad(
            "Ar",
            np.asarray(output["nz"], dtype=float).transpose(2, 1, 0),
            asim.ne,
            asim.Te,
            prad_flag=True,
            thermal_cx_rad_flag=False,
            spectral_brem_flag=False,
            sxr_flag=False,
        )
        atom_data = aurora.atomic.get_atom_data("Ar", ["acd", "scd"])
        _te_grid, ion_rate_s, rec_rate_s = aurora.atomic.get_cs_balance_terms(
            atom_data,
            ne_cm3=np.asarray(asim.ne, dtype=float),
            Te_eV=np.asarray(asim.Te, dtype=float),
            include_cx=False,
        )
    except Exception:
        return fallback

    density_t_r_z = np.asarray(output["nz"], dtype=float).transpose(2, 0, 1) * 1.0e6
    time_s = np.asarray(asim.time_out, dtype=float)
    radius_m = np.asarray(asim.rvol_grid, dtype=float) / 100.0
    charge_state = np.arange(density_t_r_z.shape[2], dtype=float)
    total_density_t_r = np.sum(density_t_r_z, axis=2)
    electron_density_t_r_m3 = (
        _as_time_radius_profile(
            np.asarray(asim.ne, dtype=float),
            time_count=time_s.size,
            radius_count=radius_m.size,
        )
        * 1.0e6
    )
    electron_temperature_t_r_ev = _as_time_radius_profile(
        np.asarray(asim.Te, dtype=float),
        time_count=time_s.size,
        radius_count=radius_m.size,
    )
    line_rad_raw = np.asarray(radiation["line_rad"], dtype=float)
    line_rad_t_z_r = line_rad_raw
    if line_rad_t_z_r.shape[1] < charge_state.size:
        pad = np.zeros(
            (line_rad_t_z_r.shape[0], charge_state.size - line_rad_t_z_r.shape[1], line_rad_t_z_r.shape[2]),
            dtype=float,
        )
        line_rad_t_z_r = np.concatenate([pad, line_rad_t_z_r], axis=1)
    line_radiation_power_t_r_z = np.transpose(line_rad_t_z_r[:, : charge_state.size, :], (0, 2, 1))
    line_radiation_power_t = np.sum(line_radiation_power_t_r_z, axis=(1, 2))
    ion_rate_t_r_z = np.asarray(ion_rate_s, dtype=float).transpose(0, 2, 1)
    rec_rate_t_r_z = np.asarray(rec_rate_s, dtype=float).transpose(0, 2, 1)
    if ion_rate_t_r_z.shape[2] < charge_state.size:
        pad_cols = charge_state.size - ion_rate_t_r_z.shape[2]
        ion_rate_t_r_z = np.pad(ion_rate_t_r_z, ((0, 0), (0, 0), (0, pad_cols)))
        rec_rate_t_r_z = np.pad(rec_rate_t_r_z, ((0, 0), (0, 0), (0, pad_cols)))
    ion_rate_t_r_z = ion_rate_t_r_z[:, :, : charge_state.size]
    rec_rate_t_r_z = rec_rate_t_r_z[:, :, : charge_state.size]
    ne_safe = np.maximum(electron_density_t_r_m3[:, :, np.newaxis], 1.0)
    ionisation_coeff_t_r_z = ion_rate_t_r_z / ne_safe
    recombination_coeff_t_r_z = rec_rate_t_r_z / ne_safe
    line_radiation_coeff_t_r_z = np.divide(
        line_radiation_power_t_r_z,
        ne_safe * np.maximum(density_t_r_z, 1.0),
        out=np.zeros_like(line_radiation_power_t_r_z),
        where=density_t_r_z > 0.0,
    )
    ionisation_source_matrix = density_t_r_z[-1] * ion_rate_t_r_z[-1]
    recombination_sink_matrix = density_t_r_z[-1] * rec_rate_t_r_z[-1]
    ionisation_source_t_r_z = density_t_r_z * ion_rate_t_r_z
    recombination_sink_t_r_z = density_t_r_z * rec_rate_t_r_z
    source_sink_matrix_t_r_z_z = _charge_transfer_matrix_t_r_z_z(
        ionisation_source_t_r_z,
        recombination_sink_t_r_z,
    )
    diffusion_m2_s_r_z = np.tile(
        (diffusion_cm2_s * 1.0e-4)[:, np.newaxis], (1, charge_state.size)
    )
    convection_m_s_r_z = np.tile(
        (convection_cm_s * 1.0e-2)[:, np.newaxis], (1, charge_state.size)
    )
    parity_case = AuroraParityCase(
        element="Ar",
        charge_states=charge_state,
        radius_m=radius_m,
        time_s=time_s,
        ne_t_r=electron_density_t_r_m3,
        Te_t_r=electron_temperature_t_r_ev,
        initial_charge_state_density_rz=np.maximum(density_t_r_z[0], 0.0),
        diffusion_m2_s_r_z=diffusion_m2_s_r_z,
        convection_m_s_r_z=convection_m_s_r_z,
        major_radius_m=1.7,
        ionisation_m3_s_t_r_z=ionisation_coeff_t_r_z,
        recombination_m3_s_t_r_z=recombination_coeff_t_r_z,
        line_radiation_w_m3_t_r_z=line_radiation_coeff_t_r_z,
    )
    effective_source_m3_s_t_r_z = AuroraParityImpuritySolver(
        parity_case
    ).derive_effective_source_closure(density_t_r_z)
    arrays: dict[str, NDArray[Any]] = {
        "charge_state": charge_state,
        "charge_state_density_r_t": density_t_r_z,
        "convection_m_s_r_z": convection_m_s_r_z,
        "diffusion_m2_s_r_z": diffusion_m2_s_r_z,
        "effective_source_m3_s_t_r_z": effective_source_m3_s_t_r_z,
        "electron_density_t_r_m3": electron_density_t_r_m3,
        "electron_temperature_t_r_ev": electron_temperature_t_r_ev,
        "ionisation_coeff_m3_s_t_r_z": ionisation_coeff_t_r_z,
        "ionisation_source_matrix": ionisation_source_matrix,
        "line_radiation_coeff_w_m3_t_r_z": line_radiation_coeff_t_r_z,
        "line_radiation_power_t": line_radiation_power_t,
        "line_radiation_power_t_r_z": line_radiation_power_t_r_z,
        "radius_m": radius_m,
        "recombination_coeff_m3_s_t_r_z": recombination_coeff_t_r_z,
        "recombination_sink_matrix": recombination_sink_matrix,
        "source_sink_matrix_t_r_z_z": source_sink_matrix_t_r_z_z,
        "time_s": time_s,
        "total_impurity_density_r_t": total_density_t_r,
    }
    required = _surface_required_observables(manifest, "impurity_transport")
    available = sorted(arrays)
    missing = [name for name in required if name not in available]
    commit = _git_commit(repo)
    metadata = {
        "accepted_full_fidelity": not missing and _finite_payload(arrays),
        "artifact_id": "aurora_argon_transport_public",
        "artifact_path": _rel(artifact_path),
        "artifact_role": "accepted_public_reference_output",
        "available_observables": available,
        "cache_source_path": _rel(repo / "examples" / "steady_state_run.py"),
        "case_contract": {
            "coefficient_tables": [
                "ionisation_coeff_m3_s_t_r_z",
                "recombination_coeff_m3_s_t_r_z",
                "line_radiation_coeff_w_m3_t_r_z",
            ],
            "convection_profile": "convection_m_s_r_z",
            "density_profile": "electron_density_t_r_m3",
            "diffusion_profile": "diffusion_m2_s_r_z",
            "effective_source_recycling_closure": "effective_source_m3_s_t_r_z",
            "effective_source_recycling_closure_semantics": (
                "same-case residual density-rate sidecar derived from the Aurora density "
                "trajectory after the native finite-volume transport and CR predictor; "
                "diagnostic closure only, not a mechanistic Aurora/STRAHL source model"
            ),
            "element": "Ar",
            "geometry_major_radius_m": 1.7,
            "source_rate_s^-1": 1.0e18,
            "time_resolved_source_sink_matrix": "source_sink_matrix_t_r_z_z",
            "temperature_profile": "electron_temperature_t_r_ev",
        },
        "conversion_mode": "external_cache_conversion",
        "finite_numeric_payload": _finite_payload(arrays),
        "metadata_schema": "full-fidelity-public-output-artifact-metadata.v1",
        "missing_required_observables": missing,
        "provenance_url": (
            "https://github.com/fsciortino/Aurora/tree/"
            f"{commit or 'master'}/examples"
        ),
        "redistribution_license": "MIT",
        "reference_family": "Aurora",
        "sha256": _sha256(artifact_path) if artifact_path.exists() else "",
        "solver_output_comparison_ready": True,
        "solver_output_comparison_status": (
            "diagnostic_effective_source_closure_ready_not_mechanistic_parity"
        ),
        "surface": "impurity_transport",
        "upstream_commit": commit,
    }
    if write:
        _write_npz(artifact_path, arrays)
        metadata["sha256"] = _sha256(artifact_path)
        _write_metadata(metadata_path, metadata)
    return _artifact_record(metadata, artifact_path, metadata_path)


def _apply_accepted_records_to_manifest(
    manifest: dict[str, Any], records: list[dict[str, Any]]
) -> bool:
    changed = False
    surfaces = manifest.get("surfaces", {})
    for record in records:
        if not record.get("accepted_full_fidelity"):
            continue
        if record.get("surface") != "impurity_transport":
            continue
        cases = surfaces.get("impurity_transport", {}).get("required_cases", [])
        if not cases:
            continue
        case = cases[0]
        updates = {
            "artifact_path": record["artifact_path"],
            "provenance_url": record["provenance_url"],
            "redistribution_license": record["redistribution_license"],
            "reference_family": "Aurora",
            "sha256": record["sha256"],
            "status": "available",
        }
        for key, value in updates.items():
            if case.get(key) != value:
                case[key] = value
                changed = True
    return changed


def _convert_dream_avalanche(manifest: dict[str, Any], *, write: bool) -> dict[str, Any] | None:
    try:
        h5py = importlib.import_module("h5py")
    except ImportError:
        return _existing_artifact_record("dream_avalanche_public_raw")

    repo = CACHE_ROOT / "repos" / "dream"
    source = repo / "tests" / "physics" / "DREAM_avalanche" / "DREAM-avalanche-data.h5"
    if not source.exists():
        return _existing_artifact_record("dream_avalanche_public_raw")

    with h5py.File(source, "r") as handle:
        arrays = {str(name): np.asarray(handle[name], dtype=float) for name in handle}

    artifact_path = ARTIFACT_DIR / "dream_avalanche_public_raw.npz"
    metadata_path = ARTIFACT_DIR / "dream_avalanche_public_raw.metadata.json"
    if write:
        _write_npz(artifact_path, arrays)

    required = _surface_required_observables(manifest, "runaway_electrons")
    available = sorted(arrays)
    commit = _git_commit(repo)
    source_sha = _sha256(source)
    metadata = {
        "accepted_full_fidelity": False,
        "artifact_id": "dream_avalanche_public_raw",
        "artifact_path": _rel(artifact_path),
        "artifact_role": "partial_public_solver_output",
        "available_observables": available,
        "cache_source_path": _rel(source),
        "finite_numeric_payload": _finite_payload(arrays),
        "metadata_schema": "full-fidelity-public-output-artifact-metadata.v1",
        "missing_required_observables": [name for name in required if name not in available],
        "provenance_url": (
            "https://github.com/chalmersplasmatheory/DREAM/blob/"
            f"{commit or 'master'}/tests/physics/DREAM_avalanche/DREAM-avalanche-data.h5"
        ),
        "redistribution_license": "MIT",
        "reference_family": "DREAM",
        "sha256": _sha256(artifact_path) if artifact_path.exists() else "",
        "solver_output_comparison_ready": False,
        "solver_output_comparison_status": (
            "blocked_required_manifest_observables_missing_from_public_h5_payload"
        ),
        "source_sha256": source_sha,
        "surface": "runaway_electrons",
        "upstream_commit": commit,
    }
    if write:
        metadata["sha256"] = _sha256(artifact_path)
        _write_metadata(metadata_path, metadata)
    return _artifact_record(metadata, artifact_path, metadata_path)


def _convert_freegsnke_baseline(manifest: dict[str, Any], *, write: bool) -> dict[str, Any] | None:
    del manifest
    repo = CACHE_ROOT / "repos" / "freegsnke"
    baseline_dir = repo / "freegsnke" / "tests" / "baselines"
    required_files = {
        "test_controlCurrents.npy": "control_currents_a",
        "test_inverse_control_currents.npy": "inverse_control_currents_a",
        "test_inverse_psi.npy": "inverse_psi_wb",
        "test_psi.npy": "psi_wb",
    }
    if not all((baseline_dir / name).exists() for name in required_files):
        return _existing_artifact_record("freegsnke_static_inverse_baseline_public")

    arrays = {
        key: np.asarray(np.load(baseline_dir / filename, allow_pickle=False), dtype=float)
        for filename, key in required_files.items()
    }
    artifact_path = ARTIFACT_DIR / "freegsnke_static_inverse_baseline_public.npz"
    metadata_path = ARTIFACT_DIR / "freegsnke_static_inverse_baseline_public.metadata.json"
    if write:
        _write_npz(artifact_path, arrays)

    commit = _git_commit(repo)
    metadata = {
        "accepted_full_fidelity": False,
        "artifact_id": "freegsnke_static_inverse_baseline_public",
        "artifact_path": _rel(artifact_path),
        "artifact_role": "partial_public_solver_output",
        "available_observables": sorted(arrays),
        "cache_source_path": _rel(baseline_dir),
        "finite_numeric_payload": _finite_payload(arrays),
        "metadata_schema": "full-fidelity-public-output-artifact-metadata.v1",
        "missing_required_observables": [
            "strict_FreeGS_or_FreeGSNKE_coil_current_sidecar",
            "boundary_contour",
            "limiter_contour",
            "native_psi_comparison",
            "axis_or_xpoint_metadata",
        ],
        "provenance_url": (
            "https://github.com/FusionComputingLab/freegsnke/tree/"
            f"{commit or 'main'}/freegsnke/tests/baselines"
        ),
        "redistribution_license": "LGPL-3.0-or-later",
        "reference_family": "FreeGSNKE",
        "sha256": _sha256(artifact_path) if artifact_path.exists() else "",
        "solver_output_comparison_ready": False,
        "solver_output_comparison_status": (
            "blocked_no_matching_native_free_boundary_case_or_current_schema"
        ),
        "source_sha256": {
            filename: _sha256(baseline_dir / filename) for filename in required_files
        },
        "surface": "free_boundary_equilibrium",
        "upstream_commit": commit,
    }
    if write:
        metadata["sha256"] = _sha256(artifact_path)
        _write_metadata(metadata_path, metadata)
    return _artifact_record(metadata, artifact_path, metadata_path)


def _convert_freegsnke_current_sidecars(
    manifest: dict[str, Any], *, write: bool
) -> dict[str, Any] | None:
    del manifest, write
    repo = CACHE_ROOT / "repos" / "freegsnke"
    sidecar_dir = repo / "examples" / "data"
    sidecar_files = (
        sidecar_dir / "simple_diverted_currents_PaxisIp.pk",
        sidecar_dir / "simple_limited_currents_PaxisIp.pk",
    )
    if not all(path.exists() for path in sidecar_files):
        return _existing_artifact_record("freegsnke_mastu_current_sidecars_public")

    # The upstream sidecars are Python pickle files.  They are useful public
    # provenance pointers, but this converter must not deserialize pickle from
    # an external cache.  Keep the previously tracked sanitized JSON artifact
    # when it exists; otherwise leave the source blocked instead of executing
    # untrusted object payloads.
    return _existing_artifact_record("freegsnke_mastu_current_sidecars_public")


def _blocking_sources(accepted_surfaces: set[str]) -> list[dict[str, str]]:
    blockers = [
        {
            "surface": "native_nonlinear_gyrokinetics",
            "source_family": "GENE/CGYRO/GS2",
            "reason": (
                "cached public sources contain input decks, docs, GYRO linear outputs, or restart "
                "files, but no complete public nonlinear output artifact with the required heat-flux, "
                "zonal-flow, saturation, and electromagnetic field observables"
            ),
        },
        {
            "surface": "runaway_electrons",
            "source_family": "DREAM",
            "reason": (
                "DREAM avalanche HDF5 data was converted as a partial raw output artifact, but it "
                "does not contain the required f_p_xi_t, runaway_current_t, synchrotron_loss_power_t, "
                "and partial_screening_drag_t observables under the current manifest"
            ),
        },
        {
            "surface": "impurity_transport",
            "source_family": "Aurora/STRAHL",
            "reason": (
                "Aurora cache contains examples and docs, but no redistributed Aurora/STRAHL output "
                "artifact with charge-state density, total density, radiation, ionisation, and "
                "recombination matrices"
            ),
        },
        {
            "surface": "free_boundary_equilibrium",
            "source_family": "FreeGS/FreeGSNKE",
            "reason": (
                "FreeGSNKE baselines and current sidecars were converted as partial raw artifacts, "
                "but strict FreeGS parity still needs boundary/limiter metadata, axis/X-point data, "
                "and native psi comparison for the same public case"
            ),
        },
    ]
    return [blocker for blocker in blockers if blocker["surface"] not in accepted_surfaces]


def run_conversion(*, write: bool = True) -> dict[str, Any]:
    """Convert available public output payloads and return a fail-closed report."""
    manifest = _load_manifest()
    converted = []
    for converter in (
        _convert_aurora_transport,
        _convert_dream_avalanche,
        _convert_freegsnke_baseline,
        _convert_freegsnke_current_sidecars,
    ):
        record = converter(manifest, write=write)
        if record is not None:
            converted.append(record)

    accepted = [record for record in converted if record["accepted_full_fidelity"]]
    partial = [record for record in converted if not record["accepted_full_fidelity"]]
    accepted_surfaces = {str(record["surface"]) for record in accepted}
    manifest_updated = _apply_accepted_records_to_manifest(manifest, accepted)
    report = {
        "accepted_full_fidelity_artifacts": len(accepted),
        "blocking_sources": _blocking_sources(accepted_surfaces),
        "conversion_modes": sorted(
            {
                str(record.get("conversion_mode", "external_cache_conversion"))
                for record in converted
            }
        ),
        "converted_artifacts": converted,
        "description": (
            "Conversion of cached public upstream outputs into tracked artifacts. Accepted public "
            "reference artifacts satisfy the manifest observable/provenance/checksum contract; "
            "partial outputs remain outside full-fidelity acceptance until required observables "
            "and solver-output comparisons are present."
        ),
        "partial_output_artifacts": len(partial),
        "reference_manifest": _rel(REFERENCE_CASES),
        "reference_manifest_updated": manifest_updated,
        "schema": "full-fidelity-reference-artifact-conversion.v1",
        "status": (
            "accepted_public_reference_artifact_available"
            if accepted
            else "partial_public_outputs_converted_not_full_fidelity"
        ),
    }
    if write:
        if manifest_updated:
            REFERENCE_CASES.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        write_reports(report)
    return report


def write_reports(report: dict[str, Any]) -> None:
    """Write JSON and Markdown conversion reports."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Full-Fidelity Reference Artifact Conversion",
        "",
        report["description"],
        "",
        f"- Schema: `{report['schema']}`",
        f"- Status: `{report['status']}`",
        f"- Accepted full-fidelity artifacts: `{report['accepted_full_fidelity_artifacts']}`",
        f"- Partial public output artifacts: `{report['partial_output_artifacts']}`",
        f"- Conversion modes: `{', '.join(report['conversion_modes'])}`",
        f"- Reference manifest updated: `{report['reference_manifest_updated']}`",
        "",
        "## Converted public output artifacts",
        "",
        "| Artifact | Surface | Family | Accepted | Comparison ready | Missing required observables | Path |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for artifact in report["converted_artifacts"]:
        missing = ", ".join(artifact["missing_required_observables"]) or "none"
        lines.append(
            "| {artifact_id} | {surface} | {family} | {accepted} | {comparison} | {missing} | `{path}` |".format(
                artifact_id=artifact["artifact_id"],
                surface=artifact["surface"],
                family=artifact["reference_family"],
                accepted=artifact["accepted_full_fidelity"],
                comparison=artifact["solver_output_comparison_ready"],
                missing=missing,
                path=artifact["artifact_path"],
            )
        )
    lines.extend(["", "## Blocking sources", ""])
    for blocker in report["blocking_sources"]:
        lines.append(
            "- {surface} ({family}): {reason}".format(
                surface=blocker["surface"],
                family=blocker["source_family"],
                reason=blocker["reason"],
            )
        )
    lines.append("")
    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run reference artifact conversion and optionally enforce exportability.

    Args:
        argv: Optional CLI argument list. When ``None``, uses ``sys.argv``.

    Returns:
        Exit status code. ``0`` when conversion succeeds and pass checks are met,
        ``1`` when ``--check`` is enabled but no public output artifacts were
        exported.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run conversion and fail if no public output artifacts can be exported.",
    )
    args = parser.parse_args(argv)
    report = run_conversion(write=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.check and report["partial_output_artifacts"] == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
