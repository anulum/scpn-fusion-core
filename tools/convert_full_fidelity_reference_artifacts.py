#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
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
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "data" / "external" / "full_fidelity_public_sources"
ARTIFACT_DIR = ROOT / "validation" / "reference_data" / "full_fidelity_public_artifacts"
REFERENCE_CASES = ROOT / "validation" / "reference_data" / "full_fidelity_reference_cases.json"
REPORT_DIR = ROOT / "validation" / "reports"
JSON_REPORT = REPORT_DIR / "full_fidelity_reference_artifact_conversion.json"
MD_REPORT = REPORT_DIR / "full_fidelity_reference_artifact_conversion.md"


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo: Path) -> str | None:
    try:
        result = subprocess.run(  # nosec B603: fixed git argv, shell disabled.
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return result.stdout.strip()


def _load_manifest() -> dict[str, Any]:
    manifest = json.loads(REFERENCE_CASES.read_text(encoding="utf-8"))
    if manifest.get("schema") != "full-fidelity-reference-cases.v1":
        raise ValueError("full-fidelity reference manifest schema mismatch")
    return manifest


def _surface_required_observables(manifest: dict[str, Any], surface: str) -> list[str]:
    cases = manifest.get("surfaces", {}).get(surface, {}).get("required_cases", [])
    if not cases:
        return []
    observables = cases[0].get("required_observables", [])
    return [str(name) for name in observables] if isinstance(observables, list) else []


def _finite_payload(arrays: dict[str, np.ndarray]) -> bool:
    return bool(arrays) and all(
        array.size > 0 and bool(np.all(np.isfinite(np.asarray(array, dtype=float))))
        for array in arrays.values()
    )


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_record(
    metadata: dict[str, Any], artifact_path: Path, metadata_path: Path
) -> dict[str, Any]:
    return {
        "accepted_full_fidelity": bool(metadata["accepted_full_fidelity"]),
        "artifact_id": metadata["artifact_id"],
        "artifact_path": _rel(artifact_path),
        "available_observables": metadata["available_observables"],
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


def _convert_dream_avalanche(manifest: dict[str, Any], *, write: bool) -> dict[str, Any] | None:
    try:
        import h5py
    except ImportError:
        return None

    repo = CACHE_ROOT / "repos" / "dream"
    source = repo / "tests" / "physics" / "DREAM_avalanche" / "DREAM-avalanche-data.h5"
    if not source.exists():
        return None

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
        return None

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


def _blocking_sources() -> list[dict[str, str]]:
    return [
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
                "FreeGSNKE baselines were converted as partial raw outputs, but strict FreeGS parity "
                "still needs coil-current sidecars, boundary/limiter metadata, axis/X-point data, "
                "and native psi comparison for the same public case"
            ),
        },
    ]


def run_conversion(*, write: bool = True) -> dict[str, Any]:
    """Convert available public output payloads and return a fail-closed report."""
    manifest = _load_manifest()
    converted = []
    for converter in (_convert_dream_avalanche, _convert_freegsnke_baseline):
        record = converter(manifest, write=write)
        if record is not None:
            converted.append(record)

    accepted = [record for record in converted if record["accepted_full_fidelity"]]
    partial = [record for record in converted if not record["accepted_full_fidelity"]]
    report = {
        "accepted_full_fidelity_artifacts": len(accepted),
        "blocking_sources": _blocking_sources(),
        "converted_artifacts": converted,
        "description": (
            "Conversion of cached public upstream outputs into tracked artifacts. Partial outputs "
            "remain outside full-fidelity acceptance until required observables and solver-output "
            "comparisons are present."
        ),
        "partial_output_artifacts": len(partial),
        "reference_manifest": _rel(REFERENCE_CASES),
        "reference_manifest_updated": False,
        "schema": "full-fidelity-reference-artifact-conversion.v1",
        "status": "partial_public_outputs_converted_not_full_fidelity",
    }
    if write:
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
