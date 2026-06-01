#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Generate a public Aurora/Open-ADAS impurity reference artifact.

This harness executes a public Aurora atomic-data path from the cached upstream
source tree. It intentionally exports only an ADAS-backed charge-state
fractional-abundance artifact and keeps full Aurora/STRAHL transport parity
fail-closed until transport, source/sink, radiation, and native same-case
comparison evidence exist.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import importlib
import json
import os
import shutil

# Aurora source provenance probing requires subprocess with fixed argv and timeouts.
import subprocess  # nosec B404
import sys
import warnings
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "data" / "external" / "full_fidelity_public_sources"
AURORA_REPO = CACHE_ROOT / "repos" / "aurora"
AURORA_ADAS_DIR = CACHE_ROOT / "adas" / "aurora"
ARTIFACT_DIR = ROOT / "validation" / "reference_data" / "full_fidelity_public_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "aurora_argon_fractional_abundance_public.npz"
METADATA_PATH = ARTIFACT_DIR / "aurora_argon_fractional_abundance_public.metadata.json"
REPORT_DIR = ROOT / "validation" / "reports"
JSON_REPORT = REPORT_DIR / "aurora_reference_execution_artifact.json"
MD_REPORT = REPORT_DIR / "aurora_reference_execution_artifact.md"
AURORA_STRAHL_OUTPUT_CONTRACT = {
    "coordinate_axes": ["time_s", "radius_m", "charge_state"],
    "observables": [
        "charge_state_density_r_t",
        "total_impurity_density_r_t",
        "line_radiation_power_t",
        "line_radiation_power_t_r_z",
        "source_sink_matrix_t_r_z_z",
        "total_impurity_inventory_t",
    ],
    "schema": "aurora-strahl-output-contract.v1",
}


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_deterministic_npz(path: Path, arrays: dict[str, NDArray[Any]]) -> None:
    """Write compressed NPZ bytes with stable member order and timestamps."""
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


def _git_commit(repo: Path) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        # Fixed git argv, shell disabled, bounded timeout.
        result = subprocess.run(  # nosec B603
            [git, "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return result.stdout.strip()


def _blocked_report(status: str, reason: str) -> dict[str, Any]:
    return {
        "accepted_full_fidelity_ready": False,
        "artifact_generated": False,
        "case_id": "aurora_argon_fractional_abundance_public",
        "description": (
            "Public Aurora/Open-ADAS execution request for an argon charge-state "
            "fractional-abundance artifact."
        ),
        "missing_full_fidelity_requirements": [
            "charge-state-resolved radial transport run with Aurora or STRAHL",
            "finite ionisation/recombination/source/sink matrix conservation evidence",
            "line-radiation and total radiated-power observables on the same case",
            "native same-case impurity solver-output comparison",
            "quantitative parity thresholds against public Aurora/STRAHL output",
        ],
        "next_action": reason,
        "reference_output_ready": False,
        "required_output_contract": AURORA_STRAHL_OUTPUT_CONTRACT,
        "schema": "aurora-reference-execution-artifact.v1",
        "same_case_comparison_ready": False,
        "source_family": "Aurora",
        "source_repo": _rel(AURORA_REPO) if AURORA_REPO.exists() else None,
        "status": status,
    }


def _load_aurora() -> Any:
    if not AURORA_REPO.exists():
        raise FileNotFoundError(f"cached Aurora source is missing: {_rel(AURORA_REPO)}")
    os.environ["AURORA_ADAS_DIR"] = str(AURORA_ADAS_DIR)
    sys.path.insert(0, str(AURORA_REPO))
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    return importlib.import_module("aurora")


def _source_data_files(aurora: Any, ion: str, filetypes: list[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    files = aurora.adas_files.adas_files_dict()[ion]
    for filetype in filetypes:
        source = Path(aurora.adas_files.get_adas_file_loc(files[filetype], filetype="adf11"))
        out.append(
            {
                "filetype": filetype,
                "path": _rel(source),
                "sha256": _sha256(source),
            }
        )
    return out


def _generate_artifact(*, write: bool) -> dict[str, Any]:
    aurora = _load_aurora()
    ion = "Ar"
    filetypes = ["acd", "scd", "ccd"]
    atom_data = aurora.atomic.get_atom_data(ion, filetypes)

    rhop = np.linspace(0.0, 1.0, 64)
    ne_cm3 = (1.0e14 - 0.4e14) * (1.0 - rhop**2.0) ** 0.5 + 0.4e14
    Te_eV = (5.0e3 - 100.0) * (1.0 - rhop**2.0) ** 1.5 + 100.0
    n0_by_ne = (1.0e-2 * np.exp(rhop**5.0 - 1.0)) ** 2

    Te_no_cx, fz_no_cx = aurora.atomic.get_frac_abundances(
        atom_data,
        ne_cm3,
        Te_eV,
        rho=rhop,
        plot=False,
    )
    Te_with_cx, fz_with_cx = aurora.atomic.get_frac_abundances(
        atom_data,
        ne_cm3,
        Te_eV,
        n0_by_ne,
        rho=rhop,
        plot=False,
    )
    arrays = {
        "charge_state_index": np.arange(fz_no_cx.shape[1], dtype=np.int64),
        "fraction_sum_no_cx": np.sum(fz_no_cx, axis=1),
        "fraction_sum_with_cx": np.sum(fz_with_cx, axis=1),
        "fz_no_cx": np.asarray(fz_no_cx, dtype=float),
        "fz_with_cx": np.asarray(fz_with_cx, dtype=float),
        "ne_cm3": np.asarray(ne_cm3, dtype=float),
        "neutral_fraction_n0_by_ne": np.asarray(n0_by_ne, dtype=float),
        "rhop": np.asarray(rhop, dtype=float),
        "returned_Te_grid_no_cx_eV": np.asarray(Te_no_cx, dtype=float),
        "returned_Te_grid_with_cx_eV": np.asarray(Te_with_cx, dtype=float),
        "Te_eV": np.asarray(Te_eV, dtype=float),
    }
    finite = all(bool(np.all(np.isfinite(array))) and array.size > 0 for array in arrays.values())
    source_files = _source_data_files(aurora, ion, filetypes)
    commit = _git_commit(AURORA_REPO)

    metadata = {
        "accepted_full_fidelity": False,
        "artifact_id": "aurora_argon_fractional_abundance_public",
        "artifact_path": _rel(ARTIFACT_PATH),
        "artifact_role": "partial_public_solver_output",
        "available_observables": sorted(arrays),
        "case_contract": {
            "impurity": ion,
            "radial_coordinate": "rhop",
            "radial_points": int(rhop.size),
            "charge_states": int(fz_no_cx.shape[1]),
            "electron_density_units": "cm^-3",
            "electron_temperature_units": "eV",
        },
        "finite_numeric_payload": finite,
        "metadata_schema": "full-fidelity-public-output-artifact-metadata.v1",
        "missing_required_observables": [
            "transported_charge_state_density_time_radius_charge",
            "line_radiation_power_time_radius_charge",
            "ionisation_recombination_source_sink_matrix_time_radius_charge_charge",
            "total_impurity_inventory_closure",
            "native_same_case_solver_output_comparison",
        ],
        "provenance_url": (
            f"https://github.com/fsciortino/Aurora/tree/{commit or 'master'}/examples"
        ),
        "redistribution_license": (
            "MIT with Aurora User Agreement citation notice; Open-ADAS public derived data"
        ),
        "reference_family": "Aurora",
        "sha256": "",
        "solver_output_comparison_ready": False,
        "solver_output_comparison_status": (
            "blocked_no_native_same_case_transport_comparison_for_partial_atomic_artifact"
        ),
        "source_data_files": source_files,
        "surface": "impurity_transport",
        "upstream_commit": commit,
    }
    if write:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        _write_deterministic_npz(ARTIFACT_PATH, arrays)
        metadata["sha256"] = _sha256(ARTIFACT_PATH)
        METADATA_PATH.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif ARTIFACT_PATH.exists():
        metadata["sha256"] = _sha256(ARTIFACT_PATH)

    return {
        "artifact_path": _rel(ARTIFACT_PATH),
        "metadata_path": _rel(METADATA_PATH),
        "metadata": metadata,
    }


def build_aurora_reference_execution_report(*, write: bool = True) -> dict[str, Any]:
    """Generate the Aurora/Open-ADAS partial reference artifact and report."""
    try:
        generated = _generate_artifact(write=write)
    except FileNotFoundError as exc:
        report = _blocked_report("blocked_missing_aurora_source_cache", str(exc))
    except Exception as exc:  # pragma: no cover - exercised by environment blockers.
        report = _blocked_report(
            "blocked_aurora_reference_artifact_generation_failed",
            f"{type(exc).__name__}: {exc}",
        )
    else:
        metadata = generated["metadata"]
        report = {
            "accepted_full_fidelity_ready": False,
            "artifact": {
                "artifact_id": metadata["artifact_id"],
                "artifact_path": generated["artifact_path"],
                "available_observables": metadata["available_observables"],
                "finite_numeric_payload": bool(metadata["finite_numeric_payload"]),
                "metadata_path": generated["metadata_path"],
                "redistribution_license": metadata["redistribution_license"],
                "sha256": metadata["sha256"],
                "solver_output_comparison_ready": False,
            },
            "artifact_generated": True,
            "case_id": "aurora_argon_fractional_abundance_public",
            "description": (
                "Public Aurora/Open-ADAS argon fractional-abundance execution artifact. "
                "This is a partial atomic-physics output, not full transport parity."
            ),
            "missing_full_fidelity_requirements": metadata["missing_required_observables"],
            "next_action": (
                "Run a public Aurora or STRAHL radial transport case with line radiation, "
                "source/sink matrices, inventory closure, and native same-case comparison."
            ),
            "reference_output_ready": True,
            "required_output_contract": AURORA_STRAHL_OUTPUT_CONTRACT,
            "schema": "aurora-reference-execution-artifact.v1",
            "same_case_comparison_ready": False,
            "source_family": "Aurora",
            "source_repo": _rel(AURORA_REPO),
            "status": "blocked_partial_public_atomic_artifact_not_transport_parity",
        }
    if write:
        write_reports(report)
    return report


def write_reports(report: dict[str, Any]) -> None:
    """Write JSON and Markdown Aurora execution reports."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Aurora Reference Execution Artifact",
        "",
        report["description"],
        "",
        f"- Schema: `{report['schema']}`",
        f"- Status: `{report['status']}`",
        f"- Artifact generated: `{report['artifact_generated']}`",
        f"- Reference output ready: `{report['reference_output_ready']}`",
        f"- Same-case comparison ready: `{report['same_case_comparison_ready']}`",
        f"- Accepted full-fidelity ready: `{report['accepted_full_fidelity_ready']}`",
        "",
        "## Required Aurora/STRAHL output contract",
        "",
        f"- Schema: `{report['required_output_contract']['schema']}`",
        f"- Coordinate axes: `{', '.join(report['required_output_contract']['coordinate_axes'])}`",
        f"- Observables: `{', '.join(report['required_output_contract']['observables'])}`",
        "",
        "## Next action",
        "",
        report["next_action"],
        "",
    ]
    if report.get("artifact"):
        artifact = report["artifact"]
        lines.extend(
            [
                "## Artifact",
                "",
                f"- Artifact: `{artifact['artifact_path']}`",
                f"- Metadata: `{artifact['metadata_path']}`",
                f"- SHA-256: `{artifact['sha256']}`",
                f"- Solver comparison ready: `{artifact['solver_output_comparison_ready']}`",
                "",
            ]
        )
    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    report = build_aurora_reference_execution_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
