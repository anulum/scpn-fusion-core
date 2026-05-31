#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Inventory public nonlinear GK reference decks and partial outputs.

This tool records redistribution-safe deck/output hashes from cached public
GS2 and GACODE/CGYRO sources. It does not promote input decks or regression
precision snippets to full nonlinear 5D parity: accepted parity still requires
runnable external-solver outputs, quantitative thresholds, convergence
evidence, and same-case native solver comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "data" / "external" / "full_fidelity_public_sources"
GS2_REPO = CACHE_ROOT / "repos" / "gs2"
GACODE_REPO = CACHE_ROOT / "repos" / "gacode"
WEB_DIR = CACHE_ROOT / "web"
ARTIFACT_DIR = ROOT / "validation" / "reference_data" / "full_fidelity_public_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "gk_public_reference_deck_inventory.json"
METADATA_PATH = ARTIFACT_DIR / "gk_public_reference_deck_inventory.metadata.json"
REPORT_DIR = ROOT / "validation" / "reports"
JSON_REPORT = REPORT_DIR / "gk_public_reference_deck_inventory.json"
MD_REPORT = REPORT_DIR / "gk_public_reference_deck_inventory.md"

KEY_VALUE_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*([^!#/]*)")


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


def _parse_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = KEY_VALUE_RE.match(line)
        if match:
            values[match.group(1).upper()] = match.group(2).strip().strip('"').strip("'")
    return values


def _number(values: dict[str, str], key: str) -> float | int | None:
    raw = values.get(key)
    if raw is None:
        return None
    try:
        value = float(raw.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None
    return int(value) if value.is_integer() else value


def _deck_record(path: Path, solver: str, role: str, commit: str | None) -> dict[str, Any]:
    values = _parse_key_values(path)
    if solver == "GS2":
        grid = {
            "nx": _number(values, "NX"),
            "ny": _number(values, "NY"),
            "ntheta": _number(values, "NTHETA"),
            "nperiod": _number(values, "NPERIOD"),
            "npassing": _number(values, "NPASSING"),
            "nesub": _number(values, "NESUB"),
            "nesuper": _number(values, "NESUPER"),
            "nspec": _number(values, "NSPEC"),
        }
        nonlinear = values.get("NONLINEAR_MODE", "").lower() == "on"
        electromagnetic = values.get("FAPAR", "0") not in {"0", "0.0", "0.00000000000E+00"}
    else:
        grid = {
            "n_radial": _number(values, "N_RADIAL"),
            "n_toroidal": _number(values, "N_TOROIDAL"),
            "n_theta": _number(values, "N_THETA"),
            "n_energy": _number(values, "N_ENERGY"),
            "n_xi": _number(values, "N_XI"),
            "n_species": _number(values, "N_SPECIES"),
            "n_field": _number(values, "N_FIELD"),
        }
        nonlinear = values.get("NONLINEAR_FLAG") == "1" or path.parent.name.startswith("nl")
        electromagnetic = values.get("N_FIELD") not in {None, "1"}
    return {
        "commit": commit,
        "grid_contract": {key: value for key, value in grid.items() if value is not None},
        "nonlinear": nonlinear,
        "electromagnetic": electromagnetic,
        "path": _rel(path),
        "role": role,
        "sha256": _sha256(path),
        "solver_family": solver,
    }


def _cgyro_output_record(path: Path, commit: str | None) -> dict[str, Any]:
    values: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            values.append(float(text.replace("D", "E").replace("d", "e")))
        except ValueError:
            continue
    arr = np.asarray(values, dtype=float)
    return {
        "commit": commit,
        "finite_numeric_payload": bool(arr.size and np.all(np.isfinite(arr))),
        "max": float(np.max(arr)) if arr.size else None,
        "min": float(np.min(arr)) if arr.size else None,
        "numeric_count": int(arr.size),
        "path": _rel(path),
        "role": "cgyro_regression_precision_output",
        "sha256": _sha256(path),
        "solver_family": "CGYRO",
    }


def _web_record(path: Path, solver: str) -> dict[str, str]:
    return {
        "path": _rel(path),
        "sha256": _sha256(path),
        "solver_family": solver,
    }


def _backend_probe() -> dict[str, Any]:
    cgyro_wrapper = GACODE_REPO / "cgyro" / "bin" / "cgyro"
    gs2_path = shutil.which("gs2")
    gacode_printversion = shutil.which("gacode_printversion")
    return {
        "cgyro_wrapper_present": cgyro_wrapper.is_file(),
        "cgyro_wrapper_path": _rel(cgyro_wrapper) if cgyro_wrapper.exists() else None,
        "gacode_printversion_available": bool(gacode_printversion),
        "gacode_printversion_path": gacode_printversion,
        "gs2_available": bool(gs2_path),
        "gs2_path": gs2_path,
        "runnable_external_backends": bool(gs2_path and gacode_printversion),
    }


def build_gk_public_deck_inventory(*, write: bool = True) -> dict[str, Any]:
    """Build and optionally write the GK public deck inventory report."""
    gs2_commit = _git_commit(GS2_REPO) if GS2_REPO.exists() else None
    gacode_commit = _git_commit(GACODE_REPO) if GACODE_REPO.exists() else None

    decks: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    web_sources: list[dict[str, str]] = []
    if GS2_REPO.exists():
        for path in sorted((GS2_REPO / "tests" / "nonlinear_tests").rglob("*.in")):
            decks.append(_deck_record(path, "GS2", "public_nonlinear_input_deck", gs2_commit))
    if GACODE_REPO.exists():
        for path in sorted((GACODE_REPO / "cgyro" / "tools" / "input").glob("nl*/input.cgyro")):
            decks.append(_deck_record(path, "CGYRO", "public_nonlinear_input_deck", gacode_commit))
        for path in sorted((GACODE_REPO / "cgyro" / "tools" / "input").glob("reg*/input.cgyro")):
            decks.append(_deck_record(path, "CGYRO", "public_regression_input_deck", gacode_commit))
        for path in sorted((GACODE_REPO / "cgyro" / "tools" / "input").glob("reg*/out.cgyro.prec")):
            outputs.append(_cgyro_output_record(path, gacode_commit))
    for filename, solver in [
        ("genecode.org.html", "GENE"),
        ("genecode_main.html", "GENE"),
        ("gs2.html", "GS2"),
        ("gs2_user_manual.html", "GS2"),
        ("cgyro.html", "CGYRO"),
        ("gacode_cgyro.html", "CGYRO"),
    ]:
        path = WEB_DIR / filename
        if path.exists():
            web_sources.append(_web_record(path, solver))

    finite_outputs = all(item["finite_numeric_payload"] for item in outputs) if outputs else False
    artifact = {
        "schema": "gk-public-reference-deck-inventory.v1",
        "surface": "native_nonlinear_gyrokinetics",
        "accepted_full_fidelity": False,
        "decks": decks,
        "output_summaries": outputs,
        "web_sources": web_sources,
    }
    metadata = {
        "accepted_full_fidelity": False,
        "artifact_id": "gk_public_reference_deck_inventory",
        "artifact_path": _rel(ARTIFACT_PATH),
        "artifact_role": "partial_public_input_output_inventory",
        "available_observables": [
            "public_input_deck_checksums",
            "public_cgyro_regression_precision_values",
            "public_web_source_checksums",
        ],
        "deck_count": len(decks),
        "finite_numeric_payload": finite_outputs,
        "metadata_schema": "full-fidelity-public-output-artifact-metadata.v1",
        "missing_required_observables": [
            "same-deck external nonlinear distribution output",
            "heat_flux_spectra_time_kx_ky_species",
            "field_energy_history_phi_apar_bpar",
            "zonal_flow_and_saturation_metrics",
            "native_same_case_solver_output_comparison",
            "grid_convergence_and_production_scale_scaling_evidence",
        ],
        "output_summary_count": len(outputs),
        "redistribution_license": "GS2 MIT; GACODE/CGYRO Apache-2.0; GENE web pages are source links only",
        "reference_family": "GENE/CGYRO/GS2",
        "sha256": "",
        "solver_output_comparison_ready": False,
        "solver_output_comparison_status": (
            "blocked_decks_and_partial_regression_outputs_not_full_nonlinear_parity"
        ),
        "surface": "native_nonlinear_gyrokinetics",
    }
    if write:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        metadata["sha256"] = _sha256(ARTIFACT_PATH)
        METADATA_PATH.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    elif ARTIFACT_PATH.exists():
        metadata["sha256"] = _sha256(ARTIFACT_PATH)

    backend = _backend_probe()
    status = (
        "blocked_public_gk_decks_indexed_missing_solver_output_parity"
        if decks
        else "blocked_missing_gk_public_source_cache"
    )
    report = {
        "accepted_full_fidelity_ready": False,
        "artifact_path": _rel(ARTIFACT_PATH),
        "backend_probe": backend,
        "deck_count": len(decks),
        "description": (
            "Public GENE/CGYRO/GS2 deck inventory for nonlinear GK parity. "
            "This is an input/output provenance artifact, not accepted full-fidelity parity."
        ),
        "gacode_commit": gacode_commit,
        "gs2_commit": gs2_commit,
        "metadata_path": _rel(METADATA_PATH),
        "missing_full_fidelity_requirements": metadata["missing_required_observables"],
        "next_action": (
            "Run GS2/CGYRO/GENE nonlinear decks with public outputs, then compare native "
            "5D nonlinear GK heat-flux spectra, field energy, saturation, and convergence."
        ),
        "output_summary_count": len(outputs),
        "reference_output_ready": False,
        "schema": "gk-public-reference-deck-inventory-report.v1",
        "sha256": metadata["sha256"],
        "status": status,
        "web_source_count": len(web_sources),
    }
    if write:
        write_reports(report)
    return report


def write_reports(report: dict[str, Any]) -> None:
    """Write JSON and Markdown GK inventory reports."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# GK Public Reference Deck Inventory",
        "",
        report["description"],
        "",
        f"- Schema: `{report['schema']}`",
        f"- Status: `{report['status']}`",
        f"- Decks indexed: `{report['deck_count']}`",
        f"- Output summaries indexed: `{report['output_summary_count']}`",
        f"- Web sources indexed: `{report['web_source_count']}`",
        f"- Artifact: `{report['artifact_path']}`",
        f"- Metadata: `{report['metadata_path']}`",
        f"- SHA-256: `{report['sha256']}`",
        f"- Accepted full-fidelity ready: `{report['accepted_full_fidelity_ready']}`",
        "",
        "## Backend probe",
        "",
        f"- GS2 available: `{report['backend_probe']['gs2_available']}`",
        f"- CGYRO wrapper present: `{report['backend_probe']['cgyro_wrapper_present']}`",
        (
            "- GACODE runtime helper available: "
            f"`{report['backend_probe']['gacode_printversion_available']}`"
        ),
        "",
        "## Next action",
        "",
        report["next_action"],
        "",
    ]
    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    report = build_gk_public_deck_inventory()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
