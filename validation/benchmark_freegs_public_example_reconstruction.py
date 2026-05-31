#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Attempt same-case reconstruction against cached public FreeGS examples.

This benchmark is intentionally fail-closed. It validates the native circular
filament Green-function convention against FreeGS public-example machine coils
and records the external nonlinear solve attempt. It does not accept strict
free-boundary parity until a converged FreeGS/FreeGSNKE public example is
compared against native same-case ``psi(R,Z)`` output with thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import importlib
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.fusion_kernel_free_boundary import green_function

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "data" / "external" / "full_fidelity_public_sources"
FREEGS_REPO = CACHE_ROOT / "repos" / "freegs"
ARTIFACT_DIR = ROOT / "validation" / "reference_data" / "full_fidelity_public_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "freegs_public_example_reconstruction_attempt.json"
METADATA_PATH = ARTIFACT_DIR / "freegs_public_example_reconstruction_attempt.metadata.json"
REPORT_DIR = ROOT / "validation" / "reports"
JSON_REPORT = REPORT_DIR / "freegs_public_example_reconstruction.json"
MD_REPORT = REPORT_DIR / "freegs_public_example_reconstruction.md"
VACUUM_NRMSE_THRESHOLD = 1.0e-12
VACUUM_MAX_ABS_THRESHOLD = 1.0e-12


@dataclass(frozen=True)
class FreeGSPublicExampleCase:
    """Single public FreeGS example reconstruction attempt specification."""

    case_id: str
    example_path: Path
    machine_class: str
    r_min: float
    r_max: float
    z_min: float
    z_max: float
    nx: int
    ny: int
    p_axis_pa: float
    plasma_current_a: float
    fvac: float
    xpoints: tuple[tuple[float, float], ...]
    isoflux: tuple[tuple[float, float, float, float], ...]
    boundary_name: str | None = None
    gamma: float | None = None
    nonlinear_attempts: tuple[tuple[int, float], ...] = ((5, 0.0), (50, 0.0))


PUBLIC_CASES = (
    FreeGSPublicExampleCase(
        case_id="freegs_01_test_tokamak_freeboundary",
        example_path=FREEGS_REPO / "01-freeboundary.py",
        machine_class="TestTokamak",
        r_min=0.1,
        r_max=2.0,
        z_min=-1.0,
        z_max=1.0,
        nx=65,
        ny=65,
        p_axis_pa=1.0e3,
        plasma_current_a=2.0e5,
        fvac=2.0,
        xpoints=((1.1, -0.6), (1.1, 0.8)),
        isoflux=((1.1, -0.6, 1.1, 0.6),),
        boundary_name="freeBoundaryHagenow",
    ),
    FreeGSPublicExampleCase(
        case_id="freegs_16_diiid_public_example",
        example_path=FREEGS_REPO / "16-DIIID.py",
        machine_class="DIIID",
        r_min=0.1,
        r_max=2.8,
        z_min=-1.8,
        z_max=1.8,
        nx=65,
        ny=65,
        p_axis_pa=159811.0,
        plasma_current_a=-1533632.0,
        fvac=-3.231962138124,
        xpoints=((1.285, -1.176), (1.2, 1.0)),
        isoflux=((1.285, -1.176, 1.2, 1.2),),
        gamma=1.0e-12,
        nonlinear_attempts=((5, 0.0), (25, 0.0)),
    ),
)


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _import_freegs() -> tuple[ModuleType | None, str | None, str | None]:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            module = importlib.import_module("freegs")
    except Exception as exc:  # pragma: no cover - depends on optional backend install.
        return None, None, f"{type(exc).__name__}: {exc}"
    _patch_freegs_find_critical_scalar_derivative(module)
    return module, str(getattr(module, "__version__", "unknown")), None


def _patch_freegs_find_critical_scalar_derivative(freegs_module: ModuleType) -> None:
    """Patch FreeGS 0.8.2 scalar derivative extraction for current SciPy."""
    critical = getattr(freegs_module, "critical", None)
    if critical is None or not hasattr(critical, "find_critical"):
        return
    func = critical.find_critical
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return
    needle = "f(R1, Z1, dx=2) / R1"
    replacement = "f(R1, Z1, dx=2)[0][0] / R1"
    if needle not in source or replacement in source:
        return
    patched_source = source.replace(needle, replacement, 1)
    namespace = dict(func.__globals__)
    # Fixed upstream FreeGS source rewrite for one scalar-indexing bug.
    exec(compile(patched_source, func.__code__.co_filename, "exec"), namespace)  # nosec B102
    critical.find_critical = namespace["find_critical"]


def _make_equilibrium(
    freegs: ModuleType, spec: FreeGSPublicExampleCase
) -> tuple[Any, Any, Any, Any]:
    tokamak = getattr(freegs.machine, spec.machine_class)()
    kwargs: dict[str, Any] = {
        "tokamak": tokamak,
        "Rmin": spec.r_min,
        "Rmax": spec.r_max,
        "Zmin": spec.z_min,
        "Zmax": spec.z_max,
        "nx": spec.nx,
        "ny": spec.ny,
    }
    if spec.boundary_name is not None:
        kwargs["boundary"] = getattr(freegs.boundary, spec.boundary_name)
    eq = freegs.Equilibrium(**kwargs)
    profiles = freegs.jtor.ConstrainPaxisIp(
        eq,
        spec.p_axis_pa,
        spec.plasma_current_a,
        spec.fvac,
    )
    constrain_kwargs: dict[str, Any] = {
        "xpoints": [tuple(point) for point in spec.xpoints],
        "isoflux": [tuple(row) for row in spec.isoflux],
    }
    if spec.gamma is not None:
        constrain_kwargs["gamma"] = spec.gamma
    constrain = freegs.control.constrain(**constrain_kwargs)
    constrain(eq)
    return tokamak, eq, profiles, constrain


def _coil_filament_terms(coil: Any) -> list[tuple[float, float, float]]:
    coil_obj = cast(Any, coil)
    current = float(coil_obj.current)
    turns = float(coil_obj.turns if hasattr(coil_obj, "turns") else 1.0)
    if hasattr(coil, "_points"):
        return [
            (float(r_fil), float(z_fil), current * turns * float(weight))
            for r_fil, z_fil, weight in coil_obj._points
        ]
    if hasattr(coil, "coils"):
        terms: list[tuple[float, float, float]] = []
        for _, subcoil, _ in coil_obj.coils:
            terms.extend(_coil_filament_terms(subcoil))
        return terms
    return [(float(coil_obj.R), float(coil_obj.Z), current * turns)]


def _machine_filaments(
    tokamak: Any,
) -> tuple[list[dict[str, Any]], list[tuple[float, float, float]]]:
    records: list[dict[str, Any]] = []
    filaments: list[tuple[float, float, float]] = []
    for name, coil in getattr(tokamak, "coils", []):
        terms = _coil_filament_terms(coil)
        filaments.extend(terms)
        current = float(getattr(coil, "current", 0.0))
        turns = float(getattr(coil, "turns", 1.0))
        records.append(
            {
                "coil_type": type(coil).__name__,
                "current_a": current,
                "effective_current_a_turns": current * turns,
                "filament_count": len(terms),
                "name": str(name),
                "turns": turns,
            }
        )
    return records, filaments


def _sample_points(spec: FreeGSPublicExampleCase) -> NDArray[np.float64]:
    r_values = np.linspace(
        spec.r_min + 0.25 * (spec.r_max - spec.r_min),
        spec.r_max - 0.25 * (spec.r_max - spec.r_min),
        3,
    )
    z_values = np.linspace(
        spec.z_min + 0.25 * (spec.z_max - spec.z_min),
        spec.z_max - 0.25 * (spec.z_max - spec.z_min),
        3,
    )
    return np.asarray([(float(r), float(z)) for z in z_values for r in r_values], dtype=np.float64)


def _native_vacuum_psi(
    filaments: list[tuple[float, float, float]],
    points: NDArray[np.float64],
) -> NDArray[np.float64]:
    out = np.zeros(points.shape[0], dtype=np.float64)
    for r_src, z_src, current in filaments:
        for idx, (r_obs, z_obs) in enumerate(points):
            out[idx] += current * green_function(r_src, z_src, float(r_obs), float(z_obs))
    return out


def _vacuum_comparison(tokamak: Any, spec: FreeGSPublicExampleCase) -> dict[str, Any]:
    coil_records, filaments = _machine_filaments(tokamak)
    points = _sample_points(spec)
    freegs_values = np.asarray(
        [float(tokamak.psi(float(r_obs), float(z_obs))) for r_obs, z_obs in points],
        dtype=np.float64,
    )
    native_values = _native_vacuum_psi(filaments, points)
    residual = native_values - freegs_values
    scale = max(float(np.max(freegs_values) - np.min(freegs_values)), 1.0e-30)
    rmse = float(np.sqrt(np.mean(residual * residual)))
    nrmse = float(rmse / scale)
    max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
    pass_gate = bool(nrmse <= VACUUM_NRMSE_THRESHOLD and max_abs <= VACUUM_MAX_ABS_THRESHOLD)
    return {
        "coil_count": len(coil_records),
        "coils": coil_records,
        "filament_count": len(filaments),
        "freegs_vacuum_psi_sample": [float(value) for value in freegs_values],
        "max_abs_error_wb": max_abs,
        "native_vacuum_psi_sample": [float(value) for value in native_values],
        "nrmse": nrmse,
        "pass": pass_gate,
        "rmse_wb": rmse,
        "sample_point_count": int(points.shape[0]),
        "thresholds": {
            "max_abs_error_wb": VACUUM_MAX_ABS_THRESHOLD,
            "nrmse": VACUUM_NRMSE_THRESHOLD,
        },
    }


def _attempt_solve(
    freegs: ModuleType,
    spec: FreeGSPublicExampleCase,
    *,
    maxits: int,
    blend: float,
) -> dict[str, Any]:
    _, eq, profiles, constrain = _make_equilibrium(freegs, spec)
    start = time.perf_counter()
    try:
        freegs.solve(
            eq,
            profiles,
            constrain,
            show=False,
            maxits=maxits,
            blend=blend,
            convergenceInfo=False,
        )
    except Exception as exc:
        return {
            "blend": blend,
            "elapsed_s": float(time.perf_counter() - start),
            "error": str(exc)[:1000],
            "error_type": type(exc).__name__,
            "external_psi_finite": False,
            "maxits": maxits,
            "native_same_case_psi_comparison_ready": False,
            "status": "failed_external_backend_solve",
        }
    elapsed = float(time.perf_counter() - start)
    try:
        psi = np.asarray(eq.psi(), dtype=float)
        finite_psi = bool(psi.size and np.all(np.isfinite(psi)))
        psi_shape = list(psi.shape)
    except Exception as exc:  # pragma: no cover - backend-version dependent.
        finite_psi = False
        psi_shape = []
        return {
            "blend": blend,
            "elapsed_s": elapsed,
            "error": str(exc)[:1000],
            "error_type": type(exc).__name__,
            "external_psi_finite": False,
            "maxits": maxits,
            "native_same_case_psi_comparison_ready": False,
            "status": "failed_external_psi_extraction",
        }
    return {
        "blend": blend,
        "elapsed_s": elapsed,
        "external_psi_finite": finite_psi,
        "external_psi_max": float(np.max(psi)) if finite_psi else None,
        "external_psi_min": float(np.min(psi)) if finite_psi else None,
        "external_psi_shape": psi_shape,
        "maxits": maxits,
        "native_same_case_psi_comparison_ready": False,
        "status": "external_backend_solved_missing_native_same_case_profile_source_comparison",
    }


def _attempt_solve_sweep(freegs: ModuleType, spec: FreeGSPublicExampleCase) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for maxits, blend in spec.nonlinear_attempts:
        attempt = _attempt_solve(freegs, spec, maxits=maxits, blend=blend)
        attempts.append(attempt)
        if (
            attempt["status"]
            == "external_backend_solved_missing_native_same_case_profile_source_comparison"
        ):
            break
    successful = [
        attempt
        for attempt in attempts
        if attempt["status"]
        == "external_backend_solved_missing_native_same_case_profile_source_comparison"
    ]
    best = successful[0] if successful else attempts[-1]
    return {
        "attempts": attempts,
        "best_attempt": best,
        "external_nonlinear_output_ready": bool(successful),
    }


def _case_record(freegs: ModuleType, spec: FreeGSPublicExampleCase) -> dict[str, Any]:
    tokamak, _, _, _ = _make_equilibrium(freegs, spec)
    vacuum = _vacuum_comparison(tokamak, spec)
    solve_sweep = _attempt_solve_sweep(freegs, spec)
    solve = solve_sweep["best_attempt"]
    return {
        "accepted_full_fidelity": False,
        "case_id": spec.case_id,
        "example_path": _rel(spec.example_path),
        "example_sha256": _sha256(spec.example_path),
        "grid": {"nx": spec.nx, "ny": spec.ny},
        "machine_class": spec.machine_class,
        "missing_full_fidelity_requirements": [
            "native profile-source/free-boundary solve using same coil currents and profiles",
            "native-vs-FreeGS psi_N RMSE, axis, X-point, boundary-containment, and q-profile thresholds",
            "grid-convergence evidence for the public example",
        ],
        "external_nonlinear_output_ready": solve_sweep["external_nonlinear_output_ready"],
        "nonlinear_solve_attempt": solve,
        "nonlinear_solve_attempts": solve_sweep["attempts"],
        "source_contract": {
            "fvac": spec.fvac,
            "isoflux": [list(row) for row in spec.isoflux],
            "p_axis_pa": spec.p_axis_pa,
            "plasma_current_a": spec.plasma_current_a,
            "xpoints": [list(point) for point in spec.xpoints],
        },
        "vacuum_green_function_comparison": vacuum,
    }


def _blocked_status(cases: list[dict[str, Any]], backend_available: bool) -> str:
    if not backend_available:
        return "blocked_freegs_backend_unavailable"
    if cases and all(case["vacuum_green_function_comparison"]["pass"] for case in cases):
        if all(case["external_nonlinear_output_ready"] for case in cases):
            return "blocked_public_freegs_external_psi_ready_missing_native_same_case_comparison"
        return "blocked_public_freegs_vacuum_matched_missing_nonlinear_same_case_psi"
    return "blocked_public_freegs_vacuum_or_backend_contract_failed"


def _write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# FreeGS Public Example Reconstruction",
        "",
        "This benchmark attempts same-case reconstruction from cached public FreeGS",
        "examples. It accepts only the vacuum Green-function convention check and keeps",
        "strict free-boundary parity blocked until nonlinear same-case native-vs-FreeGS",
        "`psi(R,Z)` comparison evidence exists.",
        "",
        f"- Schema: `{report['schema']}`",
        f"- Status: `{report['status']}`",
        f"- Backend available: `{report['freegs_backend_available']}`",
        f"- FreeGS version: `{report['freegs_version']}`",
        f"- Case count: `{report['case_count']}`",
        f"- Vacuum comparison pass: `{report['vacuum_comparison_pass']}`",
        f"- External nonlinear output ready: `{report['external_nonlinear_output_ready']}`",
        f"- Accepted full fidelity: `{report['accepted_full_fidelity_ready']}`",
        f"- Artifact: `{report['artifact_path']}`",
        "",
        "| Case | Machine | Vacuum NRMSE | Vacuum pass | Nonlinear status |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for case in report["cases"]:
        vacuum = case["vacuum_green_function_comparison"]
        solve = case["nonlinear_solve_attempt"]
        lines.append(
            "| {case_id} | {machine} | {nrmse:.6e} | {vac_pass} | `{status}` |".format(
                case_id=case["case_id"],
                machine=case["machine_class"],
                nrmse=vacuum["nrmse"],
                vac_pass=vacuum["pass"],
                status=solve["status"],
            )
        )
    lines.extend(["", "## Missing full-fidelity requirements", ""])
    for item in report["missing_full_fidelity_requirements"]:
        lines.append(f"- {item}")
    lines.append("")
    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")


def run_benchmark(*, write: bool = True) -> dict[str, Any]:
    """Run FreeGS public-example reconstruction attempts and write reports."""
    freegs, version, import_error = _import_freegs()
    cases: list[dict[str, Any]] = []
    if freegs is not None:
        for spec in PUBLIC_CASES:
            cases.append(_case_record(freegs, spec))
    backend_available = freegs is not None
    missing_requirements = [
        "native same-case free-boundary profile-source reconstruction against finite external FreeGS psi output",
        "native-vs-FreeGS psi_N RMSE threshold",
        "axis/X-point/boundary containment and q-profile thresholds",
        "grid convergence across public example resolutions",
    ]
    artifact = {
        "schema": "freegs-public-example-reconstruction-attempt.v1",
        "accepted_full_fidelity": False,
        "cases": cases,
        "freegs_backend_available": backend_available,
        "freegs_import_error": import_error,
        "freegs_version": version,
        "surface": "free_boundary_equilibrium",
    }
    metadata = {
        "accepted_full_fidelity": False,
        "artifact_id": "freegs_public_example_reconstruction_attempt",
        "artifact_path": _rel(ARTIFACT_PATH),
        "artifact_role": "partial_public_freegs_reconstruction_attempt",
        "available_observables": [
            "public_example_source_checksums",
            "FreeGS_machine_coil_currents_after_constraints",
            "native_vs_FreeGS_vacuum_Green_function_residuals",
            "external_FreeGS_nonlinear_psi_shape_and_range",
            "external_nonlinear_backend_solve_status",
        ],
        "metadata_schema": "full-fidelity-public-output-artifact-metadata.v1",
        "missing_required_observables": missing_requirements,
        "redistribution_license": "FreeGS LGPL-3.0-or-later",
        "reference_family": "FreeGS",
        "sha256": "",
        "solver_output_comparison_ready": False,
        "solver_output_comparison_status": _blocked_status(cases, backend_available),
        "surface": "free_boundary_equilibrium",
    }
    if write:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        metadata["sha256"] = _sha256(ARTIFACT_PATH) or ""
        METADATA_PATH.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    elif ARTIFACT_PATH.exists():
        metadata["sha256"] = _sha256(ARTIFACT_PATH) or ""

    vacuum_pass = bool(cases) and all(
        case["vacuum_green_function_comparison"]["pass"] for case in cases
    )
    external_ready = bool(cases) and all(case["external_nonlinear_output_ready"] for case in cases)
    report = {
        "schema": "freegs-public-example-reconstruction-report.v1",
        "accepted_full_fidelity_ready": False,
        "artifact_path": _rel(ARTIFACT_PATH),
        "case_count": len(cases),
        "cases": cases,
        "freegs_backend_available": backend_available,
        "freegs_import_error": import_error,
        "freegs_version": version,
        "external_nonlinear_output_ready": external_ready,
        "metadata_path": _rel(METADATA_PATH),
        "missing_full_fidelity_requirements": missing_requirements,
        "reference_output_ready": False,
        "sha256": metadata["sha256"],
        "status": metadata["solver_output_comparison_status"],
        "vacuum_comparison_pass": vacuum_pass,
    }
    if write:
        JSON_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        _write_markdown(report)
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the public FreeGS reconstruction benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run without writing reports.")
    args = parser.parse_args(argv)
    report = run_benchmark(write=not args.check)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
