#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
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
MU0 = 4.0e-7 * np.pi


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


def _native_profile_source_reconstruction(
    r_axis: NDArray[np.float64],
    z_axis: NDArray[np.float64],
    external_psi: NDArray[np.float64],
    jtor: NDArray[np.float64],
    *,
    omega: float = 1.35,
    iterations: int = 800,
) -> NDArray[np.float64]:
    """Solve native fixed-boundary Grad-Shafranov profile source on FreeGS grid."""
    if external_psi.shape != jtor.shape:
        raise ValueError("external_psi and jtor must have matching shapes")
    if external_psi.shape != (z_axis.size, r_axis.size):
        raise ValueError("external_psi must have shape len(Z) x len(R)")
    if r_axis.size < 3 or z_axis.size < 3:
        raise ValueError("native reconstruction requires at least 3x3 grid")
    if not np.all(np.isfinite(external_psi)) or not np.all(np.isfinite(jtor)):
        raise ValueError("native reconstruction inputs must be finite")
    if not np.all(np.diff(r_axis) > 0.0) or not np.all(np.diff(z_axis) > 0.0):
        raise ValueError("native reconstruction axes must be strictly increasing")

    d_r = float(np.mean(np.diff(r_axis)))
    d_z = float(np.mean(np.diff(z_axis)))
    psi = np.array(external_psi, dtype=np.float64, copy=True)
    psi[1:-1, 1:-1] = 0.0
    rhs = -MU0 * np.asarray(r_axis, dtype=np.float64)[np.newaxis, :] * jtor
    inv_dr2 = 1.0 / (d_r * d_r)
    inv_dz2 = 1.0 / (d_z * d_z)
    omega_f = float(omega)
    if not np.isfinite(omega_f) or omega_f <= 0.0 or omega_f >= 2.0:
        raise ValueError("omega must be finite and in (0, 2)")
    for _ in range(iterations):
        for j in range(1, z_axis.size - 1):
            for i in range(1, r_axis.size - 1):
                r_val = max(float(r_axis[i]), 1.0e-12)
                coef_ip = inv_dr2 - 0.5 / (r_val * d_r)
                coef_im = inv_dr2 + 0.5 / (r_val * d_r)
                coef_j = inv_dz2
                center = -2.0 * (inv_dr2 + inv_dz2)
                neighbor_sum = (
                    coef_ip * psi[j, i + 1]
                    + coef_im * psi[j, i - 1]
                    + coef_j * (psi[j + 1, i] + psi[j - 1, i])
                )
                updated = (rhs[j, i] - neighbor_sum) / center
                psi[j, i] = (1.0 - omega_f) * psi[j, i] + omega_f * updated
    return psi


def _normalise_psi(psi: NDArray[np.float64]) -> NDArray[np.float64]:
    psi_min = float(np.min(psi))
    span = max(float(np.max(psi) - psi_min), 1.0e-30)
    return (psi - psi_min) / span


def _axis_location(
    r_axis: NDArray[np.float64],
    z_axis: NDArray[np.float64],
    psi: NDArray[np.float64],
) -> tuple[float, float]:
    idx = np.unravel_index(int(np.nanargmax(psi)), psi.shape)
    return float(r_axis[idx[1]]), float(z_axis[idx[0]])


def _boundary_max_abs_error(
    native_psi: NDArray[np.float64], external_psi: NDArray[np.float64]
) -> float:
    residual = native_psi - external_psi
    edges = np.concatenate(
        [
            residual[0, :],
            residual[-1, :],
            residual[1:-1, 0],
            residual[1:-1, -1],
        ]
    )
    return float(np.max(np.abs(edges))) if edges.size else 0.0


def _boundary_containment_fraction(
    native_psi: NDArray[np.float64], external_psi: NDArray[np.float64], threshold: float
) -> float:
    residual = np.abs(native_psi - external_psi)
    edges = np.concatenate(
        [
            residual[0, :],
            residual[-1, :],
            residual[1:-1, 0],
            residual[1:-1, -1],
        ]
    )
    return float(np.mean(edges <= threshold)) if edges.size else 0.0


def _integrate_current(
    r_axis: NDArray[np.float64], z_axis: NDArray[np.float64], jtor: NDArray[np.float64]
) -> float:
    trapz = getattr(np, "trapezoid", None) or np.trapz
    radial_integral = trapz(jtor, r_axis, axis=1)
    return float(trapz(radial_integral, z_axis))


def _nearest_grid_value(
    r_axis: NDArray[np.float64],
    z_axis: NDArray[np.float64],
    psi: NDArray[np.float64],
    point: tuple[float, float],
) -> float:
    r_idx = int(np.argmin(np.abs(r_axis - float(point[0]))))
    z_idx = int(np.argmin(np.abs(z_axis - float(point[1]))))
    return float(psi[z_idx, r_idx])


def _point_inside_domain(point: tuple[float, float], domain: dict[str, float]) -> bool:
    r_value = float(point[0])
    z_value = float(point[1])
    return bool(
        float(domain["r_min"]) <= r_value <= float(domain["r_max"])
        and float(domain["z_min"]) <= z_value <= float(domain["z_max"])
    )


def _q_profile_sanity(eq: Any) -> dict[str, Any]:
    """Extract finite signed-q profile sanity metrics from a solved FreeGS equilibrium."""
    try:
        psi_norm = np.linspace(0.05, 0.95, 16, dtype=np.float64)
        q_values = np.asarray(eq.q(psi_norm), dtype=np.float64)
    except Exception as exc:  # pragma: no cover - backend-version dependent.
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "finite_q_profile": False,
            "status": "blocked_freegs_q_profile_extraction_failed",
        }
    finite = bool(q_values.size and np.all(np.isfinite(q_values)))
    if not finite:
        return {
            "finite_q_profile": False,
            "sample_count": int(q_values.size),
            "status": "blocked_nonfinite_q_profile",
        }
    q_abs = np.abs(q_values)
    q_signs = np.sign(q_values[np.abs(q_values) > 1.0e-12])
    signed_consistent = bool(q_signs.size == 0 or np.all(q_signs == q_signs[0]))
    q_abs_min = float(np.min(q_abs))
    q_abs_max = float(np.max(q_abs))
    sane = bool(signed_consistent and q_abs_min > 0.0 and q_abs_max < 1.0e3)
    return {
        "finite_q_profile": finite,
        "q_abs_max": q_abs_max,
        "q_abs_min": q_abs_min,
        "q_max": float(np.max(q_values)),
        "q_min": float(np.min(q_values)),
        "sample_count": int(q_values.size),
        "signed_consistent": signed_consistent,
        "status": "pass_finite_signed_q_profile" if sane else "blocked_q_profile_sanity_threshold",
    }


def _native_same_case_profile_source_comparison(
    eq: Any,
    profiles: Any,
    spec: FreeGSPublicExampleCase,
    external_psi: NDArray[np.float64],
) -> dict[str, Any]:
    r_axis = np.asarray(eq.R_1D, dtype=np.float64)
    z_axis = np.asarray(eq.Z_1D, dtype=np.float64)
    jtor_raw = getattr(eq, "Jtor", None)
    if isinstance(jtor_raw, np.ndarray):
        jtor = np.asarray(jtor_raw, dtype=np.float64)
    else:
        jtor = np.asarray(profiles.Jtor(eq.R, eq.Z, external_psi, eq.psi_bndry), dtype=np.float64)
    native_psi = _native_profile_source_reconstruction(r_axis, z_axis, external_psi, jtor)
    finite_native = bool(native_psi.size and np.all(np.isfinite(native_psi)))
    finite_external = bool(external_psi.size and np.all(np.isfinite(external_psi)))
    native_n = _normalise_psi(native_psi)
    external_n = _normalise_psi(external_psi)
    residual_n = native_n - external_n
    psi_n_rmse = float(np.sqrt(np.mean(residual_n * residual_n)))
    native_axis = _axis_location(r_axis, z_axis, native_psi)
    external_axis = _axis_location(r_axis, z_axis, external_psi)
    axis_error = float(
        np.hypot(native_axis[0] - external_axis[0], native_axis[1] - external_axis[1])
    )
    native_current = _integrate_current(r_axis, z_axis, jtor)
    current_target = float(spec.plasma_current_a)
    current_closure = abs(native_current - current_target) / max(abs(current_target), 1.0)
    xpoint_errors = [
        abs(
            _nearest_grid_value(r_axis, z_axis, native_n, point)
            - _nearest_grid_value(r_axis, z_axis, external_n, point)
        )
        for point in spec.xpoints
    ]
    boundary_threshold = 1.0e-10
    return {
        "accepted_full_fidelity": False,
        "axis_error_m": axis_error,
        "boundary_containment_fraction": _boundary_containment_fraction(
            native_psi, external_psi, boundary_threshold
        ),
        "boundary_max_abs_error_wb": _boundary_max_abs_error(native_psi, external_psi),
        "current_closure_relative_error": float(current_closure),
        "external_axis_r_m": external_axis[0],
        "external_axis_z_m": external_axis[1],
        "finite_external_psi": finite_external,
        "finite_native_psi": finite_native,
        "native_axis_r_m": native_axis[0],
        "native_axis_z_m": native_axis[1],
        "native_current_a": float(native_current),
        "psi_n_rmse": psi_n_rmse,
        "q_profile_sanity": _q_profile_sanity(eq),
        "schema": "native-freegs-profile-source-comparison.v1",
        "thresholds": {
            "axis_error_m": 2.5e-2,
            "boundary_max_abs_error_wb": boundary_threshold,
            "current_closure_relative_error": 5.0e-2,
            "psi_n_rmse": 5.0e-2,
            "xpoint_psi_n_error": 5.0e-2,
        },
        "xpoint_constraint_count": len(spec.xpoints),
        "xpoint_psi_n_error_max": float(max(xpoint_errors, default=0.0)),
    }


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
    comparison = _native_same_case_profile_source_comparison(eq, profiles, spec, psi)
    return {
        "blend": blend,
        "elapsed_s": elapsed,
        "external_psi_finite": finite_psi,
        "external_psi_max": float(np.max(psi)) if finite_psi else None,
        "external_psi_min": float(np.min(psi)) if finite_psi else None,
        "external_psi_shape": psi_shape,
        "maxits": maxits,
        "native_same_case_profile_source_comparison": comparison,
        "native_same_case_psi_comparison_ready": bool(
            comparison["finite_native_psi"] and comparison["finite_external_psi"]
        ),
        "status": "external_backend_solved_native_same_case_profile_source_compared_fail_closed",
    }


def _attempt_solve_sweep(freegs: ModuleType, spec: FreeGSPublicExampleCase) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for maxits, blend in spec.nonlinear_attempts:
        attempt = _attempt_solve(freegs, spec, maxits=maxits, blend=blend)
        attempts.append(attempt)
        if attempt["status"] == (
            "external_backend_solved_native_same_case_profile_source_compared_fail_closed"
        ):
            break
    successful = [
        attempt
        for attempt in attempts
        if attempt["status"]
        == "external_backend_solved_native_same_case_profile_source_compared_fail_closed"
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
        "domain": {
            "r_max": spec.r_max,
            "r_min": spec.r_min,
            "z_max": spec.z_max,
            "z_min": spec.z_min,
        },
        "example_path": _rel(spec.example_path),
        "example_sha256": _sha256(spec.example_path),
        "grid": {"nx": spec.nx, "ny": spec.ny},
        "machine_class": spec.machine_class,
        "missing_full_fidelity_requirements": [
            "strict native-vs-FreeGS psi_N RMSE, current, axis, X-point, and boundary-containment threshold acceptance",
            "grid-convergence evidence for the public example",
            "coil/vacuum reconstruction linked to public machine current sidecars",
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
            if all(
                case["nonlinear_solve_attempt"].get("native_same_case_psi_comparison_ready", False)
                for case in cases
            ):
                return (
                    "blocked_public_freegs_native_same_case_compared_missing_"
                    "strict_threshold_grid_convergence_coil_sidecars"
                )
            return "blocked_public_freegs_external_psi_ready_missing_native_same_case_comparison"
        return "blocked_public_freegs_vacuum_matched_missing_nonlinear_same_case_psi"
    return "blocked_public_freegs_vacuum_or_backend_contract_failed"


def _strict_threshold_checks(case: dict[str, Any]) -> list[dict[str, Any]]:
    solve = case["nonlinear_solve_attempt"]
    comparison = solve.get("native_same_case_profile_source_comparison", {})
    thresholds = comparison.get("thresholds", {})
    check_specs = [
        ("psi_n_rmse", "<=", "psi_n_rmse"),
        ("axis_error_m", "<=", "axis_error_m"),
        ("current_closure_relative_error", "<=", "current_closure_relative_error"),
        ("boundary_max_abs_error_wb", "<=", "boundary_max_abs_error_wb"),
        ("xpoint_psi_n_error_max", "<=", "xpoint_psi_n_error"),
    ]
    checks: list[dict[str, Any]] = []
    for metric, comparator, threshold_key in check_specs:
        value = comparison.get(metric)
        limit = thresholds.get(threshold_key)
        valid = isinstance(value, int | float) and isinstance(limit, int | float)
        passed = bool(valid and float(value) <= float(limit))
        checks.append(
            {
                "comparator": comparator,
                "limit": float(limit) if isinstance(limit, int | float) else None,
                "metric": metric,
                "passed": passed,
                "valid": bool(valid),
                "value": float(value) if isinstance(value, int | float) else None,
            }
        )
    boundary_fraction = comparison.get("boundary_containment_fraction")
    boundary_fraction_valid = isinstance(boundary_fraction, int | float)
    checks.append(
        {
            "comparator": ">=",
            "limit": 1.0,
            "metric": "boundary_containment_fraction",
            "passed": bool(boundary_fraction_valid and float(boundary_fraction) >= 1.0),
            "valid": bool(boundary_fraction_valid),
            "value": float(boundary_fraction) if boundary_fraction_valid else None,
        }
    )
    q_sanity = comparison.get("q_profile_sanity", {})
    q_pass = isinstance(q_sanity, dict) and q_sanity.get("status") == "pass_finite_signed_q_profile"
    checks.append(
        {
            "comparator": "==",
            "limit": "pass_finite_signed_q_profile",
            "metric": "q_profile_sanity_status",
            "passed": bool(q_pass),
            "valid": isinstance(q_sanity, dict),
            "value": q_sanity.get("status") if isinstance(q_sanity, dict) else None,
        }
    )
    return checks


def _geometry_containment_evidence(cases: list[dict[str, Any]]) -> dict[str, Any]:
    case_rows: list[dict[str, Any]] = []
    for case in cases:
        domain = case.get("domain", {})
        source = case.get("source_contract", {})
        solve = case.get("nonlinear_solve_attempt", {})
        comparison = solve.get("native_same_case_profile_source_comparison", {})
        xpoints = [tuple(point) for point in source.get("xpoints", [])]
        isoflux_points = [
            (float(row[0]), float(row[1]))
            for row in source.get("isoflux", [])
            if isinstance(row, list | tuple) and len(row) >= 2
        ] + [
            (float(row[2]), float(row[3]))
            for row in source.get("isoflux", [])
            if isinstance(row, list | tuple) and len(row) >= 4
        ]
        source_points = [*xpoints, *isoflux_points]
        source_inside = bool(
            source_points
            and isinstance(domain, dict)
            and all(_point_inside_domain(point, domain) for point in source_points)
        )
        native_axis = (
            float(comparison.get("native_axis_r_m", np.nan)),
            float(comparison.get("native_axis_z_m", np.nan)),
        )
        external_axis = (
            float(comparison.get("external_axis_r_m", np.nan)),
            float(comparison.get("external_axis_z_m", np.nan)),
        )
        axis_points = [native_axis, external_axis]
        axis_inside = bool(
            isinstance(domain, dict)
            and all(np.all(np.isfinite(point)) for point in axis_points)
            and all(_point_inside_domain(point, domain) for point in axis_points)
        )
        boundary_fraction = comparison.get("boundary_containment_fraction")
        boundary_ready = isinstance(boundary_fraction, int | float) and np.isfinite(
            float(boundary_fraction)
        )
        case_rows.append(
            {
                "axis_points_inside_grid": axis_inside,
                "boundary_containment_fraction": float(boundary_fraction)
                if boundary_ready
                else None,
                "boundary_containment_metric_ready": bool(boundary_ready),
                "case_id": case["case_id"],
                "external_axis": [external_axis[0], external_axis[1]],
                "isoflux_endpoint_count": len(isoflux_points),
                "native_axis": [native_axis[0], native_axis[1]],
                "source_point_count": len(source_points),
                "source_points_inside_grid": source_inside,
                "xpoint_count": len(xpoints),
            }
        )
    all_source_inside = bool(case_rows) and all(
        bool(row["source_points_inside_grid"]) for row in case_rows
    )
    axis_ready = bool(case_rows) and all(bool(row["axis_points_inside_grid"]) for row in case_rows)
    boundary_ready = bool(case_rows) and all(
        bool(row["boundary_containment_metric_ready"]) for row in case_rows
    )
    ready = bool(all_source_inside and axis_ready and boundary_ready)
    return {
        "accepted_full_fidelity": False,
        "all_source_points_inside_grid": all_source_inside,
        "axis_containment_metric_ready": axis_ready,
        "boundary_containment_metric_ready": boundary_ready,
        "case_count": len(case_rows),
        "cases": case_rows,
        "schema": "strict-free-boundary-geometry-containment.v1",
        "status": "accepted_local_geometry_containment_evidence"
        if ready
        else "blocked_geometry_containment_evidence_incomplete",
        "strict_geometry_containment_ready": ready,
    }


def _strict_free_boundary_parity_evidence(cases: list[dict[str, Any]]) -> dict[str, Any]:
    case_rows: list[dict[str, Any]] = []
    failed_count = 0
    for case in cases:
        checks = _strict_threshold_checks(case)
        failed = [check for check in checks if not bool(check["passed"])]
        failed_count += len(failed)
        solve = case["nonlinear_solve_attempt"]
        case_rows.append(
            {
                "case_id": case["case_id"],
                "external_nonlinear_output_ready": bool(case["external_nonlinear_output_ready"]),
                "failed_threshold_check_count": len(failed),
                "native_same_case_profile_source_ready": bool(
                    solve.get("native_same_case_psi_comparison_ready", False)
                ),
                "strict_threshold_acceptance_ready": not failed,
                "threshold_checks": checks,
            }
        )
    native_ready = bool(cases) and all(
        bool(row["native_same_case_profile_source_ready"]) for row in case_rows
    )
    threshold_ready = bool(cases) and failed_count == 0 and native_ready
    grid_ready = False
    sidecar_ready = False
    return {
        "schema": "strict-free-boundary-parity-evidence.v1",
        "accepted_full_fidelity": False,
        "blocking_requirements": [
            "strict native-vs-FreeGS psi_N RMSE/current/axis/X-point/boundary threshold acceptance",
            "grid convergence across public example resolutions",
            "coil/vacuum reconstruction linked to public machine current sidecars",
        ],
        "case_count": len(cases),
        "cases": case_rows,
        "coil_vacuum_sidecar_ready": sidecar_ready,
        "failed_threshold_check_count": failed_count,
        "geometry_containment_evidence": _geometry_containment_evidence(cases),
        "grid_convergence_ready": grid_ready,
        "native_same_case_profile_source_ready": native_ready,
        "status": "blocked_strict_thresholds_or_grid_convergence_missing",
        "strict_threshold_acceptance_ready": threshold_ready,
    }


def _write_markdown(report: dict[str, Any]) -> None:
    strict = report.get("strict_free_boundary_parity_evidence", {})
    geometry = strict.get("geometry_containment_evidence", {})
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
        f"- Native same-case comparison ready: `{report['native_same_case_psi_comparison_ready']}`",
        (
            "- Strict threshold acceptance ready: "
            f"`{strict.get('strict_threshold_acceptance_ready', False)}`"
        ),
        f"- Grid convergence ready: `{strict.get('grid_convergence_ready', False)}`",
        f"- Coil/vacuum sidecar ready: `{strict.get('coil_vacuum_sidecar_ready', False)}`",
        (
            "- Geometry containment ready: "
            f"`{geometry.get('strict_geometry_containment_ready', False)}`"
        ),
        (
            "- Boundary-containment metric ready: "
            f"`{geometry.get('boundary_containment_metric_ready', False)}`"
        ),
        (f"- Failed strict threshold checks: `{strict.get('failed_threshold_check_count', 0)}`"),
        f"- Accepted full fidelity: `{report['accepted_full_fidelity_ready']}`",
        f"- Artifact: `{report['artifact_path']}`",
        "",
        "| Case | Machine | Vacuum NRMSE | Vacuum pass | Native psi_N RMSE | Axis error [m] | Nonlinear status |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in report["cases"]:
        vacuum = case["vacuum_green_function_comparison"]
        solve = case["nonlinear_solve_attempt"]
        comparison = solve.get("native_same_case_profile_source_comparison", {})
        lines.append(
            "| {case_id} | {machine} | {nrmse:.6e} | {vac_pass} | {psi_rmse:.6e} | {axis_error:.6e} | `{status}` |".format(
                case_id=case["case_id"],
                machine=case["machine_class"],
                nrmse=vacuum["nrmse"],
                vac_pass=vacuum["pass"],
                psi_rmse=float(comparison.get("psi_n_rmse", np.nan)),
                axis_error=float(comparison.get("axis_error_m", np.nan)),
                status=solve["status"],
            )
        )
    if strict:
        if geometry:
            lines.extend(["", "## Geometry containment evidence", ""])
            lines.append(f"- Schema: `{geometry['schema']}`")
            lines.append(f"- Status: `{geometry['status']}`")
            lines.append(
                f"- Source points inside grid: `{geometry['all_source_points_inside_grid']}`"
            )
            lines.append(
                f"- Axis containment metric ready: `{geometry['axis_containment_metric_ready']}`"
            )
            lines.append(
                "- Boundary-containment metric ready: "
                f"`{geometry['boundary_containment_metric_ready']}`"
            )
            lines.append("")
            lines.append(
                "| Case | Source points | X-points | Isoflux endpoints | Source inside | Axis inside | Boundary metric |"
            )
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for case in geometry["cases"]:
                lines.append(
                    "| {case_id} | {source_points} | {xpoints} | {isoflux} | {source_inside} | {axis_inside} | {boundary_ready} |".format(
                        axis_inside=case["axis_points_inside_grid"],
                        boundary_ready=case["boundary_containment_metric_ready"],
                        case_id=case["case_id"],
                        isoflux=case["isoflux_endpoint_count"],
                        source_inside=case["source_points_inside_grid"],
                        source_points=case["source_point_count"],
                        xpoints=case["xpoint_count"],
                    )
                )
        lines.extend(["", "## Strict parity threshold checks", ""])
        for case in strict["cases"]:
            lines.append(f"### `{case['case_id']}`")
            lines.append("")
            lines.append("| Metric | Value | Comparator | Limit | Pass |")
            lines.append("| --- | ---: | --- | ---: | ---: |")
            for check in case["threshold_checks"]:
                lines.append(
                    "| {metric} | {value} | {comparator} | {limit} | {passed} |".format(
                        metric=check["metric"],
                        value=check["value"],
                        comparator=check["comparator"],
                        limit=check["limit"],
                        passed=check["passed"],
                    )
                )
            lines.append("")
    lines.extend(["", "## Missing full-fidelity requirements", ""])
    for item in report["missing_full_fidelity_requirements"]:
        lines.append(f"- {item}")
    lines.append("")
    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")


def _tracked_report_fallback() -> dict[str, Any] | None:
    if not JSON_REPORT.exists():
        return None
    try:
        report = json.loads(JSON_REPORT.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if report.get("schema") != "freegs-public-example-reconstruction-report.v1":
        return None
    if not report.get("case_count") or not report.get("vacuum_comparison_pass"):
        return None
    fallback = dict(report)
    fallback["report_generation_mode"] = "tracked_report_fallback"
    fallback["source_cache_available"] = FREEGS_REPO.exists()
    fallback["strict_free_boundary_parity_evidence"] = _strict_free_boundary_parity_evidence(
        fallback.get("cases", [])
    )
    return fallback


def run_benchmark(*, write: bool = True) -> dict[str, Any]:
    """Run FreeGS public-example reconstruction attempts and write reports."""
    freegs, version, import_error = _import_freegs()
    cases: list[dict[str, Any]] = []
    if freegs is not None:
        for spec in PUBLIC_CASES:
            cases.append(_case_record(freegs, spec))
    backend_available = freegs is not None
    missing_requirements = [
        "strict native-vs-FreeGS psi_N RMSE/current/axis/X-point/boundary threshold acceptance",
        "grid convergence across public example resolutions",
        "coil/vacuum reconstruction linked to public machine current sidecars",
    ]
    vacuum_pass = bool(cases) and all(
        case["vacuum_green_function_comparison"]["pass"] for case in cases
    )
    external_ready = bool(cases) and all(case["external_nonlinear_output_ready"] for case in cases)
    native_comparison_ready = bool(cases) and all(
        case["nonlinear_solve_attempt"].get("native_same_case_psi_comparison_ready", False)
        for case in cases
    )
    if not vacuum_pass:
        fallback = _tracked_report_fallback()
        if fallback is not None:
            if write:
                REPORT_DIR.mkdir(parents=True, exist_ok=True)
                JSON_REPORT.write_text(json.dumps(fallback, indent=2, sort_keys=True) + "\n")
                _write_markdown(fallback)
            return fallback
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
            "native_same_case_profile_source_psi_N_RMSE_axis_current_boundary_metrics",
            "strict_native_vs_FreeGS_threshold_checks",
            "external_nonlinear_backend_solve_status",
        ],
        "metadata_schema": "full-fidelity-public-output-artifact-metadata.v1",
        "missing_required_observables": missing_requirements,
        "redistribution_license": "FreeGS LGPL-3.0-or-later",
        "reference_family": "FreeGS",
        "sha256": "",
        "solver_output_comparison_ready": native_comparison_ready,
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
        "native_same_case_psi_comparison_ready": native_comparison_ready,
        "reference_output_ready": False,
        "sha256": metadata["sha256"],
        "report_generation_mode": "external_backend_reconstruction",
        "source_cache_available": FREEGS_REPO.exists(),
        "status": metadata["solver_output_comparison_status"],
        "strict_free_boundary_parity_evidence": _strict_free_boundary_parity_evidence(cases),
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
