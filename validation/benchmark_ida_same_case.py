#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — IDA free-boundary same-case evidence
"""Measure the IDA JAX free-boundary solver against public FreeGS cases.

The benchmark is deliberately fail closed.  It binds exact public-example,
solver, profile-basis, input, reference-output, and candidate-output digests;
separates a development case from a DIII-D-like evaluation candidate; measures
field residuals, current closure, nonlinear residual, compact-profile and coil
gradient audits, and synchronized warm latency; and keeps every facility/control
claim false.  A result can become bounded same-case evidence only when an
execution-preceding selection lock and every predeclared threshold are present.
Warm latency starts from an explicitly polished converged equilibrium under the
same inputs; cold continuation, warm compilation, and warm setup remain separate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.jax_free_boundary_gs import MU0_SI
from scpn_fusion.core.jax_free_boundary_predictive import (
    DEFAULT_ANDERSON_DEPTH,
    DEFAULT_CUTOFF_WIDTH,
    DEFAULT_IP_RAMP,
    DEFAULT_MIXING,
    DEFAULT_N_ITER,
    DEFAULT_TOL,
    _plasma_current,
    build_response_matrix,
    predictive_gs_residual,
    solve_predictive_equilibrium_diff,
)
from scpn_fusion.core.jax_o_point import smooth_axis_flux
from scpn_fusion.core.jax_predictive_forward_compiled import (
    solve_predictive_equilibrium_compiled,
)
from scpn_fusion.core.jax_profile_basis import (
    bspline_design_matrix,
    evaluate_profile,
)
from scpn_fusion.core.jax_x_point import smooth_xpoint_flux
from validation.benchmark_freegs_public_example_reconstruction import (
    PUBLIC_CASES,
    FreeGSPublicExampleCase,
    _import_freegs,
    _machine_filaments,
    _make_equilibrium,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "validation" / "reports" / "ida_same_case_evidence.json"
MARKDOWN_PATH = ROOT / "validation" / "reports" / "ida_same_case_evidence.md"
SCHEMA_VERSION = "scpn-fusion.ida-same-case-evidence.v2"
BENCHMARK_ID = "DIII-D-IDA-FB-JAX-B"
SOLVER_ID = "scpn_fusion.core.jax_free_boundary_predictive.solve_predictive_equilibrium_diff"
CLAIM_FIELDS = (
    "control_admission",
    "facility_validation",
    "pcs_deployment",
    "scientific_validation",
    "safety_admission",
)
THRESHOLDS: dict[str, float] = {
    "coil_gradient_relative_error_max": 5.0e-2,
    "gradient_smoothness_ratio_max": 2.5e-1,
    "latency_p95_ms_max": 20.0,
    "profile_gradient_relative_error_max": 1.0e-2,
    "psi_n_rmse_max": 5.0e-2,
    "relative_current_error_max": 5.0e-2,
    "relative_nonlinear_residual_rms_max": 5.0e-2,
}
WARM_START_ITERATION_CAP = 20
WARM_START_IP_RAMP = 1
WARM_START_MEASUREMENT_MODE = "same_input_from_converged_equilibrium"
_SHA256_LENGTH = 64
_CASE_ROLES = ("development", "evaluation_candidate")
_SOURCE_PATHS = {
    "benchmark": "validation/benchmark_ida_same_case.py",
    "freegs_public_case_runner": ("validation/benchmark_freegs_public_example_reconstruction.py"),
    "profile_basis": "src/scpn_fusion/core/jax_profile_basis.py",
    "solver": "src/scpn_fusion/core/jax_free_boundary_predictive.py",
    "compiled_forward": "src/scpn_fusion/core/jax_predictive_forward_compiled.py",
    "o_point": "src/scpn_fusion/core/jax_o_point.py",
    "x_point": "src/scpn_fusion/core/jax_x_point.py",
}
_ENVIRONMENT_FIELDS = {
    "affinity_cpu_count",
    "backend",
    "devices",
    "freegs_version",
    "host_load_1m_5m_15m",
    "isolated_host",
    "jax_version",
    "jaxlib_version",
    "machine",
    "platform",
    "python_version",
    "x64_enabled",
}
_REQUIRED_TOP_LEVEL = {
    "benchmark_id",
    "blockers",
    "case_role_contract",
    "cases",
    "claim_boundary",
    "environment",
    "generated_at",
    "payload_sha256",
    "schema_version",
    "solver_contract",
    "source_artifacts",
    "status",
    "thresholds",
}

FloatArray: TypeAlias = NDArray[np.float64]


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _payload_sha256(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned["payload_sha256"] = ""
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: object) -> str:
    array = np.asarray(value, dtype="<f8", order="C")
    descriptor = _canonical_json({"dtype": "<f8", "shape": [int(size) for size in array.shape]})
    return hashlib.sha256(descriptor + b"\0" + array.tobytes(order="C")).hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_report(path: str | Path) -> dict[str, Any]:
    """Load one report while rejecting duplicate keys and non-object roots."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
    if not isinstance(payload, dict):
        raise ValueError("report root must be an object")
    return payload


def _git_value(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _source_artifacts() -> dict[str, dict[str, str]]:
    return {
        name: {
            "path": relative_path,
            "sha256": _file_sha256(ROOT / relative_path),
        }
        for name, relative_path in sorted(_SOURCE_PATHS.items())
    }


def _runtime_environment() -> dict[str, Any]:
    load = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    devices = [str(device) for device in jax.devices()]
    return {
        "affinity_cpu_count": (
            len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        ),
        "backend": jax.default_backend(),
        "devices": devices,
        "freegs_version": importlib.metadata.version("freegs"),
        "host_load_1m_5m_15m": load,
        "isolated_host": False,
        "jax_version": importlib.metadata.version("jax"),
        "jaxlib_version": importlib.metadata.version("jaxlib"),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "x64_enabled": cast(bool, jax.config.values["jax_enable_x64"]),
    }


def _reference_mask_boundary(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    interior = np.zeros_like(mask)
    interior[1:-1, 1:-1] = mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
    return mask & ~interior


def _axis_value(
    psi: FloatArray,
    mask: NDArray[np.bool_],
    boundary_value: float,
) -> float:
    values = psi[mask]
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    return minimum if abs(minimum - boundary_value) > abs(maximum - boundary_value) else maximum


def _fit_compact_profile(
    profile_values: FloatArray,
    psin_knots: FloatArray,
    *,
    n_coefficients: int,
    degree: int,
) -> tuple[FloatArray, FloatArray]:
    design = bspline_design_matrix(psin_knots, n_coeff=n_coefficients, degree=degree)
    coefficients = np.linalg.lstsq(design, profile_values, rcond=None)[0]
    reconstructed = np.asarray(design @ coefficients, dtype=np.float64)
    return (
        np.asarray(coefficients, dtype=np.float64),
        reconstructed,
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("latency values must not be empty")
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _finite_difference_row(
    *,
    name: str,
    index: int,
    vector: FloatArray,
    autodiff: FloatArray,
    objective: Callable[[FloatArray], float],
    relative_step: float,
    relative_error_limit: float,
) -> dict[str, Any]:
    epsilon = relative_step * max(abs(float(vector[index])), 1.0)
    plus = vector.copy()
    minus = vector.copy()
    plus[index] += epsilon
    minus[index] -= epsilon
    centre_value = objective(vector)
    plus_value = objective(plus)
    minus_value = objective(minus)
    finite_difference = (plus_value - minus_value) / (2.0 * epsilon)
    autodiff_value = float(autodiff[index])
    scale = max(abs(finite_difference), abs(autodiff_value), 1.0e-14)
    relative_error = abs(autodiff_value - finite_difference) / scale
    first_difference_scale = max(
        abs(plus_value - centre_value), abs(centre_value - minus_value), 1.0e-14
    )
    smoothness_ratio = abs(plus_value - 2.0 * centre_value + minus_value) / (first_difference_scale)
    return {
        "autodiff": autodiff_value,
        "epsilon": epsilon,
        "finite_difference": finite_difference,
        "index": index,
        "input": name,
        "passed": bool(
            relative_error <= relative_error_limit
            and smoothness_ratio <= THRESHOLDS["gradient_smoothness_ratio_max"]
        ),
        "relative_error": relative_error,
        "relative_error_limit": relative_error_limit,
        "smoothness_ratio": smoothness_ratio,
        "smoothness_ratio_limit": THRESHOLDS["gradient_smoothness_ratio_max"],
    }


def _execute_case(
    spec: FreeGSPublicExampleCase,
    *,
    role: str,
    grid_points: int,
    n_iter: int,
    latency_repeats: int,
) -> dict[str, Any]:
    freegs, freegs_version, import_error = _import_freegs()
    if freegs is None:
        raise RuntimeError(f"FreeGS backend unavailable: {import_error}")
    run_spec = replace(spec, nx=grid_points, ny=grid_points)
    tokamak, equilibrium, profiles, constrain = _make_equilibrium(freegs, run_spec)
    reference_start = time.perf_counter()
    freegs.solve(
        equilibrium,
        profiles,
        constrain,
        show=False,
        maxits=run_spec.nonlinear_attempts[-1][0],
        blend=run_spec.nonlinear_attempts[-1][1],
        convergenceInfo=False,
    )
    reference_elapsed_ms = (time.perf_counter() - reference_start) * 1000.0

    r_grid = np.asarray(equilibrium.R_1D, dtype=np.float64)
    z_grid = np.asarray(equilibrium.Z_1D, dtype=np.float64)
    reference_psi_rz = np.asarray(equilibrium.psi(), dtype=np.float64)
    reference_psi = np.asarray(reference_psi_rz.T, dtype=np.float64)
    reference_jtor = np.asarray(equilibrium.Jtor, dtype=np.float64).T
    reference_mask = np.isfinite(reference_jtor) & (np.abs(reference_jtor) > 0.0)
    if not np.any(reference_mask):
        raise RuntimeError("FreeGS reference contains no finite plasma-current mask")
    boundary_mask = _reference_mask_boundary(reference_mask)
    if not np.any(boundary_mask):
        raise RuntimeError("FreeGS reference plasma mask has no boundary")
    reference_boundary = float(equilibrium.psi_bndry)
    reference_axis = float(equilibrium.psi_axis)
    reference_span = abs(reference_boundary - reference_axis)
    if not math.isfinite(reference_span) or reference_span <= 0.0:
        raise RuntimeError("FreeGS reference has no finite non-zero flux span")

    _, filaments = _machine_filaments(tokamak)
    coil_r = np.asarray([row[0] for row in filaments], dtype=np.float64)
    coil_z = np.asarray([row[1] for row in filaments], dtype=np.float64)
    coil_current = np.asarray([row[2] for row in filaments], dtype=np.float64)
    psin_knots = np.linspace(0.0, 1.0, 129, dtype=np.float64)
    pprime_exact = np.asarray(profiles.pprime(psin_knots), dtype=np.float64)
    ffprime_exact = np.asarray(profiles.ffprime(psin_knots), dtype=np.float64)
    profile_degree = 3
    profile_coefficients = 12
    pprime_coeff, pprime_values = _fit_compact_profile(
        pprime_exact,
        psin_knots,
        n_coefficients=profile_coefficients,
        degree=profile_degree,
    )
    ffprime_coeff, ffprime_values = _fit_compact_profile(
        ffprime_exact,
        psin_knots,
        n_coefficients=profile_coefficients,
        degree=profile_degree,
    )
    basis = bspline_design_matrix(
        psin_knots,
        n_coeff=profile_coefficients,
        degree=profile_degree,
    )

    r_jax = jnp.asarray(r_grid)
    z_jax = jnp.asarray(z_grid)
    coil_r_jax = jnp.asarray(coil_r)
    coil_z_jax = jnp.asarray(coil_z)
    knots_jax = jnp.asarray(psin_knots)
    basis_jax = jnp.asarray(basis)
    coil_jax = jnp.asarray(coil_current)
    p_coeff_jax = jnp.asarray(pprime_coeff)
    ff_coeff_jax = jnp.asarray(ffprime_coeff)
    response_matrix, wall_indices, source_indices = build_response_matrix(r_jax, z_jax)

    def candidate_forward(
        currents: object,
        p_coefficients: object,
        ff_coefficients: object,
        *,
        psi_init: object | None = None,
        iteration_cap: int = n_iter,
        ip_ramp: int = DEFAULT_IP_RAMP,
        return_iterations: bool = False,
    ) -> object:
        return solve_predictive_equilibrium_compiled(
            cast(jnp.ndarray, currents),
            evaluate_profile(cast(jnp.ndarray, p_coefficients), basis_jax),
            evaluate_profile(cast(jnp.ndarray, ff_coefficients), basis_jax),
            r_jax,
            z_jax,
            coil_r_jax,
            coil_z_jax,
            knots_jax,
            run_spec.plasma_current_a,
            response_matrix,
            wall_indices,
            source_indices,
            None if psi_init is None else cast(jnp.ndarray, psi_init),
            iteration_cap,
            DEFAULT_ANDERSON_DEPTH,
            DEFAULT_MIXING,
            ip_ramp,
            DEFAULT_CUTOFF_WIDTH,
            DEFAULT_TOL,
            MU0_SI,
            return_iterations=return_iterations,
        )

    compile_start = time.perf_counter()
    cold_result = cast(
        tuple[jnp.ndarray, int],
        candidate_forward(
            coil_jax,
            p_coeff_jax,
            ff_coeff_jax,
            return_iterations=True,
        ),
    )
    cold_psi_jax, cold_start_iterations = cold_result
    cold_psi_jax.block_until_ready()
    compile_and_first_ms = (time.perf_counter() - compile_start) * 1000.0

    warm_compile_start = time.perf_counter()
    warm_setup_result = cast(
        tuple[jnp.ndarray, int],
        candidate_forward(
            coil_jax,
            p_coeff_jax,
            ff_coeff_jax,
            psi_init=cold_psi_jax,
            iteration_cap=WARM_START_ITERATION_CAP,
            ip_ramp=WARM_START_IP_RAMP,
            return_iterations=True,
        ),
    )
    candidate_psi_jax, warm_start_setup_iterations = warm_setup_result
    candidate_psi_jax.block_until_ready()
    warm_compile_and_first_ms = (time.perf_counter() - warm_compile_start) * 1000.0

    warm_latency_ms: list[float] = []
    warm_start_iterations: list[int] = []
    for _ in range(latency_repeats):
        start = time.perf_counter()
        value, iteration_count = cast(
            tuple[jnp.ndarray, int],
            candidate_forward(
                coil_jax,
                p_coeff_jax,
                ff_coeff_jax,
                psi_init=candidate_psi_jax,
                iteration_cap=WARM_START_ITERATION_CAP,
                ip_ramp=WARM_START_IP_RAMP,
                return_iterations=True,
            ),
        )
        value.block_until_ready()
        warm_latency_ms.append((time.perf_counter() - start) * 1000.0)
        warm_start_iterations.append(iteration_count)

    candidate_psi = np.asarray(candidate_psi_jax, dtype=np.float64)
    difference = candidate_psi - reference_psi
    candidate_axis = float(smooth_axis_flux(candidate_psi_jax))
    candidate_boundary = float(smooth_xpoint_flux(candidate_psi_jax, r_jax, z_jax))
    candidate_span = candidate_boundary - candidate_axis
    if not math.isfinite(candidate_span) or abs(candidate_span) <= 1.0e-30:
        raise RuntimeError("predictive candidate has no finite non-zero flux span")
    reference_psi_n = (reference_psi - reference_axis) / (reference_boundary - reference_axis)
    candidate_psi_n = (candidate_psi - candidate_axis) / candidate_span
    psi_n_rmse = float(
        np.sqrt(np.mean(np.square((candidate_psi_n - reference_psi_n)[reference_mask])))
    )
    d_r = r_jax[1] - r_jax[0]
    d_z = z_jax[1] - z_jax[0]
    d_area = d_r * d_z
    pprime_jax = evaluate_profile(p_coeff_jax, basis_jax)
    ffprime_jax = evaluate_profile(ff_coeff_jax, basis_jax)
    current_density = _plasma_current(
        candidate_psi_jax,
        r_jax,
        jnp.asarray(candidate_axis),
        jnp.asarray(candidate_boundary),
        knots_jax,
        pprime_jax,
        ffprime_jax,
        jnp.asarray(run_spec.plasma_current_a),
        d_area,
        DEFAULT_CUTOFF_WIDTH,
        MU0_SI,
    )
    candidate_current = float(jnp.sum(current_density) * d_area)
    relative_current_error = abs(candidate_current - run_spec.plasma_current_a) / max(
        abs(run_spec.plasma_current_a), 1.0
    )
    nonlinear_residual = np.asarray(
        predictive_gs_residual(
            candidate_psi_jax,
            coil_jax,
            pprime_jax,
            ffprime_jax,
            r_jax,
            z_jax,
            coil_r_jax,
            coil_z_jax,
            knots_jax,
            jnp.asarray(run_spec.plasma_current_a),
            response_matrix,
            wall_indices,
            source_indices,
            DEFAULT_CUTOFF_WIDTH,
            MU0_SI,
        ),
        dtype=np.float64,
    )
    source = np.asarray(-(MU0_SI * r_jax[jnp.newaxis, :] * current_density))
    interior = np.s_[1:-1, 1:-1]
    residual_rms = float(np.sqrt(np.mean(np.square(nonlinear_residual[interior]))))
    source_rms = float(np.sqrt(np.mean(np.square(source[interior]))))
    relative_residual_rms = residual_rms / max(source_rms, 1.0e-30)

    cotangent_np = np.zeros_like(candidate_psi)
    cotangent_np[reference_mask] = 1.0 / float(np.count_nonzero(reference_mask))

    def candidate_diff(
        currents: object,
        p_coefficients: object,
        ff_coefficients: object,
    ) -> object:
        return solve_predictive_equilibrium_diff(
            cast(jnp.ndarray, currents),
            evaluate_profile(cast(jnp.ndarray, p_coefficients), basis_jax),
            evaluate_profile(cast(jnp.ndarray, ff_coefficients), basis_jax),
            r_jax,
            z_jax,
            coil_r_jax,
            coil_z_jax,
            knots_jax,
            run_spec.plasma_current_a,
            response_matrix,
            wall_indices,
            source_indices,
            jax.lax.stop_gradient(candidate_psi_jax),
            20,
            DEFAULT_ANDERSON_DEPTH,
            DEFAULT_MIXING,
            1,
            DEFAULT_CUTOFF_WIDTH,
            DEFAULT_TOL,
            MU0_SI,
        )

    candidate_value_raw, pullback = jax.vjp(
        candidate_diff,
        coil_jax,
        p_coeff_jax,
        ff_coeff_jax,
    )
    candidate_value = cast(jnp.ndarray, candidate_value_raw)
    candidate_value.block_until_ready()
    coil_gradient, p_gradient, ff_gradient = pullback(jnp.asarray(cotangent_np))
    coil_gradient_np = np.asarray(coil_gradient, dtype=np.float64)
    p_gradient_np = np.asarray(p_gradient, dtype=np.float64)
    ff_gradient_np = np.asarray(ff_gradient, dtype=np.float64)

    def scalar_objective(
        currents: FloatArray,
        p_coefficients: FloatArray,
        ff_coefficients: FloatArray,
    ) -> float:
        value = cast(
            jnp.ndarray,
            candidate_forward(
                jnp.asarray(currents),
                jnp.asarray(p_coefficients),
                jnp.asarray(ff_coefficients),
                psi_init=candidate_psi_jax,
                iteration_cap=20,
                ip_ramp=1,
            ),
        )
        return float(jnp.sum(value * jnp.asarray(cotangent_np)))

    coil_index = int(np.argmax(np.abs(coil_current)))
    p_index = profile_coefficients // 2
    ff_index = profile_coefficients // 2
    gradient_rows = [
        _finite_difference_row(
            name="coil_current_a",
            index=coil_index,
            vector=coil_current,
            autodiff=coil_gradient_np,
            objective=lambda vector: scalar_objective(vector, pprime_coeff, ffprime_coeff),
            relative_step=1.0e-4,
            relative_error_limit=THRESHOLDS["coil_gradient_relative_error_max"],
        ),
        _finite_difference_row(
            name="pprime_coefficients_pa_per_wb",
            index=p_index,
            vector=pprime_coeff,
            autodiff=p_gradient_np,
            objective=lambda vector: scalar_objective(coil_current, vector, ffprime_coeff),
            relative_step=1.0e-4,
            relative_error_limit=THRESHOLDS["profile_gradient_relative_error_max"],
        ),
        _finite_difference_row(
            name="ffprime_coefficients_t2_m2_per_wb",
            index=ff_index,
            vector=ffprime_coeff,
            autodiff=ff_gradient_np,
            objective=lambda vector: scalar_objective(coil_current, pprime_coeff, vector),
            relative_step=1.0e-4,
            relative_error_limit=THRESHOLDS["profile_gradient_relative_error_max"],
        ),
    ]
    threshold_results = {
        "gradient_audit": bool(all(row["passed"] for row in gradient_rows)),
        "latency": bool(_percentile(warm_latency_ms, 95.0) <= THRESHOLDS["latency_p95_ms_max"]),
        "psi_n_rmse": bool(psi_n_rmse <= THRESHOLDS["psi_n_rmse_max"]),
        "relative_current_error": bool(
            relative_current_error <= THRESHOLDS["relative_current_error_max"]
        ),
        "relative_nonlinear_residual_rms": bool(
            relative_residual_rms <= THRESHOLDS["relative_nonlinear_residual_rms_max"]
        ),
    }
    return {
        "admitted": False,
        "case_id": run_spec.case_id,
        "digests": {
            "candidate_psi_sha256": _array_sha256(candidate_psi),
            "coil_current_sha256": _array_sha256(coil_current),
            "ffprime_coefficients_sha256": _array_sha256(ffprime_coeff),
            "ffprime_values_sha256": _array_sha256(ffprime_values),
            "pprime_coefficients_sha256": _array_sha256(pprime_coeff),
            "pprime_values_sha256": _array_sha256(pprime_values),
            "psin_knots_sha256": _array_sha256(psin_knots),
            "r_grid_sha256": _array_sha256(r_grid),
            "reference_psi_sha256": _array_sha256(reference_psi),
            "z_grid_sha256": _array_sha256(z_grid),
        },
        "freegs_version": freegs_version,
        "gradient_audit": {
            "all_finite": bool(
                np.all(np.isfinite(coil_gradient_np))
                and np.all(np.isfinite(p_gradient_np))
                and np.all(np.isfinite(ff_gradient_np))
            ),
            "cotangent_sha256": _array_sha256(cotangent_np),
            "rows": gradient_rows,
        },
        "grid_shape": [grid_points, grid_points],
        "input_contract": {
            "coil_filament_count": len(filaments),
            "anderson_depth": DEFAULT_ANDERSON_DEPTH,
            "cutoff_width": DEFAULT_CUTOFF_WIDTH,
            "ip_ramp": DEFAULT_IP_RAMP,
            "ip_target_a": float(run_spec.plasma_current_a),
            "mixing": DEFAULT_MIXING,
            "n_iter_cap": n_iter,
            "profile_coefficient_count": profile_coefficients,
            "profile_degree": profile_degree,
            "profile_sample_count": int(psin_knots.size),
            "self_field_wall_boundary": True,
            "separatrix": "smooth_xpoint_flux",
            "warm_start_iteration_cap": WARM_START_ITERATION_CAP,
        },
        "latency": {
            "admissible_isolated_evidence": False,
            "cold_start_iterations": cold_start_iterations,
            "compile_and_first_ms": compile_and_first_ms,
            "measurement_mode": WARM_START_MEASUREMENT_MODE,
            "p50_ms": _percentile(warm_latency_ms, 50.0),
            "p95_ms": _percentile(warm_latency_ms, 95.0),
            "reference_freegs_ms": reference_elapsed_ms,
            "repeat_count": len(warm_latency_ms),
            "synchronised": True,
            "warm_compile_and_first_ms": warm_compile_and_first_ms,
            "warm_ms": warm_latency_ms,
            "warm_start_iterations": warm_start_iterations,
            "warm_start_setup_iterations": warm_start_setup_iterations,
        },
        "machine_class": run_spec.machine_class,
        "metrics": {
            "candidate_axis_wb": candidate_axis,
            "candidate_boundary_wb": candidate_boundary,
            "candidate_current_a": candidate_current,
            "psi_max_abs_error_wb": float(np.max(np.abs(difference[reference_mask]))),
            "psi_rmse_wb": float(np.sqrt(np.mean(np.square(difference[reference_mask])))),
            "psi_n_rmse": psi_n_rmse,
            "raw_psi_span_nrmse": float(
                np.sqrt(np.mean(np.square(difference[reference_mask]))) / reference_span
            ),
            "reference_axis_wb": reference_axis,
            "reference_boundary_wb": reference_boundary,
            "reference_current_a": float(run_spec.plasma_current_a),
            "relative_current_error": relative_current_error,
            "relative_nonlinear_residual_rms": relative_residual_rms,
        },
        "public_example": {
            "path": str(run_spec.example_path.relative_to(ROOT)),
            "sha256": _file_sha256(run_spec.example_path),
        },
        "reference_mask_point_count": int(np.count_nonzero(reference_mask)),
        "role": role,
        "threshold_results": threshold_results,
    }


def _selection_lock(
    path: Path | None,
    *,
    evaluation_case_id: str,
) -> dict[str, Any]:
    if path is None:
        return {
            "case_id": evaluation_case_id,
            "created_before_execution": False,
            "path": None,
            "sha256": None,
            "valid": False,
        }
    payload = load_report(path)
    expected = {
        "benchmark_id",
        "case_id",
        "created_at",
        "schema_version",
        "thresholds",
    }
    valid = bool(
        set(payload) == expected
        and payload.get("schema_version") == "scpn-fusion.ida-same-case-selection-lock.v2"
        and payload.get("benchmark_id") == BENCHMARK_ID
        and payload.get("case_id") == evaluation_case_id
        and payload.get("thresholds") == THRESHOLDS
        and isinstance(payload.get("created_at"), str)
        and bool(str(payload["created_at"]).strip())
    )
    return {
        "case_id": evaluation_case_id,
        "created_before_execution": valid,
        "path": str(path),
        "sha256": _file_sha256(path),
        "valid": valid,
    }


def build_report(
    cases: list[dict[str, Any]],
    *,
    generated_at: str,
    environment: dict[str, Any],
    source_artifacts: dict[str, dict[str, str]],
    selection_lock: dict[str, Any],
) -> dict[str, Any]:
    """Build and self-digest the strict IDA same-case evidence report."""
    if not generated_at.strip():
        raise ValueError("generated_at must not be empty")
    if [case.get("role") for case in cases] != list(_CASE_ROLES):
        raise ValueError("cases must be ordered development then evaluation_candidate")
    evaluation = cases[1]
    statistically_held_out = False
    threshold_pass = bool(
        all(evaluation.get("threshold_results", {}).values())
        and environment.get("isolated_host") is True
        and environment.get("x64_enabled") is True
        and selection_lock.get("valid") is True
        and statistically_held_out
    )
    blockers: list[str] = []
    if not selection_lock.get("valid"):
        blockers.append("execution_preceding_selection_lock_missing")
    if environment.get("isolated_host") is not True:
        blockers.append("isolated_latency_evidence_missing")
    if environment.get("x64_enabled") is not True:
        blockers.append("jax_fp64_disabled")
    for name, passed in sorted(evaluation.get("threshold_results", {}).items()):
        if passed is not True:
            blockers.append(f"evaluation_threshold_failed:{name}")
    blockers.extend(
        [
            "collaborator_solver_reference_not_bound",
            "facility_validation_not_bound",
            "pcs_and_safety_programmes_not_bound",
            "statistically_held_out_case_missing",
        ]
    )
    report: dict[str, Any] = {
        "benchmark_id": BENCHMARK_ID,
        "blockers": sorted(set(blockers)),
        "case_role_contract": {
            "development_case_id": cases[0]["case_id"],
            "evaluation_case_id": evaluation["case_id"],
            "evaluation_previously_observed_during_integration": True,
            "selection_lock": selection_lock,
            "statistically_held_out": statistically_held_out,
        },
        "cases": cases,
        "claim_boundary": {field: False for field in CLAIM_FIELDS},
        "environment": environment,
        "generated_at": generated_at,
        "payload_sha256": "",
        "schema_version": SCHEMA_VERSION,
        "solver_contract": _solver_contract(),
        "source_artifacts": source_artifacts,
        "status": (
            "accepted_bounded_same_case_evidence"
            if threshold_pass
            else "blocked_same_case_evidence"
        ),
        "thresholds": dict(THRESHOLDS),
    }
    report["payload_sha256"] = _payload_sha256(report)
    validate_report(report)
    return report


def _require_sha256(value: object, *, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _require_git_oid(value: object, *, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase Git object ID")


def _solver_contract() -> dict[str, Any]:
    return {
        "differentiated_inputs": [
            "coil_current_a",
            "pprime_coefficients_pa_per_wb",
            "ffprime_coefficients_t2_m2_per_wb",
        ],
        "reference_baseline": {
            "grid_shape": [129, 129],
            "latency_ms": 20.0,
            "picard_iterations": 10,
        },
        "conditioned_inputs": ["ip_target_a"],
        "solver_id": SOLVER_ID,
        "units": {
            "coil_current": "A",
            "ffprime": "T^2 m^2 / Wb",
            "pprime": "Pa / Wb",
            "psi": "Wb",
            "r": "m",
            "z": "m",
        },
    }


def _walk_finite(value: object, *, field: str = "report") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _walk_finite(item, field=f"{field}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field} contains a non-string key")
            _walk_finite(item, field=f"{field}.{key}")
        return
    raise ValueError(f"{field} contains unsupported value type {type(value).__name__}")


def validate_report(payload: dict[str, Any]) -> None:
    """Validate a report and reject tamper, drift, overclaim, and false admission."""
    if set(payload) != _REQUIRED_TOP_LEVEL:
        raise ValueError("report top-level fields do not match the v2 schema")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported IDA same-case evidence schema")
    if payload.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("benchmark_id does not match the IDA evidence lane")
    _walk_finite(payload)
    _require_sha256(payload.get("payload_sha256"), field="payload_sha256")
    if payload["payload_sha256"] != _payload_sha256(payload):
        raise ValueError("payload_sha256 does not match report content")
    if payload.get("thresholds") != THRESHOLDS:
        raise ValueError("thresholds do not match the frozen v2 contract")
    if payload.get("claim_boundary") != {field: False for field in CLAIM_FIELDS}:
        raise ValueError("claim_boundary must keep every promotion claim false")
    if payload.get("solver_contract") != _solver_contract():
        raise ValueError("solver_contract does not match the frozen v2 contract")
    generated_at = payload.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at.strip():
        raise ValueError("generated_at must be a non-empty string")
    environment = payload.get("environment")
    if not isinstance(environment, dict) or set(environment) != _ENVIRONMENT_FIELDS:
        raise ValueError("environment fields do not match the v2 contract")
    if not isinstance(environment["isolated_host"], bool) or not isinstance(
        environment["x64_enabled"], bool
    ):
        raise ValueError("environment admission flags must be booleans")
    source_artifacts = payload.get("source_artifacts")
    allowed_source_sets = (
        set(_SOURCE_PATHS),
        {*_SOURCE_PATHS, "repository"},
    )
    if not isinstance(source_artifacts, dict) or set(source_artifacts) not in allowed_source_sets:
        raise ValueError("source_artifacts do not match the v2 contract")
    for name, expected_path in _SOURCE_PATHS.items():
        artifact = source_artifacts.get(name)
        if (
            not isinstance(artifact, dict)
            or set(artifact) != {"path", "sha256"}
            or artifact.get("path") != expected_path
        ):
            raise ValueError(f"source_artifacts.{name} does not match the v2 contract")
        _require_sha256(
            artifact.get("sha256"),
            field=f"source_artifacts.{name}.sha256",
        )
    repository = source_artifacts.get("repository")
    if repository is not None:
        if (
            not isinstance(repository, dict)
            or set(repository) != {"git_commit", "path"}
            or repository.get("path") != "."
        ):
            raise ValueError("source_artifacts.repository does not match the v2 contract")
        _require_git_oid(
            repository.get("git_commit"),
            field="source_artifacts.repository.git_commit",
        )
    status = payload.get("status")
    if status not in {
        "accepted_bounded_same_case_evidence",
        "blocked_same_case_evidence",
    }:
        raise ValueError("status is not valid for the v2 contract")
    cases = payload.get("cases")
    if (
        not isinstance(cases, list)
        or len(cases) != 2
        or [case.get("role") for case in cases if isinstance(case, dict)] != list(_CASE_ROLES)
    ):
        raise ValueError("report must contain development and evaluation cases")
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError("case rows must be objects")
        if case.get("admitted") is not False:
            raise ValueError("case rows must not independently claim admission")
        digests = case.get("digests")
        if not isinstance(digests, dict) or not digests:
            raise ValueError("case rows must bind input and output digests")
        for name, digest in digests.items():
            _require_sha256(digest, field=f"cases[{index}].digests.{name}")
        public_example = case.get("public_example")
        if not isinstance(public_example, dict):
            raise ValueError("case rows must bind a public example")
        _require_sha256(
            public_example.get("sha256"),
            field=f"cases[{index}].public_example.sha256",
        )
        threshold_results = case.get("threshold_results")
        if (
            not isinstance(threshold_results, dict)
            or set(threshold_results)
            != {
                "gradient_audit",
                "latency",
                "psi_n_rmse",
                "relative_current_error",
                "relative_nonlinear_residual_rms",
            }
            or any(not isinstance(value, bool) for value in threshold_results.values())
        ):
            raise ValueError("case threshold results do not match the v2 contract")
        input_contract = case.get("input_contract")
        if (
            not isinstance(input_contract, dict)
            or input_contract.get("warm_start_iteration_cap") != WARM_START_ITERATION_CAP
        ):
            raise ValueError("case input contract omits the frozen warm-start iteration cap")
        latency = case.get("latency")
        if not isinstance(latency, dict):
            raise ValueError("case latency must be an object")
        expected_latency_fields = {
            "admissible_isolated_evidence",
            "cold_start_iterations",
            "compile_and_first_ms",
            "measurement_mode",
            "p50_ms",
            "p95_ms",
            "reference_freegs_ms",
            "repeat_count",
            "synchronised",
            "warm_compile_and_first_ms",
            "warm_ms",
            "warm_start_iterations",
            "warm_start_setup_iterations",
        }
        repeat_count = latency.get("repeat_count")
        warm_ms = latency.get("warm_ms")
        warm_iterations = latency.get("warm_start_iterations")
        cold_iterations = latency.get("cold_start_iterations")
        warm_setup_iterations = latency.get("warm_start_setup_iterations")
        n_iter_cap = input_contract.get("n_iter_cap")
        integer_fields = (cold_iterations, warm_setup_iterations)
        timing_fields = (
            latency.get("compile_and_first_ms"),
            latency.get("p50_ms"),
            latency.get("p95_ms"),
            latency.get("reference_freegs_ms"),
            latency.get("warm_compile_and_first_ms"),
        )
        if (
            set(latency) != expected_latency_fields
            or latency.get("admissible_isolated_evidence") is not False
            or latency.get("synchronised") is not True
            or latency.get("measurement_mode") != WARM_START_MEASUREMENT_MODE
            or isinstance(repeat_count, bool)
            or not isinstance(repeat_count, int)
            or repeat_count < 1
            or not isinstance(warm_ms, list)
            or len(warm_ms) != repeat_count
            or not isinstance(warm_iterations, list)
            or len(warm_iterations) != repeat_count
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0.0
                for value in (*timing_fields, *warm_ms)
            )
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in (*integer_fields, *warm_iterations)
            )
            or isinstance(n_iter_cap, bool)
            or not isinstance(n_iter_cap, int)
            or n_iter_cap < 1
            or cast(int, cold_iterations) > n_iter_cap
            or cast(int, warm_setup_iterations) >= WARM_START_ITERATION_CAP
            or any(iterations >= WARM_START_ITERATION_CAP for iterations in warm_iterations)
        ):
            raise ValueError("case latency does not match the warm-start measurement contract")
        expected_p50 = _percentile([float(value) for value in warm_ms], 50.0)
        expected_p95 = _percentile([float(value) for value in warm_ms], 95.0)
        if latency.get("p50_ms") != expected_p50 or latency.get("p95_ms") != expected_p95:
            raise ValueError("case latency percentiles do not match the recorded warm samples")
    role_contract = payload.get("case_role_contract")
    if not isinstance(role_contract, dict):
        raise ValueError("case_role_contract must be an object")
    if role_contract.get("statistically_held_out") is not False:
        raise ValueError("v2 integration evidence must not claim statistical holdout")
    if set(role_contract) != {
        "development_case_id",
        "evaluation_case_id",
        "evaluation_previously_observed_during_integration",
        "selection_lock",
        "statistically_held_out",
    }:
        raise ValueError("case_role_contract fields do not match the v1 contract")
    if (
        role_contract.get("development_case_id") != cases[0]["case_id"]
        or role_contract.get("evaluation_case_id") != cases[1]["case_id"]
        or role_contract.get("evaluation_previously_observed_during_integration") is not True
    ):
        raise ValueError("case_role_contract does not match the measured cases")
    blockers = payload.get("blockers")
    if (
        not isinstance(blockers, list)
        or not blockers
        or blockers != sorted(set(blockers))
        or any(not isinstance(item, str) or not item for item in blockers)
    ):
        raise ValueError("blockers must be a sorted non-empty unique string list")
    if status == "accepted_bounded_same_case_evidence":
        raise ValueError(
            "v2 integration evidence cannot be accepted while statistical holdout is false"
        )
    required_blockers = {
        "collaborator_solver_reference_not_bound",
        "facility_validation_not_bound",
        "pcs_and_safety_programmes_not_bound",
        "statistically_held_out_case_missing",
    }
    if not required_blockers.issubset(blockers):
        raise ValueError("blockers omit required v1 claim-boundary reasons")


def render_markdown(report: dict[str, Any]) -> str:
    """Render a concise human-readable report without changing claim semantics."""
    validate_report(report)
    lines = [
        "# IDA free-boundary same-case evidence",
        "",
        f"- Status: `{report['status']}`",
        f"- Schema: `{report['schema_version']}`",
        f"- Payload SHA-256: `{report['payload_sha256']}`",
        "- Facility/control/PCS/safety claims: `false`",
        "",
        "## Cases",
        "",
        "| Case | Role | Grid | ψ_N RMSE | current relative error | warm p95 (ms) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        metrics = case["metrics"]
        latency = case["latency"]
        lines.append(
            "| {case_id} | {role} | {grid} | {psi:.6g} | {current:.6g} | {latency:.6g} |".format(
                case_id=case["case_id"],
                role=case["role"],
                grid="×".join(str(value) for value in case["grid_shape"]),
                psi=metrics["psi_n_rmse"],
                current=metrics["relative_current_error"],
                latency=latency["p95_ms"],
            )
        )
    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- `{blocker}`" for blocker in report["blockers"])
    return "\n".join(lines) + "\n"


def run_benchmark(
    *,
    generated_at: str,
    selection_lock_path: Path | None = None,
    latency_repeats: int = 3,
) -> dict[str, Any]:
    """Execute the development and DIII-D-like evaluation candidate cases."""
    if latency_repeats < 1:
        raise ValueError("latency_repeats must be positive")
    environment = _runtime_environment()
    if environment["x64_enabled"] is not True:
        raise RuntimeError("IDA same-case evidence requires JAX FP64")
    development = _execute_case(
        PUBLIC_CASES[0],
        role="development",
        grid_points=65,
        n_iter=DEFAULT_N_ITER,
        latency_repeats=latency_repeats,
    )
    evaluation = _execute_case(
        PUBLIC_CASES[1],
        role="evaluation_candidate",
        grid_points=129,
        n_iter=DEFAULT_N_ITER,
        latency_repeats=latency_repeats,
    )
    selection_lock = _selection_lock(
        selection_lock_path,
        evaluation_case_id=str(evaluation["case_id"]),
    )
    report = build_report(
        [development, evaluation],
        generated_at=generated_at,
        environment=environment,
        source_artifacts=_source_artifacts(),
        selection_lock=selection_lock,
    )
    source_commit = _git_value("rev-parse", "HEAD")
    clean_status = _git_value("status", "--porcelain")
    report["source_artifacts"]["repository"] = {
        "git_commit": source_commit or "0" * 40,
        "path": ".",
    }
    if clean_status:
        report["blockers"] = sorted(set(report["blockers"]) | {"source_worktree_not_clean"})
    report["payload_sha256"] = _payload_sha256(report)
    validate_report(report)
    return report


def write_report(
    report: dict[str, Any],
    *,
    json_path: Path = REPORT_PATH,
    markdown_path: Path = MARKDOWN_PATH,
) -> None:
    """Write validated JSON and Markdown reports."""
    validate_report(report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and return zero only for bounded accepted evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--selection-lock", type=Path)
    parser.add_argument("--latency-repeats", type=int, default=3)
    parser.add_argument("--json-out", type=Path, default=REPORT_PATH)
    parser.add_argument("--markdown-out", type=Path, default=MARKDOWN_PATH)
    args = parser.parse_args(argv)
    report = run_benchmark(
        generated_at=args.generated_at,
        selection_lock_path=args.selection_lock,
        latency_repeats=args.latency_repeats,
    )
    write_report(
        report,
        json_path=args.json_out,
        markdown_path=args.markdown_out,
    )
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "accepted_bounded_same_case_evidence" else 2


if __name__ == "__main__":
    raise SystemExit(main())
