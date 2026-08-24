# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — ITER-Like Machine-Conditioned Dataset Generator
"""Generate a leakage-free fixed-support predictive equilibrium dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

_jax_config_update = cast(Callable[[str, bool], None], jax.config.update)
_jax_config_update("jax_enable_x64", True)

from scpn_fusion.core.jax_free_boundary_gs import MU0_SI, vacuum_field_si
from scpn_fusion.core.jax_free_boundary_predictive import (
    _plasma_current,
    build_response_matrix,
    predictive_gs_residual,
)
from scpn_fusion.core.jax_o_point import smooth_axis_flux
from scpn_fusion.core.jax_predictive_forward_compiled import (
    solve_predictive_equilibrium_compiled,
)
from scpn_fusion.core.jax_x_point import smooth_xpoint_flux
from scpn_fusion.io.machine_conditioned_equilibrium_dataset import (
    SCHEMA_VERSION,
    array_contract,
    sha256_file,
    verify_machine_conditioned_dataset,
)
from scpn_fusion.io.recoverable_npy_dataset import RecoverableNpyDataset
from scpn_fusion.io.safe_loaders import checked_json_load

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATION_SCHEMA = "scpn-fusion.machine-conditioned-equilibrium-generation.v2"
DIAGNOSTIC_NAMES = (
    "candidate_index",
    "converged",
    "iterations",
    "relative_gs_residual_rms",
    "plasma_current_actual_a",
    "plasma_current_relative_error",
    "psi_axis",
    "psi_x",
    "psi_span_abs_wb",
    "plasma_delta_max_abs_wb",
)
FloatArray: TypeAlias = NDArray[np.float64]
Response: TypeAlias = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
SAMPLE_ARRAY_NAMES = frozenset({"inputs", "psi_total", "psi_vacuum", "diagnostics"})


@dataclass(frozen=True)
class FeatureRange:
    """One ordered pre-solve input and its sampling interval."""

    name: str
    unit: str
    minimum: float
    maximum: float


@dataclass(frozen=True)
class MachineArrays:
    """Static numerical representation of one synthetic machine."""

    r: FloatArray
    z: FloatArray
    coil_r: FloatArray
    coil_z: FloatArray
    psin_knots: FloatArray
    support: FloatArray
    wall: FloatArray
    limiter: FloatArray


@dataclass(frozen=True)
class CandidateResult:
    """One candidate solve and its deterministic disposition."""

    candidate_index: int
    accepted: bool
    reason: str
    psi_total: FloatArray | None = None
    psi_vacuum: FloatArray | None = None
    diagnostics: FloatArray | None = None


def _as_object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return cast(dict[str, Any], value)


def _finite_pair(value: Any, *, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element array")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{name} must contain numeric values")
    minimum, maximum = float(value[0]), float(value[1])
    if not np.isfinite(minimum) or not np.isfinite(maximum) or minimum >= maximum:
        raise ValueError(f"{name} must contain finite values with min < max")
    return minimum, maximum


def _positive_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _repo_path(raw_path: str | Path, *, name: str) -> Path:
    path = (REPO_ROOT / raw_path).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"{name} escapes the repository") from exc
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} must be a regular repository file")
    return path


def load_generation_spec(path: Path) -> dict[str, Any]:
    """Load and fail-closed validate the generation specification."""
    spec = _as_object(checked_json_load(path), name="generation specification")
    if spec.get("schema_version") != GENERATION_SCHEMA:
        raise ValueError(f"schema_version must be {GENERATION_SCHEMA}")
    machine = _as_object(spec.get("machine"), name="machine")
    if not isinstance(machine.get("name"), str) or not str(machine["name"]).strip():
        raise ValueError("machine.name must be non-empty")
    grid = _as_object(machine.get("grid"), name="machine.grid")
    _finite_pair(grid.get("r_m"), name="machine.grid.r_m")
    _finite_pair(grid.get("z_m"), name="machine.grid.z_m")
    coils = machine.get("coils")
    if not isinstance(coils, list) or not coils:
        raise ValueError("machine.coils must be a non-empty array")
    names: list[str] = []
    for index, raw in enumerate(coils):
        coil = _as_object(raw, name=f"machine.coils[{index}]")
        name = coil.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"machine.coils[{index}].name must be non-empty")
        names.append(name)
        for key in ("r_m", "z_m", "nominal_current_a"):
            value = coil.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"machine.coils[{index}].{key} must be numeric")
        turns = coil.get("turns")
        if isinstance(turns, bool) or not isinstance(turns, int) or turns < 1:
            raise ValueError(f"machine.coils[{index}].turns must be a positive integer")
        _finite_pair(coil.get("sample_range_a"), name=f"machine.coils[{index}].sample_range_a")
    if len(set(names)) != len(names):
        raise ValueError("machine coil names must be unique")
    for contour_name in ("wall", "limiter", "plasma_support"):
        contour = _as_object(machine.get(contour_name), name=f"machine.{contour_name}")
        for key in ("major_radius_m", "minor_radius_m", "elongation"):
            _positive_number(contour.get(key), name=f"machine.{contour_name}.{key}")
        triangularity = contour.get("triangularity")
        if (
            isinstance(triangularity, bool)
            or not isinstance(triangularity, (int, float))
            or not -0.95 < float(triangularity) < 0.95
        ):
            raise ValueError(f"machine.{contour_name}.triangularity must lie in (-0.95, 0.95)")
        if contour_name != "plasma_support":
            count = contour.get("point_count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 9:
                raise ValueError(f"machine.{contour_name}.point_count must be >= 9")
    support = cast(dict[str, Any], machine["plasma_support"])
    _positive_number(
        support.get("transition_width"), name="machine.plasma_support.transition_width"
    )
    solver = _as_object(spec.get("solver"), name="solver")
    knots = solver.get("psin_knots")
    if not isinstance(knots, list) or len(knots) < 3:
        raise ValueError("solver.psin_knots must contain at least three values")
    knot_array = np.asarray(knots, dtype=np.float64)
    if not np.all(np.isfinite(knot_array)) or not np.all(np.diff(knot_array) > 0.0):
        raise ValueError("solver.psin_knots must be finite and strictly increasing")
    if knot_array[0] != 0.0 or knot_array[-1] != 1.0:
        raise ValueError("solver.psin_knots must span exactly [0, 1]")
    for key in ("n_iter", "anderson_depth", "ip_ramp"):
        value = solver.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"solver.{key} must be a positive integer")
    for key in ("mixing", "cutoff_width", "tol"):
        _positive_number(solver.get(key), name=f"solver.{key}")
    if solver.get("use_mg_preconditioner") is not True:
        raise ValueError("solver.use_mg_preconditioner must be true")
    if solver.get("inner_solver") != "bicgstab":
        raise ValueError("solver.inner_solver must be bicgstab")
    sampling = _as_object(spec.get("sampling"), name="sampling")
    _finite_pair(sampling.get("plasma_current_target_a"), name="sampling.plasma_current_target_a")
    for family in ("pprime", "ffprime"):
        ranges = sampling.get(f"{family}_knot_ranges")
        if not isinstance(ranges, list) or len(ranges) != len(knots) - 1:
            raise ValueError(f"sampling.{family}_knot_ranges must have len(psin_knots)-1 rows")
        for index, pair in enumerate(ranges):
            _finite_pair(pair, name=f"sampling.{family}_knot_ranges[{index}]")
    factor = sampling.get("max_attempt_factor")
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError("sampling.max_attempt_factor must be a positive integer")
    tolerances = _as_object(spec.get("tolerances"), name="tolerances")
    for key in (
        "relative_gs_residual_rms_max",
        "plasma_current_relative_error_max",
        "psi_span_abs_wb_min",
        "plasma_delta_max_abs_wb_min",
        "vacuum_reconstruction_max_abs",
        "diagnostic_replay_max_abs",
    ):
        _positive_number(tolerances.get(key), name=f"tolerances.{key}")
    return spec


def feature_ranges(spec: dict[str, Any]) -> list[FeatureRange]:
    """Build the stable leakage-free feature order."""
    sampling = cast(dict[str, Any], spec["sampling"])
    ip_min, ip_max = _finite_pair(
        sampling["plasma_current_target_a"], name="sampling.plasma_current_target_a"
    )
    ranges = [FeatureRange("plasma_current_target_a", "A", ip_min, ip_max)]
    machine = cast(dict[str, Any], spec["machine"])
    for raw in cast(list[dict[str, Any]], machine["coils"]):
        minimum, maximum = _finite_pair(raw["sample_range_a"], name=f"coil {raw['name']}")
        ranges.append(FeatureRange(f"coil_current_a.{raw['name']}", "A", minimum, maximum))
    units = {"pprime": "Pa/Wb", "ffprime": "T^2*m^2/Wb"}
    for family in ("pprime", "ffprime"):
        profile_ranges = cast(list[list[float]], sampling[f"{family}_knot_ranges"])
        for index, pair in enumerate(profile_ranges):
            minimum, maximum = _finite_pair(pair, name=f"{family} knot {index}")
            ranges.append(FeatureRange(f"{family}_knot_{index}", units[family], minimum, maximum))
    return ranges


def latin_hypercube(samples: int, dimensions: int, *, seed: int) -> FloatArray:
    """Return a seeded stratified Latin-hypercube design in ``[0, 1)``."""
    if samples < 1 or dimensions < 1:
        raise ValueError("samples and dimensions must be positive")
    rng = np.random.default_rng(seed)
    design: FloatArray = np.empty((samples, dimensions), dtype=np.float64)
    for column in range(dimensions):
        strata = rng.permutation(samples)
        design[:, column] = (strata + rng.random(samples)) / samples
    return design


def sample_feature_matrix(ranges: list[FeatureRange], *, candidates: int, seed: int) -> FloatArray:
    unit = latin_hypercube(candidates, len(ranges), seed=seed)
    lower = np.asarray([item.minimum for item in ranges], dtype=np.float64)
    upper = np.asarray([item.maximum for item in ranges], dtype=np.float64)
    return np.asarray(lower + unit * (upper - lower), dtype=np.float64)


def _analytic_contour(spec: dict[str, Any]) -> FloatArray:
    theta = np.linspace(0.0, 2.0 * np.pi, int(spec["point_count"]), endpoint=True, dtype=np.float64)
    r0, minor = float(spec["major_radius_m"]), float(spec["minor_radius_m"])
    elongation, delta = float(spec["elongation"]), float(spec["triangularity"])
    r = r0 + minor * np.cos(theta + np.arcsin(delta) * np.sin(theta))
    z = elongation * minor * np.sin(theta)
    contour = np.column_stack((r, z)).astype(np.float64, copy=False)
    contour[-1] = contour[0]
    return contour


def _machine_arrays(spec: dict[str, Any], grid_resolution: tuple[int, int]) -> MachineArrays:
    machine = cast(dict[str, Any], spec["machine"])
    grid = cast(dict[str, Any], machine["grid"])
    r_range = _finite_pair(grid["r_m"], name="machine.grid.r_m")
    z_range = _finite_pair(grid["z_m"], name="machine.grid.z_m")
    nr, nz = grid_resolution
    r = np.linspace(*r_range, nr, dtype=np.float64)
    z = np.linspace(*z_range, nz, dtype=np.float64)
    coils = cast(list[dict[str, Any]], machine["coils"])
    coil_r = np.asarray([float(coil["r_m"]) for coil in coils], dtype=np.float64)
    coil_z = np.asarray([float(coil["z_m"]) for coil in coils], dtype=np.float64)
    inside = (coil_r >= r[0]) & (coil_r <= r[-1]) & (coil_z >= z[0]) & (coil_z <= z[-1])
    if np.any(inside):
        raise ValueError("every coil filament must lie outside the computational grid")
    support_spec = cast(dict[str, Any], machine["plasma_support"])
    rr, zz = np.meshgrid(r, z)
    r0 = float(support_spec["major_radius_m"])
    minor = float(support_spec["minor_radius_m"])
    y = zz / (float(support_spec["elongation"]) * minor)
    x = (rr - r0) / minor + float(support_spec["triangularity"]) * y * y
    rho_squared = x * x + y * y
    argument = np.clip((1.0 - rho_squared) / float(support_spec["transition_width"]), -60.0, 60.0)
    support = np.asarray(1.0 / (1.0 + np.exp(-argument)), dtype=np.float64)
    solver = cast(dict[str, Any], spec["solver"])
    return MachineArrays(
        r=r,
        z=z,
        coil_r=coil_r,
        coil_z=coil_z,
        psin_knots=np.asarray(solver["psin_knots"], dtype=np.float64),
        support=support,
        wall=_analytic_contour(cast(dict[str, Any], machine["wall"])),
        limiter=_analytic_contour(cast(dict[str, Any], machine["limiter"])),
    )


def _candidate_controls(
    row: FloatArray, ranges: list[FeatureRange], coil_count: int, knot_count: int
) -> tuple[float, FloatArray, FloatArray, FloatArray]:
    values = dict(zip((item.name for item in ranges), row, strict=True))
    coil_names = [item.name.split(".", 1)[1] for item in ranges[1 : coil_count + 1]]
    currents = np.asarray([values[f"coil_current_a.{name}"] for name in coil_names])
    pprime = np.asarray([values[f"pprime_knot_{i}"] for i in range(knot_count - 1)] + [0.0])
    ffprime = np.asarray([values[f"ffprime_knot_{i}"] for i in range(knot_count - 1)] + [0.0])
    return float(values["plasma_current_target_a"]), currents, pprime, ffprime


def _solve_candidate(
    candidate_index: int,
    row: FloatArray,
    ranges: list[FeatureRange],
    machine: MachineArrays,
    spec: dict[str, Any],
    response: Response,
) -> CandidateResult:
    ip_target, currents, pprime, ffprime = _candidate_controls(
        row, ranges, len(machine.coil_r), len(machine.psin_knots)
    )
    solver = cast(dict[str, Any], spec["solver"])
    tolerance = cast(dict[str, Any], spec["tolerances"])
    r_jax, z_jax = jnp.asarray(machine.r), jnp.asarray(machine.z)
    coil_r_jax, coil_z_jax = jnp.asarray(machine.coil_r), jnp.asarray(machine.coil_z)
    current_jax = jnp.asarray(currents)
    pprime_jax, ffprime_jax = jnp.asarray(pprime), jnp.asarray(ffprime)
    knots_jax, support_jax = jnp.asarray(machine.psin_knots), jnp.asarray(machine.support)
    response_matrix, wall_idx, source_idx = response
    try:
        solved = solve_predictive_equilibrium_compiled(
            current_jax,
            pprime_jax,
            ffprime_jax,
            r_jax,
            z_jax,
            coil_r_jax,
            coil_z_jax,
            knots_jax,
            ip_target,
            response_matrix,
            wall_idx,
            source_idx,
            n_iter=int(solver["n_iter"]),
            anderson_depth=int(solver["anderson_depth"]),
            mixing=float(solver["mixing"]),
            ip_ramp=int(solver["ip_ramp"]),
            cutoff_width=float(solver["cutoff_width"]),
            tol=float(solver["tol"]),
            use_mg_preconditioner=True,
            inner_solver="bicgstab",
            return_iterations=True,
            fixed_support_weights=support_jax,
        )
        psi_jax, iterations = cast(tuple[jnp.ndarray, int], solved)
        psi_jax.block_until_ready()
    except Exception as exc:
        return CandidateResult(candidate_index, False, f"solver_exception:{type(exc).__name__}")
    psi = np.asarray(psi_jax, dtype=np.float64)
    vacuum_jax = vacuum_field_si(r_jax, z_jax, coil_r_jax, coil_z_jax, current_jax, MU0_SI)
    vacuum = np.asarray(vacuum_jax, dtype=np.float64)
    if not np.all(np.isfinite(psi)) or not np.all(np.isfinite(vacuum)):
        return CandidateResult(candidate_index, False, "non_finite_field")
    if iterations >= int(solver["n_iter"]):
        return CandidateResult(candidate_index, False, "iteration_cap")
    axis = smooth_axis_flux(psi_jax)
    xpoint = smooth_xpoint_flux(psi_jax, r_jax, z_jax)
    d_area = (r_jax[1] - r_jax[0]) * (z_jax[1] - z_jax[0])
    current_density = _plasma_current(
        psi_jax,
        r_jax,
        axis,
        xpoint,
        knots_jax,
        pprime_jax,
        ffprime_jax,
        jnp.asarray(ip_target),
        d_area,
        float(solver["cutoff_width"]),
        MU0_SI,
        fixed_support_weights=support_jax,
    )
    actual_ip = float(jnp.sum(current_density) * d_area)
    residual = predictive_gs_residual(
        psi_jax,
        current_jax,
        pprime_jax,
        ffprime_jax,
        r_jax,
        z_jax,
        coil_r_jax,
        coil_z_jax,
        knots_jax,
        jnp.asarray(ip_target),
        response_matrix,
        wall_idx,
        source_idx,
        cutoff_width=float(solver["cutoff_width"]),
        fixed_support_weights=support_jax,
    )
    source = -(MU0_SI * r_jax[jnp.newaxis, :] * current_density)
    residual_rms = jnp.sqrt(jnp.mean(jnp.square(residual[1:-1, 1:-1])))
    source_rms = jnp.sqrt(jnp.mean(jnp.square(source[1:-1, 1:-1])))
    relative_residual = float(residual_rms / jnp.maximum(source_rms, 1.0e-30))
    ip_error = abs(actual_ip - ip_target) / max(abs(ip_target), 1.0)
    span = abs(float(axis) - float(xpoint))
    plasma_delta = float(np.max(np.abs(psi - vacuum)))
    metrics = (relative_residual, ip_error, span, plasma_delta)
    if not np.all(np.isfinite(metrics)):
        return CandidateResult(candidate_index, False, "non_finite_diagnostics")
    checks = (
        (relative_residual <= float(tolerance["relative_gs_residual_rms_max"]), "gs_residual"),
        (ip_error <= float(tolerance["plasma_current_relative_error_max"]), "ip_closure"),
        (span >= float(tolerance["psi_span_abs_wb_min"]), "flux_span"),
        (plasma_delta >= float(tolerance["plasma_delta_max_abs_wb_min"]), "no_plasma"),
    )
    for passed, reason in checks:
        if not passed:
            return CandidateResult(candidate_index, False, reason)
    diagnostics = np.asarray(
        [
            float(candidate_index),
            1.0,
            float(iterations),
            relative_residual,
            actual_ip,
            ip_error,
            float(axis),
            float(xpoint),
            span,
            plasma_delta,
        ],
        dtype=np.float64,
    )
    return CandidateResult(candidate_index, True, "accepted", psi, vacuum, diagnostics)


def _unit_green_maps(machine: MachineArrays) -> FloatArray:
    r, z = jnp.asarray(machine.r), jnp.asarray(machine.z)
    coil_r, coil_z = jnp.asarray(machine.coil_r), jnp.asarray(machine.coil_z)
    maps = []
    for index in range(len(machine.coil_r)):
        unit = jnp.zeros(len(machine.coil_r), dtype=jnp.float64).at[index].set(1.0)
        maps.append(np.asarray(vacuum_field_si(r, z, coil_r, coil_z, unit), dtype=np.float64))
    return np.asarray(maps, dtype=np.float64)


def _repository_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _array_sha256(array: FloatArray) -> str:
    """Hash one C-contiguous float64 array without creating a bytes copy."""
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _manifest(
    *,
    spec: dict[str, Any],
    spec_path: Path,
    ranges: list[FeatureRange],
    arrays: dict[str, FloatArray],
    array_paths: dict[str, Path],
    samples: int,
    seed: int,
    grid_resolution: tuple[int, int],
    attempted_candidates: int,
    rejection_counts: Counter[str],
) -> dict[str, Any]:
    machine = cast(dict[str, Any], spec["machine"])
    solver = cast(dict[str, Any], spec["solver"])
    source_files = {
        "generator": Path(__file__).resolve(),
        "dataset_contract": REPO_ROOT
        / "src/scpn_fusion/io/machine_conditioned_equilibrium_dataset.py",
        "recovery_store": REPO_ROOT / "src/scpn_fusion/io/recoverable_npy_dataset.py",
        "predictive_solver": REPO_ROOT / "src/scpn_fusion/core/jax_free_boundary_predictive.py",
        "compiled_solver": REPO_ROOT / "src/scpn_fusion/core/jax_predictive_forward_compiled.py",
        "generation_spec": spec_path,
    }
    coils = []
    for raw in cast(list[dict[str, Any]], machine["coils"]):
        coil = dict(raw)
        coil["current_feature"] = f"coil_current_a.{raw['name']}"
        coil["polarity"] = "negative" if float(raw["sample_range_a"][1]) < 0.0 else "positive"
        coils.append(coil)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": (
            f"iter-like-fixed-support-v2-n{samples}-seed{seed}-"
            f"{grid_resolution[0]}x{grid_resolution[1]}"
        ),
        "claims": {
            "class": "synthetic_machine_conditioned_fixed_support_equilibrium_distribution",
            "facility_validated": False,
            "experimental_shot_data": False,
            "time_series_prediction": False,
            "limits": [
                "Synthetic SCPN solver output; not ITER Organisation or experimental data.",
                "Wall, limiter, support and coil geometry are analytic ITER-like descriptors, not ITER CAD.",
                "Plasma support is fixed across samples; free-boundary shape prediction is not claimed.",
                "This cohort validates equilibrium-surrogate conditioning only, not shot forecasting, IDA, PCS, control or safety performance.",
            ],
        },
        "source": {
            "repository": "https://github.com/anulum/scpn-fusion-core",
            "repository_head_context": _repository_head(),
            "files": {
                name: {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}
                for name, path in source_files.items()
            },
        },
        "generation": {
            "sampling_method": "seeded_stratified_latin_hypercube",
            "seed": seed,
            "requested_samples": samples,
            "accepted_samples": samples,
            "attempted_candidates": attempted_candidates,
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "grid_resolution_rz": list(grid_resolution),
            "candidate_order": "ascending_index_first_accepted",
            "storage": "fixed_shape_float64_npy_memmap_with_atomic_external_checkpoint",
        },
        "solver": {
            "family": "compiled_predictive_grad_shafranov",
            "model_boundary": "machine_defined_fixed_plasma_support",
            "coil_field_decomposition": False,
            "coil_location_requirement": "every filament outside computational rectangle",
            **solver,
        },
        "machine": {
            "name": machine["name"],
            "coordinate_system": "cylindrical_R_Z_m",
            "grid_index_order": "sample_Z_R",
            "poloidal_flux_definition": "psi = Phi_p/(2*pi), poloidal flux per radian [Wb]",
            "profile_derivative_variable": "psi [Wb]",
            "cocos": 3,
            "gauge": "absolute Green-function vacuum plus plasma self-field; no per-sample shift",
            "current_unit": "A",
            "current_semantics": "signed effective ampere-turn current; turns recorded explicitly",
            "coils": coils,
            "grid": machine["grid"],
            "wall": {**cast(dict[str, Any], machine["wall"]), "source": "analytic_synthetic"},
            "limiter": {
                **cast(dict[str, Any], machine["limiter"]),
                "source": "analytic_synthetic",
            },
            "plasma_support": {
                **cast(dict[str, Any], machine["plasma_support"]),
                "source": "analytic_synthetic_static_pre_solve",
                "array": "plasma_support_weights",
            },
            "reference_physical_descriptor": machine.get("reference_physical_descriptor"),
        },
        "features": [
            {
                "index": index,
                "name": item.name,
                "unit": item.unit,
                "role": "pre_solve_control",
                "minimum": item.minimum,
                "maximum": item.maximum,
            }
            for index, item in enumerate(ranges)
        ],
        "diagnostic_names": list(DIAGNOSTIC_NAMES),
        "tolerances": spec["tolerances"],
        "arrays": {
            name: array_contract(array_paths[name], array, role=name)
            for name, array in arrays.items()
        },
        "artifact_custody": {
            "bounded_reference": "may be tracked in Git",
            "production": "owner-controlled FTP/storage with public HTTPS download when available",
            "git_retains": "manifest, SHA-256, provenance, licensing and reproduction commands",
        },
        "reproduction": {
            "command": (
                "python tools/generate_iter_machine_conditioned_dataset.py "
                f"--spec {spec_path.relative_to(REPO_ROOT)} --samples {samples} --seed {seed} "
                f"--grid-resolution {grid_resolution[0]} {grid_resolution[1]} "
                "--checkpoint-every 100 --output-dir <dataset-directory>"
            ),
            "resume_command": (
                "python tools/generate_iter_machine_conditioned_dataset.py "
                f"--spec {spec_path.relative_to(REPO_ROOT)} --samples {samples} --seed {seed} "
                f"--grid-resolution {grid_resolution[0]} {grid_resolution[1]} "
                "--checkpoint-every 100 --resume --output-dir <dataset-directory>"
            ),
            "verification": (
                "python tools/verify_iter_machine_conditioned_dataset.py "
                "--dataset-dir <dataset-directory> --full-field-scan"
            ),
        },
    }


def generate_dataset(
    *,
    spec_path: Path,
    output_dir: Path,
    samples: int,
    seed: int,
    grid_resolution: tuple[int, int],
    checkpoint_every: int = 100,
    resume: bool = False,
    pause_after_accepted: int | None = None,
) -> dict[str, Any]:
    """Stream, checkpoint, atomically install and verify one v2 dataset."""
    if samples < 1 or min(grid_resolution) < 9 or checkpoint_every < 1:
        raise ValueError("samples must be positive and grid resolution at least 9 x 9")
    if pause_after_accepted is not None and not 1 <= pause_after_accepted < samples:
        raise ValueError("pause_after_accepted must lie in [1, samples)")
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.parent / f".{output.name}.partial"
    recovery = output.parent / f".{output.name}.recovery.json"
    spec_path = _repo_path(spec_path.resolve(), name="generation specification")
    spec = load_generation_spec(spec_path)
    ranges = feature_ranges(spec)
    max_candidates = samples * int(spec["sampling"]["max_attempt_factor"])
    candidates = sample_feature_matrix(ranges, candidates=max_candidates, seed=seed)
    machine = _machine_arrays(spec, grid_resolution)
    response = build_response_matrix(jnp.asarray(machine.r), jnp.asarray(machine.z))
    static_arrays: dict[str, FloatArray] = {
        "grid_r_m": machine.r,
        "grid_z_m": machine.z,
        "wall_rz_m": machine.wall,
        "limiter_rz_m": machine.limiter,
        "plasma_support_weights": machine.support,
        "coil_green_psi_per_current": _unit_green_maps(machine),
    }
    shapes = {
        "inputs": (samples, len(ranges)),
        "psi_total": (samples, len(machine.z), len(machine.r)),
        "psi_vacuum": (samples, len(machine.z), len(machine.r)),
        "diagnostics": (samples, len(DIAGNOSTIC_NAMES)),
        **{name: array.shape for name, array in static_arrays.items()},
    }
    logical_bytes = sum(int(np.prod(shape, dtype=np.int64)) * 8 for shape in shapes.values())
    free_bytes = shutil.disk_usage(output.parent).free
    safety_margin = max(1024**3, logical_bytes // 10)
    required_free = safety_margin if resume else logical_bytes + safety_margin
    if free_bytes < required_free:
        raise OSError(
            f"insufficient free disk: need {required_free} bytes, have {free_bytes}"
        )
    source_paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "src/scpn_fusion/io/recoverable_npy_dataset.py",
        REPO_ROOT / "src/scpn_fusion/io/machine_conditioned_equilibrium_dataset.py",
        REPO_ROOT / "src/scpn_fusion/core/jax_free_boundary_predictive.py",
        REPO_ROOT / "src/scpn_fusion/core/jax_predictive_forward_compiled.py",
        spec_path,
    )
    run_contract = {
        "generation_schema": GENERATION_SCHEMA,
        "samples": samples,
        "seed": seed,
        "grid_resolution_rz": list(grid_resolution),
        "candidate_design_sha256": _array_sha256(candidates),
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in source_paths
        },
    }
    if resume:
        store = RecoverableNpyDataset.resume(
            partial_dir=partial,
            recovery_path=recovery,
            shapes=shapes,
            sample_array_names=SAMPLE_ARRAY_NAMES,
            run_contract=run_contract,
        )
        for name, array in static_arrays.items():
            store.require_array_equal(name, array)
    else:
        store = RecoverableNpyDataset.create(
            partial_dir=partial,
            recovery_path=recovery,
            shapes=shapes,
            sample_array_names=SAMPLE_ARRAY_NAMES,
            run_contract=run_contract,
            initial_arrays=static_arrays,
        )
    accepted_count = store.accepted_samples
    next_candidate = store.next_candidate_index
    rejection_counts = Counter(store.rejection_counts)
    if accepted_count > samples or next_candidate > len(candidates):
        raise ValueError("recovered progress exceeds the immutable run bounds")
    if pause_after_accepted is not None and pause_after_accepted <= accepted_count:
        raise ValueError("pause_after_accepted must be beyond recovered progress")
    for index in range(next_candidate, len(candidates)):
        row = candidates[index]
        result = _solve_candidate(index, row, ranges, machine, spec, response)
        if result.accepted:
            sample_values = {
                "inputs": row,
                "psi_total": cast(FloatArray, result.psi_total),
                "psi_vacuum": cast(FloatArray, result.psi_vacuum),
                "diagnostics": cast(FloatArray, result.diagnostics),
            }
            store.write_sample(accepted_count, sample_values)
            accepted_count += 1
        else:
            rejection_counts[result.reason] += 1
        next_candidate = index + 1
        checkpoint_due = accepted_count > 0 and accepted_count % checkpoint_every == 0
        if checkpoint_due or accepted_count == samples:
            store.checkpoint(
                accepted_samples=accepted_count,
                next_candidate_index=next_candidate,
                rejection_counts=rejection_counts,
            )
            print(
                json.dumps(
                    {
                        "status": "running" if accepted_count < samples else "finalizing",
                        "accepted_samples": accepted_count,
                        "requested_samples": samples,
                        "attempted_candidates": next_candidate,
                        "rejection_counts": dict(sorted(rejection_counts.items())),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if pause_after_accepted is not None and accepted_count >= pause_after_accepted:
            store.checkpoint(
                accepted_samples=accepted_count,
                next_candidate_index=next_candidate,
                rejection_counts=rejection_counts,
            )
            return {
                "status": "paused",
                "accepted_samples": accepted_count,
                "requested_samples": samples,
                "next_candidate_index": next_candidate,
                "partial_dir": str(partial),
                "recovery_checkpoint": str(recovery),
            }
        if accepted_count == samples:
            break
    if accepted_count < samples:
        store.checkpoint(
            accepted_samples=accepted_count,
            next_candidate_index=next_candidate,
            rejection_counts=rejection_counts,
        )
        raise RuntimeError(
            f"accepted only {accepted_count} of {samples}; rejections={dict(rejection_counts)}"
        )
    arrays = store.arrays
    paths = {name: partial / f"{name}.npy" for name in arrays}
    manifest = _manifest(
        spec=spec,
        spec_path=spec_path,
        ranges=ranges,
        arrays=arrays,
        array_paths=paths,
        samples=samples,
        seed=seed,
        grid_resolution=grid_resolution,
        attempted_candidates=next_candidate,
        rejection_counts=rejection_counts,
    )
    with (partial / "manifest.json").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    verification = verify_machine_conditioned_dataset(partial, full_field_scan=True)
    if verification["status"] != "passed":
        raise RuntimeError(f"generated dataset failed verification: {verification['failures']}")
    partial.rename(output)
    store.remove_recovery_checkpoint()
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--grid-resolution", type=int, nargs=2, metavar=("NR", "NZ"), default=(129, 129)
    )
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--pause-after-accepted",
        type=int,
        help="operator pause boundary used to prove and exercise durable resume",
    )
    args = parser.parse_args()
    result = generate_dataset(
        spec_path=args.spec,
        output_dir=args.output_dir,
        samples=args.samples,
        seed=args.seed,
        grid_resolution=(args.grid_resolution[0], args.grid_resolution[1]),
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        pause_after_accepted=args.pause_after_accepted,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
