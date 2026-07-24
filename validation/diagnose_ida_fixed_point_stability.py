#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Measure stationary-map forcing and local gain around the FreeGS reference."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
from typing import Any, Callable, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

import validation.ida_fixed_point_stability_contract as contract

_same_case = cast(Any, importlib.import_module("validation.benchmark_ida_same_case"))
_source = cast(
    Any,
    importlib.import_module("validation.diagnose_ida_fixed_reference_source"),
)
_mechanism = cast(
    Any,
    importlib.import_module("validation.diagnose_ida_fixed_reference_source_mechanism"),
)
_predictive = cast(
    Any,
    importlib.import_module("scpn_fusion.core.jax_free_boundary_predictive"),
)

ROOT: Path = _same_case.ROOT
SAME_CASE_PATH = ROOT / contract.SAME_CASE_PATH
SOURCE_MECHANISM_PATH = ROOT / contract.SOURCE_MECHANISM_PATH
REPORT_PATH = ROOT / "validation" / "reports" / "ida_fixed_point_stability.json"
MARKDOWN_PATH = ROOT / "validation" / "reports" / "ida_fixed_point_stability.md"
FloatArray: TypeAlias = NDArray[np.float64]
_array_sha256: Callable[[object], str] = _same_case._array_sha256
_file_sha256: Callable[[Path], str] = _same_case._file_sha256
_git_value: Callable[..., str | None] = _same_case._git_value
_runtime_environment: Callable[[], dict[str, Any]] = _same_case._runtime_environment


def _finite_plane(value: object, *, field: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must be a finite non-trivial 2D array")
    return array


def _cosine(left: FloatArray, right: FloatArray) -> float:
    if left.shape != right.shape or left.size == 0:
        raise ValueError("cosine fields must be matching non-empty arrays")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-30:
        raise ValueError("cosine fields must have non-zero norms")
    return float(np.clip(np.vdot(left, right).real / denominator, -1.0, 1.0))


def _vector_metrics(field: object, *, terminal_error: object) -> dict[str, Any]:
    array = _finite_plane(field, field="diagnostic vector")
    error = _finite_plane(terminal_error, field="terminal error")
    if array.shape != error.shape:
        raise ValueError("diagnostic vector and terminal error must have matching shapes")
    error_norm_sq = float(np.vdot(error, error).real)
    if error_norm_sq <= 1.0e-30:
        raise ValueError("terminal error must have a non-zero norm")
    array_norm = float(np.linalg.norm(array))
    cosine = 0.0 if array_norm <= 1.0e-30 else _cosine(array, error)
    return {
        "cosine_to_terminal_error": cosine,
        "field_sha256": _array_sha256(array),
        "l2_wb": array_norm,
        "linf_wb": float(np.max(np.abs(array))),
        "projection_on_terminal_error": float(np.vdot(array, error).real / error_norm_sq),
        "relative_l2_to_terminal_error": array_norm / math.sqrt(error_norm_sq),
    }


def _jvp_metrics(direction: object, image: object) -> dict[str, Any]:
    source = _finite_plane(direction, field="JVP direction")
    target = _finite_plane(image, field="JVP image")
    if source.shape != target.shape:
        raise ValueError("JVP direction and image must have matching shapes")
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if source_norm <= 1.0e-30:
        raise ValueError("JVP direction must have a non-zero norm")
    return {
        "alignment_with_input": 0.0 if target_norm <= 1.0e-30 else _cosine(source, target),
        "gain_l2": target_norm / source_norm,
        "input_sha256": _array_sha256(source),
        "output_sha256": _array_sha256(target),
    }


def _trajectory_row(
    value: object,
    *,
    step: int,
    reference: object,
    candidate: object,
) -> dict[str, Any]:
    plane = _finite_plane(value, field="trajectory plane")
    reference_plane = _finite_plane(reference, field="trajectory reference")
    candidate_plane = _finite_plane(candidate, field="trajectory candidate")
    if plane.shape != reference_plane.shape or plane.shape != candidate_plane.shape:
        raise ValueError("trajectory fields must have matching shapes")
    error = candidate_plane - reference_plane
    error_norm = float(np.linalg.norm(error))
    if error_norm <= 1.0e-30:
        raise ValueError("trajectory terminal error must have a non-zero norm")
    displacement = plane - reference_plane
    return {
        "distance_to_candidate_relative_to_terminal": (
            float(np.linalg.norm(plane - candidate_plane)) / error_norm
        ),
        "distance_to_reference_relative_to_terminal": (
            float(np.linalg.norm(displacement)) / error_norm
        ),
        "projection_on_terminal_error": float(
            np.vdot(displacement, error).real / (error_norm * error_norm)
        ),
        "psi_sha256": _array_sha256(plane),
        "step": step,
    }


def _source_artifacts(
    public_example_path: Path, *, source_commit: str, clean: bool
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {
        name: {"path": path, "sha256": _file_sha256(ROOT / path)}
        for name, path in sorted(contract.SOURCE_PATHS.items())
    }
    artifacts["freegs_public_example"] = {
        "path": str(public_example_path.relative_to(ROOT)),
        "sha256": _file_sha256(public_example_path),
    }
    artifacts["repository"] = {
        "git_commit": source_commit,
        "path": ".",
        "worktree_clean": clean,
    }
    return artifacts


def _load_bound_reports(
    same_case_path: Path,
    source_mechanism_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    same_case = _same_case.load_report(same_case_path)
    _same_case.validate_report(same_case)
    mechanism = _same_case.load_report(source_mechanism_path)
    _mechanism.validate_report(mechanism)
    evaluation = _source._evaluation_case(same_case)
    if mechanism["bindings"]["same_case"]["payload_sha256"] != same_case["payload_sha256"]:
        raise ValueError("source mechanism does not bind the selected same-case payload")
    return (
        cast(dict[str, Any], same_case),
        cast(dict[str, Any], mechanism),
        cast(dict[str, Any], evaluation),
    )


def run_diagnostic(
    *,
    generated_at: str,
    same_case_path: Path = SAME_CASE_PATH,
    source_mechanism_path: Path = SOURCE_MECHANISM_PATH,
) -> dict[str, Any]:
    """Run the frozen 129-square fixed-point stability diagnostic."""
    if cast(bool, jax.config.values["jax_enable_x64"]) is not True:
        raise RuntimeError("fixed-point stability diagnostic requires JAX FP64")
    same_case, mechanism, evaluation = _load_bound_reports(
        same_case_path,
        source_mechanism_path,
    )
    (
        _,
        _,
        spec,
        tokamak,
        equilibrium,
        profiles,
        freegs_version,
    ) = _source._solve_reference(same_case_path)
    r_grid = np.asarray(equilibrium.R_1D, dtype=np.float64)
    z_grid = np.asarray(equilibrium.Z_1D, dtype=np.float64)
    reference = _finite_plane(
        np.asarray(equilibrium.psi(), dtype=np.float64).T,
        field="FreeGS reference",
    )
    reference_current = _finite_plane(
        np.asarray(equilibrium.Jtor, dtype=np.float64).T,
        field="FreeGS current",
    )
    if list(reference.shape) != contract.GRID_SHAPE:
        raise ValueError("FreeGS reference must use the frozen 129x129 grid")
    if _array_sha256(reference) != evaluation["digests"]["reference_psi_sha256"]:
        raise ValueError("reconstructed FreeGS reference digest disagrees with same-case evidence")

    knots = np.linspace(0.0, 1.0, _source.PROFILE_SAMPLE_COUNT, dtype=np.float64)
    pprime_coefficients, _ = _source._fit_compact_profile(
        np.asarray(profiles.pprime(knots), dtype=np.float64),
        knots,
        n_coefficients=_source.PROFILE_COEFFICIENT_COUNT,
        degree=_source.PROFILE_DEGREE,
    )
    ffprime_coefficients, _ = _source._fit_compact_profile(
        np.asarray(profiles.ffprime(knots), dtype=np.float64),
        knots,
        n_coefficients=_source.PROFILE_COEFFICIENT_COUNT,
        degree=_source.PROFILE_DEGREE,
    )
    candidate, _, _ = _source._solve_candidate(
        spec=spec,
        tokamak=tokamak,
        r_grid=r_grid,
        z_grid=z_grid,
        knots=knots,
        pprime_coefficients=pprime_coefficients,
        ffprime_coefficients=ffprime_coefficients,
    )
    candidate = _finite_plane(candidate, field="same-case candidate")
    if _array_sha256(candidate) != evaluation["digests"]["candidate_psi_sha256"]:
        raise ValueError("reconstructed candidate digest disagrees with same-case evidence")
    terminal_error = candidate - reference
    if float(np.linalg.norm(terminal_error)) <= 1.0e-30:
        raise RuntimeError("same-case candidate and FreeGS reference are degenerate")

    _, filaments = _source._machine_filaments(tokamak)
    coil_r = np.asarray([row[0] for row in filaments], dtype=np.float64)
    coil_z = np.asarray([row[1] for row in filaments], dtype=np.float64)
    coil_current = np.asarray([row[2] for row in filaments], dtype=np.float64)
    basis = _source._bspline_design_matrix(
        knots,
        n_coeff=_source.PROFILE_COEFFICIENT_COUNT,
        degree=_source.PROFILE_DEGREE,
    )
    r_jax = jnp.asarray(r_grid)
    z_jax = jnp.asarray(z_grid)
    knots_jax = jnp.asarray(knots)
    pprime_values = _source._evaluate_profile(
        jnp.asarray(pprime_coefficients),
        jnp.asarray(basis),
    )
    ffprime_values = _source._evaluate_profile(
        jnp.asarray(ffprime_coefficients),
        jnp.asarray(basis),
    )
    response, wall_indices, source_indices = _source._build_response_matrix(r_jax, z_jax)
    response = jnp.asarray(response)
    wall_indices = jnp.asarray(wall_indices)
    source_indices = jnp.asarray(source_indices)
    shape = reference.shape
    d_r = float(r_grid[1] - r_grid[0])
    d_z = float(z_grid[1] - z_grid[0])
    d_area = d_r * d_z
    reference_jax = jnp.asarray(reference)
    reference_flat = reference_jax.reshape(-1)
    psi_coil = _predictive.vacuum_field_si(
        r_jax,
        z_jax,
        jnp.asarray(coil_r),
        jnp.asarray(coil_z),
        jnp.asarray(coil_current),
        _source.MU0_SI,
    )
    coil_wall = psi_coil.reshape(-1)[wall_indices]
    preconditioner = _predictive.build_gs_mg_preconditioner(
        shape,
        r_jax,
        d_r,
        d_z,
    )

    def operator(psi_flat: jnp.ndarray) -> jnp.ndarray:
        return cast(
            jnp.ndarray,
            _predictive._gs_operator_flat(
                psi_flat,
                shape,
                r_jax,
                jnp.asarray(d_r),
                jnp.asarray(d_z),
            ),
        )

    def linear_solve(rhs: jnp.ndarray) -> jnp.ndarray:
        solution, _ = _predictive._bicgstab(
            operator,
            rhs,
            x0=reference_flat,
            tol=_predictive._BICGSTAB_TOL,
            maxiter=_predictive._BICGSTAB_MAXITER,
            M=preconditioner,
        )
        return cast(jnp.ndarray, solution.reshape(shape))

    reference_axis = _predictive.smooth_axis_flux(reference_jax)
    reference_boundary = _predictive.smooth_xpoint_flux(
        reference_jax,
        r_jax,
        z_jax,
        refinement=jnp.asarray(1.0),
    )
    production_current = _predictive._plasma_current(
        reference_jax,
        r_jax,
        reference_axis,
        reference_boundary,
        knots_jax,
        pprime_values,
        ffprime_values,
        jnp.asarray(spec.plasma_current_a),
        jnp.asarray(d_area),
        _source.DEFAULT_CUTOFF_WIDTH,
        _source.MU0_SI,
    )

    def coupled_rhs(current: jnp.ndarray) -> jnp.ndarray:
        source = -(_source.MU0_SI * r_jax[jnp.newaxis, :] * current)
        wall = coil_wall + response @ (current.reshape(-1)[source_indices] * d_area)
        return cast(jnp.ndarray, source.reshape(-1).at[wall_indices].set(wall))

    reference_current_jax = jnp.asarray(reference_current)
    hardwall_rhs = (
        (-(_source.MU0_SI * r_jax[jnp.newaxis, :] * reference_current_jax))
        .reshape(-1)
        .at[wall_indices]
        .set(reference_flat[wall_indices])
    )
    reference_hardwall_map = linear_solve(hardwall_rhs)
    reference_coupled_map = linear_solve(coupled_rhs(reference_current_jax))
    manual_production_map = linear_solve(coupled_rhs(production_current))

    def stationary_map(psi: jnp.ndarray) -> jnp.ndarray:
        mapped = _predictive._coupled_step(
            psi.reshape(-1),
            jnp.asarray(spec.plasma_current_a),
            coil_wall,
            shape,
            r_jax,
            z_jax,
            jnp.asarray(d_r),
            jnp.asarray(d_z),
            jnp.asarray(d_area),
            knots_jax,
            pprime_values,
            ffprime_values,
            response,
            wall_indices,
            source_indices,
            jnp.asarray(1.0),
            _source.DEFAULT_CUTOFF_WIDTH,
            _source.MU0_SI,
            preconditioner,
        )
        return cast(jnp.ndarray, mapped.reshape(shape))

    stationary_map_jit = jax.jit(stationary_map)
    production_map = stationary_map_jit(reference_jax)
    production_map.block_until_ready()
    manual_production_map.block_until_ready()
    reference_coupled_map.block_until_ready()
    reference_hardwall_map.block_until_ready()

    manual_np = np.asarray(manual_production_map, dtype=np.float64)
    production_np = np.asarray(production_map, dtype=np.float64)
    hardwall_np = np.asarray(reference_hardwall_map, dtype=np.float64)
    reference_coupled_np = np.asarray(reference_coupled_map, dtype=np.float64)
    parity_delta = production_np - manual_np
    operator_component = hardwall_np - reference
    boundary_component = reference_coupled_np - hardwall_np
    source_component = manual_np - reference_coupled_np
    total_forcing = manual_np - reference
    closure = total_forcing - operator_component - boundary_component - source_component

    _, linear_map = jax.linearize(stationary_map_jit, reference_jax)
    terminal_jvp = linear_map(jnp.asarray(terminal_error))
    source_jvp = linear_map(jnp.asarray(source_component))
    terminal_jvp.block_until_ready()
    source_jvp.block_until_ready()
    terminal_jvp_np = np.asarray(terminal_jvp, dtype=np.float64)
    source_jvp_np = np.asarray(source_jvp, dtype=np.float64)

    trajectory_planes = [reference]
    trajectory_jax = reference_jax
    for _ in range(contract.TRAJECTORY_STEPS):
        trajectory_jax = stationary_map_jit(trajectory_jax)
        trajectory_jax.block_until_ready()
        trajectory_planes.append(np.asarray(trajectory_jax, dtype=np.float64))
    trajectory = [
        _trajectory_row(
            value,
            step=index,
            reference=reference,
            candidate=candidate,
        )
        for index, value in enumerate(trajectory_planes)
    ]

    source_commit = _git_value("rev-parse", "HEAD") or "0" * 40
    source_clean = _git_value("status", "--porcelain") is None
    environment = _runtime_environment()
    environment["freegs_version"] = freegs_version
    return contract.build_report(
        generated_at=generated_at,
        environment=environment,
        source_artifacts=_source_artifacts(
            spec.example_path,
            source_commit=source_commit,
            clean=source_clean,
        ),
        bindings={
            "same_case": {
                "candidate_psi_sha256": evaluation["digests"]["candidate_psi_sha256"],
                "path": contract.SAME_CASE_PATH,
                "payload_sha256": same_case["payload_sha256"],
                "reference_psi_sha256": evaluation["digests"]["reference_psi_sha256"],
                "source_commit": same_case["source_artifacts"]["repository"]["git_commit"],
            },
            "source_mechanism": {
                "path": contract.SOURCE_MECHANISM_PATH,
                "payload_sha256": mechanism["payload_sha256"],
                "same_case_payload_sha256": mechanism["bindings"]["same_case"]["payload_sha256"],
                "source_commit": mechanism["source_artifacts"]["fusion_repository"]["git_commit"],
            },
        },
        decomposition={
            "closure_max_abs_wb": float(np.max(np.abs(closure))),
            "components": {
                "boundary_anchor": _vector_metrics(
                    boundary_component,
                    terminal_error=terminal_error,
                ),
                "native_operator_residual": _vector_metrics(
                    operator_component,
                    terminal_error=terminal_error,
                ),
                "source_mechanism": _vector_metrics(
                    source_component,
                    terminal_error=terminal_error,
                ),
            },
            "total_forcing": _vector_metrics(
                total_forcing,
                terminal_error=terminal_error,
            ),
        },
        map_parity={
            "manual_map_sha256": _array_sha256(manual_np),
            "max_abs_wb": float(np.max(np.abs(parity_delta))),
            "production_map_sha256": _array_sha256(production_np),
            "relative_l2": float(
                np.linalg.norm(parity_delta) / max(float(np.linalg.norm(manual_np)), 1.0e-30)
            ),
            "sha256_match": _array_sha256(manual_np) == _array_sha256(production_np),
        },
        jvp_gains={
            "source_mechanism": _jvp_metrics(source_component, source_jvp_np),
            "terminal_error": _jvp_metrics(terminal_error, terminal_jvp_np),
        },
        trajectory=trajectory,
        source_worktree_clean=source_clean,
    )


def write_report(
    report: dict[str, Any],
    *,
    json_path: Path = REPORT_PATH,
    markdown_path: Path = MARKDOWN_PATH,
) -> None:
    """Validate and persist JSON and Markdown diagnostic evidence."""
    contract.validate_report(report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(contract.render_markdown(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Execute the diagnostic or validate an existing report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-at")
    parser.add_argument("--same-case-report", type=Path, default=SAME_CASE_PATH)
    parser.add_argument(
        "--source-mechanism-report",
        type=Path,
        default=SOURCE_MECHANISM_PATH,
    )
    parser.add_argument("--json-report", type=Path, default=REPORT_PATH)
    parser.add_argument("--markdown-report", type=Path, default=MARKDOWN_PATH)
    parser.add_argument("--validate-report", type=Path)
    args = parser.parse_args(argv)
    if args.validate_report is not None:
        report = _same_case.load_report(args.validate_report)
        contract.validate_report(report)
        print(report["payload_sha256"])
        return 0
    if not isinstance(args.generated_at, str) or not args.generated_at.strip():
        parser.error("--generated-at is required when executing the diagnostic")
    report = run_diagnostic(
        generated_at=args.generated_at,
        same_case_path=args.same_case_report,
        source_mechanism_path=args.source_mechanism_report,
    )
    write_report(
        report,
        json_path=args.json_report,
        markdown_path=args.markdown_report,
    )
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
