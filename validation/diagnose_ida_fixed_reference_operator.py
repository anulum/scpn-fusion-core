#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Decompose the fixed FreeGS reference under native GS and wall operators."""

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

import validation.ida_fixed_reference_operator_contract as contract

_ablation = cast(
    Any,
    importlib.import_module("validation.diagnose_ida_fixed_reference_source"),
)
_same_case = cast(Any, importlib.import_module("validation.benchmark_ida_same_case"))
_predictive = cast(
    Any,
    importlib.import_module("scpn_fusion.core.jax_free_boundary_predictive"),
)

ROOT: Path = _same_case.ROOT
SAME_CASE_REPORT_PATH: Path = ROOT / contract.SAME_CASE_PATH
SOURCE_ABLATION_REPORT_PATH: Path = ROOT / contract.SOURCE_ABLATION_PATH
REPORT_PATH = ROOT / "validation" / "reports" / "ida_fixed_reference_operator_residual.json"
MARKDOWN_PATH = ROOT / "validation" / "reports" / "ida_fixed_reference_operator_residual.md"
FloatArray: TypeAlias = NDArray[np.float64]
_array_sha256: Callable[[object], str] = _same_case._array_sha256
_file_sha256: Callable[[Path], str] = _same_case._file_sha256
_git_value: Callable[..., str | None] = _same_case._git_value
_runtime_environment: Callable[[], dict[str, Any]] = _same_case._runtime_environment
_fit_compact_profile: Callable[..., tuple[FloatArray, FloatArray]] = _same_case._fit_compact_profile
_build_response_matrix: Callable[..., Any] = _same_case.build_response_matrix
_machine_filaments: Callable[..., Any] = _ablation._machine_filaments
_plasma_current: Callable[..., Any] = _predictive._plasma_current
_laplacian_star: Callable[..., Any] = _predictive._laplacian_star
_vacuum_field_si: Callable[..., Any] = _predictive.vacuum_field_si

# Re-export the contract surface used by focused callers.
SCHEMA_VERSION = contract.SCHEMA_VERSION
BENCHMARK_ID = contract.BENCHMARK_ID
EVALUATION_CASE_ID = contract.EVALUATION_CASE_ID


def _finite_2d(value: object, *, field: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must be a finite non-trivial 2D array")
    return array


def _metric(field: object, *, reference_scale: object) -> dict[str, Any]:
    """Summarise one field against an explicitly supplied same-unit scale."""
    array = np.asarray(field, dtype=np.float64)
    scale = np.asarray(reference_scale, dtype=np.float64)
    if (
        array.size == 0
        or array.shape != scale.shape
        or not np.all(np.isfinite(array))
        or not np.all(np.isfinite(scale))
    ):
        raise ValueError("metric field and reference scale must be matching finite arrays")
    denominator = max(float(np.linalg.norm(scale)), 1.0e-30)
    return {
        "field_sha256": _array_sha256(array),
        "linf": float(np.max(np.abs(array))),
        "relative_l2_to_reference_scale": float(np.linalg.norm(array)) / denominator,
        "rms": float(np.sqrt(np.mean(np.square(array)))),
    }


def _interior(value: FloatArray) -> FloatArray:
    if value.ndim != 2 or min(value.shape) < 3:
        raise ValueError("interior extraction requires a non-trivial 2D array")
    return np.asarray(value[1:-1, 1:-1], dtype=np.float64)


def _masked_interior(value: FloatArray, mask: NDArray[np.bool_]) -> FloatArray:
    interior = _interior(value)
    if mask.shape != interior.shape or not np.any(mask):
        raise ValueError("interior mask must match the interior and select points")
    return np.asarray(interior[mask], dtype=np.float64)


def _source_field(
    *,
    current: FloatArray,
    r_grid: FloatArray,
    mu0: float,
) -> FloatArray:
    if current.ndim != 2 or current.shape[0] != r_grid.size:
        raise ValueError("current first dimension must match r_grid")
    return np.asarray(-(mu0 * r_grid[:, np.newaxis] * current), dtype=np.float64)


def _closure_max_abs(actual: FloatArray, components: tuple[FloatArray, ...]) -> float:
    reconstructed = np.zeros_like(actual)
    for component in components:
        if component.shape != actual.shape:
            raise ValueError("closure components must have matching shapes")
        reconstructed = reconstructed + component
    return float(np.max(np.abs(actual - reconstructed)))


def _source_artifacts(public_example_path: Path) -> dict[str, dict[str, str]]:
    artifacts = {
        name: {"path": path, "sha256": _file_sha256(ROOT / path)}
        for name, path in sorted(contract.SOURCE_PATHS.items())
    }
    artifacts["freegs_public_example"] = {
        "path": str(public_example_path.relative_to(ROOT)),
        "sha256": _file_sha256(public_example_path),
    }
    return artifacts


def _load_bound_reports(
    same_case_report_path: Path,
    source_ablation_report_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    same_case = _same_case.load_report(same_case_report_path)
    _same_case.validate_report(same_case)
    ablation = _ablation.load_report(source_ablation_report_path)
    _ablation.validate_report(ablation)
    if ablation["source_same_case"]["payload_sha256"] != same_case["payload_sha256"]:
        raise ValueError("source ablation does not bind the selected same-case payload")
    return cast(dict[str, Any], same_case), cast(dict[str, Any], ablation)


def _freegs_fourth_order_lhs(
    *,
    freegs: Any,
    equilibrium: Any,
    plasma_psi_rz: FloatArray,
) -> FloatArray:
    generator = freegs.gradshafranov.GSsparse4thOrder(
        float(equilibrium.Rmin),
        float(equilibrium.Rmax),
        float(equilibrium.Zmin),
        float(equilibrium.Zmax),
    )
    matrix = generator(int(equilibrium.nx), int(equilibrium.ny))
    lhs = matrix @ plasma_psi_rz.reshape(-1)
    return np.asarray(lhs.reshape(plasma_psi_rz.shape), dtype=np.float64)


def _native_lhs(
    psi_rz: FloatArray,
    *,
    r_grid: FloatArray,
    z_grid: FloatArray,
) -> FloatArray:
    # Native arrays use (Z,R); transpose only at this explicit boundary.
    psi_zr = np.asarray(psi_rz.T, dtype=np.float64)
    lhs_zr = np.asarray(
        _laplacian_star(
            jnp.asarray(psi_zr),
            jnp.asarray(r_grid),
            jnp.asarray(r_grid[1] - r_grid[0]),
            jnp.asarray(z_grid[1] - z_grid[0]),
        ),
        dtype=np.float64,
    )
    return np.asarray(lhs_zr.T, dtype=np.float64)


def run_diagnostic(
    *,
    generated_at: str,
    same_case_report_path: Path = SAME_CASE_REPORT_PATH,
    source_ablation_report_path: Path = SOURCE_ABLATION_REPORT_PATH,
) -> dict[str, Any]:
    """Execute the exact fixed-reference interior and wall decomposition."""
    if cast(bool, jax.config.values["jax_enable_x64"]) is not True:
        raise RuntimeError("fixed-reference operator diagnostic requires JAX FP64")
    same_case, ablation = _load_bound_reports(
        same_case_report_path,
        source_ablation_report_path,
    )
    (
        _,
        evaluation,
        spec,
        tokamak,
        equilibrium,
        profiles,
        freegs_version,
    ) = _ablation._solve_reference(same_case_report_path)
    freegs, _, import_error = _ablation._import_freegs()
    if freegs is None:
        raise RuntimeError(f"FreeGS backend unavailable: {import_error}")

    r_grid = np.asarray(equilibrium.R_1D, dtype=np.float64)
    z_grid = np.asarray(equilibrium.Z_1D, dtype=np.float64)
    total_psi = _finite_2d(equilibrium.psi(), field="FreeGS total psi")
    plasma_psi = _finite_2d(equilibrium.plasma_psi, field="FreeGS plasma psi")
    reference_current = _finite_2d(equilibrium.Jtor, field="FreeGS Jtor")
    if list(total_psi.shape) != contract.GRID_SHAPE:
        raise ValueError("FreeGS reference must use the frozen 129x129 grid")

    knots = np.linspace(
        0.0,
        1.0,
        _ablation.PROFILE_SAMPLE_COUNT,
        dtype=np.float64,
    )
    pprime_exact = np.asarray(profiles.pprime(knots), dtype=np.float64)
    ffprime_exact = np.asarray(profiles.ffprime(knots), dtype=np.float64)
    _, pprime_compact = _fit_compact_profile(
        pprime_exact,
        knots,
        n_coefficients=_ablation.PROFILE_COEFFICIENT_COUNT,
        degree=_ablation.PROFILE_DEGREE,
    )
    _, ffprime_compact = _fit_compact_profile(
        ffprime_exact,
        knots,
        n_coefficients=_ablation.PROFILE_COEFFICIENT_COUNT,
        degree=_ablation.PROFILE_DEGREE,
    )
    d_area = float((r_grid[1] - r_grid[0]) * (z_grid[1] - z_grid[0]))

    def native_current(pprime: FloatArray, ffprime: FloatArray) -> FloatArray:
        current_zr = _plasma_current(
            jnp.asarray(total_psi.T),
            jnp.asarray(r_grid),
            jnp.asarray(float(equilibrium.psi_axis)),
            jnp.asarray(float(equilibrium.psi_bndry)),
            jnp.asarray(knots),
            jnp.asarray(pprime),
            jnp.asarray(ffprime),
            jnp.asarray(spec.plasma_current_a),
            jnp.asarray(d_area),
            _ablation.DEFAULT_CUTOFF_WIDTH,
            _ablation.MU0_SI,
        )
        return np.asarray(current_zr, dtype=np.float64).T

    exact_current = native_current(pprime_exact, ffprime_exact)
    compact_current = native_current(pprime_compact, ffprime_compact)
    reference_rhs = _source_field(
        current=reference_current,
        r_grid=r_grid,
        mu0=_ablation.MU0_SI,
    )
    exact_rhs = _source_field(
        current=exact_current,
        r_grid=r_grid,
        mu0=_ablation.MU0_SI,
    )
    compact_rhs = _source_field(
        current=compact_current,
        r_grid=r_grid,
        mu0=_ablation.MU0_SI,
    )

    freegs_lhs = _freegs_fourth_order_lhs(
        freegs=freegs,
        equilibrium=equilibrium,
        plasma_psi_rz=plasma_psi,
    )
    native_plasma_lhs = _native_lhs(plasma_psi, r_grid=r_grid, z_grid=z_grid)
    native_total_lhs = _native_lhs(total_psi, r_grid=r_grid, z_grid=z_grid)
    support_mask = np.asarray(
        np.abs(reference_current[1:-1, 1:-1]) > 0.0,
        dtype=np.bool_,
    )
    reference_scale = _masked_interior(reference_rhs, support_mask)
    freegs_baseline = _masked_interior(freegs_lhs - reference_rhs, support_mask)
    second_order_component = _masked_interior(
        native_plasma_lhs - freegs_lhs,
        support_mask,
    )
    all_interior_vacuum = _interior(native_total_lhs - native_plasma_lhs)
    vacuum_component = np.asarray(all_interior_vacuum[support_mask], dtype=np.float64)
    exact_source_component = _masked_interior(
        reference_rhs - exact_rhs,
        support_mask,
    )
    compact_source_component = _masked_interior(
        reference_rhs - compact_rhs,
        support_mask,
    )
    exact_residual = _masked_interior(native_total_lhs - exact_rhs, support_mask)
    compact_residual = _masked_interior(native_total_lhs - compact_rhs, support_mask)

    interior_components = {
        "exact_source_convention": _metric(
            exact_source_component,
            reference_scale=reference_scale,
        ),
        "freegs_fourth_order_baseline": _metric(
            freegs_baseline,
            reference_scale=reference_scale,
        ),
        "second_order_operator": _metric(
            second_order_component,
            reference_scale=reference_scale,
        ),
        "vacuum_discretisation": _metric(
            vacuum_component,
            reference_scale=reference_scale,
        ),
    }
    operator_residuals = {
        "freegs_fourth_order_reference_current_plasma_flux": _metric(
            freegs_baseline,
            reference_scale=reference_scale,
        ),
        "native_second_order_compact_source_total_flux": _metric(
            compact_residual,
            reference_scale=reference_scale,
        ),
        "native_second_order_exact_source_total_flux": _metric(
            exact_residual,
            reference_scale=reference_scale,
        ),
        "native_second_order_reference_current_plasma_flux": _metric(
            _masked_interior(native_plasma_lhs - reference_rhs, support_mask),
            reference_scale=reference_scale,
        ),
        "native_second_order_reference_current_total_flux": _metric(
            _masked_interior(native_total_lhs - reference_rhs, support_mask),
            reference_scale=reference_scale,
        ),
    }
    all_reference_scale = _interior(reference_rhs)
    outside_support = np.asarray(
        all_interior_vacuum[~support_mask],
        dtype=np.float64,
    )
    _, filaments = _machine_filaments(tokamak)
    coil_region_diagnostic = {
        "all_interior_vacuum_field": _metric(
            all_interior_vacuum,
            reference_scale=all_reference_scale,
        ),
        "coil_filament_count": len(filaments),
        "coil_filaments_inside_domain": sum(
            int(
                float(spec.r_min) < float(r_filament) < float(spec.r_max)
                and float(spec.z_min) < float(z_filament) < float(spec.z_max)
            )
            for r_filament, z_filament, _ in filaments
        ),
        "outside_reference_support_l2_fraction": float(
            np.linalg.norm(outside_support)
            / max(float(np.linalg.norm(all_interior_vacuum)), 1.0e-30)
        ),
        "reference_plasma_support_fraction": float(
            np.count_nonzero(support_mask) / support_mask.size
        ),
        "reference_plasma_support_point_count": int(np.count_nonzero(support_mask)),
        "reference_plasma_support_vacuum_field": _metric(
            vacuum_component,
            reference_scale=reference_scale,
        ),
    }

    coil_r = np.asarray([row[0] for row in filaments], dtype=np.float64)
    coil_z = np.asarray([row[1] for row in filaments], dtype=np.float64)
    coil_current = np.asarray([row[2] for row in filaments], dtype=np.float64)
    r_jax = jnp.asarray(r_grid)
    z_jax = jnp.asarray(z_grid)
    response, wall_indices_jax, source_indices_jax = _build_response_matrix(
        r_jax,
        z_jax,
    )
    wall_indices = np.asarray(wall_indices_jax, dtype=np.int64)
    source_indices = np.asarray(source_indices_jax, dtype=np.int64)
    response_np = np.asarray(response, dtype=np.float64)
    native_coil_zr = np.asarray(
        _vacuum_field_si(
            r_jax,
            z_jax,
            jnp.asarray(coil_r),
            jnp.asarray(coil_z),
            jnp.asarray(coil_current),
            _ablation.MU0_SI,
        ),
        dtype=np.float64,
    )
    reference_coil_zr = np.asarray((total_psi - plasma_psi).T, dtype=np.float64)
    reference_plasma_zr = np.asarray(plasma_psi.T, dtype=np.float64)

    def wall_response(current_rz: FloatArray) -> FloatArray:
        current_zr = np.asarray(current_rz.T, dtype=np.float64)
        return np.asarray(
            response_np @ (current_zr.reshape(-1)[source_indices] * d_area),
            dtype=np.float64,
        )

    reference_response = wall_response(reference_current)
    exact_response = wall_response(exact_current)
    compact_response = wall_response(compact_current)
    reference_coil_wall = reference_coil_zr.reshape(-1)[wall_indices]
    native_coil_wall = native_coil_zr.reshape(-1)[wall_indices]
    reference_plasma_wall = reference_plasma_zr.reshape(-1)[wall_indices]
    total_wall_scale = total_psi.T.reshape(-1)[wall_indices]
    coil_wall_component = reference_coil_wall - native_coil_wall
    plasma_wall_component = reference_plasma_wall - reference_response
    exact_wall_source_component = reference_response - exact_response
    compact_wall_source_component = reference_response - compact_response
    exact_wall_residual = total_wall_scale - (native_coil_wall + exact_response)
    compact_wall_residual = total_wall_scale - (native_coil_wall + compact_response)
    reference_wall_residual = total_wall_scale - (native_coil_wall + reference_response)

    wall_components = {
        "coil_vacuum_convention": _metric(
            coil_wall_component,
            reference_scale=total_wall_scale,
        ),
        "exact_source_convention": _metric(
            exact_wall_source_component,
            reference_scale=total_wall_scale,
        ),
        "plasma_response_quadrature": _metric(
            plasma_wall_component,
            reference_scale=total_wall_scale,
        ),
    }
    wall_residuals = {
        "compact_source_total_flux": _metric(
            compact_wall_residual,
            reference_scale=total_wall_scale,
        ),
        "exact_source_total_flux": _metric(
            exact_wall_residual,
            reference_scale=total_wall_scale,
        ),
        "reference_current_total_flux": _metric(
            reference_wall_residual,
            reference_scale=total_wall_scale,
        ),
    }
    closure = {
        "interior_compact_max_abs": _closure_max_abs(
            compact_residual,
            (
                freegs_baseline,
                second_order_component,
                vacuum_component,
                compact_source_component,
            ),
        ),
        "interior_exact_max_abs": _closure_max_abs(
            exact_residual,
            (
                freegs_baseline,
                second_order_component,
                vacuum_component,
                exact_source_component,
            ),
        ),
        "wall_compact_max_abs": _closure_max_abs(
            compact_wall_residual,
            (
                coil_wall_component,
                plasma_wall_component,
                compact_wall_source_component,
            ),
        ),
        "wall_exact_max_abs": _closure_max_abs(
            exact_wall_residual,
            (
                coil_wall_component,
                plasma_wall_component,
                exact_wall_source_component,
            ),
        ),
    }
    if any(not math.isfinite(value) for value in closure.values()):
        raise RuntimeError("operator decomposition closure must be finite")

    environment = _runtime_environment()
    environment["freegs_version"] = freegs_version
    return contract.build_report(
        generated_at=generated_at,
        environment=environment,
        source_artifacts=_source_artifacts(spec.example_path),
        source_same_case={
            "case_id": evaluation["case_id"],
            "grid_shape": evaluation["grid_shape"],
            "path": contract.SAME_CASE_PATH,
            "payload_sha256": same_case["payload_sha256"],
            "source_commit": same_case["source_artifacts"]["repository"]["git_commit"],
        },
        source_ablation={
            "path": contract.SOURCE_ABLATION_PATH,
            "payload_sha256": ablation["payload_sha256"],
            "source_commit": ablation["source_artifacts"]["repository"]["git_commit"],
            "source_same_case_payload_sha256": ablation["source_same_case"]["payload_sha256"],
        },
        operator_residuals=operator_residuals,
        interior_components=interior_components,
        wall_residuals=wall_residuals,
        wall_components=wall_components,
        closure=closure,
        coil_region_diagnostic=coil_region_diagnostic,
        source_commit=_git_value("rev-parse", "HEAD") or "0" * 40,
        source_worktree_clean=_git_value("status", "--porcelain") is None,
    )


def write_report(
    report: dict[str, Any],
    *,
    json_path: Path = REPORT_PATH,
    markdown_path: Path = MARKDOWN_PATH,
) -> None:
    """Validate and persist the JSON and human-readable evidence surfaces."""
    contract.validate_report(report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(contract.render_markdown(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the diagnostic or validate an existing report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-at")
    parser.add_argument("--same-case-report", type=Path, default=SAME_CASE_REPORT_PATH)
    parser.add_argument(
        "--source-ablation-report",
        type=Path,
        default=SOURCE_ABLATION_REPORT_PATH,
    )
    parser.add_argument("--json-report", type=Path, default=REPORT_PATH)
    parser.add_argument("--markdown-report", type=Path, default=MARKDOWN_PATH)
    parser.add_argument("--validate-report", type=Path)
    args = parser.parse_args(argv)
    if args.validate_report is not None:
        report = json.loads(args.validate_report.read_text(encoding="utf-8"))
        contract.validate_report(report)
        print(json.dumps(report["routing"], indent=2, sort_keys=True))
        return 0
    if args.generated_at is None or not args.generated_at.strip():
        parser.error("--generated-at is required when executing the diagnostic")
    report = run_diagnostic(
        generated_at=args.generated_at,
        same_case_report_path=args.same_case_report,
        source_ablation_report_path=args.source_ablation_report,
    )
    write_report(
        report,
        json_path=args.json_report,
        markdown_path=args.markdown_report,
    )
    print(json.dumps(report["routing"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
