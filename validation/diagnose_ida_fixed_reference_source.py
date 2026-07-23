#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Execute the fixed-reference profile-source ablation for IDA same-case evidence."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

import validation.ida_fixed_reference_source_contract as contract

# The legacy solver and benchmark modules are not in the strict mypy cohort.
# Typed dynamic bindings still execute their exact production implementations.
_same_case = cast(Any, importlib.import_module("validation.benchmark_ida_same_case"))
_freegs_public = cast(
    Any,
    importlib.import_module("validation.benchmark_freegs_public_example_reconstruction"),
)
_predictive = cast(
    Any,
    importlib.import_module("scpn_fusion.core.jax_free_boundary_predictive"),
)
_free_boundary = cast(
    Any,
    importlib.import_module("scpn_fusion.core.jax_free_boundary_gs"),
)

ROOT: Path = _same_case.ROOT
SAME_CASE_REPORT_PATH: Path = _same_case.REPORT_PATH
REPORT_PATH = ROOT / "validation" / "reports" / "ida_fixed_reference_source_ablation.json"
MARKDOWN_PATH = ROOT / "validation" / "reports" / "ida_fixed_reference_source_ablation.md"
_array_sha256: Callable[[object], str] = _same_case._array_sha256
_file_sha256: Callable[[Path], str] = _same_case._file_sha256
_fit_compact_profile: Callable[..., tuple[NDArray[np.float64], NDArray[np.float64]]] = (
    _same_case._fit_compact_profile
)
_git_value: Callable[..., str | None] = _same_case._git_value
_payload_sha256: Callable[[dict[str, Any]], str] = _same_case._payload_sha256
_runtime_environment: Callable[[], dict[str, Any]] = _same_case._runtime_environment
_shape_error_diagnostics: Callable[..., dict[str, Any]] = _same_case._shape_error_diagnostics
load_report: Callable[[str | Path], dict[str, Any]] = _same_case.load_report
validate_same_case_report: Callable[[dict[str, Any]], None] = _same_case.validate_report
PUBLIC_CASES: tuple[Any, ...] = _freegs_public.PUBLIC_CASES
_import_freegs: Callable[[], tuple[ModuleType | None, str | None, str | None]] = (
    _freegs_public._import_freegs
)
_make_equilibrium: Callable[[ModuleType, Any], tuple[Any, Any, Any, Any]] = (
    _freegs_public._make_equilibrium
)
_machine_filaments: Callable[[Any], tuple[Any, list[tuple[float, float, float]]]] = (
    _freegs_public._machine_filaments
)
DEFAULT_CUTOFF_WIDTH: float = _predictive.DEFAULT_CUTOFF_WIDTH
_plasma_current: Callable[..., Any] = _predictive._plasma_current
MU0_SI: float = _free_boundary.MU0_SI
_build_response_matrix: Callable[..., Any] = _same_case.build_response_matrix
_bspline_design_matrix: Callable[..., Any] = _same_case.bspline_design_matrix
_evaluate_profile: Callable[..., Any] = _same_case.evaluate_profile
_solve_compiled: Callable[..., Any] = _same_case.solve_predictive_equilibrium_compiled
_smooth_axis_flux: Callable[..., Any] = _same_case.smooth_axis_flux
_smooth_xpoint_flux: Callable[..., Any] = _same_case.smooth_xpoint_flux

# Re-export the contract surface used by callers and focused tests.
SCHEMA_VERSION = contract.SCHEMA_VERSION
BENCHMARK_ID = contract.BENCHMARK_ID
EVALUATION_CASE_ID = contract.EVALUATION_CASE_ID
PROFILE_COEFFICIENT_COUNT = contract.PROFILE_COEFFICIENT_COUNT
PROFILE_DEGREE = contract.PROFILE_DEGREE
PROFILE_SAMPLE_COUNT = contract.PROFILE_SAMPLE_COUNT
CLAIM_FIELDS = contract.CLAIM_FIELDS
ROUTING_THRESHOLDS = contract.ROUTING_THRESHOLDS
_SOURCE_PATHS = contract.SOURCE_PATHS
_require_number = contract._require_number

FloatArray: TypeAlias = NDArray[np.float64]


def build_report(
    *,
    generated_at: str,
    environment: dict[str, Any],
    source_artifacts: dict[str, dict[str, str]],
    source_same_case: dict[str, Any],
    profile_fit: dict[str, dict[str, Any]],
    fixed_reference_sources: dict[str, dict[str, Any]],
    candidate_anchor_ablation: dict[str, dict[str, Any]],
    self_consistent_candidate: dict[str, Any],
    source_commit: str,
    source_worktree_clean: bool,
) -> dict[str, Any]:
    """Build one report with the exact production cutoff width."""
    return contract.build_report(
        generated_at=generated_at,
        environment=environment,
        source_artifacts=source_artifacts,
        source_same_case=source_same_case,
        profile_fit=profile_fit,
        fixed_reference_sources=fixed_reference_sources,
        candidate_anchor_ablation=candidate_anchor_ablation,
        self_consistent_candidate=self_consistent_candidate,
        source_commit=source_commit,
        source_worktree_clean=source_worktree_clean,
        cutoff_width=DEFAULT_CUTOFF_WIDTH,
    )


def validate_report(report: dict[str, Any]) -> None:
    """Validate one report against the production cutoff contract."""
    contract.validate_report(report, cutoff_width=DEFAULT_CUTOFF_WIDTH)


def render_markdown(report: dict[str, Any]) -> str:
    """Render one report with the exact production cutoff width."""
    return contract.render_markdown(report, cutoff_width=DEFAULT_CUTOFF_WIDTH)


def _profile_fit_metrics(exact: FloatArray, reconstructed: FloatArray) -> dict[str, Any]:
    if exact.shape != reconstructed.shape or exact.ndim != 1 or exact.size < 2:
        raise ValueError("profile fit arrays must be matching non-trivial vectors")
    if not np.all(np.isfinite(exact)) or not np.all(np.isfinite(reconstructed)):
        raise ValueError("profile fit arrays must be finite")
    residual = reconstructed - exact
    return {
        "exact_sha256": _array_sha256(exact),
        "reconstructed_sha256": _array_sha256(reconstructed),
        "relative_l2_error": float(
            np.linalg.norm(residual) / max(float(np.linalg.norm(exact)), 1.0e-30)
        ),
        "relative_max_error": float(
            np.max(np.abs(residual)) / max(float(np.max(np.abs(exact))), 1.0e-30)
        ),
        "sample_count": int(exact.size),
    }


def _current_comparison(
    *,
    candidate_current_density: FloatArray,
    reference_current_density: FloatArray,
    reference_current_mask: NDArray[np.bool_],
    reference_psi: FloatArray,
    reference_axis: float,
    reference_boundary: float,
    r_grid: FloatArray,
    z_grid: FloatArray,
) -> dict[str, Any]:
    reference_psi_n = (reference_psi - reference_axis) / (reference_boundary - reference_axis)
    diagnostics = _shape_error_diagnostics(
        candidate_psi_n=reference_psi_n,
        reference_psi_n=reference_psi_n,
        reference_current_mask=reference_current_mask,
        reference_current_density=reference_current_density,
        candidate_current_density=candidate_current_density,
        r_grid=r_grid,
        z_grid=z_grid,
        candidate_axis=reference_axis,
        candidate_boundary=reference_boundary,
        reference_axis=reference_axis,
        reference_boundary=reference_boundary,
    )
    return {
        "current_density_sha256": _array_sha256(candidate_current_density),
        "distribution": diagnostics["current_distribution"],
        "support": diagnostics["current_support"],
    }


def _source_artifacts(public_example_path: Path) -> dict[str, dict[str, str]]:
    artifacts = {
        name: {"path": path, "sha256": _file_sha256(ROOT / path)}
        for name, path in sorted(_SOURCE_PATHS.items())
    }
    artifacts["freegs_public_example"] = {
        "path": str(public_example_path.relative_to(ROOT)),
        "sha256": _file_sha256(public_example_path),
    }
    return artifacts


def _evaluation_case(same_case: dict[str, Any]) -> dict[str, Any]:
    cases = same_case.get("cases")
    if not isinstance(cases, list):
        raise ValueError("same-case report cases must be a list")
    matches = [
        row
        for row in cases
        if isinstance(row, dict)
        and row.get("role") == "evaluation_candidate"
        and row.get("case_id") == EVALUATION_CASE_ID
    ]
    if len(matches) != 1:
        raise ValueError("same-case report must contain exactly one DIII-D evaluation candidate")
    return cast(dict[str, Any], matches[0])


def _solve_reference(
    same_case_report_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Any, Any, Any, Any, str | None]:
    same_case = load_report(same_case_report_path)
    validate_same_case_report(same_case)
    evaluation = _evaluation_case(same_case)
    if evaluation["grid_shape"] != [129, 129]:
        raise ValueError("same-case DIII-D evaluation must use the frozen 129x129 grid")
    freegs, freegs_version, import_error = _import_freegs()
    if freegs is None:
        raise RuntimeError(f"FreeGS backend unavailable: {import_error}")
    spec = replace(PUBLIC_CASES[1], nx=129, ny=129)
    tokamak, equilibrium, profiles, constrain = _make_equilibrium(freegs, spec)
    freegs.solve(
        equilibrium,
        profiles,
        constrain,
        show=False,
        maxits=spec.nonlinear_attempts[-1][0],
        blend=spec.nonlinear_attempts[-1][1],
        convergenceInfo=False,
    )
    return same_case, evaluation, spec, tokamak, equilibrium, profiles, freegs_version


def _solve_candidate(
    *,
    spec: Any,
    tokamak: Any,
    r_grid: FloatArray,
    z_grid: FloatArray,
    knots: FloatArray,
    pprime_coefficients: FloatArray,
    ffprime_coefficients: FloatArray,
) -> tuple[FloatArray, float, float]:
    _, filaments = _machine_filaments(tokamak)
    coil_r = np.asarray([row[0] for row in filaments], dtype=np.float64)
    coil_z = np.asarray([row[1] for row in filaments], dtype=np.float64)
    coil_current = np.asarray([row[2] for row in filaments], dtype=np.float64)
    basis = _bspline_design_matrix(
        knots,
        n_coeff=PROFILE_COEFFICIENT_COUNT,
        degree=PROFILE_DEGREE,
    )
    r_jax = jnp.asarray(r_grid)
    z_jax = jnp.asarray(z_grid)
    knots_jax = jnp.asarray(knots)
    basis_jax = jnp.asarray(basis)
    response, wall_indices, source_indices = _build_response_matrix(r_jax, z_jax)

    def forward(
        psi_init: Any | None,
        iteration_cap: int,
        ip_ramp: int,
    ) -> tuple[Any, int]:
        result = _solve_compiled(
            jnp.asarray(coil_current),
            _evaluate_profile(jnp.asarray(pprime_coefficients), basis_jax),
            _evaluate_profile(jnp.asarray(ffprime_coefficients), basis_jax),
            r_jax,
            z_jax,
            jnp.asarray(coil_r),
            jnp.asarray(coil_z),
            knots_jax,
            spec.plasma_current_a,
            response,
            wall_indices,
            source_indices,
            psi_init,
            iteration_cap,
            _same_case.DEFAULT_ANDERSON_DEPTH,
            _same_case.DEFAULT_MIXING,
            ip_ramp,
            DEFAULT_CUTOFF_WIDTH,
            _same_case.DEFAULT_TOL,
            MU0_SI,
            return_iterations=True,
        )
        return cast(tuple[Any, int], result)

    cold, _ = forward(None, _same_case.DEFAULT_N_ITER, _same_case.DEFAULT_IP_RAMP)
    cold.block_until_ready()
    candidate, _ = forward(cold, _same_case.WARM_START_ITERATION_CAP, 1)
    candidate.block_until_ready()
    return (
        np.asarray(candidate, dtype=np.float64),
        float(_smooth_axis_flux(candidate)),
        float(_smooth_xpoint_flux(candidate, r_jax, z_jax)),
    )


def run_ablation(
    *,
    generated_at: str,
    same_case_report_path: Path = SAME_CASE_REPORT_PATH,
) -> dict[str, Any]:
    """Execute the fixed-reference DIII-D profile-source ablation."""
    if cast(bool, jax.config.values["jax_enable_x64"]) is not True:
        raise RuntimeError("fixed-reference source ablation requires JAX FP64")
    (
        same_case,
        evaluation,
        spec,
        tokamak,
        equilibrium,
        profiles,
        freegs_version,
    ) = _solve_reference(same_case_report_path)
    r_grid = np.asarray(equilibrium.R_1D, dtype=np.float64)
    z_grid = np.asarray(equilibrium.Z_1D, dtype=np.float64)
    reference_psi = np.asarray(equilibrium.psi(), dtype=np.float64).T
    reference_current = np.asarray(equilibrium.Jtor, dtype=np.float64).T
    reference_mask = np.isfinite(reference_current) & (np.abs(reference_current) > 0.0)
    if not np.any(reference_mask):
        raise RuntimeError("FreeGS reference contains no finite plasma-current support")
    reference_axis = float(equilibrium.psi_axis)
    reference_boundary = float(equilibrium.psi_bndry)
    if not math.isfinite(reference_axis) or not math.isfinite(reference_boundary):
        raise RuntimeError("FreeGS reference axis and boundary must be finite")

    knots = np.linspace(0.0, 1.0, PROFILE_SAMPLE_COUNT, dtype=np.float64)
    pprime_exact = np.asarray(profiles.pprime(knots), dtype=np.float64)
    ffprime_exact = np.asarray(profiles.ffprime(knots), dtype=np.float64)
    pprime_coefficients, pprime_compact = _fit_compact_profile(
        pprime_exact,
        knots,
        n_coefficients=PROFILE_COEFFICIENT_COUNT,
        degree=PROFILE_DEGREE,
    )
    ffprime_coefficients, ffprime_compact = _fit_compact_profile(
        ffprime_exact,
        knots,
        n_coefficients=PROFILE_COEFFICIENT_COUNT,
        degree=PROFILE_DEGREE,
    )
    profile_fit = {
        "ffprime": _profile_fit_metrics(ffprime_exact, ffprime_compact),
        "pprime": _profile_fit_metrics(pprime_exact, pprime_compact),
    }
    d_area = (r_grid[1] - r_grid[0]) * (z_grid[1] - z_grid[0])

    def source(
        psi: FloatArray,
        pprime: FloatArray,
        ffprime: FloatArray,
        axis: float,
        boundary: float,
    ) -> FloatArray:
        current = _plasma_current(
            jnp.asarray(psi),
            jnp.asarray(r_grid),
            jnp.asarray(axis),
            jnp.asarray(boundary),
            jnp.asarray(knots),
            jnp.asarray(pprime),
            jnp.asarray(ffprime),
            jnp.asarray(spec.plasma_current_a),
            jnp.asarray(d_area),
            DEFAULT_CUTOFF_WIDTH,
            MU0_SI,
        )
        return np.asarray(current, dtype=np.float64)

    fixed_sources = {
        "compact_bspline": _current_comparison(
            candidate_current_density=source(
                reference_psi,
                pprime_compact,
                ffprime_compact,
                reference_axis,
                reference_boundary,
            ),
            reference_current_density=reference_current,
            reference_current_mask=reference_mask,
            reference_psi=reference_psi,
            reference_axis=reference_axis,
            reference_boundary=reference_boundary,
            r_grid=r_grid,
            z_grid=z_grid,
        ),
        "exact_sampled": _current_comparison(
            candidate_current_density=source(
                reference_psi,
                pprime_exact,
                ffprime_exact,
                reference_axis,
                reference_boundary,
            ),
            reference_current_density=reference_current,
            reference_current_mask=reference_mask,
            reference_psi=reference_psi,
            reference_axis=reference_axis,
            reference_boundary=reference_boundary,
            r_grid=r_grid,
            z_grid=z_grid,
        ),
    }
    candidate_psi, candidate_axis, candidate_boundary = _solve_candidate(
        spec=spec,
        tokamak=tokamak,
        r_grid=r_grid,
        z_grid=z_grid,
        knots=knots,
        pprime_coefficients=pprime_coefficients,
        ffprime_coefficients=ffprime_coefficients,
    )

    def anchor_row(axis: float, boundary: float) -> dict[str, Any]:
        return _current_comparison(
            candidate_current_density=source(
                candidate_psi,
                pprime_compact,
                ffprime_compact,
                axis,
                boundary,
            ),
            reference_current_density=reference_current,
            reference_current_mask=reference_mask,
            reference_psi=reference_psi,
            reference_axis=reference_axis,
            reference_boundary=reference_boundary,
            r_grid=r_grid,
            z_grid=z_grid,
        )

    anchor_ablation = {
        "candidate_axis_reference_boundary": anchor_row(
            candidate_axis,
            reference_boundary,
        ),
        "production_smooth_anchors": anchor_row(candidate_axis, candidate_boundary),
        "reference_axis_candidate_boundary": anchor_row(
            reference_axis,
            candidate_boundary,
        ),
        "reference_axis_reference_boundary": anchor_row(
            reference_axis,
            reference_boundary,
        ),
    }
    self_consistent = {
        "candidate_psi_sha256": evaluation["digests"]["candidate_psi_sha256"],
        "distribution": evaluation["shape_diagnostics"]["current_distribution"],
        "source_report_payload_sha256": same_case["payload_sha256"],
        "support": evaluation["shape_diagnostics"]["current_support"],
    }
    environment = _runtime_environment()
    environment["freegs_version"] = freegs_version
    return build_report(
        generated_at=generated_at,
        environment=environment,
        source_artifacts=_source_artifacts(spec.example_path),
        source_same_case={
            "case_id": evaluation["case_id"],
            "grid_shape": evaluation["grid_shape"],
            "path": str(same_case_report_path.relative_to(ROOT)),
            "payload_sha256": same_case["payload_sha256"],
            "source_commit": same_case["source_artifacts"]["repository"]["git_commit"],
        },
        profile_fit=profile_fit,
        fixed_reference_sources=fixed_sources,
        candidate_anchor_ablation=anchor_ablation,
        self_consistent_candidate=self_consistent,
        source_commit=_git_value("rev-parse", "HEAD") or "0" * 40,
        source_worktree_clean=_git_value("status", "--porcelain") is None,
    )


def write_report(
    report: dict[str, Any],
    *,
    json_path: Path = REPORT_PATH,
    markdown_path: Path = MARKDOWN_PATH,
) -> None:
    """Write validated JSON and Markdown evidence."""
    validate_report(report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Execute the ablation or validate an existing report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-at")
    parser.add_argument("--same-case-report", type=Path, default=SAME_CASE_REPORT_PATH)
    parser.add_argument("--json-out", type=Path, default=REPORT_PATH)
    parser.add_argument("--markdown-out", type=Path, default=MARKDOWN_PATH)
    parser.add_argument("--validate-report", type=Path)
    args = parser.parse_args(argv)
    if args.validate_report is not None:
        report = load_report(args.validate_report)
        validate_report(report)
        print(report["payload_sha256"])
        return 0
    if not isinstance(args.generated_at, str) or not args.generated_at.strip():
        parser.error("--generated-at is required unless --validate-report is used")
    report = run_ablation(
        generated_at=args.generated_at,
        same_case_report_path=args.same_case_report,
    )
    write_report(report, json_path=args.json_out, markdown_path=args.markdown_out)
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
