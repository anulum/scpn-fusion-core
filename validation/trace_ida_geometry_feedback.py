#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Trace geometry/source feedback inside the frozen IDA compiled solve."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Callable, cast

import jax
import jax.numpy as jnp
import numpy as np

import validation.ida_geometry_feedback_trace_contract as contract
import validation.ida_geometry_feedback_trace_metrics as metrics

_same_case = cast(Any, importlib.import_module("validation.benchmark_ida_same_case"))
_ablation = cast(
    Any,
    importlib.import_module("validation.diagnose_ida_fixed_reference_source"),
)
_predictive = cast(
    Any,
    importlib.import_module("scpn_fusion.core.jax_predictive_forward_compiled"),
)
_checkpoint_trace = cast(
    Any,
    importlib.import_module("scpn_fusion.core.jax_predictive_checkpoint_trace"),
)

ROOT: Path = _same_case.ROOT
REPORT_PATH = ROOT / "validation" / "reports" / "ida_geometry_feedback_trace.json"
MARKDOWN_PATH = ROOT / "validation" / "reports" / "ida_geometry_feedback_trace.md"
_array_sha256: Callable[[object], str] = _same_case._array_sha256
_file_sha256: Callable[[Path], str] = _same_case._file_sha256
_git_value: Callable[..., str | None] = _same_case._git_value
_load_report: Callable[[str | Path], dict[str, Any]] = _same_case.load_report
_runtime_environment: Callable[[], dict[str, Any]] = _same_case._runtime_environment


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


def run_trace(
    *,
    generated_at: str,
    same_case_report_path: Path = ROOT / contract.SAME_CASE_REPORT_PATH,
    source_ablation_report_path: Path = ROOT / contract.SOURCE_ABLATION_REPORT_PATH,
) -> dict[str, Any]:
    """Execute the frozen cold-plus-warm compiled trace at 129 square."""
    if cast(bool, jax.config.values["jax_enable_x64"]) is not True:
        raise RuntimeError("geometry-feedback trace requires JAX FP64")
    (
        same_case,
        evaluation,
        spec,
        tokamak,
        equilibrium,
        profiles,
        freegs_version,
    ) = _ablation._solve_reference(same_case_report_path)
    source_ablation = _load_report(source_ablation_report_path)
    _ablation.validate_report(source_ablation)

    r_grid = np.asarray(equilibrium.R_1D, dtype=np.float64)
    z_grid = np.asarray(equilibrium.Z_1D, dtype=np.float64)
    reference_psi = np.asarray(equilibrium.psi(), dtype=np.float64).T
    reference_current = np.asarray(equilibrium.Jtor, dtype=np.float64).T
    reference_mask = np.isfinite(reference_current) & (np.abs(reference_current) > 0.0)
    reference_axis = float(equilibrium.psi_axis)
    reference_boundary = float(equilibrium.psi_bndry)
    reference_span = reference_boundary - reference_axis
    if not np.any(reference_mask) or abs(reference_span) <= 1.0e-30:
        raise RuntimeError("FreeGS reference current support and flux span must be non-zero")

    _, filaments = _ablation._machine_filaments(tokamak)
    coil_r = np.asarray([row[0] for row in filaments], dtype=np.float64)
    coil_z = np.asarray([row[1] for row in filaments], dtype=np.float64)
    coil_current = np.asarray([row[2] for row in filaments], dtype=np.float64)
    knots = np.linspace(0.0, 1.0, 129, dtype=np.float64)
    pprime_exact = np.asarray(profiles.pprime(knots), dtype=np.float64)
    ffprime_exact = np.asarray(profiles.ffprime(knots), dtype=np.float64)
    pprime_coefficients, _ = _ablation._fit_compact_profile(
        pprime_exact,
        knots,
        n_coefficients=12,
        degree=3,
    )
    ffprime_coefficients, _ = _ablation._fit_compact_profile(
        ffprime_exact,
        knots,
        n_coefficients=12,
        degree=3,
    )
    basis = _ablation._bspline_design_matrix(knots, n_coeff=12, degree=3)
    pprime_values = np.asarray(
        _ablation._evaluate_profile(jnp.asarray(pprime_coefficients), jnp.asarray(basis)),
        dtype=np.float64,
    )
    ffprime_values = np.asarray(
        _ablation._evaluate_profile(jnp.asarray(ffprime_coefficients), jnp.asarray(basis)),
        dtype=np.float64,
    )
    r_jax = jnp.asarray(r_grid)
    z_jax = jnp.asarray(z_grid)
    response, wall_indices, source_indices = _ablation._build_response_matrix(r_jax, z_jax)

    def solve(
        *,
        psi_init: jnp.ndarray | None,
        iteration_cap: int,
        ip_ramp: int,
        indices: tuple[int, ...],
    ) -> Any:
        result = _predictive.solve_predictive_equilibrium_compiled(
            jnp.asarray(coil_current),
            jnp.asarray(pprime_values),
            jnp.asarray(ffprime_values),
            r_jax,
            z_jax,
            jnp.asarray(coil_r),
            jnp.asarray(coil_z),
            jnp.asarray(knots),
            spec.plasma_current_a,
            response,
            wall_indices,
            source_indices,
            psi_init,
            iteration_cap,
            _same_case.DEFAULT_ANDERSON_DEPTH,
            _same_case.DEFAULT_MIXING,
            ip_ramp,
            _same_case.DEFAULT_CUTOFF_WIDTH,
            _same_case.DEFAULT_TOL,
            _ablation.MU0_SI,
            trace_iteration_indices=indices,
            return_trace=True,
        )
        if not isinstance(result, _checkpoint_trace.CompiledPredictiveTrace):
            raise TypeError("compiled trace solve returned the wrong contract")
        result.equilibrium.block_until_ready()
        return result

    cold = solve(
        psi_init=None,
        iteration_cap=180,
        ip_ramp=30,
        indices=contract.COLD_CHECKPOINT_INDICES,
    )
    warm = solve(
        psi_init=cold.equilibrium,
        iteration_cap=20,
        ip_ramp=1,
        indices=contract.WARM_CHECKPOINT_INDICES,
    )
    d_area = float((r_grid[1] - r_grid[0]) * (z_grid[1] - z_grid[0]))
    context = {
        "current": {
            "d_area": d_area,
            "ffprime_values": ffprime_values,
            "knots": knots,
            "pprime_values": pprime_values,
            "r_grid": r_grid,
            "reference_axis": reference_axis,
            "reference_boundary": reference_boundary,
            "reference_current": reference_current,
            "reference_mask": reference_mask,
            "reference_psi_n": (reference_psi - reference_axis) / reference_span,
            "z_grid": z_grid,
        },
        "ip_target": float(spec.plasma_current_a),
        "r_jax": r_jax,
        "reference_boundary": reference_boundary,
        "residual": {
            "coil_current": coil_current,
            "coil_r": coil_r,
            "coil_z": coil_z,
            "ffprime_values": ffprime_values,
            "knots": knots,
            "pprime_values": pprime_values,
            "r_grid": r_grid,
            "response": response,
            "source_indices": source_indices,
            "wall_indices": wall_indices,
            "z_grid": z_grid,
        },
        "z_jax": z_jax,
    }
    runs = {
        "cold": metrics.trace_run(
            run_name="cold",
            trace=cold,
            requested_indices=contract.COLD_CHECKPOINT_INDICES,
            iteration_cap=180,
            context=context,
        ),
        "warm": metrics.trace_run(
            run_name="warm",
            trace=warm,
            requested_indices=contract.WARM_CHECKPOINT_INDICES,
            iteration_cap=20,
            context=context,
        ),
    }
    expected_digest = str(evaluation["digests"]["candidate_psi_sha256"])
    traced_digest = _array_sha256(warm.equilibrium)
    environment = _runtime_environment()
    environment["freegs_version"] = freegs_version
    return contract.build_report(
        generated_at=generated_at,
        environment=environment,
        source_artifacts=_source_artifacts(spec.example_path),
        source_reports={
            "same_case": {
                "path": contract.SAME_CASE_REPORT_PATH,
                "payload_sha256": same_case["payload_sha256"],
                "source_commit": same_case["source_artifacts"]["repository"]["git_commit"],
            },
            "source_ablation": {
                "path": contract.SOURCE_ABLATION_REPORT_PATH,
                "payload_sha256": source_ablation["payload_sha256"],
                "source_commit": source_ablation["source_artifacts"]["repository"]["git_commit"],
            },
        },
        runs=runs,
        terminal_parity={
            "expected_same_case_candidate_sha256": expected_digest,
            "matches_same_case_candidate": traced_digest == expected_digest,
            "traced_candidate_sha256": traced_digest,
        },
        source_commit=_git_value("rev-parse", "HEAD") or "0" * 40,
        source_worktree_clean=_git_value("status", "--porcelain") is None,
    )


def write_report(
    report: dict[str, Any],
    *,
    json_path: Path = REPORT_PATH,
    markdown_path: Path = MARKDOWN_PATH,
) -> None:
    """Write validated JSON and Markdown trace evidence."""
    contract.validate_report(report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(contract.render_markdown(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Execute the trace or validate one existing report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-at")
    parser.add_argument(
        "--same-case-report", type=Path, default=ROOT / contract.SAME_CASE_REPORT_PATH
    )
    parser.add_argument(
        "--source-ablation-report",
        type=Path,
        default=ROOT / contract.SOURCE_ABLATION_REPORT_PATH,
    )
    parser.add_argument("--json-out", type=Path, default=REPORT_PATH)
    parser.add_argument("--markdown-out", type=Path, default=MARKDOWN_PATH)
    parser.add_argument("--validate-report", type=Path)
    args = parser.parse_args(argv)
    if args.validate_report is not None:
        report = _load_report(args.validate_report)
        contract.validate_report(report)
        print(report["payload_sha256"])
        return 0
    if not isinstance(args.generated_at, str) or not args.generated_at.strip():
        parser.error("--generated-at is required unless --validate-report is used")
    report = run_trace(
        generated_at=args.generated_at,
        same_case_report_path=args.same_case_report,
        source_ablation_report_path=args.source_ablation_report,
    )
    write_report(report, json_path=args.json_out, markdown_path=args.markdown_out)
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
