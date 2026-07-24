#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Map the bound IDA operator decomposition through the native linear inverse."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Callable, cast

import jax
import jax.numpy as jnp
import numpy as np

import validation.ida_operator_response_contract as contract
from validation.ida_operator_response_fields import (
    closure_max_abs as _closure_max_abs,
)
from validation.ida_operator_response_fields import finite_plane as _finite_plane
from validation.ida_operator_response_fields import forcing_metric as _forcing_metric
from validation.ida_operator_response_fields import native_inverse as _native_inverse
from validation.ida_operator_response_fields import (
    operator_components as _operator_components,
)
from validation.ida_operator_response_fields import sum_fields as _sum_fields
from validation.ida_operator_response_fields import (
    verify_operator_binding as _verify_operator_binding,
)

_same_case = cast(Any, importlib.import_module("validation.benchmark_ida_same_case"))
_source = cast(
    Any,
    importlib.import_module("validation.diagnose_ida_fixed_reference_source"),
)
_operator = cast(
    Any,
    importlib.import_module("validation.diagnose_ida_fixed_reference_operator"),
)
_operator_contract = cast(
    Any,
    importlib.import_module("validation.ida_fixed_reference_operator_contract"),
)
_mechanism = cast(
    Any,
    importlib.import_module("validation.diagnose_ida_fixed_reference_source_mechanism"),
)
_fixed_point = cast(
    Any,
    importlib.import_module("validation.diagnose_ida_fixed_point_stability"),
)
_fixed_point_contract = cast(
    Any,
    importlib.import_module("validation.ida_fixed_point_stability_contract"),
)
_predictive = cast(
    Any,
    importlib.import_module("scpn_fusion.core.jax_free_boundary_predictive"),
)

ROOT: Path = _same_case.ROOT
SAME_CASE_PATH = ROOT / contract.SAME_CASE_PATH
OPERATOR_PATH = ROOT / contract.OPERATOR_DECOMPOSITION_PATH
SOURCE_MECHANISM_PATH = ROOT / contract.SOURCE_MECHANISM_PATH
FIXED_POINT_PATH = ROOT / contract.FIXED_POINT_PATH
REPORT_PATH = ROOT / "validation" / "reports" / "ida_operator_response.json"
MARKDOWN_PATH = ROOT / "validation" / "reports" / "ida_operator_response.md"
_array_sha256: Callable[[object], str] = _same_case._array_sha256
_file_sha256: Callable[[Path], str] = _same_case._file_sha256
_git_value: Callable[..., str | None] = _same_case._git_value
_runtime_environment: Callable[[], dict[str, Any]] = _same_case._runtime_environment


def _load_bound_reports(
    *,
    same_case_path: Path,
    operator_path: Path,
    source_mechanism_path: Path,
    fixed_point_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load and verify the complete same-case-to-fixed-point evidence chain."""
    same_case = _same_case.load_report(same_case_path)
    _same_case.validate_report(same_case)
    operator = _same_case.load_report(operator_path)
    _operator_contract.validate_report(operator)
    mechanism = _same_case.load_report(source_mechanism_path)
    _mechanism.validate_report(mechanism)
    fixed_point = _same_case.load_report(fixed_point_path)
    _fixed_point_contract.validate_report(fixed_point)
    same_digest = same_case["payload_sha256"]
    if operator["source_same_case"]["payload_sha256"] != same_digest:
        raise ValueError("operator decomposition does not bind the selected same-case payload")
    if (
        mechanism["bindings"]["same_case"]["payload_sha256"] != same_digest
        or mechanism["bindings"]["operator_decomposition"]["payload_sha256"]
        != operator["payload_sha256"]
    ):
        raise ValueError("source mechanism does not bind the selected evidence chain")
    if (
        fixed_point["bindings"]["same_case"]["payload_sha256"] != same_digest
        or fixed_point["bindings"]["source_mechanism"]["payload_sha256"]
        != mechanism["payload_sha256"]
    ):
        raise ValueError("fixed point does not bind the selected evidence chain")
    return (
        cast(dict[str, Any], same_case),
        cast(dict[str, Any], operator),
        cast(dict[str, Any], mechanism),
        cast(dict[str, Any], fixed_point),
    )


def _source_artifacts(
    public_example_path: Path,
    *,
    source_commit: str,
    clean: bool,
) -> dict[str, dict[str, Any]]:
    """Bind every executed repository source to bytes and clean Git state."""
    artifacts: dict[str, dict[str, Any]] = {
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


def run_diagnostic(
    *,
    generated_at: str,
    same_case_path: Path = SAME_CASE_PATH,
    operator_path: Path = OPERATOR_PATH,
    source_mechanism_path: Path = SOURCE_MECHANISM_PATH,
    fixed_point_path: Path = FIXED_POINT_PATH,
) -> dict[str, Any]:
    """Execute the frozen four-component native-inverse response diagnostic."""
    if cast(bool, jax.config.values["jax_enable_x64"]) is not True:
        raise RuntimeError("operator-response diagnostic requires JAX FP64")
    same_case, operator_report, mechanism, fixed_point_report = _load_bound_reports(
        same_case_path=same_case_path,
        operator_path=operator_path,
        source_mechanism_path=source_mechanism_path,
        fixed_point_path=fixed_point_path,
    )
    (
        _,
        evaluation,
        spec,
        tokamak,
        equilibrium,
        profiles,
        freegs_version,
    ) = _source._solve_reference(same_case_path)
    freegs, _, import_error = _source._import_freegs()
    if freegs is None:
        raise RuntimeError(f"FreeGS backend unavailable: {import_error}")
    runtime_artifacts = _operator._source_artifacts(
        freegs=freegs,
        evaluation=evaluation,
        public_example_path=spec.example_path,
    )
    for name in ("freegs_boundary", "freegs_operator", "freegs_public_example"):
        if runtime_artifacts[name] != operator_report["source_artifacts"][name]:
            raise ValueError(f"runtime {name} bytes disagree with operator evidence")

    r_grid = np.asarray(equilibrium.R_1D, dtype=np.float64)
    z_grid = np.asarray(equilibrium.Z_1D, dtype=np.float64)
    reference_rz = _finite_plane(equilibrium.psi(), field="FreeGS total psi")
    reference_zr = np.asarray(reference_rz.T, dtype=np.float64)
    reference_current_rz = _finite_plane(equilibrium.Jtor, field="FreeGS Jtor")
    if list(reference_rz.shape) != contract.GRID_SHAPE:
        raise ValueError("FreeGS reference must use the frozen 129x129 grid")
    if _array_sha256(reference_zr) != evaluation["digests"]["reference_psi_sha256"]:
        raise ValueError("reconstructed reference disagrees with same-case evidence")

    knots = np.linspace(0.0, 1.0, _source.PROFILE_SAMPLE_COUNT, dtype=np.float64)
    pprime_exact = np.asarray(profiles.pprime(knots), dtype=np.float64)
    ffprime_exact = np.asarray(profiles.ffprime(knots), dtype=np.float64)
    pprime_coefficients, _ = _source._fit_compact_profile(
        pprime_exact,
        knots,
        n_coefficients=_source.PROFILE_COEFFICIENT_COUNT,
        degree=_source.PROFILE_DEGREE,
    )
    ffprime_coefficients, _ = _source._fit_compact_profile(
        ffprime_exact,
        knots,
        n_coefficients=_source.PROFILE_COEFFICIENT_COUNT,
        degree=_source.PROFILE_DEGREE,
    )
    candidate_zr, _, _ = _source._solve_candidate(
        spec=spec,
        tokamak=tokamak,
        r_grid=r_grid,
        z_grid=z_grid,
        knots=knots,
        pprime_coefficients=pprime_coefficients,
        ffprime_coefficients=ffprime_coefficients,
    )
    candidate_zr = _finite_plane(candidate_zr, field="same-case candidate")
    if _array_sha256(candidate_zr) != evaluation["digests"]["candidate_psi_sha256"]:
        raise ValueError("reconstructed candidate disagrees with same-case evidence")
    terminal_error = candidate_zr - reference_zr
    if float(np.linalg.norm(terminal_error)) <= 1.0e-30:
        raise RuntimeError("same-case terminal error is degenerate")

    d_r = float(r_grid[1] - r_grid[0])
    d_z = float(z_grid[1] - z_grid[0])
    d_area = d_r * d_z
    exact_current_zr = _predictive._plasma_current(
        jnp.asarray(reference_zr),
        jnp.asarray(r_grid),
        jnp.asarray(float(equilibrium.psi_axis)),
        jnp.asarray(float(equilibrium.psi_bndry)),
        jnp.asarray(knots),
        jnp.asarray(pprime_exact),
        jnp.asarray(ffprime_exact),
        jnp.asarray(spec.plasma_current_a),
        jnp.asarray(d_area),
        _source.DEFAULT_CUTOFF_WIDTH,
        _source.MU0_SI,
    )
    exact_current_rz = np.asarray(exact_current_zr, dtype=np.float64).T
    components_rz = _operator_components(
        freegs=freegs,
        equilibrium=equilibrium,
        reference_current_rz=reference_current_rz,
        exact_current_rz=exact_current_rz,
        r_grid=r_grid,
        z_grid=z_grid,
        mu0=_source.MU0_SI,
    )
    _verify_operator_binding(
        components_rz,
        reference_current_rz=reference_current_rz,
        operator_report=operator_report,
    )
    components_zr = {
        name: np.asarray(field.T, dtype=np.float64) for name, field in components_rz.items()
    }
    native_residual = _sum_fields(
        tuple(components_zr[name] for name in contract.NATIVE_OPERATOR_COMPONENTS)
    )
    exact_residual = _sum_fields(tuple(components_zr[name] for name in contract.COMPONENTS))

    preconditioner = _predictive.build_gs_mg_preconditioner(
        reference_zr.shape,
        jnp.asarray(r_grid),
        d_r,
        d_z,
    )
    zero = np.zeros_like(reference_zr)
    responses = {
        name: -_native_inverse(
            field,
            r_grid=r_grid,
            d_r=d_r,
            d_z=d_z,
            preconditioner=preconditioner,
            x0_zr=zero,
        )
        for name, field in components_zr.items()
    }
    native_response = -_native_inverse(
        native_residual,
        r_grid=r_grid,
        d_r=d_r,
        d_z=d_z,
        preconditioner=preconditioner,
        x0_zr=zero,
    )
    exact_response = -_native_inverse(
        exact_residual,
        r_grid=r_grid,
        d_r=d_r,
        d_z=d_z,
        preconditioner=preconditioner,
        x0_zr=zero,
    )

    reference_current_zr = np.asarray(reference_current_rz.T, dtype=np.float64)
    hardwall_rhs = (
        -(_source.MU0_SI * jnp.asarray(r_grid)[jnp.newaxis, :] * reference_current_zr)
    ).reshape(-1)
    _, wall_indices, _ = _source._build_response_matrix(
        jnp.asarray(r_grid),
        jnp.asarray(z_grid),
    )
    hardwall_rhs = hardwall_rhs.at[wall_indices].set(
        jnp.asarray(reference_zr).reshape(-1)[wall_indices]
    )
    hardwall_map = _native_inverse(
        np.asarray(hardwall_rhs.reshape(reference_zr.shape), dtype=np.float64),
        r_grid=r_grid,
        d_r=d_r,
        d_z=d_z,
        preconditioner=preconditioner,
        x0_zr=reference_zr,
    )
    fixed_point_native = hardwall_map - reference_zr
    fixed_point_metric = _fixed_point._vector_metrics(
        fixed_point_native,
        terminal_error=terminal_error,
    )
    expected_fixed_point = fixed_point_report["decomposition"]["components"][
        "native_operator_residual"
    ]
    if fixed_point_metric["field_sha256"] != expected_fixed_point["field_sha256"]:
        raise ValueError("native operator response disagrees with fixed-point evidence")

    native_response_components = tuple(
        responses[name] for name in contract.NATIVE_OPERATOR_COMPONENTS
    )
    exact_response_components = tuple(responses[name] for name in contract.COMPONENTS)
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
            "operator_decomposition": {
                "path": contract.OPERATOR_DECOMPOSITION_PATH,
                "payload_sha256": operator_report["payload_sha256"],
                "same_case_payload_sha256": operator_report["source_same_case"]["payload_sha256"],
                "source_commit": operator_report["source_artifacts"]["repository"]["git_commit"],
            },
            "source_mechanism": {
                "operator_payload_sha256": mechanism["bindings"]["operator_decomposition"][
                    "payload_sha256"
                ],
                "path": contract.SOURCE_MECHANISM_PATH,
                "payload_sha256": mechanism["payload_sha256"],
                "same_case_payload_sha256": mechanism["bindings"]["same_case"]["payload_sha256"],
                "source_commit": mechanism["source_artifacts"]["fusion_repository"]["git_commit"],
            },
            "fixed_point": {
                "path": contract.FIXED_POINT_PATH,
                "payload_sha256": fixed_point_report["payload_sha256"],
                "same_case_payload_sha256": fixed_point_report["bindings"]["same_case"][
                    "payload_sha256"
                ],
                "source_commit": fixed_point_report["source_artifacts"]["repository"]["git_commit"],
                "source_mechanism_payload_sha256": fixed_point_report["bindings"][
                    "source_mechanism"
                ]["payload_sha256"],
            },
        },
        forcing_decomposition={
            "components": {
                name: _forcing_metric(field, exact_residual=exact_residual)
                for name, field in components_zr.items()
            },
            "exact_source_residual": _forcing_metric(
                exact_residual,
                exact_residual=exact_residual,
            ),
            "native_operator_residual": _forcing_metric(
                native_residual,
                exact_residual=exact_residual,
            ),
        },
        response_decomposition={
            "components": {
                name: _fixed_point._vector_metrics(
                    field,
                    terminal_error=terminal_error,
                )
                for name, field in responses.items()
            },
            "exact_source_total": _fixed_point._vector_metrics(
                exact_response,
                terminal_error=terminal_error,
            ),
            "native_operator_total": _fixed_point._vector_metrics(
                native_response,
                terminal_error=terminal_error,
            ),
        },
        closure={
            "exact_source_forcing_max_abs": _closure_max_abs(
                exact_residual,
                tuple(components_zr[name] for name in contract.COMPONENTS),
            ),
            "exact_source_response_max_abs_wb": _closure_max_abs(
                exact_response,
                exact_response_components,
            ),
            "fixed_point_native_operator_max_abs_wb": float(
                np.max(np.abs(native_response - fixed_point_native))
            ),
            "native_operator_forcing_max_abs": _closure_max_abs(
                native_residual,
                tuple(components_zr[name] for name in contract.NATIVE_OPERATOR_COMPONENTS),
            ),
            "native_operator_response_max_abs_wb": _closure_max_abs(
                native_response,
                native_response_components,
            ),
        },
    )


def write_report(
    report: dict[str, Any],
    *,
    json_path: Path = REPORT_PATH,
    markdown_path: Path = MARKDOWN_PATH,
) -> None:
    """Validate and write the JSON and Markdown evidence pair."""
    contract.validate_report(report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(contract.render_markdown(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the diagnostic or validate an existing immutable report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-at")
    parser.add_argument("--same-case-report", type=Path, default=SAME_CASE_PATH)
    parser.add_argument("--operator-report", type=Path, default=OPERATOR_PATH)
    parser.add_argument(
        "--source-mechanism-report",
        type=Path,
        default=SOURCE_MECHANISM_PATH,
    )
    parser.add_argument("--fixed-point-report", type=Path, default=FIXED_POINT_PATH)
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
        operator_path=args.operator_report,
        source_mechanism_path=args.source_mechanism_report,
        fixed_point_path=args.fixed_point_report,
    )
    write_report(
        report,
        json_path=args.json_report,
        markdown_path=args.markdown_report,
    )
    print(json.dumps(report["routing"], allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
