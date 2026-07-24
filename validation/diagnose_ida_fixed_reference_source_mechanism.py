#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Decompose fixed-reference hard mask, smooth cutoff, and Ip normalisation."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

import validation.ida_fixed_reference_source_mechanism_contract as contract

_same_case = cast(Any, importlib.import_module("validation.benchmark_ida_same_case"))
_source = cast(
    Any,
    importlib.import_module("validation.diagnose_ida_fixed_reference_source"),
)
_source_contract = cast(
    Any,
    importlib.import_module("validation.ida_fixed_reference_source_contract"),
)
_operator = cast(
    Any,
    importlib.import_module("validation.diagnose_ida_fixed_reference_operator"),
)
_operator_contract = cast(
    Any,
    importlib.import_module("validation.ida_fixed_reference_operator_contract"),
)
_predictive = cast(
    Any,
    importlib.import_module("scpn_fusion.core.jax_free_boundary_predictive"),
)
_free_boundary = cast(
    Any,
    importlib.import_module("scpn_fusion.core.jax_free_boundary_gs"),
)
_plasma_support = cast(
    Any,
    importlib.import_module("scpn_fusion.core.jax_plasma_support"),
)

ROOT: Path = _same_case.ROOT
CONTROL_ROOT = ROOT.parent / "SCPN-CONTROL"
SAME_CASE_REPORT_PATH = ROOT / contract.SAME_CASE_PATH
SOURCE_ABLATION_REPORT_PATH = ROOT / contract.SOURCE_ABLATION_PATH
OPERATOR_DECOMPOSITION_REPORT_PATH = ROOT / contract.OPERATOR_DECOMPOSITION_PATH
REPORT_PATH = ROOT / "validation" / "reports" / "ida_fixed_reference_source_mechanism.json"
MARKDOWN_PATH = ROOT / "validation" / "reports" / "ida_fixed_reference_source_mechanism.md"

SCHEMA_VERSION = contract.SCHEMA_VERSION
BENCHMARK_ID = contract.BENCHMARK_ID
CURRENT_FIELDS = contract.CURRENT_FIELDS
MECHANISM_COMPONENTS = contract.MECHANISM_COMPONENTS

_array_sha256: Callable[[object], str] = _same_case._array_sha256
_file_sha256: Callable[[Path], str] = _same_case._file_sha256
_payload_sha256: Callable[[dict[str, Any]], str] = _same_case._payload_sha256
_runtime_environment: Callable[[], dict[str, Any]] = _same_case._runtime_environment
_build_response_matrix: Callable[..., Any] = _same_case.build_response_matrix
_plasma_current: Callable[..., Any] = _predictive._plasma_current
DEFAULT_CUTOFF_WIDTH: float = _predictive.DEFAULT_CUTOFF_WIDTH
MU0_SI: float = _free_boundary.MU0_SI

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


def _git_value(repository: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _control_module() -> ModuleType:
    source_root = CONTROL_ROOT / "src"
    if not source_root.is_dir():
        raise RuntimeError(f"canonical CONTROL source root is unavailable: {source_root}")
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    module = importlib.import_module("scpn_control.core.gs_profile_source")
    module_path = Path(cast(str, module.__file__)).resolve()
    expected = (source_root / "scpn_control" / "core" / "gs_profile_source.py").resolve()
    if module_path != expected:
        raise RuntimeError(f"CONTROL module resolved to {module_path}, expected {expected}")
    return module


def _load_bound_reports(
    *,
    same_case_path: Path,
    source_ablation_path: Path,
    operator_decomposition_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    same_case = _same_case.load_report(same_case_path)
    _same_case.validate_report(same_case)
    source_ablation = _same_case.load_report(source_ablation_path)
    _source_contract.validate_report(
        source_ablation,
        cutoff_width=DEFAULT_CUTOFF_WIDTH,
    )
    operator = _same_case.load_report(operator_decomposition_path)
    _operator_contract.validate_report(operator)
    if source_ablation["source_same_case"]["payload_sha256"] != same_case["payload_sha256"]:
        raise ValueError("source ablation is not bound to the selected same-case payload")
    if operator["source_same_case"]["payload_sha256"] != same_case["payload_sha256"]:
        raise ValueError("operator decomposition is not bound to the selected same-case payload")
    if operator["source_ablation"]["payload_sha256"] != source_ablation["payload_sha256"]:
        raise ValueError("operator decomposition is not bound to the selected source ablation")
    return same_case, source_ablation, operator


def _current_metrics(
    current: FloatArray,
    *,
    reference: FloatArray,
    reference_mask: BoolArray,
    target_current_a: float,
    d_area: float,
) -> dict[str, Any]:
    if current.shape != reference.shape or current.shape != reference_mask.shape:
        raise ValueError("current fields and reference mask must have matching shapes")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(reference)):
        raise ValueError("current fields must be finite")
    reference_norm = max(float(np.linalg.norm(reference)), 1.0e-30)
    candidate_abs = np.abs(current)
    reference_abs = np.abs(reference)
    candidate_total = max(float(np.sum(candidate_abs)), 1.0e-30)
    reference_total = max(float(np.sum(reference_abs)), 1.0e-30)
    candidate_distribution = candidate_abs / candidate_total
    reference_distribution = reference_abs / reference_total
    relative_floor = float(_same_case.CURRENT_SUPPORT_RELATIVE_FLOOR)
    support = candidate_abs >= (float(np.max(candidate_abs)) * relative_floor)
    rectangular_current = float(np.sum(current) * d_area)
    return {
        "absolute_current_outside_reference_fraction": float(
            np.sum(candidate_abs[~reference_mask]) / candidate_total
        ),
        "candidate_support_point_count": int(np.count_nonzero(support)),
        "current_density_sha256": _array_sha256(current),
        "rectangular_current_a": rectangular_current,
        "reference_support_point_count": int(np.count_nonzero(reference_mask)),
        "relative_ip_error": abs(rectangular_current - target_current_a)
        / max(abs(target_current_a), 1.0),
        "relative_l2_to_reference": float(np.linalg.norm(current - reference) / reference_norm),
        "total_variation_distance": float(
            0.5 * np.sum(np.abs(candidate_distribution - reference_distribution))
        ),
    }


def _vector_metrics(field: FloatArray, *, reference_scale: FloatArray) -> dict[str, Any]:
    return cast(dict[str, Any], _operator._metric(field, reference_scale=reference_scale))


def _closure_max_abs(actual: FloatArray, components: tuple[FloatArray, ...]) -> float:
    return float(np.max(np.abs(actual - np.sum(np.stack(components), axis=0))))


def _mechanism_flux_fields(
    psi: jnp.ndarray,
    psi_axis: jnp.ndarray,
    psi_boundary: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return topology-distance and profile-interpolation flux fields."""
    topology_flux = _free_boundary.normalised_flux_unclipped(
        psi,
        psi_axis,
        psi_boundary,
    )
    profile_flux = jnp.clip(topology_flux, 0.0, 1.0)
    return topology_flux, profile_flux


def _source_artifacts(public_example_path: Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for name, relative_path in contract.SOURCE_PATHS.items():
        artifacts[name] = {
            "path": relative_path,
            "sha256": _file_sha256((ROOT / relative_path).resolve()),
        }
    artifacts["freegs_public_example"] = {
        "path": str(public_example_path.relative_to(ROOT)),
        "sha256": _file_sha256(public_example_path),
    }
    fusion_commit = _git_value(ROOT, "rev-parse", "HEAD")
    control_commit = _git_value(CONTROL_ROOT, "rev-parse", "HEAD")
    if fusion_commit is None or control_commit is None:
        raise RuntimeError("both FUSION and CONTROL repositories must resolve a Git HEAD")
    artifacts["fusion_repository"] = {
        "git_commit": fusion_commit,
        "path": ".",
        "worktree_clean": _git_value(ROOT, "status", "--porcelain") is None,
    }
    artifacts["control_repository"] = {
        "git_commit": control_commit,
        "path": contract.CONTROL_REPOSITORY_PATH,
        "worktree_clean": _git_value(CONTROL_ROOT, "status", "--porcelain") is None,
    }
    return artifacts


def _bindings(
    same_case: dict[str, Any],
    source_ablation: dict[str, Any],
    operator: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    evaluation = _source._evaluation_case(same_case)
    return {
        "operator_decomposition": {
            "path": contract.OPERATOR_DECOMPOSITION_PATH,
            "payload_sha256": operator["payload_sha256"],
            "source_ablation_payload_sha256": operator["source_ablation"]["payload_sha256"],
            "source_commit": operator["source_artifacts"]["repository"]["git_commit"],
            "source_same_case_payload_sha256": operator["source_same_case"]["payload_sha256"],
        },
        "same_case": {
            "case_id": evaluation["case_id"],
            "grid_shape": evaluation["grid_shape"],
            "path": contract.SAME_CASE_PATH,
            "payload_sha256": same_case["payload_sha256"],
            "source_commit": same_case["source_artifacts"]["repository"]["git_commit"],
        },
        "source_ablation": {
            "path": contract.SOURCE_ABLATION_PATH,
            "payload_sha256": source_ablation["payload_sha256"],
            "source_commit": source_ablation["source_artifacts"]["repository"]["git_commit"],
            "source_same_case_payload_sha256": source_ablation["source_same_case"][
                "payload_sha256"
            ],
        },
    }


def run_diagnostic(
    *,
    generated_at: str,
    same_case_report_path: Path = SAME_CASE_REPORT_PATH,
    source_ablation_report_path: Path = SOURCE_ABLATION_REPORT_PATH,
    operator_decomposition_report_path: Path = OPERATOR_DECOMPOSITION_REPORT_PATH,
) -> dict[str, Any]:
    """Execute the four-field source-mechanism decomposition on fixed FreeGS psi."""
    if cast(bool, jax.config.values["jax_enable_x64"]) is not True:
        raise RuntimeError("fixed-reference source-mechanism diagnostic requires JAX FP64")
    same_case, source_ablation, operator = _load_bound_reports(
        same_case_path=same_case_report_path,
        source_ablation_path=source_ablation_report_path,
        operator_decomposition_path=operator_decomposition_report_path,
    )
    (
        _,
        _,
        spec,
        tokamak,
        equilibrium,
        profiles,
        freegs_version,
    ) = _source._solve_reference(same_case_report_path)

    r_grid = np.asarray(equilibrium.R_1D, dtype=np.float64)
    z_grid = np.asarray(equilibrium.Z_1D, dtype=np.float64)
    psi_rz = np.asarray(equilibrium.psi(), dtype=np.float64)
    reference_current = np.asarray(equilibrium.Jtor, dtype=np.float64)
    if (
        list(reference_current.shape) != contract.GRID_SHAPE
        or psi_rz.shape != reference_current.shape
    ):
        raise ValueError("FreeGS reference must use matching frozen 129x129 fields")
    reference_mask = np.isfinite(reference_current) & (np.abs(reference_current) > 0.0)
    if not np.any(reference_mask):
        raise RuntimeError("FreeGS reference contains no plasma-current support")
    axis = float(equilibrium.psi_axis)
    boundary = float(equilibrium.psi_bndry)
    if not math.isfinite(axis) or not math.isfinite(boundary):
        raise RuntimeError("FreeGS reference anchors must be finite")
    d_r = float(r_grid[1] - r_grid[0])
    d_z = float(z_grid[1] - z_grid[0])
    d_area = d_r * d_z
    target_current = float(spec.plasma_current_a)
    knots = np.linspace(0.0, 1.0, _source.PROFILE_SAMPLE_COUNT, dtype=np.float64)
    pprime = np.asarray(profiles.pprime(knots), dtype=np.float64)
    ffprime = np.asarray(profiles.ffprime(knots), dtype=np.float64)
    rr_rz, _ = np.meshgrid(r_grid, z_grid, indexing="ij")

    control_module = _control_module()
    control_current = np.asarray(
        control_module.update_plasma_source_nonlinear(
            psi_rz,
            rr_rz,
            d_r,
            d_z,
            axis,
            boundary,
            mu0=MU0_SI,
            I_target=target_current,
            profile_mode="external",
            ped_params_p={},
            ped_params_ff={},
            ext_psi_grid=knots,
            ext_pprime=pprime,
            ext_ffprime=ffprime,
        ),
        dtype=np.float64,
    )
    raw_psi_n = (psi_rz - axis) / (boundary - axis)
    hard_mask = (raw_psi_n >= 0.0) & (raw_psi_n < 1.0)
    p_profile = np.interp(np.clip(raw_psi_n.ravel(), 0.0, 1.0), knots, pprime).reshape(
        raw_psi_n.shape
    )
    ff_profile = np.interp(
        np.clip(raw_psi_n.ravel(), 0.0, 1.0),
        knots,
        ffprime,
    ).reshape(raw_psi_n.shape)
    p_profile[~hard_mask] = 0.0
    ff_profile[~hard_mask] = 0.0
    control_formula = rr_rz * p_profile + ff_profile / (MU0_SI * rr_rz)
    control_ip = float(np.sum(control_formula) * d_area)
    if abs(control_ip) <= 1.0e-9:
        control_formula = np.zeros_like(control_formula)
    else:
        control_formula = control_formula * (target_current / control_ip)
    control_delta = control_current - control_formula
    control_relative_l2 = float(
        np.linalg.norm(control_delta) / max(float(np.linalg.norm(control_formula)), 1.0e-30)
    )
    control_metrics = _current_metrics(
        control_current,
        reference=reference_current,
        reference_mask=reference_mask,
        target_current_a=target_current,
        d_area=d_area,
    )
    reference_rectangular_current = float(np.sum(reference_current) * d_area)
    if abs(reference_rectangular_current) <= 1.0e-9:
        raise RuntimeError("FreeGS reference rectangular current is degenerate")
    hard_rectangular = reference_current * (target_current / reference_rectangular_current)

    psi_zr = jnp.asarray(psi_rz.T)
    r_jax = jnp.asarray(r_grid)
    topology_psi_n_zr, profile_psi_n_zr = _mechanism_flux_fields(
        psi_zr,
        jnp.asarray(axis),
        jnp.asarray(boundary),
    )
    r_safe = jnp.maximum(r_jax[jnp.newaxis, :], 1.0e-6)
    raw_zr = r_safe * jnp.interp(
        profile_psi_n_zr,
        jnp.asarray(knots),
        jnp.asarray(pprime),
    )
    raw_zr = raw_zr + jnp.interp(
        profile_psi_n_zr,
        jnp.asarray(knots),
        jnp.asarray(ffprime),
    ) / (MU0_SI * r_safe)
    smooth_support = _plasma_support.soft_axis_connected_support(
        psi_zr,
        topology_psi_n_zr,
        DEFAULT_CUTOFF_WIDTH,
    )
    smooth_unscaled = np.asarray(raw_zr * smooth_support, dtype=np.float64).T
    smooth_normalised = np.asarray(
        _plasma_current(
            psi_zr,
            r_jax,
            jnp.asarray(axis),
            jnp.asarray(boundary),
            jnp.asarray(knots),
            jnp.asarray(pprime),
            jnp.asarray(ffprime),
            jnp.asarray(target_current),
            jnp.asarray(d_area),
            DEFAULT_CUTOFF_WIDTH,
            MU0_SI,
        ),
        dtype=np.float64,
    ).T

    fields = {
        "freegs_hard_romberg": reference_current,
        "freegs_hard_rectangular_normalised": hard_rectangular,
        "fusion_smooth_unscaled": smooth_unscaled,
        "fusion_smooth_rectangular_normalised": smooth_normalised,
    }
    current_fields = {
        name: _current_metrics(
            field,
            reference=reference_current,
            reference_mask=reference_mask,
            target_current_a=target_current,
            d_area=d_area,
        )
        for name, field in fields.items()
    }
    current_components = {
        "hard_rectangular_normalisation": hard_rectangular - reference_current,
        "smooth_cutoff": smooth_unscaled - hard_rectangular,
        "smooth_ip_normalisation": smooth_normalised - smooth_unscaled,
    }
    current_vectors = {
        name: _vector_metrics(component, reference_scale=reference_current)
        for name, component in current_components.items()
    }

    reference_rhs = _operator._source_field(
        current=reference_current,
        r_grid=r_grid,
        mu0=MU0_SI,
    )
    support_interior = np.asarray(reference_mask[1:-1, 1:-1], dtype=np.bool_)
    reference_source_scale = _operator._masked_interior(reference_rhs, support_interior)
    source_components = {
        name: _operator._masked_interior(
            _operator._source_field(current=component, r_grid=r_grid, mu0=MU0_SI),
            support_interior,
        )
        for name, component in current_components.items()
    }
    interior_source_vectors = {
        name: _vector_metrics(component, reference_scale=reference_source_scale)
        for name, component in source_components.items()
    }

    response, wall_indices_jax, source_indices_jax = _build_response_matrix(
        r_jax,
        jnp.asarray(z_grid),
    )
    response_np = np.asarray(response, dtype=np.float64)
    source_indices = np.asarray(source_indices_jax, dtype=np.int64)
    del wall_indices_jax

    def wall_response(current_rz: FloatArray) -> FloatArray:
        current_zr = np.asarray(current_rz.T, dtype=np.float64)
        return np.asarray(
            response_np @ (current_zr.reshape(-1)[source_indices] * d_area),
            dtype=np.float64,
        )

    reference_wall = wall_response(reference_current)
    wall_components = {
        name: wall_response(component) for name, component in current_components.items()
    }
    wall_response_vectors = {
        name: _vector_metrics(component, reference_scale=reference_wall)
        for name, component in wall_components.items()
    }

    current_actual = smooth_normalised - reference_current
    source_actual = _operator._masked_interior(
        _operator._source_field(current=current_actual, r_grid=r_grid, mu0=MU0_SI),
        support_interior,
    )
    wall_actual = wall_response(smooth_normalised) - reference_wall
    closure = {
        "current_max_abs_a_per_m2": _closure_max_abs(
            current_actual,
            tuple(current_components[name] for name in MECHANISM_COMPONENTS),
        ),
        "interior_source_max_abs": _closure_max_abs(
            source_actual,
            tuple(source_components[name] for name in MECHANISM_COMPONENTS),
        ),
        "wall_response_max_abs_wb": _closure_max_abs(
            wall_actual,
            tuple(wall_components[name] for name in MECHANISM_COMPONENTS),
        ),
    }
    environment = _runtime_environment()
    environment["freegs_version"] = freegs_version
    return contract.build_report(
        generated_at=generated_at,
        environment=environment,
        source_artifacts=_source_artifacts(spec.example_path),
        bindings=_bindings(same_case, source_ablation, operator),
        current_fields=current_fields,
        current_vectors=current_vectors,
        interior_source_vectors=interior_source_vectors,
        wall_response_vectors=wall_response_vectors,
        control_parity={
            "absolute_current_outside_freegs_support_fraction": control_metrics[
                "absolute_current_outside_reference_fraction"
            ],
            "actual_current_density_sha256": _array_sha256(control_current),
            "formula_current_density_sha256": _array_sha256(control_formula),
            "hard_mask_sha256": _array_sha256(hard_mask),
            "max_abs_a_per_m2": float(np.max(np.abs(control_delta))),
            "rectangular_current_a": control_metrics["rectangular_current_a"],
            "relative_l2": control_relative_l2,
            "relative_l2_to_freegs_reference": control_metrics["relative_l2_to_reference"],
            "total_variation_to_freegs_reference": control_metrics["total_variation_distance"],
        },
        closure=closure,
        cutoff_width=DEFAULT_CUTOFF_WIDTH,
    )


def validate_report(report: dict[str, Any]) -> None:
    contract.validate_report(report, cutoff_width=DEFAULT_CUTOFF_WIDTH)


def render_markdown(report: dict[str, Any]) -> str:
    return contract.render_markdown(report, cutoff_width=DEFAULT_CUTOFF_WIDTH)


def write_report(
    report: dict[str, Any],
    *,
    json_path: Path = REPORT_PATH,
    markdown_path: Path = MARKDOWN_PATH,
) -> None:
    """Validate and persist JSON and Markdown evidence."""
    validate_report(report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Execute the diagnostic or validate an existing payload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-at")
    parser.add_argument("--same-case-report", type=Path, default=SAME_CASE_REPORT_PATH)
    parser.add_argument(
        "--source-ablation-report",
        type=Path,
        default=SOURCE_ABLATION_REPORT_PATH,
    )
    parser.add_argument(
        "--operator-decomposition-report",
        type=Path,
        default=OPERATOR_DECOMPOSITION_REPORT_PATH,
    )
    parser.add_argument("--json-report", type=Path, default=REPORT_PATH)
    parser.add_argument("--markdown-report", type=Path, default=MARKDOWN_PATH)
    parser.add_argument("--validate-report", type=Path)
    args = parser.parse_args(argv)
    if args.validate_report is not None:
        report = json.loads(args.validate_report.read_text(encoding="utf-8"))
        validate_report(report)
        print(report["payload_sha256"])
        return 0
    if not isinstance(args.generated_at, str) or not args.generated_at.strip():
        parser.error("--generated-at is required when executing the diagnostic")
    report = run_diagnostic(
        generated_at=args.generated_at,
        same_case_report_path=args.same_case_report,
        source_ablation_report_path=args.source_ablation_report,
        operator_decomposition_report_path=args.operator_decomposition_report,
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
