#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — Coil-Vacuum Grid Diagnostic
"""Diagnose IDA coil-vacuum source and source-free grid convergence."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable, cast

import jax
import jax.numpy as jnp
import numpy as np

import validation.ida_coil_vacuum_grid_contract as contract
from validation.ida_coil_vacuum_grid_fields import (
    BoolArray,
    FloatArray,
    extract_coil_manifest,
    manifest_payload,
    validate_frozen_manifest,
    zero_identity_wall,
)
from validation.ida_coil_vacuum_grid_convergence import build_convergence
from validation.ida_coil_vacuum_grid_runtime import run_grid

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
_fixed_point_contract = cast(
    Any,
    importlib.import_module("validation.ida_fixed_point_stability_contract"),
)
_response_contract = cast(
    Any,
    importlib.import_module("validation.ida_operator_response_contract"),
)
_response_fields = cast(
    Any,
    importlib.import_module("validation.ida_operator_response_fields"),
)
_predictive = cast(
    Any,
    importlib.import_module("scpn_fusion.core.jax_free_boundary_predictive"),
)

ROOT: Path = _same_case.ROOT
REPORT_PATH = ROOT / contract.REPORT_PATH
MARKDOWN_PATH = ROOT / contract.MARKDOWN_PATH
_array_sha256: Callable[[object], str] = _same_case._array_sha256
_file_sha256: Callable[[Path], str] = _same_case._file_sha256
_reject_duplicate_json_keys: Callable[
    [list[tuple[str, Any]]],
    dict[str, Any],
] = _same_case._reject_duplicate_json_keys
_runtime_environment: Callable[[], dict[str, Any]] = _same_case._runtime_environment


def _load_report(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], value)


def _load_bound_reports() -> dict[str, dict[str, Any]]:
    """Validate the exact six-report prerequisite chain and frozen payloads."""
    reports = {
        "same_case": _load_report(ROOT / contract.SAME_CASE_PATH),
        "source_ablation": _load_report(ROOT / contract.SOURCE_ABLATION_PATH),
        "operator": _load_report(ROOT / contract.OPERATOR_PATH),
        "source_mechanism": _load_report(ROOT / contract.SOURCE_MECHANISM_PATH),
        "fixed_point": _load_report(ROOT / contract.FIXED_POINT_PATH),
        "response": _load_report(ROOT / contract.RESPONSE_PATH),
    }
    _same_case.validate_report(reports["same_case"])
    _source.validate_report(reports["source_ablation"])
    _operator_contract.validate_report(reports["operator"])
    _mechanism.validate_report(reports["source_mechanism"])
    _fixed_point_contract.validate_report(reports["fixed_point"])
    _response_contract.validate_report(reports["response"])
    for name, expected in contract.EXPECTED_PAYLOADS.items():
        if reports[name]["payload_sha256"] != expected:
            raise ValueError(f"{name} report does not match the frozen payload")
    return reports


def _runtime_source_artifact(
    module: object,
    *,
    logical_path: str,
    resource_name: str,
) -> dict[str, str]:
    module_path = getattr(module, "__file__", None)
    if not isinstance(module_path, str) or not module_path:
        raise RuntimeError(f"{logical_path} has no inspectable runtime source")
    source_path = Path(module_path).resolve()
    if source_path.name != resource_name or not source_path.is_file():
        raise RuntimeError(f"{logical_path} does not resolve to {resource_name}")
    return {"path": logical_path, "sha256": _file_sha256(source_path)}


def _repository_artifact() -> dict[str, Any]:
    """Bind an inspectable HEAD and distinguish clean status from probe failure."""
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("canonical repository provenance is not inspectable") from exc
    source_commit = commit_result.stdout.strip()
    if contract._GIT_OID_RE.fullmatch(source_commit) is None:
        raise RuntimeError("canonical repository HEAD is not a full Git object ID")
    return {
        "git_commit": source_commit,
        "path": ".",
        "worktree_clean": not bool(status_result.stdout.strip()),
    }


def _source_artifacts(
    *,
    freegs: Any,
    tokamak: Any,
    public_example_path: Path,
) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {
        name: {"path": path, "sha256": _file_sha256(ROOT / path)}
        for name, path in sorted(contract.SOURCE_PATHS.items())
    }
    shaped_coil = tokamak.coils[0][1]
    runtime = {
        "freegs_boundary": (freegs.boundary, "freegs.boundary", "boundary.py"),
        "freegs_machine": (
            importlib.import_module(type(tokamak).__module__),
            "freegs.machine",
            "machine.py",
        ),
        "freegs_operator": (
            freegs.gradshafranov,
            "freegs.gradshafranov",
            "gradshafranov.py",
        ),
        "freegs_shaped_coil": (
            importlib.import_module(type(shaped_coil).__module__),
            "freegs.shaped_coil",
            "shaped_coil.py",
        ),
    }
    for name, (module, logical_path, resource_name) in runtime.items():
        artifacts[name] = _runtime_source_artifact(
            module,
            logical_path=logical_path,
            resource_name=resource_name,
        )
    artifacts["freegs_public_example"] = {
        "path": str(public_example_path.relative_to(ROOT)),
        "sha256": _file_sha256(public_example_path),
    }
    artifacts["repository"] = _repository_artifact()
    return artifacts


def _bindings(reports: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": contract.EXPECTED_BINDING_PATHS[name],
            "payload_sha256": reports[name]["payload_sha256"],
        }
        for name in sorted(contract.EXPECTED_PAYLOADS)
    }


def _anchor(
    *,
    equilibrium: Any,
    r_grid: FloatArray,
    z_grid: FloatArray,
    response_closure_max_abs_wb: float,
) -> tuple[dict[str, Any], FloatArray, FloatArray]:
    total_rz = np.asarray(equilibrium.psi(), dtype=np.float64)
    plasma_rz = np.asarray(equilibrium.plasma_psi, dtype=np.float64)
    forcing_zr = zero_identity_wall(
        np.asarray(
            (
                _operator._native_lhs(total_rz, r_grid=r_grid, z_grid=z_grid)
                - _operator._native_lhs(plasma_rz, r_grid=r_grid, z_grid=z_grid)
            ).T,
            dtype=np.float64,
        ),
        field="129 anchor forcing",
    )
    d_r = float(r_grid[1] - r_grid[0])
    d_z = float(z_grid[1] - z_grid[0])
    preconditioner = _predictive.build_gs_mg_preconditioner(
        forcing_zr.shape,
        jnp.asarray(r_grid),
        d_r,
        d_z,
    )
    response_zr = -_response_fields.native_inverse(
        forcing_zr,
        r_grid=r_grid,
        d_r=d_r,
        d_z=d_z,
        preconditioner=preconditioner,
        x0_zr=np.zeros_like(forcing_zr),
    )
    forcing_digest = _array_sha256(forcing_zr)
    response_digest = _array_sha256(response_zr)
    if forcing_digest != contract.EXPECTED_ANCHOR_FORCING_SHA256:
        raise ValueError("129 coil-vacuum forcing anchor drifted")
    if response_digest != contract.EXPECTED_ANCHOR_RESPONSE_SHA256:
        raise ValueError("129 coil-vacuum response anchor drifted")
    return (
        {
            "forcing_sha256": forcing_digest,
            "response_closure_max_abs_wb": response_closure_max_abs_wb,
            "response_sha256": response_digest,
        },
        np.asarray((total_rz - plasma_rz).T, dtype=np.float64),
        forcing_zr,
    )


def _plasma_support_mask(
    *,
    equilibrium: Any,
    profiles: Any,
    resolution: int,
    r_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
) -> BoolArray:
    """Evaluate the frozen FreeGS reference-current support on one nested grid."""
    r_grid = np.linspace(*r_bounds, resolution, dtype=np.float64)
    z_grid = np.linspace(*z_bounds, resolution, dtype=np.float64)
    r_mesh, z_mesh = np.meshgrid(r_grid, z_grid, indexing="ij")
    total_psi = np.asarray(equilibrium.psiRZ(r_mesh, z_mesh), dtype=np.float64)
    runtime_profiles = profiles
    if getattr(profiles, "eq", None) is equilibrium:
        runtime_profiles = copy.copy(profiles)

        class _FrozenBoundaryEquilibrium:
            def __init__(self, psi_bndry: float) -> None:
                self.psi_bndry = psi_bndry

            def _updateBoundaryPsi(self, psi: object) -> None:
                """Keep the solved boundary flux fixed on a diagnostic grid."""

        runtime_profiles.eq = _FrozenBoundaryEquilibrium(float(equilibrium.psi_bndry))
    current_density = np.asarray(
        runtime_profiles.Jtor(
            r_mesh,
            z_mesh,
            total_psi,
            psi_bndry=equilibrium.psi_bndry,
        ),
        dtype=np.float64,
    )
    if current_density.shape != (resolution, resolution) or not np.all(
        np.isfinite(current_density)
    ):
        raise ValueError("reference plasma-current support evaluation is invalid")
    support = np.asarray(np.abs(current_density.T) > 0.0, dtype=np.bool_)
    support[0, :] = False
    support[-1, :] = False
    support[:, 0] = False
    support[:, -1] = False
    if not np.any(support):
        raise ValueError("reference plasma-current support must not be empty")
    return support


def run_diagnostic(*, generated_at: str) -> dict[str, Any]:
    """Execute the exact 129 anchor and mandatory four-grid diagnostic."""
    if cast(bool, jax.config.values["jax_enable_x64"]) is not True:
        raise RuntimeError("coil-vacuum grid convergence requires JAX FP64")
    reports = _load_bound_reports()
    (
        _,
        _,
        spec,
        tokamak,
        equilibrium,
        profiles,
        freegs_version,
    ) = _source._solve_reference(ROOT / contract.SAME_CASE_PATH)
    freegs, _, import_error = _source._import_freegs()
    if freegs is None:
        raise RuntimeError(f"FreeGS backend unavailable: {import_error}")
    runtime_artifacts = _operator._source_artifacts(
        freegs=freegs,
        evaluation=_source._evaluation_case(reports["same_case"]),
        public_example_path=spec.example_path,
    )
    for name in ("freegs_boundary", "freegs_operator", "freegs_public_example"):
        if runtime_artifacts[name] != reports["operator"]["source_artifacts"][name]:
            raise ValueError(f"runtime {name} bytes disagree with bound operator evidence")
    parents = extract_coil_manifest(tokamak)
    validate_frozen_manifest(
        parents,
        r_bounds=(float(spec.r_min), float(spec.r_max)),
        z_bounds=(float(spec.z_min), float(spec.z_max)),
    )
    r_129 = np.asarray(equilibrium.R_1D, dtype=np.float64)
    z_129 = np.asarray(equilibrium.Z_1D, dtype=np.float64)
    anchor, reference_129, reference_129_forcing = _anchor(
        equilibrium=equilibrium,
        r_grid=r_129,
        z_grid=z_129,
        response_closure_max_abs_wb=float(
            reports["response"]["closure"]["native_operator_response_max_abs_wb"]
        ),
    )
    coarse_d_r = (float(spec.r_max) - float(spec.r_min)) / (contract.GRID_RESOLUTIONS[0] - 1)
    coarse_d_z = (float(spec.z_max) - float(spec.z_min)) / (contract.GRID_RESOLUTIONS[0] - 1)
    fixed_radius = 2.0 * max(coarse_d_r, coarse_d_z)
    results = [
        run_grid(
            resolution=resolution,
            tokamak=tokamak,
            parents=parents,
            r_bounds=(float(spec.r_min), float(spec.r_max)),
            z_bounds=(float(spec.z_min), float(spec.z_max)),
            fixed_physical_radius_m=fixed_radius,
            reference_129_zr=reference_129 if resolution == 129 else None,
            reference_129_forcing_zr=(reference_129_forcing if resolution == 129 else None),
            plasma_support_mask=_plasma_support_mask(
                equilibrium=equilibrium,
                profiles=profiles,
                resolution=resolution,
                r_bounds=(float(spec.r_min), float(spec.r_max)),
                z_bounds=(float(spec.z_min), float(spec.z_max)),
            ),
        )
        for resolution in contract.GRID_RESOLUTIONS
    ]
    grid_129 = next(row for row in results if row.resolution == 129)
    if grid_129.report["forcing_partition"]["total"]["field_sha256"] != anchor["forcing_sha256"]:
        raise ValueError("four-grid 129 forcing does not reproduce the bound anchor")
    if grid_129.report["response_partition"]["total"]["field_sha256"] != anchor["response_sha256"]:
        raise ValueError("four-grid 129 response does not reproduce the bound anchor")
    environment = _runtime_environment()
    environment["freegs_version"] = freegs_version
    return contract.build_report(
        generated_at=generated_at,
        environment=environment,
        source_artifacts=_source_artifacts(
            freegs=freegs,
            tokamak=tokamak,
            public_example_path=spec.example_path,
        ),
        bindings=_bindings(reports),
        anchor=anchor,
        coil_manifest=manifest_payload(parents),
        grids=[row.report for row in results],
        convergence=build_convergence(results),
    )


def write_report(
    report: dict[str, Any],
    *,
    json_path: Path = REPORT_PATH,
    markdown_path: Path = MARKDOWN_PATH,
) -> None:
    """Validate and write the JSON/Markdown evidence pair."""
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
    parser.add_argument("--json-report", type=Path, default=REPORT_PATH)
    parser.add_argument("--markdown-report", type=Path, default=MARKDOWN_PATH)
    parser.add_argument("--validate-report", type=Path)
    args = parser.parse_args(argv)
    if args.validate_report is not None:
        report = _load_report(args.validate_report)
        contract.validate_report(report)
        print(report["payload_sha256"])
        return 0
    if not isinstance(args.generated_at, str) or not args.generated_at.strip():
        parser.error("--generated-at is required when executing the diagnostic")
    report = run_diagnostic(generated_at=args.generated_at)
    write_report(
        report,
        json_path=args.json_report,
        markdown_path=args.markdown_report,
    )
    print(json.dumps(report["routing"], allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
