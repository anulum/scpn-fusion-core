#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
"""Strict fail-closed free-boundary parity gate.

This benchmark does not solve a new equilibrium. It evaluates whether existing
public FreeGS reconstruction and public machine metadata artifacts satisfy the
full-fidelity free-boundary acceptance contract:

* same-case external nonlinear output is available,
* native same-case ``psi(R,Z)`` comparison is available,
* native-vs-FreeGS thresholds pass for ``psi_N``, current, axis, X-point, and
  boundary containment,
* public grid-convergence evidence contains the required resolution ladder, and
* public external coil-current/vacuum sidecars are linked to the same case.

Missing data is reported as a blocked row, never as a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, TypeGuard

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "validation" / "reports"
DEFAULT_FREEGS_REPORT = REPORT_DIR / "freegs_public_example_reconstruction.json"
DEFAULT_MACHINE_METADATA_REPORT = (
    REPORT_DIR / "free_boundary_public_machine_metadata_inventory.json"
)
DEFAULT_JSON_REPORT = REPORT_DIR / "free_boundary_strict_parity_benchmark.json"
DEFAULT_MD_REPORT = REPORT_DIR / "free_boundary_strict_parity_benchmark.md"


def _finite_number(value: object) -> TypeGuard[int | float]:
    """Reject booleans and nonfinite or unrepresentable JSON numbers."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _rel(path: Path) -> str:
    """Return a repository-relative display path."""
    resolved = path if path.is_absolute() else ROOT / path
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk with a fail-closed error on wrong shape."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _sha256_json(payload: Any) -> str:
    """Return the canonical SHA-256 digest for a JSON-serialisable payload."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_commit() -> str:
    """Return the current Git commit, falling back to `unknown` outside Git."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return proc.stdout.strip() or "unknown"


def _require_bool(data: dict[str, Any], key: str) -> bool:
    """Return a strict boolean field, treating missing/non-boolean as False."""
    return bool(data.get(key) is True)


def _source_record(artifact_id: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Bind parsed report content to one file snapshot, not to absent field arrays."""
    resolved = path if path.is_absolute() else ROOT / path
    payload_digest = _sha256_json(payload)
    file_digest: str | None = None
    matches = False
    try:
        raw = resolved.read_bytes()
    except OSError:
        raw = None
    if raw is not None:
        file_digest = hashlib.sha256(raw).hexdigest()
        try:
            disk_payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            disk_payload = None
        matches = isinstance(disk_payload, dict) and _sha256_json(disk_payload) == payload_digest
    return {
        "artifact_id": artifact_id,
        "path": _rel(path),
        "payload_sha256": payload_digest,
        "file_sha256": file_digest,
        "payload_matches_file": matches,
        "schema": payload.get("schema"),
        "status": payload.get("status"),
    }


def _identity_cases(section: object) -> dict[str, dict[str, Any]]:
    """Index a report section without silently discarding malformed identities."""
    if not isinstance(section, dict) or not isinstance(section.get("cases"), list):
        raise ValueError("identity section requires a cases list")
    indexed: dict[str, dict[str, Any]] = {}
    for case in section["cases"]:
        if not isinstance(case, dict):
            raise ValueError("identity cases must be objects")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in indexed:
            raise ValueError("identity cases require unique nonempty case_id values")
        indexed[case_id] = case
    return indexed


def _case_identity_errors(freegs: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    """Compare reported cohort, source and configuration identity, not array custody."""
    from validation.free_boundary_vacuum_evidence import evaluate_vacuum_evidence

    strict = freegs.get("strict_free_boundary_parity_evidence")
    strict = strict if isinstance(strict, dict) else {}
    sections = {
        "reconstruction": freegs,
        "strict": strict,
        "reference": metadata.get("same_case_public_reference_output"),
        "grid": strict.get("grid_convergence_evidence"),
        "geometry": strict.get("geometry_containment_evidence"),
    }
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    errors: list[str] = []
    for name, section in sections.items():
        try:
            indexed[name] = _identity_cases(section)
        except ValueError:
            errors.append(f"{name}_case_identity_malformed")
            indexed[name] = {}
    baseline = indexed["reconstruction"]
    if not baseline:
        errors.append("reconstruction_cases_empty")
    fields_by_section = {
        "strict": ("machine_class", "example_path", "example_sha256"),
        "reference": ("machine_class", "example_path", "example_sha256"),
        "grid": ("machine_class",),
        "geometry": (),
    }
    for name, fields in fields_by_section.items():
        candidates = indexed[name]
        if set(candidates) != set(baseline):
            errors.append(f"{name}_case_membership_mismatch")
        for case_id in sorted(set(candidates) & set(baseline)):
            sidecar = baseline[case_id].get("vacuum_green_function_comparison")
            if name == "strict":
                if _sha256_json(sidecar) != _sha256_json(
                    candidates[case_id].get("vacuum_green_function_comparison")
                ):
                    errors.append(f"strict:{case_id}:vacuum_sidecar_mismatch")
                vacuum_diagnostics = evaluate_vacuum_evidence(sidecar)
                if not vacuum_diagnostics["structure_consistent"]:
                    errors.append(f"strict:{case_id}:vacuum_sidecar_structure")
                if not vacuum_diagnostics["metrics_consistent"]:
                    errors.append(f"strict:{case_id}:vacuum_numerics_inconsistent")
            if name == "reference":
                coils = sidecar.get("coils") if isinstance(sidecar, dict) else None
                count = candidates[case_id].get("coil_count")
                if not isinstance(coils, list) or type(count) is not int or count != len(coils):
                    errors.append(f"reference:{case_id}:coil_count_mismatch")
            for field in fields:
                value = baseline[case_id].get(field)
                if (
                    not isinstance(value, str)
                    or not value
                    or candidates[case_id].get(field) != value
                ):
                    errors.append(f"{name}:{case_id}:{field}_mismatch")
            if name == "strict":
                source = baseline[case_id].get("source_contract")
                candidate = candidates[case_id].get("source_contract")
                if (
                    not isinstance(source, dict)
                    or not source
                    or _sha256_json(source) != _sha256_json(candidate)
                ):
                    errors.append(f"strict:{case_id}:source_contract_mismatch")
            if name == "geometry":
                solve = baseline[case_id].get("nonlinear_solve_attempt")
                comparison = (
                    solve.get("native_same_case_profile_source_comparison")
                    if isinstance(solve, dict)
                    else None
                )
                if not isinstance(comparison, dict):
                    errors.append(f"geometry:{case_id}:comparison_missing")
                    continue
                expected = {
                    "external_axis": [
                        comparison.get("external_axis_r_m"),
                        comparison.get("external_axis_z_m"),
                    ],
                    "native_axis": [
                        comparison.get("native_axis_r_m"),
                        comparison.get("native_axis_z_m"),
                    ],
                    "boundary_containment_fraction": comparison.get(
                        "boundary_containment_fraction"
                    ),
                }
                for field, value in expected.items():
                    values = value if isinstance(value, list) else [value]
                    if not all(_finite_number(item) for item in values) or _sha256_json(
                        value
                    ) != _sha256_json(candidates[case_id].get(field)):
                        errors.append(f"geometry:{case_id}:{field}_mismatch")
    return errors


def _source_example_files(freegs: dict[str, Any]) -> list[dict[str, Any]]:
    """Hash named repository source examples without executing scientific code."""
    try:
        cases = _identity_cases(freegs)
    except ValueError:
        return []
    records: list[dict[str, Any]] = []
    for case_id, case in cases.items():
        path_value = case.get("example_path")
        actual_hash: str | None = None
        if isinstance(path_value, str) and path_value:
            try:
                path = (ROOT / path_value).resolve()
                path.relative_to(ROOT.resolve())
                if path.is_file():
                    digest = hashlib.sha256()
                    with path.open("rb") as source:
                        for chunk in iter(lambda: source.read(65536), b""):
                            digest.update(chunk)
                    actual_hash = digest.hexdigest()
            except (OSError, ValueError, RuntimeError):
                actual_hash = None
        records.append(
            {
                "case_id": case_id,
                "path": path_value,
                "declared_sha256": case.get("example_sha256"),
                "file_sha256": actual_hash,
                "declared_hash_matches_file": actual_hash is not None
                and actual_hash == case.get("example_sha256"),
            }
        )
    return records


def evaluate_strict_parity(
    freegs_report: dict[str, Any],
    machine_metadata_report: dict[str, Any],
    *,
    freegs_report_path: Path = DEFAULT_FREEGS_REPORT,
    machine_metadata_report_path: Path = DEFAULT_MACHINE_METADATA_REPORT,
) -> dict[str, Any]:
    """Classify summary evidence without granting scientific admission.

    Parameters
    ----------
    freegs_report, machine_metadata_report : dict
        Original reconstruction and machine metadata reports.
    freegs_report_path, machine_metadata_report_path : Path
        Paths recorded for provenance. Report checksums do not establish
        custody of absent external/native field arrays or q-profile samples.

    Returns
    -------
    dict
        Version-2 non-admitting report preserving diagnostic legacy rows.
        Known v1 inputs lack verified field custody; unknown schemas are
        rejected too. No positive scientific evidence contract is admitted
        by this implementation.
    """
    strict = freegs_report.get("strict_free_boundary_parity_evidence", {})
    if not isinstance(strict, dict):
        strict = {}
    geometry = strict.get("geometry_containment_evidence", {})
    if not isinstance(geometry, dict):
        geometry = {}
    grid = strict.get("grid_convergence_evidence", {})
    if not isinstance(grid, dict):
        grid = {}

    threshold_ready = _require_bool(strict, "strict_threshold_acceptance_ready")
    grid_ready = _require_bool(strict, "grid_convergence_ready")
    sidecar_ready = _require_bool(strict, "coil_vacuum_sidecar_ready")
    native_ready = _require_bool(strict, "native_same_case_profile_source_ready")
    external_ready = _require_bool(freegs_report, "external_nonlinear_output_ready")
    geometry_ready = _require_bool(geometry, "strict_geometry_containment_ready")
    boundary_metric_ready = _require_bool(geometry, "boundary_containment_metric_ready")
    machine_metadata_ready = _require_bool(machine_metadata_report, "machine_metadata_ready")
    machine_reference_ready = _require_bool(machine_metadata_report, "reference_output_ready")
    legacy = strict.get("schema") == "strict-free-boundary-parity-evidence.v1"
    classification = "legacy_non_admitting" if legacy else "unsupported_evidence_non_admitting"

    blockers: list[str] = [
        "legacy_evidence_has_no_verified_field_custody"
        if legacy
        else "unsupported_evidence_contract"
    ]
    if not threshold_ready:
        blockers.append("strict_threshold_acceptance_failed")
    if not grid_ready:
        blockers.append("grid_convergence_evidence_missing")
    if not sidecar_ready:
        blockers.append("public_external_coil_vacuum_sidecars_missing")
    if not native_ready:
        blockers.append("native_same_case_profile_source_comparison_missing")
    if not external_ready:
        blockers.append("external_freegs_nonlinear_output_missing")
    if not geometry_ready:
        blockers.append("geometry_containment_evidence_missing")
    if not boundary_metric_ready:
        blockers.append("boundary_containment_metric_missing")
    if not machine_metadata_ready:
        blockers.append("public_machine_metadata_inventory_missing")
    if not machine_reference_ready:
        blockers.append("same_case_public_reference_output_missing")

    acceptance_contract = {
        "gate_semantics": "fail_closed",
        "required_threshold_metrics": [
            "psi_N_RMSE",
            "current_closure",
            "magnetic_axis_error",
            "x_point_error",
            "boundary_containment",
            "q_profile_sanity",
        ],
        "requires_native_same_case_profile_source": True,
        "requires_public_external_coil_vacuum_sidecars": True,
        "requires_grid_convergence_ladder": True,
        "requires_same_case_public_reference_output": True,
    }
    acceptance_matrix = {
        "same_case_reference_output": external_ready and machine_reference_ready,
        "native_same_case_profile_source": native_ready,
        "strict_threshold_metrics": threshold_ready and geometry_ready and boundary_metric_ready,
        "grid_convergence_ladder": grid_ready,
        "coil_vacuum_sidecars": sidecar_ready,
        "machine_metadata": machine_metadata_ready,
    }
    checks = {
        "external_nonlinear_output_ready": external_ready,
        "native_same_case_profile_source_ready": native_ready,
        "strict_threshold_acceptance_ready": threshold_ready,
        "geometry_containment_ready": geometry_ready,
        "boundary_containment_metric_ready": boundary_metric_ready,
        "grid_convergence_ready": grid_ready,
        "coil_vacuum_sidecar_ready": sidecar_ready,
        "machine_metadata_ready": machine_metadata_ready,
        "same_case_public_reference_output_ready": machine_reference_ready,
    }
    reported_readiness = checks
    checks = dict.fromkeys(checks, False)
    acceptance_matrix = dict.fromkeys(acceptance_matrix, False)
    from validation.free_boundary_threshold_evidence import evaluate_threshold_evidence

    numeric_diagnostics = evaluate_threshold_evidence(strict)
    threshold_cases = numeric_diagnostics["threshold_cases"]
    grid_cases = numeric_diagnostics["grid_cases"]
    if {case["case_id"] for case in threshold_cases} != {case["case_id"] for case in grid_cases}:
        blockers.append("grid_case_membership_mismatch")
    input_reports = [
        _source_record("freegs_public_example_reconstruction", freegs_report_path, freegs_report),
        _source_record(
            "free_boundary_public_machine_metadata_inventory",
            machine_metadata_report_path,
            machine_metadata_report,
        ),
    ]
    source_checksums = {
        f"{source['artifact_id']}_{key}": source[key]
        for source in input_reports
        for key in ("payload_sha256", "file_sha256")
    }
    if any(source["payload_matches_file"] is not True for source in input_reports):
        blockers.append("input_report_payload_not_bound_to_file")
    identity_errors = _case_identity_errors(freegs_report, machine_metadata_report)
    if identity_errors:
        blockers.append("case_identity_inconsistent")
    example_files = _source_example_files(freegs_report)
    if not example_files or any(
        row["declared_hash_matches_file"] is not True for row in example_files
    ):
        blockers.append("source_example_bytes_unverified")
    machine_metadata = {
        "schema": machine_metadata_report.get("schema"),
        "status": machine_metadata_report.get("status"),
        "machine_config_count": int(machine_metadata_report.get("machine_config_count", 0)),
        "machines": machine_metadata_report.get("machines", []),
        "missing_full_fidelity_requirements": machine_metadata_report.get(
            "missing_full_fidelity_requirements", []
        ),
    }

    return {
        "schema": "free-boundary-strict-parity-benchmark.v2",
        "benchmark_id": "free_boundary_strict_parity",
        "benchmark_scope": "free_boundary_full_fidelity_acceptance",
        "accepted_full_fidelity": False,
        "evidence_classification": classification,
        "reported_readiness": reported_readiness,
        "status": "blocked_free_boundary_strict_parity",
        "inputs": {
            "freegs_public_example_reconstruction": _rel(freegs_report_path),
            "free_boundary_public_machine_metadata_inventory": _rel(machine_metadata_report_path),
        },
        "source_checksums": source_checksums,
        "provenance": {
            "generator": "validation/benchmark_free_boundary_strict_parity.py",
            "source_commit": _source_commit(),
            "python_version": sys.version.split()[0],
            "input_reports": input_reports,
        },
        "checks": checks,
        "acceptance_contract": acceptance_contract,
        "acceptance_matrix": acceptance_matrix,
        "evidence_checksums": {
            "acceptance_contract_sha256": _sha256_json(acceptance_contract),
            "acceptance_matrix_sha256": _sha256_json(acceptance_matrix),
            "checks_sha256": _sha256_json(checks),
            "threshold_cases_sha256": _sha256_json(threshold_cases),
            "grid_convergence_cases_sha256": _sha256_json(grid_cases),
            "machine_metadata_sha256": _sha256_json(machine_metadata),
        },
        "blockers": blockers,
        "case_identity_errors": identity_errors,
        "source_example_files": example_files,
        "case_count": len(threshold_cases),
        "failed_threshold_check_count": sum(
            row["failed_threshold_check_count"] for row in threshold_cases
        ),
        "threshold_cases": threshold_cases,
        "grid_convergence": {
            "schema": grid.get("schema"),
            "status": grid.get("status"),
            "required_resolution_count": numeric_diagnostics["required_resolution_count"],
            "cases": grid_cases,
        },
        "machine_metadata": machine_metadata,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the strict parity report through its presentation module."""
    from validation.free_boundary_strict_parity_report import render_markdown as render

    return render(report)


def run_benchmark(
    *,
    freegs_report_path: Path = DEFAULT_FREEGS_REPORT,
    machine_metadata_report_path: Path = DEFAULT_MACHINE_METADATA_REPORT,
    json_report_path: Path = DEFAULT_JSON_REPORT,
    md_report_path: Path = DEFAULT_MD_REPORT,
    write: bool = True,
) -> dict[str, Any]:
    """Run the strict parity benchmark from tracked report artifacts."""
    report = evaluate_strict_parity(
        _load_json(freegs_report_path),
        _load_json(machine_metadata_report_path),
        freegs_report_path=freegs_report_path,
        machine_metadata_report_path=machine_metadata_report_path,
    )
    if write:
        json_report_path.parent.mkdir(parents=True, exist_ok=True)
        json_report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        md_report_path.write_text(render_markdown(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the strict parity benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run without writing reports.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero unless full-fidelity free-boundary parity is accepted.",
    )
    args = parser.parse_args(argv)
    report = run_benchmark(write=not args.check)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if (not args.strict or report["accepted_full_fidelity"]) else 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
