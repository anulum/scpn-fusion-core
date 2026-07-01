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
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "validation" / "reports"
DEFAULT_FREEGS_REPORT = REPORT_DIR / "freegs_public_example_reconstruction.json"
DEFAULT_MACHINE_METADATA_REPORT = (
    REPORT_DIR / "free_boundary_public_machine_metadata_inventory.json"
)
DEFAULT_JSON_REPORT = REPORT_DIR / "free_boundary_strict_parity_benchmark.json"
DEFAULT_MD_REPORT = REPORT_DIR / "free_boundary_strict_parity_benchmark.md"


def _rel(path: Path) -> str:
    """Return a repository-relative display path."""
    resolved = path if path.is_absolute() else ROOT / path
    return str(resolved.relative_to(ROOT))


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


def _sha256_file(path: Path) -> str | None:
    """Return the SHA-256 digest for `path`, or None when the file is absent."""
    resolved = path if path.is_absolute() else ROOT / path
    if not resolved.exists():
        return None
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


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
    """Build provenance and checksum metadata for an input report artifact."""
    return {
        "artifact_id": artifact_id,
        "path": _rel(path),
        "payload_sha256": _sha256_json(payload),
        "file_sha256": _sha256_file(path),
        "schema": payload.get("schema"),
        "status": payload.get("status"),
    }


def _source_checksums(
    freegs_report: dict[str, Any],
    machine_metadata_report: dict[str, Any],
    *,
    freegs_report_path: Path,
    machine_metadata_report_path: Path,
) -> dict[str, str | None]:
    """Return stable input payload and file checksum fields."""
    return {
        "freegs_public_example_reconstruction_payload_sha256": _sha256_json(freegs_report),
        "freegs_public_example_reconstruction_file_sha256": _sha256_file(freegs_report_path),
        "free_boundary_public_machine_metadata_inventory_payload_sha256": _sha256_json(
            machine_metadata_report
        ),
        "free_boundary_public_machine_metadata_inventory_file_sha256": _sha256_file(
            machine_metadata_report_path
        ),
    }


def _threshold_case_rows(strict: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract per-case strict threshold rows from the FreeGS report."""
    rows: list[dict[str, Any]] = []
    for case in strict.get("cases", []):
        if not isinstance(case, dict):
            continue
        failed_checks = [
            check
            for check in case.get("threshold_checks", [])
            if isinstance(check, dict) and check.get("passed") is not True
        ]
        rows.append(
            {
                "case_id": str(case.get("case_id", "")),
                "external_nonlinear_output_ready": bool(
                    case.get("external_nonlinear_output_ready") is True
                ),
                "native_same_case_profile_source_ready": bool(
                    case.get("native_same_case_profile_source_ready") is True
                ),
                "strict_threshold_acceptance_ready": bool(
                    case.get("strict_threshold_acceptance_ready") is True
                ),
                "failed_threshold_check_count": len(failed_checks),
                "failed_threshold_checks": [
                    {
                        "metric": str(check.get("metric", "")),
                        "value": check.get("value"),
                        "limit": check.get("limit"),
                        "comparator": str(check.get("comparator", "")),
                    }
                    for check in failed_checks
                ],
            }
        )
    return rows


def _grid_case_rows(strict: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract per-case grid-convergence readiness rows."""
    grid = strict.get("grid_convergence_evidence", {})
    if not isinstance(grid, dict):
        return []
    rows: list[dict[str, Any]] = []
    for case in grid.get("cases", []):
        if not isinstance(case, dict):
            continue
        rows.append(
            {
                "case_id": str(case.get("case_id", "")),
                "machine_class": str(case.get("machine_class", "")),
                "observed_resolution_count": int(case.get("observed_resolution_count", 0)),
                "required_resolution_count": int(case.get("required_resolution_count", 0)),
                "missing_resolution_count": int(case.get("missing_resolution_count", 0)),
                "grid_convergence_case_ready": bool(
                    case.get("grid_convergence_case_ready") is True
                ),
                "blocking_reason": str(case.get("blocking_reason", "")),
            }
        )
    return rows


def evaluate_strict_parity(
    freegs_report: dict[str, Any],
    machine_metadata_report: dict[str, Any],
    *,
    freegs_report_path: Path = DEFAULT_FREEGS_REPORT,
    machine_metadata_report_path: Path = DEFAULT_MACHINE_METADATA_REPORT,
) -> dict[str, Any]:
    """Evaluate the strict free-boundary acceptance contract."""
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
    accepted = bool(
        threshold_ready
        and grid_ready
        and sidecar_ready
        and native_ready
        and external_ready
        and geometry_ready
        and boundary_metric_ready
        and machine_metadata_ready
        and machine_reference_ready
    )

    blockers: list[str] = []
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
    threshold_cases = _threshold_case_rows(strict)
    grid_cases = _grid_case_rows(strict)
    source_checksums = _source_checksums(
        freegs_report,
        machine_metadata_report,
        freegs_report_path=freegs_report_path,
        machine_metadata_report_path=machine_metadata_report_path,
    )
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
        "schema": "free-boundary-strict-parity-benchmark.v1",
        "benchmark_id": "free_boundary_strict_parity",
        "benchmark_scope": "free_boundary_full_fidelity_acceptance",
        "accepted_full_fidelity": accepted,
        "status": "accepted_full_fidelity_free_boundary_parity"
        if accepted
        else "blocked_free_boundary_strict_parity",
        "inputs": {
            "freegs_public_example_reconstruction": _rel(freegs_report_path),
            "free_boundary_public_machine_metadata_inventory": _rel(machine_metadata_report_path),
        },
        "source_checksums": source_checksums,
        "provenance": {
            "generator": "validation/benchmark_free_boundary_strict_parity.py",
            "source_commit": _source_commit(),
            "python_version": sys.version.split()[0],
            "input_reports": [
                _source_record(
                    "freegs_public_example_reconstruction",
                    freegs_report_path,
                    freegs_report,
                ),
                _source_record(
                    "free_boundary_public_machine_metadata_inventory",
                    machine_metadata_report_path,
                    machine_metadata_report,
                ),
            ],
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
        "case_count": int(freegs_report.get("case_count", 0)),
        "failed_threshold_check_count": int(strict.get("failed_threshold_check_count", 0)),
        "threshold_cases": threshold_cases,
        "grid_convergence": {
            "schema": grid.get("schema"),
            "status": grid.get("status"),
            "required_resolution_count": int(grid.get("required_resolution_count", 0)),
            "cases": grid_cases,
        },
        "machine_metadata": machine_metadata,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the strict parity report as Markdown."""
    lines = [
        "# Free-boundary Strict Parity Benchmark",
        "",
        "This gate is fail-closed. It accepts full-fidelity free-boundary parity",
        "only when same-case public FreeGS output, native profile-source",
        "comparison, strict thresholds, grid convergence, and public external",
        "coil/vacuum sidecars are all present.",
        "",
        f"- Schema: `{report['schema']}`",
        f"- Status: `{report['status']}`",
        f"- Accepted full fidelity: `{report['accepted_full_fidelity']}`",
        f"- Case count: `{report['case_count']}`",
        f"- Failed threshold checks: `{report['failed_threshold_check_count']}`",
        "",
        "## Checks",
        "",
        "| Check | Ready |",
        "| --- | ---: |",
    ]
    for key, value in report["checks"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Acceptance matrix",
            "",
            "| Requirement | Ready |",
            "| --- | ---: |",
        ]
    )
    for key, value in report["acceptance_matrix"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Blockers", ""])
    if report["blockers"]:
        for blocker in report["blockers"]:
            lines.append(f"- `{blocker}`")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Threshold cases",
            "",
            "| Case | External output | Native comparison | Thresholds ready | Failed checks |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for case in report["threshold_cases"]:
        lines.append(
            "| {case_id} | `{external}` | `{native}` | `{threshold}` | {failed} |".format(
                case_id=case["case_id"],
                external=case["external_nonlinear_output_ready"],
                native=case["native_same_case_profile_source_ready"],
                threshold=case["strict_threshold_acceptance_ready"],
                failed=case["failed_threshold_check_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Grid-convergence cases",
            "",
            "| Case | Machine | Observed | Required | Missing | Ready | Blocker |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for case in report["grid_convergence"]["cases"]:
        lines.append(
            "| {case_id} | {machine} | {observed} | {required} | {missing} | `{ready}` | {reason} |".format(
                case_id=case["case_id"],
                machine=case["machine_class"],
                observed=case["observed_resolution_count"],
                required=case["required_resolution_count"],
                missing=case["missing_resolution_count"],
                ready=case["grid_convergence_case_ready"],
                reason=case["blocking_reason"],
            )
        )
    lines.extend(["", "## Machine metadata", ""])
    metadata = report["machine_metadata"]
    lines.append(f"- Schema: `{metadata['schema']}`")
    lines.append(f"- Status: `{metadata['status']}`")
    lines.append(f"- Machine config count: `{metadata['machine_config_count']}`")
    lines.append(f"- Machines: `{', '.join(str(v) for v in metadata['machines'])}`")
    lines.extend(["", "## Provenance and checksums", ""])
    provenance = report["provenance"]
    lines.append(f"- Generator: `{provenance['generator']}`")
    lines.append(f"- Source commit: `{provenance['source_commit']}`")
    lines.append(f"- Python version: `{provenance['python_version']}`")
    lines.extend(["", "| Input report | Payload SHA-256 | File SHA-256 |", "| --- | --- | --- |"])
    for source in provenance["input_reports"]:
        lines.append(
            "| {path} | `{payload}` | `{file}` |".format(
                path=source["path"],
                payload=source["payload_sha256"],
                file=source["file_sha256"],
            )
        )
    lines.extend(["", "| Evidence section | SHA-256 |", "| --- | --- |"])
    for key, value in report["evidence_checksums"].items():
        lines.append(f"| `{key}` | `{value}` |")
    return "\n".join(lines) + "\n"


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
    raise SystemExit(main())
