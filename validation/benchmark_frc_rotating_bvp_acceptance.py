#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Rotating BVP Acceptance Gate
"""Generate the fail-closed FUS-C.1 rotating-BVP acceptance report."""

from __future__ import annotations

import datetime as dt
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUST_WORKSPACE = ROOT / "scpn-fusion-rs"
REPORT_DIR = ROOT / "validation" / "reports"
JSON_REPORT = REPORT_DIR / "frc_rotating_bvp_acceptance.json"
MD_REPORT = REPORT_DIR / "frc_rotating_bvp_acceptance.md"
REFERENCE_MANIFEST = (
    ROOT / "validation" / "reference_data" / "full_fidelity_public_artifacts"
    / "frc_reference_papers_manifest.json"
)

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_frc = importlib.import_module("scpn_fusion.core.frc_rigid_rotor")
ELEMENTARY_CHARGE_C = _frc.ELEMENTARY_CHARGE_C
MU_0 = _frc.MU_0
RigidRotorFRCInputs = _frc.RigidRotorFRCInputs
rotating_frc_bvp_acceptance_status = _frc.rotating_frc_bvp_acceptance_status
solve_frc_equilibrium = _frc.solve_frc_equilibrium
validate_equilibrium = _frc.validate_equilibrium


def _run_text(command: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


def _pressure_matched_inputs(theta_dot: float) -> RigidRotorFRCInputs:
    t_i_ev = 10_000.0
    t_e_ev = 5_000.0
    b_ext = 5.0
    n0 = b_ext**2 / (2.0 * MU_0) / ((t_i_ev + t_e_ev) * ELEMENTARY_CHARGE_C)
    return RigidRotorFRCInputs(
        n0=n0,
        T_i_eV=t_i_ev,
        T_e_eV=t_e_ev,
        theta_dot=theta_dot,
        R_s=0.20,
        B_ext=b_ext,
        delta=0.02,
    )


def _python_no_rotation_contract() -> dict[str, Any]:
    rho = np.linspace(0.0, 0.4, 401, dtype=np.float64)
    state = solve_frc_equilibrium(_pressure_matched_inputs(theta_dot=0.0), rho)
    validation = validate_equilibrium(state)
    passed = (
        bool(state.converged)
        and bool(validation.passed)
        and bool(state.field_reversal_passed)
        and state.residual <= 1.0e-12
        and state.separatrix_radius_error_m <= 1.0e-3
    )
    return {
        "passed": passed,
        "model": state.model,
        "grid_points": int(rho.size),
        "converged": bool(state.converged),
        "validation_passed": bool(validation.passed),
        "field_reversal_passed": bool(state.field_reversal_passed),
        "residual": float(state.residual),
        "separatrix_radius_error_m": float(state.separatrix_radius_error_m),
        "s_parameter": float(state.s_parameter),
        "pressure_balance_residual_linf": float(state.pressure_balance_residual_linf),
        "force_balance_residual_linf": float(state.force_balance_residual_linf),
    }


def _python_nonzero_rotation_contract() -> dict[str, Any]:
    rho = np.linspace(0.0, 0.4, 65, dtype=np.float64)
    try:
        solve_frc_equilibrium(_pressure_matched_inputs(theta_dot=1.0), rho)
    except NotImplementedError as exc:
        return {
            "passed": "rotating rigid-rotor BVP" in str(exc),
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
    return {
        "passed": False,
        "exception_type": None,
        "message": "nonzero theta_dot unexpectedly solved",
    }


def _reference_gate() -> dict[str, Any]:
    manifest = cast(dict[str, Any], json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8")))
    papers = manifest.get("papers")
    if not isinstance(papers, list):
        raise ValueError("FRC reference manifest must contain a papers list")
    steinhauer = next(
        (
            paper
            for paper in papers
            if isinstance(paper, dict)
            and paper.get("key") == "steinhauer_2011_review_field_reversed_configurations"
        ),
        None,
    )
    if steinhauer is None:
        raise ValueError("Steinhauer 2011 reference row missing from FRC manifest")
    local_artifacts = steinhauer.get("local_artifacts", [])
    has_pdf = any(
        isinstance(item, dict) and item.get("content_type_detected") == "PDF document"
        for item in local_artifacts
    )
    download_status = str(steinhauer.get("download_status", "missing"))
    passed = download_status == "blocked_by_publisher_http_403" and not has_pdf
    return {
        "passed": passed,
        "reference_key": steinhauer["key"],
        "citation": steinhauer["citation"],
        "doi": steinhauer.get("doi"),
        "download_status": download_status,
        "has_verified_pdf_payload": has_pdf,
        "artifact_count": len(local_artifacts) if isinstance(local_artifacts, list) else 0,
        "notes": steinhauer.get("notes", ""),
    }


def _rust_status(run_rust: bool) -> dict[str, Any]:
    if not run_rust:
        return {
            "status": "not_run",
            "passed": None,
            "reason": "run_rust=False",
        }
    command = [
        "cargo",
        "run",
        "--quiet",
        "-p",
        "fusion-physics",
        "--example",
        "frc_rotating_bvp_status",
    ]
    env = os.environ.copy()
    env.setdefault("CARGO_TARGET_DIR", str(RUST_WORKSPACE / "target"))
    completed = subprocess.run(
        command,
        cwd=RUST_WORKSPACE,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = cast(dict[str, Any], json.loads(completed.stdout.strip()))
    payload["passed"] = (
        payload.get("status") == "blocked_missing_verified_steinhauer_rotating_closure"
        and payload.get("rotating_bvp_implemented") is False
        and payload.get("no_rotation_converged") is True
        and payload.get("nonzero_rotation_fail_closed") is True
    )
    payload["command"] = command
    payload["stderr"] = completed.stderr.strip()
    return payload


def build_report(*, run_rust: bool = True) -> dict[str, Any]:
    """Build the FUS-C.1 rotating-BVP acceptance report payload."""
    python_status = rotating_frc_bvp_acceptance_status()
    python_no_rotation = _python_no_rotation_contract()
    python_nonzero = _python_nonzero_rotation_contract()
    reference_gate = _reference_gate()
    rust = _rust_status(run_rust)
    rust_passed = rust["passed"] is True if run_rust else True
    fail_closed_passed = (
        python_status["status"] == "blocked_missing_verified_steinhauer_rotating_closure"
        and python_status["rotating_bvp_implemented"] is False
        and python_no_rotation["passed"] is True
        and python_nonzero["passed"] is True
        and reference_gate["passed"] is True
        and rust_passed
    )
    return {
        "schema": "frc-rotating-bvp-acceptance.v1",
        "generated_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "status": (
            "blocked_rotating_bvp_reference_missing_fail_closed_contract_passed"
            if fail_closed_passed
            else "failed_rotating_bvp_acceptance_contract"
        ),
        "accepted_full_fidelity_rotating_bvp": False,
        "fail_closed_contract_passed": fail_closed_passed,
        "missing_requirements": [
            "Steinhauer 2011 Section II.B plus Figure 3 closure",
            "machine-readable Steinhauer rotating-BVP reference profile",
            "Python/Rust rotating-BVP parity after verified closure lands",
            "same-case rotating-BVP benchmark evidence before any acceleration claim",
        ],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "rustc": _run_text(["rustc", "--version"]),
            "cargo": _run_text(["cargo", "--version"]),
        },
        "python_status": python_status,
        "python_no_rotation_contract": python_no_rotation,
        "python_nonzero_rotation_contract": python_nonzero,
        "rust_status": rust,
        "steinhauer_reference_gate": reference_gate,
    }


def _write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# FRC Rotating BVP Acceptance",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Status: `{report['status']}`",
        f"- Accepted full-fidelity rotating BVP: `{report['accepted_full_fidelity_rotating_bvp']}`",
        f"- Fail-closed contract passed: `{report['fail_closed_contract_passed']}`",
        f"- Python: `{report['environment']['python']}`",
        f"- Rust: `{report['environment']['rustc']}`",
        "",
        "## Contract Checks",
        "",
        "| Check | Result | Evidence |",
        "|---|:---:|---|",
        (
            "| Python status | `{}` | `{}` |".format(
                report["python_status"]["status"],
                report["python_status"]["solver_action"],
            )
        ),
        (
            "| Python no-rotation solve | `{}` | residual `{:.6e}`, s `{:.6e}` |".format(
                report["python_no_rotation_contract"]["passed"],
                report["python_no_rotation_contract"]["residual"],
                report["python_no_rotation_contract"]["s_parameter"],
            )
        ),
        (
            "| Python nonzero rotation | `{}` | `{}` |".format(
                report["python_nonzero_rotation_contract"]["passed"],
                report["python_nonzero_rotation_contract"]["exception_type"],
            )
        ),
        (
            "| Rust status | `{}` | `{}` |".format(
                report["rust_status"]["passed"],
                report["rust_status"]["status"],
            )
        ),
        (
            "| Steinhauer reference gate | `{}` | `{}` |".format(
                report["steinhauer_reference_gate"]["passed"],
                report["steinhauer_reference_gate"]["download_status"],
            )
        ),
        "",
        "## Missing Requirements",
        "",
    ]
    lines.extend(f"- {item}" for item in report["missing_requirements"])
    lines.extend(
        [
            "",
            "The accepted production contract remains the Steinhauer no-rotation analytical "
            "FRC equilibrium. Nonzero `theta_dot` remains fail-closed until the missing "
            "reference requirements are satisfied.",
        ]
    )
    MD_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    report = build_report(run_rust=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report)
    print(f"FRC rotating BVP acceptance {report['status']}: {MD_REPORT.relative_to(ROOT)}")
    return 0 if report["fail_closed_contract_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
