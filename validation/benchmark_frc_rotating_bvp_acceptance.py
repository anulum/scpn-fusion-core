#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Rotating Rigid-Rotor Acceptance Gate
"""Generate the FUS-C.1 rotating rigid-rotor acceptance report.

The rotating closure implements the source-verified Rostoker & Qerushi (2002)
one-dimensional one-ion centrifugal density modulation (reproduced in
US 6,664,740 B2). Acceptance requires, together: the no-rotation contract still
solves; a sub-sonic rotating equilibrium validates with a non-negative pressure
and an in-tolerance centrifugal force-balance residual; the rotating profile
reduces to the no-rotation contract as ``theta_dot -> 0`` (``omega^2`` scaling,
byte-identical rigid-rotor field/flux); and the Steinhauer 2011 Figure 3
digitised parity boundary is held (not claimed, publisher payload still
unavailable).
"""

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
    ROOT
    / "validation"
    / "reference_data"
    / "full_fidelity_public_artifacts"
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

# Sub-sonic drive used for the accepted rotating equilibrium (Mach ~ 0.07).
ROTATING_ACCEPTANCE_THETA_DOT = 3.0e5
ROTATION_FORCE_BALANCE_TOLERANCE = 2.0e-2


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
        and state.model == "steinhauer_2011_no_rotation_analytical"
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


def _python_rotating_contract() -> dict[str, Any]:
    rho = np.linspace(0.0, 0.4, 401, dtype=np.float64)
    state = solve_frc_equilibrium(
        _pressure_matched_inputs(theta_dot=ROTATING_ACCEPTANCE_THETA_DOT), rho
    )
    validation = validate_equilibrium(state)
    passed = (
        state.model == "rostoker_qerushi_2002_rotating_rigid_rotor"
        and bool(np.all(state.p >= 0.0))
        and bool(validation.passed)
        and bool(validation.rotation_force_balance_passed)
        and state.pressure_clipped_fraction == 0.0
        and state.rotation_force_balance_residual_linf <= ROTATION_FORCE_BALANCE_TOLERANCE
    )
    return {
        "passed": passed,
        "model": state.model,
        "theta_dot": ROTATING_ACCEPTANCE_THETA_DOT,
        "rotation_mach_number": float(state.rotation_mach_number),
        "pressure_non_negative": bool(np.all(state.p >= 0.0)),
        "pressure_clipped_fraction": float(state.pressure_clipped_fraction),
        "validation_passed": bool(validation.passed),
        "rotation_force_balance_passed": bool(validation.rotation_force_balance_passed),
        "rotation_force_balance_residual_linf": float(state.rotation_force_balance_residual_linf),
    }


def _python_reduction_to_contract() -> dict[str, Any]:
    rho = np.linspace(0.0, 0.4, 401, dtype=np.float64)
    baseline = solve_frc_equilibrium(_pressure_matched_inputs(theta_dot=0.0), rho)
    peak = float(np.max(baseline.p))
    deviations: dict[str, float] = {}
    field_bit_exact = True
    for theta_dot in (1.0e2, 1.0e3, 1.0e4):
        state = solve_frc_equilibrium(_pressure_matched_inputs(theta_dot=theta_dot), rho)
        deviations[f"{theta_dot:.0e}"] = float(np.max(np.abs(state.p - baseline.p))) / peak
        field_bit_exact = field_bit_exact and bool(np.array_equal(state.B_z, baseline.B_z))
    # omega^2 scaling: each decade in omega multiplies the deviation by ~100.
    ratios = [
        deviations["1e+03"] / max(deviations["1e+02"], 1.0e-30),
        deviations["1e+04"] / max(deviations["1e+03"], 1.0e-30),
    ]
    quadratic = all(50.0 <= ratio <= 200.0 for ratio in ratios)
    passed = quadratic and field_bit_exact
    return {
        "passed": passed,
        "quadratic_scaling": quadratic,
        "field_bit_exact_with_no_rotation": field_bit_exact,
        "deviations": deviations,
        "decade_ratios": ratios,
    }


def _steinhauer_figure3_boundary() -> dict[str, Any]:
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
    status = cast(dict[str, Any], rotating_frc_bvp_acceptance_status())
    # The boundary holds when the implementation does NOT claim Steinhauer Figure 3
    # parity (its verbatim payload remains unavailable from the publisher).
    passed = has_pdf is False and status["steinhauer_figure3_parity_claimed"] is False
    return {
        "passed": passed,
        "reference_key": steinhauer["key"],
        "citation": steinhauer["citation"],
        "doi": steinhauer.get("doi"),
        "download_status": str(steinhauer.get("download_status", "missing")),
        "has_verified_pdf_payload": has_pdf,
        "figure3_parity_claimed": bool(status["steinhauer_figure3_parity_claimed"]),
        "notes": steinhauer.get("notes", ""),
    }


def _rust_parity(run_rust: bool) -> dict[str, Any]:
    if not run_rust:
        return {"status": "not_run", "passed": None, "reason": "run_rust=False"}
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
    try:
        completed = subprocess.run(
            command,
            cwd=RUST_WORKSPACE,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {
            "status": "unavailable",
            "passed": None,
            "reason": str(exc),
            "command": command,
        }
    payload = cast(dict[str, Any], json.loads(completed.stdout.strip()))
    payload["passed"] = (
        payload.get("rotating_bvp_implemented") is True
        and payload.get("no_rotation_reduction_passed") is True
        and payload.get("rotating_pressure_non_negative") is True
    )
    payload["command"] = command
    payload["stderr"] = completed.stderr.strip()
    return payload


def build_report(*, run_rust: bool = True) -> dict[str, Any]:
    """Build the FUS-C.1 rotating rigid-rotor acceptance report payload."""
    python_status = cast(dict[str, Any], rotating_frc_bvp_acceptance_status())
    no_rotation = _python_no_rotation_contract()
    rotating = _python_rotating_contract()
    reduction = _python_reduction_to_contract()
    figure3_boundary = _steinhauer_figure3_boundary()
    rust = _rust_parity(run_rust)
    rust_passed = rust["passed"] is True if run_rust else True
    accepted = (
        python_status["rotating_bvp_implemented"] is True
        and no_rotation["passed"] is True
        and rotating["passed"] is True
        and reduction["passed"] is True
        and figure3_boundary["passed"] is True
        and rust_passed
    )
    return {
        "schema": "frc-rotating-bvp-acceptance.v2",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "status": (
            "implemented_rostoker_qerushi_rotating_closure_accepted"
            if accepted
            else "failed_rotating_closure_acceptance_contract"
        ),
        "accepted_rotating_closure": accepted,
        "claim_boundary": python_status["claim_boundary"],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "rustc": _run_text(["rustc", "--version"]),
            "cargo": _run_text(["cargo", "--version"]),
        },
        "python_status": python_status,
        "python_no_rotation_contract": no_rotation,
        "python_rotating_contract": rotating,
        "python_reduction_to_contract": reduction,
        "rust_parity": rust,
        "steinhauer_figure3_boundary": figure3_boundary,
    }


def _write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# FRC Rotating Rigid-Rotor Acceptance",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Status: `{report['status']}`",
        f"- Accepted rotating closure: `{report['accepted_rotating_closure']}`",
        f"- Python: `{report['environment']['python']}`",
        f"- Rust: `{report['environment']['rustc']}`",
        "",
        "## Contract Checks",
        "",
        "| Check | Result | Evidence |",
        "|---|:---:|---|",
        "| Python status | `{}` | implemented=`{}` |".format(
            report["python_status"]["status"],
            report["python_status"]["rotating_bvp_implemented"],
        ),
        "| No-rotation contract | `{}` | residual `{:.3e}`, s `{:.3e}` |".format(
            report["python_no_rotation_contract"]["passed"],
            report["python_no_rotation_contract"]["residual"],
            report["python_no_rotation_contract"]["s_parameter"],
        ),
        "| Rotating equilibrium | `{}` | Mach `{:.3f}`, rot-FB `{:.3e}` |".format(
            report["python_rotating_contract"]["passed"],
            report["python_rotating_contract"]["rotation_mach_number"],
            report["python_rotating_contract"]["rotation_force_balance_residual_linf"],
        ),
        "| Reduces to contract (omega^2) | `{}` | ratios `{}` |".format(
            report["python_reduction_to_contract"]["passed"],
            [round(ratio, 1) for ratio in report["python_reduction_to_contract"]["decade_ratios"]],
        ),
        "| Rust parity | `{}` | `{}` |".format(
            report["rust_parity"]["passed"],
            report["rust_parity"]["status"],
        ),
        "| Steinhauer Fig. 3 boundary | `{}` | parity-claimed=`{}` |".format(
            report["steinhauer_figure3_boundary"]["passed"],
            report["steinhauer_figure3_boundary"]["figure3_parity_claimed"],
        ),
        "",
        "## Claim Boundary",
        "",
        report["claim_boundary"],
    ]
    MD_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run the rotating rigid-rotor acceptance report generator."""
    report = build_report(run_rust=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report)
    print(f"FRC rotating rigid-rotor acceptance {report['status']}: {MD_REPORT.relative_to(ROOT)}")
    return 0 if report["accepted_rotating_closure"] else 1


if __name__ == "__main__":
    sys.exit(main())
