# SPDX-License-Identifier: AGPL-3.0-or-later
"""Benchmark the native Grad-Shafranov operator/current closure contract.

This benchmark uses manufactured flux fields with an analytic cylindrical
Grad-Shafranov operator:

    psi(R, Z) = a R^4 + b Z^2 + c R^2 Z^2
    analytic Delta*psi = 8 a R^2 + 2 b + 2 c R^2
    centered-stencil Delta*psi = 8 a R^2 + 2 b + 2 c R^2 - 2 a dR^2
    J_phi = -Delta*psi / (mu0 R)

It validates the non-reduced operator-current relation rather than comparing
against an inverse EFIT reconstruction.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scpn_fusion.core.jax_gs_solver import (  # noqa: E402
    gs_delta_star_np,
    gs_toroidal_current_density_np,
    gs_total_toroidal_current_np,
)

MU0 = 4.0e-7 * np.pi
REPORT_DIR = ROOT / "validation" / "reports"
REPORT_JSON = REPORT_DIR / "gs_operator_current_closure.json"
REPORT_MD = REPORT_DIR / "gs_operator_current_closure.md"


def _machine_metadata() -> dict[str, Any]:
    uname = platform.uname()
    return {
        "platform": platform.platform(),
        "system": uname.system,
        "release": uname.release,
        "machine": uname.machine,
        "processor": uname.processor,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
    }


def _run_case(
    case_name: str,
    nr: int,
    nz: int,
    radial_coeff: float,
    vertical_coeff: float,
    mixed_coeff: float = 0.0,
) -> dict[str, Any]:
    r_min, r_max = 1.0, 3.0
    z_min, z_max = -1.0, 1.0
    r = np.linspace(r_min, r_max, nr)
    z = np.linspace(z_min, z_max, nz)
    rr, zz = np.meshgrid(r, z)
    psi = radial_coeff * rr**4 + vertical_coeff * zz**2 + mixed_coeff * rr**2 * zz**2
    dr = (r_max - r_min) / (nr - 1)
    dz = (z_max - z_min) / (nz - 1)

    start = time.perf_counter()
    delta_star = gs_delta_star_np(psi, r_min, r_max, z_min, z_max)
    current_density = gs_toroidal_current_density_np(psi, r_min, r_max, z_min, z_max)
    total_current = gs_total_toroidal_current_np(psi, r_min, r_max, z_min, z_max)
    elapsed_s = time.perf_counter() - start

    expected_analytic_delta = (
        8.0 * radial_coeff * rr[1:-1, 1:-1] ** 2
        + 2.0 * vertical_coeff
        + 2.0 * mixed_coeff * rr[1:-1, 1:-1] ** 2
    )
    expected_delta = expected_analytic_delta - 2.0 * radial_coeff * dr**2
    expected_density = np.broadcast_to(
        -expected_delta / (MU0 * rr[1:-1, 1:-1]),
        current_density[1:-1, 1:-1].shape,
    )
    expected_total = float(np.sum(expected_density) * dr * dz)

    delta_error = np.abs(delta_star[1:-1, 1:-1] - expected_delta)
    analytic_delta_error = np.abs(delta_star[1:-1, 1:-1] - expected_analytic_delta)
    current_error = np.abs(current_density[1:-1, 1:-1] - expected_density)
    current_scale = np.maximum(np.abs(expected_density), 1.0)
    total_scale = max(abs(expected_total), 1.0)

    return {
        "case": case_name,
        "nr": nr,
        "nz": nz,
        "radial_coeff": radial_coeff,
        "vertical_coeff": vertical_coeff,
        "mixed_coeff": mixed_coeff,
        "dr": dr,
        "elapsed_s": elapsed_s,
        "delta_star_max_abs_error": float(np.max(delta_error)),
        "analytic_delta_star_max_abs_error": float(np.max(analytic_delta_error)),
        "expected_second_order_radial_truncation": abs(2.0 * radial_coeff * dr**2),
        "current_density_max_abs_error": float(np.max(current_error)),
        "current_density_max_relative_error": float(np.max(current_error / current_scale)),
        "total_current": total_current,
        "expected_total_current": expected_total,
        "total_current_relative_error": abs(total_current - expected_total) / total_scale,
    }


def _radial_convergence_order(cases: list[dict[str, Any]]) -> float | None:
    """Estimate radial-quartic analytic-error convergence order from two finest grids."""
    radial_cases = [case for case in cases if case["case"].startswith("radial_quartic")]
    if len(radial_cases) < 2:
        return None
    radial_cases.sort(key=lambda case: case["dr"], reverse=True)
    for case in radial_cases:
        dr = float(case["dr"])
        error = float(case["analytic_delta_star_max_abs_error"])
        if not math.isfinite(dr) or dr <= 0.0:
            raise ValueError("radial-quartic rows require strictly decreasing positive dr")
        if not math.isfinite(error) or error <= 0.0:
            raise ValueError("radial-quartic rows require positive finite analytic Delta error")
    for coarse, fine in zip(radial_cases, radial_cases[1:]):
        if float(coarse["dr"]) <= float(fine["dr"]):
            raise ValueError("radial-quartic rows require strictly decreasing positive dr")
    coarse = radial_cases[-2]
    fine = radial_cases[-1]
    return float(
        math.log(
            coarse["analytic_delta_star_max_abs_error"] / fine["analytic_delta_star_max_abs_error"]
        )
        / math.log(coarse["dr"] / fine["dr"])
    )


def _radial_total_current_relative_error_max(cases: list[dict[str, Any]]) -> float:
    """Return the worst total-current closure error over radial-quartic grids."""
    radial_errors = [
        float(case["total_current_relative_error"])
        for case in cases
        if case["case"].startswith("radial_quartic")
    ]
    return max(radial_errors) if radial_errors else float("inf")


def build_gate_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Build fail-closed machine-readable gates for the operator/current contract."""
    cases = report.get("cases")
    thresholds = report.get("thresholds")
    case_thresholds_pass = (
        isinstance(cases, list)
        and len(cases) > 0
        and isinstance(thresholds, dict)
        and all(
            isinstance(case, dict)
            and float(case.get("delta_star_max_abs_error", float("inf")))
            <= float(thresholds.get("delta_star_max_abs_error", float("-inf")))
            and float(case.get("current_density_max_relative_error", float("inf")))
            <= float(thresholds.get("current_density_max_relative_error", float("-inf")))
            and float(case.get("total_current_relative_error", float("inf")))
            <= float(thresholds.get("total_current_relative_error", float("-inf")))
            for case in cases
        )
    )

    radial_order = report.get("radial_convergence_order")
    radial_convergence_pass = (
        isinstance(radial_order, int | float) and 1.99 <= float(radial_order) <= 2.01
    )
    radial_current_closure_pass = bool(report.get("radial_current_closure_stability_pass", False))

    gates = {
        "case_thresholds": case_thresholds_pass,
        "radial_convergence_order": radial_convergence_pass,
        "radial_total_current_closure_stability": radial_current_closure_pass,
    }
    failed_gates = [name for name, gate_passed in gates.items() if not gate_passed]
    return {
        "gates": gates,
        "gate_count": len(gates),
        "gate_pass_count": len(gates) - len(failed_gates),
        "failed_gates": failed_gates,
        "passes": not failed_gates,
    }


def _write_markdown(report: dict[str, Any]) -> None:
    gate_summary = report["gate_summary"]
    lines = [
        "# Grad-Shafranov Operator Current Closure Benchmark",
        "",
        f"Benchmark ID: `{report['benchmark_id']}`",
        f"Schema: `{report['schema_version']}`",
        f"Scope: `{report['benchmark_scope']}`",
        f"Solver mode: `{report['solver_mode']}`",
        f"Gates passed: `{gate_summary['gate_pass_count']}/{gate_summary['gate_count']}`",
        "",
        "Manufactured contracts: `psi(R, Z) = a R^4 + b Z^2 + c R^2 Z^2`, "
        "`Delta*psi = 8aR^2 + 2b + 2cR^2`, and `J_phi = -Delta*psi / (mu0 R)`.",
        "",
        "For radial-quartic terms the centered second-order stencil has exact "
        "discrete truncation `-2a dR^2`; the benchmark reports both the "
        "discrete-contract error and the analytic truncation magnitude.",
        "",
        "## Local machine",
        "",
    ]
    machine = report["machine"]
    for key in [
        "platform",
        "system",
        "release",
        "machine",
        "processor",
        "python",
        "numpy",
        "cpu_count",
    ]:
        lines.append(f"- `{key}`: `{machine[key]}`")
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Case | Grid | a | b | c | elapsed s | max discrete Delta* abs error | analytic Delta* error | max J rel error | total current rel error |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for case in report["cases"]:
        lines.append(
            f"| {case['case']} | {case['nr']}x{case['nz']} | "
            f"{case['radial_coeff']:.6g} | {case['vertical_coeff']:.6g} | "
            f"{case['mixed_coeff']:.6g} | "
            f"{case['elapsed_s']:.6e} | "
            f"{case['delta_star_max_abs_error']:.6e} | "
            f"{case['analytic_delta_star_max_abs_error']:.6e} | "
            f"{case['current_density_max_relative_error']:.6e} | "
            f"{case['total_current_relative_error']:.6e} |"
        )
    if report["radial_convergence_order"] is not None:
        lines.extend(
            [
                "",
                "## Radial-quartic convergence",
                "",
                "The radial-quartic analytic error measures the expected centered-stencil "
                "truncation against the continuous `Delta*` operator. The measured "
                f"order from the two finest radial grids is `{report['radial_convergence_order']:.6f}`.",
                "",
                "The radial-quartic total-current closure stability gate reports the "
                "worst relative current error across the refinement sequence: "
                f"`{report['radial_total_current_relative_error_max']:.6e}`.",
            ]
        )
    lines.extend(
        [
            "",
            f"Pass threshold: `{report['thresholds']}`.",
            f"Overall status: `{'PASS' if report['passes'] else 'FAIL'}`.",
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Run Grad-Shafranov current-closure benchmark and write reports.

    Returns ``0`` when all convergence and current-closure gates are satisfied.
    """
    thresholds = {
        "delta_star_max_abs_error": 1.0e-10,
        "current_density_max_relative_error": 1.0e-11,
        "total_current_relative_error": 1.0e-12,
        "radial_total_current_relative_error_max": 1.0e-12,
    }
    cases = [
        _run_case("vertical_quadratic", 17, 19, 0.0, -0.25),
        _run_case("radial_quartic_17", 17, 19, 0.03125, -0.125),
        _run_case("radial_quartic_33", 33, 35, 0.03125, -0.125),
        _run_case("radial_quartic_65", 65, 67, 0.03125, -0.125),
        _run_case("mixed_solovev", 29, 31, 0.0, -0.125, 0.05),
    ]
    radial_convergence_order = _radial_convergence_order(cases)
    radial_current_error_max = _radial_total_current_relative_error_max(cases)
    radial_current_closure_stability_pass = bool(
        radial_current_error_max <= thresholds["radial_total_current_relative_error_max"]
    )
    passed = (
        all(
            case["delta_star_max_abs_error"] <= thresholds["delta_star_max_abs_error"]
            and case["current_density_max_relative_error"]
            <= thresholds["current_density_max_relative_error"]
            and case["total_current_relative_error"] <= thresholds["total_current_relative_error"]
            for case in cases
        )
        and radial_convergence_order is not None
        and 1.99 <= radial_convergence_order <= 2.01
        and radial_current_closure_stability_pass
    )
    report = {
        "schema_version": "gs-operator-current-closure.v2",
        "benchmark_id": "gs_operator_current_closure",
        "benchmark": "gs_operator_current_closure",
        "benchmark_scope": "native_grad_shafranov_operator_current_closure",
        "physics_scope": "manufactured_full_order_grad_shafranov_operator_current_relation",
        "solver_mode": "manufactured_flux_operator_current_closure",
        "machine": _machine_metadata(),
        "thresholds": thresholds,
        "cases": cases,
        "radial_convergence_order": radial_convergence_order,
        "radial_total_current_relative_error_max": radial_current_error_max,
        "radial_current_closure_stability_pass": radial_current_closure_stability_pass,
        "passed": passed,
    }
    gate_summary = build_gate_summary(report)
    report["gate_summary"] = gate_summary
    report["passes"] = bool(gate_summary["passes"])
    report["passed"] = report["passes"]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if bool(report["passes"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
