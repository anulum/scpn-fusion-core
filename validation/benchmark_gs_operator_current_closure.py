# SPDX-License-Identifier: AGPL-3.0-or-later
"""Benchmark the native Grad-Shafranov operator/current closure contract.

This benchmark uses manufactured flux fields with an analytic cylindrical
Grad-Shafranov operator:

    psi(R, Z) = a R^4 + b Z^2
    analytic Delta*psi = 8 a R^2 + 2 b
    centered-stencil Delta*psi = 8 a R^2 + 2 b - 2 a dR^2
    J_phi = -Delta*psi / (mu0 R)

It validates the non-reduced operator-current relation rather than comparing
against an inverse EFIT reconstruction.
"""

from __future__ import annotations

import json
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
) -> dict[str, Any]:
    r_min, r_max = 1.0, 3.0
    z_min, z_max = -1.0, 1.0
    r = np.linspace(r_min, r_max, nr)
    z = np.linspace(z_min, z_max, nz)
    rr, zz = np.meshgrid(r, z)
    psi = radial_coeff * rr**4 + vertical_coeff * zz**2
    dr = (r_max - r_min) / (nr - 1)
    dz = (z_max - z_min) / (nz - 1)

    start = time.perf_counter()
    delta_star = gs_delta_star_np(psi, r_min, r_max, z_min, z_max)
    current_density = gs_toroidal_current_density_np(psi, r_min, r_max, z_min, z_max)
    total_current = gs_total_toroidal_current_np(psi, r_min, r_max, z_min, z_max)
    elapsed_s = time.perf_counter() - start

    expected_analytic_delta = 8.0 * radial_coeff * rr[1:-1, 1:-1] ** 2 + 2.0 * vertical_coeff
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


def _write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Grad-Shafranov Operator Current Closure Benchmark",
        "",
        "Manufactured contracts: `psi(R, Z) = a R^4 + b Z^2`, "
        "`Delta*psi = 8aR^2 + 2b`, and `J_phi = -Delta*psi / (mu0 R)`.",
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
            "| Case | Grid | a | b | elapsed s | max discrete Delta* abs error | analytic Delta* error | max J rel error | total current rel error |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for case in report["cases"]:
        lines.append(
            f"| {case['case']} | {case['nr']}x{case['nz']} | "
            f"{case['radial_coeff']:.6g} | {case['vertical_coeff']:.6g} | "
            f"{case['elapsed_s']:.6e} | "
            f"{case['delta_star_max_abs_error']:.6e} | "
            f"{case['analytic_delta_star_max_abs_error']:.6e} | "
            f"{case['current_density_max_relative_error']:.6e} | "
            f"{case['total_current_relative_error']:.6e} |"
        )
    lines.extend(
        [
            "",
            f"Pass threshold: `{report['thresholds']}`.",
            f"Overall status: `{'PASS' if report['passed'] else 'FAIL'}`.",
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    thresholds = {
        "delta_star_max_abs_error": 1.0e-10,
        "current_density_max_relative_error": 1.0e-12,
        "total_current_relative_error": 1.0e-12,
    }
    cases = [
        _run_case("vertical_quadratic", 17, 19, 0.0, -0.25),
        _run_case("radial_quartic", 33, 35, 0.03125, -0.125),
    ]
    passed = all(
        case["delta_star_max_abs_error"] <= thresholds["delta_star_max_abs_error"]
        and case["current_density_max_relative_error"]
        <= thresholds["current_density_max_relative_error"]
        and case["total_current_relative_error"] <= thresholds["total_current_relative_error"]
        for case in cases
    )
    report = {
        "benchmark": "gs_operator_current_closure",
        "machine": _machine_metadata(),
        "thresholds": thresholds,
        "cases": cases,
        "passed": passed,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
