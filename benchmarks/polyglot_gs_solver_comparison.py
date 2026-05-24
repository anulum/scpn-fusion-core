#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Polyglot Grad-Shafranov Benchmark
"""Benchmark native Python and Julia fixed-boundary Grad-Shafranov solvers."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from scpn_fusion.core.jax_gs_solver import gs_solve_np

_CASE_PATH = _REPO / "validation" / "polyglot" / "gs_picard_reference.toml"
_JULIA_PROJECT = _REPO / "scpn-fusion-jl"
_REPORT_JSON = _REPO / "validation" / "reports" / "polyglot_gs_solver_comparison.json"
_REPORT_MD = _REPO / "validation" / "reports" / "polyglot_gs_solver_comparison.md"


def _parse_scalar(value: str) -> int | float | str:
    cleaned = value.strip().strip('"')
    if cleaned.lower() in {"true", "false"}:
        return cleaned.lower() == "true"
    try:
        if any(marker in cleaned.lower() for marker in (".", "e")):
            return float(cleaned)
        return int(cleaned)
    except ValueError:
        return cleaned


def _read_case(path: Path) -> dict[str, Any]:
    section = None
    values: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section != "grad_shafranov" or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = _parse_scalar(value)
    return values


def _run_python(case: dict[str, Any]) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    psi = gs_solve_np(**case)
    return psi, time.perf_counter() - t0


def _run_julia() -> tuple[np.ndarray, float]:
    command = [
        "julia",
        f"--project={_JULIA_PROJECT}",
        "--startup-file=no",
        str(_JULIA_PROJECT / "bin" / "gs_picard_csv.jl"),
        str(_CASE_PATH),
    ]
    t0 = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=_REPO,
        check=True,
        text=True,
        capture_output=True,
    )
    elapsed = time.perf_counter() - t0
    rows = [[float(cell) for cell in row] for row in csv.reader(completed.stdout.splitlines())]
    return np.asarray(rows, dtype=float), elapsed


def _hardware_metadata() -> dict[str, str]:
    cpu_model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    try:
        julia_version = subprocess.run(
            ["julia", "--version"], check=True, text=True, capture_output=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        julia_version = f"unavailable: {exc}"
    return {
        "cpu_model": cpu_model,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "julia": julia_version,
        "os": platform.platform(),
    }


def main() -> None:
    case = _read_case(_CASE_PATH)
    python_psi, python_seconds = _run_python(case)
    julia_psi, julia_seconds = _run_julia()

    rel_l2 = float(np.linalg.norm(julia_psi[1:-1, 1:-1] - python_psi[1:-1, 1:-1])) / (
        float(np.linalg.norm(python_psi[1:-1, 1:-1])) + 1e-30
    )
    boundary_max = float(
        max(
            np.max(np.abs(julia_psi[0, :])),
            np.max(np.abs(julia_psi[-1, :])),
            np.max(np.abs(julia_psi[:, 0])),
            np.max(np.abs(julia_psi[:, -1])),
        )
    )

    report = {
        "_metadata": {
            "spdx_license": "AGPL-3.0-or-later",
            "commercial_license": "Commercial license available",
            "concepts_copyright": "Concepts 1996-2026 Miroslav Sotek. All rights reserved.",
            "code_copyright": "Code 2020-2026 Miroslav Sotek. All rights reserved.",
            "orcid": "0009-0009-3560-0851",
            "contact": "www.anulum.li | protoscience@anulum.li",
            "project": "SCPN Fusion Core - Polyglot Grad-Shafranov Benchmark",
        },
        "case": case,
        "hardware": _hardware_metadata(),
        "solvers": [
            {"language": "Python", "implementation": "gs_solve_np", "wall_time_s": python_seconds},
            {
                "language": "Julia",
                "implementation": "SCPNFusionSolvers.solve_grad_shafranov",
                "wall_time_s": julia_seconds,
            },
        ],
        "parity": {
            "relative_l2_interior": rel_l2,
            "julia_boundary_abs_max": boundary_max,
            "shape": list(julia_psi.shape),
        },
    }
    _REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "<!-- Commercial license available -->",
        "<!-- Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->",
        "<!-- Code 2020-2026 Miroslav Sotek. All rights reserved. -->",
        "<!-- ORCID: 0009-0009-3560-0851 -->",
        "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
        "<!-- SCPN Fusion Core - Polyglot Grad-Shafranov Benchmark -->",
        "",
        "# Polyglot Grad-Shafranov Solver Benchmark",
        "",
        "Local workstation benchmark for native Python and native Julia fixed-boundary Grad-Shafranov Picard/Jacobi solvers. The Julia path executes the `SCPNFusionSolvers` package directly; it is not a Python FFI wrapper.",
        "",
        "## Hardware",
        "",
        f"- CPU: {report['hardware']['cpu_model']}",
        f"- Machine: {report['hardware']['machine']}",
        f"- OS: {report['hardware']['os']}",
        f"- Python: {report['hardware']['python']}",
        f"- Julia: {report['hardware']['julia']}",
        "",
        "## Case",
        "",
        f"- Grid: {case['NZ']}x{case['NR']}",
        f"- Picard iterations: {case['n_picard']}",
        f"- Jacobi sweeps per Picard step: {case['n_jacobi']}",
        f"- Target plasma current: {case['Ip_target']:.6g} A",
        "",
        "## Timing",
        "",
        "| Language | Implementation | Wall time (s) |",
        "|----------|----------------|---------------|",
    ]
    for row in report["solvers"]:
        lines.append(f"| {row['language']} | `{row['implementation']}` | {row['wall_time_s']:.6f} |")
    lines.extend(
        [
            "",
            "## Numerical Parity",
            "",
            f"- Interior relative L2 error, Julia vs Python: {rel_l2:.6e}",
            f"- Julia boundary absolute maximum: {boundary_max:.6e}",
            "",
            "These local timings include process start-up for the Julia CLI path. Use cloud or long-lived process benchmarks for throughput comparisons.",
        ]
    )
    _REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(_REPORT_MD)
    print(json.dumps(report["parity"], sort_keys=True))


if __name__ == "__main__":
    main()
