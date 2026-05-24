#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Polyglot Grad-Shafranov Benchmark
"""Benchmark native Python, Julia, Go, Rust, and Lean Grad-Shafranov solvers."""

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
_GO_PROJECT = _REPO / "scpn-fusion-go"
_RUST_PROJECT = _REPO / "scpn-fusion-rs"
_LEAN_PROJECT = _REPO / "scpn-fusion-lean"
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


def _matrix_from_csv(stdout: str) -> np.ndarray:
    rows = [[float(cell) for cell in row] for row in csv.reader(stdout.splitlines())]
    return np.asarray(rows, dtype=float)


def _run_python(case: dict[str, Any]) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    psi = gs_solve_np(**case)
    return psi, time.perf_counter() - t0


def _run_command(command: list[str], cwd: Path) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return _matrix_from_csv(completed.stdout), time.perf_counter() - t0


def _run_julia() -> tuple[np.ndarray, float]:
    return _run_command(
        [
            "julia",
            f"--project={_JULIA_PROJECT}",
            "--startup-file=no",
            str(_JULIA_PROJECT / "bin" / "gs_picard_csv.jl"),
            str(_CASE_PATH),
        ],
        _REPO,
    )


def _run_go() -> tuple[np.ndarray, float]:
    return _run_command(["go", "run", "./cmd/gs_picard_csv", str(_CASE_PATH)], _GO_PROJECT)


def _run_rust() -> tuple[np.ndarray, float]:
    return _run_command(
        [
            "cargo",
            "run",
            "--release",
            "-q",
            "-p",
            "fusion-polyglot",
            "--bin",
            "gs_picard_csv",
            "--",
            str(_CASE_PATH),
        ],
        _RUST_PROJECT,
    )


def _run_lean() -> tuple[np.ndarray, float]:
    return _run_command(["lake", "exe", "gs_picard_csv", str(_CASE_PATH)], _LEAN_PROJECT)


def _tool_version(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"


def _hardware_metadata() -> dict[str, str]:
    cpu_model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "cpu_model": cpu_model,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "julia": _tool_version(["julia", "--version"]),
        "go": _tool_version(["go", "version"]),
        "rust": _tool_version(["rustc", "--version"]),
        "lean": _tool_version(["lean", "--version"]),
        "os": platform.platform(),
    }


def _boundary_abs_max(psi: np.ndarray) -> float:
    return float(
        max(
            np.max(np.abs(psi[0, :])),
            np.max(np.abs(psi[-1, :])),
            np.max(np.abs(psi[:, 0])),
            np.max(np.abs(psi[:, -1])),
        )
    )


def _relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference[1:-1, 1:-1])) + 1e-30
    return float(np.linalg.norm(candidate[1:-1, 1:-1] - reference[1:-1, 1:-1])) / denominator


def main() -> None:
    case = _read_case(_CASE_PATH)
    python_psi, python_seconds = _run_python(case)
    julia_psi, julia_seconds = _run_julia()
    go_psi, go_seconds = _run_go()
    rust_psi, rust_seconds = _run_rust()
    lean_psi, lean_seconds = _run_lean()

    parity_by_language = {
        "Julia": {
            "relative_l2_interior": _relative_l2(julia_psi, python_psi),
            "boundary_abs_max": _boundary_abs_max(julia_psi),
        },
        "Go": {
            "relative_l2_interior": _relative_l2(go_psi, python_psi),
            "boundary_abs_max": _boundary_abs_max(go_psi),
        },
        "Rust": {
            "relative_l2_interior": _relative_l2(rust_psi, python_psi),
            "boundary_abs_max": _boundary_abs_max(rust_psi),
        },
        "Lean": {
            "relative_l2_interior": _relative_l2(lean_psi, python_psi),
            "boundary_abs_max": _boundary_abs_max(lean_psi),
        },
    }

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
            {"language": "Go", "implementation": "gssolver.Solve", "wall_time_s": go_seconds},
            {
                "language": "Rust",
                "implementation": "fusion_polyglot::solve_grad_shafranov",
                "wall_time_s": rust_seconds,
            },
            {
                "language": "Lean",
                "implementation": "SCPNFusionSolvers.solveGradShafranov",
                "wall_time_s": lean_seconds,
            },
        ],
        "parity": {"by_language": parity_by_language, "shape": list(python_psi.shape)},
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
        "Local workstation benchmark for native Python, Julia, Go, Rust, and Lean fixed-boundary Grad-Shafranov Picard/Jacobi solvers. Each non-Python path executes its own implementation rather than a Python FFI wrapper.",
        "",
        "## Hardware",
        "",
        f"- CPU: {report['hardware']['cpu_model']}",
        f"- Machine: {report['hardware']['machine']}",
        f"- OS: {report['hardware']['os']}",
        f"- Python: {report['hardware']['python']}",
        f"- Julia: {report['hardware']['julia']}",
        f"- Go: {report['hardware']['go']}",
        f"- Rust: {report['hardware']['rust']}",
        f"- Lean: {report['hardware']['lean']}",
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
            "| Language | Interior relative L2 vs Python | Boundary absolute maximum |",
            "|----------|--------------------------------|---------------------------|",
        ]
    )
    for language, parity in parity_by_language.items():
        lines.append(
            f"| {language} | {parity['relative_l2_interior']:.6e} | {parity['boundary_abs_max']:.6e} |"
        )
    lines.extend(
        [
            "",
            "These local timings include process start-up and compilation-cache checks for CLI paths. Use long-lived processes or cloud CPU/GPU runners for throughput comparisons.",
        ]
    )
    _REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(_REPORT_MD)
    print(json.dumps(report["parity"], sort_keys=True))


if __name__ == "__main__":
    main()
