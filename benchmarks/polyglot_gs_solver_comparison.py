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
import hashlib
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from scpn_fusion.core.jax_gs_solver import gs_equation_residual_np, gs_solve_np
from scpn_fusion.core.physical_case import case_from_toml

FloatArray = NDArray[np.float64]

_CASE_PATH = _REPO / "validation" / "polyglot" / "gs_picard_reference.toml"
_JULIA_PROJECT = _REPO / "scpn-fusion-jl"
_GO_PROJECT = _REPO / "scpn-fusion-go"
_RUST_PROJECT = _REPO / "scpn-fusion-rs"
_RUST_RELEASE_BINARY = _RUST_PROJECT / "target" / "release" / "gs_picard_csv"
_LEAN_PROJECT = _REPO / "scpn-fusion-lean"
_REPORT_JSON = _REPO / "validation" / "reports" / "polyglot_gs_solver_comparison.json"
_REPORT_MD = _REPO / "validation" / "reports" / "polyglot_gs_solver_comparison.md"


def _read_case(path: Path) -> dict[str, Any]:
    """Load the exact common typed physical case contract before benchmarking."""
    return dict(case_from_toml(path).as_mapping())


def _matrix_from_csv(stdout: str) -> FloatArray:
    """Decode the actual native solver CSV into a float64 matrix."""
    rows = [[float(cell) for cell in row] for row in csv.reader(stdout.splitlines())]
    return np.asarray(rows, dtype=float)


def _run_python(case: dict[str, Any]) -> tuple[FloatArray, float]:
    """Time the public NumPy physical solver with the admitted case."""
    t0 = time.perf_counter()
    psi = gs_solve_np(**case)
    return psi, time.perf_counter() - t0


def _run_command(command: list[str], cwd: Path) -> tuple[FloatArray, float]:
    """Time a real native CLI and require successful CSV output."""
    t0 = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return _matrix_from_csv(completed.stdout), time.perf_counter() - t0


def _run_julia(case_path: Path = _CASE_PATH) -> tuple[FloatArray, float]:
    """Execute the Julia project solver with the selected physical deck."""
    return _run_command(
        [
            "julia",
            f"--project={_JULIA_PROJECT}",
            "--startup-file=no",
            str(_JULIA_PROJECT / "bin" / "gs_picard_csv.jl"),
            str(case_path),
        ],
        _REPO,
    )


def _run_go(case_path: Path = _CASE_PATH) -> tuple[FloatArray, float]:
    """Build Go outside the measurement and time the actual solver binary."""
    with tempfile.TemporaryDirectory() as build_dir:
        binary = Path(build_dir) / "gs_picard_csv"
        subprocess.run(
            ["go", "build", "-o", str(binary), "./cmd/gs_picard_csv"],
            cwd=_GO_PROJECT,
            check=True,
            text=True,
            capture_output=True,
        )
        return _run_command([str(binary), str(case_path)], _GO_PROJECT)


def _build_rust_binary() -> None:
    """Build the optimized native Rust physical solver before timing."""
    subprocess.run(
        ["cargo", "build", "--release", "-q", "-p", "fusion-polyglot"],
        cwd=_RUST_PROJECT,
        check=True,
        text=True,
        capture_output=True,
    )


def _run_rust(case_path: Path = _CASE_PATH) -> tuple[FloatArray, float]:
    """Time the optimized Rust solver against the selected physical deck."""
    _build_rust_binary()
    return _run_command([str(_RUST_RELEASE_BINARY), str(case_path)], _RUST_PROJECT)


def _run_lean(case_path: Path = _CASE_PATH) -> tuple[FloatArray, float]:
    """Run the pinned Lean project executable with the selected physical deck."""
    return _run_command(["lake", "exe", "gs_picard_csv", str(case_path)], _LEAN_PROJECT)


def _tool_version(command: list[str], cwd: Path | None = None) -> str:
    """Read a real tool version or return its actual availability failure."""
    try:
        return subprocess.run(
            command, cwd=cwd, check=True, text=True, capture_output=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"


def _hardware_metadata() -> dict[str, str]:
    """Record actual host and solver toolchain versions for timing provenance."""
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
        "lean": _tool_version(["lake", "env", "lean", "--version"], _LEAN_PROJECT),
        "os": platform.platform(),
    }


def _boundary_abs_max(psi: FloatArray) -> float:
    """Measure the largest absolute flux on the complete boundary ring."""
    return float(
        max(
            np.max(np.abs(psi[0, :])),
            np.max(np.abs(psi[-1, :])),
            np.max(np.abs(psi[:, 0])),
            np.max(np.abs(psi[:, -1])),
        )
    )


def _vertical_symmetry_abs_max(psi: FloatArray) -> float:
    """Measure vertical reflection error across the full flux grid."""
    return float(np.max(np.abs(psi - np.flipud(psi))))


def _axis_midplane_offset_cells(psi: FloatArray) -> int:
    """Measure the interior flux-axis offset from the vertical midplane."""
    axis_z_index = int(np.unravel_index(np.argmax(psi), psi.shape)[0])
    midplane_index = psi.shape[0] // 2
    return abs(axis_z_index - midplane_index)


def _axis_radial_center_offset_cells(psi: FloatArray) -> int:
    """Measure the interior flux-axis offset from the radial grid center."""
    axis_r_index = int(np.unravel_index(np.argmax(psi), psi.shape)[1])
    radial_center_index = psi.shape[1] // 2
    return abs(axis_r_index - radial_center_index)


def _axis_boundary_distance_cells(psi: FloatArray) -> int:
    """Measure the nearest boundary distance of the interior flux maximum."""
    axis_z_index, axis_r_index = np.unravel_index(np.argmax(psi), psi.shape)
    return int(
        min(
            axis_z_index,
            axis_r_index,
            psi.shape[0] - 1 - axis_z_index,
            psi.shape[1] - 1 - axis_r_index,
        )
    )


def _axis_local_dominance_margin(psi: FloatArray) -> float:
    """Compare peak flux with its direct interior neighbors."""
    axis_z_index, axis_r_index = np.unravel_index(np.argmax(psi), psi.shape)
    axis_value = float(psi[axis_z_index, axis_r_index])
    neighbor_values = [
        float(psi[axis_z_index - 1, axis_r_index]),
        float(psi[axis_z_index + 1, axis_r_index]),
        float(psi[axis_z_index, axis_r_index - 1]),
        float(psi[axis_z_index, axis_r_index + 1]),
    ]
    return axis_value - max(neighbor_values)


def _axis_discrete_laplacian(psi: FloatArray) -> float:
    """Measure the discrete Laplacian at the interior flux maximum."""
    axis_z_index, axis_r_index = np.unravel_index(np.argmax(psi), psi.shape)
    return float(
        psi[axis_z_index - 1, axis_r_index]
        + psi[axis_z_index + 1, axis_r_index]
        + psi[axis_z_index, axis_r_index - 1]
        + psi[axis_z_index, axis_r_index + 1]
        - 4.0 * psi[axis_z_index, axis_r_index]
    )


def _axis_flux_value(psi: FloatArray) -> float:
    """Return the largest interior poloidal flux value."""
    return float(np.max(psi))


def _midplane_radial_monotonicity_violations(psi: FloatArray) -> int:
    """Count departures from radial monotonicity toward the midplane peak."""
    axis_z_index, axis_r_index = np.unravel_index(np.argmax(psi), psi.shape)
    midplane = psi[axis_z_index, :]
    violations = 0
    for ir in range(1, axis_r_index + 1):
        if midplane[ir] + 1e-14 < midplane[ir - 1]:
            violations += 1
    for ir in range(axis_r_index + 1, psi.shape[1]):
        if midplane[ir] > midplane[ir - 1] + 1e-14:
            violations += 1
    return violations


def _axis_column_vertical_monotonicity_violations(psi: FloatArray) -> int:
    """Count vertical monotonicity departures along the peak column."""
    axis_z_index, axis_r_index = np.unravel_index(np.argmax(psi), psi.shape)
    axis_column = psi[:, axis_r_index]
    violations = 0
    for iz in range(1, axis_z_index + 1):
        if axis_column[iz] + 1e-14 < axis_column[iz - 1]:
            violations += 1
    for iz in range(axis_z_index + 1, psi.shape[0]):
        if axis_column[iz] > axis_column[iz - 1] + 1e-14:
            violations += 1
    return violations


def _negative_flux_abs_max(psi: FloatArray) -> float:
    """Measure the largest magnitude of negative interior flux."""
    return max(0.0, -float(np.min(psi)))


def _gs_equation_residual_abs_max(psi: FloatArray, case: dict[str, Any]) -> float:
    """Evaluate the public reference equation residual over the interior."""
    return gs_equation_residual_np(
        psi,
        float(case["R_min"]),
        float(case["R_max"]),
        float(case["Z_min"]),
        float(case["Z_max"]),
        int(case["NR"]),
        int(case["NZ"]),
        float(case["Ip_target"]),
        float(case["mu0"]),
        float(case["beta_mix"]),
    )["abs_max"]


def _gs_equation_residual_relative_max(psi: FloatArray, case: dict[str, Any]) -> float:
    """Normalize the equation residual by the source magnitude."""
    return gs_equation_residual_np(
        psi,
        float(case["R_min"]),
        float(case["R_max"]),
        float(case["Z_min"]),
        float(case["Z_max"]),
        int(case["NR"]),
        int(case["NZ"]),
        float(case["Ip_target"]),
        float(case["mu0"]),
        float(case["beta_mix"]),
    )["relative_max"]


def _relative_l2(candidate: FloatArray, reference: FloatArray) -> float:
    """Measure interior L2 error normalized by the reference flux norm."""
    denominator = float(np.linalg.norm(reference[1:-1, 1:-1])) + 1e-30
    return float(np.linalg.norm(candidate[1:-1, 1:-1] - reference[1:-1, 1:-1])) / denominator


def _interior_max_abs_error(candidate: FloatArray, reference: FloatArray) -> float:
    """Measure the largest interior absolute error against the reference."""
    return float(np.max(np.abs(candidate[1:-1, 1:-1] - reference[1:-1, 1:-1])))


def main(
    *,
    case_path: Path = _CASE_PATH,
    report_json: Path = _REPORT_JSON,
    report_md: Path = _REPORT_MD,
) -> None:
    """Run all five real solver CLIs and emit timing, parity and provenance reports."""
    case_path = case_path.resolve()
    case: dict[str, Any] = _read_case(case_path)
    python_psi, python_seconds = _run_python(case)
    julia_psi, julia_seconds = _run_julia(case_path)
    go_psi, go_seconds = _run_go(case_path)
    rust_psi, rust_seconds = _run_rust(case_path)
    lean_psi, lean_seconds = _run_lean(case_path)

    parity_by_language = {
        "Julia": {
            "relative_l2_interior": _relative_l2(julia_psi, python_psi),
            "max_abs_error_interior": _interior_max_abs_error(julia_psi, python_psi),
            "axis_flux_abs_error": abs(_axis_flux_value(julia_psi) - _axis_flux_value(python_psi)),
            "boundary_abs_max": _boundary_abs_max(julia_psi),
            "vertical_symmetry_abs_max": _vertical_symmetry_abs_max(julia_psi),
            "axis_midplane_offset_cells": _axis_midplane_offset_cells(julia_psi),
            "axis_radial_center_offset_cells": _axis_radial_center_offset_cells(julia_psi),
            "axis_boundary_distance_cells": _axis_boundary_distance_cells(julia_psi),
            "axis_local_dominance_margin": _axis_local_dominance_margin(julia_psi),
            "axis_discrete_laplacian": _axis_discrete_laplacian(julia_psi),
            "midplane_radial_monotonicity_violations": _midplane_radial_monotonicity_violations(
                julia_psi
            ),
            "axis_column_vertical_monotonicity_violations": _axis_column_vertical_monotonicity_violations(
                julia_psi
            ),
            "negative_flux_abs_max": _negative_flux_abs_max(julia_psi),
            "gs_equation_residual_abs_max": _gs_equation_residual_abs_max(julia_psi, case),
            "gs_equation_residual_relative_max": _gs_equation_residual_relative_max(
                julia_psi, case
            ),
        },
        "Go": {
            "relative_l2_interior": _relative_l2(go_psi, python_psi),
            "max_abs_error_interior": _interior_max_abs_error(go_psi, python_psi),
            "axis_flux_abs_error": abs(_axis_flux_value(go_psi) - _axis_flux_value(python_psi)),
            "boundary_abs_max": _boundary_abs_max(go_psi),
            "vertical_symmetry_abs_max": _vertical_symmetry_abs_max(go_psi),
            "axis_midplane_offset_cells": _axis_midplane_offset_cells(go_psi),
            "axis_radial_center_offset_cells": _axis_radial_center_offset_cells(go_psi),
            "axis_boundary_distance_cells": _axis_boundary_distance_cells(go_psi),
            "axis_local_dominance_margin": _axis_local_dominance_margin(go_psi),
            "axis_discrete_laplacian": _axis_discrete_laplacian(go_psi),
            "midplane_radial_monotonicity_violations": _midplane_radial_monotonicity_violations(
                go_psi
            ),
            "axis_column_vertical_monotonicity_violations": _axis_column_vertical_monotonicity_violations(
                go_psi
            ),
            "negative_flux_abs_max": _negative_flux_abs_max(go_psi),
            "gs_equation_residual_abs_max": _gs_equation_residual_abs_max(go_psi, case),
            "gs_equation_residual_relative_max": _gs_equation_residual_relative_max(go_psi, case),
        },
        "Rust": {
            "relative_l2_interior": _relative_l2(rust_psi, python_psi),
            "max_abs_error_interior": _interior_max_abs_error(rust_psi, python_psi),
            "axis_flux_abs_error": abs(_axis_flux_value(rust_psi) - _axis_flux_value(python_psi)),
            "boundary_abs_max": _boundary_abs_max(rust_psi),
            "vertical_symmetry_abs_max": _vertical_symmetry_abs_max(rust_psi),
            "axis_midplane_offset_cells": _axis_midplane_offset_cells(rust_psi),
            "axis_radial_center_offset_cells": _axis_radial_center_offset_cells(rust_psi),
            "axis_boundary_distance_cells": _axis_boundary_distance_cells(rust_psi),
            "axis_local_dominance_margin": _axis_local_dominance_margin(rust_psi),
            "axis_discrete_laplacian": _axis_discrete_laplacian(rust_psi),
            "midplane_radial_monotonicity_violations": _midplane_radial_monotonicity_violations(
                rust_psi
            ),
            "axis_column_vertical_monotonicity_violations": _axis_column_vertical_monotonicity_violations(
                rust_psi
            ),
            "negative_flux_abs_max": _negative_flux_abs_max(rust_psi),
            "gs_equation_residual_abs_max": _gs_equation_residual_abs_max(rust_psi, case),
            "gs_equation_residual_relative_max": _gs_equation_residual_relative_max(rust_psi, case),
        },
        "Lean": {
            "relative_l2_interior": _relative_l2(lean_psi, python_psi),
            "max_abs_error_interior": _interior_max_abs_error(lean_psi, python_psi),
            "axis_flux_abs_error": abs(_axis_flux_value(lean_psi) - _axis_flux_value(python_psi)),
            "boundary_abs_max": _boundary_abs_max(lean_psi),
            "vertical_symmetry_abs_max": _vertical_symmetry_abs_max(lean_psi),
            "axis_midplane_offset_cells": _axis_midplane_offset_cells(lean_psi),
            "axis_radial_center_offset_cells": _axis_radial_center_offset_cells(lean_psi),
            "axis_boundary_distance_cells": _axis_boundary_distance_cells(lean_psi),
            "axis_local_dominance_margin": _axis_local_dominance_margin(lean_psi),
            "axis_discrete_laplacian": _axis_discrete_laplacian(lean_psi),
            "midplane_radial_monotonicity_violations": _midplane_radial_monotonicity_violations(
                lean_psi
            ),
            "axis_column_vertical_monotonicity_violations": _axis_column_vertical_monotonicity_violations(
                lean_psi
            ),
            "negative_flux_abs_max": _negative_flux_abs_max(lean_psi),
            "gs_equation_residual_abs_max": _gs_equation_residual_abs_max(lean_psi, case),
            "gs_equation_residual_relative_max": _gs_equation_residual_relative_max(lean_psi, case),
        },
    }

    report: dict[str, Any] = {
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
            {
                "language": "Python",
                "implementation": "gs_solve_np",
                "wall_time_s": python_seconds,
                "axis_flux_value": _axis_flux_value(python_psi),
                "vertical_symmetry_abs_max": _vertical_symmetry_abs_max(python_psi),
                "axis_midplane_offset_cells": _axis_midplane_offset_cells(python_psi),
                "axis_radial_center_offset_cells": _axis_radial_center_offset_cells(python_psi),
                "axis_boundary_distance_cells": _axis_boundary_distance_cells(python_psi),
                "axis_local_dominance_margin": _axis_local_dominance_margin(python_psi),
                "axis_discrete_laplacian": _axis_discrete_laplacian(python_psi),
                "gs_equation_residual_abs_max": _gs_equation_residual_abs_max(python_psi, case),
                "gs_equation_residual_relative_max": _gs_equation_residual_relative_max(
                    python_psi, case
                ),
                "midplane_radial_monotonicity_violations": _midplane_radial_monotonicity_violations(
                    python_psi
                ),
                "axis_column_vertical_monotonicity_violations": _axis_column_vertical_monotonicity_violations(
                    python_psi
                ),
                "negative_flux_abs_max": _negative_flux_abs_max(python_psi),
            },
            {
                "language": "Julia",
                "implementation": "SCPNFusionSolvers.solve_grad_shafranov",
                "wall_time_s": julia_seconds,
                "axis_flux_value": _axis_flux_value(julia_psi),
                "vertical_symmetry_abs_max": _vertical_symmetry_abs_max(julia_psi),
                "axis_midplane_offset_cells": _axis_midplane_offset_cells(julia_psi),
                "axis_radial_center_offset_cells": _axis_radial_center_offset_cells(julia_psi),
                "axis_boundary_distance_cells": _axis_boundary_distance_cells(julia_psi),
                "axis_local_dominance_margin": _axis_local_dominance_margin(julia_psi),
                "axis_discrete_laplacian": _axis_discrete_laplacian(julia_psi),
                "gs_equation_residual_abs_max": _gs_equation_residual_abs_max(julia_psi, case),
                "gs_equation_residual_relative_max": _gs_equation_residual_relative_max(
                    julia_psi, case
                ),
                "midplane_radial_monotonicity_violations": _midplane_radial_monotonicity_violations(
                    julia_psi
                ),
                "axis_column_vertical_monotonicity_violations": _axis_column_vertical_monotonicity_violations(
                    julia_psi
                ),
                "negative_flux_abs_max": _negative_flux_abs_max(julia_psi),
            },
            {
                "language": "Go",
                "implementation": "gssolver.Solve",
                "wall_time_s": go_seconds,
                "axis_flux_value": _axis_flux_value(go_psi),
                "vertical_symmetry_abs_max": _vertical_symmetry_abs_max(go_psi),
                "axis_midplane_offset_cells": _axis_midplane_offset_cells(go_psi),
                "axis_radial_center_offset_cells": _axis_radial_center_offset_cells(go_psi),
                "axis_boundary_distance_cells": _axis_boundary_distance_cells(go_psi),
                "axis_local_dominance_margin": _axis_local_dominance_margin(go_psi),
                "axis_discrete_laplacian": _axis_discrete_laplacian(go_psi),
                "gs_equation_residual_abs_max": _gs_equation_residual_abs_max(go_psi, case),
                "gs_equation_residual_relative_max": _gs_equation_residual_relative_max(
                    go_psi, case
                ),
                "midplane_radial_monotonicity_violations": _midplane_radial_monotonicity_violations(
                    go_psi
                ),
                "axis_column_vertical_monotonicity_violations": _axis_column_vertical_monotonicity_violations(
                    go_psi
                ),
                "negative_flux_abs_max": _negative_flux_abs_max(go_psi),
            },
            {
                "language": "Rust",
                "implementation": "fusion_polyglot::solve_grad_shafranov",
                "wall_time_s": rust_seconds,
                "axis_flux_value": _axis_flux_value(rust_psi),
                "vertical_symmetry_abs_max": _vertical_symmetry_abs_max(rust_psi),
                "axis_midplane_offset_cells": _axis_midplane_offset_cells(rust_psi),
                "axis_radial_center_offset_cells": _axis_radial_center_offset_cells(rust_psi),
                "axis_boundary_distance_cells": _axis_boundary_distance_cells(rust_psi),
                "axis_local_dominance_margin": _axis_local_dominance_margin(rust_psi),
                "axis_discrete_laplacian": _axis_discrete_laplacian(rust_psi),
                "gs_equation_residual_abs_max": _gs_equation_residual_abs_max(rust_psi, case),
                "gs_equation_residual_relative_max": _gs_equation_residual_relative_max(
                    rust_psi, case
                ),
                "midplane_radial_monotonicity_violations": _midplane_radial_monotonicity_violations(
                    rust_psi
                ),
                "axis_column_vertical_monotonicity_violations": _axis_column_vertical_monotonicity_violations(
                    rust_psi
                ),
                "negative_flux_abs_max": _negative_flux_abs_max(rust_psi),
            },
            {
                "language": "Lean",
                "implementation": "SCPNFusionSolvers.solveGradShafranov",
                "wall_time_s": lean_seconds,
                "axis_flux_value": _axis_flux_value(lean_psi),
                "vertical_symmetry_abs_max": _vertical_symmetry_abs_max(lean_psi),
                "axis_midplane_offset_cells": _axis_midplane_offset_cells(lean_psi),
                "axis_radial_center_offset_cells": _axis_radial_center_offset_cells(lean_psi),
                "axis_boundary_distance_cells": _axis_boundary_distance_cells(lean_psi),
                "axis_local_dominance_margin": _axis_local_dominance_margin(lean_psi),
                "axis_discrete_laplacian": _axis_discrete_laplacian(lean_psi),
                "gs_equation_residual_abs_max": _gs_equation_residual_abs_max(lean_psi, case),
                "gs_equation_residual_relative_max": _gs_equation_residual_relative_max(
                    lean_psi, case
                ),
                "midplane_radial_monotonicity_violations": _midplane_radial_monotonicity_violations(
                    lean_psi
                ),
                "axis_column_vertical_monotonicity_violations": _axis_column_vertical_monotonicity_violations(
                    lean_psi
                ),
                "negative_flux_abs_max": _negative_flux_abs_max(lean_psi),
            },
        ],
        "parity": {"by_language": parity_by_language, "shape": list(python_psi.shape)},
    }
    report["case_sha256"] = hashlib.sha256(case_path.read_bytes()).hexdigest()
    source_paths = (
        "benchmarks/polyglot_gs_solver_comparison.py",
        "src/scpn_fusion/core/physical_case.py",
        "src/scpn_fusion/core/jax_gs_solver.py",
        "scpn-fusion-go/gssolver/case.go",
        "scpn-fusion-go/gssolver/solver.go",
        "scpn-fusion-go/go.mod",
        "scpn-fusion-go/go.sum",
        "scpn-fusion-jl/src/physical_case.jl",
        "scpn-fusion-jl/src/SCPNFusionSolvers.jl",
        "scpn-fusion-rs/crates/fusion-polyglot/src/case.rs",
        "scpn-fusion-rs/crates/fusion-polyglot/src/lib.rs",
        "scpn-fusion-rs/crates/fusion-polyglot/Cargo.toml",
        "scpn-fusion-rs/Cargo.lock",
        "scpn-fusion-lean/SCPNFusionSolvers.lean",
        "scpn-fusion-lean/SCPNFusionSolvers/PhysicalCase.lean",
        "scpn-fusion-lean/SCPNFusionSolvers/CSV.lean",
        "scpn-fusion-lean/Main.lean",
        "scpn-fusion-lean/lean-toolchain",
    )
    report["source_sha256"] = {
        path: hashlib.sha256((_REPO / path).read_bytes()).hexdigest() for path in source_paths
    }
    solvers = cast(list[dict[str, Any]], report["solvers"])
    hardware = cast(dict[str, Any], report["hardware"])
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

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
        f"- CPU: {hardware['cpu_model']}",
        f"- Machine: {hardware['machine']}",
        f"- OS: {hardware['os']}",
        f"- Python: {hardware['python']}",
        f"- Julia: {hardware['julia']}",
        f"- Go: {hardware['go']}",
        f"- Rust: {hardware['rust']}",
        f"- Lean: {hardware['lean']}",
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
    for row in solvers:
        lines.append(
            f"| {row['language']} | `{row['implementation']}` | {row['wall_time_s']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Numerical Parity",
            "",
            "| Language | Interior relative L2 vs Python | Interior max abs error vs Python | Axis flux abs error vs Python | Boundary absolute maximum |",
            "|----------|--------------------------------|----------------------------------|-------------------------------|---------------------------|",
        ]
    )
    for language, parity in parity_by_language.items():
        lines.append(
            f"| {language} | {parity['relative_l2_interior']:.6e} | "
            f"{parity['max_abs_error_interior']:.6e} | {parity['axis_flux_abs_error']:.6e} | "
            f"{parity['boundary_abs_max']:.6e} |"
        )
    lines.extend(
        [
            "",
            "## Physics Invariants",
            "",
            "| Language | Axis flux value | Vertical symmetry absolute maximum | Axis midplane offset (cells) | Axis radial-center offset (cells) | Axis boundary distance (cells) | Axis local dominance margin | Axis discrete Laplacian | GS residual absolute maximum | GS residual relative maximum | Midplane radial monotonicity violations | Axis-column vertical monotonicity violations | Negative flux absolute maximum |",
            "|----------|-----------------|------------------------------------|------------------------------|------------------------------------|--------------------------------|------------------------------|--------------------------|------------------------------|------------------------------|-------------------------------------------|-----------------------------------------------|--------------------------------|",
        ]
    )
    for row in solvers:
        lines.append(
            f"| {row['language']} | {row['axis_flux_value']:.6e} | "
            f"{row['vertical_symmetry_abs_max']:.6e} | "
            f"{row['axis_midplane_offset_cells']} | {row['axis_radial_center_offset_cells']} | "
            f"{row['axis_boundary_distance_cells']} | {row['axis_local_dominance_margin']:.6e} | "
            f"{row['axis_discrete_laplacian']:.6e} | "
            f"{row['gs_equation_residual_abs_max']:.6e} | "
            f"{row['gs_equation_residual_relative_max']:.6e} | "
            f"{row['midplane_radial_monotonicity_violations']} | "
            f"{row['axis_column_vertical_monotonicity_violations']} | "
            f"{row['negative_flux_abs_max']:.6e} |"
        )
    lines.extend(
        [
            "",
            "These local timings include process start-up for CLI paths. The Go and Rust rows build solver binaries before timing and exclude toolchain orchestration from the measured solver invocation. Use long-lived processes or cloud CPU/GPU runners for throughput comparisons.",
        ]
    )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_md)
    print(json.dumps(report["parity"], sort_keys=True))


if __name__ == "__main__":
    main()
