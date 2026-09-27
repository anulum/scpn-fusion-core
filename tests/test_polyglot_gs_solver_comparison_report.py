# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Polyglot Benchmark Report Tests
"""Real five-language benchmark workflow and report provenance regressions."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks import polyglot_gs_solver_comparison as benchmark


def _sample_psi() -> benchmark.FloatArray:
    """Return a symmetric fixed-boundary flux grid with an interior axis."""
    return np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 1.0, 0.5, 0.0],
            [0.0, 1.0, 2.0, 1.0, 0.0],
            [0.0, 0.5, 1.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


def _sample_case() -> dict[str, Any]:
    """Return the minimal Grad-Shafranov case consumed by report metrics."""
    return {
        "R_min": 1.0,
        "R_max": 3.0,
        "Z_min": -1.0,
        "Z_max": 1.0,
        "NR": 5,
        "NZ": 5,
        "Ip_target": 1.0e6,
        "mu0": 4.0e-7 * np.pi,
        "n_picard": 2,
        "n_jacobi": 3,
        "alpha": 0.1,
        "omega_j": 2.0 / 3.0,
        "beta_mix": 0.5,
    }


def test_case_parser_matrix_parser_and_command_runner(tmp_path: Path) -> None:
    """Load a real typed case and compare actual native CSV with the NumPy solve."""
    path = tmp_path / "case.toml"
    expected = _sample_case()
    path.write_text(
        '["grad_shafranov"]\n'
        + "\n".join(f"{key} = {value}" for key, value in expected.items())
        + "\n"
    )
    case = benchmark._read_case(path)
    assert case == expected
    matrix, seconds = benchmark._run_go(path)
    reference, _ = benchmark._run_python(case)
    np.testing.assert_allclose(matrix, reference, rtol=5e-12, atol=5e-12)
    assert seconds > 0.0
    path.write_text(path.read_text() + "enabled = true\n")
    with pytest.raises(ValueError):
        benchmark._read_case(path)
    with pytest.raises(subprocess.CalledProcessError):
        benchmark._run_go(path)


def test_python_runner_executes_reference_solver() -> None:
    """The actual Python timing path returns a finite fixed-boundary physical solve."""
    psi, seconds = benchmark._run_python(_sample_case())
    assert psi.shape == (5, 5) and np.all(np.isfinite(psi)) and seconds > 0.0
    assert np.all(psi[[0, -1]] == 0.0) and np.all(psi[:, [0, -1]] == 0.0)


def test_native_command_wrappers_build_expected_commands() -> None:
    """Exercise real Julia and Lean wrappers through their pinned CLI toolchains."""
    reference, _ = benchmark._run_python(benchmark._read_case(benchmark._CASE_PATH))
    for run in (benchmark._run_julia, benchmark._run_lean):
        psi, seconds = run()
        assert seconds > 0.0 and psi.shape == reference.shape
        np.testing.assert_allclose(psi, reference, rtol=5e-12, atol=5e-12)


def test_tool_version_records_subprocess_output() -> None:
    """Record version output from the real available Julia executable."""
    version = benchmark._tool_version(["julia", "--version"])
    assert "julia" in version.lower() and not version.startswith("unavailable:")


def test_tool_version_reports_unavailable_command(tmp_path: Path) -> None:
    """Report an actual missing executable as an availability diagnostic."""
    assert benchmark._tool_version([str(tmp_path / "missing-executable")]).startswith(
        "unavailable:"
    )


def test_hardware_metadata_records_tool_versions() -> None:
    """Qualify actual host metadata and the pinned Lean toolchain version."""
    metadata = benchmark._hardware_metadata()
    for key in ("julia", "go", "rust", "lean", "python", "machine"):
        assert metadata[key] and not metadata[key].startswith("unavailable:")
    assert "4.29.1" in metadata["lean"]


def test_boundary_and_error_metrics_cover_report_scalars() -> None:
    """Report scalar helpers expose boundary and interior parity metrics."""
    reference = _sample_psi()
    candidate = reference.copy()
    candidate[0, 2] = 0.25
    candidate[2, 2] = 2.25
    oscillatory = reference.copy()
    oscillatory[2, 0] = 1.5
    oscillatory[0, 2] = 1.5
    oscillatory[1, 2] = 0.25

    assert benchmark._boundary_abs_max(candidate) == 0.25
    assert benchmark._relative_l2(candidate, reference) > 0.0
    assert benchmark._interior_max_abs_error(candidate, reference) == 0.25
    assert benchmark._midplane_radial_monotonicity_violations(oscillatory) == 1
    assert benchmark._axis_column_vertical_monotonicity_violations(oscillatory) == 1


def test_main_writes_json_and_markdown_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run all five actual solver paths and verify report schema, physics and byte provenance."""
    report_json, report_md = tmp_path / "polyglot.json", tmp_path / "polyglot.md"
    benchmark.main(report_json=report_json, report_md=report_md)
    output = capsys.readouterr().out
    report = json.loads(report_json.read_text())
    rendered = report_md.read_text()
    assert str(report_md) in output
    assert report["case"] == benchmark._read_case(benchmark._CASE_PATH)
    assert report["case_sha256"] == hashlib.sha256(benchmark._CASE_PATH.read_bytes()).hexdigest()
    assert report["parity"]["shape"] == [17, 17]
    assert [row["language"] for row in report["solvers"]] == [
        "Python",
        "Julia",
        "Go",
        "Rust",
        "Lean",
    ]
    for row in report["solvers"]:
        assert row["wall_time_s"] > 0.0
    for row in report["parity"]["by_language"].values():
        assert row["relative_l2_interior"] < 5e-12
        assert row["boundary_abs_max"] == 0.0
    assert "# Polyglot Grad-Shafranov Solver Benchmark" in rendered
    assert "| Python | `gs_solve_np` |" in rendered
