# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Polyglot Benchmark Report Tests
"""Report-generation tests for the polyglot Grad-Shafranov benchmark."""

from __future__ import annotations

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


def test_case_parser_matrix_parser_and_command_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case parsing and command execution use the tracked file/CSV contracts."""
    case_path = tmp_path / "case.toml"
    case_path.write_text(
        "\n".join(
            [
                "[ignored]",
                "value = 7",
                "",
                "[grad_shafranov]",
                "NR = 5",
                "NZ = 5",
                "enabled = true",
                'label = "reference"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[tuple[list[str], Path]] = []

    def fake_run(
        command: list[str],
        cwd: Path,
        check: bool,
        text: bool,
        capture_output: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, cwd))
        assert check is True
        assert text is True
        assert capture_output is True
        return subprocess.CompletedProcess(command, 0, stdout="0,1\n2,3\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    case = benchmark._read_case(case_path)
    matrix, seconds = benchmark._run_command(["solver", str(case_path)], tmp_path)

    assert case == {"NR": 5, "NZ": 5, "enabled": True, "label": "reference"}
    assert np.array_equal(matrix, np.array([[0.0, 1.0], [2.0, 3.0]]))
    assert seconds >= 0.0
    assert calls == [(["solver", str(case_path)], tmp_path)]


def test_python_runner_executes_reference_solver() -> None:
    """The Python timing path executes the NumPy reference solver."""
    psi, seconds = benchmark._run_python(_sample_case())

    assert psi.shape == (5, 5)
    assert np.all(np.isfinite(psi))
    assert seconds >= 0.0


def test_native_command_wrappers_build_expected_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Julia and Lean wrappers route through their native command lines."""
    calls: list[tuple[list[str], Path]] = []

    def fake_run_command(command: list[str], cwd: Path) -> tuple[benchmark.FloatArray, float]:
        calls.append((command, cwd))
        return _sample_psi(), 0.01

    monkeypatch.setattr(benchmark, "_run_command", fake_run_command)

    julia_psi, julia_seconds = benchmark._run_julia()
    lean_psi, lean_seconds = benchmark._run_lean()

    assert julia_psi.shape == (5, 5)
    assert lean_psi.shape == (5, 5)
    assert julia_seconds == 0.01
    assert lean_seconds == 0.01
    assert calls[0][0][:3] == [
        "julia",
        f"--project={benchmark._JULIA_PROJECT}",
        "--startup-file=no",
    ]
    assert calls[0][1] == benchmark._REPO
    assert calls[1][0] == ["lake", "exe", "gs_picard_csv", str(benchmark._CASE_PATH)]
    assert calls[1][1] == benchmark._LEAN_PROJECT


def test_tool_version_records_subprocess_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool-version probing records command stdout."""

    def fake_run(
        command: list[str],
        check: bool,
        text: bool,
        capture_output: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check is True
        assert text is True
        assert capture_output is True
        return subprocess.CompletedProcess(command, 0, stdout=f"{command[0]} 1.0\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert benchmark._tool_version(["julia", "--version"]) == "julia 1.0"


def test_tool_version_reports_unavailable_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool-version probing reports unavailable binaries without raising."""

    def fake_run(
        command: list[str],
        check: bool,
        text: bool,
        capture_output: bool,
    ) -> subprocess.CompletedProcess[str]:
        raise OSError(f"{command[0]} missing")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert benchmark._tool_version(["julia", "--version"]) == "unavailable: julia missing"


def test_hardware_metadata_records_tool_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hardware metadata records runtime and toolchain version fields."""
    monkeypatch.setattr(benchmark, "_tool_version", lambda command: f"{command[0]} 1.0")
    metadata = benchmark._hardware_metadata()

    assert metadata["julia"] == "julia 1.0"
    assert metadata["go"] == "go 1.0"
    assert metadata["rust"] == "rustc 1.0"
    assert metadata["lean"] == "lean 1.0"
    assert metadata["python"]
    assert metadata["machine"]


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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The benchmark main path writes the tracked JSON/Markdown report schema."""
    psi = _sample_psi()
    case = _sample_case()
    report_json = tmp_path / "polyglot.json"
    report_md = tmp_path / "polyglot.md"

    monkeypatch.setattr(benchmark, "_CASE_PATH", tmp_path / "case.toml")
    monkeypatch.setattr(benchmark, "_REPORT_JSON", report_json)
    monkeypatch.setattr(benchmark, "_REPORT_MD", report_md)
    monkeypatch.setattr(benchmark, "_read_case", lambda _path: case)
    monkeypatch.setattr(benchmark, "_run_python", lambda _case: (psi, 0.10))
    monkeypatch.setattr(benchmark, "_run_julia", lambda: (psi, 0.20))
    monkeypatch.setattr(benchmark, "_run_go", lambda: (psi, 0.30))
    monkeypatch.setattr(benchmark, "_run_rust", lambda: (psi, 0.40))
    monkeypatch.setattr(benchmark, "_run_lean", lambda: (psi, 0.50))
    monkeypatch.setattr(
        benchmark,
        "_hardware_metadata",
        lambda: {
            "cpu_model": "test cpu",
            "machine": "x86_64",
            "python": "3.12",
            "julia": "julia 1.0",
            "go": "go1.0",
            "rust": "rustc 1.0",
            "lean": "Lean 4",
            "os": "Linux",
        },
    )

    benchmark.main()
    out = capsys.readouterr().out
    report = json.loads(report_json.read_text(encoding="utf-8"))
    rendered = report_md.read_text(encoding="utf-8")

    assert str(report_md) in out
    assert report["case"] == case
    assert report["parity"]["shape"] == [5, 5]
    assert [row["language"] for row in report["solvers"]] == [
        "Python",
        "Julia",
        "Go",
        "Rust",
        "Lean",
    ]
    assert "# Polyglot Grad-Shafranov Solver Benchmark" in rendered
    assert "| Python | `gs_solve_np` | 0.100000 |" in rendered
