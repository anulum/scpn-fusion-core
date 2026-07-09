# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Solver Comparison Benchmark Tests
"""Tests for the Python solver-comparison benchmark CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks import solver_comparison as solver


def test_sor_benchmark_runs_real_fusion_kernel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SOR row exercises the production FusionKernel equilibrium path."""
    monkeypatch.chdir(tmp_path)

    row = solver._benchmark_sor(5, 5, max_iter=1)

    assert row["solver"] == "SOR"
    assert row["grid"] == "5x5"
    assert row["iterations"] == 1
    assert isinstance(row["wall_time_ms"], float)
    assert row["wall_time_ms"] >= 0.0
    assert list(tmp_path.iterdir()) == []


def test_newton_benchmark_runs_real_fusion_kernel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Newton row exercises the production FusionKernel Newton path."""
    monkeypatch.chdir(tmp_path)

    row = solver._benchmark_newton(5, 5)

    assert row["solver"] == "Newton-K"
    assert row["grid"] == "5x5"
    assert row["iterations"] == 20
    assert isinstance(row["wall_time_ms"], float)
    assert row["wall_time_ms"] >= 0.0
    assert list(tmp_path.iterdir()) == []


def test_neural_benchmark_reports_successful_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The neural row reports grid shape and timing from the runtime kernel."""

    class _Config:
        grid_shape = (7, 9)

    class _Accel:
        cfg = _Config()

    class _Kernel:
        accel = _Accel()

        def __init__(self, _config_path: Path) -> None:
            pass

        def solve_equilibrium(self) -> None:
            return None

    monkeypatch.setattr(solver, "NeuralEquilibriumKernel", _Kernel)

    row = solver._benchmark_neural(129, 129)

    assert row["solver"] == "Neural (MLP)"
    assert row["grid"] == "9x7"
    assert row["iterations"] == 1
    assert isinstance(row["wall_time_ms"], float)
    assert row["wall_time_ms"] >= 0.0


def test_neural_benchmark_fails_closed_for_missing_config(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The neural row reports an unavailable runtime without aborting the benchmark table."""

    class _MissingKernel:
        def __init__(self, _config_path: Path) -> None:
            raise FileNotFoundError("weights unavailable")

    monkeypatch.setattr(solver, "NeuralEquilibriumKernel", _MissingKernel)

    row = solver._benchmark_neural(129, 129)
    out = capsys.readouterr().out

    assert row == {
        "solver": "Neural (MLP)",
        "grid": "N/A",
        "iterations": "N/A",
        "wall_time_ms": "N/A",
    }
    assert "weights unavailable" in out


def test_main_renders_markdown_table(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI renders all benchmark rows as the documented Markdown table."""

    def _fake_sor(nr: int, nz: int, max_iter: int = 200) -> solver.BenchmarkRow:
        return {
            "solver": "SOR",
            "grid": f"{nr}x{nz}",
            "iterations": max_iter,
            "wall_time_ms": 1.25,
        }

    def _fake_newton(nr: int, nz: int) -> solver.BenchmarkRow:
        return {
            "solver": "Newton-K",
            "grid": f"{nr}x{nz}",
            "iterations": 4,
            "wall_time_ms": 2.5,
        }

    def _fake_neural(nr: int, nz: int) -> solver.BenchmarkRow:
        return {
            "solver": "Neural (MLP)",
            "grid": f"{nr}x{nz}",
            "iterations": 1,
            "wall_time_ms": 0.125,
        }

    monkeypatch.setattr(solver, "_benchmark_sor", _fake_sor)
    monkeypatch.setattr(solver, "_benchmark_newton", _fake_newton)
    monkeypatch.setattr(solver, "_benchmark_neural", _fake_neural)

    solver.main()
    out = capsys.readouterr().out

    assert "### Python Solver Comparison" in out
    assert "| Solver | Grid | Iterations | Wall Time (ms) |" in out
    assert "| SOR | 33x33 | 200 | 1.2 |" in out
    assert "| Newton-K | 65x65 | 4 | 2.5 |" in out
    assert "Neural:  0.125 ms" in out
