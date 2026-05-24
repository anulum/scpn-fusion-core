# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Polyglot Grad-Shafranov Benchmark Tests
"""Polyglot Grad-Shafranov benchmark contract tests."""

from __future__ import annotations

import subprocess

import numpy as np

import benchmarks.polyglot_gs_solver_comparison as benchmark


def test_vertical_symmetry_metric_detects_up_down_asymmetry() -> None:
    """Benchmark reports must expose up-down symmetry residuals for symmetric GS cases."""
    symmetric = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    asymmetric = np.array(
        [
            [0.0, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    assert benchmark._vertical_symmetry_abs_max(symmetric) == 0.0
    assert benchmark._vertical_symmetry_abs_max(asymmetric) == 0.2


def test_axis_midplane_offset_counts_cells_from_symmetric_midplane() -> None:
    """Benchmark reports must expose magnetic-axis displacement from the midplane."""
    centered = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    shifted = np.array(
        [
            [0.0, 3.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    assert benchmark._axis_midplane_offset_cells(centered) == 0
    assert benchmark._axis_midplane_offset_cells(shifted) == 1


def test_axis_radial_center_offset_counts_cells_from_geometric_center() -> None:
    """Benchmark reports must expose magnetic-axis radial displacement."""
    centered = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    shifted = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
        ]
    )

    assert benchmark._axis_radial_center_offset_cells(centered) == 0
    assert benchmark._axis_radial_center_offset_cells(shifted) == 1


def test_negative_flux_overshoot_metric_reports_maximum_negative_well() -> None:
    """Benchmark reports must expose nonphysical negative-flux overshoot."""
    nonnegative = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    overshoot = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, -0.25, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    assert benchmark._negative_flux_abs_max(nonnegative) == 0.0
    assert benchmark._negative_flux_abs_max(overshoot) == 0.25


def test_interior_max_abs_error_excludes_dirichlet_boundary() -> None:
    """Benchmark parity must expose localized interior solver error."""
    reference = np.array(
        [
            [9.0, 9.0, 9.0],
            [9.0, 1.0, 9.0],
            [9.0, 9.0, 9.0],
        ]
    )
    candidate = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.25, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    assert benchmark._interior_max_abs_error(candidate, reference) == 0.25


def test_axis_boundary_distance_counts_cells_to_nearest_wall() -> None:
    """Benchmark reports must expose whether the magnetic axis is confined."""
    confined = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    wall_axis = np.array(
        [
            [0.0, 3.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    assert benchmark._axis_boundary_distance_cells(confined) == 1
    assert benchmark._axis_boundary_distance_cells(wall_axis) == 0


def test_rust_benchmark_executes_compiled_release_solver(monkeypatch) -> None:
    """Rust timing must measure the compiled solver binary, not Cargo orchestration."""
    calls: list[tuple[list[str], object]] = []

    def fake_run(command, cwd, check, text, capture_output):
        calls.append((list(command), cwd))
        if command[0] == "cargo":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        stdout = "0,0,0\n0,1,0\n0,0,0\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    psi, wall_seconds = benchmark._run_rust()

    assert psi.shape == (3, 3)
    assert np.isfinite(wall_seconds)
    assert calls[0][0] == ["cargo", "build", "--release", "-q", "-p", "fusion-polyglot"]
    assert calls[0][1] == benchmark._RUST_PROJECT
    assert calls[1][0] == [
        str(benchmark._RUST_RELEASE_BINARY),
        str(benchmark._CASE_PATH),
    ]
    assert calls[1][1] == benchmark._RUST_PROJECT


def test_go_benchmark_executes_compiled_solver_binary(monkeypatch) -> None:
    """Go timing must measure the compiled solver binary, not go run orchestration."""
    calls: list[tuple[list[str], object]] = []

    class FakeTemporaryDirectory:
        def __enter__(self) -> str:
            return "/tmp/scpn-fusion-go-benchmark"

        def __exit__(self, exc_type, exc, traceback) -> bool:
            return False

    def fake_run(command, cwd, check, text, capture_output):
        calls.append((list(command), cwd))
        if command[0] == "go":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        stdout = "0,0,0\n0,1,0\n0,0,0\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(benchmark.tempfile, "TemporaryDirectory", FakeTemporaryDirectory)
    monkeypatch.setattr(subprocess, "run", fake_run)

    psi, wall_seconds = benchmark._run_go()

    assert psi.shape == (3, 3)
    assert np.isfinite(wall_seconds)
    assert calls[0][0] == [
        "go",
        "build",
        "-o",
        "/tmp/scpn-fusion-go-benchmark/gs_picard_csv",
        "./cmd/gs_picard_csv",
    ]
    assert calls[0][1] == benchmark._GO_PROJECT
    assert calls[1][0] == [
        "/tmp/scpn-fusion-go-benchmark/gs_picard_csv",
        str(benchmark._CASE_PATH),
    ]
    assert calls[1][1] == benchmark._GO_PROJECT
