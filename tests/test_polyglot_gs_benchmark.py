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
