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
