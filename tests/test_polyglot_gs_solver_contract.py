# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Polyglot Grad-Shafranov Contract Tests
"""Polyglot fixed-boundary Grad-Shafranov solver workflow contract.

This workflow test verifies that the native Julia implementation solves the
same fixed-boundary Picard/Jacobi Grad-Shafranov contract as the Python
reference implementation. The Julia path is executed as a native Julia package,
not through Python FFI.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import numpy as np

from scpn_fusion.core.jax_gs_solver import gs_solve_np


_REPO = Path(__file__).resolve().parents[1]
_JULIA_PROJECT = _REPO / "scpn-fusion-jl"
_REFERENCE_CASE = _REPO / "validation" / "polyglot" / "gs_picard_reference.toml"
_CASE = {
    "R_min": 1.0,
    "R_max": 3.0,
    "Z_min": -1.2,
    "Z_max": 1.2,
    "NR": 17,
    "NZ": 17,
    "Ip_target": 1.0e6,
    "mu0": 4.0e-7 * np.pi,
    "n_picard": 8,
    "n_jacobi": 16,
    "alpha": 0.1,
    "omega_j": 2.0 / 3.0,
    "beta_mix": 0.5,
}


def _run_julia_case() -> np.ndarray:
    completed = subprocess.run(
        [
            "julia",
            f"--project={_JULIA_PROJECT}",
            "--startup-file=no",
            str(_JULIA_PROJECT / "bin" / "gs_picard_csv.jl"),
            str(_REFERENCE_CASE),
        ],
        check=True,
        cwd=_REPO,
        text=True,
        capture_output=True,
    )
    rows = [[float(cell) for cell in row] for row in csv.reader(completed.stdout.splitlines())]
    return np.asarray(rows, dtype=float)


def test_native_julia_grad_shafranov_matches_python_reference() -> None:
    """Julia native GS solve preserves the Python reference physics contract."""
    python_psi = gs_solve_np(**_CASE)
    julia_psi = _run_julia_case()

    assert julia_psi.shape == python_psi.shape
    assert np.all(np.isfinite(julia_psi))
    assert np.max(np.abs(julia_psi[0, :])) < 1e-14
    assert np.max(np.abs(julia_psi[-1, :])) < 1e-14
    assert np.max(np.abs(julia_psi[:, 0])) < 1e-14
    assert np.max(np.abs(julia_psi[:, -1])) < 1e-14

    denominator = np.linalg.norm(python_psi[1:-1, 1:-1]) + 1e-30
    relative_l2 = np.linalg.norm(julia_psi[1:-1, 1:-1] - python_psi[1:-1, 1:-1]) / denominator
    assert relative_l2 < 5e-12
