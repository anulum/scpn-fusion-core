# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Polyglot Grad-Shafranov Contract Tests
"""Polyglot fixed-boundary Grad-Shafranov solver workflow contract.

This workflow test verifies that the native Julia, Go, Rust, and Lean
implementations solve the same fixed-boundary Picard/Jacobi Grad-Shafranov
contract as the Python reference implementation. Each path is executed through
its native toolchain, not through Python FFI.
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
_ALTERNATE_CASE = {
    "R_min": 0.9,
    "R_max": 2.7,
    "Z_min": -1.0,
    "Z_max": 1.0,
    "NR": 13,
    "NZ": 15,
    "Ip_target": 7.5e5,
    "mu0": 4.0e-7 * np.pi,
    "n_picard": 6,
    "n_jacobi": 9,
    "alpha": 0.12,
    "omega_j": 0.7,
    "beta_mix": 0.35,
}


def _write_case(path: Path, case: dict[str, float | int]) -> None:
    path.write_text(
        "\n".join(
            [
                "[grad_shafranov]",
                f"R_min = {case['R_min']}",
                f"R_max = {case['R_max']}",
                f"Z_min = {case['Z_min']}",
                f"Z_max = {case['Z_max']}",
                f"NR = {case['NR']}",
                f"NZ = {case['NZ']}",
                f"Ip_target = {case['Ip_target']}",
                f"mu0 = {case['mu0']}",
                f"n_picard = {case['n_picard']}",
                f"n_jacobi = {case['n_jacobi']}",
                f"alpha = {case['alpha']}",
                f"omega_j = {case['omega_j']}",
                f"beta_mix = {case['beta_mix']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_case_missing_beta_mix(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "[grad_shafranov]",
                "R_min = 1.0",
                "R_max = 3.0",
                "Z_min = -1.2",
                "Z_max = 1.2",
                "NR = 17",
                "NZ = 17",
                "Ip_target = 1.0e6",
                "mu0 = 1.2566370614359173e-6",
                "n_picard = 8",
                "n_jacobi = 16",
                "alpha = 0.1",
                "omega_j = 0.6666666666666666",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _run_julia_case(case_path: Path = _REFERENCE_CASE) -> np.ndarray:
    completed = subprocess.run(
        [
            "julia",
            f"--project={_JULIA_PROJECT}",
            "--startup-file=no",
            str(_JULIA_PROJECT / "bin" / "gs_picard_csv.jl"),
            str(case_path),
        ],
        check=True,
        cwd=_REPO,
        text=True,
        capture_output=True,
    )
    rows = [[float(cell) for cell in row] for row in csv.reader(completed.stdout.splitlines())]
    return np.asarray(rows, dtype=float)


def _run_go_case(case_path: Path = _REFERENCE_CASE) -> np.ndarray:
    completed = subprocess.run(
        ["go", "run", "./cmd/gs_picard_csv", str(case_path)],
        check=True,
        cwd=_REPO / "scpn-fusion-go",
        text=True,
        capture_output=True,
    )
    rows = [[float(cell) for cell in row] for row in csv.reader(completed.stdout.splitlines())]
    return np.asarray(rows, dtype=float)


def _run_rust_case(case_path: Path = _REFERENCE_CASE) -> np.ndarray:
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "fusion-polyglot",
            "--bin",
            "gs_picard_csv",
            "--",
            str(case_path),
        ],
        check=True,
        cwd=_REPO / "scpn-fusion-rs",
        text=True,
        capture_output=True,
    )
    rows = [[float(cell) for cell in row] for row in csv.reader(completed.stdout.splitlines())]
    return np.asarray(rows, dtype=float)


def _run_lean_case(case_path: Path = _REFERENCE_CASE) -> np.ndarray:
    completed = subprocess.run(
        ["lake", "exe", "gs_picard_csv", str(case_path)],
        check=True,
        cwd=_REPO / "scpn-fusion-lean",
        text=True,
        capture_output=True,
    )
    rows = [[float(cell) for cell in row] for row in csv.reader(completed.stdout.splitlines())]
    return np.asarray(rows, dtype=float)


def _assert_matches_python_reference(candidate_psi: np.ndarray) -> None:
    _assert_matches_case(candidate_psi, _CASE)


def _assert_matches_case(candidate_psi: np.ndarray, case: dict[str, float | int]) -> None:
    python_psi = gs_solve_np(**case)

    assert candidate_psi.shape == python_psi.shape
    assert np.all(np.isfinite(candidate_psi))
    assert np.max(np.abs(candidate_psi[0, :])) < 1e-14
    assert np.max(np.abs(candidate_psi[-1, :])) < 1e-14
    assert np.max(np.abs(candidate_psi[:, 0])) < 1e-14
    assert np.max(np.abs(candidate_psi[:, -1])) < 1e-14
    assert np.max(np.abs(candidate_psi - np.flipud(candidate_psi))) < 5e-13
    axis_z_index, axis_r_index = np.unravel_index(np.argmax(candidate_psi), candidate_psi.shape)
    axis_z_index = int(axis_z_index)
    axis_r_index = int(axis_r_index)
    assert axis_z_index == candidate_psi.shape[0] // 2
    assert axis_r_index == candidate_psi.shape[1] // 2
    assert 0 < axis_z_index < candidate_psi.shape[0] - 1
    assert 0 < axis_r_index < candidate_psi.shape[1] - 1
    axis_value = candidate_psi[axis_z_index, axis_r_index]
    nearest_neighbors = [
        candidate_psi[axis_z_index - 1, axis_r_index],
        candidate_psi[axis_z_index + 1, axis_r_index],
        candidate_psi[axis_z_index, axis_r_index - 1],
        candidate_psi[axis_z_index, axis_r_index + 1],
    ]
    assert axis_value - max(nearest_neighbors) > 1e-6
    midplane = candidate_psi[axis_z_index, :]
    assert np.all(np.diff(midplane[: axis_r_index + 1]) >= -1e-14)
    assert np.all(np.diff(midplane[axis_r_index:]) <= 1e-14)
    axis_column = candidate_psi[:, axis_r_index]
    assert np.all(np.diff(axis_column[: axis_z_index + 1]) >= -1e-14)
    assert np.all(np.diff(axis_column[axis_z_index:]) <= 1e-14)
    assert float(np.min(candidate_psi)) >= -1e-14
    assert float(np.max(candidate_psi[1:-1, 1:-1])) > 1e-8

    denominator = np.linalg.norm(python_psi[1:-1, 1:-1]) + 1e-30
    interior_error = candidate_psi[1:-1, 1:-1] - python_psi[1:-1, 1:-1]
    relative_l2 = np.linalg.norm(interior_error) / denominator
    assert relative_l2 < 5e-12
    assert np.max(np.abs(interior_error)) < 5e-12


def test_native_julia_grad_shafranov_matches_python_reference() -> None:
    """Julia native GS solve preserves the Python reference physics contract."""
    _assert_matches_python_reference(_run_julia_case())


def test_native_julia_grad_shafranov_uses_requested_case_file(tmp_path: Path) -> None:
    """Julia native GS solve must consume the requested case file, not a built-in case."""
    case_path = tmp_path / "julia_alt_case.toml"
    _write_case(case_path, _ALTERNATE_CASE)

    _assert_matches_case(_run_julia_case(case_path), _ALTERNATE_CASE)


def test_native_julia_grad_shafranov_rejects_missing_required_case_field(
    tmp_path: Path,
) -> None:
    """Julia native GS case loading must fail closed when a required physics field is absent."""
    case_path = tmp_path / "julia_missing_beta_mix.toml"
    _write_case_missing_beta_mix(case_path)

    completed = subprocess.run(
        [
            "julia",
            f"--project={_JULIA_PROJECT}",
            "--startup-file=no",
            str(_JULIA_PROJECT / "bin" / "gs_picard_csv.jl"),
            str(case_path),
        ],
        cwd=_REPO,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "beta_mix" in completed.stderr


def test_native_go_grad_shafranov_matches_python_reference() -> None:
    """Go native GS solve preserves the Python reference physics contract."""
    _assert_matches_python_reference(_run_go_case())


def test_native_go_grad_shafranov_uses_requested_case_file(tmp_path: Path) -> None:
    """Go native GS solve must consume the requested case file, not a built-in case."""
    case_path = tmp_path / "go_alt_case.toml"
    _write_case(case_path, _ALTERNATE_CASE)

    _assert_matches_case(_run_go_case(case_path), _ALTERNATE_CASE)


def test_native_go_grad_shafranov_rejects_missing_required_case_field(
    tmp_path: Path,
) -> None:
    """Go native GS case loading must fail closed when a required physics field is absent."""
    case_path = tmp_path / "go_missing_beta_mix.toml"
    _write_case_missing_beta_mix(case_path)

    completed = subprocess.run(
        ["go", "run", "./cmd/gs_picard_csv", str(case_path)],
        cwd=_REPO / "scpn-fusion-go",
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "beta_mix" in completed.stderr


def test_native_rust_grad_shafranov_matches_python_reference() -> None:
    """Rust native GS solve preserves the Python reference physics contract."""
    _assert_matches_python_reference(_run_rust_case())


def test_native_rust_grad_shafranov_uses_requested_case_file(tmp_path: Path) -> None:
    """Rust native GS solve must consume the requested case file, not a built-in case."""
    case_path = tmp_path / "rust_alt_case.toml"
    _write_case(case_path, _ALTERNATE_CASE)

    _assert_matches_case(_run_rust_case(case_path), _ALTERNATE_CASE)


def test_native_rust_grad_shafranov_rejects_missing_required_case_field(
    tmp_path: Path,
) -> None:
    """Rust native GS case loading must fail closed when a required physics field is absent."""
    case_path = tmp_path / "rust_missing_beta_mix.toml"
    _write_case_missing_beta_mix(case_path)

    completed = subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "fusion-polyglot",
            "--bin",
            "gs_picard_csv",
            "--",
            str(case_path),
        ],
        cwd=_REPO / "scpn-fusion-rs",
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "beta_mix" in completed.stderr


def test_native_lean_grad_shafranov_matches_python_reference() -> None:
    """Lean native GS solve preserves the Python reference physics contract."""
    _assert_matches_python_reference(_run_lean_case())


def test_native_lean_grad_shafranov_uses_requested_case_file(tmp_path: Path) -> None:
    """Lean native GS solve must consume the requested case file, not a built-in case."""
    case_path = tmp_path / "lean_alt_case.toml"
    _write_case(case_path, _ALTERNATE_CASE)

    _assert_matches_case(_run_lean_case(case_path), _ALTERNATE_CASE)


def test_native_lean_grad_shafranov_rejects_missing_required_case_field(
    tmp_path: Path,
) -> None:
    """Lean native GS case loading must fail closed when a required physics field is absent."""
    case_path = tmp_path / "lean_missing_beta_mix.toml"
    _write_case_missing_beta_mix(case_path)

    completed = subprocess.run(
        ["lake", "exe", "gs_picard_csv", str(case_path)],
        cwd=_REPO / "scpn-fusion-lean",
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "missing required Grad-Shafranov case field: beta_mix" in completed.stderr
