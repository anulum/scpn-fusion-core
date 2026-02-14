# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Equilibrium Comparison Tests
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Unit tests for validation/equilibrium_comparison.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

# ── Load module by file path (same pattern as test_rmse_dashboard) ───

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "equilibrium_comparison.py"
SPEC = importlib.util.spec_from_file_location("equilibrium_comparison", MODULE_PATH)
assert SPEC and SPEC.loader
eq_cmp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eq_cmp)


# ── Normalised psi RMSE ──────────────────────────────────────────────


def test_identical_psi_rmse_zero() -> None:
    """Identical arrays must yield RMSE = 0."""
    psi = np.random.default_rng(42).normal(size=(64, 64))
    assert eq_cmp.normalized_psi_rmse(psi, psi) == 0.0


def test_known_offset_rmse() -> None:
    """Adding 1% Gaussian noise yields NRMSE close to 0.01."""
    rng = np.random.default_rng(99)
    psi_ref = rng.normal(loc=5.0, scale=1.0, size=(100, 100))
    noise = 0.01 * psi_ref * rng.normal(size=psi_ref.shape)
    psi_ours = psi_ref + noise

    nrmse = eq_cmp.normalized_psi_rmse(psi_ours, psi_ref)
    # The expected value is roughly 0.01; allow generous tolerance for
    # finite-sample fluctuation.
    assert 0.005 < nrmse < 0.02


def test_zero_ref_psi_rmse() -> None:
    """Zero reference with non-zero input returns inf."""
    psi_ref = np.zeros((10, 10))
    psi_ours = np.ones((10, 10))
    assert eq_cmp.normalized_psi_rmse(psi_ours, psi_ref) == float("inf")


def test_both_zero_psi_rmse() -> None:
    """Both zero returns 0."""
    z = np.zeros((5, 5))
    assert eq_cmp.normalized_psi_rmse(z, z) == 0.0


# ── Magnetic axis position ───────────────────────────────────────────


def test_axis_position_exact() -> None:
    """Parabolic psi with minimum at a known grid location."""
    r = np.linspace(1.0, 3.0, 51)
    z = np.linspace(-1.0, 1.0, 41)
    R2d, Z2d = np.meshgrid(r, z, indexing="ij")

    # Minimum at R=2.0, Z=0.0
    r0, z0 = 2.0, 0.0
    psi = (R2d - r0) ** 2 + (Z2d - z0) ** 2

    result = eq_cmp.axis_position_error(psi, r, z, r0, z0)
    assert abs(result["dr_m"]) < 1e-12
    assert abs(result["dz_m"]) < 1e-12
    assert abs(result["total_m"]) < 1e-12


def test_axis_position_offset() -> None:
    """Axis detected at grid point nearest to off-grid minimum."""
    r = np.linspace(1.0, 3.0, 21)
    z = np.linspace(-1.0, 1.0, 21)
    R2d, Z2d = np.meshgrid(r, z, indexing="ij")
    psi = (R2d - 2.05) ** 2 + (Z2d - 0.05) ** 2

    result = eq_cmp.axis_position_error(psi, r, z, 2.05, 0.05)
    # Grid spacing is 0.1 m; error should be at most one cell diagonal.
    assert result["total_m"] < 0.15


# ── Boundary Hausdorff ───────────────────────────────────────────────


def test_hausdorff_identical_zero() -> None:
    """Same contour yields Hausdorff distance = 0."""
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    r_bdry = 6.2 + 2.0 * np.cos(theta)
    z_bdry = 0.0 + 3.4 * np.sin(theta)

    h = eq_cmp.boundary_hausdorff(r_bdry, z_bdry, r_bdry, z_bdry)
    assert h == pytest.approx(0.0, abs=1e-14)


def test_hausdorff_known_offset() -> None:
    """Shifting contour by 0.01 m in R gives Hausdorff ~ 0.01 m."""
    theta = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    r_bdry = 6.2 + 2.0 * np.cos(theta)
    z_bdry = 0.0 + 3.4 * np.sin(theta)

    shift = 0.01  # metres
    h = eq_cmp.boundary_hausdorff(r_bdry + shift, z_bdry, r_bdry, z_bdry)
    assert h == pytest.approx(shift, abs=1e-6)


# ── q-profile RMSE ───────────────────────────────────────────────────


def test_q_profile_identical_zero() -> None:
    """Identical q profiles yield RMSE = 0."""
    q = np.linspace(1.0, 3.5, 50)
    assert eq_cmp.q_profile_rmse(q, q) == pytest.approx(0.0, abs=1e-14)


def test_q_profile_known_offset() -> None:
    """Constant offset of 0.1 yields RMSE = 0.1."""
    q_ref = np.linspace(1.0, 3.0, 40)
    q_ours = q_ref + 0.1
    assert eq_cmp.q_profile_rmse(q_ours, q_ref) == pytest.approx(0.1, abs=1e-10)


def test_q_profile_empty() -> None:
    """Empty profiles return 0."""
    assert eq_cmp.q_profile_rmse(np.array([]), np.array([])) == 0.0


# ── Stored energy error ──────────────────────────────────────────────


def test_stored_energy_identical_zero() -> None:
    """Identical pressure fields yield zero relative error."""
    r = np.linspace(1.0, 3.0, 30)
    z = np.linspace(-1.0, 1.0, 30)
    psi = np.ones((30, 30))
    p = np.ones((30, 30)) * 1e5
    assert eq_cmp.stored_energy_error(psi, psi, p, p, r, z) == pytest.approx(
        0.0, abs=1e-14
    )


def test_stored_energy_known_difference() -> None:
    """Doubling pressure should give relative error = 1.0."""
    r = np.linspace(1.0, 3.0, 30)
    z = np.linspace(-1.0, 1.0, 30)
    psi = np.ones((30, 30))
    p_ref = np.ones((30, 30)) * 1e5
    p_ours = p_ref * 2.0
    err = eq_cmp.stored_energy_error(psi, psi, p_ours, p_ref, r, z)
    assert err == pytest.approx(1.0, rel=1e-10)


# ── Full comparison suite ────────────────────────────────────────────


def _make_equilibrium(seed: int = 0, shift: float = 0.0) -> dict:
    """Build a synthetic equilibrium dict for testing.

    Grid sizes are chosen so that (R=2.0, Z=0.0) falls exactly on a
    grid point, ensuring the argmin-based axis finder is exact.
    """
    rng = np.random.default_rng(seed)
    r = np.linspace(1.0, 3.0, 41)   # step=0.05, 2.0 at index 20
    z = np.linspace(-1.0, 1.0, 41)  # step=0.05, 0.0 at index 20
    R2d, Z2d = np.meshgrid(r, z, indexing="ij")
    psi = (R2d - 2.0) ** 2 + (Z2d - 0.0) ** 2 + shift
    theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    return {
        "psi_rz": psi,
        "r_grid": r,
        "z_grid": z,
        "q_profile": np.linspace(1.0, 3.5, 50),
        "r_axis": 2.0,
        "z_axis": 0.0,
        "r_boundary": 2.0 + 0.9 * np.cos(theta),
        "z_boundary": 0.0 + 0.9 * np.sin(theta),
        "pressure": rng.uniform(1e4, 1e5, size=(41, 41)),
    }


def test_full_comparison_returns_all_keys() -> None:
    """full_comparison must return the five expected top-level keys."""
    ours = _make_equilibrium(seed=0)
    ref = _make_equilibrium(seed=1)

    result = eq_cmp.full_comparison(ours, ref)

    expected_keys = {
        "normalized_psi_rmse",
        "axis_error",
        "boundary_hausdorff_m",
        "q_profile_rmse",
        "stored_energy_relative_error",
    }
    assert set(result.keys()) == expected_keys

    # axis_error must itself be a dict with dr_m, dz_m, total_m
    assert set(result["axis_error"].keys()) == {"dr_m", "dz_m", "total_m"}

    # All scalar values must be finite floats
    assert isinstance(result["normalized_psi_rmse"], float)
    assert isinstance(result["boundary_hausdorff_m"], float)
    assert isinstance(result["q_profile_rmse"], float)
    assert isinstance(result["stored_energy_relative_error"], float)


def test_full_comparison_identical_equilibria() -> None:
    """Comparing an equilibrium to itself yields zero / near-zero errors."""
    eq = _make_equilibrium(seed=42)
    result = eq_cmp.full_comparison(eq, eq)

    assert result["normalized_psi_rmse"] == 0.0
    assert result["axis_error"]["total_m"] == pytest.approx(0.0, abs=1e-12)
    assert result["boundary_hausdorff_m"] == pytest.approx(0.0, abs=1e-14)
    assert result["q_profile_rmse"] == pytest.approx(0.0, abs=1e-14)
    assert result["stored_energy_relative_error"] == pytest.approx(0.0, abs=1e-14)
