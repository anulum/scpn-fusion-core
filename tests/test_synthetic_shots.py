# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
# ----------------------------------------------------------------------
# SCPN Fusion Core -- Tests for Synthetic Shot Generator
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ----------------------------------------------------------------------
"""Unit tests for validation/generate_synthetic_shots.py.

Tests verify the analytical Solov'ev / Cerfon-Freidberg equilibrium
solver produces physically consistent results without requiring the
Rust GS solver or any external dependencies beyond numpy.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

# ── Load module by file path (same pattern as test_equilibrium_comparison) ──

import sys

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "generate_synthetic_shots.py"
SPEC = importlib.util.spec_from_file_location("generate_synthetic_shots", MODULE_PATH)
assert SPEC and SPEC.loader
synth = importlib.util.module_from_spec(SPEC)
sys.modules["generate_synthetic_shots"] = synth
SPEC.loader.exec_module(synth)

SolovevEquilibrium = synth.SolovevEquilibrium
generate_all_shots = synth.generate_all_shots


# ── Test 1: Circular equilibrium has interior minimum ─────────────────


def test_solovev_circular() -> None:
    """Generate a circular equilibrium (kappa=1, delta=0).

    Verify that psi has its minimum strictly inside the computational
    domain -- i.e. the magnetic axis is not on the grid boundary.
    This is the most basic sanity check: if the axis sits on the edge,
    the equilibrium is unphysical.
    """
    eq = SolovevEquilibrium(
        R0=1.7,
        a=0.6,
        B0=2.0,
        Ip=1.0,
        kappa=1.0,
        delta=0.0,
    ).solve()

    # psi_rz must be a 2-D array of the expected shape
    assert eq.psi_rz.shape == (129, 129)

    # The minimum must be interior (not on any edge row/column)
    assert eq.has_interior_minimum(), (
        "Circular Solov'ev equilibrium must have psi minimum inside the "
        "domain, not on a boundary row/column."
    )

    # The axis should be close to R0 and Z=0 for a circular plasma
    assert abs(eq.r_axis - eq.R0) < eq.a, (
        f"Axis R={eq.r_axis:.3f} is too far from R0={eq.R0:.3f}"
    )
    assert abs(eq.z_axis) < eq.a, (
        f"Axis Z={eq.z_axis:.3f} should be near Z=0 for up-down symmetric "
        f"equilibrium."
    )

    # psi at the axis must be negative (convention: psi < 0 inside)
    assert eq.psi_axis < 0.0, (
        f"psi_axis = {eq.psi_axis:.4e} should be negative."
    )


# ── Test 2: Elongated equilibrium boundary shape ──────────────────────


def test_solovev_elongated() -> None:
    """Generate an elongated equilibrium (kappa=1.7, delta=0.33).

    Verify that the boundary shape kappa estimate roughly matches the
    requested elongation.  We allow 30% tolerance because the Solov'ev
    solution approximates (not exactly matches) the Miller boundary.
    """
    kappa_target = 1.7
    delta_target = 0.33

    eq = SolovevEquilibrium(
        R0=6.2,
        a=2.0,
        B0=5.3,
        Ip=15.0,
        kappa=kappa_target,
        delta=delta_target,
    ).solve()

    kappa_est = eq.boundary_kappa_estimate()
    assert kappa_est > 0.0, "Boundary kappa estimate must be positive."

    # The estimated elongation should be within 30% of the target.
    # The Solov'ev/Cerfon-Freidberg solution matches the prescribed
    # boundary shape in a least-squares sense, so moderate deviations
    # are expected for high shaping.
    assert abs(kappa_est - kappa_target) / kappa_target < 0.30, (
        f"Boundary elongation estimate {kappa_est:.3f} deviates more than "
        f"30% from target {kappa_target:.3f}."
    )

    # Interior minimum check
    assert eq.has_interior_minimum()


# ── Test 3: Probes lie on the LCFS ────────────────────────────────────


def test_probe_on_boundary() -> None:
    """Verify that synthetic probes are placed on the LCFS.

    The probes are generated from the Miller boundary parameterisation.
    Their interpolated psi values should be close to psi_boundary = 0
    (within interpolation and lstsq residual tolerance).
    """
    eq = SolovevEquilibrium(
        R0=2.0,
        a=0.6,
        B0=2.5,
        Ip=1.5,
        kappa=1.5,
        delta=0.25,
    ).solve()

    # Probe arrays must have the expected length
    assert len(eq.probe_r) == 40
    assert len(eq.probe_z) == 40
    assert len(eq.probe_psi) == 40

    # All probes should be on or near the boundary (psi ~ 0).
    # The tolerance must accommodate both the lstsq boundary residual
    # and the bilinear interpolation error.
    psi_range = abs(eq.psi_axis - eq.psi_boundary)
    if psi_range < 1e-30:
        pytest.skip("Degenerate equilibrium: psi_axis == psi_boundary.")

    # Normalise probe psi by the psi range; they should be near 0
    # (boundary) relative to the axis value.
    normalised_probe_psi = np.abs(eq.probe_psi - eq.psi_boundary) / psi_range

    # Allow 15% tolerance (boundary fitting + interpolation error)
    assert np.mean(normalised_probe_psi) < 0.15, (
        f"Mean normalised probe |psi| = {np.mean(normalised_probe_psi):.4f} "
        f"exceeds 15% tolerance.  Probes may not lie on the LCFS."
    )

    # Verify probes are geometrically on the prescribed boundary curve
    # by checking they match the Miller parameterisation.
    theta_check = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    Rb_check, Zb_check = synth.shaped_boundary(
        theta_check, eq.R0, eq.a, eq.kappa, eq.delta
    )
    np.testing.assert_allclose(eq.probe_r, Rb_check, atol=1e-12)
    np.testing.assert_allclose(eq.probe_z, Zb_check, atol=1e-12)


# ── Test 4: Shot metadata completeness ────────────────────────────────


def test_shot_metadata_complete() -> None:
    """Verify that the JSON metadata dict contains all required keys.

    We generate a small batch (save=False, in-memory only) and inspect
    the first shot's metadata dict.
    """
    results = generate_all_shots(seed=42, save=False)

    assert len(results) == 50, f"Expected 50 shots, got {len(results)}"

    required_keys = {
        "shot_id",
        "category",
        "description",
        "R0_m",
        "a_m",
        "B0_T",
        "Ip_MA",
        "kappa",
        "delta",
        "beta_N",
        "grid",
        "n_probes",
        "solver",
        "A_param",
        "psi_axis",
        "r_axis_m",
        "z_axis_m",
        "has_interior_minimum",
        "boundary_kappa_estimate",
        "coefficients",
    }

    required_grid_keys = {
        "nr",
        "nz",
        "R_min_m",
        "R_max_m",
        "Z_min_m",
        "Z_max_m",
    }

    required_array_keys = {
        "psi_rz",
        "r_grid",
        "z_grid",
        "p_prime",
        "ff_prime",
        "probe_r",
        "probe_z",
        "probe_psi",
        "boundary_r",
        "boundary_z",
    }

    for shot in results:
        missing = required_keys - set(shot.keys())
        assert not missing, (
            f"Shot {shot.get('shot_id', '?')} missing metadata keys: {missing}"
        )

        # Grid sub-dict
        assert isinstance(shot["grid"], dict)
        grid_missing = required_grid_keys - set(shot["grid"].keys())
        assert not grid_missing, (
            f"Shot {shot.get('shot_id', '?')} missing grid keys: {grid_missing}"
        )

        # Arrays sub-dict (only present when save=False)
        assert "arrays" in shot, (
            f"Shot {shot.get('shot_id', '?')} missing 'arrays' dict "
            f"(expected when save=False)."
        )
        arr_missing = required_array_keys - set(shot["arrays"].keys())
        assert not arr_missing, (
            f"Shot {shot.get('shot_id', '?')} missing array keys: {arr_missing}"
        )

    # Verify category distribution
    cats = [s["category"] for s in results]
    assert cats.count("circular") == 10
    assert cats.count("moderate_elongation") == 15
    assert cats.count("high_elongation") == 15
    assert cats.count("high_beta") == 5
    assert cats.count("low_current") == 5


# ── Test 5: Array shapes ─────────────────────────────────────────────


def test_array_shapes() -> None:
    """Verify that all output arrays have the documented shapes."""
    eq = SolovevEquilibrium(
        R0=3.0,
        a=1.0,
        B0=3.0,
        Ip=2.0,
        kappa=1.3,
        delta=0.15,
    ).solve()

    assert eq.psi_rz.shape == (129, 129), f"psi_rz shape: {eq.psi_rz.shape}"
    assert eq.r_grid.shape == (129,), f"r_grid shape: {eq.r_grid.shape}"
    assert eq.z_grid.shape == (129,), f"z_grid shape: {eq.z_grid.shape}"
    assert eq.p_prime.shape == (129,), f"p_prime shape: {eq.p_prime.shape}"
    assert eq.ff_prime.shape == (129,), f"ff_prime shape: {eq.ff_prime.shape}"
    assert eq.probe_r.shape == (40,), f"probe_r shape: {eq.probe_r.shape}"
    assert eq.probe_z.shape == (40,), f"probe_z shape: {eq.probe_z.shape}"
    assert eq.probe_psi.shape == (40,), f"probe_psi shape: {eq.probe_psi.shape}"


# ── Test 6: Reproducibility ──────────────────────────────────────────


def test_reproducibility() -> None:
    """Two runs with the same seed must produce identical results."""
    eq1 = SolovevEquilibrium(
        R0=2.0, a=0.5, B0=2.0, Ip=1.0, kappa=1.0, delta=0.0
    ).solve()
    eq2 = SolovevEquilibrium(
        R0=2.0, a=0.5, B0=2.0, Ip=1.0, kappa=1.0, delta=0.0
    ).solve()

    np.testing.assert_array_equal(eq1.psi_rz, eq2.psi_rz)
    np.testing.assert_array_equal(eq1.probe_psi, eq2.probe_psi)
    assert eq1.psi_axis == eq2.psi_axis
    assert eq1.beta_N == eq2.beta_N


# ── Test 7: Solov'ev profiles are constant ────────────────────────────


def test_solovev_constant_profiles() -> None:
    """p'(psi) and FF'(psi) must be constant for a Solov'ev solution."""
    eq = SolovevEquilibrium(
        R0=4.0, a=1.5, B0=4.0, Ip=8.0, kappa=1.6, delta=0.3
    ).solve()

    # All elements should be identical (constant profile)
    assert np.all(eq.p_prime == eq.p_prime[0]), (
        "p_prime should be constant for Solov'ev equilibrium."
    )
    assert np.all(eq.ff_prime == eq.ff_prime[0]), (
        "ff_prime should be constant for Solov'ev equilibrium."
    )

    # They should be finite, non-zero
    assert np.isfinite(eq.p_prime[0])
    assert np.isfinite(eq.ff_prime[0])
