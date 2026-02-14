# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
# ----------------------------------------------------------------------
# SCPN Fusion Core -- Tests for Inverse Reconstruction Validation
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ----------------------------------------------------------------------
"""Tests for the inverse reconstruction validation pipeline.

Tests verify that:
1. A circular shot with small perturbation converges.
2. The reconstruction RMSE improves over the initial guess.
3. The output result dict has all required keys.
4. Multiple categories can be reconstructed (moderate elongation).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ── Path setup ─────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_VALIDATION_DIR = _REPO_ROOT / "validation"

if str(_VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(_VALIDATION_DIR))


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def circular_shot():
    """Create a simple circular Solov'ev shot for testing."""
    from generate_synthetic_shots import SolovevEquilibrium

    eq = SolovevEquilibrium(
        R0=1.65,
        a=0.5,
        B0=2.0,
        Ip=1.0,
        kappa=1.0,
        delta=0.0,
        nr=33,
        nz=33,
        n_probes=20,
    )
    eq.solve()
    return eq


@pytest.fixture(scope="module")
def moderate_shot():
    """Create a moderately elongated shot for testing."""
    from generate_synthetic_shots import SolovevEquilibrium

    eq = SolovevEquilibrium(
        R0=1.67,
        a=0.67,
        B0=2.2,
        Ip=1.5,
        kappa=1.5,
        delta=0.25,
        nr=33,
        nz=33,
        n_probes=20,
    )
    eq.solve()
    return eq


def _eq_to_shot_data(eq):
    """Convert a SolovevEquilibrium to a ShotData-like object.

    We import ShotData here to avoid module-level import issues.
    """
    from run_forward_validation import ShotData

    # Reconstruct the scale factor from the equilibrium
    from generate_synthetic_shots import (
        _psi_particular_p, _psi_particular_ff, _HOMOGENEOUS,
    )

    xb = eq.boundary_r / eq.R0
    yb = eq.boundary_z / eq.R0
    psi_on_bdry = (
        _psi_particular_p(xb, yb)
        + eq.A_param * _psi_particular_ff(xb, yb)
    )
    for k, psi_hk in enumerate(_HOMOGENEOUS):
        psi_on_bdry += eq.coefficients[k] * psi_hk(xb, yb)
    psi_raw_center_approx = (
        0.125 + eq.A_param * (-0.125)
        + sum(
            c * h
            for c, h in zip(
                eq.coefficients,
                [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            )
        )
    )
    if abs(psi_raw_center_approx) > 1e-30:
        ir_c = np.argmin(np.abs(eq.r_grid - eq.R0))
        iz_c = np.argmin(np.abs(eq.z_grid - 0.0))
        scale_factor = float(eq.psi_rz[ir_c, iz_c] / psi_raw_center_approx)
    else:
        MU_0 = 4.0 * np.pi * 1e-7
        scale_factor = MU_0 * eq.R0 * (eq.Ip * 1e6) / (2.0 * np.pi)

    return ShotData(
        shot_id="TEST_SHOT",
        category="test",
        description="Test shot",
        r_grid=eq.r_grid,
        z_grid=eq.z_grid,
        psi_rz=eq.psi_rz,
        p_prime=eq.p_prime,
        ff_prime=eq.ff_prime,
        probe_r=eq.probe_r,
        probe_z=eq.probe_z,
        probe_psi=eq.probe_psi,
        boundary_r=eq.boundary_r,
        boundary_z=eq.boundary_z,
        r_axis=eq.r_axis,
        z_axis=eq.z_axis,
        psi_axis=eq.psi_axis,
        R0=eq.R0,
        a=eq.a,
        B0=eq.B0,
        Ip=eq.Ip,
        kappa=eq.kappa,
        delta=eq.delta,
        beta_N=eq.beta_N,
        A_param=eq.A_param,
        scale_factor=scale_factor,
    )


# =====================================================================
# Test 1: Circular shot with small perturbation converges
# =====================================================================


def test_circular_small_perturbation_converges(circular_shot):
    """A circular shot with +/-5% A perturbation should converge rapidly."""
    from run_inverse_validation import (
        InverseConfig,
        run_inverse_reconstruction,
    )

    shot_data = _eq_to_shot_data(circular_shot)
    config = InverseConfig(
        max_iterations=20,
        tolerance=1e-6,
        perturbation_range=0.05,  # small perturbation
        seed=42,
    )
    rng = np.random.default_rng(config.seed)

    result = run_inverse_reconstruction(shot_data, config, rng)

    assert result.converged, (
        f"Expected convergence for circular shot with 5% perturbation, "
        f"but got residual_final={result.residual_final:.2e} "
        f"after {result.iterations} iterations"
    )
    assert result.A_relative_error < 0.01, (
        f"A-parameter relative error {result.A_relative_error:.2e} "
        f"exceeds 1% threshold"
    )


# =====================================================================
# Test 2: Reconstruction RMSE improves over initial guess
# =====================================================================


def test_reconstruction_improves_over_initial(circular_shot):
    """The final RMSE must be strictly less than the initial guess RMSE."""
    from run_inverse_validation import (
        InverseConfig,
        run_inverse_reconstruction,
    )

    shot_data = _eq_to_shot_data(circular_shot)
    config = InverseConfig(
        max_iterations=20,
        tolerance=1e-6,
        perturbation_range=0.20,  # 20% perturbation
        seed=123,
    )
    rng = np.random.default_rng(config.seed)

    result = run_inverse_reconstruction(shot_data, config, rng)

    assert result.psi_rmse_final < result.psi_rmse_initial, (
        f"Expected RMSE to decrease: initial={result.psi_rmse_initial:.6f}, "
        f"final={result.psi_rmse_final:.6f}"
    )
    assert result.improvement_ratio > 1.0, (
        f"Expected improvement_ratio > 1.0, got {result.improvement_ratio:.2f}"
    )


# =====================================================================
# Test 3: Output result dict has all required keys
# =====================================================================


def test_result_dict_has_required_keys(circular_shot):
    """The result dict must contain all fields needed for the summary CSV."""
    from run_inverse_validation import (
        InverseConfig,
        run_inverse_reconstruction,
    )

    shot_data = _eq_to_shot_data(circular_shot)
    config = InverseConfig(
        max_iterations=5,
        tolerance=1e-6,
        perturbation_range=0.10,
        seed=999,
    )
    rng = np.random.default_rng(config.seed)

    result = run_inverse_reconstruction(shot_data, config, rng)
    result_dict = result.to_dict()

    required_keys = {
        "shot_id",
        "category",
        "A_true",
        "A_initial",
        "A_recovered",
        "perturbation_applied",
        "iterations",
        "converged",
        "residual_initial",
        "residual_final",
        "psi_rmse_initial",
        "psi_rmse_final",
        "axis_error_m",
        "boundary_hausdorff_m",
        "wall_time_s",
        "A_relative_error",
        "improvement_ratio",
    }

    missing = required_keys - set(result_dict.keys())
    assert not missing, f"Missing keys in result dict: {missing}"

    # Verify types of key fields
    assert isinstance(result_dict["shot_id"], str)
    assert isinstance(result_dict["iterations"], int)
    assert isinstance(result_dict["converged"], bool)
    assert isinstance(result_dict["psi_rmse_final"], float)
    assert isinstance(result_dict["wall_time_s"], float)
    assert result_dict["wall_time_s"] >= 0.0


# =====================================================================
# Test 4: Moderate elongation shot can be reconstructed
# =====================================================================


def test_moderate_elongation_converges(moderate_shot):
    """A moderately elongated (DIII-D-like) shot should also converge."""
    from run_inverse_validation import (
        InverseConfig,
        run_inverse_reconstruction,
    )

    shot_data = _eq_to_shot_data(moderate_shot)
    config = InverseConfig(
        max_iterations=20,
        tolerance=1e-6,
        perturbation_range=0.15,  # 15% perturbation
        seed=2026,
    )
    rng = np.random.default_rng(config.seed)

    result = run_inverse_reconstruction(shot_data, config, rng)

    # Should converge or at least improve significantly
    assert result.psi_rmse_final < result.psi_rmse_initial, (
        f"Expected RMSE improvement for moderate elongation shot: "
        f"initial={result.psi_rmse_initial:.6f}, "
        f"final={result.psi_rmse_final:.6f}"
    )
    # Relaxed convergence criterion for shaped plasmas
    assert result.A_relative_error < 0.05, (
        f"A-parameter relative error {result.A_relative_error:.2e} "
        f"exceeds 5% threshold for moderate elongation"
    )
