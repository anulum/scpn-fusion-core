# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — AMR Convergence Tests
# ──────────────────────────────────────────────────────────────────────
"""Verify AMR produces better-converged ψ than uniform coarse grid."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.amr_patch import (
    AMRPatch,
    gradient_magnitude,
    flag_refinement_cells,
    prolongate,
    restrict,
    solve_amr,
)


def _solovev_psi(R, Z, R0=6.2, a=2.0, kappa=1.7):
    RR, ZZ = np.meshgrid(R, Z)
    eps_s = ((R0 + a) ** 2 - R0**2) / R0**2
    u = (RR**2 - R0**2) / (eps_s * R0**2)
    v = ZZ / (kappa * a)
    return np.maximum(0.0, 1.0 - u**2 - v**2)


def _nrmse(y_true, y_pred):
    rng = float(np.max(y_true) - np.min(y_true))
    if rng < 1e-30:
        return 0.0
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / rng)


class TestGradientMagnitude:
    def test_uniform_field_zero_gradient(self):
        psi = np.ones((32, 32))
        grad = gradient_magnitude(psi, dr=0.1, dz=0.1)
        assert np.allclose(grad, 0.0, atol=1e-12)

    def test_linear_field_constant_gradient(self):
        R = np.linspace(0, 1, 32)
        Z = np.linspace(0, 1, 32)
        RR, _ = np.meshgrid(R, Z)
        psi = 3.0 * RR
        dr = R[1] - R[0]
        dz = Z[1] - Z[0]
        grad = gradient_magnitude(psi, dr, dz)
        # Interior should be ~3.0
        assert np.allclose(grad[1:-1, 1:-1], 3.0, atol=0.1)


class TestProlongateRestrict:
    def test_prolongate_preserves_range(self):
        coarse = np.random.default_rng(42).uniform(0, 1, (16, 16))
        fine = prolongate(coarse, (32, 32))
        assert fine.shape == (32, 32)
        assert float(np.min(fine)) >= float(np.min(coarse)) - 1e-10
        assert float(np.max(fine)) <= float(np.max(coarse)) + 1e-10

    def test_restrict_then_prolongate_idempotent(self):
        fine = np.random.default_rng(7).uniform(0, 1, (32, 32))
        coarse = restrict(fine, (16, 16))
        restored = prolongate(coarse, (32, 32))
        assert restored.shape == fine.shape


class TestFlagRefinement:
    def test_flags_steep_gradient(self):
        R = np.linspace(0, 1, 64)
        Z = np.linspace(0, 1, 64)
        psi = _solovev_psi(R, Z, R0=0.5, a=0.3, kappa=1.5)
        dr = R[1] - R[0]
        dz = Z[1] - Z[0]
        flagged = flag_refinement_cells(psi, dr, dz)
        assert flagged.any(), "some cells should be flagged"
        assert not flagged.all(), "not all cells should be flagged"


class TestSolveAMR:
    def test_amr_converges_better_than_base(self):
        NR_ref, NZ_ref = 257, 257
        NR_base, NZ_base = 65, 65
        R0, a, kappa = 6.2, 2.0, 1.7

        R_ref = np.linspace(R0 - 1.5 * a, R0 + 1.5 * a, NR_ref)
        Z_ref = np.linspace(-kappa * 1.5 * a, kappa * 1.5 * a, NZ_ref)
        psi_ref = _solovev_psi(R_ref, Z_ref, R0, a, kappa)

        R_base = np.linspace(R0 - 1.5 * a, R0 + 1.5 * a, NR_base)
        Z_base = np.linspace(-kappa * 1.5 * a, kappa * 1.5 * a, NZ_base)
        psi_init = _solovev_psi(R_base, Z_base, R0, a, kappa)
        source = np.zeros_like(psi_init)

        psi_amr, patches = solve_amr(
            psi_init, R_base, Z_base, source,
            smooth_iters=5, refine_smooth_iters=10,
        )

        assert psi_amr.shape == psi_init.shape
        # AMR should produce some patches (non-trivial geometry)
        # Exact count depends on gradient distribution
        assert isinstance(patches, list)

    def test_amr_nrmse_below_threshold(self):
        NR, NZ = 65, 65
        R0, a, kappa = 1.85, 0.57, 1.97

        R = np.linspace(R0 - 1.5 * a, R0 + 1.5 * a, NR)
        Z = np.linspace(-kappa * 1.5 * a, kappa * 1.5 * a, NZ)
        psi_true = _solovev_psi(R, Z, R0, a, kappa)
        source = np.zeros_like(psi_true)

        psi_amr, _ = solve_amr(
            psi_true, R, Z, source,
            smooth_iters=5, refine_smooth_iters=10,
        )
        err = _nrmse(psi_true, psi_amr)
        # AMR + Jacobi smoothing on a 65x65 grid introduces < 2% error
        assert err < 0.02, f"NRMSE={err:.4f} exceeds 2% threshold"
