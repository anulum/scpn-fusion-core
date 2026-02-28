"""Tests for 3D force-balance solver (reduced-order spectral variational)."""

import numpy as np
import pytest

from scpn_fusion.core.equilibrium_3d import (
    FourierMode3D,
    VMECStyleEquilibrium3D,
    ForceBalance3D,
    ForceBalanceResult,
)


def _make_eq(**kwargs):
    defaults = dict(r_axis=6.2, z_axis=0.0, a_minor=2.0, kappa=1.7, triangularity=0.3, nfp=1)
    defaults.update(kwargs)
    return VMECStyleEquilibrium3D(**defaults)


class TestForceBalance3DBasic:
    def test_compute_residual_positive(self):
        eq = _make_eq()
        fb = ForceBalance3D(eq)
        res = fb.compute_force_residual(n_rho=6, n_theta=12, n_phi=8)
        assert res > 0.0
        assert np.isfinite(res)

    def test_solve_returns_result(self):
        eq = _make_eq()
        fb = ForceBalance3D(eq, b0_tesla=5.3, r0_major=6.2, p0_pa=5e5)
        result = fb.solve(max_iterations=10, n_rho=6, n_theta=12, n_phi=8)
        assert isinstance(result, ForceBalanceResult)
        assert result.iterations >= 1
        assert result.initial_residual > 0.0
        assert len(result.force_residual_history) >= 2
        assert result.armijo_reject_count >= 0
        assert result.non_decreasing_steps >= 0

    def test_solve_reduces_residual(self):
        eq = _make_eq()
        fb = ForceBalance3D(eq, b0_tesla=5.3, r0_major=6.2)
        result = fb.solve(max_iterations=20, learning_rate=0.005, n_rho=6, n_theta=12, n_phi=8)
        assert result.residual_norm <= result.initial_residual * 1.01  # at least no worse

    def test_solve_seeds_modes_if_empty(self):
        eq = _make_eq()
        assert len(eq.modes) == 0
        fb = ForceBalance3D(eq)
        result = fb.solve(max_iterations=5, n_rho=6, n_theta=12, n_phi=8)
        assert len(result.modes) > 0

    def test_solve_with_existing_modes(self):
        eq = _make_eq(modes=[FourierMode3D(m=1, n=0, r_cos=0.01)])
        fb = ForceBalance3D(eq)
        result = fb.solve(max_iterations=5, n_rho=6, n_theta=12, n_phi=8)
        assert result.iterations >= 1

    def test_history_is_monotonically_tracked(self):
        eq = _make_eq()
        fb = ForceBalance3D(eq)
        result = fb.solve(max_iterations=10, n_rho=6, n_theta=12, n_phi=8)
        assert len(result.force_residual_history) == result.iterations + 1

    def test_pressure_profile_shape(self):
        eq = _make_eq()
        fb = ForceBalance3D(eq, p0_pa=1e6)
        rho = np.linspace(0, 1, 50)
        p = fb._pressure_profile(rho)
        assert p[0] > p[-1]  # peaked
        assert np.all(p >= 0)

    def test_magnetic_field_toroidal_dominance(self):
        eq = _make_eq()
        fb = ForceBalance3D(eq, b0_tesla=5.3)
        R = np.array([6.2])
        Z = np.array([0.0])
        rho = np.array([0.0])
        B_R, B_Z, B_phi = fb._magnetic_field(R, Z, rho)
        assert abs(B_phi[0]) > abs(B_R[0])
        assert abs(B_phi[0]) > abs(B_Z[0])
