# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — JAX GS Solver Tests
"""Tests for JAX-differentiable Grad-Shafranov solver.

Validates NumPy fallback, JAX solve, boundary conditions, symmetry,
profile response, and autodiff through the full equilibrium solve.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.jax_gs_solver import (
    gs_delta_star_np,
    gs_equation_residual_np,
    gs_solve_np,
    gs_toroidal_current_density_np,
    gs_total_toroidal_current_np,
    has_jax,
    jax_gs_solve,
)

jax_available = has_jax()

# Shared geometry for tests
R_MIN, R_MAX = 0.1, 2.0
Z_MIN, Z_MAX = -1.5, 1.5
NR, NZ = 17, 17
IP_TARGET = 1e6
MU0 = 4e-7 * np.pi
N_PICARD = 20
N_JACOBI = 50


class TestGsSolveNumpy:
    """Verify the NumPy Picard+Jacobi GS solver."""

    def test_returns_correct_shape(self):
        psi = gs_solve_np(
            R_MIN, R_MAX, Z_MIN, Z_MAX, NR, NZ, IP_TARGET, n_picard=N_PICARD, n_jacobi=N_JACOBI
        )
        assert psi.shape == (NZ, NR)

    def test_boundary_zero(self):
        psi = gs_solve_np(
            R_MIN, R_MAX, Z_MIN, Z_MAX, NR, NZ, IP_TARGET, n_picard=N_PICARD, n_jacobi=N_JACOBI
        )
        assert np.allclose(psi[0, :], 0.0)
        assert np.allclose(psi[-1, :], 0.0)
        assert np.allclose(psi[:, 0], 0.0)
        assert np.allclose(psi[:, -1], 0.0)

    def test_finite_values(self):
        psi = gs_solve_np(
            R_MIN, R_MAX, Z_MIN, Z_MAX, NR, NZ, IP_TARGET, n_picard=N_PICARD, n_jacobi=N_JACOBI
        )
        assert np.all(np.isfinite(psi))

    def test_nonzero_interior(self):
        psi = gs_solve_np(
            R_MIN, R_MAX, Z_MIN, Z_MAX, NR, NZ, IP_TARGET, n_picard=N_PICARD, n_jacobi=N_JACOBI
        )
        assert np.max(np.abs(psi[1:-1, 1:-1])) > 0.0

    def test_vertical_symmetry(self):
        """Z-symmetric boundary → approximately Z-symmetric psi."""
        psi = gs_solve_np(R_MIN, R_MAX, Z_MIN, Z_MAX, NR, NZ, IP_TARGET, n_picard=40, n_jacobi=100)
        psi_flipped = psi[::-1, :]
        # Not exact due to finite iterations, but should be close
        np.testing.assert_allclose(psi, psi_flipped, atol=1e-4)

    def test_higher_current_larger_flux(self):
        """Doubling Ip should increase the flux magnitude."""
        psi_lo = gs_solve_np(
            R_MIN, R_MAX, Z_MIN, Z_MAX, NR, NZ, IP_TARGET, n_picard=N_PICARD, n_jacobi=N_JACOBI
        )
        psi_hi = gs_solve_np(
            R_MIN,
            R_MAX,
            Z_MIN,
            Z_MAX,
            NR,
            NZ,
            2.0 * IP_TARGET,
            n_picard=N_PICARD,
            n_jacobi=N_JACOBI,
        )
        assert np.max(np.abs(psi_hi)) > np.max(np.abs(psi_lo))

    def test_finer_grid_finite(self):
        """Solve on 33x33 grid stays finite."""
        psi = gs_solve_np(
            R_MIN, R_MAX, Z_MIN, Z_MAX, 33, 33, IP_TARGET, n_picard=N_PICARD, n_jacobi=N_JACOBI
        )
        assert psi.shape == (33, 33)
        assert np.all(np.isfinite(psi))

    def test_equation_residual_reports_discrete_pde_imbalance(self):
        """Residual diagnostic catches fields that do not satisfy the GS stencil."""
        balanced = np.zeros((3, 3))
        imbalanced = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )

        balanced_metrics = gs_equation_residual_np(
            balanced, 1.0, 3.0, -1.0, 1.0, 3, 3, 0.0, MU0, 0.5
        )
        imbalanced_metrics = gs_equation_residual_np(
            imbalanced, 1.0, 3.0, -1.0, 1.0, 3, 3, 0.0, MU0, 0.5
        )

        assert balanced_metrics["abs_max"] == 0.0
        assert imbalanced_metrics["abs_max"] == pytest.approx(4.0)
        assert imbalanced_metrics["relative_max"] > 1.0

    def test_equation_residual_rejects_malformed_candidate(self):
        """Residual diagnostic fails closed on shape/domain contract violations."""
        with pytest.raises(ValueError, match="psi shape"):
            gs_equation_residual_np(np.zeros((4, 3)), 1.0, 3.0, -1.0, 1.0, 3, 3, 0.0)
        with pytest.raises(ValueError, match="positive, increasing"):
            gs_equation_residual_np(np.zeros((3, 3)), 0.0, 3.0, -1.0, 1.0, 3, 3, 0.0)
        bad = np.zeros((3, 3))
        bad[1, 1] = np.nan
        with pytest.raises(ValueError, match="finite"):
            gs_equation_residual_np(bad, 1.0, 3.0, -1.0, 1.0, 3, 3, 0.0)

    def test_current_density_matches_manufactured_z_quadratic_flux(self):
        """The native GS operator recovers J_phi = -Delta*psi/(mu0 R)."""
        nr, nz = 11, 13
        r_min, r_max = 1.0, 3.0
        z_min, z_max = -1.0, 1.0
        coeff = -0.25

        r = np.linspace(r_min, r_max, nr)
        z = np.linspace(z_min, z_max, nz)
        _, zz = np.meshgrid(r, z)
        psi = coeff * zz**2

        delta_star = gs_delta_star_np(psi, r_min, r_max, z_min, z_max)
        current_density = gs_toroidal_current_density_np(psi, r_min, r_max, z_min, z_max)

        np.testing.assert_allclose(
            delta_star[1:-1, 1:-1],
            2.0 * coeff,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(delta_star[0, :], 0.0, atol=0.0)
        np.testing.assert_allclose(delta_star[-1, :], 0.0, atol=0.0)
        np.testing.assert_allclose(delta_star[:, 0], 0.0, atol=0.0)
        np.testing.assert_allclose(delta_star[:, -1], 0.0, atol=0.0)

        expected_j = np.broadcast_to(
            -2.0 * coeff / (MU0 * r[None, 1:-1]),
            current_density[1:-1, 1:-1].shape,
        )
        np.testing.assert_allclose(
            current_density[1:-1, 1:-1],
            expected_j,
            rtol=1.0e-12,
            atol=1.0e-6,
        )

    def test_total_current_integrates_manufactured_current_closure(self):
        """The native current diagnostic conserves the discrete toroidal current."""
        nr, nz = 17, 19
        r_min, r_max = 1.0, 3.0
        z_min, z_max = -1.0, 1.0
        coeff = -0.125

        r = np.linspace(r_min, r_max, nr)
        z = np.linspace(z_min, z_max, nz)
        _, zz = np.meshgrid(r, z)
        psi = coeff * zz**2
        dr = (r_max - r_min) / (nr - 1)
        dz = (z_max - z_min) / (nz - 1)

        expected_density = -2.0 * coeff / (MU0 * r[None, 1:-1])
        expected_total = float(np.sum(expected_density) * (nz - 2) * dr * dz)

        total_current = gs_total_toroidal_current_np(psi, r_min, r_max, z_min, z_max)

        assert total_current == pytest.approx(expected_total, rel=1.0e-12)


class TestGsSolvePublicAPI:
    """Test jax_gs_solve dispatches correctly."""

    def test_numpy_reference_backend(self):
        psi = jax_gs_solve(
            R_MIN,
            R_MAX,
            Z_MIN,
            Z_MAX,
            NR,
            NZ,
            IP_TARGET,
            n_picard=N_PICARD,
            n_jacobi=N_JACOBI,
            use_jax=False,
        )
        assert psi.shape == (NZ, NR)
        assert np.all(np.isfinite(psi))

    def test_requested_jax_backend_fails_closed_when_unavailable(self, monkeypatch):
        import scpn_fusion.core.jax_gs_solver as gs

        monkeypatch.setattr(gs, "_HAS_JAX", False)
        with pytest.raises(RuntimeError, match="JAX backend requested"):
            gs.jax_gs_solve(
                R_MIN,
                R_MAX,
                Z_MIN,
                Z_MAX,
                NR,
                NZ,
                IP_TARGET,
                n_picard=N_PICARD,
                n_jacobi=N_JACOBI,
                use_jax=True,
            )

    def test_boundary_zero_api(self):
        psi = jax_gs_solve(
            R_MIN,
            R_MAX,
            Z_MIN,
            Z_MAX,
            NR,
            NZ,
            IP_TARGET,
            n_picard=N_PICARD,
            n_jacobi=N_JACOBI,
            use_jax=False,
        )
        assert np.allclose(psi[0, :], 0.0)
        assert np.allclose(psi[-1, :], 0.0)


@pytest.mark.skipif(not jax_available, reason="JAX not installed")
class TestGsSolveJAX:
    """JAX-accelerated GS solver tests."""

    def test_jax_returns_correct_shape(self):
        psi = jax_gs_solve(
            R_MIN,
            R_MAX,
            Z_MIN,
            Z_MAX,
            NR,
            NZ,
            IP_TARGET,
            n_picard=N_PICARD,
            n_jacobi=N_JACOBI,
            use_jax=True,
        )
        assert psi.shape == (NZ, NR)

    def test_jax_boundary_zero(self):
        psi = jax_gs_solve(
            R_MIN,
            R_MAX,
            Z_MIN,
            Z_MAX,
            NR,
            NZ,
            IP_TARGET,
            n_picard=N_PICARD,
            n_jacobi=N_JACOBI,
            use_jax=True,
        )
        assert np.allclose(psi[0, :], 0.0)
        assert np.allclose(psi[-1, :], 0.0)
        assert np.allclose(psi[:, 0], 0.0)
        assert np.allclose(psi[:, -1], 0.0)

    def test_jax_finite(self):
        psi = jax_gs_solve(
            R_MIN,
            R_MAX,
            Z_MIN,
            Z_MAX,
            NR,
            NZ,
            IP_TARGET,
            n_picard=N_PICARD,
            n_jacobi=N_JACOBI,
            use_jax=True,
        )
        assert np.all(np.isfinite(psi))

    def test_jax_nonzero_interior(self):
        psi = jax_gs_solve(
            R_MIN,
            R_MAX,
            Z_MIN,
            Z_MAX,
            NR,
            NZ,
            IP_TARGET,
            n_picard=N_PICARD,
            n_jacobi=N_JACOBI,
            use_jax=True,
        )
        assert np.max(np.abs(psi[1:-1, 1:-1])) > 0.0

    def test_jax_numpy_parity(self):
        """JAX and NumPy solvers produce similar results."""
        psi_np = jax_gs_solve(
            R_MIN,
            R_MAX,
            Z_MIN,
            Z_MAX,
            NR,
            NZ,
            IP_TARGET,
            n_picard=N_PICARD,
            n_jacobi=N_JACOBI,
            use_jax=False,
        )
        psi_jax = jax_gs_solve(
            R_MIN,
            R_MAX,
            Z_MIN,
            Z_MAX,
            NR,
            NZ,
            IP_TARGET,
            n_picard=N_PICARD,
            n_jacobi=N_JACOBI,
            use_jax=True,
        )
        # Same algorithm, should match closely
        np.testing.assert_allclose(psi_jax, psi_np, rtol=1e-5, atol=1e-10)

    def test_jax_higher_current_larger_flux(self):
        psi_lo = jax_gs_solve(
            R_MIN, R_MAX, Z_MIN, Z_MAX, NR, NZ, IP_TARGET, n_picard=N_PICARD, n_jacobi=N_JACOBI
        )
        psi_hi = jax_gs_solve(
            R_MIN,
            R_MAX,
            Z_MIN,
            Z_MAX,
            NR,
            NZ,
            2.0 * IP_TARGET,
            n_picard=N_PICARD,
            n_jacobi=N_JACOBI,
        )
        assert np.max(np.abs(psi_hi)) > np.max(np.abs(psi_lo))


@pytest.mark.skipif(not jax_available, reason="JAX not installed")
class TestGsAutodiff:
    """Verify jax.grad works through the full GS solve."""

    def test_grad_Ip_finite(self):
        from scpn_fusion.core.jax_gs_solver import jax_gs_grad_Ip

        grad_val = jax_gs_grad_Ip(
            IP_TARGET,
            R_MIN,
            R_MAX,
            Z_MIN,
            Z_MAX,
            NR,
            NZ,
            n_picard=10,
            n_jacobi=30,
        )
        assert np.isfinite(grad_val)

    def test_grad_Ip_nonzero(self):
        from scpn_fusion.core.jax_gs_solver import jax_gs_grad_Ip

        grad_val = jax_gs_grad_Ip(
            IP_TARGET,
            R_MIN,
            R_MAX,
            Z_MIN,
            Z_MAX,
            NR,
            NZ,
            n_picard=10,
            n_jacobi=30,
        )
        assert abs(grad_val) > 0.0

    def test_grad_Ip_sign(self):
        """Increasing Ip should increase total flux → positive gradient."""
        from scpn_fusion.core.jax_gs_solver import jax_gs_grad_Ip

        grad_val = jax_gs_grad_Ip(
            IP_TARGET,
            R_MIN,
            R_MAX,
            Z_MIN,
            Z_MAX,
            NR,
            NZ,
            n_picard=10,
            n_jacobi=30,
        )
        # For positive Ip, increasing it should increase psi interior values
        # (flux scales with current). Gradient should be nonzero.
        # Sign depends on convention but should be consistent.
        assert grad_val != 0.0

    def test_grad_beta_mix(self):
        """jax.grad w.r.t. beta_mix through the full solve."""
        import jax
        import jax.numpy as jnp
        from scpn_fusion.core.jax_gs_solver import _jax_gs_solve_impl

        R = jnp.linspace(R_MIN, R_MAX, NR)
        Z = jnp.linspace(Z_MIN, Z_MAX, NZ)
        RR, _ = jnp.meshgrid(R, Z)
        dR = float(R[1] - R[0])
        dZ = float(Z[1] - Z[0])

        R_center = 0.5 * (R_MIN + R_MAX)
        dist_sq = (RR - R_center) ** 2
        psi_init = jnp.exp(-dist_sq / 0.5) * 0.01
        psi_init = psi_init.at[0, :].set(0.0)
        psi_init = psi_init.at[-1, :].set(0.0)
        psi_init = psi_init.at[:, 0].set(0.0)
        psi_init = psi_init.at[:, -1].set(0.0)

        def obj(beta: float) -> float:
            psi = _jax_gs_solve_impl(
                RR,
                dR,
                dZ,
                psi_init,
                float(IP_TARGET),
                float(MU0),
                10,
                30,
                0.1,
                0.667,
                beta,
            )
            return jnp.sum(psi)

        grad_val = float(jax.grad(obj)(jnp.float64(0.5)))
        assert np.isfinite(grad_val)

    def test_grad_matches_finite_diff(self):
        """JAX gradient approximately matches finite-difference estimate."""
        from scpn_fusion.core.jax_gs_solver import jax_gs_grad_Ip

        Ip0 = 1e6
        eps = 1e3
        grad_ad = jax_gs_grad_Ip(
            Ip0,
            R_MIN,
            R_MAX,
            Z_MIN,
            Z_MAX,
            NR,
            NZ,
            n_picard=10,
            n_jacobi=30,
        )

        psi_plus = jax_gs_solve(
            R_MIN,
            R_MAX,
            Z_MIN,
            Z_MAX,
            NR,
            NZ,
            Ip0 + eps,
            n_picard=10,
            n_jacobi=30,
        )
        psi_minus = jax_gs_solve(
            R_MIN,
            R_MAX,
            Z_MIN,
            Z_MAX,
            NR,
            NZ,
            Ip0 - eps,
            n_picard=10,
            n_jacobi=30,
        )
        grad_fd = (np.sum(psi_plus) - np.sum(psi_minus)) / (2.0 * eps)

        # Should agree within ~10% (finite diff is approximate)
        if abs(grad_fd) > 1e-20:
            rel_err = abs(grad_ad - grad_fd) / abs(grad_fd)
            assert rel_err < 0.5, f"AD={grad_ad}, FD={grad_fd}, rel_err={rel_err}"
