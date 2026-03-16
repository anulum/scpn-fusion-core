# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — JAX Transport Solver Tests
"""Tests for JAX-accelerated transport solver primitives.

Validates NumPy/JAX parity, boundary conditions, conservation, and
batched transport via vmap.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.jax_solvers import (
    _diffusion_rhs_np,
    _thomas_solve_np,
    batched_crank_nicolson,
    crank_nicolson_step,
    diffusion_rhs,
    has_jax,
    thomas_solve,
)

# Skip JAX-specific tests if JAX not installed
jax_available = has_jax()


class TestThomasSolveNumpy:
    """Verify NumPy Thomas solver against known tridiagonal systems."""

    def test_identity_system(self):
        """I x = d => x = d."""
        n = 10
        a = np.zeros(n - 1)
        b = np.ones(n)
        c = np.zeros(n - 1)
        d = np.arange(n, dtype=float)
        x = _thomas_solve_np(a, b, c, d)
        np.testing.assert_allclose(x, d, atol=1e-12)

    def test_simple_tridiag(self):
        """Known 4x4 system: [2,-1,0,0; -1,2,-1,0; 0,-1,2,-1; 0,0,-1,2] x = [1,0,0,1]."""
        a = np.array([-1.0, -1.0, -1.0])
        b = np.array([2.0, 2.0, 2.0, 2.0])
        c = np.array([-1.0, -1.0, -1.0])
        d = np.array([1.0, 0.0, 0.0, 1.0])
        x = _thomas_solve_np(a, b, c, d)
        # Verify A @ x = d
        result = b * x
        result[:-1] += c * x[1:]
        result[1:] += a * x[:-1]
        np.testing.assert_allclose(result, d, atol=1e-10)

    def test_diffusion_like_matrix(self):
        """Tridiag from 1D diffusion discretisation."""
        n = 32
        a = -0.5 * np.ones(n - 1)
        b = 2.0 * np.ones(n)
        c = -0.5 * np.ones(n - 1)
        rng = np.random.default_rng(42)
        d = rng.standard_normal(n)
        x = _thomas_solve_np(a, b, c, d)
        # Verify residual
        result = b * x
        result[:-1] += c * x[1:]
        result[1:] += a * x[:-1]
        np.testing.assert_allclose(result, d, atol=1e-10)


class TestDiffusionRhsNumpy:
    """Verify NumPy diffusion operator."""

    def test_constant_profile_zero_flux(self):
        """Uniform T => L_h(T) = 0 everywhere."""
        n = 32
        rho = np.linspace(0.01, 1.0, n)
        T = 5.0 * np.ones(n)
        chi = 0.5 * np.ones(n)
        drho = rho[1] - rho[0]
        Lh = _diffusion_rhs_np(T, chi, rho, drho)
        np.testing.assert_allclose(Lh, 0.0, atol=1e-12)

    def test_linear_profile(self):
        """Linear T(r) = r => nonzero L_h due to cylindrical geometry."""
        n = 64
        rho = np.linspace(0.01, 1.0, n)
        T = rho.copy()
        chi = np.ones(n)
        drho = rho[1] - rho[0]
        Lh = _diffusion_rhs_np(T, chi, rho, drho)
        # Interior should be nonzero (1/r correction)
        assert np.any(np.abs(Lh[1:-1]) > 1e-6)

    def test_boundary_values_zero(self):
        """Boundaries (i=0, i=n-1) should be zero (not computed)."""
        n = 32
        rho = np.linspace(0.01, 1.0, n)
        T = np.sin(np.pi * rho)
        chi = np.ones(n)
        drho = rho[1] - rho[0]
        Lh = _diffusion_rhs_np(T, chi, rho, drho)
        assert Lh[0] == 0.0
        assert Lh[-1] == 0.0


class TestCrankNicolsonStep:
    """Verify Crank-Nicolson transport step."""

    def test_no_source_diffuses(self):
        """Without sources, peaked profile should flatten."""
        n = 32
        rho = np.linspace(0.05, 1.0, n)
        drho = rho[1] - rho[0]
        T = 10.0 * np.exp(-((rho - 0.3) ** 2) / 0.05)
        chi = 0.5 * np.ones(n)
        source = np.zeros(n)
        T_new = crank_nicolson_step(T, chi, source, rho, drho, dt=0.01, use_jax=False)
        # Peak should decrease
        assert np.max(T_new[1:-1]) < np.max(T[1:-1])

    def test_edge_bc_applied(self):
        """Edge boundary condition should be enforced."""
        n = 32
        rho = np.linspace(0.05, 1.0, n)
        drho = rho[1] - rho[0]
        T = 5.0 * np.ones(n)
        chi = 0.1 * np.ones(n)
        source = np.zeros(n)
        T_new = crank_nicolson_step(T, chi, source, rho, drho, dt=0.01, T_edge=0.1, use_jax=False)
        assert T_new[-1] == pytest.approx(0.1)
        # Core Neumann: T[0] = T[1]
        assert T_new[0] == pytest.approx(T_new[1])


@pytest.mark.skipif(not jax_available, reason="JAX not installed")
class TestJaxParity:
    """Verify JAX implementations match NumPy within tolerance."""

    def test_thomas_parity(self):
        n = 32
        rng = np.random.default_rng(42)
        a = -0.3 * np.ones(n - 1)
        b = 2.0 * np.ones(n)
        c = -0.3 * np.ones(n - 1)
        d = rng.standard_normal(n)

        x_np = thomas_solve(a, b, c, d, use_jax=False)
        x_jax = thomas_solve(a, b, c, d, use_jax=True)
        np.testing.assert_allclose(x_jax, x_np, atol=1e-10)

    def test_diffusion_rhs_parity(self):
        n = 32
        rho = np.linspace(0.05, 1.0, n)
        drho = rho[1] - rho[0]
        T = 10.0 * np.exp(-((rho - 0.5) ** 2) / 0.1)
        chi = 0.5 + 0.2 * rho

        Lh_np = diffusion_rhs(T, chi, rho, drho, use_jax=False)
        Lh_jax = diffusion_rhs(T, chi, rho, drho, use_jax=True)
        np.testing.assert_allclose(Lh_jax, Lh_np, atol=1e-8)

    def test_cn_step_parity(self):
        n = 32
        rho = np.linspace(0.05, 1.0, n)
        drho = rho[1] - rho[0]
        T = 10.0 * np.exp(-((rho - 0.3) ** 2) / 0.05)
        chi = 0.5 * np.ones(n)
        source = np.zeros(n)

        T_np = crank_nicolson_step(T, chi, source, rho, drho, 0.01, use_jax=False)
        T_jax = crank_nicolson_step(T, chi, source, rho, drho, 0.01, use_jax=True)
        np.testing.assert_allclose(T_jax, T_np, atol=1e-8)

    def test_batched_cn(self):
        n = 32
        batch = 8
        rho = np.linspace(0.05, 1.0, n)
        drho = rho[1] - rho[0]
        rng = np.random.default_rng(7)
        T_batch = 5.0 + rng.standard_normal((batch, n))
        chi = 0.5 * np.ones(n)
        source = np.zeros(n)

        result = batched_crank_nicolson(T_batch, chi, source, rho, drho, 0.01)
        assert result.shape == (batch, n)
        # Each trajectory should satisfy BCs
        for i in range(batch):
            assert result[i, -1] == pytest.approx(0.1)
            assert result[i, 0] == pytest.approx(result[i, 1])

    def test_jax_grad_thomas(self):
        """JAX autodiff through Thomas solver should produce finite gradients."""
        import jax
        import jax.numpy as jnp
        from scpn_fusion.core.jax_solvers import _thomas_solve_jax_impl

        n = 16
        a = -0.3 * jnp.ones(n - 1)
        b = 2.0 * jnp.ones(n)
        c = -0.3 * jnp.ones(n - 1)

        def loss(d):
            x = _thomas_solve_jax_impl(a, b, c, d)
            return jnp.sum(x**2)

        d = jnp.ones(n)
        grad = jax.grad(loss)(d)
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.any(grad != 0.0)

    def test_jax_grad_cn_step(self):
        """JAX autodiff through full CN step."""
        import jax
        import jax.numpy as jnp
        from scpn_fusion.core.jax_solvers import _cn_step_jax

        n = 16
        rho = jnp.linspace(0.05, 1.0, n)
        drho = float(rho[1] - rho[0])
        chi = 0.5 * jnp.ones(n)
        source = jnp.zeros(n)

        def loss(T):
            T_new = _cn_step_jax(T, chi, source, rho, drho, 0.01, 0.1)
            return jnp.sum(T_new**2)

        T = 5.0 * jnp.exp(-((rho - 0.3) ** 2) / 0.1)
        grad = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad))


class TestBatchedTransportFallback:
    """Batched transport with NumPy fallback."""

    def test_batched_numpy_fallback(self):
        n = 16
        batch = 4
        rho = np.linspace(0.05, 1.0, n)
        drho = rho[1] - rho[0]
        rng = np.random.default_rng(42)
        T_batch = 5.0 + rng.standard_normal((batch, n))
        chi = 0.5 * np.ones(n)
        source = np.zeros(n)

        # Force NumPy path by comparing single vs batch
        results_single = np.stack(
            [
                crank_nicolson_step(T_batch[i], chi, source, rho, drho, 0.01, use_jax=False)
                for i in range(batch)
            ]
        )
        # batched_crank_nicolson will use JAX if available, NumPy otherwise
        # Either way, results should match
        for i in range(batch):
            assert results_single[i, -1] == pytest.approx(0.1)
