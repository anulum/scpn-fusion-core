# ─────────────────────────────────────────────────────────────────────
# Tests for JAX differentiable GS equilibrium solver
# ─────────────────────────────────────────────────────────────────────
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from scpn_fusion.core.jax_equilibrium_solver import (
    _ellipk_approx,
    _ellipe_approx,
    greens_psi,
    vacuum_field,
    solve_equilibrium_jax,
    find_axis,
    axis_position_loss,
    axis_sensitivity,
)


# ── Elliptic integral accuracy ────────────────────────────────────


def test_ellipk_known_values():
    """K(0) = π/2, K(0.5) ≈ 1.8541."""
    assert abs(float(_ellipk_approx(jnp.array(0.0))) - 1.5708) < 0.01
    assert abs(float(_ellipk_approx(jnp.array(0.5))) - 1.8541) < 0.02


def test_ellipe_known_values():
    """E(0) = π/2, E(0.5) ≈ 1.3506."""
    assert abs(float(_ellipe_approx(jnp.array(0.0))) - 1.5708) < 0.01
    assert abs(float(_ellipe_approx(jnp.array(0.5))) - 1.3506) < 0.02


# ── Green's function ─────────────────────────────────────────────


def test_greens_psi_finite():
    R = jnp.array([3.0, 5.0, 7.0])
    Z = jnp.array([0.0, 0.0, 0.0])
    psi = greens_psi(R, Z, 5.0, 0.0, 1.0)
    assert jnp.all(jnp.isfinite(psi))
    assert psi.shape == (3,)


def test_greens_psi_symmetry():
    """Ψ(R, +Z) = Ψ(R, -Z) for coil at Z=0."""
    R = jnp.array([5.0])
    psi_pos = greens_psi(R, jnp.array([1.0]), 5.0, 0.0, 1.0)
    psi_neg = greens_psi(R, jnp.array([-1.0]), 5.0, 0.0, 1.0)
    assert abs(float(psi_pos[0] - psi_neg[0])) < 1e-10


# ── Vacuum field ─────────────────────────────────────────────────


def test_vacuum_field_shape():
    R = jnp.linspace(2.0, 10.0, 32)
    Z = jnp.linspace(-4.0, 4.0, 32)
    coil_R = jnp.array([3.5, 8.0, 9.5])
    coil_Z = jnp.array([3.0, 3.0, 0.0])
    coil_I = jnp.array([-1.0, 4.0, 6.0])
    psi = vacuum_field(R, Z, coil_R, coil_Z, coil_I)
    assert psi.shape == (32, 32)
    assert jnp.all(jnp.isfinite(psi))


# ── Equilibrium solve ────────────────────────────────────────────


def test_solve_equilibrium_converges():
    """Solve on small 33x33 grid, verify axis inside domain."""
    R = jnp.linspace(2.0, 10.0, 33)
    Z = jnp.linspace(-4.0, 4.0, 33)
    coil_R = jnp.array([3.5, 8.0, 9.5, 8.0, 3.5, 9.5, 2.1])
    coil_Z = jnp.array([3.0, 3.0, 0.0, -3.0, -3.0, 3.0, 0.0])
    coil_I = jnp.array([-1.0, 4.0, 6.0, 4.0, -1.0, 3.0, 0.0])
    psi = solve_equilibrium_jax(
        R, Z, coil_R, coil_Z, coil_I, Ip=15.0, max_picard=10, sor_per_picard=10
    )
    assert psi.shape == (33, 33)
    assert jnp.all(jnp.isfinite(psi))
    R_ax, Z_ax = find_axis(psi, R, Z)
    assert 2.0 < float(R_ax) < 10.0
    assert -4.0 < float(Z_ax) < 4.0


# ── Autodiff through equilibrium ─────────────────────────────────


def test_axis_loss_differentiable():
    """jax.grad(axis_position_loss) returns finite gradients."""
    R = jnp.linspace(2.0, 10.0, 17)
    Z = jnp.linspace(-4.0, 4.0, 17)
    coil_R = jnp.array([3.5, 8.0, 9.5])
    coil_Z = jnp.array([3.0, 3.0, 0.0])
    coil_I = jnp.array([-1.0, 4.0, 6.0])
    grad_fn = jax.grad(axis_position_loss, argnums=0)
    g = grad_fn(coil_I, R, Z, coil_R, coil_Z, 15.0, 6.2, 0.0)
    assert g.shape == (3,)
    assert jnp.all(jnp.isfinite(g))


def test_axis_sensitivity_finite():
    """Jacobian dR/dI and dZ/dI are finite."""
    R = jnp.linspace(2.0, 10.0, 17)
    Z = jnp.linspace(-4.0, 4.0, 17)
    coil_R = jnp.array([3.5, 8.0, 9.5])
    coil_Z = jnp.array([3.0, 3.0, 0.0])
    coil_I = jnp.array([-1.0, 4.0, 6.0])
    dR_dI, dZ_dI = axis_sensitivity(coil_I, R, Z, coil_R, coil_Z, 15.0)
    assert dR_dI.shape == (3,)
    assert dZ_dI.shape == (3,)
    assert jnp.all(jnp.isfinite(dR_dI))
    assert jnp.all(jnp.isfinite(dZ_dI))


def test_gradient_descent_reduces_loss():
    """Optimization should decrease axis-position loss."""
    R = jnp.linspace(2.0, 10.0, 17)
    Z = jnp.linspace(-4.0, 4.0, 17)
    coil_R = jnp.array([3.5, 8.0, 9.5])
    coil_Z = jnp.array([3.0, 3.0, 0.0])
    coil_I = jnp.array([-1.0, 4.0, 6.0])
    loss_0 = float(axis_position_loss(coil_I, R, Z, coil_R, coil_Z, 15.0, 6.2, 0.0))
    grad_fn = jax.grad(axis_position_loss, argnums=0)
    g = grad_fn(coil_I, R, Z, coil_R, coil_Z, 15.0, 6.2, 0.0)
    coil_I_new = coil_I - 0.5 * g
    loss_1 = float(axis_position_loss(coil_I_new, R, Z, coil_R, coil_Z, 15.0, 6.2, 0.0))
    assert loss_1 < loss_0
