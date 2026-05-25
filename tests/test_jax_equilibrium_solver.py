# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
from __future__ import annotations

import pytest
import numpy as np
from scipy.special import ellipe, ellipk

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from scpn_fusion.core.jax_equilibrium_solver import (
    _ellipk_approx,
    _ellipe_approx,
    _boundary_flux_level,
    _interior_axis_flux,
    _plasma_source,
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


def test_elliptic_integrals_match_scipy_reference_across_green_function_domain():
    """Complete elliptic integrals should match reference values used by coil Green kernels."""
    m = jnp.array([0.0, 1.0e-9, 1.0e-4, 0.01, 0.5, 0.9, 0.99, 0.999999])
    m_np = np.asarray(m)

    np.testing.assert_allclose(
        np.asarray(_ellipk_approx(m)), ellipk(m_np), rtol=1.0e-8, atol=1.0e-8
    )
    np.testing.assert_allclose(
        np.asarray(_ellipe_approx(m)), ellipe(m_np), rtol=1.0e-8, atol=1.0e-8
    )


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


def test_find_axis_ignores_free_boundary_wall_flux_maximum():
    """PF-coil wall flux must not be reported as the plasma magnetic axis."""
    R = jnp.linspace(1.0, 5.0, 5)
    Z = jnp.linspace(-2.0, 2.0, 5)
    psi = jnp.zeros((5, 5))
    psi = psi.at[0, :].set(100.0)
    psi = psi.at[-1, :].set(100.0)
    psi = psi.at[:, 0].set(100.0)
    psi = psi.at[:, -1].set(100.0)
    psi = psi.at[2, 2].set(1.0)

    R_ax, Z_ax = find_axis(psi, R, Z)

    assert float(R_ax) == pytest.approx(3.0)
    assert float(Z_ax) == pytest.approx(0.0)


def test_find_axis_supports_negative_flux_well_convention():
    """GEQDSK-like decreasing-psi conventions still have an interior axis."""
    R = jnp.linspace(1.0, 5.0, 5)
    Z = jnp.linspace(-2.0, 2.0, 5)
    psi = jnp.zeros((5, 5))
    psi = psi.at[2, 2].set(-4.0)
    psi = psi.at[2, 1].set(-1.0)
    psi = psi.at[2, 3].set(-1.0)
    psi = psi.at[1, 2].set(-1.0)
    psi = psi.at[3, 2].set(-1.0)

    R_ax, Z_ax = find_axis(psi, R, Z)

    assert float(R_ax) == pytest.approx(3.0)
    assert float(Z_ax) == pytest.approx(0.0)


def test_free_boundary_source_uses_nonzero_wall_flux_level():
    """Source normalization must use the actual free-boundary wall-flux level."""
    psi = jnp.full((5, 5), 10.0)
    psi = psi.at[2, 2].set(6.0)
    psi = psi.at[1, 2].set(8.0)
    psi = psi.at[3, 2].set(8.0)
    psi = psi.at[2, 1].set(8.0)
    psi = psi.at[2, 3].set(8.0)

    assert float(_boundary_flux_level(psi)) == pytest.approx(10.0)
    assert float(_interior_axis_flux(psi)) == pytest.approx(6.0)


def test_plasma_source_remains_bounded_for_near_degenerate_flux_span():
    """Near-equal wall and axis flux must not explode the source normalization."""
    R = jnp.linspace(2.0, 4.0, 5)
    psi = jnp.ones((5, 5)) * 10.0
    source = _plasma_source(psi, R, Ip=15.0, psi_axis=10.0, psi_boundary=10.0 + 1e-14)

    assert jnp.all(jnp.isfinite(source))
    assert float(jnp.max(jnp.abs(source))) < 1.0e3


def test_free_boundary_solve_preserves_vacuum_flux_boundary():
    """Free-boundary Picard/SOR must keep external coil flux on the wall."""
    R = jnp.linspace(2.0, 10.0, 17)
    Z = jnp.linspace(-4.0, 4.0, 17)
    coil_R = jnp.array([3.5, 8.0, 9.5])
    coil_Z = jnp.array([3.0, -3.0, 0.0])
    coil_I = jnp.array([-1.0, 4.0, 6.0])

    psi_vac = vacuum_field(R, Z, coil_R, coil_Z, coil_I)
    psi = solve_equilibrium_jax(
        R, Z, coil_R, coil_Z, coil_I, Ip=15.0, max_picard=4, sor_per_picard=4
    )

    np.testing.assert_allclose(
        np.asarray(psi[0, :]), np.asarray(psi_vac[0, :]), rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(psi[-1, :]), np.asarray(psi_vac[-1, :]), rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(psi[:, 0]), np.asarray(psi_vac[:, 0]), rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(psi[:, -1]), np.asarray(psi_vac[:, -1]), rtol=0.0, atol=1e-12
    )


def test_free_boundary_solve_responds_to_coil_boundary_shift():
    """Changing PF coil currents must change the free-boundary wall flux."""
    R = jnp.linspace(2.0, 10.0, 17)
    Z = jnp.linspace(-4.0, 4.0, 17)
    coil_R = jnp.array([3.5, 8.0, 9.5])
    coil_Z = jnp.array([3.0, -3.0, 0.0])
    coil_I = jnp.array([-1.0, 4.0, 6.0])
    shifted_I = coil_I.at[2].set(8.0)

    psi = solve_equilibrium_jax(
        R, Z, coil_R, coil_Z, coil_I, Ip=15.0, max_picard=4, sor_per_picard=4
    )
    shifted = solve_equilibrium_jax(
        R, Z, coil_R, coil_Z, shifted_I, Ip=15.0, max_picard=4, sor_per_picard=4
    )

    boundary_delta = np.max(np.abs(np.asarray(shifted[[0, -1], :]) - np.asarray(psi[[0, -1], :])))
    assert boundary_delta > 1e-6


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
