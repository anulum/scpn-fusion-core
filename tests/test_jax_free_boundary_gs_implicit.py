# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the Krylov-forward, implicit-differentiation free-boundary GS solver.

The load-bearing claims: (1) the Krylov forward converges to the same equilibrium as
the weighted-Jacobi reference solver; (2) the implicit-diff gradient of the profile
parameters (``p'``, ``FF'``) is *exact* (matches finite differences), which is the
whole point — it replaces the numerical stencil the DIII-D IDA loop uses; and (3) that
gradient is independent of the forward iteration count (the implicit-function-theorem
property). The coil-current gradient's documented ~2 % accuracy is pinned so the
limitation cannot silently drift.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from scpn_fusion.core.jax_equilibrium_solver import _boundary_flux_level, _interior_axis_flux
from scpn_fusion.core.jax_free_boundary_gs import (
    general_gs_source,
    normalised_flux,
    solve_free_boundary_gs,
    vacuum_field_si,
)
from scpn_fusion.core.jax_free_boundary_gs_implicit import (
    nonlinear_gs_residual,
    solve_free_boundary_gs_implicit,
)


def _case(n: int = 33):
    R = jnp.linspace(0.9, 2.5, n)
    Z = jnp.linspace(-1.3, 1.3, n)
    coil_R = jnp.array([1.0, 1.0, 2.4, 2.4])
    coil_Z = jnp.array([1.2, -1.2, 0.9, -0.9])
    coil_I = jnp.array([-1.5e6, -1.5e6, 1.0e6, 1.0e6])
    psin = jnp.linspace(0.0, 1.0, 12)
    pprime = -1.0e5 * (1.0 - psin**2)
    ffprime = -0.5 * (1.0 - psin**2)
    return R, Z, coil_R, coil_Z, coil_I, psin, pprime, ffprime


def _gs_residual_interior(psi, R, Z, psin, pprime, ffprime, psi_vac):
    p = np.asarray(psi, dtype=np.float64)
    r = np.asarray(R, dtype=np.float64)
    z = np.asarray(Z, dtype=np.float64)
    d_r = r[1] - r[0]
    d_z = z[1] - z[0]
    d2r = (p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2]) / d_r**2
    d1r = (p[:, 2:] - p[:, :-2]) / (2 * d_r)
    d2z = (p[2:, :] - 2 * p[1:-1, :] + p[:-2, :]) / d_z**2
    lap = d2r[1:-1, :] - d1r[1:-1, :] / r[np.newaxis, 1:-1] + d2z[:, 1:-1]
    axis = _interior_axis_flux(psi)
    bnd = _boundary_flux_level(psi_vac)
    src = np.asarray(general_gs_source(psi, R, axis, bnd, psin, pprime, ffprime), dtype=np.float64)
    psi_n = np.asarray(normalised_flux(psi, axis, bnd))
    res = lap - src[1:-1, 1:-1]
    plasma = (psi_n[1:-1, 1:-1] < 0.999) & (psi_n[1:-1, 1:-1] > 0.0)
    assert plasma.sum() > 50
    return float(np.max(np.abs(res[plasma])) / np.max(np.abs(src[1:-1, 1:-1][plasma])))


# ── Forward correctness ───────────────────────────────────────────


def test_implicit_forward_shape_and_finite():
    R, Z, cR, cZ, cI, psin, pp, ff = _case()
    psi = solve_free_boundary_gs_implicit(cI, pp, ff, R, Z, cR, cZ, psin, 40)
    assert psi.shape == (Z.shape[0], R.shape[0])
    assert bool(jnp.all(jnp.isfinite(psi)))


def test_implicit_forward_satisfies_grad_shafranov():
    R, Z, cR, cZ, cI, psin, pp, ff = _case()
    psi = solve_free_boundary_gs_implicit(cI, pp, ff, R, Z, cR, cZ, psin, 40)
    psi_vac = vacuum_field_si(R, Z, cR, cZ, cI)
    rel = _gs_residual_interior(psi, R, Z, psin, pp, ff, psi_vac)
    assert rel < 1e-7  # Krylov forward converges tightly at every resolution


def test_implicit_matches_jacobi_reference():
    """Two independent algorithms (Krylov vs weighted-Jacobi) reach the same ψ."""
    R, Z, cR, cZ, cI, psin, pp, ff = _case()
    psi_imp = solve_free_boundary_gs_implicit(cI, pp, ff, R, Z, cR, cZ, psin, 40)
    psi_jac = solve_free_boundary_gs(
        R, Z, cR, cZ, cI, psin, pp, ff, max_picard=100, sor_per_picard=100, sor_omega=0.9
    )
    rel = float(jnp.max(jnp.abs(psi_imp - psi_jac)) / jnp.max(jnp.abs(psi_jac)))
    assert rel < 5e-3


def test_implicit_forward_deterministic():
    R, Z, cR, cZ, cI, psin, pp, ff = _case()
    a = solve_free_boundary_gs_implicit(cI, pp, ff, R, Z, cR, cZ, psin, 40)
    b = solve_free_boundary_gs_implicit(cI, pp, ff, R, Z, cR, cZ, psin, 40)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_nonlinear_residual_near_zero_at_solution():
    R, Z, cR, cZ, cI, psin, pp, ff = _case()
    psi = solve_free_boundary_gs_implicit(cI, pp, ff, R, Z, cR, cZ, psin, 40)
    res = nonlinear_gs_residual(psi, cI, pp, ff, R, Z, cR, cZ, psin)
    # interior residual scaled by source magnitude is tiny; boundary rows are ψ−ψ_vac ≈ 0
    assert float(jnp.max(jnp.abs(res[1:-1, 1:-1]))) < 1e-3


# ── Implicit-diff gradients ───────────────────────────────────────


def test_profile_gradients_are_exact():
    """∂/∂FF' and ∂/∂p' match central finite differences — the headline property."""
    R, Z, cR, cZ, cI, psin, pp, ff = _case()

    def loss_ff(ffv):
        return jnp.mean(solve_free_boundary_gs_implicit(cI, pp, ffv, R, Z, cR, cZ, psin, 50) ** 2)

    def loss_pp(ppv):
        return jnp.mean(solve_free_boundary_gs_implicit(cI, ppv, ff, R, Z, cR, cZ, psin, 50) ** 2)

    for loss, theta in ((loss_ff, ff), (loss_pp, pp)):
        g_ad = np.asarray(jax.grad(loss)(theta))
        assert bool(np.all(np.isfinite(g_ad)))
        idx = 6
        eps = 1e-2 * abs(float(theta[idx]))
        g_fd = (float(loss(theta.at[idx].add(eps))) - float(loss(theta.at[idx].add(-eps)))) / (
            2 * eps
        )
        assert abs(g_ad[idx] - g_fd) / (abs(g_fd) + 1e-30) < 1e-4


def test_profile_gradient_independent_of_picard_count():
    """Implicit-function-theorem property: the gradient depends on the converged root,
    not on how many Picard iterations reached it."""
    R, Z, cR, cZ, cI, psin, pp, ff = _case()

    def grad_at(n):
        return np.asarray(
            jax.grad(
                lambda ffv: jnp.mean(
                    solve_free_boundary_gs_implicit(cI, pp, ffv, R, Z, cR, cZ, psin, n) ** 2
                )
            )(ff)
        )

    g40 = grad_at(40)
    g70 = grad_at(70)
    np.testing.assert_allclose(g40, g70, rtol=1e-4, atol=1e-12)


def test_coil_current_gradient_finite_and_within_documented_tolerance():
    """∂/∂I_coil is present but only ~2 % accurate (non-smooth O-point/LCFS
    normalisation). Pinned so the documented limit cannot silently regress or improve
    unnoticed."""
    R, Z, cR, cZ, cI, psin, pp, ff = _case()

    def loss(civ):
        return jnp.mean(solve_free_boundary_gs_implicit(civ, pp, ff, R, Z, cR, cZ, psin, 50) ** 2)

    g_ad = np.asarray(jax.grad(loss)(cI))
    assert bool(np.all(np.isfinite(g_ad)))
    idx = 2
    eps = 1e-3 * abs(float(cI[idx]))
    g_fd = (float(loss(cI.at[idx].add(eps))) - float(loss(cI.at[idx].add(-eps)))) / (2 * eps)
    rel = abs(g_ad[idx] - g_fd) / (abs(g_fd) + 1e-30)
    assert rel < 3e-2  # documented ~2 % (see module docstring); not yet exact
