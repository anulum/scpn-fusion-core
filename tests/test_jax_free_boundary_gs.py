# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the SI free-boundary differentiable Grad-Shafranov solver.

Covers the toroidal Green's function, vacuum superposition, the general
``p'``/``FF'`` source, the free-boundary Picard/weighted-Jacobi solve, and the
emergent plasma current. The load-bearing checks are (1) the solved ``ψ`` satisfies
the discrete Grad-Shafranov equation in the plasma interior and (2) ``jax.grad``
matches a finite-difference gradient — the differentiability that motivates the
DIII-D IDA collaboration.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from scpn_fusion.core.jax_equilibrium_solver import _boundary_flux_level, _interior_axis_flux
from scpn_fusion.core.jax_free_boundary_gs import (
    MU0_SI,
    general_gs_source,
    greens_psi_si,
    normalised_flux,
    solve_free_boundary_gs,
    toroidal_plasma_current,
    vacuum_field_si,
)


# ── Shared DIII-D-like case (SI units) ────────────────────────────


def _case():
    R = jnp.linspace(0.9, 2.5, 33)
    Z = jnp.linspace(-1.3, 1.3, 33)
    coil_R = jnp.array([1.0, 1.0, 2.4, 2.4])
    coil_Z = jnp.array([1.2, -1.2, 0.9, -0.9])
    coil_I = jnp.array([-1.5e6, -1.5e6, 1.0e6, 1.0e6])  # amperes
    psin = jnp.linspace(0.0, 1.0, 12)
    pprime = -1.0e5 * (1.0 - psin**2)
    ffprime = -0.5 * (1.0 - psin**2)
    return R, Z, coil_R, coil_Z, coil_I, psin, pprime, ffprime


def _gs_residual_interior(psi, R, Z, psin, pprime, ffprime, psi_vac):
    """Max |Δ*ψ − S| relative to |S|, restricted to the plasma interior (0 < ψ_N < 1)."""
    p = np.asarray(psi, dtype=np.float64)
    r = np.asarray(R, dtype=np.float64)
    z = np.asarray(Z, dtype=np.float64)
    d_r = r[1] - r[0]
    d_z = z[1] - z[0]
    d2r = (p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2]) / d_r**2
    d1r = (p[:, 2:] - p[:, :-2]) / (2 * d_r)
    d2z = (p[2:, :] - 2 * p[1:-1, :] + p[:-2, :]) / d_z**2
    rg = r[np.newaxis, 1:-1]
    lap = d2r[1:-1, :] - d1r[1:-1, :] / rg + d2z[:, 1:-1]
    axis = _interior_axis_flux(psi)
    bnd = _boundary_flux_level(psi_vac)
    src = np.asarray(general_gs_source(psi, R, axis, bnd, psin, pprime, ffprime), dtype=np.float64)
    psi_n = np.asarray(normalised_flux(psi, axis, bnd))
    res = lap - src[1:-1, 1:-1]
    plasma = (psi_n[1:-1, 1:-1] < 0.999) & (psi_n[1:-1, 1:-1] > 0.0)
    assert plasma.sum() > 50  # a real plasma region exists
    return float(np.max(np.abs(res[plasma])) / np.max(np.abs(src[1:-1, 1:-1][plasma])))


# ── SI constant ───────────────────────────────────────────────────


def test_mu0_si_value():
    assert abs(float(MU0_SI) - 4.0e-7 * np.pi) < 1e-18


# ── Green's function (SI) ─────────────────────────────────────────


def test_greens_psi_si_finite_and_shape():
    R = jnp.array([1.0, 1.5, 2.0])
    Z = jnp.array([0.0, 0.0, 0.0])
    psi = greens_psi_si(R, Z, 1.5, 0.0, 1.0e6)
    assert psi.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(psi)))


def test_greens_psi_si_up_down_symmetry():
    """Ψ(R, +Z) = Ψ(R, −Z) for a midplane coil."""
    R = jnp.array([1.5])
    up = greens_psi_si(R, jnp.array([0.4]), 1.5, 0.0, 1.0e6)
    dn = greens_psi_si(R, jnp.array([-0.4]), 1.5, 0.0, 1.0e6)
    assert abs(float(up[0] - dn[0])) < 1e-9


def test_greens_psi_si_linear_in_current():
    R = jnp.array([1.2, 1.8])
    Z = jnp.array([0.1, -0.2])
    single = greens_psi_si(R, Z, 1.5, 0.0, 1.0e6)
    double = greens_psi_si(R, Z, 1.5, 0.0, 2.0e6)
    np.testing.assert_allclose(np.asarray(double), 2.0 * np.asarray(single), rtol=1e-6)


def test_greens_psi_si_scales_with_mu0():
    R = jnp.array([1.4])
    Z = jnp.array([0.0])
    a = greens_psi_si(R, Z, 1.5, 0.0, 1.0e6, mu0=MU0_SI)
    b = greens_psi_si(R, Z, 1.5, 0.0, 1.0e6, mu0=2.0 * MU0_SI)
    np.testing.assert_allclose(np.asarray(b), 2.0 * np.asarray(a), rtol=1e-6)


# ── Vacuum field ──────────────────────────────────────────────────


def test_vacuum_field_si_shape_nz_nr():
    R, Z, cR, cZ, cI, *_ = _case()
    psi = vacuum_field_si(R, Z, cR, cZ, cI)
    assert psi.shape == (Z.shape[0], R.shape[0])  # (NZ, NR)
    assert bool(jnp.all(jnp.isfinite(psi)))


def test_vacuum_field_si_superposition_of_single_coils():
    R, Z, cR, cZ, cI, *_ = _case()
    total = vacuum_field_si(R, Z, cR, cZ, cI)
    r2d = R[jnp.newaxis, :]
    z2d = Z[:, jnp.newaxis]
    summed = sum(
        greens_psi_si(r2d, z2d, float(cR[k]), float(cZ[k]), float(cI[k]))
        for k in range(cR.shape[0])
    )
    np.testing.assert_allclose(np.asarray(total), np.asarray(summed), rtol=1e-6, atol=1e-9)


def test_vacuum_field_si_linear_in_coil_current():
    R, Z, cR, cZ, cI, *_ = _case()
    single = vacuum_field_si(R, Z, cR, cZ, cI)
    double = vacuum_field_si(R, Z, cR, cZ, 2.0 * cI)
    np.testing.assert_allclose(np.asarray(double), 2.0 * np.asarray(single), rtol=1e-6, atol=1e-9)


# ── Normalised flux ───────────────────────────────────────────────


def test_normalised_flux_range_and_endpoints():
    psi = jnp.linspace(-2.0, 3.0, 50).reshape(5, 10)
    psi_n = normalised_flux(psi, jnp.array(0.0), jnp.array(2.0))
    assert float(jnp.min(psi_n)) >= 0.0
    assert float(jnp.max(psi_n)) <= 1.0


def test_normalised_flux_degenerate_denominator_is_finite():
    """ψ_axis == ψ_boundary must not produce NaN/inf (sign-aware floor)."""
    psi = jnp.linspace(-1.0, 1.0, 20).reshape(4, 5)
    psi_n = normalised_flux(psi, jnp.array(0.5), jnp.array(0.5))
    assert bool(jnp.all(jnp.isfinite(psi_n)))


def test_normalised_flux_handles_decreasing_convention():
    """Decreasing-ψ (axis above boundary) still maps into [0, 1]."""
    psi = jnp.linspace(0.0, 1.0, 20).reshape(4, 5)
    psi_n = normalised_flux(psi, jnp.array(1.0), jnp.array(0.0))
    assert float(jnp.min(psi_n)) >= 0.0
    assert float(jnp.max(psi_n)) <= 1.0


# ── General GS source ─────────────────────────────────────────────


def test_general_gs_source_zero_for_zero_profiles():
    R, Z, cR, cZ, cI, psin, *_ = _case()
    psi_vac = vacuum_field_si(R, Z, cR, cZ, cI)
    axis = _interior_axis_flux(psi_vac)
    bnd = _boundary_flux_level(psi_vac)
    src = general_gs_source(psi_vac, R, axis, bnd, psin, jnp.zeros(12), jnp.zeros(12))
    assert float(jnp.max(jnp.abs(src))) == 0.0


def test_general_gs_source_matches_analytic_formula_inside():
    """S = −(μ₀ R² p' + FF') where 0 ≤ ψ_N < 1, and 0 outside."""
    R, Z, cR, cZ, cI, psin, pprime, ffprime = _case()
    psi = solve_free_boundary_gs(R, Z, cR, cZ, cI, psin, pprime, ffprime)
    psi_vac = vacuum_field_si(R, Z, cR, cZ, cI)
    axis = _interior_axis_flux(psi)
    bnd = _boundary_flux_level(psi_vac)
    src = np.asarray(general_gs_source(psi, R, axis, bnd, psin, pprime, ffprime))
    psi_n = np.asarray(normalised_flux(psi, axis, bnd))
    pp = np.interp(psi_n, np.asarray(psin), np.asarray(pprime))
    ffp = np.interp(psi_n, np.asarray(psin), np.asarray(ffprime))
    r2d = np.asarray(R)[np.newaxis, :]
    expected = np.where(psi_n < 1.0, -(float(MU0_SI) * r2d**2 * pp + ffp), 0.0)
    np.testing.assert_allclose(src, expected, rtol=1e-6, atol=1e-9)


def test_general_gs_source_subcell_one_is_bit_identical_default():
    """subcell=1 (and the implicit default) must be the EXACT historical point source."""
    R, Z, cR, cZ, cI, psin, pprime, ffprime = _case()
    psi = solve_free_boundary_gs(R, Z, cR, cZ, cI, psin, pprime, ffprime)
    psi_vac = vacuum_field_si(R, Z, cR, cZ, cI)
    axis = _interior_axis_flux(psi)
    bnd = _boundary_flux_level(psi_vac)
    a = general_gs_source(psi, R, axis, bnd, psin, pprime, ffprime)
    b = general_gs_source(psi, R, axis, bnd, psin, pprime, ffprime, subcell=1)
    assert float(jnp.max(jnp.abs(a - b))) == 0.0


def test_general_gs_source_subcell_matches_point_in_smooth_interior():
    """Away from the LCFS the 4x4 average equals the point sample to sub-cell Taylor
    accuracy; in LCFS-straddling cells it must genuinely differ (the mechanism)."""
    R, Z, cR, cZ, cI, psin, pprime, ffprime = _case()
    psi = solve_free_boundary_gs(R, Z, cR, cZ, cI, psin, pprime, ffprime)
    psi_vac = vacuum_field_si(R, Z, cR, cZ, cI)
    axis = _interior_axis_flux(psi)
    bnd = _boundary_flux_level(psi_vac)
    point = np.asarray(general_gs_source(psi, R, axis, bnd, psin, pprime, ffprime))
    avg = np.asarray(general_gs_source(psi, R, axis, bnd, psin, pprime, ffprime, subcell=4))
    assert bool(np.all(np.isfinite(avg)))
    psi_n = np.asarray(normalised_flux(psi, axis, bnd))
    scale = float(np.max(np.abs(point)))
    core = psi_n < 0.85
    shell = (psi_n > 0.9) & (psi_n <= 1.0)
    core_diff = float(np.max(np.abs(avg - point)[core])) / scale
    shell_diff = float(np.max(np.abs(avg - point)[shell])) / scale
    assert core_diff < 0.05
    # the averaging path must genuinely engage (paths differ); the edge-DOMINANCE of the
    # effect is a property of steep real profiles and is pinned by the DIII-D 145419
    # validation artifact (0.72 % -> 0.48 %), not by this synthetic case whose profiles
    # vanish at the edge
    assert shell_diff > 0.0


def test_general_gs_source_subcell_deterministic_and_fails_closed():
    R, Z, cR, cZ, cI, psin, pprime, ffprime = _case()
    psi_vac = vacuum_field_si(R, Z, cR, cZ, cI)
    axis = _interior_axis_flux(psi_vac)
    bnd = _boundary_flux_level(psi_vac)
    a = general_gs_source(psi_vac, R, axis, bnd, psin, pprime, ffprime, subcell=4)
    b = general_gs_source(psi_vac, R, axis, bnd, psin, pprime, ffprime, subcell=4)
    assert float(jnp.max(jnp.abs(a - b))) == 0.0
    with pytest.raises(ValueError, match="subcell"):
        general_gs_source(psi_vac, R, axis, bnd, psin, pprime, ffprime, subcell=0)


def test_general_gs_source_vanishes_outside_lcfs():
    R, Z, cR, cZ, cI, psin, pprime, ffprime = _case()
    psi = solve_free_boundary_gs(R, Z, cR, cZ, cI, psin, pprime, ffprime)
    psi_vac = vacuum_field_si(R, Z, cR, cZ, cI)
    axis = _interior_axis_flux(psi)
    bnd = _boundary_flux_level(psi_vac)
    src = np.asarray(general_gs_source(psi, R, axis, bnd, psin, pprime, ffprime))
    psi_n = np.asarray(normalised_flux(psi, axis, bnd))
    assert np.all(src[psi_n >= 1.0] == 0.0)


# ── Free-boundary solve: correctness ──────────────────────────────


def test_solve_shape_and_finite():
    R, Z, cR, cZ, cI, psin, pprime, ffprime = _case()
    psi = solve_free_boundary_gs(R, Z, cR, cZ, cI, psin, pprime, ffprime)
    assert psi.shape == (Z.shape[0], R.shape[0])
    assert bool(jnp.all(jnp.isfinite(psi)))


def test_solve_satisfies_grad_shafranov_in_plasma():
    """The load-bearing physics check: Δ*ψ = −μ₀R²p' − FF' inside the LCFS."""
    R, Z, cR, cZ, cI, psin, pprime, ffprime = _case()
    psi = solve_free_boundary_gs(R, Z, cR, cZ, cI, psin, pprime, ffprime)
    psi_vac = vacuum_field_si(R, Z, cR, cZ, cI)
    rel = _gs_residual_interior(psi, R, Z, psin, pprime, ffprime, psi_vac)
    assert rel < 1e-6  # measured ~7e-11 with default iterations


def test_solve_is_deterministic():
    R, Z, cR, cZ, cI, psin, pprime, ffprime = _case()
    a = solve_free_boundary_gs(R, Z, cR, cZ, cI, psin, pprime, ffprime)
    b = solve_free_boundary_gs(R, Z, cR, cZ, cI, psin, pprime, ffprime)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_solve_zero_profiles_gives_zero_source_fixed_point():
    """With no plasma, the source is identically zero (pure vacuum drive)."""
    R, Z, cR, cZ, cI, psin, *_ = _case()
    psi = solve_free_boundary_gs(R, Z, cR, cZ, cI, psin, jnp.zeros(12), jnp.zeros(12))
    psi_vac = vacuum_field_si(R, Z, cR, cZ, cI)
    axis = _interior_axis_flux(psi)
    bnd = _boundary_flux_level(psi_vac)
    src = general_gs_source(psi, R, axis, bnd, psin, jnp.zeros(12), jnp.zeros(12))
    assert float(jnp.max(jnp.abs(src))) == 0.0


# ── Free-boundary solve: differentiability (the collaboration point) ──


def test_solve_gradient_finite_and_nonzero():
    R, Z, cR, cZ, cI, psin, pprime, ffprime = _case()

    def loss(pp, ffp, ci):
        return jnp.mean(solve_free_boundary_gs(R, Z, cR, cZ, ci, psin, pp, ffp) ** 2)

    g_pp, g_ff, g_ci = jax.grad(loss, argnums=(0, 1, 2))(pprime, ffprime, cI)
    for g in (g_pp, g_ff, g_ci):
        assert bool(jnp.all(jnp.isfinite(g)))
        assert float(jnp.max(jnp.abs(g))) > 0.0


def test_solve_autodiff_matches_finite_difference():
    """Analytic ``jax.grad`` must match a central finite difference — the claim that
    replaces the IDA numerical five-point stencil."""
    R, Z, cR, cZ, cI, psin, pprime, ffprime = _case()

    def loss(ffp):
        return jnp.mean(solve_free_boundary_gs(R, Z, cR, cZ, cI, psin, pprime, ffp) ** 2)

    g_ad = np.asarray(jax.grad(loss)(ffprime))
    idx = 6
    eps = 1e-2 * abs(float(ffprime[idx]))
    g_fd = (float(loss(ffprime.at[idx].add(eps))) - float(loss(ffprime.at[idx].add(-eps)))) / (
        2 * eps
    )
    assert abs(g_ad[idx] - g_fd) / (abs(g_fd) + 1e-30) < 1e-4


def test_solve_rejects_over_relaxation_with_nonfinite(caplog):
    """Weighted-Jacobi is stable only for sor_omega ≤ 1; over-relaxation diverges.
    Documented as a contract, asserted so the guidance can't silently rot."""
    R, Z, cR, cZ, cI, psin, pprime, ffprime = _case()
    psi = solve_free_boundary_gs(
        R, Z, cR, cZ, cI, psin, pprime, ffprime, max_picard=40, sor_per_picard=40, sor_omega=1.9
    )
    assert not bool(jnp.all(jnp.isfinite(psi)))


# ── Emergent plasma current ───────────────────────────────────────


def test_toroidal_plasma_current_finite_and_signed():
    R, Z, cR, cZ, cI, psin, pprime, ffprime = _case()
    psi = solve_free_boundary_gs(R, Z, cR, cZ, cI, psin, pprime, ffprime)
    psi_vac = vacuum_field_si(R, Z, cR, cZ, cI)
    axis = _interior_axis_flux(psi)
    bnd = _boundary_flux_level(psi_vac)
    ip = toroidal_plasma_current(psi, R, Z, axis, bnd, psin, pprime, ffprime)
    assert bool(jnp.isfinite(ip))
    assert float(jnp.abs(ip)) > 0.0


def test_toroidal_plasma_current_zero_for_zero_profiles():
    R, Z, cR, cZ, cI, psin, *_ = _case()
    psi = solve_free_boundary_gs(R, Z, cR, cZ, cI, psin, jnp.zeros(12), jnp.zeros(12))
    psi_vac = vacuum_field_si(R, Z, cR, cZ, cI)
    axis = _interior_axis_flux(psi)
    bnd = _boundary_flux_level(psi_vac)
    ip = toroidal_plasma_current(psi, R, Z, axis, bnd, psin, jnp.zeros(12), jnp.zeros(12))
    assert float(ip) == 0.0


def test_toroidal_plasma_current_scales_with_profile_amplitude():
    R, Z, cR, cZ, cI, psin, pprime, ffprime = _case()
    psi = solve_free_boundary_gs(R, Z, cR, cZ, cI, psin, pprime, ffprime)
    psi_vac = vacuum_field_si(R, Z, cR, cZ, cI)
    axis = _interior_axis_flux(psi)
    bnd = _boundary_flux_level(psi_vac)
    ip1 = toroidal_plasma_current(psi, R, Z, axis, bnd, psin, pprime, ffprime)
    ip2 = toroidal_plasma_current(psi, R, Z, axis, bnd, psin, 2.0 * pprime, 2.0 * ffprime)
    np.testing.assert_allclose(float(ip2), 2.0 * float(ip1), rtol=1e-6)
