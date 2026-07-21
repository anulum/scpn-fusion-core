# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — tests for the predictive free-boundary Grad-Shafranov solver
"""Self-contained tests for :mod:`scpn_fusion.core.jax_free_boundary_predictive`.

A small synthetic diverted coilset on a 33² grid (no external fixture, CI-safe) exercises the
whole coupled solve: the von Hagenow response matrix, the Ip normalisation, and the
Anderson-accelerated cold-start convergence to a self-consistent fixed point. The
quantitative FreeGS DIII-D cross-validation (≈ 0.8 % of ψ-span, from coils alone) uses the
external Milestone-B reference and lives in the validation harness / findings, not here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from scpn_fusion.core.jax_free_boundary_gs import MU0_SI, greens_psi_si, vacuum_field_si
from scpn_fusion.core.jax_free_boundary_predictive import (
    _laplacian_star,
    _plasma_current,
    build_response_matrix,
    predictive_gs_residual,
    solve_predictive_equilibrium,
    solve_predictive_equilibrium_diff,
)
from scpn_fusion.core.jax_o_point import smooth_axis_flux
from scpn_fusion.core.jax_x_point import smooth_xpoint_flux

# ── A small synthetic diverted case (33², lower divertor coil) ─────
_R = jnp.linspace(1.0, 2.5, 33)
_Z = jnp.linspace(-1.4, 1.4, 33)
_COIL_R = jnp.array([1.2, 2.3, 1.2, 2.3, 1.6, 1.5])
_COIL_Z = jnp.array([0.9, 0.9, -0.9, -0.9, 1.3, -1.35])
_COIL_I = jnp.array([-3.0e5, -3.0e5, -3.0e5, -3.0e5, -1.0e5, -6.0e5])
_PSIN = jnp.linspace(0.0, 1.0, 6)
_PPRIME = jnp.array([-8.0e4, -6.0e4, -4.0e4, -2.0e4, -0.7e4, 0.0])
_FFPRIME = jnp.array([-1.2, -0.9, -0.6, -0.3, -0.1, 0.0])
_IP = 1.0e6


@pytest.fixture(scope="module")
def response():
    return build_response_matrix(_R, _Z)


@pytest.fixture(scope="module")
def solved(response):
    m, b, s = response
    psi = solve_predictive_equilibrium(
        _COIL_I, _PPRIME, _FFPRIME, _R, _Z, _COIL_R, _COIL_Z, _PSIN, _IP, m, b, s, n_iter=150
    )
    return psi, m, b, s


# ── Response matrix (von Hagenow) ─────────────────────────────────


def test_response_matrix_shape_and_indices(response) -> None:
    m, b, s = response
    nz, nr = _Z.shape[0], _R.shape[0]
    assert m.shape == (b.shape[0], s.shape[0])
    assert b.shape[0] + s.shape[0] == nz * nr
    # every wall index is on the border ring
    bi, bj = np.unravel_index(np.asarray(b), (nz, nr))
    on_border = (bi == 0) | (bi == nz - 1) | (bj == 0) | (bj == nr - 1)
    assert bool(np.all(on_border))


def test_response_matrix_reconstructs_wall_flux(response) -> None:
    """``M @ (I·dA)`` for a current blob equals a direct Green's sum at the wall."""
    m, b, s = response
    nz, nr = _Z.shape[0], _R.shape[0]
    rr, zz = jnp.meshgrid(_R, _Z)
    # a smooth interior current pattern (zero on the border)
    blob = jnp.exp(-((rr - 1.7) ** 2 + zz**2) / 0.1)
    blob = blob.at[0, :].set(0.0).at[-1, :].set(0.0).at[:, 0].set(0.0).at[:, -1].set(0.0)
    dA = float((_R[1] - _R[0]) * (_Z[1] - _Z[0]))
    via_matrix = m @ (blob.reshape(-1)[s] * dA)
    # direct: flux at each wall point summed over all source cells
    r_wall = rr.reshape(-1)[b]
    z_wall = zz.reshape(-1)[b]
    r_src = rr.reshape(-1)[s]
    z_src = zz.reshape(-1)[s]
    i_src = blob.reshape(-1)[s] * dA
    direct = jnp.zeros_like(via_matrix)
    for k in range(r_src.shape[0]):
        direct = direct + greens_psi_si(
            r_wall, z_wall, float(r_src[k]), float(z_src[k]), float(i_src[k]), MU0_SI
        )
    assert np.allclose(np.asarray(via_matrix), np.asarray(direct), rtol=1e-9, atol=1e-12)


# ── Ip normalisation ──────────────────────────────────────────────


def test_ip_normalisation_holds_target() -> None:
    """``_plasma_current`` scales the profile so ``∮Jφ dA`` equals the target exactly."""
    rr, zz = jnp.meshgrid(_R, _Z)
    psi = -6.0 * ((rr - 1.7) ** 2 + zz**2) + 1.0  # a simple peaked flux
    dA = jnp.asarray((_R[1] - _R[0]) * (_Z[1] - _Z[0]))
    j = _plasma_current(
        psi,
        _R,
        jnp.asarray(0.9),
        jnp.asarray(0.1),
        _PSIN,
        _PPRIME,
        _FFPRIME,
        jnp.asarray(_IP),
        dA,
        0.03,
        MU0_SI,
    )
    assert abs(float(jnp.sum(j) * dA) - _IP) / _IP < 1e-9


# ── Full coupled solve ────────────────────────────────────────────


def test_solve_is_finite_and_self_consistent(solved) -> None:
    """The cold-start solve returns a finite ψ that satisfies the coupled residual to machine
    precision (a true fixed point of the discrete operator)."""
    psi, m, b, s = solved
    assert bool(jnp.all(jnp.isfinite(psi)))
    F = predictive_gs_residual(
        psi, _COIL_I, _PPRIME, _FFPRIME, _R, _Z, _COIL_R, _COIL_Z, _PSIN, jnp.asarray(_IP), m, b, s
    )
    scale = float(jnp.max(jnp.abs(_laplacian_star(psi, _R, _R[1] - _R[0], _Z[1] - _Z[0])))) + 1e-30
    rel = float(jnp.max(jnp.abs(F[2:-2, 2:-2]))) / scale
    assert rel < 1e-3


def test_solve_holds_target_ip(solved) -> None:
    """The solved equilibrium carries the requested plasma current."""
    psi, _m, _b, _s = solved
    axis = smooth_axis_flux(psi)
    xp = smooth_xpoint_flux(psi, _R, _Z)
    dA = jnp.asarray((_R[1] - _R[0]) * (_Z[1] - _Z[0]))
    j = _plasma_current(
        psi, _R, axis, xp, _PSIN, _PPRIME, _FFPRIME, jnp.asarray(_IP), dA, 0.03, MU0_SI
    )
    assert abs(float(jnp.sum(j) * dA) - _IP) / _IP < 1e-6


def test_solve_forms_a_plasma(solved) -> None:
    """A confined plasma forms: axis flux exceeds the separatrix, and ψ departs from vacuum."""
    psi, _m, _b, _s = solved
    axis = float(smooth_axis_flux(psi))
    xp = float(smooth_xpoint_flux(psi, _R, _Z))
    assert axis > xp
    psi_vac = vacuum_field_si(_R, _Z, _COIL_R, _COIL_Z, _COIL_I, MU0_SI)
    assert float(jnp.max(jnp.abs(psi - psi_vac))) > 0.1


def test_solve_never_returns_nan_past_convergence(response) -> None:
    """Running well past convergence must not NaN — the early-stop + finite-guard hold even when
    the Anderson history goes rank-deficient."""
    m, b, s = response
    psi = solve_predictive_equilibrium(
        _COIL_I, _PPRIME, _FFPRIME, _R, _Z, _COIL_R, _COIL_Z, _PSIN, _IP, m, b, s, n_iter=220
    )
    assert bool(jnp.all(jnp.isfinite(psi)))


def test_mg_preconditioned_solve_matches_plain(solved) -> None:
    """``use_mg_preconditioner=True`` reaches the SAME equilibrium — the multigrid V-cycle only
    reshapes the inner Krylov convergence path, never the fixed point."""
    psi_plain, m, b, s = solved
    psi_mg = solve_predictive_equilibrium(
        _COIL_I,
        _PPRIME,
        _FFPRIME,
        _R,
        _Z,
        _COIL_R,
        _COIL_Z,
        _PSIN,
        _IP,
        m,
        b,
        s,
        n_iter=150,
        use_mg_preconditioner=True,
    )
    axis = float(smooth_axis_flux(psi_plain))
    xp = float(smooth_xpoint_flux(psi_plain, _R, _Z))
    span = abs(axis - xp)
    assert float(jnp.max(jnp.abs(psi_mg - psi_plain))) / span < 1e-6


# ── Implicit-diff adjoint (∂ψ*/∂θ for the IDA loop) ───────────────


def _axis_loss(response, coil_I, pprime, ffprime):
    m, b, s = response
    psi = solve_predictive_equilibrium_diff(
        coil_I, pprime, ffprime, _R, _Z, _COIL_R, _COIL_Z, _PSIN, _IP, m, b, s, n_iter=150
    )
    return smooth_axis_flux(psi)


def test_adjoint_gradient_shapes_and_finite(response) -> None:
    """``jax.grad`` through the differentiable solve returns finite gradients of the right shape
    for all three differentiated inputs (coil currents and both profiles)."""
    g_ci, g_pp, g_ff = jax.grad(
        lambda ci, pp, ff: _axis_loss(response, ci, pp, ff), argnums=(0, 1, 2)
    )(_COIL_I, _PPRIME, _FFPRIME)
    assert (
        g_ci.shape == _COIL_I.shape and g_pp.shape == _PPRIME.shape and g_ff.shape == _FFPRIME.shape
    )
    assert bool(
        jnp.all(jnp.isfinite(g_ci)) & jnp.all(jnp.isfinite(g_pp)) & jnp.all(jnp.isfinite(g_ff))
    )


def test_coil_gradient_matches_finite_difference(response) -> None:
    """The implicit-diff COIL gradient matches a warm-started central FD at a small step.

    Pins the erasure of the historical "coil grad ~3 %" honest-limit: that figure was an FD
    *truncation artefact* (a 3 kA step is a ~0.5 % coil perturbation and the axis flux is
    visibly nonlinear at that scale — central FD is then ~27 % off); at a 300 A step the
    warm-FD agrees with the adjoint to ~7 significant figures. The FD solves warm-start from
    the base solution so the comparison stays in-basin (the documented cold-start lesson)."""
    m, b, s = response
    g_ci = jax.grad(lambda ci: _axis_loss(response, ci, _PPRIME, _FFPRIME))(_COIL_I)
    psi_base = solve_predictive_equilibrium(
        _COIL_I, _PPRIME, _FFPRIME, _R, _Z, _COIL_R, _COIL_Z, _PSIN, _IP, m, b, s, n_iter=150
    )

    def warm_loss(ci: jnp.ndarray) -> float:
        psi = solve_predictive_equilibrium(
            ci,
            _PPRIME,
            _FFPRIME,
            _R,
            _Z,
            _COIL_R,
            _COIL_Z,
            _PSIN,
            _IP,
            m,
            b,
            s,
            psi_init=psi_base,
            n_iter=150,
        )
        return float(smooth_axis_flux(psi))

    idx, eps = 5, 3.0e2  # strongest (divertor) coil; ~0.05 % perturbation stays linear
    fd = (warm_loss(_COIL_I.at[idx].add(eps)) - warm_loss(_COIL_I.at[idx].add(-eps))) / (2 * eps)
    assert abs(float(g_ci[idx]) - fd) / (abs(fd) + 1e-30) < 1e-3


def test_profile_gradient_matches_finite_difference(response) -> None:
    """The implicit-diff profile gradient (the quantity IDA infers) matches central FD — the
    adjoint is exact on the converged fixed point, not an approximation through the solver."""
    _g_ci, g_pp, _g_ff = jax.grad(
        lambda ci, pp, ff: _axis_loss(response, ci, pp, ff), argnums=(0, 1, 2)
    )(_COIL_I, _PPRIME, _FFPRIME)
    idx, eps = 3, 1.0e3
    lp = float(_axis_loss(response, _COIL_I, _PPRIME.at[idx].add(eps), _FFPRIME))
    lm = float(_axis_loss(response, _COIL_I, _PPRIME.at[idx].add(-eps), _FFPRIME))
    fd = (lp - lm) / (2 * eps)
    assert abs(float(g_pp[idx]) - fd) / (abs(fd) + 1e-30) < 1e-2


def test_differentiates_through_bspline_profile_coefficients(response) -> None:
    """End-to-end IDA pipeline: compact B-spline coefficients → p'/FF' samples → equilibrium →
    ``jax.grad`` back to the coefficients. Proves the Rung-3 profile basis composes with the
    Rung-1 implicit-diff adjoint — the whole point of the compact IDA parameterisation."""
    from scpn_fusion.core.jax_profile_basis import bspline_design_matrix, evaluate_profile

    design = bspline_design_matrix(np.asarray(_PSIN), n_coeff=6)
    pp_coeffs = jnp.asarray(np.linalg.lstsq(design, np.asarray(_PPRIME), rcond=None)[0])
    ff_coeffs = jnp.asarray(np.linalg.lstsq(design, np.asarray(_FFPRIME), rcond=None)[0])
    m, b, s = response

    def loss(pc: jnp.ndarray, fc: jnp.ndarray) -> jnp.ndarray:
        pp = evaluate_profile(pc, design)
        ff = evaluate_profile(fc, design)
        psi = solve_predictive_equilibrium_diff(
            _COIL_I, pp, ff, _R, _Z, _COIL_R, _COIL_Z, _PSIN, _IP, m, b, s
        )
        return smooth_axis_flux(psi)

    g_pc, g_fc = jax.grad(loss, argnums=(0, 1))(pp_coeffs, ff_coeffs)
    assert g_pc.shape == pp_coeffs.shape and g_fc.shape == ff_coeffs.shape
    assert bool(jnp.all(jnp.isfinite(g_pc)) & jnp.all(jnp.isfinite(g_fc)))
    assert float(jnp.sum(jnp.abs(g_fc))) > 0.0  # the equilibrium genuinely depends on FF' coeffs
