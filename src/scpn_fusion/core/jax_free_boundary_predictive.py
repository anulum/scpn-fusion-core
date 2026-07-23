# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — predictive free-boundary Grad-Shafranov (coils + profiles + Ip alone)
"""Predictive free-boundary Grad-Shafranov: solve the equilibrium from coils + profiles + Ip.

The companion solvers :mod:`scpn_fusion.core.jax_free_boundary_gs` and
:mod:`scpn_fusion.core.jax_free_boundary_gs_implicit` impose the computational-wall flux from
the **coils only** (``ψ_bnd = ψ_vac``). That omits the plasma's *own* contribution to the wall
flux — on the DIII-D reference the plasma self-field at the wall is ≈ 2.7× the coil field, so a
coil-only wall condition is ≈ 70 % wrong at the boundary. Those solvers are therefore accurate
in the interior *given* a good wall, but not **predictive** from coils alone.

This module closes that gap. It solves the free-boundary equilibrium self-consistently with
four coupled pieces (all functions of the current iterate ``ψ``):

1. **Plasma self-field boundary coupling (von Hagenow / boundary integral).** The wall flux is
   ``ψ_bnd = ψ_coil + M @ (Jφ·dA)``, where ``M[wall, interior]`` is the poloidal-flux response
   of each wall point to a unit current at each interior cell (a Green's-function matrix,
   geometry-fixed → precomputed once by :func:`build_response_matrix`).
2. **Ip current normalisation.** ``Jφ`` is scaled each step so ``∮Jφ dA = Ip_target``. Without
   this the coupled map is under-determined and runs away (the self-field feeds back positively).
3. **Self-consistent magnetic axis and separatrix.** ``ψ_axis`` from the smooth O-point
   (:func:`scpn_fusion.core.jax_o_point.smooth_axis_flux`) and ``ψ_bndry`` from the smooth
   X-point (:func:`scpn_fusion.core.jax_x_point.smooth_xpoint_flux`) — both found from ``ψ``
   each iteration, so the last-closed-flux-surface is not handed in.
4. **Anderson-accelerated fixed-point solve.** Naive Picard on this full nonlinear
   boundary+profile map is *unstable*: the physical fixed point is a saddle and simple
   iteration is driven to a spurious high-peaking attractor (a known reason production
   free-boundary codes use Newton/Anderson, not Picard). Anderson mixing (depth ``m``) with an
   ``Ip`` ramp converges to the true fixed point from a cold vacuum start.

The public FreeGS same-case accuracy is measured by
``validation/benchmark_ida_same_case.py`` and remains fail closed.  Convergence of the
discrete fixed point is not evidence of agreement with FreeGS: the two are reported as
separate residual and ``ψ_N`` metrics.

The gradient (``∂ψ*/∂θ`` for the IDA loop) is provided by :func:`solve_predictive_equilibrium_diff`,
which wraps the forward in a :func:`jax.custom_vjp` implicit-diff adjoint on the converged fixed
point ``F(ψ*, θ) = 0`` (:func:`predictive_gs_residual`) — the gradient cost is independent of the
Anderson iteration count. Validated vs central finite differences: on the synthetic 33² case the profile gradients
``∂/∂p'`` / ``∂/∂FF'`` (what IDA infers) match to ``< 2e-3`` and the coil-current gradient to a
few percent (the divertor current moves the near-wall X-point, the harder term); on the 65²
FreeGS DIII-D case the ``∂/∂FF'`` gradient matches an in-basin (warm-started) FD to ``~3e-5``.
The FD must be warm-started from the base ``ψ*`` — a cold-start FD at 65² can jump basins
(the solve's cold-start sensitivity) and is then not a local derivative.

Honest limits: (a) the Anderson hyper-parameters (``m``, mixing, ramp) are tuned on this case,
not yet auto-selected or Newton-backed; (b) validated on one case/grid so far; (c) the forward
solve is a Python Anderson loop (not real-time); (d) not real-DIII-D-data validated.

SI units throughout (μ₀ = 4π·10⁻⁷, currents [A], ψ [Wb]); ``ψ`` shape ``(NZ, NR)``.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, cast

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import bicgstab

from scpn_fusion.core.jax_free_boundary_gs import (
    MU0_SI,
    greens_psi_si,
    normalised_flux,
    vacuum_field_si,
)
from scpn_fusion.core.jax_multigrid_precond import build_gs_mg_preconditioner
from scpn_fusion.core.jax_o_point import smooth_axis_flux
from scpn_fusion.core.jax_x_point import smooth_xpoint_flux

# Anderson / continuation defaults used by the fail-closed public same-case benchmark.
DEFAULT_N_ITER = 120
DEFAULT_ANDERSON_DEPTH = 8
DEFAULT_MIXING = 0.5
DEFAULT_IP_RAMP = 30
DEFAULT_CUTOFF_WIDTH = 0.03
DEFAULT_TOL = 1.0e-9
_BICGSTAB_TOL = 1.0e-11
_BICGSTAB_MAXITER = 700
# The adjoint (∂F/∂ψ)ᵀ system mixes interior rows (Δ*, ~1/h²) and wall rows (~1); uncorrected it
# is badly scaled and its iteration count grows ~1/h², so a fixed cap is silently hit at finer
# grids and the gradient under-converges (a ~(h_coarse/h_fine)² error — ~4× at 65² if capped at
# 800). The Jacobi preconditioner in `_solve_diff_bwd` normalises that diagonal, making the
# BiCGSTAB convergence roughly grid-insensitive; this cap then only bounds pathological cases.
# Validated: preconditioned, 65² reaches a ~3e-5 warm-FD match within this cap.
_ADJOINT_MAXITER = 2000


# ── Geometry: von Hagenow response matrix ─────────────────────────


def _wall_source_indices(nz: int, nr: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Flat indices of the computational-wall ring and the interior source cells."""
    mask = (
        jnp.zeros((nz, nr), bool)
        .at[0, :]
        .set(True)
        .at[-1, :]
        .set(True)
        .at[:, 0]
        .set(True)
        .at[:, -1]
        .set(True)
    )
    flat = mask.reshape(-1)
    return jnp.where(flat)[0], jnp.where(~flat)[0]


def build_response_matrix(
    R_grid: jnp.ndarray, Z_grid: jnp.ndarray, mu0: float = MU0_SI
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Von Hagenow response matrix ``M[wall, interior]`` and the wall/interior flat indices.

    ``M[b, s]`` is the poloidal flux [Wb] induced at wall point ``b`` by a unit (1 A) toroidal
    current at interior cell ``s`` — the SI toroidal Green's function
    :func:`~scpn_fusion.core.jax_free_boundary_gs.greens_psi_si`. Depends only on the grid
    geometry (not on currents or profiles), so it is computed **once** and reused every Picard
    step as ``ψ_plasma_wall = M @ (Jφ_interior · dA)``.

    Returns ``(M, b_idx, s_idx)`` with ``M`` shape ``(N_wall, N_interior)``.
    """
    nz, nr = Z_grid.shape[0], R_grid.shape[0]
    rr, zz = jnp.meshgrid(R_grid, Z_grid)
    b_idx, s_idx = _wall_source_indices(nz, nr)
    r_wall = rr.reshape(-1)[b_idx]
    z_wall = zz.reshape(-1)[b_idx]
    r_src = rr.reshape(-1)[s_idx]
    z_src = zz.reshape(-1)[s_idx]

    def column(rs: jnp.ndarray, zs: jnp.ndarray) -> jnp.ndarray:
        return cast(jnp.ndarray, greens_psi_si(r_wall, z_wall, rs, zs, 1.0, mu0))

    m = jax.vmap(column, in_axes=(0, 0), out_axes=1)(r_src, z_src)
    return m, b_idx, s_idx


# ── GS operator (matrix-free, identity boundary rows) ─────────────


def _laplacian_star(
    psi: jnp.ndarray, R_grid: jnp.ndarray, d_r: jnp.ndarray, d_z: jnp.ndarray
) -> jnp.ndarray:
    """Grad-Shafranov operator ``Δ*ψ = ∂²ψ/∂R² − (1/R)∂ψ/∂R + ∂²ψ/∂Z²`` (interior only)."""
    d2r = (
        jnp.zeros_like(psi)
        .at[:, 1:-1]
        .set((psi[:, 2:] - 2.0 * psi[:, 1:-1] + psi[:, :-2]) / d_r**2)
    )
    d1r = jnp.zeros_like(psi).at[:, 1:-1].set((psi[:, 2:] - psi[:, :-2]) / (2.0 * d_r))
    d2z = (
        jnp.zeros_like(psi)
        .at[1:-1, :]
        .set((psi[2:, :] - 2.0 * psi[1:-1, :] + psi[:-2, :]) / d_z**2)
    )
    r_safe = jnp.maximum(R_grid[jnp.newaxis, :], 1e-6)
    return d2r - d1r / r_safe + d2z


def _gs_operator_flat(
    psi_flat: jnp.ndarray,
    shape: tuple[int, int],
    R_grid: jnp.ndarray,
    d_r: jnp.ndarray,
    d_z: jnp.ndarray,
) -> jnp.ndarray:
    """Linear GS operator with identity wall rows (Dirichlet imposed through the RHS)."""
    psi = psi_flat.reshape(shape)
    out = _laplacian_star(psi, R_grid, d_r, d_z)
    out = out.at[0, :].set(psi[0, :]).at[-1, :].set(psi[-1, :])
    out = out.at[:, 0].set(psi[:, 0]).at[:, -1].set(psi[:, -1])
    return out.reshape(-1)


# ── Plasma current (Ip-normalised, smooth LCFS cutoff) ────────────


def _plasma_current(
    psi: jnp.ndarray,
    R_grid: jnp.ndarray,
    psi_axis: jnp.ndarray,
    psi_bndry: jnp.ndarray,
    psin_knots: jnp.ndarray,
    pprime_vals: jnp.ndarray,
    ffprime_vals: jnp.ndarray,
    ip_target: jnp.ndarray,
    dA: jnp.ndarray,
    cutoff_width: float,
    mu0: float,
) -> jnp.ndarray:
    """Toroidal current density ``Jφ = R p' + FF'/(μ₀R)`` with a smooth LCFS roll-off, scaled
    so ``∮Jφ dA = ip_target``.

    The ``tanh`` roll-off (vs a hard ``ψ_N < 1`` mask) keeps the current — and thus the coupled
    fixed point — differentiable; the Ip scaling pins the total current, killing the self-field
    runaway.
    """
    psi_n = normalised_flux(psi, psi_axis, psi_bndry)
    pprime = jnp.interp(psi_n, psin_knots, pprime_vals)
    ffprime = jnp.interp(psi_n, psin_knots, ffprime_vals)
    r_safe = jnp.maximum(R_grid[jnp.newaxis, :], 1e-6)
    j_raw = r_safe * pprime + ffprime / (mu0 * r_safe)
    j_masked = j_raw * 0.5 * (1.0 - jnp.tanh((psi_n - 1.0) / cutoff_width))
    ip_now = jnp.sum(j_masked) * dA
    scale = ip_target / jnp.where(jnp.abs(ip_now) < 1.0, 1.0, ip_now)
    return j_masked * scale


def predictive_gs_residual(
    psi: jnp.ndarray,
    coil_I: jnp.ndarray,
    pprime_vals: jnp.ndarray,
    ffprime_vals: jnp.ndarray,
    R_grid: jnp.ndarray,
    Z_grid: jnp.ndarray,
    coil_R: jnp.ndarray,
    coil_Z: jnp.ndarray,
    psin_knots: jnp.ndarray,
    ip_target: jnp.ndarray,
    response_matrix: jnp.ndarray,
    wall_idx: jnp.ndarray,
    source_idx: jnp.ndarray,
    cutoff_width: float = DEFAULT_CUTOFF_WIDTH,
    mu0: float = MU0_SI,
) -> jnp.ndarray:
    """Coupled predictive GS residual ``F(ψ)`` whose root is the free-boundary equilibrium.

    Interior: ``Δ*ψ − S(ψ)`` with ``S = −μ₀ R Jφ`` and ``Jφ`` the Ip-normalised current.
    Wall rows: ``ψ − (ψ_coil + M @ (Jφ·dA))`` — the von Hagenow self-consistent Dirichlet
    condition, coils **plus** plasma self-field. The axis / separatrix use the smooth O-/X-point
    finders, so ``F`` is a pure differentiable function of ``ψ`` (and ``θ = (I_coil, p', FF')``);
    ``F(ψ*, θ) = 0`` at the equilibrium and defines the implicit-diff adjoint.
    """
    shape = (Z_grid.shape[0], R_grid.shape[0])
    d_r = R_grid[1] - R_grid[0]
    d_z = Z_grid[1] - Z_grid[0]
    dA = d_r * d_z

    psi_coil = vacuum_field_si(R_grid, Z_grid, coil_R, coil_Z, coil_I, mu0)
    axis = smooth_axis_flux(psi)
    bndry = smooth_xpoint_flux(psi, R_grid, Z_grid)
    j_phi = _plasma_current(
        psi,
        R_grid,
        axis,
        bndry,
        psin_knots,
        pprime_vals,
        ffprime_vals,
        ip_target,
        dA,
        cutoff_width,
        mu0,
    )

    # Interior source is −μ₀ R Jφ with the SAME Ip-normalised current the solve uses (not the
    # raw general_gs_source), so F = 0 exactly at the coupled fixed point.
    source = -(mu0 * R_grid[jnp.newaxis, :] * j_phi)
    res = _laplacian_star(psi, R_grid, d_r, d_z) - source

    wall_flux = psi_coil.reshape(-1)[wall_idx] + response_matrix @ (
        j_phi.reshape(-1)[source_idx] * dA
    )
    res_flat = res.reshape(-1).at[wall_idx].set(psi.reshape(-1)[wall_idx] - wall_flux)
    return res_flat.reshape(shape)


# ── Anderson-accelerated forward solve ────────────────────────────


def _coupled_step(
    psi_flat: jnp.ndarray,
    ip_now: jnp.ndarray,
    coil_wall: jnp.ndarray,
    shape: tuple[int, int],
    R_grid: jnp.ndarray,
    Z_grid: jnp.ndarray,
    d_r: jnp.ndarray,
    d_z: jnp.ndarray,
    dA: jnp.ndarray,
    psin_knots: jnp.ndarray,
    pprime_vals: jnp.ndarray,
    ffprime_vals: jnp.ndarray,
    response_matrix: jnp.ndarray,
    wall_idx: jnp.ndarray,
    source_idx: jnp.ndarray,
    cutoff_width: float,
    mu0: float,
    precond: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """One coupled Picard step ``G(ψ)``: axis/X-point → Ip-normalised Jφ → coupled wall flux →
    linear GS solve. The equilibrium is its fixed point ``ψ = G(ψ)``. ``precond`` (optional) is
    a linear ``M ≈ A⁻¹`` handed to BiCGSTAB — it changes the Krylov convergence path only, not
    the solution."""
    rhs = _coupled_rhs(
        psi_flat,
        ip_now,
        coil_wall,
        shape,
        R_grid,
        Z_grid,
        dA,
        psin_knots,
        pprime_vals,
        ffprime_vals,
        response_matrix,
        wall_idx,
        source_idx,
        cutoff_width,
        mu0,
    )

    def operator(pf: jnp.ndarray) -> jnp.ndarray:
        return _gs_operator_flat(pf, shape, R_grid, d_r, d_z)

    sol, _info = jax.scipy.sparse.linalg.bicgstab(  # type: ignore[no-untyped-call]
        operator, rhs, x0=psi_flat, tol=_BICGSTAB_TOL, maxiter=_BICGSTAB_MAXITER, M=precond
    )
    return cast(jnp.ndarray, sol)


def _coupled_rhs(
    psi_flat: jnp.ndarray,
    ip_now: jnp.ndarray,
    coil_wall: jnp.ndarray,
    shape: tuple[int, int],
    R_grid: jnp.ndarray,
    Z_grid: jnp.ndarray,
    dA: jnp.ndarray,
    psin_knots: jnp.ndarray,
    pprime_vals: jnp.ndarray,
    ffprime_vals: jnp.ndarray,
    response_matrix: jnp.ndarray,
    wall_idx: jnp.ndarray,
    source_idx: jnp.ndarray,
    cutoff_width: float,
    mu0: float,
) -> jnp.ndarray:
    """Right-hand side of the linear GS system at the current iterate: interior source
    ``−μ₀ R Jφ`` (Ip-normalised, smooth LCFS cutoff) with the coupled von Hagenow wall flux
    on the identity wall rows. Shared by every inner-solver variant — the physics of the
    coupled step lives HERE, exactly once."""
    psi = psi_flat.reshape(shape)
    axis = smooth_axis_flux(psi)
    bndry = smooth_xpoint_flux(psi, R_grid, Z_grid)
    j_phi = _plasma_current(
        psi,
        R_grid,
        axis,
        bndry,
        psin_knots,
        pprime_vals,
        ffprime_vals,
        ip_now,
        dA,
        cutoff_width,
        mu0,
    )
    wall_flux = coil_wall + response_matrix @ (j_phi.reshape(-1)[source_idx] * dA)
    return (-(mu0 * R_grid[jnp.newaxis, :] * j_phi)).reshape(-1).at[wall_idx].set(wall_flux)


def solve_predictive_equilibrium(
    coil_I: jnp.ndarray,
    pprime_vals: jnp.ndarray,
    ffprime_vals: jnp.ndarray,
    R_grid: jnp.ndarray,
    Z_grid: jnp.ndarray,
    coil_R: jnp.ndarray,
    coil_Z: jnp.ndarray,
    psin_knots: jnp.ndarray,
    ip_target: float,
    response_matrix: jnp.ndarray,
    wall_idx: jnp.ndarray,
    source_idx: jnp.ndarray,
    psi_init: jnp.ndarray | None = None,
    n_iter: int = DEFAULT_N_ITER,
    anderson_depth: int = DEFAULT_ANDERSON_DEPTH,
    mixing: float = DEFAULT_MIXING,
    ip_ramp: int = DEFAULT_IP_RAMP,
    cutoff_width: float = DEFAULT_CUTOFF_WIDTH,
    tol: float = DEFAULT_TOL,
    mu0: float = MU0_SI,
    *,
    use_mg_preconditioner: bool = False,
) -> jnp.ndarray:
    """Solve the predictive free-boundary equilibrium from coils + profiles + Ip.

    Anderson-accelerated fixed-point iteration of the coupled step :func:`_coupled_step` with an
    ``Ip`` ramp for cold-start robustness. Returns the equilibrium ``ψ`` [Wb], shape
    ``(NZ, NR)``. Pass ``response_matrix, wall_idx, source_idx`` from :func:`build_response_matrix`
    (precomputed once per grid). ``psi_init`` defaults to the vacuum field (a genuine cold start).

    Iteration stops early once the fixed-point residual ``‖G(ψ)−ψ‖`` falls below ``tol`` relative
    to ``‖ψ‖`` — both to save work and to avoid the rank-deficient Anderson least-squares that a
    machine-converged history would produce; any non-finite Anderson step falls back to a damped
    Picard update so the solve never returns ``NaN``.

    Parameters
    ----------
    ip_target : total toroidal plasma current ``∮Jφ dA`` [A] to hold.
    n_iter : maximum outer Anderson iterations (may stop earlier at ``tol``).
    anderson_depth : Anderson history depth ``m`` (validated ``≈ 8``; too small may not converge).
    mixing : Anderson mixing ``β`` (validated ``≈ 0.5``).
    ip_ramp : ramp ``Ip`` linearly to ``ip_target`` over the first ``ip_ramp`` iterations.
    cutoff_width : ``ψ_N`` roll-off width of the smooth LCFS current cutoff.
    tol : relative fixed-point residual at which to stop early.
    use_mg_preconditioner : precondition each inner BiCGSTAB with one geometric-multigrid
        V-cycle (:func:`~scpn_fusion.core.jax_multigrid_precond.build_gs_mg_preconditioner`).
        Identical fixed point (the preconditioner only reshapes the Krylov convergence path);
        the forward-speed lane pending its dedicated-hardware benchmark, hence opt-in.
    """
    shape = (Z_grid.shape[0], R_grid.shape[0])
    d_r = R_grid[1] - R_grid[0]
    d_z = Z_grid[1] - Z_grid[0]
    dA = d_r * d_z
    psi_coil = vacuum_field_si(R_grid, Z_grid, coil_R, coil_Z, coil_I, mu0)
    coil_wall = psi_coil.reshape(-1)[wall_idx]
    precond: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    if use_mg_preconditioner:
        precond = build_gs_mg_preconditioner(shape, R_grid, float(d_r), float(d_z))

    def step(psi_flat: jnp.ndarray, ip_now: jnp.ndarray) -> jnp.ndarray:
        return _coupled_step(
            psi_flat,
            ip_now,
            coil_wall,
            shape,
            R_grid,
            Z_grid,
            d_r,
            d_z,
            dA,
            psin_knots,
            pprime_vals,
            ffprime_vals,
            response_matrix,
            wall_idx,
            source_idx,
            cutoff_width,
            mu0,
            precond,
        )

    x = (psi_coil if psi_init is None else psi_init).reshape(-1)
    f_hist: list[jnp.ndarray] = []
    x_hist: list[jnp.ndarray] = []
    for k in range(n_iter):
        ip_k = ip_target * min(1.0, (k + 1.0) / max(ip_ramp, 1))
        g_x = step(x, jnp.asarray(ip_k))
        f = g_x - x
        # Converged (and Ip fully ramped): stop before the Anderson history goes rank-deficient.
        if k >= ip_ramp and float(jnp.linalg.norm(f)) <= tol * (float(jnp.linalg.norm(x)) + 1.0):
            break
        f_hist.append(f)
        x_hist.append(x)
        if len(f_hist) > anderson_depth:
            f_hist.pop(0)
            x_hist.pop(0)
        m = len(f_hist)
        if m == 1:
            x = x + mixing * f
        else:
            df = jnp.stack([f_hist[i + 1] - f_hist[i] for i in range(m - 1)], axis=1)
            dx = jnp.stack([x_hist[i + 1] - x_hist[i] for i in range(m - 1)], axis=1)
            gamma, _res, _rank, _sv = jnp.linalg.lstsq(df, f, rcond=None)
            x_next = x + mixing * f - (dx + mixing * df) @ gamma
            # A rank-deficient history can produce a non-finite step near convergence; fall back
            # to a damped Picard update so the solve is always finite.
            x = jnp.where(jnp.all(jnp.isfinite(x_next)), x_next, x + mixing * f)
    return cast(jnp.ndarray, x.reshape(shape))


# ── Implicit-differentiation adjoint (∂ψ*/∂θ for the IDA loop) ─────


@partial(jax.custom_vjp, nondiff_argnums=tuple(range(3, 20)))
def solve_predictive_equilibrium_diff(
    coil_I: jnp.ndarray,
    pprime_vals: jnp.ndarray,
    ffprime_vals: jnp.ndarray,
    R_grid: jnp.ndarray,
    Z_grid: jnp.ndarray,
    coil_R: jnp.ndarray,
    coil_Z: jnp.ndarray,
    psin_knots: jnp.ndarray,
    ip_target: float,
    response_matrix: jnp.ndarray,
    wall_idx: jnp.ndarray,
    source_idx: jnp.ndarray,
    psi_init: jnp.ndarray | None = None,
    n_iter: int = DEFAULT_N_ITER,
    anderson_depth: int = DEFAULT_ANDERSON_DEPTH,
    mixing: float = DEFAULT_MIXING,
    ip_ramp: int = DEFAULT_IP_RAMP,
    cutoff_width: float = DEFAULT_CUTOFF_WIDTH,
    tol: float = DEFAULT_TOL,
    mu0: float = MU0_SI,
) -> jnp.ndarray:
    """Differentiable predictive free-boundary solve — ``ψ*`` with an exact implicit-diff adjoint.

    Identical forward to :func:`solve_predictive_equilibrium`, but ``jax.grad`` w.r.t. the
    differentiated inputs ``(coil_I, pprime_vals, ffprime_vals)`` uses the implicit function
    theorem on the converged fixed point ``F(ψ*, θ) = 0`` (:func:`predictive_gs_residual`): one
    adjoint solve ``(∂F/∂ψ)ᵀ λ = ψ̄`` then ``θ̄ = −(∂F/∂θ)ᵀ λ``. The gradient cost is independent
    of the Anderson iteration count — the property the DIII-D IDA MAP/MCMC loop needs — and the
    adjoint is exact only insofar as the forward has converged (``F(ψ*) ≈ 0``) and the axis /
    separatrix finders are smooth (they are). ``ip_target`` and all solver settings are treated
    as constants (non-differentiated).
    """
    return solve_predictive_equilibrium(
        coil_I,
        pprime_vals,
        ffprime_vals,
        R_grid,
        Z_grid,
        coil_R,
        coil_Z,
        psin_knots,
        ip_target,
        response_matrix,
        wall_idx,
        source_idx,
        psi_init,
        n_iter,
        anderson_depth,
        mixing,
        ip_ramp,
        cutoff_width,
        tol,
        mu0,
    )


def _solve_diff_fwd(
    coil_I: jnp.ndarray,
    pprime_vals: jnp.ndarray,
    ffprime_vals: jnp.ndarray,
    R_grid: jnp.ndarray,
    Z_grid: jnp.ndarray,
    coil_R: jnp.ndarray,
    coil_Z: jnp.ndarray,
    psin_knots: jnp.ndarray,
    ip_target: float,
    response_matrix: jnp.ndarray,
    wall_idx: jnp.ndarray,
    source_idx: jnp.ndarray,
    psi_init: jnp.ndarray | None,
    n_iter: int,
    anderson_depth: int,
    mixing: float,
    ip_ramp: int,
    cutoff_width: float,
    tol: float,
    mu0: float,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
    psi = solve_predictive_equilibrium(
        coil_I,
        pprime_vals,
        ffprime_vals,
        R_grid,
        Z_grid,
        coil_R,
        coil_Z,
        psin_knots,
        ip_target,
        response_matrix,
        wall_idx,
        source_idx,
        psi_init,
        n_iter,
        anderson_depth,
        mixing,
        ip_ramp,
        cutoff_width,
        tol,
        mu0,
    )
    return psi, (psi, coil_I, pprime_vals, ffprime_vals)


def _solve_diff_bwd(
    R_grid: jnp.ndarray,
    Z_grid: jnp.ndarray,
    coil_R: jnp.ndarray,
    coil_Z: jnp.ndarray,
    psin_knots: jnp.ndarray,
    ip_target: float,
    response_matrix: jnp.ndarray,
    wall_idx: jnp.ndarray,
    source_idx: jnp.ndarray,
    psi_init: jnp.ndarray | None,
    n_iter: int,
    anderson_depth: int,
    mixing: float,
    ip_ramp: int,
    cutoff_width: float,
    tol: float,
    mu0: float,
    residuals: tuple[jnp.ndarray, ...],
    psi_bar: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Implicit-diff VJP: adjoint solve ``(∂F/∂ψ)ᵀ λ = ψ̄`` then ``θ̄ = −(∂F/∂θ)ᵀ λ``."""
    psi, coil_I, pprime_vals, ffprime_vals = residuals
    shape = psi.shape

    def residual_in_psi(p: jnp.ndarray) -> jnp.ndarray:
        return predictive_gs_residual(
            p,
            coil_I,
            pprime_vals,
            ffprime_vals,
            R_grid,
            Z_grid,
            coil_R,
            coil_Z,
            psin_knots,
            jnp.asarray(ip_target),
            response_matrix,
            wall_idx,
            source_idx,
            cutoff_width,
            mu0,
        )

    _, vjp_psi = jax.vjp(residual_in_psi, psi)

    def adjoint_operator(lam_flat: jnp.ndarray) -> jnp.ndarray:
        return cast(jnp.ndarray, vjp_psi(lam_flat.reshape(shape))[0].reshape(-1))

    # Diagonal (Jacobi) preconditioner: F's interior rows scale as the Δ* centre coefficient
    # (~1/h²) and its wall rows as 1 — a badly-scaled system whose conditioning worsens ~1/h² at
    # finer grids (uncorrected, BiCGSTAB needs thousands of iterations and silently caps out).
    # Normalising by that known diagonal makes the iteration count grid-insensitive.
    d_r = R_grid[1] - R_grid[0]
    d_z = Z_grid[1] - Z_grid[0]
    precond_diag = jnp.full(psi.size, 2.0 / d_r**2 + 2.0 / d_z**2).at[wall_idx].set(1.0)

    def preconditioner(x_flat: jnp.ndarray) -> jnp.ndarray:
        return x_flat / precond_diag

    lam_flat, _info = bicgstab(  # type: ignore[no-untyped-call]
        adjoint_operator,
        psi_bar.reshape(-1),
        tol=_BICGSTAB_TOL,
        maxiter=_ADJOINT_MAXITER,
        M=preconditioner,
    )
    lam = lam_flat.reshape(shape)

    def residual_in_theta(ci: jnp.ndarray, pp: jnp.ndarray, ff: jnp.ndarray) -> jnp.ndarray:
        return predictive_gs_residual(
            psi,
            ci,
            pp,
            ff,
            R_grid,
            Z_grid,
            coil_R,
            coil_Z,
            psin_knots,
            jnp.asarray(ip_target),
            response_matrix,
            wall_idx,
            source_idx,
            cutoff_width,
            mu0,
        )

    _, vjp_theta = jax.vjp(residual_in_theta, coil_I, pprime_vals, ffprime_vals)
    g_ci, g_pp, g_ff = vjp_theta(lam)
    return (-g_ci, -g_pp, -g_ff)


solve_predictive_equilibrium_diff.defvjp(_solve_diff_fwd, _solve_diff_bwd)
