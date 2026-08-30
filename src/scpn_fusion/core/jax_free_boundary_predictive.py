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
4. **Two-axis continuation.** A cold start first ramps ``Ip`` while using the stable soft
   separatrix estimate, then homotopies to the sub-cell saddle value. Warm starts use the
   physical saddle value immediately. This follows the desired diverted root rather than the
   competing vacuum-like fixed point.
5. **Continuation-aware Anderson fixed-point solve.** Naive Picard on this full nonlinear
   boundary+profile map is *unstable*: the physical fixed point is a saddle and simple
   iteration is driven to a spurious high-peaking attractor (a known reason production
   free-boundary codes use Newton/Anderson, not Picard). Anderson mixing (depth ``m``) with an
   ``Ip`` ramp converges to the true fixed point from a cold vacuum start. Moving-map history
   stays active for stability, then resets when Ip or refinement reaches its fixed endpoint.

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

from dataclasses import dataclass
from functools import partial
from typing import Callable, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import bicgstab, gmres
from numpy.typing import NDArray
from scipy.optimize import NoConvergence, newton_krylov

from scpn_fusion.core.jax_free_boundary_gs import (
    MU0_SI,
    greens_psi_si,
    normalised_flux,
    normalised_flux_unclipped,
    vacuum_field_si,
)
from scpn_fusion.core.jax_plasma_support import soft_axis_connected_support
from scpn_fusion.core.jax_continuation_history import (
    continuation_history_requires_reset,
)
from scpn_fusion.core.jax_multigrid_precond import build_gs_mg_preconditioner
from scpn_fusion.core.jax_o_point import smooth_axis_coordinates, smooth_axis_flux
from scpn_fusion.core.jax_x_point import smooth_xpoint_flux

_bicgstab = cast(
    Callable[..., tuple[jnp.ndarray, jnp.ndarray]],
    bicgstab,
)
_gmres = cast(
    Callable[..., tuple[jnp.ndarray, jnp.ndarray]],
    gmres,
)

# Anderson / continuation defaults used by the fail-closed public same-case benchmark.
DEFAULT_N_ITER = 180
DEFAULT_ANDERSON_DEPTH = 8
DEFAULT_MIXING = 0.5
DEFAULT_IP_RAMP = 30
DEFAULT_SEPARATRIX_START = 100
DEFAULT_SEPARATRIX_RAMP = 20
DEFAULT_CUTOFF_WIDTH = 0.03
DEFAULT_TOL = 1.0e-9
_BICGSTAB_TOL = 1.0e-11
_BICGSTAB_MAXITER = 700
# The adjoint mixes interior rows (Δ*, ~1/h²) and identity wall rows.  The backward solve uses
# right-preconditioned GMRES on `A.T @ D^-1 y = psi_bar`, so its stopping criterion measures the
# physical residual; `lambda = D^-1 y` then recovers the original adjoint variable.  A left
# preconditioner is forbidden here because it can report convergence with an O(1) physical
# residual and silently bias coil gradients.
_ADJOINT_MAXITER = 500
_ADJOINT_RESTART = 200
_ADJOINT_TOL = 1.0e-9
_ADJOINT_RESIDUAL_FACTOR = 10.0


@dataclass(frozen=True)
class PredictiveIterationSnapshot:
    """One accepted iteration of the eager predictive-equilibrium solve.

    The arrays use the physical ``(NZ, NR)`` grid shape. ``psi`` is the state
    presented to the coupled map, ``mapped_psi`` is ``G(psi)``, and ``next_psi``
    is the state accepted by the continuation/Anderson loop. On a converged
    iteration, ``next_psi`` equals ``psi`` because the solver stops before
    extending its rank-deficient Anderson history.

    Attributes
    ----------
    iteration_index
        Zero-based outer-iteration index.
    ip_now
        Plasma current used by the coupled map [A].
    separatrix_refinement
        Homotopy fraction from the stable soft separatrix to the sub-cell
        saddle value.
    psi
        State presented to the coupled map [Wb].
    mapped_psi
        Coupled-map result ``G(psi)`` [Wb].
    fixed_point_residual
        Difference ``G(psi) - psi`` [Wb].
    next_psi
        State accepted for the next iteration [Wb].
    converged
        Whether this iteration satisfied the fail-closed stopping criterion.
    """

    iteration_index: int
    ip_now: float
    separatrix_refinement: float
    psi: jnp.ndarray
    mapped_psi: jnp.ndarray
    fixed_point_residual: jnp.ndarray
    next_psi: jnp.ndarray
    converged: bool


PredictiveIterationObserver = Callable[[PredictiveIterationSnapshot], None]


@dataclass(frozen=True)
class PredictiveNewtonResult:
    """Causal two-stage Newton solution and fail-closed convergence evidence."""

    equilibrium: jnp.ndarray
    fixed_support_max_scaled_residual: float
    dynamic_support_max_scaled_residual: float
    relative_nonlinear_residual_rms: float
    fixed_support_converged: bool
    dynamic_support_converged: bool
    finite: bool
    seed_source: str
    newton_linearization: str
    fixed_support_iterations: int
    dynamic_support_iterations: int


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
    fixed_support_weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Calculate the smooth, Ip-normalised toroidal current density.

    The density ``Jφ = R p' + FF'/(μ₀R)`` is scaled so
    ``∮Jφ dA = ip_target``. Support is the **axis-connected** smooth LCFS
    weight (:func:`~scpn_fusion.core.jax_plasma_support.soft_axis_connected_support`):
    a pure ``ψ_N < 1`` / pure ``tanh(ψ_N − 1)`` cut incorrectly admits
    private-flux islands on diverted topologies. Ip scaling pins the total
    current and kills the self-field runaway.
    """
    psi_n_raw = normalised_flux_unclipped(psi, psi_axis, psi_bndry)
    psi_n = normalised_flux(psi, psi_axis, psi_bndry)
    pprime = jnp.interp(psi_n, psin_knots, pprime_vals)
    ffprime = jnp.interp(psi_n, psin_knots, ffprime_vals)
    r_safe = jnp.maximum(R_grid[jnp.newaxis, :], 1e-6)
    j_raw = r_safe * pprime + ffprime / (mu0 * r_safe)
    support = (
        soft_axis_connected_support(psi, psi_n_raw, cutoff_width)
        if fixed_support_weights is None
        else fixed_support_weights
    )
    j_masked = j_raw * support
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
    *,
    fixed_psi_axis: jnp.ndarray | None = None,
    fixed_psi_boundary: jnp.ndarray | None = None,
    fixed_support_weights: jnp.ndarray | None = None,
    decompose_coil_field: bool = False,
) -> jnp.ndarray:
    """Coupled predictive GS residual ``F(ψ)`` whose root is the free-boundary equilibrium.

    The default retains the historical total-field formulation. With
    ``decompose_coil_field=True``, the interior is ``Δ*(ψ − ψ_coil) − S(ψ)`` and wall rows are
    ``(ψ − ψ_coil) − M @ (Jφ·dA)``. This exact superposition is required when coil filaments lie
    inside the rectangular grid, where their vacuum field is not harmonic throughout the domain.
    Fixed anchors/support form an explicit external-reconstruction control; omitted values use
    the differentiable smooth O-/X-point and axis-connected-support path.
    """
    shape = (Z_grid.shape[0], R_grid.shape[0])
    d_r = R_grid[1] - R_grid[0]
    d_z = Z_grid[1] - Z_grid[0]
    dA = d_r * d_z

    if (fixed_psi_axis is None) != (fixed_psi_boundary is None):
        raise ValueError("fixed_psi_axis and fixed_psi_boundary must be supplied together")
    if fixed_support_weights is not None and fixed_support_weights.shape != psi.shape:
        raise ValueError("fixed_support_weights must match the equilibrium grid")

    psi_coil = vacuum_field_si(R_grid, Z_grid, coil_R, coil_Z, coil_I, mu0)
    axis = smooth_axis_flux(psi) if fixed_psi_axis is None else fixed_psi_axis
    bndry = (
        smooth_xpoint_flux(psi, R_grid, Z_grid)
        if fixed_psi_boundary is None
        else fixed_psi_boundary
    )
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
        fixed_support_weights,
    )

    # Interior source is −μ₀ R Jφ with the SAME Ip-normalised current the solve uses (not the
    # raw general_gs_source), so F = 0 exactly at the coupled fixed point.
    source = -(mu0 * R_grid[jnp.newaxis, :] * j_phi)
    solved_field = psi - psi_coil if decompose_coil_field else psi
    res = _laplacian_star(solved_field, R_grid, d_r, d_z) - source

    plasma_wall_flux = response_matrix @ (j_phi.reshape(-1)[source_idx] * dA)
    wall_target = (
        plasma_wall_flux
        if decompose_coil_field
        else psi_coil.reshape(-1)[wall_idx] + plasma_wall_flux
    )
    res_flat = res.reshape(-1).at[wall_idx].set(solved_field.reshape(-1)[wall_idx] - wall_target)
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
    separatrix_refinement: jnp.ndarray,
    cutoff_width: float,
    mu0: float,
    precond: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    fixed_psi_axis: jnp.ndarray | None = None,
    fixed_psi_boundary: jnp.ndarray | None = None,
    fixed_support_weights: jnp.ndarray | None = None,
    psi_coil: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Advance one coupled Picard step ``G(ψ)``.

    The path is axis/X-point → Ip-normalised Jφ → coupled wall flux → linear
    GS solve. The equilibrium is its fixed point ``ψ = G(ψ)``. ``precond``
    (optional) is a linear ``M ≈ A⁻¹`` handed to BiCGSTAB — it changes the
    Krylov convergence path only, not the solution.
    """
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
        separatrix_refinement,
        cutoff_width,
        mu0,
        fixed_psi_axis,
        fixed_psi_boundary,
        fixed_support_weights,
        psi_coil is not None,
    )

    def operator(pf: jnp.ndarray) -> jnp.ndarray:
        return _gs_operator_flat(pf, shape, R_grid, d_r, d_z)

    linear_init = psi_flat if psi_coil is None else psi_flat - psi_coil.reshape(-1)
    sol, _info = _bicgstab(
        operator,
        rhs,
        x0=linear_init,
        tol=_BICGSTAB_TOL,
        maxiter=_BICGSTAB_MAXITER,
        M=precond,
    )
    return sol if psi_coil is None else sol + psi_coil.reshape(-1)


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
    separatrix_refinement: jnp.ndarray,
    cutoff_width: float,
    mu0: float,
    fixed_psi_axis: jnp.ndarray | None = None,
    fixed_psi_boundary: jnp.ndarray | None = None,
    fixed_support_weights: jnp.ndarray | None = None,
    decompose_coil_field: bool = False,
) -> jnp.ndarray:
    """Construct the linear Grad-Shafranov right-hand side.

    The current iterate supplies the interior source ``−μ₀ R Jφ``
    (Ip-normalised, smooth LCFS cutoff) and the coupled von Hagenow wall flux
    on the identity wall rows. Every inner-solver variant shares this function,
    so the coupled-step physics lives here exactly once.
    """
    psi = psi_flat.reshape(shape)
    axis = smooth_axis_flux(psi) if fixed_psi_axis is None else fixed_psi_axis
    bndry = (
        smooth_xpoint_flux(
            psi,
            R_grid,
            Z_grid,
            refinement=separatrix_refinement,
        )
        if fixed_psi_boundary is None
        else fixed_psi_boundary
    )
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
        fixed_support_weights,
    )
    plasma_wall_flux = response_matrix @ (j_phi.reshape(-1)[source_idx] * dA)
    wall_flux = plasma_wall_flux if decompose_coil_field else coil_wall + plasma_wall_flux
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
    iteration_observer: PredictiveIterationObserver | None = None,
    fixed_psi_axis: jnp.ndarray | None = None,
    fixed_psi_boundary: jnp.ndarray | None = None,
    fixed_support_weights: jnp.ndarray | None = None,
    decompose_coil_field: bool = False,
) -> jnp.ndarray:
    """Solve the predictive free-boundary equilibrium from coils + profiles + Ip.

    Anderson-accelerated fixed-point iteration of the coupled step :func:`_coupled_step` with
    ``Ip`` and separatrix-refinement continuation for cold-start robustness. Moving-map history
    resets when each continuation parameter reaches its fixed endpoint. Returns
    the equilibrium ``ψ`` [Wb], shape ``(NZ, NR)``. Pass ``response_matrix, wall_idx, source_idx``
    from :func:`build_response_matrix` (precomputed once per grid). ``psi_init`` defaults to the
    vacuum field (a genuine cold start); an explicit warm start uses the refined separatrix
    immediately.

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
    iteration_observer : optional synchronous callback receiving the immutable accepted state
        of every eager continuation/Anderson iteration. This diagnostic surface is deliberately
        absent from :func:`solve_predictive_equilibrium_diff` and the compiled forward solver;
        omitting it preserves the normal solve path.
    fixed_psi_axis, fixed_psi_boundary, fixed_support_weights : optional external-reconstruction
        topology contract. Both anchors must be supplied together and the support must match the
        grid. Omit all three for the production smooth-topology path.
    decompose_coil_field : solve the plasma self-field and add the known coil field everywhere.
        Required when any filament lies inside the rectangular computational domain; the default
        preserves the historical wall-harmonic contract for backward compatibility.
    """
    shape = (Z_grid.shape[0], R_grid.shape[0])
    d_r = R_grid[1] - R_grid[0]
    d_z = Z_grid[1] - Z_grid[0]
    dA = d_r * d_z
    if (fixed_psi_axis is None) != (fixed_psi_boundary is None):
        raise ValueError("fixed_psi_axis and fixed_psi_boundary must be supplied together")
    if fixed_support_weights is not None and fixed_support_weights.shape != shape:
        raise ValueError("fixed_support_weights must match the equilibrium grid")

    psi_coil = vacuum_field_si(R_grid, Z_grid, coil_R, coil_Z, coil_I, mu0)
    coil_wall = psi_coil.reshape(-1)[wall_idx]
    precond: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    if use_mg_preconditioner:
        precond = build_gs_mg_preconditioner(shape, R_grid, float(d_r), float(d_z))

    def step(
        psi_flat: jnp.ndarray,
        ip_now: jnp.ndarray,
        separatrix_refinement: jnp.ndarray,
    ) -> jnp.ndarray:
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
            separatrix_refinement,
            cutoff_width,
            mu0,
            precond,
            fixed_psi_axis,
            fixed_psi_boundary,
            fixed_support_weights,
            psi_coil if decompose_coil_field else None,
        )

    x = (psi_coil if psi_init is None else psi_init).reshape(-1)
    f_hist: list[jnp.ndarray] = []
    x_hist: list[jnp.ndarray] = []
    use_separatrix_continuation = psi_init is None
    for k in range(n_iter):
        ip_k = ip_target * min(1.0, (k + 1.0) / max(ip_ramp, 1))
        refinement = (
            min(
                1.0,
                max(
                    0.0,
                    (k + 1.0 - DEFAULT_SEPARATRIX_START) / DEFAULT_SEPARATRIX_RAMP,
                ),
            )
            if use_separatrix_continuation
            else 1.0
        )
        g_x = step(x, jnp.asarray(ip_k), jnp.asarray(refinement))
        f = g_x - x
        # Converged (and Ip fully ramped): stop before the Anderson history goes rank-deficient.
        continuation_complete = (
            not use_separatrix_continuation
            or k + 1 >= DEFAULT_SEPARATRIX_START + DEFAULT_SEPARATRIX_RAMP
        )
        converged = (
            k >= ip_ramp
            and continuation_complete
            and float(jnp.linalg.norm(f)) <= tol * (float(jnp.linalg.norm(x)) + 1.0)
        )
        if converged:
            next_x = x
        else:
            if bool(
                continuation_history_requires_reset(
                    k,
                    ip_ramp=ip_ramp,
                    use_separatrix_continuation=use_separatrix_continuation,
                    separatrix_start=DEFAULT_SEPARATRIX_START,
                    separatrix_ramp=DEFAULT_SEPARATRIX_RAMP,
                )
            ):
                f_hist.clear()
                x_hist.clear()
            f_hist.append(f)
            x_hist.append(x)
            if len(f_hist) > anderson_depth:
                f_hist.pop(0)
                x_hist.pop(0)
            m = len(f_hist)
            if m == 1:
                next_x = x + mixing * f
            else:
                df = jnp.stack([f_hist[i + 1] - f_hist[i] for i in range(m - 1)], axis=1)
                dx = jnp.stack([x_hist[i + 1] - x_hist[i] for i in range(m - 1)], axis=1)
                gamma, _res, _rank, _sv = jnp.linalg.lstsq(df, f, rcond=None)
                x_next = x + mixing * f - (dx + mixing * df) @ gamma
                # A rank-deficient history can produce a non-finite step near convergence; fall
                # back to a damped Picard update so the solve is always finite.
                next_x = jnp.where(jnp.all(jnp.isfinite(x_next)), x_next, x + mixing * f)
        if iteration_observer is not None:
            iteration_observer(
                PredictiveIterationSnapshot(
                    iteration_index=k,
                    ip_now=ip_k,
                    separatrix_refinement=refinement,
                    psi=x.reshape(shape),
                    mapped_psi=g_x.reshape(shape),
                    fixed_point_residual=f.reshape(shape),
                    next_psi=next_x.reshape(shape),
                    converged=converged,
                )
            )
        x = next_x
        if converged:
            break
    return cast(jnp.ndarray, x.reshape(shape))


def solve_predictive_equilibrium_newton(
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
    *,
    psi_init: jnp.ndarray | None = None,
    warm_start_linearization: str = "exact_jvp_gmres",
    seed_picard_iterations: int = 80,
    fixed_support_newton_iterations: int = 100,
    dynamic_support_newton_iterations: int = 100,
    newton_f_tol: float = 1.0e-8,
    cutoff_width: float = DEFAULT_CUTOFF_WIDTH,
    mu0: float = MU0_SI,
) -> PredictiveNewtonResult:
    """Solve the predictive equilibrium through a causal two-stage Newton path.

    With no explicit warm start, a coil-boundary implicit solve and the
    established total-field predictive solve supply two causal seed fields. A
    geometry-neutral selector rejects axis assignments outside the central
    chamber, then chooses the lower corrected-residual seed. ``psi_init`` may
    instead carry a prior native equilibrium for continuation or local finite
    differences. Support is frozen for the first Newton-Krylov stage and
    released for a second dynamic-support stage. The established total-field
    seed uses exact JAX linearisations; the coil-boundary seed uses SciPy's
    finite-difference Krylov path. For local continuation, pass the base
    result's ``newton_linearization`` back as ``warm_start_linearization`` so
    perturbations retain its numerical branch. No reference field, mask,
    machine name, or geometry-specific threshold enters.

    The returned status remains fail closed: a stalled line search retains its
    best finite field but leaves the corresponding convergence boolean false.
    """
    from scpn_fusion.core.jax_free_boundary_gs_implicit import (
        solve_free_boundary_gs_implicit,
    )

    if (
        min(
            seed_picard_iterations,
            fixed_support_newton_iterations,
            dynamic_support_newton_iterations,
        )
        < 1
    ):
        raise ValueError("all predictive Newton iteration counts must be >= 1")
    if not np.isfinite(newton_f_tol) or newton_f_tol <= 0.0:
        raise ValueError("newton_f_tol must be finite and > 0")
    if warm_start_linearization not in ("exact_jvp_gmres", "finite_difference_krylov"):
        raise ValueError(
            "warm_start_linearization must be 'exact_jvp_gmres' or 'finite_difference_krylov'"
        )
    shape = (int(Z_grid.size), int(R_grid.size))
    if psi_init is not None and psi_init.shape != shape:
        raise ValueError(f"psi_init shape {psi_init.shape} != grid shape {shape}")
    try:
        jax_version = tuple(int(part) for part in jax.__version__.split("+", 1)[0].split(".")[:2])
    except ValueError as exc:
        raise RuntimeError(f"unsupported JAX version string {jax.__version__!r}") from exc
    if jax_version < (0, 7):
        raise RuntimeError(
            "predictive Newton requires JAX >= 0.7; older solver trajectories are not "
            "admitted by the multi-geometry evidence contract"
        )

    seed = (
        solve_free_boundary_gs_implicit(
            coil_I,
            pprime_vals,
            ffprime_vals,
            R_grid,
            Z_grid,
            coil_R,
            coil_Z,
            psin_knots,
            n_picard=seed_picard_iterations,
            mu0=mu0,
            use_smooth_axis=True,
        )
        if psi_init is None
        else jnp.asarray(psi_init)
    )
    d_r = R_grid[1] - R_grid[0]
    d_z = Z_grid[1] - Z_grid[0]
    row_scale = (
        jnp.full(seed.size, 2.0 / d_r**2 + 2.0 / d_z**2).at[wall_idx].set(1.0).reshape(shape)
    )

    @jax.jit
    def dynamic_scaled_residual(psi: jnp.ndarray) -> jnp.ndarray:
        return (
            predictive_gs_residual(
                psi,
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
                decompose_coil_field=True,
            )
            / row_scale
        )

    # Seed admission is intentionally stricter than the O-point search window:
    # a cold predictive field can assign a vacuum/divertor extremum just inside
    # the middle-half boundary. Requiring the central vertical third rejects
    # that off-midplane branch while retaining vertically shifted plasma axes.
    z_margin = max(3, shape[0] // 3)
    r_margin = max(3, shape[1] // 10)

    def seed_score(candidate: jnp.ndarray) -> float:
        axis_r, axis_z = smooth_axis_coordinates(candidate, R_grid, Z_grid)
        axis_in_chamber = bool(
            (axis_z >= Z_grid[z_margin])
            & (axis_z <= Z_grid[-1 - z_margin])
            & (axis_r >= R_grid[r_margin])
            & (axis_r <= R_grid[-1 - r_margin])
        )
        if not (bool(jnp.all(jnp.isfinite(candidate))) and axis_in_chamber):
            return float("inf")
        return float(jnp.linalg.norm(dynamic_scaled_residual(candidate)))

    if psi_init is None:
        from scpn_fusion.core.jax_predictive_forward_compiled import (
            solve_predictive_equilibrium_compiled,
        )

        total_field_seed = cast(
            jnp.ndarray,
            solve_predictive_equilibrium_compiled(
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
            ),
        )
        fixed_boundary_score = seed_score(seed)
        total_field_score = seed_score(total_field_seed)
        if not np.isfinite(fixed_boundary_score) and not np.isfinite(total_field_score):
            raise RuntimeError(
                "predictive Newton seed selection found no finite central-axis field"
            )
        use_finite_difference = fixed_boundary_score <= total_field_score
        seed = seed if use_finite_difference else total_field_seed
        seed_source = (
            "fixed_boundary_implicit" if use_finite_difference else "total_field_predictive"
        )
    else:
        if not np.isfinite(seed_score(seed)):
            raise ValueError("psi_init must be finite and place the magnetic axis in the chamber")
        use_finite_difference = warm_start_linearization == "finite_difference_krylov"
        seed_source = "explicit_warm_start"
    newton_linearization = (
        "finite_difference_krylov" if use_finite_difference else "exact_jvp_gmres"
    )
    seed_axis = smooth_axis_flux(seed)
    seed_boundary = smooth_xpoint_flux(seed, R_grid, Z_grid)
    seed_support = soft_axis_connected_support(
        seed,
        normalised_flux_unclipped(seed, seed_axis, seed_boundary),
        cutoff_width,
    )

    def solve_stage(
        initial: jnp.ndarray,
        fixed_support_weights: jnp.ndarray | None,
        max_iterations: int,
    ) -> tuple[jnp.ndarray, bool, float, int]:
        @jax.jit
        def scaled_residual(psi: jnp.ndarray) -> jnp.ndarray:
            return (
                predictive_gs_residual(
                    psi,
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
                    fixed_support_weights=fixed_support_weights,
                    decompose_coil_field=True,
                )
                / row_scale
            )

        if use_finite_difference:
            iteration_count = 0

            def host_residual(flat_psi: NDArray[np.float64]) -> NDArray[np.float64]:
                value = scaled_residual(jnp.asarray(flat_psi.reshape(shape)))
                return np.asarray(value, dtype=np.float64).reshape(-1)

            def count_iteration(
                _state: NDArray[np.float64], _residual: NDArray[np.float64]
            ) -> None:
                nonlocal iteration_count
                iteration_count += 1

            converged = True
            try:
                solved = newton_krylov(
                    host_residual,
                    np.asarray(initial, dtype=np.float64).reshape(-1),
                    f_tol=newton_f_tol,
                    maxiter=max_iterations,
                    verbose=False,
                    callback=count_iteration,
                )
            except NoConvergence as exc:
                converged = False
                solved = np.asarray(exc.args[0], dtype=np.float64)
            equilibrium = jnp.asarray(np.asarray(solved, dtype=np.float64).reshape(shape))
            maximum = float(jnp.max(jnp.abs(scaled_residual(equilibrium))))
            converged = bool(np.isfinite(maximum) and maximum <= newton_f_tol)
            return equilibrium, converged, maximum, iteration_count

        equilibrium = initial
        converged = False
        line_search_factors = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
        iteration_count = 0
        for _iteration in range(max_iterations):
            iteration_count = _iteration + 1
            residual, linearised = jax.linearize(scaled_residual, equilibrium)
            residual_norm = float(jnp.linalg.norm(residual))
            if residual_norm <= newton_f_tol:
                converged = True
                break

            def operator(vector: jnp.ndarray) -> jnp.ndarray:
                return cast(jnp.ndarray, linearised(vector.reshape(shape)).reshape(-1))

            direction, _info = _gmres(
                operator,
                -residual.reshape(-1),
                tol=1.0e-8,
                atol=0.0,
                restart=60,
                maxiter=60,
                solve_method="incremental",
            )
            direction = direction.reshape(shape)
            best = equilibrium
            best_norm = residual_norm
            for factor in line_search_factors:
                candidate = equilibrium + factor * direction
                candidate_norm = float(jnp.linalg.norm(scaled_residual(candidate)))
                if np.isfinite(candidate_norm) and candidate_norm < best_norm:
                    best = candidate
                    best_norm = candidate_norm
            if best_norm >= residual_norm:
                break
            equilibrium = best
        maximum = float(jnp.max(jnp.abs(scaled_residual(equilibrium))))
        converged = bool(np.isfinite(maximum) and maximum <= newton_f_tol)
        return equilibrium, converged, maximum, iteration_count

    fixed_equilibrium, fixed_converged, fixed_maximum, fixed_iterations = solve_stage(
        seed,
        seed_support,
        fixed_support_newton_iterations,
    )
    equilibrium, dynamic_converged, dynamic_maximum, dynamic_iterations = solve_stage(
        fixed_equilibrium,
        None,
        dynamic_support_newton_iterations,
    )
    axis = smooth_axis_flux(equilibrium)
    boundary = smooth_xpoint_flux(equilibrium, R_grid, Z_grid)
    d_area = d_r * d_z
    current = _plasma_current(
        equilibrium,
        R_grid,
        axis,
        boundary,
        psin_knots,
        pprime_vals,
        ffprime_vals,
        jnp.asarray(ip_target),
        d_area,
        cutoff_width,
        mu0,
    )
    raw_residual = predictive_gs_residual(
        equilibrium,
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
        decompose_coil_field=True,
    )
    source = -(mu0 * R_grid[jnp.newaxis, :] * current)
    interior_residual = raw_residual[1:-1, 1:-1]
    interior_source = source[1:-1, 1:-1]
    relative_residual = float(
        jnp.sqrt(jnp.mean(interior_residual**2))
        / jnp.maximum(jnp.sqrt(jnp.mean(interior_source**2)), 1.0e-30)
    )
    finite = bool(jnp.all(jnp.isfinite(equilibrium)) & jnp.isfinite(jnp.asarray(relative_residual)))
    return PredictiveNewtonResult(
        equilibrium=equilibrium,
        fixed_support_max_scaled_residual=fixed_maximum,
        dynamic_support_max_scaled_residual=dynamic_maximum,
        relative_nonlinear_residual_rms=relative_residual,
        fixed_support_converged=fixed_converged,
        dynamic_support_converged=dynamic_converged,
        finite=finite,
        seed_source=seed_source,
        newton_linearization=newton_linearization,
        fixed_support_iterations=fixed_iterations,
        dynamic_support_iterations=dynamic_iterations,
    )


# ── Implicit-differentiation adjoint (∂ψ*/∂θ for the IDA loop) ─────


@partial(jax.custom_vjp, nondiff_argnums=tuple(range(3, 20)) + (23,))
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
    fixed_psi_axis: jnp.ndarray | None = None,
    fixed_psi_boundary: jnp.ndarray | None = None,
    fixed_support_weights: jnp.ndarray | None = None,
    decompose_coil_field: bool = False,
) -> jnp.ndarray:
    """Differentiable predictive free-boundary solve — ``ψ*`` with an exact implicit-diff adjoint.

    Identical forward to :func:`solve_predictive_equilibrium`, but ``jax.grad`` w.r.t. the
    differentiated inputs ``(coil_I, pprime_vals, ffprime_vals)`` uses the implicit function
    theorem on the converged fixed point ``F(ψ*, θ) = 0`` (:func:`predictive_gs_residual`): one
    adjoint solve ``(∂F/∂ψ)ᵀ λ = ψ̄`` then ``θ̄ = −(∂F/∂θ)ᵀ λ``. The gradient cost is independent
    of the Anderson iteration count — the property the DIII-D IDA MAP/MCMC loop needs — and the
    adjoint is exact only insofar as the forward has converged (``F(ψ*) ≈ 0``) and the axis /
    separatrix finders are smooth (they are). Fixed topology inputs, ``ip_target``, and solver
    settings are treated as constants; gradients are returned only for coil currents and the two
    sampled profiles.
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
        fixed_psi_axis=fixed_psi_axis,
        fixed_psi_boundary=fixed_psi_boundary,
        fixed_support_weights=fixed_support_weights,
        decompose_coil_field=decompose_coil_field,
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
    fixed_psi_axis: jnp.ndarray | None,
    fixed_psi_boundary: jnp.ndarray | None,
    fixed_support_weights: jnp.ndarray | None,
    decompose_coil_field: bool,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray | None, ...]]:
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
        fixed_psi_axis=fixed_psi_axis,
        fixed_psi_boundary=fixed_psi_boundary,
        fixed_support_weights=fixed_support_weights,
        decompose_coil_field=decompose_coil_field,
    )
    return psi, (
        psi,
        coil_I,
        pprime_vals,
        ffprime_vals,
        fixed_psi_axis,
        fixed_psi_boundary,
        fixed_support_weights,
    )


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
    decompose_coil_field: bool,
    residuals: tuple[jnp.ndarray | None, ...],
    psi_bar: jnp.ndarray,
) -> tuple[jnp.ndarray | None, ...]:
    """Implicit-diff VJP: adjoint solve ``(∂F/∂ψ)ᵀ λ = ψ̄`` then ``θ̄ = −(∂F/∂θ)ᵀ λ``."""
    (
        psi,
        coil_I,
        pprime_vals,
        ffprime_vals,
        fixed_psi_axis,
        fixed_psi_boundary,
        fixed_support_weights,
    ) = residuals
    if psi is None or coil_I is None or pprime_vals is None or ffprime_vals is None:
        raise ValueError("custom VJP residuals lost a differentiated solver input")
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
            fixed_psi_axis=fixed_psi_axis,
            fixed_psi_boundary=fixed_psi_boundary,
            fixed_support_weights=fixed_support_weights,
            decompose_coil_field=decompose_coil_field,
        )

    _, vjp_psi = jax.vjp(residual_in_psi, psi)

    def adjoint_operator(lam_flat: jnp.ndarray) -> jnp.ndarray:
        return cast(jnp.ndarray, vjp_psi(lam_flat.reshape(shape))[0].reshape(-1))

    # Diagonal right preconditioner: F's interior rows scale as the Δ* centre coefficient
    # (~1/h²) and its wall rows as 1 — a badly-scaled system whose conditioning worsens ~1/h² at
    # finer grids (uncorrected, BiCGSTAB needs thousands of iterations and silently caps out).
    # Normalising by that known diagonal makes the iteration count grid-insensitive.
    d_r = R_grid[1] - R_grid[0]
    d_z = Z_grid[1] - Z_grid[0]
    precond_diag = jnp.full(psi.size, 2.0 / d_r**2 + 2.0 / d_z**2).at[wall_idx].set(1.0)

    def right_preconditioned_operator(y_flat: jnp.ndarray) -> jnp.ndarray:
        return adjoint_operator(y_flat / precond_diag)

    # Right preconditioning preserves the physical residual used by GMRES's stopping
    # criterion.  The historical left-preconditioned BiCGSTAB could report completion while
    # ||A^T lambda - psi_bar|| / ||psi_bar|| remained O(1), silently biasing coil gradients
    # on grids whose coil filaments lie inside the rectangular computational domain.
    y_flat, _info = _gmres(
        right_preconditioned_operator,
        psi_bar.reshape(-1),
        tol=_ADJOINT_TOL,
        atol=0.0,
        restart=_ADJOINT_RESTART,
        maxiter=_ADJOINT_MAXITER,
        solve_method="incremental",
    )
    lam_flat = y_flat / precond_diag
    adjoint_residual = adjoint_operator(lam_flat) - psi_bar.reshape(-1)
    relative_adjoint_residual = jnp.linalg.norm(adjoint_residual) / jnp.maximum(
        jnp.linalg.norm(psi_bar),
        jnp.asarray(1.0e-30, dtype=psi_bar.dtype),
    )
    # A failed adjoint must poison the downstream gradient rather than silently return a
    # plausible biased value.  Real-surface finite/FD gates then fail closed on non-finite output.
    lam_flat = jnp.where(
        relative_adjoint_residual <= _ADJOINT_RESIDUAL_FACTOR * _ADJOINT_TOL,
        lam_flat,
        jnp.full_like(lam_flat, jnp.nan),
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
            fixed_psi_axis=fixed_psi_axis,
            fixed_psi_boundary=fixed_psi_boundary,
            fixed_support_weights=fixed_support_weights,
            decompose_coil_field=decompose_coil_field,
        )

    _, vjp_theta = jax.vjp(residual_in_theta, coil_I, pprime_vals, ffprime_vals)
    g_ci, g_pp, g_ff = vjp_theta(lam)
    return (-g_ci, -g_pp, -g_ff, None, None, None)


solve_predictive_equilibrium_diff.defvjp(_solve_diff_fwd, _solve_diff_bwd)


def differentiate_predictive_equilibrium_root(
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
    equilibrium: jnp.ndarray,
    *,
    cutoff_width: float = DEFAULT_CUTOFF_WIDTH,
    mu0: float = MU0_SI,
    fixed_psi_axis: jnp.ndarray | None = None,
    fixed_psi_boundary: jnp.ndarray | None = None,
    fixed_support_weights: jnp.ndarray | None = None,
    decompose_coil_field: bool = True,
) -> jnp.ndarray:
    """Attach the implicit equilibrium adjoint to an already converged root.

    The forward returns ``equilibrium`` without another fixed-point iteration;
    reverse mode differentiates the production residual with respect to coil
    currents, p-prime, and FF-prime. This is the differentiable companion to
    :func:`solve_predictive_equilibrium_newton`: the nonlinear root is obtained
    once, while each VJP pays only for the implicit adjoint. Callers remain
    responsible for checking the Newton result's fail-closed convergence and
    residual fields before admitting its gradient.
    """
    expected_shape = (int(Z_grid.size), int(R_grid.size))
    if equilibrium.shape != expected_shape:
        raise ValueError(f"equilibrium shape {equilibrium.shape} != grid shape {expected_shape}")
    return solve_predictive_equilibrium_diff(
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
        equilibrium,
        0,
        DEFAULT_ANDERSON_DEPTH,
        DEFAULT_MIXING,
        1,
        cutoff_width,
        DEFAULT_TOL,
        mu0,
        fixed_psi_axis,
        fixed_psi_boundary,
        fixed_support_weights,
        decompose_coil_field,
    )
