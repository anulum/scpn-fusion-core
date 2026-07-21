# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Free-boundary GS with Krylov forward + implicit-diff gradient
"""Free-boundary Grad-Shafranov solve with a Krylov linear solve and an implicit,
matrix-free adjoint gradient.

This is the fast/accurate companion to :mod:`scpn_fusion.core.jax_free_boundary_gs`.
Two algorithmic upgrades over the weighted-Jacobi reference solver there:

1. **Krylov forward.** Each Picard step solves the linear GS system with a
   matrix-free BiCGSTAB (:func:`jax.scipy.sparse.linalg.bicgstab`) instead of many
   weighted-Jacobi sweeps. The inner solve converges in ``O(N)`` mat-vecs rather than
   ``O(N²)`` sweeps, so the plasma-interior residual stays ``~1e-10`` across grid
   sizes (33² … 257²) with a fixed, small Picard count — where plain Jacobi degrades.

2. **Implicit-differentiation gradient.** The converged ``ψ*`` satisfies
   ``F(ψ*, θ) = 0`` (the nonlinear GS residual, ``θ = (I_coil, p', FF')``). Rather than
   back-propagating through the iterative solver — which yields only an *approximate*
   gradient (early-terminated Krylov, ``~1e-3``) and stores every iterate — a
   :func:`jax.custom_vjp` applies the implicit function theorem: one adjoint solve
   ``(∂F/∂ψ)ᵀ λ = ψ̄`` (matrix-free BiCGSTAB, converges to ``~1e-13``), then
   ``θ̄ = −(∂F/∂θ)ᵀ λ``. The cost is independent of the Picard count — the property the
   DIII-D IDA MAP/MCMC loop needs, where the gradient is evaluated for every parameter,
   every step.

   Gradient accuracy (vs central finite differences): the **profile gradients**
   ``∂/∂p'`` and ``∂/∂FF'`` — the parameters IDA infers — are **exact to ``~1e-6``**.
   The **coil-current gradient** ``∂/∂I_coil`` depends on the axis finder. With the default
   hard-``argmax`` axis (:func:`~scpn_fusion.core.jax_equilibrium_solver._interior_axis_flux`,
   identical to the Jacobi reference equilibrium) it is only ``~2 %`` accurate: ``I_coil``
   shifts the whole field and hence the magnetic axis, whose position the ``argmax`` cannot
   differentiate smoothly. Passing ``use_smooth_axis=True`` swaps in the smooth sub-cell
   O-point (:func:`~scpn_fusion.core.jax_o_point.smooth_axis_flux`) and the coil-current
   gradient becomes **exact (~1e-6)** — at the cost of a < 0.5 % shift in the axis flux and
   < 0.05 % (of ψ-span, at 65²+) shift in the equilibrium. The profile gradients are exact
   either way because they barely move the axis.

SI units throughout (μ₀ = 4π·10⁻⁷, currents [A], ψ [Wb]); ``ψ`` shape ``(NZ, NR)``.
The GS equation, source, normalised flux and vacuum field are shared with
:mod:`scpn_fusion.core.jax_free_boundary_gs`. Runs on whatever backend JAX has
(CPU/GPU) with no code change; the matrix-free mat-vecs and Krylov solve are the
GPU-friendly form (few large fused kernels, not thousands of tiny sequential sweeps).
"""

from __future__ import annotations

from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import bicgstab

from scpn_fusion.core.jax_equilibrium_solver import _boundary_flux_level, _interior_axis_flux
from scpn_fusion.core.jax_free_boundary_gs import MU0_SI, general_gs_source, vacuum_field_si
from scpn_fusion.core.jax_o_point import smooth_axis_flux

# Krylov / Picard defaults.
_BICGSTAB_TOL = 1.0e-11
_BICGSTAB_MAXITER = 400
_ADJOINT_MAXITER = 800


def _axis_flux(psi: jnp.ndarray, use_smooth_axis: bool) -> jnp.ndarray:
    """Magnetic-axis flux for the ``ψ_N`` normalisation.

    ``use_smooth_axis`` is a static (non-differentiated) flag, so this selection resolves at
    trace time.  Default (``False``) keeps the hard-``argmax``
    :func:`~scpn_fusion.core.jax_equilibrium_solver._interior_axis_flux` — the exact same
    equilibrium the weighted-Jacobi reference solver produces.  ``True`` swaps in the smooth
    sub-cell :func:`~scpn_fusion.core.jax_o_point.smooth_axis_flux`, which makes the
    implicit-diff coil-current gradient exact (the hard argmax is the sole reason it was
    ~2 %); it shifts the axis flux < 0.5 % and the equilibrium < 0.05 % of ψ-span at 65²+.
    """
    return cast(jnp.ndarray, smooth_axis_flux(psi) if use_smooth_axis else _interior_axis_flux(psi))


# ── Matrix-free GS operator ───────────────────────────────────────


def _laplacian_star(psi: jnp.ndarray, R_grid: jnp.ndarray, d_r: Any, d_z: Any) -> jnp.ndarray:
    """Grad-Shafranov operator ``Δ*ψ = ∂²ψ/∂R² − (1/R)∂ψ/∂R + ∂²ψ/∂Z²`` on the interior.

    Uses slicing (not periodic ``roll``) so no wrap-around is introduced; boundary rows
    are left as the raw ``ψ`` value there and overwritten by the caller's BC handling.
    """
    d2r = jnp.zeros_like(psi)
    d1r = jnp.zeros_like(psi)
    d2z = jnp.zeros_like(psi)
    d2r = d2r.at[:, 1:-1].set((psi[:, 2:] - 2.0 * psi[:, 1:-1] + psi[:, :-2]) / d_r**2)
    d1r = d1r.at[:, 1:-1].set((psi[:, 2:] - psi[:, :-2]) / (2.0 * d_r))
    d2z = d2z.at[1:-1, :].set((psi[2:, :] - 2.0 * psi[1:-1, :] + psi[:-2, :]) / d_z**2)
    r_safe = jnp.maximum(R_grid[jnp.newaxis, :], 1e-6)
    return d2r - d1r / r_safe + d2z


def _gs_linear_operator(
    psi_flat: jnp.ndarray, shape: tuple[int, int], R_grid: jnp.ndarray, d_r: Any, d_z: Any
) -> jnp.ndarray:
    """Linear GS operator with identity boundary rows, flattened for Krylov solves.

    Interior rows return ``Δ*ψ``; the four boundary edges return ``ψ`` itself, so the
    Dirichlet condition is imposed entirely through the right-hand side.
    """
    psi = psi_flat.reshape(shape)
    out = _laplacian_star(psi, R_grid, d_r, d_z)
    out = out.at[0, :].set(psi[0, :])
    out = out.at[-1, :].set(psi[-1, :])
    out = out.at[:, 0].set(psi[:, 0])
    out = out.at[:, -1].set(psi[:, -1])
    return out.reshape(-1)


def nonlinear_gs_residual(
    psi: jnp.ndarray,
    coil_I: jnp.ndarray,
    pprime_vals: jnp.ndarray,
    ffprime_vals: jnp.ndarray,
    R_grid: jnp.ndarray,
    Z_grid: jnp.ndarray,
    coil_R: jnp.ndarray,
    coil_Z: jnp.ndarray,
    psin_knots: jnp.ndarray,
    mu0: float = MU0_SI,
    use_smooth_axis: bool = False,
) -> jnp.ndarray:
    """Nonlinear GS residual ``F(ψ)`` whose root is the equilibrium.

    Interior: ``Δ*ψ − S(ψ)`` with ``S = −μ₀R²p' − FF'``.  Boundary edges: ``ψ − ψ_vac``,
    so ``F = 0`` enforces both the PDE inside and the free-boundary Dirichlet condition
    from the external coils.  ``use_smooth_axis`` selects the axis finder (see
    :func:`_axis_flux`); it must match the forward solve for the adjoint to be consistent.
    """
    d_r = R_grid[1] - R_grid[0]
    d_z = Z_grid[1] - Z_grid[0]
    psi_vac = vacuum_field_si(R_grid, Z_grid, coil_R, coil_Z, coil_I, mu0)
    axis = _axis_flux(psi, use_smooth_axis)
    bnd = _boundary_flux_level(psi_vac)
    source = general_gs_source(psi, R_grid, axis, bnd, psin_knots, pprime_vals, ffprime_vals, mu0)
    res = _laplacian_star(psi, R_grid, d_r, d_z) - source
    res = res.at[0, :].set(psi[0, :] - psi_vac[0, :])
    res = res.at[-1, :].set(psi[-1, :] - psi_vac[-1, :])
    res = res.at[:, 0].set(psi[:, 0] - psi_vac[:, 0])
    res = res.at[:, -1].set(psi[:, -1] - psi_vac[:, -1])
    return cast(jnp.ndarray, res)


# ── Forward solve + implicit-diff gradient ────────────────────────


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8, 9, 10, 11))
def solve_free_boundary_gs_implicit(
    coil_I: jnp.ndarray,
    pprime_vals: jnp.ndarray,
    ffprime_vals: jnp.ndarray,
    R_grid: jnp.ndarray,
    Z_grid: jnp.ndarray,
    coil_R: jnp.ndarray,
    coil_Z: jnp.ndarray,
    psin_knots: jnp.ndarray,
    n_picard: int = 40,
    picard_omega: float = 0.6,
    mu0: float = MU0_SI,
    use_smooth_axis: bool = False,
) -> jnp.ndarray:
    """Solve the SI free-boundary GS equilibrium (Krylov forward, implicit-diff gradient).

    Differentiable inputs are ``coil_I``, ``pprime_vals`` and ``ffprime_vals`` (the first
    three arguments); ``jax.grad`` w.r.t. them uses the exact adjoint (see module
    docstring). ``n_picard`` must be a Python int (it is the ``lax.scan`` length).

    Parameters
    ----------
    coil_I : (N_coils,) PF coil currents [A].
    pprime_vals, ffprime_vals : (K,) samples of ``p'`` / ``FF'`` on ``psin_knots``.
    R_grid, Z_grid : 1D uniform grids [m]; ``ψ`` returned as ``(NZ, NR)``.
    coil_R, coil_Z : (N_coils,) coil positions [m].
    psin_knots : (K,) monotonic normalised-flux knots on [0, 1].
    n_picard : outer Picard iterations (each an inner BiCGSTAB linear solve).
    picard_omega : Picard under-relaxation blend.
    mu0 : SI vacuum permeability [H/m].
    use_smooth_axis : if ``True``, use the smooth sub-cell O-point
        (:func:`~scpn_fusion.core.jax_o_point.smooth_axis_flux`) for the ``ψ_N``
        normalisation, which makes the coil-current gradient exact (~1e-6 vs ~2 %);
        default ``False`` keeps the hard-argmax axis (identical to the Jacobi reference).
        Best on grids ≥ 49²; the O-point must be interior.
    """
    psi_vac = vacuum_field_si(R_grid, Z_grid, coil_R, coil_Z, coil_I, mu0)
    shape = (Z_grid.shape[0], R_grid.shape[0])
    d_r = R_grid[1] - R_grid[0]
    d_z = Z_grid[1] - Z_grid[0]

    def operator(psi_flat: jnp.ndarray) -> jnp.ndarray:
        return _gs_linear_operator(psi_flat, shape, R_grid, d_r, d_z)

    def picard(psi_flat: jnp.ndarray, _: Any) -> tuple[jnp.ndarray, None]:
        psi = psi_flat.reshape(shape)
        axis = _axis_flux(psi, use_smooth_axis)
        bnd = _boundary_flux_level(psi_vac)
        source = general_gs_source(
            psi, R_grid, axis, bnd, psin_knots, pprime_vals, ffprime_vals, mu0
        )
        rhs = source
        rhs = rhs.at[0, :].set(psi_vac[0, :])
        rhs = rhs.at[-1, :].set(psi_vac[-1, :])
        rhs = rhs.at[:, 0].set(psi_vac[:, 0])
        rhs = rhs.at[:, -1].set(psi_vac[:, -1])
        sol, _info = bicgstab(  # type: ignore[no-untyped-call]  # jax.scipy.sparse.linalg is untyped
            operator, rhs.reshape(-1), x0=psi_flat, tol=_BICGSTAB_TOL, maxiter=_BICGSTAB_MAXITER
        )
        return picard_omega * sol + (1.0 - picard_omega) * psi_flat, None

    root, _ = jax.lax.scan(picard, psi_vac.reshape(-1), None, length=n_picard)
    return cast(jnp.ndarray, root.reshape(shape))


def _solve_fwd(
    coil_I: jnp.ndarray,
    pprime_vals: jnp.ndarray,
    ffprime_vals: jnp.ndarray,
    R_grid: jnp.ndarray,
    Z_grid: jnp.ndarray,
    coil_R: jnp.ndarray,
    coil_Z: jnp.ndarray,
    psin_knots: jnp.ndarray,
    n_picard: int,
    picard_omega: float,
    mu0: float,
    use_smooth_axis: bool,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
    psi = solve_free_boundary_gs_implicit(
        coil_I,
        pprime_vals,
        ffprime_vals,
        R_grid,
        Z_grid,
        coil_R,
        coil_Z,
        psin_knots,
        n_picard,
        picard_omega,
        mu0,
        use_smooth_axis,
    )
    return psi, (psi, coil_I, pprime_vals, ffprime_vals)


def _solve_bwd(
    R_grid: jnp.ndarray,
    Z_grid: jnp.ndarray,
    coil_R: jnp.ndarray,
    coil_Z: jnp.ndarray,
    psin_knots: jnp.ndarray,
    n_picard: int,
    picard_omega: float,
    mu0: float,
    use_smooth_axis: bool,
    residuals: tuple[jnp.ndarray, ...],
    psi_bar: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Implicit-diff VJP: one adjoint solve ``(∂F/∂ψ)ᵀ λ = ψ̄`` then ``θ̄ = −(∂F/∂θ)ᵀ λ``."""
    psi, coil_I, pprime_vals, ffprime_vals = residuals
    shape = psi.shape

    def residual_in_psi(p: jnp.ndarray) -> jnp.ndarray:
        return nonlinear_gs_residual(
            p,
            coil_I,
            pprime_vals,
            ffprime_vals,
            R_grid,
            Z_grid,
            coil_R,
            coil_Z,
            psin_knots,
            mu0,
            use_smooth_axis,
        )

    _, vjp_psi = jax.vjp(residual_in_psi, psi)

    def adjoint_operator(lam_flat: jnp.ndarray) -> jnp.ndarray:
        return cast(jnp.ndarray, vjp_psi(lam_flat.reshape(shape))[0].reshape(-1))

    lam_flat, _info = bicgstab(  # type: ignore[no-untyped-call]  # jax.scipy.sparse.linalg is untyped
        adjoint_operator, psi_bar.reshape(-1), tol=_BICGSTAB_TOL, maxiter=_ADJOINT_MAXITER
    )
    lam = lam_flat.reshape(shape)

    def residual_in_theta(ci: jnp.ndarray, pp: jnp.ndarray, ff: jnp.ndarray) -> jnp.ndarray:
        return nonlinear_gs_residual(
            psi, ci, pp, ff, R_grid, Z_grid, coil_R, coil_Z, psin_knots, mu0, use_smooth_axis
        )

    _, vjp_theta = jax.vjp(residual_in_theta, coil_I, pprime_vals, ffprime_vals)
    g_ci, g_pp, g_ff = vjp_theta(lam)
    return (-g_ci, -g_pp, -g_ff)


solve_free_boundary_gs_implicit.defvjp(_solve_fwd, _solve_bwd)
