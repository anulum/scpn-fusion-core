# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — SI free-boundary differentiable Grad-Shafranov (general p'/FF')
"""SI free-boundary Grad-Shafranov solve with a general ``p'(ψ)`` / ``FF'(ψ)`` source.

This module unifies two capabilities that previously lived apart in the codebase:

- the JAX-differentiable free-boundary Picard/SOR machinery of
  :mod:`scpn_fusion.core.jax_equilibrium_solver` (autodiff through a fixed number
  of iterations, toroidal Green's-function vacuum boundary), and
- the *general* Grad-Shafranov current source of
  :class:`scpn_fusion.core.fusion_kernel` — ``Jφ = R p'(ψ) + FF'(ψ)/(μ₀ R)`` — which
  was NumPy-only and therefore not differentiable.

The result is a single ``ψ(R,Z)`` provider in **SI units** (μ₀ = 4π·10⁻⁷, currents in
amperes, ψ in webers) whose output is differentiable with respect to the profile
samples ``p'`` / ``FF'`` and the coil currents, via ``jax.grad``.  It is the
Milestone-A deliverable for the DIII-D Integrated Data Analysis (IDA) collaboration:
inputs ``(p', FF', I_coil)`` → ``ψ(R,Z)`` with analytic gradients, replacing a
numerical finite-difference stencil.

Grad-Shafranov equation solved (SI, cylindrical):

    Δ*ψ ≡ ∂²ψ/∂R² − (1/R) ∂ψ/∂R + ∂²ψ/∂Z²  =  −μ₀ R² p'(ψ) − FF'(ψ)

with a free-boundary Dirichlet condition on the computational wall taken from the
external poloidal-field coils (vacuum Green's function), not a zero-flux wall.

Profiles are supplied as samples of ``p'`` and ``FF'`` on a monotonic normalised-flux
grid ``ψ_N ∈ [0, 1]`` and interpolated differentiably; a spline/PCHIP basis adapter
can be layered on top without changing this contract (the sampled values remain the
differentiated quantity).

References: Grad & Rubin (1958); Shafranov (1966); Lao et al. (1985).
"""

from __future__ import annotations

from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import jit

# Reuse the unit-agnostic primitives from the normalized-units solver: the complete
# elliptic integrals, the SOR sweep (pure operator, carries no μ₀), and the
# axis/wall flux extractors (they only compare flux values).
from scpn_fusion.core.jax_equilibrium_solver import (
    _boundary_flux_level,
    _ellipe_approx,
    _ellipk_approx,
    _interior_axis_flux,
    _sor_step,
)

# SI vacuum permeability [H/m].
MU0_SI: float = 4.0e-7 * jnp.pi


# ── SI toroidal Green's function ──────────────────────────────────


@jit
def greens_psi_si(
    R: jnp.ndarray, Z: jnp.ndarray, Rc: float, Zc: float, current: float, mu0: float = MU0_SI
) -> jnp.ndarray:
    """Poloidal flux [Wb] from one filamentary coil at ``(Rc, Zc)`` carrying ``current`` [A].

    Ψ = (μ₀ I / 2π) √(R Rc) [(2/k − k) K(k²) − (2/k) E(k²)],  k² = 4 R Rc / ((R+Rc)² + (Z−Zc)²)

    Identical form to :func:`jax_equilibrium_solver.greens_psi` but with an explicit
    SI ``μ₀`` and current in amperes.  Grad & Shafranov (1958); Lao et al. (1985).
    """
    R_safe = jnp.maximum(R, 1e-6)
    denom = (R_safe + Rc) ** 2 + (Z - Zc) ** 2
    k2 = jnp.clip(4.0 * R_safe * Rc / jnp.maximum(denom, 1e-30), 1e-9, 0.999999)
    k = jnp.sqrt(k2)
    k_val = _ellipk_approx(k2)
    e_val = _ellipe_approx(k2)
    prefactor = mu0 * current / (2.0 * jnp.pi)
    psi = prefactor * jnp.sqrt(R_safe * Rc) * ((2.0 / k - k) * k_val - (2.0 / k) * e_val)
    return cast(jnp.ndarray, jnp.where(jnp.isfinite(psi), psi, 0.0))


@partial(jit, static_argnums=())
def vacuum_field_si(
    R_grid: jnp.ndarray,
    Z_grid: jnp.ndarray,
    coil_R: jnp.ndarray,
    coil_Z: jnp.ndarray,
    coil_I: jnp.ndarray,
    mu0: float = MU0_SI,
) -> jnp.ndarray:
    """Sum the SI vacuum flux ``ψ`` from all coils onto the ``(NZ, NR)`` grid.

    Parameters
    ----------
    R_grid : (NR,) radial grid [m]
    Z_grid : (NZ,) vertical grid [m]
    coil_R, coil_Z : (N_coils,) coil positions [m]
    coil_I : (N_coils,) coil currents [A]
    mu0 : SI vacuum permeability [H/m]
    """
    r2d = R_grid[jnp.newaxis, :]
    z2d = Z_grid[:, jnp.newaxis]

    def single_coil(carry: jnp.ndarray, coil: jnp.ndarray) -> tuple[jnp.ndarray, None]:
        rc, zc, ic = coil
        return carry + greens_psi_si(r2d, z2d, rc, zc, ic, mu0), None

    coils = jnp.stack([coil_R, coil_Z, coil_I], axis=-1)
    psi_init = jnp.zeros((Z_grid.shape[0], R_grid.shape[0]))
    psi_vac, _ = jax.lax.scan(single_coil, psi_init, coils)
    return psi_vac


# ── General Grad-Shafranov source ─────────────────────────────────


@jit
def normalised_flux(
    psi: jnp.ndarray, psi_axis: jnp.ndarray, psi_boundary: jnp.ndarray
) -> jnp.ndarray:
    """Normalised poloidal flux ``ψ_N = (ψ − ψ_axis)/(ψ_boundary − ψ_axis)``, clipped to [0, 1].

    A sign-aware floor on the denominator keeps the map finite for either increasing-
    or decreasing-ψ conventions without leaking a discontinuity into the gradient.
    """
    raw = psi_boundary - psi_axis
    scale = jnp.maximum(jnp.maximum(jnp.max(jnp.abs(psi)), jnp.abs(psi_boundary)), 1.0)
    floor = 1.0e-9 * scale
    sign = jnp.where(raw < 0.0, -1.0, 1.0)
    denom = jnp.where(jnp.abs(raw) > floor, raw, sign * floor)
    return jnp.clip((psi - psi_axis) / denom, 0.0, 1.0)


@jit
def general_gs_source(
    psi: jnp.ndarray,
    R_grid: jnp.ndarray,
    psi_axis: jnp.ndarray,
    psi_boundary: jnp.ndarray,
    psin_knots: jnp.ndarray,
    pprime_vals: jnp.ndarray,
    ffprime_vals: jnp.ndarray,
    mu0: float = MU0_SI,
) -> jnp.ndarray:
    """Grad-Shafranov RHS ``S = −μ₀ R² p'(ψ_N) − FF'(ψ_N)`` on the ``(NZ, NR)`` grid [SI].

    ``p'`` and ``FF'`` are given as samples on the monotonic knot grid ``psin_knots``
    (``ψ_N ∈ [0, 1]``) and interpolated with :func:`jnp.interp`, which is differentiable
    with respect to ``pprime_vals`` / ``ffprime_vals``.  The source is zero outside the
    last closed flux surface (``ψ_N ≥ 1``), so the plasma is self-consistently bounded.
    """
    psi_n = normalised_flux(psi, psi_axis, psi_boundary)
    pprime = jnp.interp(psi_n, psin_knots, pprime_vals)
    ffprime = jnp.interp(psi_n, psin_knots, ffprime_vals)
    r2d = R_grid[jnp.newaxis, :]
    source = -(mu0 * r2d**2 * pprime + ffprime)
    inside = psi_n < 1.0
    return jnp.where(inside, source, 0.0)


# ── Free-boundary Picard/SOR solve ────────────────────────────────


@partial(jit, static_argnums=(8, 9))
def solve_free_boundary_gs(
    R_grid: jnp.ndarray,
    Z_grid: jnp.ndarray,
    coil_R: jnp.ndarray,
    coil_Z: jnp.ndarray,
    coil_I: jnp.ndarray,
    psin_knots: jnp.ndarray,
    pprime_vals: jnp.ndarray,
    ffprime_vals: jnp.ndarray,
    max_picard: int = 100,
    sor_per_picard: int = 100,
    picard_omega: float = 0.5,
    sor_omega: float = 0.9,
    mu0: float = MU0_SI,
) -> jnp.ndarray:
    """Solve the SI free-boundary GS equilibrium with general ``p'`` / ``FF'`` profiles.

    Parameters
    ----------
    R_grid, Z_grid : 1D uniform grids [m]; ``ψ`` is returned as ``(NZ, NR)``.
    coil_R, coil_Z, coil_I : PF coil positions [m] and currents [A].
    psin_knots : (K,) monotonic normalised-flux knots spanning [0, 1].
    pprime_vals : (K,) samples of ``p'(ψ_N)`` = dp/dψ [Pa/Wb].
    ffprime_vals : (K,) samples of ``FF'(ψ_N)`` = F dF/dψ [T²·m²/Wb].
    max_picard, sor_per_picard : static outer/inner iteration counts.
    picard_omega : Picard under-relaxation blend for the outer nonlinear loop.
    sor_omega : inner relaxation factor. The inner sweep is a **weighted-Jacobi**
        update (XLA-vectorised, no sequential sweep dependence), so it is stable only
        for ``sor_omega ≤ 1``; over-relaxation (``> 1``) diverges. Use ``≈ 0.9``.
    mu0 : SI vacuum permeability [H/m].

    Returns
    -------
    psi : (NZ, NR) equilibrium poloidal flux [Wb].  Differentiable w.r.t.
        ``pprime_vals``, ``ffprime_vals`` and ``coil_I`` via ``jax.grad``.

    Notes
    -----
    Convergence: with the defaults the plasma-interior GS residual reaches ``~1e-10``
    relative to the source on a 33×33 grid. The weighted-Jacobi inner solve converges
    slowly (spectral radius → 1), so hitting sub-20 ms latencies at 129²+ needs a
    stronger linear solver (multigrid) and/or implicit differentiation of the
    fixed point (to avoid back-propagating through every sweep) and/or a GPU backend —
    tracked as Milestone-A latency work, not yet done here.
    """
    d_r = R_grid[1] - R_grid[0]
    d_z = Z_grid[1] - Z_grid[0]

    psi_vac = vacuum_field_si(R_grid, Z_grid, coil_R, coil_Z, coil_I, mu0)

    def picard_body(psi_curr: jnp.ndarray, _: Any) -> tuple[jnp.ndarray, None]:
        psi_axis = _interior_axis_flux(psi_curr)
        psi_bnd = _boundary_flux_level(psi_vac)
        source = general_gs_source(
            psi_curr, R_grid, psi_axis, psi_bnd, psin_knots, pprime_vals, ffprime_vals, mu0
        )

        def sor_body(p: jnp.ndarray, __: Any) -> tuple[jnp.ndarray, None]:
            return _sor_step(p, source, R_grid, d_r, d_z, sor_omega, psi_vac), None

        psi_relaxed, _ = jax.lax.scan(sor_body, psi_curr, None, length=sor_per_picard)
        psi_new = picard_omega * psi_relaxed + (1.0 - picard_omega) * psi_curr
        return psi_new, None

    psi_final, _ = jax.lax.scan(picard_body, psi_vac, None, length=max_picard)
    return cast(jnp.ndarray, psi_final)


@jit
def toroidal_plasma_current(
    psi: jnp.ndarray,
    R_grid: jnp.ndarray,
    Z_grid: jnp.ndarray,
    psi_axis: jnp.ndarray,
    psi_boundary: jnp.ndarray,
    psin_knots: jnp.ndarray,
    pprime_vals: jnp.ndarray,
    ffprime_vals: jnp.ndarray,
    mu0: float = MU0_SI,
) -> jnp.ndarray:
    """Total toroidal plasma current ``Ip = ∫ Jφ dA`` [A], with ``Jφ = R p' + FF'/(μ₀ R)``.

    Emerges from the solved equilibrium (an output, not an input) — useful for
    validating the general-profile solve against a target current.
    """
    psi_n = normalised_flux(psi, psi_axis, psi_boundary)
    pprime = jnp.interp(psi_n, psin_knots, pprime_vals)
    ffprime = jnp.interp(psi_n, psin_knots, ffprime_vals)
    r2d = R_grid[jnp.newaxis, :]
    r_safe = jnp.maximum(r2d, 1e-6)
    j_phi = jnp.where(psi_n < 1.0, r2d * pprime + ffprime / (mu0 * r_safe), 0.0)
    d_r = R_grid[1] - R_grid[0]
    d_z = Z_grid[1] - Z_grid[0]
    return jnp.sum(j_phi) * d_r * d_z
