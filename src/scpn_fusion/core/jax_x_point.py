# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — smooth differentiable X-point (separatrix) flux finder
"""Smooth, differentiable X-point (separatrix) poloidal flux for free-boundary solves.

A *diverted* plasma is bounded not by a material limiter but by the **X-point** — a saddle
of the poloidal flux ``ψ`` where the poloidal field vanishes (``∇ψ = 0``). The flux value at
that saddle is the last-closed-flux-surface level ``ψ_bndry`` that normalises the current
profile (``ψ_N = (ψ − ψ_axis)/(ψ_bndry − ψ_axis)``). Where the magnetic axis is the *maximum*
found by :func:`scpn_fusion.core.jax_o_point.smooth_axis_flux`, the X-point is the *saddle*;
both are critical points of ``ψ`` and share the same "``∇ψ = 0``" locator idea.

For the predictive free-boundary solve
(:mod:`scpn_fusion.core.jax_free_boundary_predictive`) ``ψ_bndry`` is **not** an input — it is
the X-point flux of the current iterate and must be found *self-consistently* each step, and
smoothly, so the coupled fixed point stays differentiable. A hard ``argmin`` of ``|∇ψ|`` drops
the sub-cell sensitivity and injects the same gradient noise the hard axis ``argmax`` did.

This module gives a smooth estimate of ``ψ_bndry`` via a **soft-argmin of ``|∇ψ|²``** over the
lower-null search region:

1. ``|∇ψ|²`` is evaluated by central differences on the SI grid (spacing from ``R_grid`` /
   ``Z_grid``); it vanishes at every critical point — the axis (a maximum) and the X-point
   (a saddle).
2. The **magnetic axis** is excluded two ways so the soft-argmin cannot latch onto the (also
   low-``|∇ψ|``) axis or its flanks: (a) the search is restricted to a *fixed* lower band of
   the domain — a geometric fraction of the grid height, deliberately **not** tied to a
   per-iteration ``argmax`` of ``ψ`` (which would jump during a cold-start solve and
   destabilise the Anderson fixed point); and (b) a disk of ``axis_margin`` cells around the
   *soft* axis position (a softmax over the flux depth) is masked out. The disk exclusion is
   what makes the estimate robust to the exact band fraction — without it the band edge grazes
   the axis flank and the soft-argmin flips between the X-point and a spurious near-axis null.
3. A softmax over ``−β·|∇ψ|²`` (RMS-normalised so ``β`` is scale-free) weights the flux there:
   ``ψ_bndry ≈ Σ w·ψ``, concentrated on the saddle, smooth in ``ψ``.

Measured on the FreeGS 0.8.2 DIII-D reference (65², lower single null): the hard ``|∇ψ|``
minimum sits at the true X-point and returns ``ψ_X = 0.1405`` (bit-identical to the reference
separatrix flux); this smooth estimator returns ``0.1387`` (≈ 1.3 % low), enough to drive the
coupled solve to < 1 % of the reference. A sub-cell saddle-vertex fit (mirroring the O-point's
quadratic vertex) would tighten it further and is the next refinement.

Scope: a single lower X-point (LSN). The search region is the lower part of the domain below
the flux peak; ``search_below=False`` flips it for an upper single null. Double-null / limited
plasmas are out of scope for this first version.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

# Validated defaults (FreeGS DIII-D 65², LSN): the lower ~55 % of the domain holds the X-point;
# the axis-disk exclusion (radius axis_margin cells around the soft axis position) keeps the
# result robust to the exact band fraction; edge keeps wall nulls out; β concentrates on the
# saddle without starving the softmax; axis_beta sets the softness of the axis-position seed.
DEFAULT_BETA = 8.0
DEFAULT_LOWER_FRAC = 0.55
DEFAULT_EDGE = 2
DEFAULT_AXIS_MARGIN = 6
DEFAULT_AXIS_BETA = 6.0


@partial(jax.jit, static_argnames=("lower_frac", "edge", "search_below"))
def smooth_xpoint_flux(
    psi: jnp.ndarray,
    R_grid: jnp.ndarray,
    Z_grid: jnp.ndarray,
    beta: float = DEFAULT_BETA,
    lower_frac: float = DEFAULT_LOWER_FRAC,
    edge: int = DEFAULT_EDGE,
    axis_margin: float = DEFAULT_AXIS_MARGIN,
    axis_beta: float = DEFAULT_AXIS_BETA,
    search_below: bool = True,
) -> jnp.ndarray:
    """Smooth, differentiable X-point (separatrix) flux ``ψ_bndry`` [same units as ``ψ``].

    Parameters
    ----------
    psi : (NZ, NR) poloidal flux; a diverted plasma with one X-point in the search band.
    R_grid, Z_grid : 1D uniform grids [m] giving the central-difference spacing.
    beta : soft-argmin temperature on the RMS-normalised ``|∇ψ|²``.
    lower_frac : fraction of the grid height forming the (fixed) search band; the axis must lie
        outside it. ``search_below`` → the bottom ``lower_frac`` rows; else the top ``lower_frac``.
    edge : cells of computational-wall border excluded from the search (drops boundary nulls).
    axis_margin : radius (in cells) of the disk masked out around the soft axis position, so the
        soft-argmin cannot latch onto the axis or its low-``|∇ψ|`` flank.
    axis_beta : softmax temperature locating the axis position (over the flux depth).
    search_below : if ``True`` search the lower band (lower single null), else the upper band.

    Returns
    -------
    Scalar ``ψ_bndry`` — the flux at the X-point saddle, a soft-argmin over ``|∇ψ|²``. Smooth
    in ``psi`` (differentiable) so the coupled free-boundary fixed point stays differentiable.
    """
    nz, nr = psi.shape
    d_r = R_grid[1] - R_grid[0]
    d_z = Z_grid[1] - Z_grid[0]

    # |∇ψ|² by central differences (interior only; edges left zero and excluded below).
    g_z = jnp.zeros_like(psi).at[1:-1, :].set((psi[2:, :] - psi[:-2, :]) / (2.0 * d_z))
    g_r = jnp.zeros_like(psi).at[:, 1:-1].set((psi[:, 2:] - psi[:, :-2]) / (2.0 * d_r))
    grad2 = g_r * g_r + g_z * g_z

    ii = jnp.arange(nz)[:, jnp.newaxis]
    jj = jnp.arange(nr)[jnp.newaxis, :]

    # Soft axis position over the flux depth |ψ − ψ_wall| (peaks at the axis). Used only to
    # place the exclusion disk — its position need not be differentiable; ψ_bndry is a smooth
    # soft-weighted sum, which is.
    wall = 0.25 * (
        jnp.mean(psi[0, :]) + jnp.mean(psi[-1, :]) + jnp.mean(psi[:, 0]) + jnp.mean(psi[:, -1])
    )
    depth = jnp.abs(psi - wall)
    rms = jnp.sqrt(jnp.mean(depth**2)) + 1e-30
    w_axis = jax.nn.softmax((axis_beta * depth / rms).reshape(-1))
    i_grid = jnp.broadcast_to(ii, (nz, nr))
    j_grid = jnp.broadcast_to(jj, (nz, nr))
    i_axis = jnp.sum(w_axis * i_grid.reshape(-1))
    j_axis = jnp.sum(w_axis * j_grid.reshape(-1))
    off_axis = ((ii - i_axis) ** 2 + (jj - j_axis) ** 2) > axis_margin**2

    # Fixed geometric search band (a stable region — NOT tied to a per-iteration argmax of ψ,
    # which would jump during a cold start and destabilise the Anderson fixed point).
    band = int(round(lower_frac * nz))
    lower = ii < band
    upper = ii >= nz - band
    half = jnp.where(search_below, lower, upper)
    interior = (ii >= edge) & (ii < nz - edge) & (jj >= edge) & (jj < nr - edge)
    region = half & interior & off_axis

    # Soft-argmin of |∇ψ|² over the region → weights concentrated on the saddle.
    ref = jnp.mean(jnp.where(region, grad2, 0.0)) + 1e-30
    big = jnp.max(grad2) * 10.0 + 1.0
    score = jnp.where(region, grad2 / ref, big)
    weights = jax.nn.softmax((-beta * score).reshape(-1))
    return jnp.sum(weights * psi.reshape(-1))
