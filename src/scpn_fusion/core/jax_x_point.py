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

This module gives a locally smooth estimate of ``ψ_bndry`` via a **saddle-classified sub-cell
quadratic fit** over the lower-null search region:

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
3. The Hessian determinant rejects O-points and other gradient minima that are not saddles.
   A grid-normalised radius around the axis rejects remote wall/coil nulls without encoding a
   machine-specific coordinate.
4. The lowest-gradient saddle seeds an unweighted 2-D quadratic fit. Its stationary point is
   evaluated analytically, giving sub-cell value accuracy and the exact local envelope
   derivative while the selected saddle basin remains unchanged.
5. If a cold-start iterate has no valid saddle, the original RMS-normalised soft-argmin remains
   the stable fallback until a diverted topology forms.

Scope: a single lower X-point (LSN). The search region is the lower part of the domain below
the flux peak; ``search_below=False`` flips it for an upper single null. Double-null / limited
plasmas are out of scope for this first version.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from scpn_fusion.core.jax_o_point import _quadratic_pinv

# Validated defaults (FreeGS DIII-D 65², LSN): the lower ~55 % of the domain holds the X-point;
# the axis-disk exclusion (radius axis_margin cells around the soft axis position) keeps the
# result robust to the exact band fraction; the grid-normalised search radius rejects remote
# wall/coil nulls; edge keeps the wall out; β controls the no-saddle fallback; axis_beta sets
# the softness of the axis-position seed.
DEFAULT_BETA = 8.0
DEFAULT_LOWER_FRAC = 0.55
DEFAULT_EDGE = 2
DEFAULT_AXIS_MARGIN = 6
DEFAULT_AXIS_BETA = 6.0
DEFAULT_SEARCH_RADIUS = 0.40
DEFAULT_PATCH = 5


@partial(jax.jit, static_argnames=("lower_frac", "edge", "search_below", "patch"))
def smooth_xpoint_flux(
    psi: jnp.ndarray,
    R_grid: jnp.ndarray,
    Z_grid: jnp.ndarray,
    beta: float = DEFAULT_BETA,
    lower_frac: float = DEFAULT_LOWER_FRAC,
    edge: int = DEFAULT_EDGE,
    axis_margin: float = DEFAULT_AXIS_MARGIN,
    axis_beta: float = DEFAULT_AXIS_BETA,
    search_radius: float = DEFAULT_SEARCH_RADIUS,
    patch: int = DEFAULT_PATCH,
    search_below: bool = True,
) -> jnp.ndarray:
    """Locally smooth X-point (separatrix) flux ``ψ_bndry`` [same units as ``ψ``].

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
    search_radius : maximum axis-to-saddle distance in grid-normalised coordinates. This
        rejects remote wall/coil nulls while remaining independent of machine dimensions.
    patch : odd quadratic-fit window (≥ 3, ≤ min(NZ, NR)).
    search_below : if ``True`` search the lower band (lower single null), else the upper band.

    Returns
    -------
    Scalar ``ψ_bndry`` — the flux at the quadratic saddle vertex. The value is differentiable
    in ``psi`` inside a stable saddle basin; a smooth soft-argmin is used when no saddle exists.
    """
    nz, nr = psi.shape
    centre = (patch - 1) // 2
    pinv = jnp.asarray(_quadratic_pinv(patch))
    d_r = R_grid[1] - R_grid[0]
    d_z = Z_grid[1] - Z_grid[0]

    # |∇ψ|² by central differences (interior only; edges left zero and excluded below).
    g_z = jnp.zeros_like(psi).at[1:-1, :].set((psi[2:, :] - psi[:-2, :]) / (2.0 * d_z))
    g_r = jnp.zeros_like(psi).at[:, 1:-1].set((psi[:, 2:] - psi[:, :-2]) / (2.0 * d_r))
    grad2 = g_r * g_r + g_z * g_z
    h_rr = (
        jnp.zeros_like(psi)
        .at[:, 1:-1]
        .set((psi[:, 2:] - 2.0 * psi[:, 1:-1] + psi[:, :-2]) / (d_r * d_r))
    )
    h_zz = (
        jnp.zeros_like(psi)
        .at[1:-1, :]
        .set((psi[2:, :] - 2.0 * psi[1:-1, :] + psi[:-2, :]) / (d_z * d_z))
    )
    h_rz = (
        jnp.zeros_like(psi)
        .at[1:-1, 1:-1]
        .set((psi[2:, 2:] - psi[2:, :-2] - psi[:-2, 2:] + psi[:-2, :-2]) / (4.0 * d_r * d_z))
    )
    saddle = h_rr * h_zz - h_rz * h_rz < 0.0

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

    # Stable cold-start fallback: the original soft-argmin remains defined before a diverted
    # topology (and therefore an X-point) has formed.
    ref = jnp.mean(jnp.where(region, grad2, 0.0)) + 1e-30
    big = jnp.max(grad2) * 10.0 + 1.0
    score = jnp.where(region, grad2 / ref, big)
    weights = jax.nn.softmax((-beta * score).reshape(-1))
    fallback = jnp.sum(weights * psi.reshape(-1))

    # A true separatrix saddle is local to the plasma axis. Remote wall/coil nulls can have a
    # smaller sampled |∇ψ|² than the physical X-point, so classify by Hessian and reject points
    # outside a machine-independent radius expressed as fractions of the grid dimensions.
    distance2 = ((ii - i_axis) / nz) ** 2 + ((jj - j_axis) / nr) ** 2
    saddle_region = region & saddle & (distance2 <= search_radius**2)
    saddle_score = jnp.where(saddle_region, grad2, big)
    flat_index = jnp.argmin(saddle_score.reshape(-1))
    i0 = jnp.clip(flat_index // nr, centre, nz - 1 - centre)
    j0 = jnp.clip(flat_index % nr, centre, nr - 1 - centre)

    stencil = jax.lax.dynamic_slice(psi, (i0 - centre, j0 - centre), (patch, patch))
    a, b, c, d, e, f = pinv @ stencil.reshape(-1)
    determinant = 4.0 * d * e - f * f
    safe_determinant = jnp.where(jnp.abs(determinant) < 1e-30, 1e-30, determinant)
    x_star = jnp.clip(
        (-2.0 * e * b + f * c) / safe_determinant,
        -float(centre),
        float(centre),
    )
    y_star = jnp.clip(
        (f * b - 2.0 * d * c) / safe_determinant,
        -float(centre),
        float(centre),
    )
    vertex = a + b * x_star + c * y_star + d * x_star**2 + e * y_star**2 + f * x_star * y_star
    valid = jnp.any(saddle_region) & (determinant < 0.0) & jnp.isfinite(vertex)
    return jnp.where(valid, vertex, fallback)
