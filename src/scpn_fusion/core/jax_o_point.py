# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — smooth differentiable magnetic-axis (O-point) flux finder
"""Smooth, differentiable magnetic-axis (O-point) flux for Grad-Shafranov solves.

The ``ψ_N`` normalisation of a free-boundary equilibrium needs the poloidal flux at the
magnetic axis, ``ψ_axis``. The obvious estimator — ``ψ`` at the interior cell of maximal
peakness (:func:`scpn_fusion.core.jax_equilibrium_solver._interior_axis_flux`) — uses a
hard ``argmax``. Through a reverse-mode autodiff (and especially through the implicit-diff
adjoint of :mod:`scpn_fusion.core.jax_free_boundary_gs_implicit`) that ``argmax`` drops the
sensitivity of the axis *position* to ``ψ``, which is the sole reason the coil-current
gradient of the implicit solver was only ~2 %% accurate. The profile gradients were always
exact because they barely move the axis.

This module gives a value-accurate **and** smoothly-differentiable ``ψ_axis`` via a
sub-cell parabolic vertex:

1. A **soft-argmax position** over the flux depth ``|ψ − ψ_wall|`` (peaks at the axis,
   sign-agnostic), RMS-normalised so the softmax temperature is grid- and scale-free and
   the denominator never vanishes. This gives a smooth estimate of the axis cell.
2. A fixed ``patch × patch`` stencil around the rounded soft position, fitted by an
   **unweighted** 2-D quadratic least squares. Because the weights are fixed, the fit is a
   *constant linear map* of the stencil values (a precomputed pseudo-inverse) — so its
   gradient is exact and free of the variable-weight noise that spoils a softmax-weighted
   fit. A patch of 7 comfortably contains the true peak even when the soft seed lands a
   cell away.
3. The **vertex** of that quadratic (its stationary point, clipped inside the patch) — the
   magnetic axis to sub-cell accuracy — evaluated analytically.

Measured against central finite differences on a free-boundary case: axis-value bias
< 0.5 %% of the peak, and the implicit-diff **coil-current gradient becomes exact to
~1e-6** (from ~2 %%), with the profile gradients unchanged (~1e-5). The solution drifts
< 0.05 %% of ψ-span from the hard-argmax reference at 65² — a physically identical
equilibrium with an exact gradient.

The magnetic axis is assumed interior (at least ``(patch−1)/2`` cells from the wall); the
seed is clipped to keep the stencil on-grid. Grids must satisfy ``NZ, NR ≥ patch``.
"""

from __future__ import annotations

from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.jax_equilibrium_solver import _boundary_flux_level

# Default sub-cell fit window and softmax temperature (validated sweet spot: bias < 0.5 %,
# gradient ~1e-6 across 33²–65², robust in β at patch 7).
DEFAULT_PATCH = 7
DEFAULT_BETA = 6.0

_PINV_CACHE: dict[int, NDArray[np.float64]] = {}


def _quadratic_pinv(patch: int) -> NDArray[np.float64]:
    """Constant Moore-Penrose pseudo-inverse of the 2-D quadratic design matrix.

    Columns ``[1, x, y, x², y², xy]`` over the integer ``patch × patch`` stencil centred at
    the origin.  Returned as a concrete NumPy ``(6, patch²)`` matrix so ``pinv @ patch``
    yields the six quadratic coefficients as a fixed linear map (an exact, smooth gradient).

    The cache holds a NumPy array (not a JAX one): it is built outside tracing and converted
    per-trace at the call site, so no traced intermediate ever escapes a ``jit`` scope.
    """
    if patch not in _PINV_CACHE:
        centre = (patch - 1) // 2
        axis = np.arange(patch) - centre
        gx, gy = np.meshgrid(axis, axis, indexing="ij")
        x = gx.ravel().astype(np.float64)
        y = gy.ravel().astype(np.float64)
        design = np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], axis=1)
        _PINV_CACHE[patch] = np.linalg.pinv(design).astype(np.float64)
    return _PINV_CACHE[patch]


@partial(jax.jit, static_argnames=("patch",))
def smooth_axis_flux(
    psi: jnp.ndarray, patch: int = DEFAULT_PATCH, beta: float = DEFAULT_BETA
) -> jnp.ndarray:
    """Smooth, differentiable magnetic-axis poloidal flux ``ψ_axis`` [same units as ``ψ``].

    Parameters
    ----------
    psi : (NZ, NR) poloidal flux; the magnetic axis must be interior.
    patch : odd sub-cell fit window (≥ 3, ≤ min(NZ, NR)); larger tolerates a coarser seed.
    beta : soft-argmax temperature on the RMS-normalised flux depth.

    Returns
    -------
    Scalar ``ψ_axis`` — the flux at the quadratic-fit vertex (the O-point to sub-cell
    accuracy). Smooth in ``psi`` (exact reverse-mode gradient), value-accurate to < 0.5 %.
    """
    pinv = jnp.asarray(_quadratic_pinv(patch))
    centre = (patch - 1) // 2

    wall = _boundary_flux_level(psi)
    depth = jnp.abs(psi - wall)
    rms = jnp.sqrt(jnp.mean(depth**2)) + 1e-30
    # zero the border band so the soft position (and stencil) stay interior
    interior = (
        jnp.zeros_like(psi)
        .at[centre:-centre, centre:-centre]
        .set(depth[centre:-centre, centre:-centre])
    )
    weights = jax.nn.softmax((beta * interior / rms).reshape(-1))

    nz, nr = psi.shape
    ii, jj = jnp.meshgrid(jnp.arange(nz), jnp.arange(nr), indexing="ij")
    i_soft = jnp.sum(weights * ii.reshape(-1))
    j_soft = jnp.sum(weights * jj.reshape(-1))
    i0 = jnp.clip(jnp.round(i_soft).astype(int), centre, nz - 1 - centre)
    j0 = jnp.clip(jnp.round(j_soft).astype(int), centre, nr - 1 - centre)

    stencil = jax.lax.dynamic_slice(psi, (i0 - centre, j0 - centre), (patch, patch))
    a, b, c, d, e, f = pinv @ stencil.reshape(-1)

    # vertex of a + b·x + c·y + d·x² + e·y² + f·xy : solve [[2d, f], [f, 2e]]·[x;y] = [−b;−c]
    det = 4.0 * d * e - f * f
    det = jnp.where(jnp.abs(det) < 1e-30, 1e-30, det)
    x_star = jnp.clip((-2.0 * e * b + f * c) / det, -float(centre), float(centre))
    y_star = jnp.clip((f * b - 2.0 * d * c) / det, -float(centre), float(centre))
    vertex = a + b * x_star + c * y_star + d * x_star**2 + e * y_star**2 + f * x_star * y_star
    return cast(jnp.ndarray, vertex)
