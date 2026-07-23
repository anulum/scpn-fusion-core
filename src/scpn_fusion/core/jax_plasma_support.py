# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — axis-connected plasma support (hard + AD-safe soft)
"""Axis-connected plasma support for free-boundary current / GS sources.

On a diverted equilibrium the level set ``ψ_N < 1`` is **not** the closed plasma
core: private-flux regions above an upper X-point and below a lower X-point also
sit inside that flux window. Placing toroidal current there (hard
``psi_n < 1`` or a pure ``tanh`` LCFS roll-off) is a topology error — flagged by
peer review and by the IDA same-case current-support diagnostics.

This module provides:

1. **Hard** 4-connected flood fill from the magnetic-axis seed through a boolean
   level set — exact topology for diagnostics and non-AD paths.
2. **Soft** (AD-safe) support: a smooth LCFS weight times a fixed-iteration
   soft flood fill from a soft axis seed. Spread is blocked where the LCFS weight
   vanishes, so private-flux islands that are flux-level members but not
   path-connected to the axis stay suppressed while remaining differentiable.

The soft path is what the predictive IDA solver uses; the hard path pins tests
and the optional exact mask on :func:`general_gs_source`.
"""

from __future__ import annotations

from collections import deque
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

# Softmax temperature for the soft axis seed (depth |ψ − ψ_wall|).
DEFAULT_SEED_BETA = 8.0


def soft_lcfs_weight(psi_n: jnp.ndarray, cutoff_width: float) -> jnp.ndarray:
    """Smooth plasma weight from normalised flux: ~1 in the core, ~0 outside LCFS.

    Parameters
    ----------
    psi_n
        Normalised flux on the ``(NZ, NR)`` grid (preferably already clipped).
    cutoff_width
        Positive roll-off width in ``ψ_N`` units (same contract as the predictive
        solver's ``cutoff_width``).
    """
    width = jnp.maximum(jnp.asarray(cutoff_width, dtype=psi_n.dtype), 1.0e-12)
    return 0.5 * (1.0 - jnp.tanh((psi_n - 1.0) / width))


def hard_axis_connected_mask(
    level_set: NDArray[np.bool_],
    seed_i: int,
    seed_j: int,
) -> NDArray[np.bool_]:
    """Exact 4-connected flood fill of ``level_set`` from ``(seed_i, seed_j)``.

    Indices use the array layout ``level_set[i, j]`` with ``i`` along the first
    axis (Z rows) and ``j`` along the second (R columns), matching the GS grids.

    If the seed cell is out of bounds or not in the level set, the result is all
    ``False`` (fail-closed empty support).
    """
    mask = np.asarray(level_set, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("level_set must be 2-D")
    nz, nr = mask.shape
    out = np.zeros((nz, nr), dtype=bool)
    if not (0 <= seed_i < nz and 0 <= seed_j < nr):
        return out
    if not mask[seed_i, seed_j]:
        return out
    queue: deque[tuple[int, int]] = deque([(seed_i, seed_j)])
    out[seed_i, seed_j] = True
    while queue:
        i, j = queue.popleft()
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < nz and 0 <= nj < nr and mask[ni, nj] and not out[ni, nj]:
                out[ni, nj] = True
                queue.append((ni, nj))
    return out


def hard_axis_connected_from_psi_n(
    psi_n: NDArray[np.float64],
    *,
    seed_i: int | None = None,
    seed_j: int | None = None,
    level: float = 1.0,
) -> NDArray[np.bool_]:
    """Hard axis-connected component of ``ψ_N < level``.

    When the seed is omitted it is the cell of minimal ``ψ_N`` (the magnetic axis
    for the usual increasing-``ψ_N`` convention after normalisation).
    """
    pn = np.asarray(psi_n, dtype=np.float64)
    if pn.ndim != 2:
        raise ValueError("psi_n must be 2-D")
    level_set = pn < float(level)
    if seed_i is None or seed_j is None:
        # Prefer the deepest core cell inside the level set.
        work = np.where(level_set, pn, np.inf)
        if not np.isfinite(work).any():
            return np.zeros(pn.shape, dtype=bool)
        flat = int(np.argmin(work))
        seed_i, seed_j = divmod(flat, pn.shape[1])
    return hard_axis_connected_mask(level_set, int(seed_i), int(seed_j))


def _neighbor_max(field: jnp.ndarray) -> jnp.ndarray:
    """Max of the four orthogonal neighbours (zeros outside the domain)."""
    up = jnp.concatenate([jnp.zeros((1, field.shape[1]), dtype=field.dtype), field[:-1, :]], axis=0)
    down = jnp.concatenate(
        [field[1:, :], jnp.zeros((1, field.shape[1]), dtype=field.dtype)], axis=0
    )
    left = jnp.concatenate(
        [jnp.zeros((field.shape[0], 1), dtype=field.dtype), field[:, :-1]], axis=1
    )
    right = jnp.concatenate(
        [field[:, 1:], jnp.zeros((field.shape[0], 1), dtype=field.dtype)], axis=1
    )
    return jnp.maximum(jnp.maximum(up, down), jnp.maximum(left, right))


def soft_axis_seed(psi: jnp.ndarray, *, beta: float = DEFAULT_SEED_BETA) -> jnp.ndarray:
    """Soft one-hot-like seed peaked at the magnetic axis from flux depth.

    Uses ``|ψ − ψ_wall|`` (sign-agnostic axis peak) with an RMS-normalised softmax,
    matching the smooth O-point seed philosophy without a hard ``argmax``.
    """
    wall = 0.25 * (
        jnp.mean(psi[0, :]) + jnp.mean(psi[-1, :]) + jnp.mean(psi[:, 0]) + jnp.mean(psi[:, -1])
    )
    depth = jnp.abs(psi - wall)
    rms = jnp.sqrt(jnp.mean(depth**2)) + jnp.asarray(1.0e-30, dtype=psi.dtype)
    weights = jax.nn.softmax((beta * depth / rms).reshape(-1)).reshape(psi.shape)
    return weights


@partial(jax.jit, static_argnames=("n_steps",))
def soft_axis_connected_support(
    psi: jnp.ndarray,
    psi_n: jnp.ndarray,
    cutoff_width: float,
    *,
    n_steps: int | None = None,
    seed_beta: float = DEFAULT_SEED_BETA,
) -> jnp.ndarray:
    """Differentiable axis-connected plasma support in ``[0, 1]``.

    Parameters
    ----------
    psi
        Poloidal flux ``(NZ, NR)`` — used only to place the soft axis seed.
    psi_n
        Normalised flux on the same grid.
    cutoff_width
        Smooth LCFS roll-off width (``ψ_N`` units).
    n_steps
        Soft flood-fill iterations. Default ``NZ + NR`` covers the Manhattan
        diameter of the grid. Must be a static Python ``int`` under ``jit``.
    seed_beta
        Softmax temperature for the axis seed.

    Returns
    -------
    jnp.ndarray
        Support weight ``(NZ, NR)``. Near 1 on the axis-connected core, near 0
        outside the LCFS and on private-flux islands.
    """
    if n_steps is None:
        n_steps = int(psi.shape[0] + psi.shape[1])
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    lcfs = soft_lcfs_weight(psi_n, cutoff_width)
    seed = soft_axis_seed(psi, beta=seed_beta) * lcfs
    # Normalise seed so the peak is O(1) even on fine grids (softmax mass spreads).
    seed_peak = jnp.max(seed) + jnp.asarray(1.0e-30, dtype=psi.dtype)
    state = seed / seed_peak

    def body(carry: jnp.ndarray, _: None) -> tuple[jnp.ndarray, None]:
        spread = jnp.maximum(carry, _neighbor_max(carry))
        nxt = lcfs * spread
        return nxt, None

    filled, _ = jax.lax.scan(body, state, xs=None, length=int(n_steps))
    # Keep in [0, 1]; LCFS already bounds the open set.
    clipped: jnp.ndarray = jnp.clip(filled, 0.0, 1.0)
    return clipped


def axis_connected_support(
    psi: jnp.ndarray,
    psi_n: jnp.ndarray,
    cutoff_width: float,
    *,
    mode: str = "soft",
    n_steps: int | None = None,
    seed_beta: float = DEFAULT_SEED_BETA,
    hard_level: float = 1.0,
) -> jnp.ndarray:
    """Dispatch soft (default) or hard axis-connected support.

    ``mode='hard'`` returns a boolean float mask ``{0, 1}`` (exact flood fill).
    ``mode='soft'`` returns the AD-safe weight field.
    """
    if mode == "soft":
        return cast(
            jnp.ndarray,
            soft_axis_connected_support(
                psi,
                psi_n,
                cutoff_width,
                n_steps=n_steps,
                seed_beta=seed_beta,
            ),
        )
    if mode == "hard":
        pn = np.asarray(psi_n, dtype=np.float64)
        # Seed from soft axis argmax for consistency with the differentiable seed.
        seed = np.asarray(soft_axis_seed(psi, beta=seed_beta))
        flat = int(np.argmax(seed))
        si, sj = divmod(flat, pn.shape[1])
        mask = hard_axis_connected_from_psi_n(pn, seed_i=si, seed_j=sj, level=hard_level)
        out: jnp.ndarray = jnp.asarray(mask, dtype=psi.dtype)
        return out
    raise ValueError(f"unknown support mode {mode!r}; expected 'soft' or 'hard'")
