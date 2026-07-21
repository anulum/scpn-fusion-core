# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — JAX geometric-multigrid preconditioner for the free-boundary GS solve
"""Geometric-multigrid preconditioner for the predictive free-boundary Krylov solve.

The forward Picard step of :mod:`scpn_fusion.core.jax_free_boundary_predictive` solves the
linear system ``A ψ = rhs`` where ``A`` is the Grad-Shafranov operator ``Δ*`` on the interior
with **identity wall rows** (Dirichlet imposed through the right-hand side). Unpreconditioned
BiCGSTAB on that system needs an iteration count that grows like ``1/h²`` with grid refinement;
a single geometric-multigrid V-cycle is the classical optimal preconditioner for this elliptic
operator, making the Krylov iteration count roughly grid-size independent.

:func:`build_gs_mg_preconditioner` returns a **linear**, jit-compatible operator
``M(r) ≈ A⁻¹ r``:

- wall-ring rows of ``A`` are the identity, so ``M`` reproduces ``r`` on the ring exactly and
  feeds those values into the interior stencil as Dirichlet data on the finest level;
- the interior is approximated by one (or more) V-cycles of red-black Gauss-Seidel smoothing
  with full-weighting restriction and bilinear prolongation, **zero initial guess and fixed
  sweep counts** — which is what makes the map linear in ``r`` (a requirement of
  ``jax.scipy.sparse.linalg.bicgstab``'s ``M`` argument).

The stencil matches ``_laplacian_star`` of the predictive solver exactly (including its
``max(R, 1e-6)`` axis guard), and the level hierarchy is built once per grid at Python trace
time (static shapes at every level, so the whole preconditioner jits cleanly). Coarsening is
vertex-centred (``n → (n+1)//2``) and is only applied while both dimensions are odd, which
keeps the transfer operators exact; on grids that cannot coarsen (even dimensions) the
preconditioner honestly degrades to smoothing-only on the finest level — still linear, still a
valid (if weaker) preconditioner.

Honest quality statement (measured during development, 33²–129²): on ring-free vectors — the
representative Krylov workload once the identity wall rows are satisfied — a single default
V-cycle contracts the true residual to ≈ 0.05–0.14 of the input, grid-independently. On
ring-HEAVY vectors the Dirichlet data couples into the near-ring interior at ``O(1/h²)`` and
one V-cycle leaves a leftover that can exceed ``‖r‖`` (≈ 15 ‖r‖ measured at 65²); this is
expected of a single preconditioner application and is absorbed by the outer Krylov iteration
— the ring rows themselves are reproduced exactly.

This module is the forward-speed lane (fidelity-curve Rung 2). The adjoint solve keeps its own
validated Jacobi preconditioner; correctness of the equilibrium is unchanged by construction —
a preconditioner alters the Krylov convergence path, not the solution of ``A ψ = rhs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

_AXIS_GUARD = 1e-6  # matches _laplacian_star in jax_free_boundary_predictive


@dataclass(frozen=True)
class _Level:
    """Static per-level geometry: stencil coefficients and red/black interior masks."""

    shape: tuple[int, int]
    a_e: jnp.ndarray  # east coefficient, interior shape (nz-2, nr-2)
    a_w: jnp.ndarray  # west coefficient
    a_ns: float  # north/south coefficient 1/dz²
    a_c: jnp.ndarray  # centre coefficient (positive; row is −a_c ψ_C + …)
    red: jnp.ndarray  # interior checkerboard masks, shape (nz-2, nr-2)
    black: jnp.ndarray


def _make_level(r_axis: NDArray[np.float64], nz: int, d_r: float, d_z: float) -> _Level:
    """Build the static stencil data for one grid level.

    The interior row of ``Δ*ψ = rhs`` reads
    ``a_e ψ_E + a_w ψ_W + a_ns (ψ_N + ψ_S) − a_c ψ_C = rhs`` with
    ``a_e = 1/dr² − 1/(2 R dr)``, ``a_w = 1/dr² + 1/(2 R dr)``, ``a_ns = 1/dz²`` and
    ``a_c = 2/dr² + 2/dz²`` — the same discretisation as ``_laplacian_star``.
    """
    nr = r_axis.shape[0]
    r_int = np.maximum(r_axis[1:-1], _AXIS_GUARD)[np.newaxis, :]  # broadcast over Z rows
    dr2 = d_r * d_r
    dz2 = d_z * d_z
    a_e = jnp.asarray(np.broadcast_to(1.0 / dr2 - 1.0 / (2.0 * r_int * d_r), (nz - 2, nr - 2)))
    a_w = jnp.asarray(np.broadcast_to(1.0 / dr2 + 1.0 / (2.0 * r_int * d_r), (nz - 2, nr - 2)))
    a_c = jnp.asarray(np.full((nz - 2, nr - 2), 2.0 / dr2 + 2.0 / dz2))
    ii, jj = np.mgrid[1 : nz - 1, 1 : nr - 1]
    red = jnp.asarray(((ii + jj) % 2) == 0)
    return _Level(
        shape=(nz, nr),
        a_e=a_e,
        a_w=a_w,
        a_ns=1.0 / dz2,
        a_c=a_c,
        red=red,
        black=jnp.asarray(((ii + jj) % 2) == 1),
    )


def _build_hierarchy(
    r_axis: NDArray[np.float64], nz: int, d_r: float, d_z: float, min_grid: int
) -> list[_Level]:
    """Vertex-centred level hierarchy, coarsening ``n → (n+1)//2`` while both dims stay odd."""
    levels = [_make_level(r_axis, nz, d_r, d_z)]
    nr = r_axis.shape[0]
    while nz % 2 == 1 and nr % 2 == 1 and (nz + 1) // 2 >= min_grid and (nr + 1) // 2 >= min_grid:
        r_axis = r_axis[::2]  # exact for a uniform axis; dr doubles
        nz = (nz + 1) // 2
        nr = (nr + 1) // 2
        d_r *= 2.0
        d_z *= 2.0
        levels.append(_make_level(r_axis, nz, d_r, d_z))
    return levels


def _smooth(psi: jnp.ndarray, rhs: jnp.ndarray, level: _Level, n_sweeps: int) -> jnp.ndarray:
    """Red-black Gauss-Seidel sweeps on the interior; the boundary ring is left untouched."""
    for _ in range(n_sweeps):
        for mask in (level.red, level.black):
            update = (
                level.a_e * psi[1:-1, 2:]
                + level.a_w * psi[1:-1, :-2]
                + level.a_ns * (psi[2:, 1:-1] + psi[:-2, 1:-1])
                - rhs[1:-1, 1:-1]
            ) / level.a_c
            psi = psi.at[1:-1, 1:-1].set(jnp.where(mask, update, psi[1:-1, 1:-1]))
    return psi


def _interior_residual(psi: jnp.ndarray, rhs: jnp.ndarray, level: _Level) -> jnp.ndarray:
    """``rhs − Δ*ψ`` on the interior, zero on the ring (the coarse-grid defect)."""
    lap = (
        level.a_e * psi[1:-1, 2:]
        + level.a_w * psi[1:-1, :-2]
        + level.a_ns * (psi[2:, 1:-1] + psi[:-2, 1:-1])
        - level.a_c * psi[1:-1, 1:-1]
    )
    return jnp.zeros_like(psi).at[1:-1, 1:-1].set(rhs[1:-1, 1:-1] - lap)


def _restrict(fine: jnp.ndarray, coarse_shape: tuple[int, int]) -> jnp.ndarray:
    """Full-weighting 9-point restriction onto a vertex-centred coarse grid (ring stays zero)."""
    nz_c, nr_c = coarse_shape
    interior = (
        4.0 * fine[2:-2:2, 2:-2:2]
        + 2.0
        * (
            fine[1:-3:2, 2:-2:2]
            + fine[3:-1:2, 2:-2:2]
            + fine[2:-2:2, 1:-3:2]
            + fine[2:-2:2, 3:-1:2]
        )
        + (
            fine[1:-3:2, 1:-3:2]
            + fine[1:-3:2, 3:-1:2]
            + fine[3:-1:2, 1:-3:2]
            + fine[3:-1:2, 3:-1:2]
        )
    ) / 16.0
    return jnp.zeros((nz_c, nr_c), dtype=fine.dtype).at[1:-1, 1:-1].set(interior)


def _prolongate(coarse: jnp.ndarray, fine_shape: tuple[int, int]) -> jnp.ndarray:
    """Bilinear prolongation from a vertex-centred coarse grid (``n_f = 2 n_c − 1``)."""
    fine = jnp.zeros(fine_shape, dtype=coarse.dtype)
    fine = fine.at[::2, ::2].set(coarse)
    fine = fine.at[::2, 1::2].set(0.5 * (coarse[:, :-1] + coarse[:, 1:]))
    fine = fine.at[1::2, ::2].set(0.5 * (coarse[:-1, :] + coarse[1:, :]))
    fine = fine.at[1::2, 1::2].set(
        0.25 * (coarse[:-1, :-1] + coarse[1:, :-1] + coarse[:-1, 1:] + coarse[1:, 1:])
    )
    return fine


def _vcycle(
    psi: jnp.ndarray,
    rhs: jnp.ndarray,
    levels: list[_Level],
    depth: int,
    pre_smooth: int,
    post_smooth: int,
    coarse_sweeps: int,
) -> jnp.ndarray:
    """One V-cycle on ``levels[depth:]``; coarse corrections use a homogeneous-Dirichlet ring."""
    level = levels[depth]
    if depth == len(levels) - 1:
        return _smooth(psi, rhs, level, coarse_sweeps)
    psi = _smooth(psi, rhs, level, pre_smooth)
    defect = _interior_residual(psi, rhs, level)
    coarse_level = levels[depth + 1]
    d_coarse = _restrict(defect, coarse_level.shape)
    e_coarse = _vcycle(
        jnp.zeros_like(d_coarse),
        d_coarse,
        levels,
        depth + 1,
        pre_smooth,
        post_smooth,
        coarse_sweeps,
    )
    correction = _prolongate(e_coarse, level.shape)
    # The correction must not disturb the Dirichlet ring carried by psi on this level.
    psi = psi.at[1:-1, 1:-1].add(correction[1:-1, 1:-1])
    return _smooth(psi, rhs, level, post_smooth)


def build_gs_mg_preconditioner(
    shape: tuple[int, int],
    R_grid: jnp.ndarray,
    d_r: float,
    d_z: float,
    *,
    n_vcycles: int = 1,
    pre_smooth: int = 2,
    post_smooth: int = 2,
    coarse_sweeps: int = 32,
    min_grid: int = 5,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build ``M(r) ≈ A⁻¹ r`` for the GS operator with identity wall rows.

    Parameters
    ----------
    shape : tuple[int, int]
        Grid shape ``(NZ, NR)`` of the flattened system.
    R_grid : jnp.ndarray
        Major-radius axis, length ``NR`` (uniform spacing assumed, as in the solver).
    d_r, d_z : float
        Grid spacings [m].
    n_vcycles : int, optional
        V-cycles per application; 1 is the standard preconditioner setting.
    pre_smooth, post_smooth : int, optional
        Red-black Gauss-Seidel sweeps before/after each coarse correction.
    coarse_sweeps : int, optional
        Smoother sweeps on the coarsest level (in lieu of a direct solve).
    min_grid : int, optional
        Smallest permitted level dimension.

    Returns
    -------
    Callable[[jnp.ndarray], jnp.ndarray]
        A linear, jit-compatible map on flat vectors of length ``NZ·NR``, suitable as the
        ``M`` argument of ``jax.scipy.sparse.linalg.bicgstab``.

    Raises
    ------
    ValueError
        If the shape/axis are inconsistent, any dimension is below 3, the spacings are not
        strictly positive and finite, or a tuning knob is non-positive.
    """
    nz, nr = int(shape[0]), int(shape[1])
    r_axis = np.asarray(R_grid, dtype=np.float64)
    if r_axis.ndim != 1 or r_axis.shape[0] != nr:
        raise ValueError(f"R_grid must be 1-D of length NR={nr}; got shape {r_axis.shape}.")
    if nz < 3 or nr < 3:
        raise ValueError(f"grid must be at least 3x3; got ({nz}, {nr}).")
    d_r_f, d_z_f = float(d_r), float(d_z)
    if not (np.isfinite(d_r_f) and d_r_f > 0.0 and np.isfinite(d_z_f) and d_z_f > 0.0):
        raise ValueError("d_r and d_z must be finite and > 0.")
    if min(n_vcycles, pre_smooth, post_smooth, coarse_sweeps, min_grid) < 1:
        raise ValueError("all multigrid tuning parameters must be >= 1.")

    levels = _build_hierarchy(r_axis, nz, d_r_f, d_z_f, min_grid)

    def apply(r_flat: jnp.ndarray) -> jnp.ndarray:
        r = r_flat.reshape((nz, nr))
        # Identity wall rows are inverted exactly: the ring of z equals the ring of r, and it
        # acts as the Dirichlet data seen by the interior stencil on the finest level.
        z = jnp.zeros_like(r)
        z = z.at[0, :].set(r[0, :]).at[-1, :].set(r[-1, :])
        z = z.at[:, 0].set(r[:, 0]).at[:, -1].set(r[:, -1])
        for _ in range(n_vcycles):
            z = _vcycle(z, r, levels, 0, pre_smooth, post_smooth, coarse_sweeps)
        return z.reshape(-1)

    return apply
