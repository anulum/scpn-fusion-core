# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — fully compiled (lax.while_loop) predictive free-boundary forward solve
"""Fully compiled predictive free-boundary forward solve — the whole Anderson loop on-device.

The eager reference implementation (:func:`~scpn_fusion.core.jax_free_boundary_predictive.
solve_predictive_equilibrium`) runs the Anderson fixed-point iteration as a Python loop with
per-iteration host synchronisations. Measured on an A100 (Rung-2 wall-clock record): that
forward is HOST-LOOP-bound — ~0 % GPU utilisation, wall-clock dominated by dispatch overhead
rather than linear algebra, while the multigrid-preconditioned inner solve already needs only
5–6 Krylov iterations at 129² (grid-independent).

This module removes the host from the loop: the ENTIRE iteration — Ip ramp, coupled Picard
step (axis/X-point → Ip-normalised Jφ → von Hagenow wall flux → MG-preconditioned BiCGSTAB),
Anderson mixing with rolling fixed-shape history, rank-deficiency guard and the early-stop
test — runs inside a single :func:`jax.lax.while_loop` under :func:`jax.jit`.

Semantics match the eager solver's (same coupled step, same warm-up behaviour, same
break-before-update early stop, same damped-Picard NaN fallback) to numerical tolerance —
NOT bit-exactly (compiled reductions associate differently); the equivalence is pinned by
tests at span-relative tolerance, the same policy as the MG-vs-plain equality test. The
differentiable path (``solve_predictive_equilibrium_diff``) keeps the eager forward — the
implicit-function-theorem adjoint depends only on the converged fixed point, not on how the
forward trajectory was produced.

Compilation is cached per (grid geometry, solver settings) via an ``lru_cache`` factory, so
repeated solves (the production/benchmark pattern) pay tracing+compilation once. A uniform
grid is REQUIRED and verified fail-closed (the whole discretisation assumes it).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

_State = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]

from scpn_fusion.core.jax_free_boundary_gs import MU0_SI, vacuum_field_si
from scpn_fusion.core.jax_free_boundary_predictive import (
    DEFAULT_ANDERSON_DEPTH,
    DEFAULT_CUTOFF_WIDTH,
    DEFAULT_IP_RAMP,
    DEFAULT_MIXING,
    DEFAULT_N_ITER,
    DEFAULT_TOL,
    _coupled_step,
)
from scpn_fusion.core.jax_multigrid_precond import build_gs_mg_preconditioner


def _require_uniform(name: str, axis: NDArray[np.float64]) -> float:
    """Fail closed on non-uniform axes — the discretisation assumes uniform spacing."""
    steps = np.diff(axis)
    if axis.ndim != 1 or axis.size < 3:
        raise ValueError(f"{name} must be a 1-D axis with at least 3 points")
    if not np.all(np.isfinite(steps)) or steps[0] <= 0.0:
        raise ValueError(f"{name} must be strictly increasing and finite")
    if np.max(np.abs(steps - steps[0])) > 1e-9 * abs(steps[0]):
        raise ValueError(f"{name} must be uniformly spaced for the compiled forward")
    return float(steps[0])


@lru_cache(maxsize=16)
def _make_runner(
    nz: int,
    nr: int,
    r0: float,
    d_r: float,
    d_z: float,
    n_iter: int,
    anderson_depth: int,
    mixing: float,
    ip_ramp: int,
    cutoff_width: float,
    tol: float,
    mu0: float,
    use_mg_preconditioner: bool,
) -> Callable[..., jnp.ndarray]:
    """Build and jit the while_loop runner for one (geometry, settings) combination."""
    shape = (nz, nr)
    n_flat = nz * nr
    m = anderson_depth
    r_axis = np.asarray(r0 + d_r * np.arange(nr), dtype=np.float64)
    precond: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    if use_mg_preconditioner:
        precond = build_gs_mg_preconditioner(shape, jnp.asarray(r_axis), d_r, d_z)
    ramp_div = float(max(ip_ramp, 1))
    # Anderson column validity mask template: column i (of m-1) pairs rows (i, i+1) of the
    # rolling history; with n_valid entries occupying the LAST n_valid rows, the valid
    # columns are i >= m - n_valid.
    col_idx = jnp.arange(m - 1)

    def run(
        coil_I: jnp.ndarray,
        pprime_vals: jnp.ndarray,
        ffprime_vals: jnp.ndarray,
        R_grid: jnp.ndarray,
        Z_grid: jnp.ndarray,
        psin_knots: jnp.ndarray,
        ip_target: jnp.ndarray,
        response_matrix: jnp.ndarray,
        wall_idx: jnp.ndarray,
        source_idx: jnp.ndarray,
        coil_wall: jnp.ndarray,
        x0: jnp.ndarray,
    ) -> jnp.ndarray:
        dA = jnp.asarray(d_r * d_z)

        def g_of(x: jnp.ndarray, ip_now: jnp.ndarray) -> jnp.ndarray:
            return _coupled_step(
                x,
                ip_now,
                coil_wall,
                shape,
                R_grid,
                Z_grid,
                jnp.asarray(d_r),
                jnp.asarray(d_z),
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

        def cond(state: _State) -> jnp.ndarray:
            k, _x, _f_h, _x_h, _nv, done = state
            return (k < n_iter) & (~done)

        def body(state: _State) -> _State:
            k, x, f_hist, x_hist, n_valid, _done = state
            ip_k = ip_target * jnp.minimum(1.0, (k + 1.0) / ramp_div)
            f = g_of(x, ip_k) - x
            # Break-before-update semantics: once converged (Ip fully ramped), neither the
            # history nor x are touched — the loop exits returning the pre-update iterate.
            done_now = (k >= ip_ramp) & (jnp.linalg.norm(f) <= tol * (jnp.linalg.norm(x) + 1.0))
            f_new = jnp.roll(f_hist, -1, axis=0).at[-1].set(f)
            x_new = jnp.roll(x_hist, -1, axis=0).at[-1].set(x)
            nv_new = jnp.minimum(n_valid + 1, m)
            # Anderson step over the valid suffix of the rolling history. Invalid (warm-up)
            # columns are zeroed; SVD least squares assigns them zero weight, so the first
            # iteration (no valid pairs) reduces exactly to the damped Picard update.
            valid = (col_idx >= (m - nv_new))[jnp.newaxis, :]
            df = jnp.where(valid, (f_new[1:] - f_new[:-1]).T, 0.0)
            dx = jnp.where(valid, (x_new[1:] - x_new[:-1]).T, 0.0)
            gamma, _res, _rank, _sv = jnp.linalg.lstsq(df, f, rcond=None)
            x_anderson = x + mixing * f - (dx + mixing * df) @ gamma
            x_fallback = x + mixing * f
            x_next = jnp.where(jnp.all(jnp.isfinite(x_anderson)), x_anderson, x_fallback)
            return (
                k + 1,
                jnp.where(done_now, x, x_next),
                jnp.where(done_now, f_hist, f_new),
                jnp.where(done_now, x_hist, x_new),
                jnp.where(done_now, n_valid, nv_new),
                done_now,
            )

        state0 = (
            jnp.asarray(0),
            x0,
            jnp.zeros((m, n_flat), dtype=x0.dtype),
            jnp.zeros((m, n_flat), dtype=x0.dtype),
            jnp.asarray(0),
            jnp.asarray(False),
        )
        _k, x_final, _fh, _xh, _nv, _done = jax.lax.while_loop(cond, body, state0)
        return x_final.reshape(shape)

    return jax.jit(run)


def solve_predictive_equilibrium_compiled(
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
    use_mg_preconditioner: bool = True,
) -> jnp.ndarray:
    """Compiled predictive free-boundary solve — same fixed point as the eager solver.

    Same inputs and semantics as
    :func:`~scpn_fusion.core.jax_free_boundary_predictive.solve_predictive_equilibrium`;
    the whole Anderson iteration runs on-device under ``jax.jit``. Compilation is cached per
    (grid geometry, settings), so repeated solves pay tracing once.

    ``use_mg_preconditioner`` defaults to **True** here (unlike the eager reference): the
    committed iteration evidence shows plain BiCGSTAB cannot reach the inner tolerance at
    129² within any practical cap (DNF at 20 000), i.e. the preconditioner is a robustness
    requirement at production grids, and under compilation its V-cycle no longer pays the
    eager dispatch penalty.

    Raises
    ------
    ValueError
        On non-uniform / degenerate axes, inconsistent shapes, or non-positive settings.
    """
    r_np = np.asarray(R_grid, dtype=np.float64)
    z_np = np.asarray(Z_grid, dtype=np.float64)
    d_r = _require_uniform("R_grid", r_np)
    d_z = _require_uniform("Z_grid", z_np)
    if min(n_iter, anderson_depth, ip_ramp) < 1:
        raise ValueError("n_iter, anderson_depth and ip_ramp must be >= 1")
    if not (np.isfinite(tol) and tol > 0.0):
        raise ValueError("tol must be finite and > 0")
    nz, nr = z_np.size, r_np.size
    if psi_init is not None and tuple(np.shape(psi_init)) != (nz, nr):
        raise ValueError(f"psi_init shape {np.shape(psi_init)} does not match grid ({nz}, {nr})")

    psi_coil = vacuum_field_si(R_grid, Z_grid, coil_R, coil_Z, coil_I, mu0)
    coil_wall = psi_coil.reshape(-1)[wall_idx]
    x0 = (psi_coil if psi_init is None else jnp.asarray(psi_init)).reshape(-1)

    runner = _make_runner(
        nz,
        nr,
        float(r_np[0]),
        d_r,
        d_z,
        int(n_iter),
        int(anderson_depth),
        float(mixing),
        int(ip_ramp),
        float(cutoff_width),
        float(tol),
        float(mu0),
        bool(use_mg_preconditioner),
    )
    return runner(
        coil_I,
        pprime_vals,
        ffprime_vals,
        R_grid,
        Z_grid,
        psin_knots,
        jnp.asarray(float(ip_target)),
        response_matrix,
        wall_idx,
        source_idx,
        coil_wall,
        x0,
    )
