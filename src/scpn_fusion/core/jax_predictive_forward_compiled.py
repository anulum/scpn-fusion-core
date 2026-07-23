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

This module removes the host from the loop: the ENTIRE iteration — Ip ramp, soft-to-refined
separatrix homotopy, coupled Picard step (axis/X-point → Ip-normalised Jφ → von Hagenow wall
flux → MG-preconditioned BiCGSTAB), Anderson mixing with rolling fixed-shape history,
rank-deficiency guard and the early-stop test — runs inside a single
:func:`jax.lax.while_loop` under :func:`jax.jit`.

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
from typing import Callable, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

_RunnerOutput = (
    tuple[jnp.ndarray, jnp.ndarray]
    | tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]
)

from scpn_fusion.core.jax_free_boundary_gs import MU0_SI, vacuum_field_si
from scpn_fusion.core.jax_free_boundary_predictive import (
    DEFAULT_ANDERSON_DEPTH,
    DEFAULT_CUTOFF_WIDTH,
    DEFAULT_IP_RAMP,
    DEFAULT_MIXING,
    DEFAULT_N_ITER,
    DEFAULT_SEPARATRIX_RAMP,
    DEFAULT_SEPARATRIX_START,
    DEFAULT_TOL,
    _coupled_rhs,
    _coupled_step,
    _gs_operator_flat,
)
from scpn_fusion.core.jax_multigrid_precond import build_gs_mg_preconditioner
from scpn_fusion.core.jax_predictive_checkpoint_trace import (
    CompiledPredictiveTrace,
    _LoopState as _State,
    _LoopTransition as _Transition,
    build_compiled_predictive_trace,
    run_checkpointed_while_loop,
    validate_trace_request,
)


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
    inner_solver: str,
    inner_cycles: int,
    use_separatrix_continuation: bool,
    anderson_solver: str = "lstsq",
    trace_iteration_indices: tuple[int, ...] = (),
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
    ) -> _RunnerOutput:
        dA = jnp.asarray(d_r * d_z)

        def g_of(
            x: jnp.ndarray,
            ip_now: jnp.ndarray,
            separatrix_refinement: jnp.ndarray,
        ) -> jnp.ndarray:
            if inner_solver == "bicgstab":
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
                    separatrix_refinement,
                    cutoff_width,
                    mu0,
                    precond,
                )
            # inner_solver == "mg_richardson": warm-started MG-preconditioned Richardson —
            # each cycle is ONE matvec + ONE V-cycle (vs 2+2 per BiCGSTAB iteration). The
            # composite outer/inner fixed point is unchanged (at outer convergence the RHS
            # is stationary and the inner iteration polishes towards the exact solve); the
            # equivalence to the BiCGSTAB path is pinned by tests at span tolerance.
            rhs = _coupled_rhs(
                x,
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
            )
            assert precond is not None  # guarded in the wrapper
            y = x
            for _ in range(inner_cycles):
                y = y + precond(
                    rhs - _gs_operator_flat(y, shape, R_grid, jnp.asarray(d_r), jnp.asarray(d_z))
                )
            return y

        def cond(state: _State) -> jnp.ndarray:
            k, _x, _f_h, _x_h, _nv, done = state
            return (k < n_iter) & (~done)

        def advance(state: _State) -> _Transition:
            k, x, f_hist, x_hist, n_valid, _done = state
            ip_k = ip_target * jnp.minimum(1.0, (k + 1.0) / ramp_div)
            separatrix_refinement = (
                jnp.clip(
                    (k + 1.0 - DEFAULT_SEPARATRIX_START) / DEFAULT_SEPARATRIX_RAMP,
                    0.0,
                    1.0,
                )
                if use_separatrix_continuation
                else jnp.asarray(1.0)
            )
            f = g_of(x, ip_k, separatrix_refinement) - x
            # Break-before-update semantics: once converged (Ip fully ramped), neither the
            # history nor x are touched — the loop exits returning the pre-update iterate.
            continuation_complete = (
                k + 1 >= DEFAULT_SEPARATRIX_START + DEFAULT_SEPARATRIX_RAMP
                if use_separatrix_continuation
                else jnp.asarray(True)
            )
            done_now = (
                (k >= ip_ramp)
                & continuation_complete
                & (jnp.linalg.norm(f) <= tol * (jnp.linalg.norm(x) + 1.0))
            )
            f_new = jnp.roll(f_hist, -1, axis=0).at[-1].set(f)
            x_new = jnp.roll(x_hist, -1, axis=0).at[-1].set(x)
            nv_new = jnp.minimum(n_valid + 1, m)
            # Anderson step over the valid suffix of the rolling history. Invalid (warm-up)
            # columns are zeroed; SVD least squares assigns them zero weight, so the first
            # iteration (no valid pairs) reduces exactly to the damped Picard update.
            valid = (col_idx >= (m - nv_new))[jnp.newaxis, :]
            df = jnp.where(valid, (f_new[1:] - f_new[:-1]).T, 0.0)
            dx = jnp.where(valid, (x_new[1:] - x_new[:-1]).T, 0.0)
            if anderson_solver == "normal_eq":
                # Normal equations on the (m-1)x(m-1) Gram system with relative Tikhonov
                # regularisation — mathematically the same least squares as lstsq up to
                # conditioning (standard Anderson practice); batchable (no tall SVD).
                gram = df.T @ df
                lam = 1e-12 * jnp.trace(gram) + 1e-300
                gamma = jnp.linalg.solve(gram + lam * jnp.eye(m - 1, dtype=gram.dtype), df.T @ f)
            else:
                gamma, _res, _rank, _sv = jnp.linalg.lstsq(df, f, rcond=None)
            x_anderson = x + mixing * f - (dx + mixing * df) @ gamma
            x_fallback = x + mixing * f
            x_next = jnp.where(jnp.all(jnp.isfinite(x_anderson)), x_anderson, x_fallback)
            accepted = jnp.where(done_now, x, x_next)
            return (
                (
                    k + 1,
                    accepted,
                    jnp.where(done_now, f_hist, f_new),
                    jnp.where(done_now, x_hist, x_new),
                    jnp.where(done_now, n_valid, nv_new),
                    done_now,
                ),
                x,
                f,
                accepted,
                ip_k,
                separatrix_refinement,
                done_now,
            )

        def body(state: _State) -> _State:
            next_state, _x, _f, _accepted, _ip, _refinement, _done = advance(state)
            return next_state

        state0 = (
            jnp.asarray(0),
            x0,
            jnp.zeros((m, n_flat), dtype=x0.dtype),
            jnp.zeros((m, n_flat), dtype=x0.dtype),
            jnp.asarray(0),
            jnp.asarray(False),
        )
        if trace_iteration_indices:
            return run_checkpointed_while_loop(
                state0=state0,
                condition=cond,
                advance=advance,
                trace_iteration_indices=trace_iteration_indices,
                shape=shape,
            )
        k_final, x_final, _fh, _xh, _nv, _done = jax.lax.while_loop(cond, body, state0)
        return x_final.reshape(shape), k_final

    return cast(Callable[..., jnp.ndarray], jax.jit(run))


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
    inner_solver: str = "bicgstab",
    inner_cycles: int = 3,
    anderson_solver: str = "lstsq",
    return_iterations: bool = False,
    trace_iteration_indices: tuple[int, ...] = (),
    return_trace: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, int] | CompiledPredictiveTrace:
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

    ``inner_solver``: ``"bicgstab"`` (reference) or ``"mg_richardson"`` — warm-started
    MG-preconditioned Richardson with a fixed ``inner_cycles`` count (one matvec + one
    V-cycle per cycle, half the kernel traffic of a BiCGSTAB iteration); requires the MG
    preconditioner and reaches the same fixed point (test-pinned at span tolerance).

    ``return_trace=True`` captures only ``trace_iteration_indices`` inside the
    compiled loop and returns :class:`CompiledPredictiveTrace`. The indices
    must be a strictly increasing tuple within ``[0, n_iter)``. Tracing is
    opt-in: the normal runner is compiled without trace state or checkpoint
    writes.

    Raises
    ------
    ValueError
        On non-uniform / degenerate axes, inconsistent shapes, non-positive settings, an
        unknown ``inner_solver``, or ``mg_richardson`` without the MG preconditioner.
    """
    r_np = np.asarray(R_grid, dtype=np.float64)
    z_np = np.asarray(Z_grid, dtype=np.float64)
    d_r = _require_uniform("R_grid", r_np)
    d_z = _require_uniform("Z_grid", z_np)
    if min(n_iter, anderson_depth, ip_ramp) < 1:
        raise ValueError("n_iter, anderson_depth and ip_ramp must be >= 1")
    if not (np.isfinite(tol) and tol > 0.0):
        raise ValueError("tol must be finite and > 0")
    if inner_solver not in ("bicgstab", "mg_richardson"):
        raise ValueError(f"unknown inner_solver {inner_solver!r}")
    if inner_solver == "mg_richardson" and not use_mg_preconditioner:
        raise ValueError("inner_solver='mg_richardson' requires use_mg_preconditioner=True")
    if inner_cycles < 1:
        raise ValueError("inner_cycles must be >= 1")
    if anderson_solver not in ("lstsq", "normal_eq"):
        raise ValueError(f"unknown anderson_solver {anderson_solver!r}")
    validate_trace_request(
        trace_iteration_indices=trace_iteration_indices,
        n_iter=n_iter,
        return_iterations=return_iterations,
        return_trace=return_trace,
    )
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
        str(inner_solver),
        int(inner_cycles),
        psi_init is None,
        str(anderson_solver),
        trace_iteration_indices,
    )
    result = runner(
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
    if return_trace:
        return build_compiled_predictive_trace(
            result,
            checkpoint_indices=trace_iteration_indices,
        )
    psi, k = cast(tuple[jnp.ndarray, jnp.ndarray], result)
    if return_iterations:
        return psi, int(k)
    return psi


@lru_cache(maxsize=16)
def _make_batched_runner(
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
    inner_solver: str,
    inner_cycles: int,
    anderson_solver: str,
    shared_init: bool,
) -> Callable[..., jnp.ndarray]:
    """Build and jit the vmapped batch runner for one (geometry, settings) combination.

    The cache is the whole point: re-jitting ``jax.vmap`` per call recompiles the batched
    while-loop graph every time (~3 min at 129² on a GTX 1060 — the measured "batch cliff",
    call-count- and batch-size-bound, not solver-bound). With the runner cached, the warm
    batched solve amortises kernel-launch latency exactly as designed (measured 13–17
    ms/solve at 129² on the same card vs 34 ms single). Geometry and machine arrays are
    traced ARGUMENTS (broadcast axes), never closure captures, so one compiled runner
    serves every call with matching static settings.
    """
    runner = _make_runner(
        nz,
        nr,
        r0,
        d_r,
        d_z,
        n_iter,
        anderson_depth,
        mixing,
        ip_ramp,
        cutoff_width,
        tol,
        mu0,
        use_mg_preconditioner,
        inner_solver,
        inner_cycles,
        not shared_init,
        anderson_solver,
    )

    def one(
        ci: jnp.ndarray,
        pp: jnp.ndarray,
        ff: jnp.ndarray,
        R_grid: jnp.ndarray,
        Z_grid: jnp.ndarray,
        coil_R: jnp.ndarray,
        coil_Z: jnp.ndarray,
        psin_knots: jnp.ndarray,
        ip_arr: jnp.ndarray,
        response_matrix: jnp.ndarray,
        wall_idx: jnp.ndarray,
        source_idx: jnp.ndarray,
        x0_shared: jnp.ndarray,
    ) -> jnp.ndarray:
        psi_coil = vacuum_field_si(R_grid, Z_grid, coil_R, coil_Z, ci, mu0)
        coil_wall = psi_coil.reshape(-1)[wall_idx]
        x0 = x0_shared if shared_init else psi_coil.reshape(-1)
        psi, _k = runner(
            ci,
            pp,
            ff,
            R_grid,
            Z_grid,
            psin_knots,
            ip_arr,
            response_matrix,
            wall_idx,
            source_idx,
            coil_wall,
            x0,
        )
        return cast(jnp.ndarray, psi)

    in_axes = (0, 0, 0) + (None,) * 10
    return cast(Callable[..., jnp.ndarray], jax.jit(jax.vmap(one, in_axes=in_axes)))


def solve_predictive_equilibrium_batched(
    coil_I_batch: jnp.ndarray,
    pprime_batch: jnp.ndarray,
    ffprime_batch: jnp.ndarray,
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
    inner_solver: str = "mg_richardson",
    inner_cycles: int = 2,
    anderson_solver: str = "lstsq",
) -> jnp.ndarray:
    """Batched compiled solve — ``vmap`` over (coil currents, p′, FF′) samples.

    The MCMC/ensemble pattern of the IDA loop: many parameter samples of the SAME machine
    geometry solved simultaneously. Kernel-launch latency (the measured wall-clock driver at
    these grid sizes) is amortised across the whole batch, so the effective per-sample cost
    drops far below a single solve. All samples share ``psi_init`` (typically the current
    MAP equilibrium — the in-basin warm start; pass ``ip_ramp=1`` semantics via the shared
    warm default below) and the machine geometry; each sample carries its own
    ``(coil_I, p′, FF′)`` row. Batched defaults follow the measured production path
    (``mg_richardson``, ``inner_cycles=2``).

    Correctness: element ``i`` of the returned batch equals the single-solve result for
    sample ``i`` (test-pinned at span tolerance) — ``lax.while_loop`` under ``vmap`` runs
    until EVERY element converges, so per-element results are unaffected by batching.

    Returns ψ batch of shape ``(B, NZ, NR)``.
    """
    r_np = np.asarray(R_grid, dtype=np.float64)
    z_np = np.asarray(Z_grid, dtype=np.float64)
    d_r = _require_uniform("R_grid", r_np)
    d_z = _require_uniform("Z_grid", z_np)
    if coil_I_batch.ndim != 2 or pprime_batch.ndim != 2 or ffprime_batch.ndim != 2:
        raise ValueError("coil_I_batch, pprime_batch and ffprime_batch must be 2-D (batch, values)")
    if not (coil_I_batch.shape[0] == pprime_batch.shape[0] == ffprime_batch.shape[0]):
        raise ValueError("batch dimensions must agree across coil_I, pprime and ffprime")
    if min(n_iter, anderson_depth, ip_ramp) < 1:
        raise ValueError("n_iter, anderson_depth and ip_ramp must be >= 1")
    if not (np.isfinite(tol) and tol > 0.0):
        raise ValueError("tol must be finite and > 0")
    if inner_solver not in ("bicgstab", "mg_richardson"):
        raise ValueError(f"unknown inner_solver {inner_solver!r}")
    if inner_solver == "mg_richardson" and not use_mg_preconditioner:
        raise ValueError("inner_solver='mg_richardson' requires use_mg_preconditioner=True")
    if inner_cycles < 1:
        raise ValueError("inner_cycles must be >= 1")
    if anderson_solver not in ("lstsq", "normal_eq"):
        raise ValueError(f"unknown anderson_solver {anderson_solver!r}")
    nz, nr = z_np.size, r_np.size
    if psi_init is not None and tuple(np.shape(psi_init)) != (nz, nr):
        raise ValueError(f"psi_init shape {np.shape(psi_init)} does not match grid ({nz}, {nr})")
    batched = _make_batched_runner(
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
        str(inner_solver),
        int(inner_cycles),
        str(anderson_solver),
        psi_init is not None,
    )
    ip_arr = jnp.asarray(float(ip_target))
    x0_shared = (
        jnp.zeros(nz * nr, dtype=jnp.result_type(float))
        if psi_init is None
        else jnp.asarray(psi_init).reshape(-1)
    )
    return batched(
        coil_I_batch,
        pprime_batch,
        ffprime_batch,
        R_grid,
        Z_grid,
        coil_R,
        coil_Z,
        psin_knots,
        ip_arr,
        response_matrix,
        wall_idx,
        source_idx,
        x0_shared,
    )
