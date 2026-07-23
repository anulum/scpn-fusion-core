# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Checkpoint capture for the compiled predictive-equilibrium loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, cast

import jax
import jax.numpy as jnp

_LoopState = tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]
_LoopTransition = tuple[
    _LoopState,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]
_TracedState = tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
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
_TraceRunnerOutput = tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
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


@dataclass(frozen=True)
class CompiledPredictiveTrace:
    """Frozen checkpoints captured inside the compiled predictive solve.

    Every array is a direct output of the same :func:`jax.lax.while_loop` that
    produces ``equilibrium``. Unreached requested checkpoints have
    ``recorded=False`` and zero-filled payload rows; consumers must filter them
    before interpretation.

    Attributes
    ----------
    equilibrium
        Final accepted equilibrium, shape ``(NZ, NR)`` [Wb].
    iteration_count
        Number of outer iterations executed.
    checkpoint_indices
        Requested zero-based outer-iteration indices.
    recorded
        Boolean mask identifying checkpoints reached before loop termination.
    psi_before
        State presented to the coupled map at each checkpoint [Wb].
    fixed_point_residual
        Coupled-map residual ``G(psi_before) - psi_before`` [Wb].
    psi_after
        State accepted by Anderson or the break-before-update stop [Wb].
    ip_now
        Plasma current active at each checkpoint [A].
    separatrix_refinement
        Soft-to-saddle homotopy fraction active at each checkpoint.
    converged
        Whether the checkpoint triggered the break-before-update stop.
    terminal_iteration_index
        Zero-based index of the final transition executed.
    terminal_psi_before
        State presented to the final coupled-map transition [Wb].
    terminal_fixed_point_residual
        Final coupled-map residual [Wb].
    terminal_psi_after
        State accepted by the final transition [Wb].
    terminal_ip_now
        Plasma current active on the final transition [A].
    terminal_separatrix_refinement
        Homotopy fraction active on the final transition.
    terminal_converged
        Whether the final transition triggered the convergence stop.
    """

    equilibrium: jnp.ndarray
    iteration_count: int
    checkpoint_indices: tuple[int, ...]
    recorded: jnp.ndarray
    psi_before: jnp.ndarray
    fixed_point_residual: jnp.ndarray
    psi_after: jnp.ndarray
    ip_now: jnp.ndarray
    separatrix_refinement: jnp.ndarray
    converged: jnp.ndarray
    terminal_iteration_index: int
    terminal_psi_before: jnp.ndarray
    terminal_fixed_point_residual: jnp.ndarray
    terminal_psi_after: jnp.ndarray
    terminal_ip_now: float
    terminal_separatrix_refinement: float
    terminal_converged: bool


def validate_trace_request(
    *,
    trace_iteration_indices: tuple[int, ...],
    n_iter: int,
    return_iterations: bool,
    return_trace: bool,
) -> None:
    """Reject ambiguous or out-of-range compiled-trace requests."""
    if return_iterations and return_trace:
        raise ValueError("return_iterations and return_trace are mutually exclusive")
    if not isinstance(trace_iteration_indices, tuple) or any(
        isinstance(index, bool) or not isinstance(index, int) for index in trace_iteration_indices
    ):
        raise ValueError("trace_iteration_indices must be a tuple of integers")
    if bool(trace_iteration_indices) != return_trace:
        raise ValueError("trace_iteration_indices and return_trace must be enabled together")
    if trace_iteration_indices and (
        tuple(sorted(set(trace_iteration_indices))) != trace_iteration_indices
        or trace_iteration_indices[0] < 0
        or trace_iteration_indices[-1] >= n_iter
    ):
        raise ValueError("trace_iteration_indices must be unique, increasing, and within n_iter")


def run_checkpointed_while_loop(
    *,
    state0: _LoopState,
    condition: Callable[[_LoopState], jnp.ndarray],
    advance: Callable[[_LoopState], _LoopTransition],
    trace_iteration_indices: tuple[int, ...],
    shape: tuple[int, int],
) -> _TraceRunnerOutput:
    """Run one compiled loop while capturing only the requested iterations."""
    trace_count = len(trace_iteration_indices)
    if trace_count < 1:
        raise ValueError("checkpointed loop requires at least one trace index")
    trace_indices = jnp.asarray(trace_iteration_indices, dtype=jnp.int32)
    n_flat = shape[0] * shape[1]
    x0 = state0[1]

    def traced_condition(state: _TracedState) -> jnp.ndarray:
        return condition(state[:6])

    def traced_body(state: _TracedState) -> _TracedState:
        base = state[:6]
        (
            recorded,
            psi_before,
            residual,
            psi_after,
            ip_trace,
            refinement_trace,
            converged_trace,
            _terminal_iteration,
            _terminal_psi_before,
            _terminal_residual,
            _terminal_psi_after,
            _terminal_ip,
            _terminal_refinement,
            _terminal_converged,
        ) = state[6:]
        iteration = base[0]
        (
            next_base,
            x_before,
            fixed_point_residual,
            accepted,
            ip_now,
            refinement,
            done_now,
        ) = advance(base)
        matches = trace_indices == iteration
        slot = jnp.argmax(matches)
        should_record = jnp.any(matches)
        recorded = recorded.at[slot].set(recorded[slot] | should_record)
        psi_before = psi_before.at[slot].set(jnp.where(should_record, x_before, psi_before[slot]))
        residual = residual.at[slot].set(
            jnp.where(should_record, fixed_point_residual, residual[slot])
        )
        psi_after = psi_after.at[slot].set(jnp.where(should_record, accepted, psi_after[slot]))
        ip_trace = ip_trace.at[slot].set(jnp.where(should_record, ip_now, ip_trace[slot]))
        refinement_trace = refinement_trace.at[slot].set(
            jnp.where(should_record, refinement, refinement_trace[slot])
        )
        converged_trace = converged_trace.at[slot].set(
            converged_trace[slot] | (should_record & done_now)
        )
        return (
            *next_base,
            recorded,
            psi_before,
            residual,
            psi_after,
            ip_trace,
            refinement_trace,
            converged_trace,
            iteration,
            x_before,
            fixed_point_residual,
            accepted,
            ip_now,
            refinement,
            done_now,
        )

    traced_state0: _TracedState = (
        *state0,
        jnp.zeros((trace_count,), dtype=bool),
        jnp.zeros((trace_count, n_flat), dtype=x0.dtype),
        jnp.zeros((trace_count, n_flat), dtype=x0.dtype),
        jnp.zeros((trace_count, n_flat), dtype=x0.dtype),
        jnp.zeros((trace_count,), dtype=x0.dtype),
        jnp.zeros((trace_count,), dtype=x0.dtype),
        jnp.zeros((trace_count,), dtype=bool),
        jnp.asarray(-1),
        jnp.zeros((n_flat,), dtype=x0.dtype),
        jnp.zeros((n_flat,), dtype=x0.dtype),
        jnp.zeros((n_flat,), dtype=x0.dtype),
        jnp.asarray(0.0, dtype=x0.dtype),
        jnp.asarray(0.0, dtype=x0.dtype),
        jnp.asarray(False),
    )
    traced_final = jax.lax.while_loop(traced_condition, traced_body, traced_state0)
    (
        iteration_count,
        equilibrium,
        _f_history,
        _x_history,
        _history_count,
        _done,
        recorded,
        psi_before,
        residual,
        psi_after,
        ip_trace,
        refinement_trace,
        converged_trace,
        terminal_iteration,
        terminal_psi_before,
        terminal_residual,
        terminal_psi_after,
        terminal_ip,
        terminal_refinement,
        terminal_converged,
    ) = traced_final
    return (
        equilibrium.reshape(shape),
        iteration_count,
        recorded,
        psi_before.reshape((trace_count, *shape)),
        residual.reshape((trace_count, *shape)),
        psi_after.reshape((trace_count, *shape)),
        ip_trace,
        refinement_trace,
        converged_trace,
        terminal_iteration,
        terminal_psi_before.reshape(shape),
        terminal_residual.reshape(shape),
        terminal_psi_after.reshape(shape),
        terminal_ip,
        terminal_refinement,
        terminal_converged,
    )


def build_compiled_predictive_trace(
    result: object,
    *,
    checkpoint_indices: tuple[int, ...],
) -> CompiledPredictiveTrace:
    """Convert one traced runner result into its immutable public contract."""
    (
        equilibrium,
        iteration_count,
        recorded,
        psi_before,
        fixed_point_residual,
        psi_after,
        ip_now,
        separatrix_refinement,
        converged,
        terminal_iteration_index,
        terminal_psi_before,
        terminal_fixed_point_residual,
        terminal_psi_after,
        terminal_ip_now,
        terminal_separatrix_refinement,
        terminal_converged,
    ) = cast(_TraceRunnerOutput, result)
    return CompiledPredictiveTrace(
        equilibrium=equilibrium,
        iteration_count=int(iteration_count),
        checkpoint_indices=checkpoint_indices,
        recorded=recorded,
        psi_before=psi_before,
        fixed_point_residual=fixed_point_residual,
        psi_after=psi_after,
        ip_now=ip_now,
        separatrix_refinement=separatrix_refinement,
        converged=converged,
        terminal_iteration_index=int(terminal_iteration_index),
        terminal_psi_before=terminal_psi_before,
        terminal_fixed_point_residual=terminal_fixed_point_residual,
        terminal_psi_after=terminal_psi_after,
        terminal_ip_now=float(terminal_ip_now),
        terminal_separatrix_refinement=float(terminal_separatrix_refinement),
        terminal_converged=bool(terminal_converged),
    )
