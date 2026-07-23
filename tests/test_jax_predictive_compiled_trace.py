# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Real-solver contract tests for compiled predictive-loop checkpoints."""

from __future__ import annotations

from typing import Callable, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scpn_fusion.core.jax_free_boundary_predictive import build_response_matrix
from scpn_fusion.core.jax_predictive_checkpoint_trace import CompiledPredictiveTrace
from scpn_fusion.core.jax_predictive_forward_compiled import solve_predictive_equilibrium_compiled

_jax_config_update = cast(Callable[[str, bool], None], jax.config.update)
_jax_config_update("jax_enable_x64", True)

_R = jnp.linspace(1.0, 2.5, 17)
_Z = jnp.linspace(-1.4, 1.4, 17)
_COIL_R = jnp.array([1.2, 2.3, 1.2, 2.3, 1.6, 1.5])
_COIL_Z = jnp.array([0.9, 0.9, -0.9, -0.9, 1.3, -1.35])
_COIL_I = jnp.array([-3.0e5, -3.0e5, -3.0e5, -3.0e5, -1.0e5, -6.0e5])
_PSIN = jnp.linspace(0.0, 1.0, 6)
_PPRIME = jnp.array([-8.0e4, -6.0e4, -4.0e4, -2.0e4, -0.7e4, 0.0])
_FFPRIME = jnp.array([-1.2, -0.9, -0.6, -0.3, -0.1, 0.0])
_IP = 1.0e6

Response: TypeAlias = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]


@pytest.fixture(scope="module")
def response() -> Response:
    """Build the production von Hagenow response on the bounded test grid."""
    return build_response_matrix(_R, _Z)


def _solve(
    response: Response,
    *,
    psi_init: jnp.ndarray | None = None,
    n_iter: int = 4,
    ip_ramp: int = 2,
    tol: float = 1.0e-30,
    use_mg_preconditioner: bool = False,
    trace_iteration_indices: tuple[int, ...] = (),
    return_iterations: bool = False,
    return_trace: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, int] | CompiledPredictiveTrace:
    """Call the public compiled solver with the real coupled input surface."""
    matrix, wall, source = response
    return solve_predictive_equilibrium_compiled(
        _COIL_I,
        _PPRIME,
        _FFPRIME,
        _R,
        _Z,
        _COIL_R,
        _COIL_Z,
        _PSIN,
        _IP,
        matrix,
        wall,
        source,
        psi_init=psi_init,
        n_iter=n_iter,
        ip_ramp=ip_ramp,
        tol=tol,
        use_mg_preconditioner=use_mg_preconditioner,
        trace_iteration_indices=trace_iteration_indices,
        return_iterations=return_iterations,
        return_trace=return_trace,
    )


def test_compiled_trace_preserves_and_exposes_the_exact_loop(response: Response) -> None:
    """Checkpoint state is continuous and leaves the compiled result unchanged."""
    baseline = cast(jnp.ndarray, _solve(response, use_mg_preconditioner=True))
    trace = _solve(
        response,
        use_mg_preconditioner=True,
        trace_iteration_indices=(0, 1, 2, 3),
        return_trace=True,
    )
    assert isinstance(trace, CompiledPredictiveTrace)
    trace.equilibrium.block_until_ready()

    assert np.array_equal(np.asarray(trace.equilibrium), np.asarray(baseline))
    assert trace.iteration_count == 4
    assert trace.checkpoint_indices == (0, 1, 2, 3)
    assert np.array_equal(np.asarray(trace.recorded), np.ones(4, dtype=np.bool_))
    assert trace.psi_before.shape == (4, 17, 17)
    assert trace.fixed_point_residual.shape == trace.psi_before.shape
    assert trace.psi_after.shape == trace.psi_before.shape
    assert np.array_equal(
        np.asarray(trace.psi_before[1:]),
        np.asarray(trace.psi_after[:-1]),
    )
    assert np.array_equal(np.asarray(trace.psi_after[-1]), np.asarray(trace.equilibrium))
    assert np.allclose(np.asarray(trace.ip_now), [_IP / 2.0, _IP, _IP, _IP])
    assert np.array_equal(np.asarray(trace.separatrix_refinement), np.zeros(4))
    assert not bool(jnp.any(trace.converged))
    assert bool(jnp.all(jnp.isfinite(trace.fixed_point_residual)))


def test_compiled_trace_records_break_before_update(response: Response) -> None:
    """A converged checkpoint records the unchanged pre-update accepted state."""
    seed = cast(jnp.ndarray, _solve(response))
    trace = _solve(
        response,
        psi_init=seed,
        n_iter=4,
        ip_ramp=1,
        tol=1.0e9,
        trace_iteration_indices=(1, 3),
        return_trace=True,
    )
    assert isinstance(trace, CompiledPredictiveTrace)
    trace.equilibrium.block_until_ready()

    assert trace.iteration_count == 2
    assert np.array_equal(np.asarray(trace.recorded), [True, False])
    assert bool(trace.converged[0])
    assert float(trace.ip_now[0]) == pytest.approx(_IP)
    assert float(trace.separatrix_refinement[0]) == 1.0
    assert np.array_equal(np.asarray(trace.psi_before[0]), np.asarray(trace.psi_after[0]))
    assert np.array_equal(np.asarray(trace.equilibrium), np.asarray(trace.psi_before[0]))
    assert not bool(trace.converged[1])
    assert np.array_equal(np.asarray(trace.psi_before[1]), np.zeros((17, 17)))
    assert np.array_equal(np.asarray(trace.fixed_point_residual[1]), np.zeros((17, 17)))
    assert np.array_equal(np.asarray(trace.psi_after[1]), np.zeros((17, 17)))


@pytest.mark.parametrize(
    ("indices", "return_iterations", "return_trace", "match"),
    [
        ((), False, True, "enabled together"),
        ((0,), False, False, "enabled together"),
        ((1, 0), False, True, "unique, increasing"),
        ((0, 0), False, True, "unique, increasing"),
        ((4,), False, True, "within n_iter"),
        ((0,), True, True, "mutually exclusive"),
    ],
)
def test_compiled_trace_guards_fail_closed(
    response: Response,
    indices: tuple[int, ...],
    return_iterations: bool,
    return_trace: bool,
    match: str,
) -> None:
    """Invalid trace requests fail before compiling or executing the solver."""
    with pytest.raises(ValueError, match=match):
        _solve(
            response,
            trace_iteration_indices=indices,
            return_iterations=return_iterations,
            return_trace=return_trace,
        )
