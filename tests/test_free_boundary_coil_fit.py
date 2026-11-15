# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — tests of the inverse coil-current fit
"""The coil-current fit meets symmetric targets exactly, descends, honours limits and fails closed.

The case is an up-down symmetric diverted coilset on a 33x33 grid. Forward solves start
from a solution of the predictive solver at its default tolerance, because a cold start
at the fit's 1e-9 tolerance needs more than the 300-iteration budget on this case.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scpn_fusion.core.free_boundary_coil_fit import (
    CoilFitProblem,
    IsofluxTargets,
    fit_coil_currents,
    forward,
    interpolate_bicubic,
    require_float64,
)
from scpn_fusion.core.jax_free_boundary_predictive import (
    build_response_matrix,
    solve_predictive_equilibrium,
)

_UPDATE = cast(Callable[[str, bool], None], jax.config.update)
_READ = cast(Callable[[str], object], jax.config.read)
_UPDATE("jax_enable_x64", True)

_R = np.linspace(1.0, 2.5, 33)
_Z = np.linspace(-1.4, 1.4, 33)
_COIL_R = np.array([1.2, 2.3, 1.2, 2.3, 1.6, 1.6])
_COIL_Z = np.array([0.9, 0.9, -0.9, -0.9, 1.3, -1.3])
_COIL_I = np.full(6, -3.0e5)
_PSIN = np.linspace(0.0, 1.0, 6)
_PPRIME = np.array([-8.0e4, -6.0e4, -4.0e4, -2.0e4, -0.7e4, 0.0])
_FFPRIME = np.array([-1.2, -0.9, -0.6, -0.3, -0.1, 0.0])
_IP = 1.0e6
_CONTROL = np.array([True, True, False, False, False, False])


def _targets(*, symmetric: bool) -> IsofluxTargets:
    z = [0.6, -0.6, 0.85, -0.85] if symmetric else [0.6, -0.5, 0.85, -0.7]
    return IsofluxTargets(np.array([1.9, 1.9, 1.55, 1.55]), np.array(z), np.array([]), np.array([]))


@pytest.fixture(scope="module")
def warm_problem() -> tuple[CoilFitProblem, jax.Array]:
    m, b, s = build_response_matrix(jnp.asarray(_R), jnp.asarray(_Z))
    psi0 = solve_predictive_equilibrium(
        jnp.asarray(_COIL_I),
        jnp.asarray(_PPRIME),
        jnp.asarray(_FFPRIME),
        jnp.asarray(_R),
        jnp.asarray(_Z),
        jnp.asarray(_COIL_R),
        jnp.asarray(_COIL_Z),
        jnp.asarray(_PSIN),
        _IP,
        m,
        b,
        s,
        n_iter=150,
    )
    problem = CoilFitProblem(
        _R,
        _Z,
        _COIL_R,
        _COIL_Z,
        np.eye(6),
        _PSIN,
        _PPRIME,
        _FFPRIME,
        _IP,
        m,
        b,
        s,
        _targets(symmetric=True),
        _CONTROL,
        _COIL_I,
        np.zeros(2),
    )
    return problem, psi0


def _quadratic(rr: jax.Array, zz: jax.Array) -> jax.Array:
    return 1.0 + 2.0 * rr + 3.0 * zz + 0.5 * rr**2 - 0.25 * zz**2 + 0.7 * rr * zz


def test_bicubic_interpolation_reproduces_quadratic_fields() -> None:
    r, z = jnp.asarray(_R), jnp.asarray(_Z)
    grid = _quadratic(r[None, :], z[:, None])
    for rp, zp in [(1.33, 0.21), (2.01, -0.77), (1.7, 1.0)]:
        point_r, point_z = jnp.asarray(rp), jnp.asarray(zp)
        value = float(interpolate_bicubic(grid, r, z, point_r, point_z))
        assert value == pytest.approx(float(_quadratic(point_r, point_z)), abs=1e-12)


def test_targets_count_pairs_and_refuse_bad_points() -> None:
    targets = IsofluxTargets(np.ones(4) * 1.5, np.arange(4.0), np.array([1.4]), np.array([0.0]))
    assert targets.pairs.tolist() == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    assert targets.residual_count == 6 + 2
    with pytest.raises(ValueError, match="at least 2"):
        IsofluxTargets(np.array([1.5]), np.array([0.0]), np.array([]), np.array([]))
    with pytest.raises(ValueError, match="R > 0"):
        IsofluxTargets(np.array([1.5, -1.0]), np.zeros(2), np.array([]), np.array([]))
    with pytest.raises(ValueError, match="as many R as Z"):
        IsofluxTargets(np.ones(2), np.zeros(2), np.array([1.0]), np.array([]))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"control": np.zeros(6, dtype=bool)}, "at least one"),
        ({"tikhonov": np.zeros(3)}, "one >= 0 per control"),
        ({"lower": np.zeros(3)}, "lower needs one entry"),
        ({"lower": np.zeros(2), "upper": -np.ones(2)}, "must not exceed"),
        ({"inner_iterations": 0}, "inner_iterations"),
        ({"filament_r": np.ones(5)}, "filament coordinates"),
    ],
)
def test_problem_refuses_inconsistent_settings(
    warm_problem: tuple[CoilFitProblem, jax.Array], changes: dict[str, Any], message: str
) -> None:
    problem, _psi0 = warm_problem
    with pytest.raises(ValueError, match=message):
        replace(problem, **changes)


def test_mirrored_isoflux_pairs_have_zero_residual_on_a_symmetric_flux_map(
    warm_problem: tuple[CoilFitProblem, jax.Array],
) -> None:
    """With symmetric coils, currents and flux map, mirrored points share one flux value.

    The flux map is symmetrised explicitly, because the exactly symmetric double-null
    state is degenerate for the solver and does not settle within its iteration budget.
    """
    problem, psi0 = warm_problem
    psi_symmetric = 0.5 * (psi0 + psi0[::-1, :])

    residual = problem.residual(psi_symmetric, problem.circuit_currents(_COIL_I[_CONTROL]))

    pair_list = [list(pair) for pair in problem.targets.pairs]
    mirrored = [pair_list.index([0, 1]), pair_list.index([2, 3])]
    assert np.max(np.abs(residual[mirrored])) < 1e-12
    assert np.max(np.abs(residual)) > 1e-4


def test_fit_descends_and_keeps_currents_within_limits(
    warm_problem: tuple[CoilFitProblem, jax.Array],
) -> None:
    """From a perturbed start the objective never rises, and the limits hold."""
    base, psi0 = warm_problem
    lower, upper = np.full(2, -3.3e5), np.full(2, -2.7e5)
    problem = replace(base, targets=_targets(symmetric=False), lower=lower, upper=upper)

    result = fit_coil_currents(problem, _COIL_I[_CONTROL] * 1.05, psi0, max_outer=2)

    history = np.asarray(result.objective_history)
    assert np.all(np.diff(history) <= 0.0)
    assert result.objective <= history[0]
    assert np.all(result.controls >= lower) and np.all(result.controls <= upper)
    assert result.stop_reason in {
        "objective_converged",
        "no_descent_step",
        "stalled",
        "max_outer_iterations",
    }
    assert len(result.jacobian_singular_values) == 2
    assert result.psi is not None


def test_fit_refuses_a_start_that_does_not_converge(
    warm_problem: tuple[CoilFitProblem, jax.Array],
) -> None:
    """A forward solve that does not converge is never used: the fit stops and says why.

    The same holds when the continuation reference itself does not converge: no step is
    walked from it.
    """
    base, psi0 = warm_problem
    problem = replace(base, inner_iterations=1)

    for reference in (None, _COIL_I[_CONTROL]):
        result = fit_coil_currents(
            problem, _COIL_I[_CONTROL] * 1.2, psi0, continuation_reference=reference
        )

        assert result.stop_reason == "start_not_converged"
        assert result.outer_iterations == 0
        assert result.continuation_steps == 0
        assert result.psi is None


def test_fitting_refuses_float32_jax(warm_problem: tuple[CoilFitProblem, jax.Array]) -> None:
    problem, psi0 = warm_problem
    previous = bool(_READ("jax_enable_x64"))
    _UPDATE("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="float64"):
            require_float64()
        with pytest.raises(RuntimeError, match="float64"):
            fit_coil_currents(problem, _COIL_I[_CONTROL], psi0)
    finally:
        _UPDATE("jax_enable_x64", previous)


def test_fit_refuses_inconsistent_arguments(warm_problem: tuple[CoilFitProblem, jax.Array]) -> None:
    problem, psi0 = warm_problem
    with pytest.raises(ValueError, match="max_outer and stall_iterations"):
        fit_coil_currents(problem, _COIL_I[_CONTROL], psi0, max_outer=0)
    with pytest.raises(ValueError, match="one entry per controlled circuit"):
        fit_coil_currents(problem, _COIL_I[:3], psi0)


@pytest.mark.parametrize(
    ("settings", "reason"),
    [
        ({"objective_relative_tolerance": 0.9}, "objective_converged"),
        ({"stall_relative_decrease": 1.0, "stall_iterations": 1}, "stalled"),
    ],
)
def test_fit_stops_on_its_declared_criteria(
    warm_problem: tuple[CoilFitProblem, jax.Array], settings: dict[str, Any], reason: str
) -> None:
    """Loose stopping criteria end the fit after its first accepted step, with that reason."""
    base, psi0 = warm_problem
    problem = replace(base, targets=_targets(symmetric=False))

    result = fit_coil_currents(problem, _COIL_I[_CONTROL] * 1.05, psi0, max_outer=3, **settings)

    assert result.stop_reason == reason
    assert result.outer_iterations == 1
    assert result.objective < result.objective_history[0]


def test_fit_refuses_a_jacobian_built_from_unconverged_solves(
    warm_problem: tuple[CoilFitProblem, jax.Array],
) -> None:
    """A difference step of 3x the current reverses its sign; once such a solve fails, the fit stops."""
    base, psi0 = warm_problem
    problem = replace(base, targets=_targets(symmetric=False), inner_iterations=40)

    result = fit_coil_currents(
        problem, _COIL_I[_CONTROL] * 1.05, psi0, finite_difference_relative_step=3.0
    )

    assert result.stop_reason == "jacobian_solve_not_converged"
    # The last accepted state is kept, with its converged flux map.
    assert result.objective == result.objective_history[-1]
    assert result.psi is not None


def test_fit_walks_to_a_start_that_does_not_converge_directly(
    warm_problem: tuple[CoilFitProblem, jax.Array],
) -> None:
    """A far start is walked to in converged steps from a converged reference.

    The reference flux map is converged first. With a 12-iteration budget the far start
    (1.5x the coil currents) does not converge from it directly, while the walk from the
    1.05x reference does, in steps of at most a tenth of the way, halved when one fails.
    """
    base, psi0 = warm_problem
    targets = _targets(symmetric=False)
    far, near = _COIL_I[_CONTROL] * 1.5, _COIL_I[_CONTROL] * 1.05
    reference = forward(replace(base, targets=targets), near, psi0)
    assert reference.converged
    problem = replace(base, targets=targets, inner_iterations=12)
    assert not forward(problem, far, reference.psi).converged

    result = fit_coil_currents(
        problem, far, reference.psi, continuation_reference=near, max_outer=1, stall_iterations=1
    )

    assert result.continuation_steps >= 1
    assert result.stop_reason != "start_not_converged"
