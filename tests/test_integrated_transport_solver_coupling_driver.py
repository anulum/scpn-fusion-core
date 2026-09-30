# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Transport Driver Step Contracts
"""Steady-state driver refusal, continuation and public evolution parity."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_fusion.core.integrated_transport_solver import (
    AdaptiveTimeController,
    IntegratedTransportSolver,
    TransportSolver,
)


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """Use the maintained ITER case with a bounded spatial regression grid."""
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "validation/iter_config.json").read_text())
    config["grid_resolution"] = [20, 20]
    path = tmp_path / "iter_transport.json"
    path.write_text(json.dumps(config))
    return path


def _solver(path: Path, solver_type: type[TransportSolver] = TransportSolver) -> TransportSolver:
    solver = solver_type(path, nr=16, multi_ion=False)
    solver.transport_backend = "reduced_multichannel"
    solver.Ti = 4.0 * (1.0 - solver.rho**2) + 0.2
    solver.Te = solver.Ti.copy()
    solver.update_transport_model(20.0)
    return solver


def _assert_result_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(actual[name], value)
        else:
            assert actual[name] == expected[name]


@pytest.mark.parametrize("solver_type", [TransportSolver, IntegratedTransportSolver])
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("n_steps", [0, -1, -10])
def test_nonpositive_steps_refused_before_mutation(
    config_path: Path, solver_type: type[TransportSolver], adaptive: bool, n_steps: int
) -> None:
    """Both public exports refuse an empty run and retain the next valid trajectory."""
    solver = _solver(config_path, solver_type)
    reference = _solver(config_path, solver_type)
    before = dict(vars(solver))
    arrays = {name: value.copy() for name, value in before.items() if isinstance(value, np.ndarray)}
    containers = {
        name: deepcopy(value) for name, value in before.items() if isinstance(value, (dict, list))
    }

    with pytest.raises(ValueError, match="^n_steps must be positive$"):
        solver.run_to_steady_state(20.0, n_steps=n_steps, dt=0.002, adaptive=adaptive)

    assert vars(solver).keys() == before.keys()
    for name, value in before.items():
        assert getattr(solver, name) is value
    for name, value in arrays.items():
        np.testing.assert_array_equal(getattr(solver, name), value)
    for name, value in containers.items():
        assert getattr(solver, name) == value
    actual = solver.run_to_steady_state(20.0, n_steps=1, dt=0.002, adaptive=adaptive)
    expected = reference.run_to_steady_state(20.0, n_steps=1, dt=0.002, adaptive=adaptive)
    _assert_result_equal(actual, expected)


@pytest.mark.parametrize("n_steps", [1, 2])
def test_positive_fixed_steps_match_public_evolution(config_path: Path, n_steps: int) -> None:
    """A positive fixed run executes the same coefficient and evolution sequence."""
    solver = _solver(config_path)
    reference = _solver(config_path)
    expected_avg = expected_core = 0.0
    for _ in range(n_steps):
        reference.update_transport_model(20.0)
        expected_avg, expected_core = reference.evolve_profiles(0.002, 20.0)

    actual = solver.run_to_steady_state(20.0, n_steps=n_steps, dt=0.002)

    assert actual["n_steps"] == n_steps
    assert actual["T_avg"] == expected_avg
    assert actual["T_core"] == expected_core
    assert actual["tau_e"] == reference.compute_confinement_time(20.0)
    np.testing.assert_array_equal(actual["Ti_profile"], reference.Ti)
    np.testing.assert_array_equal(actual["ne_profile"], reference.ne)
    assert not np.shares_memory(actual["Ti_profile"], solver.Ti)


@pytest.mark.parametrize("n_steps", [1, 2])
def test_positive_adaptive_steps_match_public_trials(config_path: Path, n_steps: int) -> None:
    """Positive adaptive runs preserve each public trial and PI history entry."""
    solver = _solver(config_path)
    reference = _solver(config_path)
    controller = AdaptiveTimeController(dt_init=0.002, tol=1e-3)
    for _ in range(n_steps):
        reference.update_transport_model(20.0)
        error = controller.estimate_error(reference, 20.0)
        controller.adapt_dt(error)

    actual = solver.run_to_steady_state(20.0, n_steps=n_steps, dt=0.002, adaptive=True)

    assert actual["n_steps"] == n_steps
    assert actual["T_avg"] == float(np.mean(reference.Ti))
    assert actual["T_core"] == float(reference.Ti[0])
    assert actual["dt_final"] == controller.dt
    assert actual["dt_history"] == controller.dt_history
    assert actual["error_history"] == controller.error_history
    assert actual["trial_difference_history"] == controller.trial_difference_history
    np.testing.assert_array_equal(actual["Ti_profile"], reference.Ti)
    np.testing.assert_array_equal(actual["ne_profile"], reference.ne)


@pytest.mark.parametrize("n_steps", [0, -1])
def test_self_consistent_ignores_unused_step_count(config_path: Path, n_steps: int) -> None:
    """The coupled mode retains its independent inner and outer iteration counts."""
    solver = _solver(config_path)
    reference = _solver(config_path)
    expected = reference.run_self_consistent(20.0, n_inner=1, n_outer=1, dt=0.002)

    actual = solver.run_to_steady_state(
        20.0, n_steps=n_steps, dt=0.002, self_consistent=True, sc_n_inner=1, sc_n_outer=1
    )

    _assert_result_equal(actual, expected)
    np.testing.assert_array_equal(solver.Psi, reference.Psi)
    np.testing.assert_array_equal(solver.J_phi, reference.J_phi)
