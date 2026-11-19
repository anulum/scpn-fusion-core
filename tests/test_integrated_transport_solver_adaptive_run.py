# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Adaptive Run Stop Reasons and Diagnostics
"""Minimum-step exhaustion, stop reasons and histories kept across a refusal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import scpn_fusion.core.integrated_transport_solver_adaptive as adaptive_mod
from scpn_fusion.core.integrated_transport_solver import PhysicsError, TransportSolver

P_AUX_MW = 20.0
DT_S = 0.002


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """Use the maintained ITER case with a bounded spatial regression grid."""
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "validation/iter_config.json").read_text())
    config["grid_resolution"] = [20, 20]
    path = tmp_path / "iter_transport.json"
    path.write_text(json.dumps(config))
    return path


def _solver(path: Path) -> TransportSolver:
    """Build the bounded ITER solver with peaked, equal ion and electron profiles."""
    solver = TransportSolver(path, nr=16, multi_ion=False)
    solver.transport_backend = "reduced_multichannel"
    solver.Ti = 4.0 * (1.0 - solver.rho**2) + 0.2
    solver.Te = solver.Ti.copy()
    solver.update_transport_model(P_AUX_MW)
    return solver


def _reference_steps(
    path: Path, n_steps: int, tol: float
) -> tuple[TransportSolver, adaptive_mod.AdaptiveTimeController, int]:
    """Repeat the public controller sequence by hand, stopping as the driver must."""
    solver = _solver(path)
    controller = adaptive_mod.AdaptiveTimeController(dt_init=DT_S, tol=tol)
    steps = 0
    while steps < n_steps:
        solver.update_transport_model(P_AUX_MW)
        controller.adapt_dt(controller.estimate_error(solver, P_AUX_MW))
        steps += 1
        if controller.minimum_step_exhausted:
            break
    return solver, controller, steps


def _assert_histories(record: dict[str, Any], controller: Any) -> None:
    """Require the record's timestep and histories to equal the controller's."""
    assert record["dt_final"] == controller.dt
    assert record["dt_history"] == controller.dt_history
    assert record["error_history"] == controller.error_history
    assert record["trial_difference_history"] == controller.trial_difference_history


def test_controller_starts_unexhausted() -> None:
    """A new controller has measured nothing, so nothing is exhausted."""
    assert adaptive_mod.AdaptiveTimeController().minimum_step_exhausted is False


@pytest.mark.parametrize(
    ("dt_init", "error", "exhausted"),
    [
        (0.001, 2e-3, True),
        (0.0005, 2e-3, True),
        (0.001, 1e-3, False),
        (0.001, 5e-4, False),
        (0.002, 2e-3, False),
    ],
)
def test_exhaustion_needs_a_minimum_step_above_tolerance(
    dt_init: float, error: float, exhausted: bool
) -> None:
    """Only a step at or below ``dt_min`` whose error exceeds ``tol`` is exhausted."""
    controller = adaptive_mod.AdaptiveTimeController(dt_init=dt_init, dt_min=0.001, tol=1e-3)

    controller.adapt_dt(error)

    assert controller.minimum_step_exhausted is exhausted


def test_exhaustion_is_recomputed_on_every_update() -> None:
    """A later step within tolerance clears the flag set by an earlier one."""
    controller = adaptive_mod.AdaptiveTimeController(dt_init=0.001, dt_min=0.001, tol=1e-3)
    controller.adapt_dt(2e-3)
    assert controller.minimum_step_exhausted is True

    assert controller.dt == controller.dt_min
    controller.adapt_dt(5e-4)

    assert controller.minimum_step_exhausted is False


def test_diagnostics_are_copies() -> None:
    """Changing a returned record never changes the controller histories."""
    controller = adaptive_mod.AdaptiveTimeController(dt_init=0.01, dt_min=0.001, tol=1e-3)
    controller.adapt_dt(2e-3)
    controller.trial_difference_history.append({"ti_l2_kev": 1.0})
    record = controller.diagnostics()

    record["dt_history"].append(9.0)
    record["error_history"].append(9.0)
    record["trial_difference_history"][0]["ti_l2_kev"] = 9.0

    assert record["dt_final"] == controller.dt
    assert controller.dt_history == [0.01]
    assert controller.error_history == [2e-3]
    assert controller.trial_difference_history == [{"ti_l2_kev": 1.0}]


@pytest.mark.parametrize("n_steps", [0, -1])
def test_run_refuses_nonpositive_steps_before_mutation(config_path: Path, n_steps: int) -> None:
    """A direct call refuses an empty run and records nothing on the solver."""
    solver = _solver(config_path)
    before = dict(vars(solver))

    with pytest.raises(ValueError, match="^n_steps must be positive$"):
        adaptive_mod.run_adaptive_steps(solver, P_AUX_MW, n_steps, DT_S, 1e-3)

    assert vars(solver).keys() == before.keys()
    assert not hasattr(solver, "adaptive_run_diagnostics")


def test_completed_run_reports_its_stop_reason_and_diagnostics(config_path: Path) -> None:
    """A run within tolerance executes every step and records the same histories."""
    solver = _solver(config_path)
    reference, controller, steps = _reference_steps(config_path, 2, 1e-3)

    result = solver.run_to_steady_state(P_AUX_MW, n_steps=2, dt=DT_S, adaptive=True)

    assert steps == 2
    assert result["n_steps"] == 2
    assert result["stop_reason"] == adaptive_mod.STOP_COMPLETED == "completed"
    _assert_histories(result, controller)
    np.testing.assert_array_equal(result["Ti_profile"], reference.Ti)
    diagnostics = solver.adaptive_run_diagnostics
    assert diagnostics["steps_completed"] == 2
    assert diagnostics["stop_reason"] == "completed"
    _assert_histories(diagnostics, controller)
    assert diagnostics["dt_history"] is not result["dt_history"]


def test_minimum_step_exhaustion_stops_the_run(config_path: Path) -> None:
    """An unreachable tolerance shrinks the step to ``dt_min`` and the run stops there."""
    solver = _solver(config_path)
    reference, controller, steps = _reference_steps(config_path, 10, 1e-30)

    result = solver.run_to_steady_state(P_AUX_MW, n_steps=10, dt=DT_S, adaptive=True, tol=1e-30)

    assert 1 < steps < 10
    assert result["n_steps"] == steps
    assert result["stop_reason"] == adaptive_mod.STOP_MINIMUM_STEP_EXHAUSTED
    assert result["stop_reason"] == "minimum_step_exhausted"
    assert result["dt_history"][-1] == controller.dt_min
    assert all(dt > controller.dt_min for dt in result["dt_history"][:-1])
    _assert_histories(result, controller)
    np.testing.assert_array_equal(result["Ti_profile"], reference.Ti)
    assert result["T_avg"] == float(np.mean(reference.Ti))
    assert result["T_core"] == float(reference.Ti[0])
    assert solver.adaptive_run_diagnostics["steps_completed"] == steps
    assert solver.adaptive_run_diagnostics["stop_reason"] == "minimum_step_exhausted"


def test_refusal_keeps_the_histories_of_completed_steps(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real recovery refusal in the second step propagates and keeps step one's record.

    The healthy first step needs one density recovery, so the budget is one; three
    negative Ti points exceed it in the second step.
    """
    solver = _solver(config_path)
    _, controller, _ = _reference_steps(config_path, 1, 1e-3)
    update = solver.update_transport_model
    calls: list[float] = []

    def corrupt_second_step(p_aux: float) -> None:
        """Update the model, then make the second step's Ti negative at three points."""
        update(p_aux)
        calls.append(p_aux)
        if len(calls) == 2:
            solver.Ti[:3] = -1.0

    monkeypatch.setattr(solver, "update_transport_model", corrupt_second_step)

    with pytest.raises(PhysicsError, match="recovery"):
        solver.run_to_steady_state(
            P_AUX_MW,
            n_steps=3,
            dt=DT_S,
            adaptive=True,
            enforce_numerical_recovery=True,
            max_numerical_recoveries=1,
        )

    diagnostics = solver.adaptive_run_diagnostics
    assert len(calls) == 2
    assert diagnostics["steps_completed"] == 1
    assert diagnostics["stop_reason"] == adaptive_mod.STOP_RAISED == "raised"
    assert diagnostics["exception"].startswith("PhysicsError: ")
    assert "recovery" in diagnostics["exception"]
    _assert_histories(diagnostics, controller)
    np.testing.assert_array_equal(solver.Ti[:3], -1.0)


def test_final_result_refusal_reports_raised_with_histories(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real confinement-time refusal after the last step leaves ``raised``, not a success.

    After the run's last half step the volume element is given a wrong shape, so the
    final ``compute_confinement_time`` raises its own profile/geometry PhysicsError.
    """
    solver = _solver(config_path)
    _, controller, _ = _reference_steps(config_path, 2, 1e-3)
    evolve = solver.evolve_profiles
    calls: list[int] = []

    def evolve_then_break_geometry(*args: Any, **kwargs: Any) -> Any:
        """Evolve; after the sixth evolution (step two's last half step) truncate dV."""
        result = evolve(*args, **kwargs)
        calls.append(1)
        if len(calls) == 6:
            solver._dV_cache = np.asarray(solver._rho_volume_element())[:-1]
        return result

    monkeypatch.setattr(solver, "evolve_profiles", evolve_then_break_geometry)

    with pytest.raises(PhysicsError, match="Profile and geometry shape mismatch"):
        solver.run_to_steady_state(P_AUX_MW, n_steps=2, dt=DT_S, adaptive=True)

    diagnostics = solver.adaptive_run_diagnostics
    assert len(calls) == 6
    assert diagnostics["stop_reason"] == "raised"
    assert diagnostics["steps_completed"] == 2
    assert diagnostics["exception"].startswith("PhysicsError: Profile and geometry shape mismatch")
    _assert_histories(diagnostics, controller)


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("n_steps", [1.5, float("nan"), float("inf")])
def test_non_integer_step_counts_are_refused_before_mutation(
    config_path: Path, adaptive: bool, n_steps: float
) -> None:
    """Fractional, NaN and infinite counts are refused like ``range`` does, before mutation."""
    solver = _solver(config_path)
    before = dict(vars(solver))
    arrays = {name: value.copy() for name, value in before.items() if isinstance(value, np.ndarray)}

    count: Any = n_steps  # a deliberate non-integer, outside the declared int type
    with pytest.raises(TypeError, match="cannot be interpreted as an integer"):
        solver.run_to_steady_state(P_AUX_MW, n_steps=count, dt=DT_S, adaptive=adaptive)

    assert vars(solver).keys() == before.keys()
    for name, value in arrays.items():
        np.testing.assert_array_equal(getattr(solver, name), value)
    assert not hasattr(solver, "adaptive_run_diagnostics")


def test_direct_call_refuses_a_fractional_count(config_path: Path) -> None:
    """The adaptive helper itself applies the integer contract, not only its caller."""
    solver = _solver(config_path)
    fractional: Any = 1.5

    with pytest.raises(TypeError, match="cannot be interpreted as an integer"):
        adaptive_mod.run_adaptive_steps(solver, P_AUX_MW, fractional, DT_S, 1e-3)

    assert not hasattr(solver, "adaptive_run_diagnostics")


def test_integer_like_step_counts_still_run(config_path: Path) -> None:
    """Index-compatible counts such as numpy integers keep working."""
    solver = _solver(config_path)

    count: Any = np.int64(1)  # index-compatible at run time, outside the declared int type
    result = solver.run_to_steady_state(P_AUX_MW, n_steps=count, dt=DT_S, adaptive=True)

    assert result["n_steps"] == 1
    assert result["stop_reason"] == "completed"
