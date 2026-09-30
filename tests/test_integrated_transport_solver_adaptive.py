# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Adaptive Transport Tests
"""Adaptive trials against independent public transport evolutions."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from types import GetSetDescriptorType

import numpy as np
import pytest

import scpn_fusion.core.integrated_transport_solver_adaptive as adaptive_mod
from scpn_fusion.core.integrated_transport_solver import AdaptiveTimeController as public_controller
from scpn_fusion.core.integrated_transport_solver import TransportSolver, PhysicsError
from scpn_fusion.core.integrated_transport_solver_runtime import (
    AdaptiveTimeController as runtime_controller,
)


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """Use the maintained ITER configuration on a bounded regression grid."""
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "validation/iter_config.json").read_text())
    config["grid_resolution"] = [20, 20]
    config["physics"]["impurity_transport_enabled"] = True
    config["physics"]["pedestal_mode"] = "eped"
    path = tmp_path / "transport.json"
    path.write_text(json.dumps(config))
    return path


def _solver(path: Path, multi_ion: bool, backend: str) -> TransportSolver:
    solver = TransportSolver(path, nr=16, multi_ion=multi_ion)
    solver.transport_backend = backend
    solver.Ti = 4.0 * (1.0 - solver.rho**2) + 0.2
    solver.Te = 2.0 * (1.0 - solver.rho**2) + 0.15
    solver.n_impurity[:] = 1e-6
    solver.update_transport_model(20.0)
    return solver


def _assert_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _assert_equal(a, b)
    elif hasattr(expected, "__dict__"):
        _assert_equal(vars(actual), vars(expected))
    else:
        assert actual == expected


def _values(value: Any) -> Any:
    if isinstance(value, dict):
        return {name: _values(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_values(item) for item in value]
    if hasattr(value, "__dict__"):
        return _values(vars(value))
    if type(value).__name__ == "PyFusionKernel":
        return {
            "native_identity": id(value),
            "public_properties": {
                name: _values(getattr(value, name))
                for name, descriptor in vars(type(value)).items()
                if isinstance(descriptor, GetSetDescriptorType) and not name.startswith("_")
            },
        }
    return deepcopy(value)


@pytest.mark.parametrize("power", [20.0, 50.0])
@pytest.mark.parametrize("multi_ion", [False, True])
@pytest.mark.parametrize("backend", ["reduced_multichannel", "neural_transport"])
def test_estimate_error_matches_independent_half_step_state(
    config_path: Path, multi_ion: bool, backend: str, power: float
) -> None:
    """Full-step species, closure and pedestal mutations never enter the accepted path."""
    solver = _solver(config_path, multi_ion, backend)
    full = _solver(config_path, multi_ion, backend)
    half = _solver(config_path, multi_ion, backend)
    if power > 30.0:
        full.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        solver.neoclassical_params = full.neoclassical_params
        half.neoclassical_params = full.neoclassical_params
    controller = adaptive_mod.AdaptiveTimeController(dt_init=0.002)
    original_ti = solver.Ti
    original_values = original_ti.copy()
    full.evolve_profiles(0.002, power)
    half.evolve_profiles(0.001, power)
    half.evolve_profiles(0.001, power)
    expected_error = max(float(np.linalg.norm(full.Ti - half.Ti)) / 3.0, 1e-15)

    error = controller.estimate_error(solver, P_aux=power)

    assert error == pytest.approx(expected_error, rel=1e-12)
    assert controller.trial_difference_history == [
        {
            "ti_l2_kev": float(np.linalg.norm(full.Ti - half.Ti)),
            "te_l2_kev": float(np.linalg.norm(full.Te - half.Te)),
            "ne_l2_1e19_m3": float(np.linalg.norm(full.ne - half.ne)),
        }
    ]
    for name in (
        "Ti",
        "Te",
        "ne",
        "n_D",
        "n_T",
        "n_He",
        "n_impurity",
        "chi_i",
        "chi_e",
        "D_n",
        "_Z_eff",
        "T_edge_keV",
        "_neural_transport_model",
        "pedestal_model",
        "q_profile",
        "_dV_cache",
    ):
        _assert_equal(getattr(solver, name), getattr(half, name))
    for name in vars(half):
        if name.startswith("_last_"):
            _assert_equal(getattr(solver, name), getattr(half, name))
    if power > 30.0:
        assert solver._last_pedestal_contract["used"]
        assert solver.pedestal_model is not None
    if multi_ion:
        assert not np.array_equal(solver.Te, solver.Ti)
    np.testing.assert_array_equal(original_ti, original_values)


@pytest.mark.parametrize("failed_call", [1, 2, 3])
def test_estimate_error_rolls_back_interrupted_real_trial(
    config_path: Path, monkeypatch: pytest.MonkeyPatch, failed_call: int
) -> None:
    """An interruption after any real evolution restores all original runtime objects."""
    solver = _solver(config_path, True, "neural_transport")
    before = dict(vars(solver))
    physical = {
        name: deepcopy(value)
        for name, value in before.items()
        if isinstance(value, (np.ndarray, dict, float, int, str))
        or name in {"pedestal_model", "_neural_transport_model"}
    }
    evolve = solver.evolve_profiles
    calls = 0

    def interrupted(*args: Any, **kwargs: Any) -> tuple[float, float]:
        nonlocal calls
        result = evolve(*args, **kwargs)
        calls += 1
        if calls == failed_call:
            raise RuntimeError("operator interruption")
        return result

    monkeypatch.setattr(solver, "evolve_profiles", interrupted)
    controller = adaptive_mod.AdaptiveTimeController(dt_init=0.002)
    with pytest.raises(RuntimeError, match="operator interruption"):
        controller.estimate_error(solver, P_aux=20.0)
    assert controller.trial_difference_history == []
    assert controller.dt_history == []
    assert controller.error_history == []
    assert controller.dt == 0.002
    assert set(vars(solver)) == set(before) | {"evolve_profiles"}
    for name, value in before.items():
        assert getattr(solver, name) is value
    for name, value in physical.items():
        _assert_equal(getattr(solver, name), value)


def test_estimate_error_restores_state_on_real_recovery_refusal(config_path: Path) -> None:
    """A real recovery-budget refusal is propagated without advancing species or heat."""
    solver = _solver(config_path, True, "reduced_multichannel")
    solver.Ti[0] = -1.0
    before = dict(vars(solver))
    values = _values(before)
    controller = adaptive_mod.AdaptiveTimeController(dt_init=0.002)
    with pytest.raises(PhysicsError, match="recovery"):
        controller.estimate_error(
            solver, P_aux=20.0, enforce_numerical_recovery=True, max_numerical_recoveries=0
        )
    assert vars(solver).keys() == before.keys()
    for name, value in before.items():
        assert getattr(solver, name) is value
    _assert_equal(_values(vars(solver)), values)
    assert controller.trial_difference_history == []
    assert controller.dt_history == []
    assert controller.error_history == []
    assert solver.Ti[0] == -1.0


def test_adapt_dt_respects_bounds() -> None:
    """Valid errors keep the next timestep within the configured seconds bounds."""
    atc = adaptive_mod.AdaptiveTimeController(dt_init=0.01, dt_min=0.001, dt_max=0.02, tol=1e-3)

    atc.adapt_dt(error=1e-12)
    assert 0.001 <= atc.dt <= 0.02
    atc.adapt_dt(error=1e3)
    assert 0.001 <= atc.dt <= 0.02


@pytest.mark.parametrize(
    "controller_type",
    [adaptive_mod.AdaptiveTimeController, public_controller, runtime_controller],
    ids=["adaptive", "public", "runtime"],
)
@pytest.mark.parametrize("error", [0.0, -0.0, -1.0, np.nan, np.inf, -np.inf])
def test_adapt_dt_refuses_invalid_error_without_mutation(
    controller_type: type[adaptive_mod.AdaptiveTimeController], error: float
) -> None:
    """Every public route refuses invalid error and resumes the original PI trajectory."""
    controller = controller_type()
    reference = controller_type()
    controller.adapt_dt(2e-3)
    reference.adapt_dt(2e-3)
    before = deepcopy(vars(controller))
    histories = (
        controller.dt_history,
        controller.error_history,
        controller.trial_difference_history,
    )

    with pytest.raises(ValueError, match="error must be finite and positive"):
        controller.adapt_dt(error)

    assert vars(controller) == before
    assert controller.dt_history is histories[0]
    assert controller.error_history is histories[1]
    assert controller.trial_difference_history is histories[2]
    for next_error in (5e-4, 4e-3):
        controller.adapt_dt(next_error)
        reference.adapt_dt(next_error)
        assert controller.dt == reference.dt
        assert controller.dt_history == reference.dt_history
        assert controller.error_history == reference.error_history


@pytest.mark.parametrize("error", [1e-12, 1e-3, 1e3])
def test_adapt_dt_valid_errors_match_pi_update(error: float) -> None:
    """Finite positive errors retain the PI update, factor bounds and timestep bounds."""
    controller = public_controller(dt_init=0.01, dt_min=0.001, dt_max=0.015, tol=1e-3)
    old_dt = controller.dt
    factor = 0.9 * (1e-3 / error) ** 0.35 * (1e-3 / error) ** 0.2
    expected = min(0.015, max(0.001, old_dt * max(0.1, min(2.0, factor))))

    controller.adapt_dt(error)

    assert controller.dt == pytest.approx(expected, rel=1e-14)
    assert controller.dt_history == [old_dt]
    assert controller.error_history == [error]


def test_adaptive_driver_exposes_trial_differences(config_path: Path) -> None:
    """The public adaptive driver returns accepted trial diagnostics for each step."""
    solver = _solver(config_path, True, "reduced_multichannel")
    result = solver.run_to_steady_state(P_aux=20.0, n_steps=2, dt=0.002, adaptive=True)
    history = result["trial_difference_history"]
    assert len(history) == 2
    for entry, error in zip(history, result["error_history"]):
        assert set(entry) == {"ti_l2_kev", "te_l2_kev", "ne_l2_1e19_m3"}
        assert error == max(entry["ti_l2_kev"] / 3.0, 1e-15)
        assert all(np.isfinite(value) and value >= 0 for value in entry.values())


def test_estimate_error_refuses_nonfinite_trial_difference(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt result after real evolution is refused without committing any trial."""
    solver = _solver(config_path, True, "reduced_multichannel")
    before = dict(vars(solver))
    values = _values(before)
    evolve = solver.evolve_profiles
    calls = 0

    def corrupt(*args: Any, **kwargs: Any) -> tuple[float, float]:
        nonlocal calls
        result = evolve(*args, **kwargs)
        calls += 1
        if calls == 3:
            solver.Te[0] = np.nan
        return result

    monkeypatch.setattr(solver, "evolve_profiles", corrupt)
    controller = adaptive_mod.AdaptiveTimeController(dt_init=0.002)
    with pytest.raises(FloatingPointError, match="Nonfinite full-versus-half"):
        controller.estimate_error(solver, P_aux=20.0)
    for name, value in before.items():
        assert getattr(solver, name) is value
    assert controller.trial_difference_history == []

    assert set(vars(solver)) == set(before) | {"evolve_profiles"}
    for name, value in values.items():
        _assert_equal(_values(getattr(solver, name)), value)
    assert controller.dt_history == []
    assert controller.error_history == []


@pytest.mark.parametrize("multi_ion", [False, True])
@pytest.mark.parametrize("backend", ["reduced_multichannel", "neural_transport"])
def test_real_evolution_mutations_remain_in_trial_scope(
    config_path: Path, multi_ion: bool, backend: str
) -> None:
    """Every runtime attribute change must be covered by adaptive state isolation."""
    solver = _solver(config_path, multi_ion, backend)
    solver.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
    before = _values(vars(solver))
    solver.evolve_profiles(0.002, 50.0)
    changed = set(before) ^ set(vars(solver))
    for name in before.keys() & vars(solver).keys():
        try:
            _assert_equal(_values(getattr(solver, name)), before[name])
        except AssertionError:
            changed.add(name)
    assert {"Ti", "Te", "pedestal_model"} <= changed
    assert all(
        name in adaptive_mod._TRIAL_FIELDS or name.startswith("_last_") for name in changed
    ), changed


@pytest.mark.parametrize("failed_call", [2, 3])
def test_half_trial_real_recovery_refusal_restores_values(
    config_path: Path, monkeypatch: pytest.MonkeyPatch, failed_call: int
) -> None:
    """Corrupt half-trial input triggers the real runtime budget and full rollback."""
    solver = _solver(config_path, True, "reduced_multichannel")
    before = dict(vars(solver))
    values = _values(before)
    evolve = solver.evolve_profiles
    calls = 0

    def damaged_input(*args: Any, **kwargs: Any) -> tuple[float, float]:
        nonlocal calls
        calls += 1
        if calls == failed_call:
            solver.Ti[:] = -1.0
        return evolve(*args, **kwargs)

    monkeypatch.setattr(solver, "evolve_profiles", damaged_input)
    controller = adaptive_mod.AdaptiveTimeController(dt_init=0.002)
    with pytest.raises(PhysicsError, match="Numerical recovery budget exceeded"):
        controller.estimate_error(
            solver, P_aux=20.0, enforce_numerical_recovery=True, max_numerical_recoveries=8
        )
    assert calls == failed_call
    assert set(vars(solver)) == set(before) | {"evolve_profiles"}
    for name, value in before.items():
        assert getattr(solver, name) is value
        _assert_equal(_values(getattr(solver, name)), values[name])
    assert controller.trial_difference_history == []
    assert controller.dt_history == []
    assert controller.error_history == []
