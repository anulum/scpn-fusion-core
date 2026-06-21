# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Controller Tuning Tests
"""Tests for the Optuna PID / H-infinity controller-tuning helpers and entry points."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scpn_fusion.control.controller_tuning as mod
from scpn_fusion.control.controller_tuning import (
    HAS_OPTUNA,
    _action_bounds,
    _format_pid_action,
    _pid_rollout_score,
    _reset_observation,
    _step_environment,
    _tracking_error,
    tune_hinf,
    tune_pid,
)


class TestResetObservation:
    """Environment-reset observation unpacking."""

    def test_plain_observation(self) -> None:
        """A non-tuple reset value is returned unchanged."""
        env = SimpleNamespace(reset=lambda: np.array([1.0]))
        np.testing.assert_array_equal(_reset_observation(env), np.array([1.0]))

    def test_tuple_observation(self) -> None:
        """A (obs, info) reset tuple yields its first element."""
        env = SimpleNamespace(reset=lambda: (np.array([2.0]), {}))
        np.testing.assert_array_equal(_reset_observation(env), np.array([2.0]))

    def test_empty_tuple_rejected(self) -> None:
        """An empty reset tuple is rejected."""
        env = SimpleNamespace(reset=lambda: ())
        with pytest.raises(ValueError, match="empty tuple"):
            _reset_observation(env)


class TestTrackingError:
    """Tracking-error extraction from observations."""

    def test_valid(self) -> None:
        """A finite leading observation component is returned as the error."""
        assert _tracking_error(np.array([0.7, 0.0])) == pytest.approx(0.7)

    def test_unindexable_rejected(self) -> None:
        """An observation without an index-0 element is rejected."""
        with pytest.raises(ValueError, match="index 0"):
            _tracking_error(object())

    def test_non_finite_rejected(self) -> None:
        """A non-finite tracking error is rejected."""
        with pytest.raises(ValueError, match="finite"):
            _tracking_error(np.array([np.inf]))


class TestStepEnvironment:
    """Gym / Gymnasium step-tuple handling."""

    def test_non_tuple_rejected(self) -> None:
        """A non-tuple step result is rejected."""
        env = SimpleNamespace(step=lambda _a: "nope")
        with pytest.raises(ValueError, match="must return a tuple"):
            _step_environment(env, 0.0)

    def test_gymnasium_five_tuple(self) -> None:
        """A five-element Gymnasium step combines terminated/truncated into done."""
        env = SimpleNamespace(step=lambda _a: (np.array([1.0]), 0.0, False, True, {}))
        obs, done = _step_environment(env, 0.0)
        assert done is True
        np.testing.assert_array_equal(obs, np.array([1.0]))

    def test_gym_four_tuple(self) -> None:
        """A four-element Gym step returns the done flag directly."""
        env = SimpleNamespace(step=lambda _a: (np.array([1.0]), 0.0, False, {}))
        _obs, done = _step_environment(env, 0.0)
        assert done is False

    def test_wrong_length_rejected(self) -> None:
        """A step tuple of unsupported length is rejected."""
        env = SimpleNamespace(step=lambda _a: (1.0, 2.0, 3.0))
        with pytest.raises(ValueError, match="Gym or Gymnasium"):
            _step_environment(env, 0.0)


class TestActionBounds:
    """Scalar action-bound extraction."""

    def test_no_action_space(self) -> None:
        """An environment without an action space yields open bounds."""
        assert _action_bounds(SimpleNamespace()) == (None, None)

    def test_numeric_bounds(self) -> None:
        """Numeric low/high arrays yield scalar bounds."""
        space = SimpleNamespace(low=np.array([-2.0]), high=np.array([2.0]))
        assert _action_bounds(SimpleNamespace(action_space=space)) == (-2.0, 2.0)

    def test_inverted_bounds_rejected(self) -> None:
        """A lower bound at or above the upper bound is rejected."""
        space = SimpleNamespace(low=np.array([2.0]), high=np.array([1.0]))
        with pytest.raises(ValueError, match="lower bound"):
            _action_bounds(SimpleNamespace(action_space=space))

    def test_non_numeric_bounds_rejected(self) -> None:
        """Non-numeric action bounds are rejected."""
        space = SimpleNamespace(low=np.array(["x"]), high=None)
        with pytest.raises(ValueError, match="numeric"):
            _action_bounds(SimpleNamespace(action_space=space))


class TestFormatPidAction:
    """PID command formatting and clipping to the action space."""

    def test_scalar_no_space(self) -> None:
        """Without an action space the command is returned as a scalar."""
        assert _format_pid_action(SimpleNamespace(), 3.0) == pytest.approx(3.0)

    def test_clipped_to_bounds(self) -> None:
        """A command outside the bounds is clipped into range."""
        space = SimpleNamespace(low=np.array([-1.0]), high=np.array([1.0]), shape=())
        assert _format_pid_action(SimpleNamespace(action_space=space), 5.0) == pytest.approx(1.0)

    def test_one_element_shape_returns_array(self) -> None:
        """A one-element action space returns a length-1 array."""
        space = SimpleNamespace(low=None, high=None, shape=(1,))
        result = _format_pid_action(SimpleNamespace(action_space=space), 0.5)
        np.testing.assert_array_equal(result, np.array([0.5]))

    def test_unsupported_shape_rejected(self) -> None:
        """A multi-element action space is rejected."""
        space = SimpleNamespace(low=None, high=None, shape=(2,))
        with pytest.raises(ValueError, match="scalar or one-element"):
            _format_pid_action(SimpleNamespace(action_space=space), 0.5)


class _NonTerminatingEnv:
    """Environment whose episode never reports done, to exercise the step cap."""

    def reset(self) -> Any:
        """Return a fixed initial observation."""
        return np.array([1.0])

    def step(self, _action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        """Return a constant observation and a never-done flag."""
        return np.array([1.0]), 0.0, False, {}


def test_pid_rollout_score_applies_step_cap_penalty() -> None:
    """A non-terminating episode incurs the max-step truncation penalty."""
    score = _pid_rollout_score(
        _NonTerminatingEnv(), kp=1.0, ki=0.0, kd=0.0, n_episodes=1, max_steps=3
    )
    assert score > 0.0


def test_tune_pid_fallback_without_optuna(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without Optuna the PID tuner returns the default gains."""
    monkeypatch.setattr(mod, "HAS_OPTUNA", False)
    gains = tune_pid(None, n_trials=1)
    assert gains == {"Kp": 1.0, "Ki": 0.1, "Kd": 0.05}


def test_tune_hinf_fallback_without_optuna(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without Optuna the H-infinity tuner returns the default parameters."""
    monkeypatch.setattr(mod, "HAS_OPTUNA", False)
    params = tune_hinf(plant={}, n_trials=1)
    assert params == {"gamma": 1.1, "bandwidth": 0.5}


def test_tune_pid_returns_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    """The PID tuner returns positive gains for every term."""
    monkeypatch.setattr(mod, "HAS_OPTUNA", False)
    result = tune_pid(env=None)
    assert isinstance(result, dict)
    for key in ("Kp", "Ki", "Kd"):
        assert key in result
        assert result[key] > 0


def test_tune_pid_validates_positive_trial_count() -> None:
    """A non-positive trial count is rejected."""
    with pytest.raises(ValueError, match="n_trials"):
        tune_pid(None, n_trials=0)


def test_tune_hinf_validates_positive_trial_count() -> None:
    """A non-positive H-infinity trial count is rejected."""
    with pytest.raises(ValueError, match="n_trials"):
        tune_hinf(plant={}, n_trials=0)


def test_tune_pid_requires_gym_like_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The PID tuner requires a reset/step-capable environment."""
    monkeypatch.setattr(mod, "HAS_OPTUNA", True)
    with pytest.raises(TypeError, match="reset"):
        tune_pid(object(), n_trials=1)


class _FixedTrial:
    """Optuna-trial stub returning fixed hyperparameter values."""

    values = {"Kp": 1.0, "Ki": 0.5, "Kd": 0.25, "gamma": 1.5, "bandwidth": 0.5}

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        """Return the pre-set value for the named hyperparameter."""
        return self.values[name]


class _Study:
    """Optuna-study stub that runs the objective once and records best params."""

    def __init__(self, recorded: dict[str, float]) -> None:
        self.best_params = recorded

    def optimize(self, objective: Any, n_trials: int) -> None:
        """Invoke the objective a single time."""
        objective(_FixedTrial())


def test_tune_pid_objective_uses_integral_and_derivative_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The PID objective applies proportional, integral, and derivative gains in order."""

    class RecordingEnv:
        """Environment recording the PID commands it receives."""

        def __init__(self) -> None:
            self.actions: list[float] = []
            self._next_errors = [0.5, 0.25, 0.125]

        def reset(self) -> Any:
            """Reset the error schedule and command log."""
            self.actions.clear()
            self._next_errors = [0.5, 0.25, 0.125]
            return np.array([1.0]), {}

        def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            """Log the command and emit the next scheduled error."""
            self.actions.append(float(action))
            error = self._next_errors.pop(0)
            terminated = not self._next_errors
            return np.array([error]), -abs(error), terminated, False, {}

    env = RecordingEnv()
    monkeypatch.setattr(mod, "HAS_OPTUNA", True)
    monkeypatch.setattr(
        mod,
        "optuna",
        SimpleNamespace(create_study=lambda direction: _Study({"Kp": 1.0, "Ki": 0.5, "Kd": 0.25})),
        raising=False,
    )

    result = tune_pid(env, n_trials=1, n_episodes=1, max_steps=10)

    assert result == {"Kp": 1.0, "Ki": 0.5, "Kd": 0.25}
    assert env.actions[0] == pytest.approx(1.5)
    assert env.actions[1] == pytest.approx(1.125)


def test_tune_hinf_objective_minimises_target_distance(monkeypatch: pytest.MonkeyPatch) -> None:
    """The H-infinity objective scores the distance to the target gamma/bandwidth."""
    monkeypatch.setattr(mod, "HAS_OPTUNA", True)
    monkeypatch.setattr(
        mod,
        "optuna",
        SimpleNamespace(create_study=lambda direction: _Study({"gamma": 1.5, "bandwidth": 0.5})),
        raising=False,
    )
    result = tune_hinf(plant={"target_gamma": 1.5, "target_bandwidth": 0.5}, n_trials=1)
    assert result == {"gamma": 1.5, "bandwidth": 0.5}


def test_tune_hinf_rejects_non_dict_plant(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-dict plant specification is rejected when Optuna is active."""
    monkeypatch.setattr(mod, "HAS_OPTUNA", True)
    with pytest.raises(TypeError, match="dictionary"):
        tune_hinf(plant=[1, 2, 3], n_trials=1)  # type: ignore[arg-type]


def test_tune_hinf_rejects_invalid_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Out-of-range target gamma or bandwidth is rejected when Optuna is active."""
    monkeypatch.setattr(mod, "HAS_OPTUNA", True)
    monkeypatch.setattr(
        mod, "optuna", SimpleNamespace(create_study=lambda direction: _Study({})), raising=False
    )
    with pytest.raises(ValueError, match="target gamma"):
        tune_hinf(plant={"target_gamma": 0.5}, n_trials=1)


@pytest.mark.skipif(not HAS_OPTUNA, reason="Optuna not installed")
def test_tune_pid_with_optuna() -> None:
    """With real Optuna the PID tuner returns positive gains end to end."""

    class MockEnv:
        """Decaying-error environment for a real Optuna run."""

        _step = 0

        def reset(self) -> Any:
            """Reset the decay counter."""
            self._step = 0
            return np.array([0.1, 0.0]), {}

        def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            """Emit an exponentially-decaying tracking error."""
            self._step += 1
            obs = np.array([0.1 * np.exp(-0.1 * self._step), 0.0])
            done = self._step >= 10
            return obs, -abs(obs[0]), done, False, {}

    result = tune_pid(MockEnv(), n_trials=3)
    assert isinstance(result, dict)
    assert result["Kp"] > 0


@pytest.mark.skipif(not HAS_OPTUNA, reason="Optuna not installed")
def test_tune_hinf_with_optuna() -> None:
    """With real Optuna the H-infinity tuner returns an in-range gamma."""
    result = tune_hinf(plant={"A": [[0, 1], [-1, 0]]}, n_trials=3)
    assert isinstance(result, dict)
    assert 1.01 <= result["gamma"] <= 2.0
