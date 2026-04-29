# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Controller Tuning Tests
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scpn_fusion.control.controller_tuning import HAS_OPTUNA, tune_hinf, tune_pid


def test_tune_pid_fallback_without_optuna(monkeypatch):
    import scpn_fusion.control.controller_tuning as mod

    monkeypatch.setattr(mod, "HAS_OPTUNA", False)
    gains = tune_pid(None, n_trials=1)
    assert gains == {"Kp": 1.0, "Ki": 0.1, "Kd": 0.05}


def test_tune_hinf_fallback_without_optuna(monkeypatch):
    import scpn_fusion.control.controller_tuning as mod

    monkeypatch.setattr(mod, "HAS_OPTUNA", False)
    params = tune_hinf(plant={}, n_trials=1)
    assert params == {"gamma": 1.1, "bandwidth": 0.5}


def test_tune_pid_returns_dict(monkeypatch):
    import scpn_fusion.control.controller_tuning as mod

    monkeypatch.setattr(mod, "HAS_OPTUNA", False)
    result = tune_pid(env=None)
    assert isinstance(result, dict)
    for key in ("Kp", "Ki", "Kd"):
        assert key in result
        assert result[key] > 0


def test_tune_pid_validates_positive_trial_count():
    with pytest.raises(ValueError, match="n_trials"):
        tune_pid(None, n_trials=0)


def test_tune_pid_requires_gym_like_environment(monkeypatch):
    import scpn_fusion.control.controller_tuning as mod

    monkeypatch.setattr(mod, "HAS_OPTUNA", True)
    with pytest.raises(TypeError, match="reset"):
        tune_pid(object(), n_trials=1)


def test_tune_pid_objective_uses_integral_and_derivative_terms(monkeypatch):
    import scpn_fusion.control.controller_tuning as mod

    class FixedTrial:
        values = {"Kp": 1.0, "Ki": 0.5, "Kd": 0.25}

        def suggest_float(self, name, low, high, log=False):
            return self.values[name]

    class Study:
        best_params: dict[str, float] = {}

        def optimize(self, objective, n_trials):
            assert n_trials == 1
            objective(FixedTrial())
            self.best_params = dict(FixedTrial.values)

    class RecordingEnv:
        def __init__(self):
            self.actions: list[float] = []
            self._next_errors = [0.5, 0.25, 0.125]

        def reset(self):
            self.actions.clear()
            self._next_errors = [0.5, 0.25, 0.125]
            return np.array([1.0]), {}

        def step(self, action):
            self.actions.append(float(action))
            error = self._next_errors.pop(0)
            terminated = not self._next_errors
            return np.array([error]), -abs(error), terminated, False, {}

    env = RecordingEnv()
    monkeypatch.setattr(mod, "HAS_OPTUNA", True)
    monkeypatch.setattr(
        mod,
        "optuna",
        SimpleNamespace(create_study=lambda direction: Study()),
        raising=False,
    )

    result = tune_pid(env, n_trials=1, n_episodes=1, max_steps=10)

    assert result == FixedTrial.values
    assert env.actions[0] == pytest.approx(1.5)
    assert env.actions[1] == pytest.approx(1.125)


@pytest.mark.skipif(not HAS_OPTUNA, reason="Optuna not installed")
def test_tune_pid_with_optuna():
    class MockEnv:
        _step = 0

        def reset(self):
            self._step = 0
            return np.array([0.1, 0.0]), {}

        def step(self, action):
            self._step += 1
            obs = np.array([0.1 * np.exp(-0.1 * self._step), 0.0])
            done = self._step >= 10
            return obs, -abs(obs[0]), done, False, {}

    result = tune_pid(MockEnv(), n_trials=3)
    assert isinstance(result, dict)
    assert "Kp" in result
    assert result["Kp"] > 0


@pytest.mark.skipif(not HAS_OPTUNA, reason="Optuna not installed")
def test_tune_hinf_with_optuna():
    result = tune_hinf(plant={"A": [[0, 1], [-1, 0]]}, n_trials=3)
    assert isinstance(result, dict)
    assert "gamma" in result
    assert 1.01 <= result["gamma"] <= 2.0
