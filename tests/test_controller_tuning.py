# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Controller Tuning Tests
from __future__ import annotations

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
