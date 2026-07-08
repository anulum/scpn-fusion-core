# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Controller Tuning
"""
Automated controller gain tuning using Bayesian optimisation.

Optimises PID and H-infinity parameters against Gymnasium environments
to minimise tracking error and maximise stability.
"""

from __future__ import annotations

import logging
from math import isfinite
from typing import Any

import numpy as np


try:
    import optuna

    HAS_OPTUNA = True  # pragma: no cover - optional optuna hyperparameter engine
except ImportError:
    HAS_OPTUNA = False

logger = logging.getLogger(__name__)

DEFAULT_PID_GAINS = {"Kp": 1.0, "Ki": 0.1, "Kd": 0.05}
DEFAULT_HINF_PARAMS = {"gamma": 1.1, "bandwidth": 0.5}


def _require_positive_int(name: str, value: int) -> None:
    """Reject non-positive integer tuning controls."""
    if value < 1:
        raise ValueError(f"{name} must be at least 1")


def _require_pid_environment(env: Any) -> None:
    """Validate that a PID tuning target exposes reset and step methods."""
    missing = [name for name in ("reset", "step") if not callable(getattr(env, name, None))]
    if missing:
        joined = ", ".join(missing)
        raise TypeError(f"PID tuning environment must provide callable {joined} method(s)")


def _reset_observation(env: Any) -> Any:
    """Return the observation component from Gym or Gymnasium reset output."""
    result = env.reset()
    if isinstance(result, tuple):
        if not result:
            raise ValueError("environment reset returned an empty tuple")
        return result[0]
    return result


def _tracking_error(obs: Any) -> float:
    """Extract a finite scalar tracking error from the leading observation value."""
    try:
        error = float(obs[0])
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("environment observations must expose tracking error at index 0") from exc
    if not isfinite(error):
        raise ValueError("environment tracking error must be finite")
    return error


def _step_environment(env: Any, action: Any) -> tuple[Any, bool]:
    """Step a Gym or Gymnasium environment and normalize its done flag."""
    result = env.step(action)
    if not isinstance(result, tuple):
        raise ValueError("environment step must return a tuple")
    if len(result) == 5:
        obs, _reward, terminated, truncated, _info = result
        return obs, bool(terminated or truncated)
    if len(result) == 4:
        obs, _reward, done, _info = result
        return obs, bool(done)
    raise ValueError("environment step must return a Gym or Gymnasium step tuple")


def _action_bounds(env: Any) -> tuple[float | None, float | None]:
    """Return scalar action bounds when the environment exposes them."""
    action_space = getattr(env, "action_space", None)
    if action_space is None:
        return None, None

    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    try:
        lower = float(np.asarray(low, dtype=float).flat[0]) if low is not None else None
        upper = float(np.asarray(high, dtype=float).flat[0]) if high is not None else None
    except (TypeError, ValueError) as exc:
        raise ValueError("scalar PID action bounds must be numeric") from exc

    if lower is not None and upper is not None and lower >= upper:
        raise ValueError("PID action lower bound must be less than upper bound")
    return lower, upper


def _format_pid_action(env: Any, command: float) -> Any:
    """Clip and shape a scalar PID command for the environment action space."""
    lower, upper = _action_bounds(env)
    if lower is not None or upper is not None:
        low = -np.inf if lower is None else lower
        high = np.inf if upper is None else upper
        command = float(np.clip(command, low, high))

    action_space = getattr(env, "action_space", None)
    shape = getattr(action_space, "shape", None)
    if shape is None or tuple(shape) == ():
        return command
    if tuple(shape) == (1,):
        return np.asarray([command], dtype=float)
    raise ValueError("tune_pid supports scalar or one-element action spaces")


def _pid_rollout_score(
    env: Any,
    *,
    kp: float,
    ki: float,
    kd: float,
    n_episodes: int,
    max_steps: int,
) -> float:
    """Score PID gains by averaged integral absolute tracking error."""
    total_iae = 0.0

    for _ in range(n_episodes):
        obs = _reset_observation(env)
        previous_error = _tracking_error(obs)
        integral_error = 0.0
        done = False
        steps = 0

        while not done and steps < max_steps:
            error = _tracking_error(obs)
            integral_error += error
            derivative_error = 0.0 if steps == 0 else error - previous_error
            command = kp * error + ki * integral_error + kd * derivative_error
            action = _format_pid_action(env, command)
            obs, done = _step_environment(env, action)
            total_iae += abs(error)
            previous_error = error
            steps += 1

        if steps == max_steps and not done:
            total_iae += abs(previous_error) * max_steps

    return total_iae / n_episodes


def tune_pid(
    env: Any,
    n_trials: int = 50,
    *,
    n_episodes: int = 5,
    max_steps: int = 500,
) -> dict[str, float]:
    """Tune PID gains (Kp, Ki, Kd) using Optuna.

    Parameters
    ----------
    env : gymnasium.Env
        The tokamak control environment.
    n_trials : int
        Number of optimisation trials.

    Returns
    -------
    dict — Tuned gains.
    """
    _require_positive_int("n_trials", n_trials)
    _require_positive_int("n_episodes", n_episodes)
    _require_positive_int("max_steps", max_steps)

    if not HAS_OPTUNA:
        logger.warning("Optuna not installed; returning default gains.")
        return dict(DEFAULT_PID_GAINS)

    _require_pid_environment(env)

    def objective(trial: optuna.Trial) -> float:
        kp = trial.suggest_float("Kp", 0.1, 10.0, log=True)
        ki = trial.suggest_float("Ki", 0.01, 1.0, log=True)
        kd = trial.suggest_float("Kd", 0.01, 1.0, log=True)
        return _pid_rollout_score(
            env,
            kp=kp,
            ki=ki,
            kd=kd,
            n_episodes=n_episodes,
            max_steps=max_steps,
        )

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return dict(study.best_params)


def tune_hinf(plant: dict[str, Any], n_trials: int = 50) -> dict[str, float]:
    """Tune H-infinity parameters (gamma, bandwidth) using Optuna."""
    _require_positive_int("n_trials", n_trials)

    if not HAS_OPTUNA:
        return dict(DEFAULT_HINF_PARAMS)

    if not isinstance(plant, dict):
        raise TypeError("plant must be a dictionary of H-infinity tuning targets")

    target_gamma = float(plant.get("target_gamma", DEFAULT_HINF_PARAMS["gamma"]))
    target_bandwidth = float(plant.get("target_bandwidth", DEFAULT_HINF_PARAMS["bandwidth"]))
    if target_gamma <= 1.0 or target_bandwidth <= 0.0:
        raise ValueError("H-infinity target gamma must exceed 1 and bandwidth must be positive")

    def objective(trial: optuna.Trial) -> float:
        gamma = trial.suggest_float("gamma", 1.01, 2.0)
        bandwidth = trial.suggest_float("bandwidth", 0.05, 5.0, log=True)
        return float(abs(gamma - target_gamma) + abs(bandwidth - target_bandwidth))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return dict(study.best_params)
