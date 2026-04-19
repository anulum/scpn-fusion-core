# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Constrained Safe Reinforcement Learning
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class SafetyConstraint:
    """Named constraint: cost_fn(obs, act, next_obs) must stay below limit."""

    name: str
    cost_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float]
    limit: float


class ConstrainedGymTokamakEnv:
    """Wrapper to compute and report constraint costs."""

    def __init__(self, base_env: Any, constraints: list[SafetyConstraint]):
        self.base_env = base_env
        self.constraints = constraints
        self.n_constraints = len(constraints)

        self.action_space = base_env.action_space
        self.observation_space = base_env.observation_space

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset base environment and cache initial observation."""
        obs, info = self.base_env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step base env and append constraint costs to info dict."""
        obs, reward, terminated, truncated, info = self.base_env.step(action)

        costs = []
        for c in self.constraints:
            c_val = c.cost_fn(self._last_obs, action, obs)
            costs.append(c_val)

        info["constraint_costs"] = costs
        self._last_obs = obs

        return obs, reward, terminated, truncated, info


class LagrangianPPO:
    """PPO augmented with Lagrangian multipliers for constrained RL."""

    def __init__(self, env: ConstrainedGymTokamakEnv, lambda_lr: float = 0.01, gamma: float = 0.99):
        self.env = env
        self.n_constraints = env.n_constraints
        self.lambdas = np.zeros(self.n_constraints)
        self.lambda_lr = lambda_lr
        self.gamma = gamma

        self.trained = False

    def _augmented_reward(self, reward: float, costs: list[float]) -> float:
        """r_aug = r - sum(lambda_i * c_i)."""
        penalty = sum(lam * c for lam, c in zip(self.lambdas, costs))
        return float(reward - penalty)

    def update_lambdas(self, episode_costs: list[float]) -> None:
        """Dual gradient ascent: lambda_i <- max(0, lambda_i + lr*(C_i - d_i))."""
        for i, c in enumerate(episode_costs):
            limit = self.env.constraints[i].limit
            grad = c - limit
            self.lambdas[i] = max(0.0, self.lambdas[i] + self.lambda_lr * grad)

    def train(self, total_timesteps: int) -> None:
        """Mock training loop — collects rollouts and updates lambdas."""
        current_step = 0
        while current_step < total_timesteps:
            obs, info = self.env.reset()
            done = False
            ep_costs = [0.0] * self.n_constraints

            steps = 0
            while not done and steps < 100:
                action = self.env.action_space.sample()
                obs, reward, term, trunc, info = self.env.step(action)
                done = term or trunc

                costs = info.get("constraint_costs", [0.0] * self.n_constraints)
                for i in range(self.n_constraints):
                    ep_costs[i] += costs[i]

                current_step += 1
                steps += 1

            self.update_lambdas(ep_costs)

        self.trained = True

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return action for given observation. Currently samples randomly."""
        return np.asarray(self.env.action_space.sample())


def q95_cost_fn(obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray) -> float:
    q95 = next_obs[2]
    return float(max(0.0, 2.0 - q95))


def beta_n_cost_fn(obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray) -> float:
    beta_N = next_obs[1]
    return float(max(0.0, beta_N - 3.5))


def ip_cost_fn(obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray) -> float:
    Ip = next_obs[0]
    return float(max(0.0, -Ip))


def default_safety_constraints() -> list[SafetyConstraint]:
    return [
        SafetyConstraint("q95_lower_bound", q95_cost_fn, limit=0.0),
        SafetyConstraint("beta_n_upper_bound", beta_n_cost_fn, limit=0.0),
        SafetyConstraint("ip_positive", ip_cost_fn, limit=0.0),
    ]
