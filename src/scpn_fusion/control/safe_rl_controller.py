# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Constrained Safe Reinforcement Learning
"""Constrained reinforcement-learning wrappers and tokamak safety costs.

The module provides a constrained proximal-policy-optimisation (PPO) controller
with a Lagrangian primal-dual update: the primal step ascends the clipped PPO
surrogate on a :class:`~scpn_fusion.control.constrained_policy.LinearGaussianPolicy`,
while the dual step raises the constraint multipliers on any violated safety
cost. Actions are sampled from the policy (not the action space), so training
genuinely improves the augmented return. Every source of randomness flows
through a seeded RNG, keeping the controller auditable and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.control.constrained_policy import LinearGaussianPolicy, PolicyGradient

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass
class SafetyConstraint:
    """Named constraint: cost_fn(obs, act, next_obs) must stay below limit."""

    name: str
    cost_fn: Callable[[FloatArray, FloatArray, FloatArray], float]
    limit: float


class ConstrainedGymTokamakEnv:
    """Wrapper to compute and report constraint costs."""

    def __init__(self, base_env: Any, constraints: list[SafetyConstraint]):
        """Wrap a Gym-like environment with named safety constraints."""
        self.base_env = base_env
        self.constraints = constraints
        self.n_constraints = len(constraints)

        self.action_space = base_env.action_space
        self.observation_space = base_env.observation_space

    def reset(self, **kwargs: Any) -> tuple[FloatArray, dict[str, Any]]:
        """Reset base environment and cache initial observation."""
        obs, info = self.base_env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action: FloatArray) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        """Step base env and append constraint costs to info dict."""
        obs, reward, terminated, truncated, info = self.base_env.step(action)

        costs = []
        for c in self.constraints:
            c_val = c.cost_fn(self._last_obs, action, obs)
            costs.append(c_val)

        info["constraint_costs"] = costs
        self._last_obs = obs

        return obs, reward, terminated, truncated, info


@dataclass
class _Transition:
    """One collected step: observation, action, and its behaviour log-prob."""

    obs: FloatArray
    action: FloatArray
    log_prob_old: float
    augmented_reward: float


class LagrangianPPO:
    """Clipped-surrogate PPO with a Lagrangian primal-dual constraint update.

    The primal step ascends the clipped PPO objective on a linear-Gaussian
    policy using Monte-Carlo return-to-go advantages; the dual step raises the
    Lagrange multipliers on any constraint whose episode cost exceeds its limit.
    The augmented reward ``r - sum_i lambda_i c_i`` couples the two.
    """

    def __init__(
        self,
        env: ConstrainedGymTokamakEnv,
        lambda_lr: float = 0.01,
        gamma: float = 0.99,
        *,
        policy_lr: float = 0.05,
        clip_epsilon: float = 0.2,
        n_epochs: int = 4,
        batch_episodes: int = 16,
        log_std: float = -0.5,
        max_episode_steps: int = 100,
        seed: int = 0,
    ):
        """Initialise constrained-policy state and primal-dual hyperparameters."""
        self.env = env
        self.n_constraints = env.n_constraints
        self.lambdas = np.zeros(self.n_constraints)
        self.lambda_lr = lambda_lr
        self.gamma = gamma
        self.policy_lr = float(policy_lr)
        self.clip_epsilon = float(clip_epsilon)
        self.n_epochs = int(n_epochs)
        self.batch_episodes = max(1, int(batch_episodes))
        self.log_std = float(log_std)
        self.max_episode_steps = int(max_episode_steps)
        self._rng = np.random.default_rng(seed)
        self.policy: LinearGaussianPolicy | None = None
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

    def _ensure_policy(self, obs: FloatArray) -> LinearGaussianPolicy:
        """Lazily build the policy sized from the first observation and action."""
        if self.policy is None:
            obs_dim = int(np.asarray(obs, dtype=np.float64).ravel().size)
            act_dim = int(np.asarray(self.env.action_space.sample()).ravel().size)
            self.policy = LinearGaussianPolicy(obs_dim, act_dim, log_std=self.log_std)
        return self.policy

    def _action_bounds(self) -> tuple[FloatArray, FloatArray] | None:
        """Return ``(low, high)`` action bounds if the action space exposes them."""
        low = getattr(self.env.action_space, "low", None)
        high = getattr(self.env.action_space, "high", None)
        if low is None or high is None:
            return None
        return (
            np.asarray(low, dtype=np.float64).ravel(),
            np.asarray(high, dtype=np.float64).ravel(),
        )

    def _clip_action(self, action: FloatArray) -> FloatArray:
        """Clip a raw action to the action-space bounds when available."""
        bounds = self._action_bounds()
        if bounds is None:
            return action
        return np.clip(action, bounds[0], bounds[1])

    def _collect_episode(self) -> tuple[list[_Transition], list[float]]:
        """Roll out one episode under the stochastic policy."""
        obs, _ = self.env.reset()
        policy = self._ensure_policy(obs)
        transitions: list[_Transition] = []
        ep_costs = [0.0] * self.n_constraints
        done = False
        steps = 0
        while not done and steps < self.max_episode_steps:
            obs_vec = np.asarray(obs, dtype=np.float64).ravel()
            action = policy.sample(obs_vec, self._rng)
            log_prob_old = policy.log_prob(obs_vec, action)
            clipped = self._clip_action(action)
            next_obs, reward, term, trunc, info = self.env.step(clipped)
            costs = info.get("constraint_costs", [0.0] * self.n_constraints)
            for i in range(self.n_constraints):
                ep_costs[i] += costs[i]
            transitions.append(
                _Transition(
                    obs=obs_vec,
                    action=np.asarray(action, dtype=np.float64).ravel(),
                    log_prob_old=log_prob_old,
                    augmented_reward=self._augmented_reward(float(reward), list(costs)),
                )
            )
            obs = next_obs
            done = bool(term or trunc)
            steps += 1
        return transitions, ep_costs

    def _returns_to_go(self, transitions: list[_Transition]) -> FloatArray:
        """Discounted return-to-go of the augmented reward for one episode."""
        returns = np.zeros(len(transitions), dtype=np.float64)
        running = 0.0
        for t in range(len(transitions) - 1, -1, -1):
            running = transitions[t].augmented_reward + self.gamma * running
            returns[t] = running
        return returns

    def _batch_advantages(self, batch: list[list[_Transition]]) -> list[FloatArray]:
        """Per-timestep baseline across the batch, then whitened advantages.

        Subtracting the mean return-to-go *at each timestep* removes the
        time-position bias (early steps carry more remaining reward than late
        steps regardless of action quality); whitening the residuals keeps the
        policy step well-scaled.
        """
        episode_returns = [self._returns_to_go(ep) for ep in batch]
        max_len = max(len(ep) for ep in batch)
        baseline = np.zeros(max_len, dtype=np.float64)
        for t in range(max_len):
            step_returns = [r[t] for r in episode_returns if len(r) > t]
            baseline[t] = float(np.mean(step_returns))
        advantages = [returns - baseline[: len(returns)] for returns in episode_returns]
        flat = np.concatenate(advantages) if advantages else np.zeros(0)
        std = float(np.std(flat))
        if std > 1e-8:
            mean = float(np.mean(flat))
            advantages = [(adv - mean) / std for adv in advantages]
        return advantages

    def _ppo_update(
        self,
        batch: list[list[_Transition]],
        advantages: list[FloatArray],
    ) -> None:
        """Ascend the clipped PPO surrogate over every transition in the batch."""
        policy = self.policy
        assert policy is not None  # set by _collect_episode via _ensure_policy
        flat = [
            (transition, float(adv))
            for transitions, ep_adv in zip(batch, advantages)
            for transition, adv in zip(transitions, ep_adv)
        ]
        for _ in range(self.n_epochs):
            # Accumulate the mean clipped-surrogate gradient, then take one step.
            acc_w = np.zeros_like(policy.w)
            acc_b = np.zeros_like(policy.b)
            for transition, adv in flat:
                log_prob_new = policy.log_prob(transition.obs, transition.action)
                ratio = float(np.exp(log_prob_new - transition.log_prob_old))
                # Clipping zeroes a sample's contribution when the surrogate is capped.
                if adv >= 0.0 and ratio > 1.0 + self.clip_epsilon:
                    continue
                if adv < 0.0 and ratio < 1.0 - self.clip_epsilon:
                    continue
                grad = policy.grad_log_prob(transition.obs, transition.action)
                scale = ratio * adv
                acc_w += scale * grad.grad_w
                acc_b += scale * grad.grad_b
            policy.apply_gradient(
                PolicyGradient(acc_w / len(flat), acc_b / len(flat)),
                self.policy_lr,
            )

    def train(self, total_timesteps: int) -> None:
        """Primal-dual training: PPO policy ascent + Lagrangian dual update."""
        current_step = 0
        while current_step < total_timesteps:
            batch: list[list[_Transition]] = []
            batch_costs: list[list[float]] = []
            for _ in range(self.batch_episodes):
                transitions, ep_costs = self._collect_episode()
                current_step += len(transitions)
                if transitions:
                    batch.append(transitions)
                    batch_costs.append(ep_costs)
            if not batch:
                break
            advantages = self._batch_advantages(batch)
            self._ppo_update(batch, advantages)
            mean_costs = [
                float(np.mean([c[i] for c in batch_costs])) for i in range(self.n_constraints)
            ]
            self.update_lambdas(mean_costs)
        self.trained = True

    def predict(self, obs: FloatArray) -> FloatArray:
        """Return the deterministic policy action (mean) for an observation."""
        policy = self._ensure_policy(obs)
        obs_vec = np.asarray(obs, dtype=np.float64).ravel()
        return self._clip_action(policy.mean(obs_vec))


def q95_cost_fn(obs: FloatArray, act: FloatArray, next_obs: FloatArray) -> float:
    """Compute a lower-bound violation cost on edge safety factor ``q95``.

    :param obs: Previous observation; unused in this contract.
    :param act: Action taken; unused in this contract.
    :param next_obs: Post-step observation, where index 2 is assumed to hold ``q95``.
    :returns: Positive violation amount in the same unit as ``q95`` delta.
    """
    q95 = next_obs[2]
    return float(max(0.0, 2.0 - q95))


def beta_n_cost_fn(obs: FloatArray, act: FloatArray, next_obs: FloatArray) -> float:
    """Compute an upper-bound violation cost on normalized beta ``beta_N``.

    :param obs: Previous observation; unused in this contract.
    :param act: Action taken; unused in this contract.
    :param next_obs: Post-step observation, where index 1 is assumed to hold ``beta_N``.
    :returns: Positive cost when ``beta_N`` exceeds ``3.5``.
    """
    beta_N = next_obs[1]
    return float(max(0.0, beta_N - 3.5))


def ip_cost_fn(obs: FloatArray, act: FloatArray, next_obs: FloatArray) -> float:
    """Compute a lower-bound violation cost on plasma current.

    :param obs: Previous observation; unused in this contract.
    :param act: Action taken; unused in this contract.
    :param next_obs: Post-step observation, where index 0 is assumed to hold ``Ip``.
    :returns: Positive violation value when ``Ip`` is non-positive.
    """
    Ip = next_obs[0]
    return float(max(0.0, -Ip))


def default_safety_constraints() -> list[SafetyConstraint]:
    """Return the default ``q95``, ``beta_N``, and plasma-current constraints.

    :returns: A list of default :class:`SafetyConstraint` instances with zero limits.
    """
    return [
        SafetyConstraint("q95_lower_bound", q95_cost_fn, limit=0.0),
        SafetyConstraint("beta_n_upper_bound", beta_n_cost_fn, limit=0.0),
        SafetyConstraint("ip_positive", ip_cost_fn, limit=0.0),
    ]
