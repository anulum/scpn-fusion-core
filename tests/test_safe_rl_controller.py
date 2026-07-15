# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Constrained Safe RL Tests
from __future__ import annotations

import numpy as np

from scpn_fusion.control.safe_rl_controller import (
    ConstrainedGymTokamakEnv,
    LagrangianPPO,
    SafetyConstraint,
    default_safety_constraints,
)


class MockEnv:
    def __init__(self):
        class Space:
            def sample(self):
                return np.array([0.0])

        self.action_space = Space()
        self.observation_space = Space()
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return np.array([10.0, 2.0, 3.0]), {}

    def step(self, action):
        self.step_count += 1
        obs = np.array([10.0, 2.0, 1.5])
        reward = 1.0
        term = self.step_count >= 10
        return obs, reward, term, False, {}


class _BoxSpace:
    """Minimal Box-like action space with bounds and a seeded sampler."""

    def __init__(self, low: float, high: float, seed: int = 0) -> None:
        self.low = np.array([low], dtype=np.float64)
        self.high = np.array([high], dtype=np.float64)
        self._rng = np.random.default_rng(seed)

    def sample(self) -> np.ndarray:
        return self._rng.uniform(self.low, self.high)


class QuadraticTrackingEnv:
    """Fast analytic env: reward peaks when the scalar action hits ``target``.

    Multi-step episodes with a constant observation, so the policy must learn a
    non-trivial mean action. An optional upper action bound is enforced as a
    constraint cost, letting the Lagrangian dual demonstrably bind.
    """

    def __init__(self, target: float = 0.7, n_steps: int = 8) -> None:
        self.target = float(target)
        self.n_steps = int(n_steps)
        self.action_space = _BoxSpace(-2.0, 2.0)
        self.observation_space = _BoxSpace(-1.0, 1.0)
        self._step = 0

    def reset(self, **_: object) -> tuple[np.ndarray, dict]:
        self._step = 0
        return np.array([1.0], dtype=np.float64), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        a = float(np.asarray(action, dtype=np.float64).ravel()[0])
        reward = -((a - self.target) ** 2)
        self._step += 1
        term = self._step >= self.n_steps
        obs = np.array([1.0], dtype=np.float64)
        return obs, reward, term, False, {}


def _action_upper_bound_cost(limit: float):
    def cost_fn(_obs, act, _next_obs) -> float:
        a = float(np.asarray(act, dtype=np.float64).ravel()[0])
        return float(max(0.0, a - limit))

    return cost_fn


def _mean_deterministic_reward(ppo: LagrangianPPO, env, n_episodes: int = 20) -> float:
    """Roll out the deterministic policy and average the raw episode reward."""
    total = 0.0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = ppo.predict(np.asarray(obs, dtype=np.float64))
            obs, reward, term, trunc, _ = env.step(action)
            total += reward
            done = term or trunc
    return total / n_episodes


def test_constrained_env_wrapper():
    base_env = MockEnv()
    constraints = default_safety_constraints()
    env = ConstrainedGymTokamakEnv(base_env, constraints)

    obs, info = env.reset()
    action = np.array([0.0])
    obs_next, reward, term, trunc, info = env.step(action)

    costs = info["constraint_costs"]
    assert len(costs) == 3
    assert costs[0] == 0.5
    assert costs[1] == 0.0


def test_lagrangian_ppo_lambda_update():
    base_env = MockEnv()
    constraints = default_safety_constraints()
    env = ConstrainedGymTokamakEnv(base_env, constraints)

    ppo = LagrangianPPO(env, lambda_lr=0.1)

    ep_costs = [5.0, 0.0, 0.0]

    ppo.update_lambdas(ep_costs)

    assert ppo.lambdas[0] > 0.0
    assert ppo.lambdas[1] == 0.0
    assert ppo.lambdas[2] == 0.0


def test_augmented_reward():
    base_env = MockEnv()
    constraints = default_safety_constraints()
    env = ConstrainedGymTokamakEnv(base_env, constraints)

    ppo = LagrangianPPO(env)
    ppo.lambdas = np.array([2.0, 1.0, 0.5])

    aug = ppo._augmented_reward(10.0, [1.0, 0.0, 2.0])
    assert aug == 7.0

    ppo.lambdas = np.zeros(3)
    aug_zero = ppo._augmented_reward(10.0, [1.0, 0.0, 2.0])
    assert aug_zero == 10.0


def test_training_updates_lambdas_and_marks_trained():
    base_env = MockEnv()
    env = ConstrainedGymTokamakEnv(base_env, default_safety_constraints())

    ppo = LagrangianPPO(env, seed=1)
    ppo.train(total_timesteps=50)

    assert ppo.trained
    # q95 cost accrues every step (next_obs[2] = 1.5 < 2.0), so its multiplier rises.
    assert ppo.lambdas[0] > 0.0


def test_training_with_zero_length_episodes_still_marks_trained():
    # max_episode_steps=0 yields empty rollouts, exercising the empty-batch guard.
    env = ConstrainedGymTokamakEnv(MockEnv(), default_safety_constraints())
    ppo = LagrangianPPO(env, seed=5, max_episode_steps=0)
    ppo.train(total_timesteps=10)
    assert ppo.trained
    np.testing.assert_array_equal(ppo.lambdas, np.zeros(3))


def test_predict_returns_deterministic_mean_action():
    env = ConstrainedGymTokamakEnv(MockEnv(), default_safety_constraints())
    ppo = LagrangianPPO(env, seed=2)
    obs = np.array([10.0, 2.0, 3.0])
    action_a = ppo.predict(obs)
    action_b = ppo.predict(obs)
    assert np.asarray(action_a).shape == (1,)
    # Deterministic: identical observation yields identical action.
    np.testing.assert_array_equal(action_a, action_b)


def test_ppo_learns_to_track_unconstrained_target():
    """With a slack constraint, PPO drives the mean action toward the target."""
    base_env = QuadraticTrackingEnv(target=0.7)
    # limit well above the target, so the constraint never binds.
    constraints = [SafetyConstraint("effort", _action_upper_bound_cost(5.0), limit=10.0)]
    env = ConstrainedGymTokamakEnv(base_env, constraints)
    ppo = LagrangianPPO(env, seed=3, policy_lr=0.05)

    before = _mean_deterministic_reward(ppo, base_env)
    ppo.train(total_timesteps=6000)
    after = _mean_deterministic_reward(ppo, base_env)

    assert after > before
    learned = float(ppo.predict(np.array([1.0]))[0])
    # Moved decisively away from the zero-initialised action toward 0.7.
    assert abs(learned - 0.7) < abs(0.0 - 0.7)


def test_lagrangian_dual_binds_and_shields_the_action():
    """A binding upper-bound constraint keeps the learned action below target."""
    base_env = QuadraticTrackingEnv(target=0.7)
    constraints = [SafetyConstraint("effort", _action_upper_bound_cost(0.3), limit=0.0)]
    env = ConstrainedGymTokamakEnv(base_env, constraints)
    ppo = LagrangianPPO(env, seed=4, policy_lr=0.05, lambda_lr=0.05)

    ppo.train(total_timesteps=8000)

    assert ppo.lambdas[0] > 0.0
    learned = float(ppo.predict(np.array([1.0]))[0])
    # The dual penalty pulls the optimum below the unconstrained target of 0.7.
    assert learned < 0.7
