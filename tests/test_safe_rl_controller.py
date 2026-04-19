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


def test_mock_training_loop():
    base_env = MockEnv()
    constraints = default_safety_constraints()
    env = ConstrainedGymTokamakEnv(base_env, constraints)

    ppo = LagrangianPPO(env)
    ppo.train(total_timesteps=50)

    assert ppo.trained
    assert ppo.lambdas[0] > 0.0
