# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Gymnasium Tokamak Env Tests

import numpy as np

import pytest

gym = pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env
from scpn_fusion.control.gym_tokamak_env import TokamakEnv, register


class _AxisKernel:
    """Kernel stand-in that exposes ``find_magnetic_axis`` (the preferred path)."""

    Psi = np.zeros((4, 4), dtype=np.float64)
    R = np.linspace(4.0, 8.0, 4, dtype=np.float64)
    Z = np.linspace(-2.0, 2.0, 4, dtype=np.float64)
    cfg = {"physics": {"plasma_current_target": 15.0, "beta_scale": 2.0}}

    def find_magnetic_axis(self) -> tuple[float, float, float]:
        return 6.2, 0.1, 0.5

    def find_x_point(self, _psi: np.ndarray) -> tuple[tuple[float, float], None]:
        return (5.0, -3.0), None


def test_env_registration():
    """Verify that Tokamak-v0 is registered correctly."""
    register()
    env_ids = [spec.id for spec in gym.envs.registry.values()]
    assert "Tokamak-v0" in env_ids


def test_env_reset():
    """Verify that reset returns valid observation and info."""
    register()
    env = gym.make("Tokamak-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    assert obs.shape == (8,)
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)
    assert np.all(np.isfinite(obs))


def test_env_step():
    """Verify that step returns valid values and terminates."""
    register()
    env = gym.make("Tokamak-v0", max_steps=5)
    env.reset()

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (8,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "disrupted" in info


def test_env_compliance():
    """Verify that the environment complies with Gymnasium API standards."""
    register()
    env = gym.make("Tokamak-v0", max_steps=5)
    # This checks observation space, action space, and API behavior
    check_env(env.unwrapped)


def test_disruption_penalty():
    """Verify that disruption (large error) results in a penalty."""
    register()
    env = gym.make("Tokamak-v0")
    env.reset()

    # Large impossible action to force disruption
    action = np.array([1e9, 1e9, 1e9, 1e9], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert reward < -5.0  # Should include the -10 penalty
    assert terminated or info["disrupted"]


def test_get_obs_prefers_find_magnetic_axis_when_kernel_exposes_it():
    """When the kernel exposes ``find_magnetic_axis`` the observation uses it directly.

    The default ITER kernel lacks the method, so the env normally falls back to the
    grid-argmax estimate; injecting a kernel that provides it exercises the preferred
    branch and the axis coordinates flow straight into the observation.
    """
    env = TokamakEnv(max_steps=5)
    env.controller.kernel = _AxisKernel()
    obs = env._get_obs()

    assert obs[0] == pytest.approx(6.2)  # curr_R from find_magnetic_axis
    assert obs[1] == pytest.approx(0.1)  # curr_Z from find_magnetic_axis
    assert obs[2] == pytest.approx(15.0)  # plasma_current_target from cfg
    assert obs[6] == pytest.approx(5.0)  # x-point R
    assert obs[7] == pytest.approx(-3.0)  # x-point Z
