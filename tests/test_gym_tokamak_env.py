# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Gymnasium Tokamak Env Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────

import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from scpn_fusion.control.gym_tokamak_env import register

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
    
    assert reward < -5.0 # Should include the -10 penalty
    assert terminated or info["disrupted"]
