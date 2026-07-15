# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Shielded Tokamak Environment Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control.shielded_tokamak_env import (
    NONNEGATIVE,
    ActionChannelRule,
    ShieldedTokamakEnv,
    _lyapunov_theta,
    default_obs_to_safety_state,
    position_error_lyapunov_v,
)
from scpn_fusion.phase.lyapunov_guard import LyapunovGuard


def _obs(
    r: float = 6.0,
    z: float = 0.0,
    ip: float = 5.0,
    beta: float = 1.0,
    dr: float = 0.0,
    dz: float = 0.0,
) -> np.ndarray:
    return np.array([r, z, ip, beta, dr, dz, 0.0, -3.0], dtype=np.float64)


class _Space:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class ScriptedTokamak:
    """Fake tokamak env that emits a scripted observation sequence."""

    def __init__(self, init_obs: np.ndarray, step_obs: list[np.ndarray]) -> None:
        self._init_obs = init_obs
        self._step_obs = list(step_obs)
        self._idx = 0
        self.received_actions: list[np.ndarray] = []
        self.action_space = _Space((4,))
        self.observation_space = _Space((8,))

    def reset(self, **_: object) -> tuple[np.ndarray, dict]:
        self._idx = 0
        self.received_actions = []
        return self._init_obs.copy(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.received_actions.append(np.asarray(action, dtype=np.float64).ravel().copy())
        obs = self._step_obs[min(self._idx, len(self._step_obs) - 1)]
        self._idx += 1
        return obs.copy(), 1.0, False, False, {"base": True}


# --- helper-function unit tests -------------------------------------------------


def test_position_error_lyapunov_v_is_zero_at_target_and_monotone():
    assert position_error_lyapunov_v(_obs(dr=0.0, dz=0.0)) == 0.0
    small = position_error_lyapunov_v(_obs(dr=0.2, dz=0.1))
    large = position_error_lyapunov_v(_obs(dr=1.0, dz=1.0))
    assert 0.0 < small < large < 2.0


def test_lyapunov_theta_round_trips_through_guard_candidate():
    from scpn_fusion.phase.kuramoto import lyapunov_v

    for v in (0.0, 0.4, 1.3, 2.0):
        theta = _lyapunov_theta(v)
        assert lyapunov_v(theta, 0.0) == pytest.approx(v, abs=1e-9)


def test_default_obs_to_safety_state_maps_keys_and_velocity():
    curr = _obs(z=0.10, ip=12.0, beta=3.1)
    prev = _obs(z=0.0)
    state = default_obs_to_safety_state(curr, prev, dt=0.05)
    assert state["I_p"] == 12.0
    assert state["beta_N"] == 3.1
    assert state["dZ_dt"] == pytest.approx((0.10 - 0.0) / 0.05)


def test_default_obs_to_safety_state_zero_velocity_without_prev():
    state = default_obs_to_safety_state(_obs(z=0.5), None, dt=0.05)
    assert state["dZ_dt"] == 0.0


# --- shield behaviour -----------------------------------------------------------


def test_safe_state_passes_action_through_unchanged():
    env = ShieldedTokamakEnv(ScriptedTokamak(_obs(), [_obs()]))
    env.reset()
    action = np.array([0.5, -0.5, 0.3, 0.7])
    env.step(action)
    np.testing.assert_array_equal(env.base_env.received_actions[0], action)
    assert env.report.clamp_events == 0


def test_beta_breach_clamps_heating_to_cooling_only():
    # Beta 5.0 exceeds the default beta_no_wall limit (2.8) -> power_ramp inhibited.
    env = ShieldedTokamakEnv(ScriptedTokamak(_obs(beta=5.0), [_obs(beta=5.0)]))
    env.reset()
    env.step(np.array([0.0, 0.0, 0.0, 0.9]))
    # Heating delta (index 3) clamped to <= 0.
    assert env.base_env.received_actions[0][3] == 0.0
    assert "power_ramp" in env.report.channel_trips


def test_current_breach_freezes_coil_ramps():
    # Ip 20 MA exceeds the default current limit (15) -> current_ramp inhibited.
    env = ShieldedTokamakEnv(ScriptedTokamak(_obs(ip=20.0), [_obs(ip=20.0)]))
    env.reset()
    env.step(np.array([0.6, -0.4, 0.5, 0.0]))
    received = env.base_env.received_actions[0]
    np.testing.assert_array_equal(received[:3], np.zeros(3))
    assert env.report.channel_trips.get("current_ramp", 0) == 3


def test_vertical_velocity_breach_freezes_coils_on_second_step():
    # Z jumps 0.1 m over dt=0.05 s -> dZ/dt = 2 m/s > 1 m/s vertical limit.
    init = _obs(z=0.0)
    env = ShieldedTokamakEnv(ScriptedTokamak(init, [_obs(z=0.1), _obs(z=0.2)]))
    env.reset()
    env.step(np.array([0.5, 0.5, 0.5, 0.0]))  # dZ/dt still 0 (prev==last)
    np.testing.assert_array_equal(env.base_env.received_actions[0][:3], np.array([0.5, 0.5, 0.5]))
    env.step(np.array([0.5, 0.5, 0.5, 0.0]))  # now dZ/dt = 2 m/s
    np.testing.assert_array_equal(env.base_env.received_actions[1][:3], np.zeros(3))
    assert env.report.channel_trips.get("position_move", 0) == 3


def test_lyapunov_guard_halts_on_diverging_error():
    guard = LyapunovGuard(window=50, dt=1e-3, lambda_threshold=0.0, max_violations=3)
    # Monotonically growing position error -> positive Lyapunov exponent.
    growing = [_obs(dr=0.1 * k, dz=0.1 * k) for k in range(1, 8)]
    env = ShieldedTokamakEnv(ScriptedTokamak(_obs(), growing), guard=guard, halt_penalty=-5.0)
    env.reset()
    halted = False
    reward_at_halt = 0.0
    for _ in range(7):
        _, reward, terminated, _, info = env.step(np.zeros(4))
        if terminated:
            halted = info["shield_halt"]
            reward_at_halt = reward
            break
    assert halted
    assert env.report.halted
    assert reward_at_halt == pytest.approx(1.0 - 5.0)


def test_step_info_exposes_shield_channels():
    env = ShieldedTokamakEnv(ScriptedTokamak(_obs(), [_obs()]))
    env.reset()
    _, _, _, _, info = env.step(np.zeros(4))
    for key in (
        "interlock_allowed",
        "shielded_action",
        "shield_clamp_events",
        "lyapunov_v",
        "lyapunov_approved",
        "shield_halt",
    ):
        assert key in info
    assert info["base"] is True  # base-env info preserved


def test_reset_clears_report_and_guard():
    env = ShieldedTokamakEnv(ScriptedTokamak(_obs(ip=20.0), [_obs(ip=20.0)]))
    env.reset()
    env.step(np.array([0.5, 0.5, 0.5, 0.0]))
    assert env.report.clamp_events > 0
    env.reset()
    assert env.report.clamp_events == 0
    assert env.report.channel_trips == {}


def test_custom_interlock_and_obs_to_state_are_used():
    from scpn_fusion.scpn.safety_interlocks import SafetyInterlockRuntime

    # A custom obs_to_state that always reports a current breach, plus an
    # explicitly supplied interlock, exercises both injected-dependency branches.
    def always_over_current(_obs, _prev, _dt):
        return {"I_p": 99.0, "beta_N": 0.0, "dZ_dt": 0.0}

    env = ShieldedTokamakEnv(
        ScriptedTokamak(_obs(), [_obs()]),
        interlock=SafetyInterlockRuntime(),
        obs_to_state=always_over_current,
    )
    env.reset()
    env.step(np.array([0.4, 0.4, 0.4, 0.0]))
    np.testing.assert_array_equal(env.base_env.received_actions[0][:3], np.zeros(3))
    assert env.report.channel_trips.get("current_ramp", 0) == 3


def test_denied_channel_with_already_safe_action_records_no_clamp():
    # Current breach, but the coil deltas are already zero -> freeze changes nothing.
    env = ShieldedTokamakEnv(ScriptedTokamak(_obs(ip=20.0), [_obs(ip=20.0)]))
    env.reset()
    _, _, _, _, info = env.step(np.zeros(4))
    assert info["interlock_allowed"]["current_ramp"] is False
    assert env.report.clamp_events == 0


def test_step_before_reset_uses_zero_observation_fallback():
    env = ShieldedTokamakEnv(ScriptedTokamak(_obs(), [_obs()]))
    # No reset(): _last_obs is None, so the shield falls back to a zero state.
    obs, reward, terminated, truncated, info = env.step(np.zeros(4))
    assert "interlock_allowed" in info
    assert env.report.clamp_events == 0


def test_custom_nonnegative_rule_and_out_of_range_index():
    rules = {
        0: ActionChannelRule(("current_ramp",), NONNEGATIVE),
        9: ActionChannelRule(("current_ramp",), NONNEGATIVE),  # index beyond action size
    }
    env = ShieldedTokamakEnv(ScriptedTokamak(_obs(ip=20.0), [_obs(ip=20.0)]), action_rules=rules)
    env.reset()
    env.step(np.array([-0.7, 0.0, 0.0, 0.0]))
    # NONNEGATIVE clamps the negative delta up to 0; out-of-range index is ignored.
    assert env.base_env.received_actions[0][0] == 0.0
    assert env.report.channel_trips.get("current_ramp", 0) == 1
