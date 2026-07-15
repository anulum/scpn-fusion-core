# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Safety-Shielded Tokamak Environment
"""Safety-shielded wrapper composing the interlock net and the Lyapunov guard.

The wrapper turns a bare :class:`~scpn_fusion.control.gym_tokamak_env.TokamakEnv`
(or any Gym-like tokamak env) into a *shielded* env with two independent layers:

1. **Interlock veto/clamp.** Before every step the current observation is mapped
   to plasma-limit tokens and fed to a
   :class:`~scpn_fusion.scpn.safety_interlocks.SafetyInterlockRuntime`. When a
   control channel is inhibited (a limit is breached) the mapped action
   component is clamped to its safe half-space (heating may only decrease when
   beta/thermal is high; coil ramps freeze when current or vertical-position
   limits trip).

2. **Lyapunov halt.** After the step the position-error energy is folded into a
   Lyapunov candidate and passed to a
   :class:`~scpn_fusion.phase.lyapunov_guard.LyapunovGuard`. A sustained
   positive Lyapunov exponent (diverging error) terminates the episode as a
   fail-safe.

Every intervention is counted and surfaced in the ``info`` mapping so a controller
comparison can measure how hard each policy leans on the shield.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.phase.lyapunov_guard import LyapunovGuard, LyapunovVerdict
from scpn_fusion.scpn.safety_interlocks import SafetyInterlockRuntime

FloatArray: TypeAlias = NDArray[np.float64]

# Clamp modes for a shielded action component.
FREEZE = "freeze"  # hold the actuator (delta -> 0)
NONPOSITIVE = "nonpositive"  # allow only a decrease (delta -> min(x, 0))
NONNEGATIVE = "nonnegative"  # allow only an increase (delta -> max(x, 0))


@dataclass(frozen=True)
class ActionChannelRule:
    """Maps one action component to the interlock channels that gate it."""

    channels: tuple[str, ...]
    clamp: str


# Default mapping for the TokamakEnv action [PF1, PF3, PF5, Heating]:
# coil ramps freeze on a current or vertical-position breach; heating may only
# cool when a thermal or beta limit is breached.
DEFAULT_ACTION_RULES: dict[int, ActionChannelRule] = {
    0: ActionChannelRule(("current_ramp", "position_move"), FREEZE),
    1: ActionChannelRule(("current_ramp", "position_move"), FREEZE),
    2: ActionChannelRule(("current_ramp", "position_move"), FREEZE),
    3: ActionChannelRule(("heat_ramp", "power_ramp"), NONPOSITIVE),
}


def default_obs_to_safety_state(
    obs: FloatArray,
    prev_obs: FloatArray | None,
    dt: float,
) -> dict[str, float]:
    """Map a TokamakEnv observation to interlock plasma-state keys.

    Observation layout ``[R, Z, Ip_MA, Beta, dR, dZ, XP_R, XP_Z]``. Only the
    current, beta and vertical-velocity limits are observable here; the thermal
    and density channels are left at their safe defaults (they are not part of
    this observation vector).
    """
    o = np.asarray(obs, dtype=np.float64).ravel()
    dz_dt = 0.0
    if prev_obs is not None and dt > 0.0:
        p = np.asarray(prev_obs, dtype=np.float64).ravel()
        dz_dt = float((o[1] - p[1]) / dt)
    return {
        "I_p": abs(float(o[2])),
        "beta_N": float(o[3]),
        "dZ_dt": dz_dt,
    }


def position_error_lyapunov_v(obs: FloatArray) -> float:
    """Fold the radial/vertical position error into a Lyapunov candidate in [0, 2).

    ``V = 2 (1 - exp(-0.5 (dR^2 + dZ^2)))`` is zero at the target, strictly
    increasing in the error energy, and bounded to the guard's ``[0, 2]`` range
    so a growing error yields a positive Lyapunov exponent.
    """
    o = np.asarray(obs, dtype=np.float64).ravel()
    energy = float(o[4] ** 2 + o[5] ** 2)
    return float(2.0 * (1.0 - np.exp(-0.5 * energy)))


def _lyapunov_theta(v: float) -> FloatArray:
    """Encode a scalar V in [0, 2] as a one-element phase so ``guard.check`` sees it.

    ``lyapunov_v([theta], psi=0) = 1 - cos(theta)``, so ``theta = arccos(1 - V)``
    round-trips the candidate through the guard's stateful sliding window.
    """
    v_clipped = float(np.clip(v, 0.0, 2.0))
    return np.array([float(np.arccos(1.0 - v_clipped))], dtype=np.float64)


@dataclass
class ShieldReport:
    """Per-episode tally of shield activity."""

    clamp_events: int = 0
    channel_trips: dict[str, int] = field(default_factory=dict)
    halted: bool = False

    def record_trip(self, channel: str) -> None:
        """Increment the clamp counter for one interlock channel."""
        self.channel_trips[channel] = self.channel_trips.get(channel, 0) + 1
        self.clamp_events += 1


class ShieldedTokamakEnv:
    """Gym-like env that shields actions with an interlock net and Lyapunov guard."""

    def __init__(
        self,
        base_env: Any,
        *,
        interlock: SafetyInterlockRuntime | None = None,
        guard: LyapunovGuard | None = None,
        action_rules: Mapping[int, ActionChannelRule] | None = None,
        obs_to_state: Callable[[FloatArray, FloatArray | None, float], dict[str, float]]
        | None = None,
        control_dt_s: float = 0.05,
        halt_penalty: float = 0.0,
    ) -> None:
        """Wrap ``base_env`` with a safety interlock and a Lyapunov halt guard."""
        self.base_env = base_env
        self.action_space = base_env.action_space
        self.observation_space = base_env.observation_space
        self.interlock = interlock if interlock is not None else SafetyInterlockRuntime()
        self.guard = guard if guard is not None else LyapunovGuard()
        self.action_rules = (
            dict(action_rules) if action_rules is not None else dict(DEFAULT_ACTION_RULES)
        )
        self._obs_to_state = (
            obs_to_state if obs_to_state is not None else default_obs_to_safety_state
        )
        self.control_dt_s = float(control_dt_s)
        self.halt_penalty = float(halt_penalty)
        self._last_obs: FloatArray | None = None
        self._prev_obs: FloatArray | None = None
        self.report = ShieldReport()

    def reset(self, **kwargs: Any) -> tuple[FloatArray, dict[str, Any]]:
        """Reset the base env, guard and shield tallies."""
        obs, info = self.base_env.reset(**kwargs)
        self.guard.reset()
        self.report = ShieldReport()
        obs_vec = np.asarray(obs, dtype=np.float64).ravel()
        self._last_obs = obs_vec
        self._prev_obs = obs_vec
        return obs, info

    def shield_action(self, action: FloatArray, allowed: Mapping[str, bool]) -> FloatArray:
        """Clamp each action component whose gating channel is inhibited."""
        shielded = np.asarray(action, dtype=np.float64).ravel().copy()
        for idx, rule in self.action_rules.items():
            if idx >= shielded.size:
                continue
            denied = any(not allowed.get(channel, True) for channel in rule.channels)
            if not denied:
                continue
            original = float(shielded[idx])
            if rule.clamp == FREEZE:
                new_value = 0.0
            elif rule.clamp == NONPOSITIVE:
                new_value = min(original, 0.0)
            elif rule.clamp == NONNEGATIVE:
                new_value = max(original, 0.0)
            else:  # pragma: no cover - guarded by ActionChannelRule construction
                raise ValueError(f"unknown clamp mode: {rule.clamp}")
            if new_value != original:
                shielded[idx] = new_value
                # Attribute the trip to the first denied channel for the tally.
                denied_channel = next(
                    channel for channel in rule.channels if not allowed.get(channel, True)
                )
                self.report.record_trip(denied_channel)
        return shielded

    def step(self, action: FloatArray) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        """Shield the action, step the base env, then apply the Lyapunov halt."""
        current_obs = (
            self._last_obs
            if self._last_obs is not None
            else np.asarray(action, dtype=np.float64).ravel() * 0.0
        )
        state = self._obs_to_state(current_obs, self._prev_obs, self.control_dt_s)
        allowed = self.interlock.update_from_state(state)
        shielded = self.shield_action(action, allowed)

        obs, reward, terminated, truncated, info = self.base_env.step(shielded)
        obs_vec = np.asarray(obs, dtype=np.float64).ravel()

        v = position_error_lyapunov_v(obs_vec)
        verdict: LyapunovVerdict = self.guard.check(_lyapunov_theta(v), 0.0)
        shield_halt = not verdict.approved
        reward = float(reward)
        if shield_halt and not terminated:
            terminated = True
            reward += self.halt_penalty
            self.report.halted = True

        shielded_info = dict(info)
        shielded_info.update(
            {
                "interlock_allowed": dict(allowed),
                "shielded_action": shielded,
                "shield_clamp_events": self.report.clamp_events,
                "shield_channel_trips": dict(self.report.channel_trips),
                "lyapunov_v": v,
                "lyapunov_lambda": verdict.lambda_exp,
                "lyapunov_approved": verdict.approved,
                "shield_halt": shield_halt,
            }
        )

        # Advance the two-step observation history for the next dZ/dt estimate.
        self._prev_obs = self._last_obs
        self._last_obs = obs_vec
        return obs, reward, terminated, truncated, shielded_info
