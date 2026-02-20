# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Gymnasium Tokamak Environment
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Gymnasium environment for Tokamak Flight Simulator.

Enables standard Reinforcement Learning (RL) training using Stable-Baselines3,
Ray Rllib, or other Gymnasium-compatible libraries.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from scpn_fusion.control.tokamak_flight_sim import (
    IsoFluxController,
    TARGET_R,
    TARGET_Z,
)

try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel

logger = logging.getLogger(__name__)


class TokamakEnv(gym.Env):
    """
    Gymnasium environment wrapping the SCPN Tokamak Flight Simulator.

    Observation Space (Box):
        [R_axis, Z_axis, Ip_MA, Beta, Error_R, Error_Z, XP_R, XP_Z]
    
    Action Space (Box):
        [PF1_delta, PF3_delta, PF5_delta, Heating_delta]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        config_file: Optional[str] = None,
        max_steps: int = 100,
        control_dt_s: float = 0.05,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        if config_file is None:
            # Default to bundled iter_config.json
            repo_root = Path(__file__).resolve().parents[3]
            config_file = str(repo_root / "iter_config.json")
        self.config_file = config_file
        self.max_steps = max_steps
        self.control_dt_s = control_dt_s
        self.render_mode = render_mode

        # Initialize Simulator (Controller is used here as a state manager)
        self.controller = IsoFluxController(
            config_file=self.config_file,
            verbose=False,
            control_dt_s=self.control_dt_s,
        )

        # Observation: [R, Z, Ip, Beta, dR, dZ, XP_R, XP_Z]
        # Expanded limits to ensure compliance with noisy/perturbed states
        obs_low = np.array([0.0, -10.0, 0.0, 0.0, -10.0, -10.0, 0.0, -15.0], dtype=np.float32)
        obs_high = np.array([15.0, 10.0, 30.0, 20.0, 10.0, 10.0, 15.0, 5.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action: Normalized [-1, 1] mapped to physical deltas
        # [PF1_delta, PF3_delta, PF5_delta, Heating_delta]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # Scaling factor for actions (1.0 in RL -> 100kA in physical sim)
        self._action_scale = np.array([1e5, 1e5, 1e5, 0.5], dtype=np.float32)

        self.current_step = 0
        self.state = None

    def _get_obs(self) -> np.ndarray:
        kernel = self.controller.kernel
        
        # Measure state (similar to tokamak_flight_sim.py)
        if hasattr(kernel, "find_magnetic_axis"):
            curr_R, curr_Z, _ = kernel.find_magnetic_axis()
        else:
            idx_max = np.argmax(kernel.Psi)
            iz, ir = np.unravel_index(idx_max, kernel.Psi.shape)
            curr_R = float(kernel.R[ir])
            curr_Z = float(kernel.Z[iz])
            
        xp_pos, _ = kernel.find_x_point(kernel.Psi)
        
        physics_cfg = kernel.cfg.get("physics", {})
        ip = float(physics_cfg.get("plasma_current_target", 5.0))
        beta = float(physics_cfg.get("beta_scale", 1.0))

        err_r = float(TARGET_R - curr_R)
        err_z = float(TARGET_Z - curr_Z)

        return np.array([
            curr_R, curr_Z, ip, beta,
            err_r, err_z,
            float(xp_pos[0]), float(xp_pos[1])
        ], dtype=np.float32)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Re-init controller/kernel
        self.controller = IsoFluxController(
            config_file=self.config_file,
            verbose=False,
            control_dt_s=self.control_dt_s,
        )
        self.controller.kernel.solve_equilibrium()
        
        self.current_step = 0
        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Map normalized [-1, 1] to physical deltas
        scaled_action = action * self._action_scale
        pf1_delta, pf3_delta, pf5_delta, heating_delta = scaled_action

        # Map actions to actuators (with lag if we reuse FirstOrderActuator)
        # For RL, we can either control raw currents or go through actuators.
        # Here we go through actuators to match flight_sim.
        radial_applied = self.controller._act_radial.step(pf3_delta)
        top_applied = self.controller._act_top.step(pf1_delta)
        bottom_applied = self.controller._act_bottom.step(pf5_delta)
        beta_applied = self.controller._act_heating.step(1.0 + heating_delta) # base 1.0

        # Update Coils
        self.controller._add_coil_current(0, top_applied)
        self.controller._add_coil_current(2, radial_applied)
        self.controller._add_coil_current(4, bottom_applied)
        
        # Update Physics
        self.controller.kernel.cfg.setdefault("physics", {})["beta_scale"] = beta_applied
        
        # Solve Equilibrium
        self.controller.kernel.solve_equilibrium()
        
        # New observation
        obs = self._get_obs()
        
        # Reward function: - MSE of position + energy efficiency
        r_err = obs[4]
        z_err = obs[5]
        dist_to_target = np.sqrt(r_err**2 + z_err**2)
        
        reward = -dist_to_target
        
        # Penalty for disruption (if out of bounds)
        disrupted = False
        if abs(r_err) > 0.5 or abs(z_err) > 0.5:
            reward -= 10.0
            disrupted = True

        self.current_step += 1
        terminated = disrupted
        truncated = self.current_step >= self.max_steps
        
        return obs, float(reward), terminated, truncated, {"disrupted": disrupted}

    def render(self):
        if self.render_mode == "human":
            # For now, just log state. In future, trigger visualize_flight.
            obs = self._get_obs()
            print(f"Step {self.current_step}: R={obs[0]:.2f}, Z={obs[1]:.2f}, Reward={-np.sqrt(obs[4]**2+obs[5]**2):.4f}")

# Registration
def register():
    gym.envs.registration.register(
        id="Tokamak-v0",
        entry_point="scpn_fusion.control.gym_tokamak_env:TokamakEnv",
    )

if __name__ == "__main__":
    # Smoke test
    register()
    env = gym.make("Tokamak-v0", max_steps=10)
    obs, info = env.reset()
    print("Initial Obs:", obs)
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action} -> Reward: {reward:.4f}")
        if terminated or truncated:
            break
    env.close()
