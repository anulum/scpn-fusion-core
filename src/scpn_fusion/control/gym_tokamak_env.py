# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Gymnasium Tokamak Environment
"""
Gymnasium environment for Tokamak Flight Simulator.

Enables standard Reinforcement Learning (RL) training using Stable-Baselines3,
Ray Rllib, or other Gymnasium-compatible libraries.
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from scpn_fusion._data_paths import default_iter_config_path
from scpn_fusion.control.tokamak_flight_sim import IsoFluxController

FloatArray = NDArray[np.float32]
ActionArray = NDArray[np.float32] | NDArray[np.float64]

logger = logging.getLogger(__name__)


class TokamakEnv(gym.Env[FloatArray, ActionArray]):
    """
    Gymnasium environment wrapping the SCPN Tokamak Flight Simulator.

    Observation Space (Box):
        [R_axis, Z_axis, Ip_MA, Beta, Error_R, Error_Z, XP_R, XP_Z]

    Action Space (Box):
        [PF1_delta, PF3_delta, PF5_delta, Heating_delta]

    Notes
    -----
    Actions are four finite values in [-1, 1]. Each magnetic action requests
    a 0.1 MA (100 kA) offset from the configured initial current, not an
    increment accumulated at every step. The flight actuators retain their
    0.05 MA saturation, 1 MA/s slew and 0.06 s first-order lag. The heating
    setpoint is ``1 + 0.5 * action[3]`` in dimensionless beta units, subject
    to the existing heating actuator limits [1, 5]. Observations are float32,
    measured after the equilibrium solve; no extra sensor delay is introduced.

    Reward is minus the Euclidean R/Z position error in metres, with a further
    penalty of 10 when either absolute error exceeds 0.5 m. That condition
    terminates the episode; reaching ``max_steps`` truncates it independently.
    This position-error flag is not a physical disruption prediction.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        config_file: Optional[str] = None,
        max_steps: int = 100,
        control_dt_s: float = 0.05,
        render_mode: Optional[str] = None,
    ) -> None:
        """Create an environment using the configured equilibrium kernel.

        Parameters
        ----------
        config_file : str or None
            Machine JSON configuration; None selects the bundled ITER deck.
        max_steps : int
            Episode step budget used for truncation.
        control_dt_s : float
            Positive finite actuator update period in seconds.
        render_mode : str or None
            ``human`` logs the current axis; ``rgb_array`` returns a flux-field
            image. None disables rendering.
        """
        super().__init__()

        if config_file is None:
            config_file = str(default_iter_config_path())
        self.config_file = config_file
        self.max_steps = max_steps
        self.control_dt_s = control_dt_s
        self.render_mode = render_mode

        self.controller = IsoFluxController(
            config_file=self.config_file,
            verbose=False,
            control_dt_s=self.control_dt_s,
        )

        obs_low = np.array([0.0, -10.0, 0.0, 0.0, -10.0, -10.0, 0.0, -15.0], dtype=np.float32)
        obs_high = np.array([15.0, 10.0, 30.0, 20.0, 10.0, 10.0, 15.0, 5.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self._action_scale = np.array([0.1, 0.1, 0.1, 0.5], dtype=np.float64)

        self.current_step = 0
        self.state = None

    def _get_obs(self) -> FloatArray:
        kernel = self.controller.kernel

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

        err_r = float(self.controller.target_R - curr_R)
        err_z = float(self.controller.target_Z - curr_Z)

        return np.array(
            [curr_R, curr_Z, ip, beta, err_r, err_z, float(xp_pos[0]), float(xp_pos[1])],
            dtype=np.float32,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[FloatArray, Dict[str, Any]]:
        """Restore the configured plant and actuator state for a new episode.

        Parameters
        ----------
        seed : int or None
            Gymnasium random seed. Identical seeds and actions reproduce the
            deterministic equilibrium trajectory; action-space sampling is
            seeded separately through ``action_space.seed``.
        options : dict or None
            Reserved Gymnasium reset options; currently unused.

        Returns
        -------
        observation : ndarray of float32, shape (8,)
            Initial axis, current, beta, position errors and X-point values.
        info : dict
            Empty reset metadata.
        """
        super().reset(seed=seed)

        self.controller = IsoFluxController(
            config_file=self.config_file,
            verbose=False,
            control_dt_s=self.control_dt_s,
        )
        self.controller.kernel.solve_equilibrium()

        self.current_step = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action: ActionArray) -> Tuple[FloatArray, float, bool, bool, Dict[str, Any]]:
        """Apply one normalised offset command through the flight actuators.

        Parameters
        ----------
        action : ndarray, shape (4,)
            Finite PF1/PF3/PF5/heating commands in [-1, 1]. Magnetic commands
            scale to 0.1 MA before the actuator's physical saturation and lag.

        Returns
        -------
        observation : ndarray of float32, shape (8,)
            Post-solve plant observation in the class-documented order.
        reward : float
            Negative position-error norm, minus 10 on position-limit breach.
        terminated : bool
            Either absolute position error exceeds 0.5 m.
        truncated : bool
            The configured step budget has been exhausted.
        info : dict
            ``disrupted`` reports the position-limit breach, not a physics
            instability diagnosis.

        Raises
        ------
        ValueError
            Action shape, finiteness or normalised range is invalid. Validation
            occurs before any actuator, plant or episode-counter mutation.
        """
        action_values = np.asarray(action, dtype=np.float64)
        if action_values.shape != (4,):
            raise ValueError("action must have shape (4,).")
        if not np.all(np.isfinite(action_values)) or np.any(np.abs(action_values) > 1.0):
            raise ValueError("action must contain finite values in [-1, 1].")
        scaled_action = action_values * self._action_scale
        pf1_delta, pf3_delta, pf5_delta, heating_delta = scaled_action

        radial_applied = self.controller._act_radial.step(pf3_delta)
        top_applied = self.controller._act_top.step(pf1_delta)
        bottom_applied = self.controller._act_bottom.step(pf5_delta)
        beta_applied = self.controller._act_heating.step(1.0 + heating_delta)

        self.controller._set_coil_current_offset(0, top_applied)
        self.controller._set_coil_current_offset(2, radial_applied)
        self.controller._set_coil_current_offset(4, bottom_applied)

        self.controller.kernel.cfg.setdefault("physics", {})["beta_scale"] = beta_applied

        self.controller.kernel.solve_equilibrium()

        obs = self._get_obs()

        r_err = obs[4]
        z_err = obs[5]
        dist_to_target = np.sqrt(r_err**2 + z_err**2)

        reward = -dist_to_target

        disrupted = False
        if abs(r_err) > 0.5 or abs(z_err) > 0.5:
            reward -= 10.0
            disrupted = True

        self.current_step += 1
        terminated = disrupted
        truncated = self.current_step >= self.max_steps

        return obs, float(reward), terminated, truncated, {"disrupted": disrupted}

    def render(self) -> NDArray[np.uint8] | None:
        """Render the current solved state without advancing the plant.

        Returns
        -------
        image : ndarray of uint8 or None
            RGB flux-field image with shape (400, 600, 3) in ``rgb_array``
            mode; None for logging-only ``human`` mode or disabled rendering.
            Radial and vertical coordinates are in metres.
        """
        if self.render_mode == "human":
            obs = self._get_obs()
            logger.info(
                "Step %d: R=%.2f, Z=%.2f, Reward=%.4f",
                self.current_step,
                obs[0],
                obs[1],
                -np.sqrt(obs[4] ** 2 + obs[5] ** 2),
            )
        elif self.render_mode == "rgb_array":
            from matplotlib import rc_context
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure

            kernel = self.controller.kernel
            figure = Figure(figsize=(6, 4), dpi=100)
            FigureCanvasAgg(figure)
            axes = figure.subplots()
            field = axes.pcolormesh(kernel.R, kernel.Z, kernel.Psi, shading="auto")
            axes.plot(self.controller.target_R, self.controller.target_Z, "rx", label="Target")
            axes.set(
                xlabel="R [m]", ylabel="Z [m]", title=f"Poloidal flux — step {self.current_step}"
            )
            axes.set_aspect("equal")
            axes.legend()
            figure.colorbar(field, ax=axes, label="Poloidal flux (solver units)")
            with BytesIO() as buffer, rc_context({"savefig.bbox": None}):
                figure.savefig(buffer, format="rgba", dpi=100)
                rgba = np.frombuffer(buffer.getvalue(), dtype=np.uint8).reshape(400, 600, 4)
            return rgba[:, :, :3].copy()
        return None


# Registration
def register() -> None:
    """Register the Tokamak-v0 Gymnasium environment."""
    gym.envs.registration.register(
        id="Tokamak-v0",
        entry_point="scpn_fusion.control.gym_tokamak_env:TokamakEnv",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    # Smoke test
    register()
    env = gym.make("Tokamak-v0", max_steps=10)
    obs, info = env.reset()
    logger.info("Initial observation: %s", obs)
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        logger.info("Sampled action reward: action=%s reward=%.4f", action, reward)
        if terminated or truncated:
            break
    env.close()
