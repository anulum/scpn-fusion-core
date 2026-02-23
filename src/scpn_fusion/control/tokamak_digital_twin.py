# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Tokamak Digital Twin
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import logging
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

from scpn_fusion.io.imas_connector import (
    digital_twin_history_to_ids,
    digital_twin_history_to_ids_pulse,
    digital_twin_summary_to_ids,
)

GRID_SIZE = 40
TIME_STEPS = 10000
LEARNING_RATE = 0.0001
HIDDEN_SIZE = 64
BATCH_SIZE = 32
MEMORY_SIZE = 1000
R_MAJ = 2.0   # major radius [m]
R_MIN = 0.8   # minor radius [m]


def _resolve_rng(seed: int, rng: Optional[np.random.Generator]) -> np.random.Generator:
    if rng is not None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator when provided")
        return rng
    return np.random.default_rng(int(seed))


class TokamakTopoloy:
    """Magnetic geometry: q-profile and island evolution via Modified Rutherford Equation."""

    def __init__(self, size=GRID_SIZE):
        self.size = size
        y, x = np.ogrid[-size/2:size/2, -size/2:size/2]
        self.r_map = np.sqrt(x**2 + y**2) / (size/2)
        self.mask = self.r_map <= 1.0

        self.q0 = 1.0
        self.qa = 3.0
        self.update_q_profile(0.0)

        self.resonances = [1.5, 2.0, 2.5, 3.0]
        self.island_widths = {res: 0.01 for res in self.resonances}
        self.eta = 1e-5

    def step_island_evolution(self, dt=0.1):
        """Evolve island widths via MRE with neoclassical bootstrap drive."""
        beta_p = 0.6
        w_crit = 0.05
        for res in self.resonances:
            delta_prime = -0.2 - (5.0 * self.island_widths[res])
            f_bs = beta_p * (self.island_widths[res] / (self.island_widths[res]**2 + w_crit**2))
            dw_dt = self.eta * (delta_prime + f_bs)
            self.island_widths[res] = max(0.001, self.island_widths[res] + dw_dt * dt)

    def update_q_profile(self, current_drive_action):
        """Update parabolic q(r) = q0 + (qa-q0)*r^2 with current drive modulation."""
        mod_q0 = self.q0 - (0.2 * current_drive_action)
        mod_qa = self.qa + (0.5 * current_drive_action)
        self.q_map = mod_q0 + (mod_qa - mod_q0) * (self.r_map**2)

    def get_rational_surfaces(self):
        """Boolean map of rational-surface islands from current MRE widths."""
        danger_map = np.zeros((self.size, self.size), dtype=bool)
        for res in self.resonances:
            width = self.island_widths[res]
            mask = (np.abs(self.q_map - res) < width) & self.mask
            danger_map = np.logical_or(danger_map, mask)
        return danger_map

class Plasma2D:
    """2D diffusion-reaction model on a poloidal cross-section."""

    def __init__(self, topology, gyro_surrogate=None):
        self.topo = topology
        self.T = np.zeros((GRID_SIZE, GRID_SIZE))
        self.T_core_hist = []
        self._gyro_surrogate = gyro_surrogate

    def step(self, action):
        """Evolve plasma one timestep with current drive action in [-1, 1]."""
        self.topo.update_q_profile(action)
        self.topo.step_island_evolution()
        danger_zones = self.topo.get_rational_surfaces()

        center = GRID_SIZE // 2
        self.T[center, center] += 5.0

        D_base = 0.01
        D_turb = 0.5
        diffusivity = np.ones_like(self.T) * D_base
        diffusivity[danger_zones] = D_turb

        if self._gyro_surrogate is not None:
            correction = np.asarray(
                self._gyro_surrogate(self.T, self.topo.q_map, danger_zones),
                dtype=float,
            )
            if correction.shape != self.T.shape:
                raise ValueError(
                    f"gyro_surrogate shape must be {self.T.shape}, got {correction.shape}"
                )
            correction = np.nan_to_num(correction, nan=1.0, posinf=2.0, neginf=0.5)
            np.clip(correction, 0.2, 5.0, out=correction)
            diffusivity *= correction

        T_up = np.roll(self.T, -1, axis=0)
        T_down = np.roll(self.T, 1, axis=0)
        T_left = np.roll(self.T, -1, axis=1)
        T_right = np.roll(self.T, 1, axis=1)
        laplacian = (T_up + T_down + T_left + T_right - 4*self.T)

        # Bremsstrahlung ~ sqrt(T); tungsten line radiation peaks at ~2 keV
        radiation = 0.002 * np.sqrt(self.T + 1e-6)
        tungsten_rad = 0.05 * np.exp(-((self.T - 2.0)**2) / 0.5)

        self.T += diffusivity * laplacian - radiation - tungsten_rad
        self.T[~self.topo.mask] = 0.0
        self.T = np.clip(self.T, 0, 100.0)

        core_temp = self.T[center, center]
        avg_temp = np.mean(self.T[self.topo.mask])
        self.T_core_hist.append(core_temp)
        return self.T.flatten(), avg_temp

class SimpleNeuralNet:
    """NumPy MLP policy network for continuous control."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        *,
        rng: np.random.Generator,
    ) -> None:
        self.W1 = rng.standard_normal((input_size, hidden_size)) * np.sqrt(1 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.standard_normal((hidden_size, output_size)) * np.sqrt(1 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.out = np.tanh(self.z2)
        return self.out

    def train_step(self, x, target_action, advantage):
        """REINFORCE-like policy gradient step."""
        pred = self.forward(x)
        grad_out = -(advantage)

        d_z2 = grad_out * (1 - self.out**2)
        d_W2 = np.dot(self.a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)

        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * (1 - self.a1**2)
        d_W1 = np.dot(x.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)

        self.W1 -= LEARNING_RATE * d_W1
        self.b1 -= LEARNING_RATE * d_b1
        self.W2 -= LEARNING_RATE * d_W2
        self.b2 -= LEARNING_RATE * d_b2
        return np.mean(np.abs(grad_out))

def run_digital_twin(
    time_steps=TIME_STEPS,
    seed=42,
    save_plot=True,
    output_path="Tokamak_Digital_Twin.png",
    verbose=True,
    gyro_surrogate=None,
    chaos_monkey: bool = False,
    sensor_dropout_prob: float = 0.0,
    sensor_noise_std: float = 0.0,
    rng: Optional[np.random.Generator] = None,
):
    """Run digital-twin control simulation, return summary dict."""
    steps = int(time_steps)
    if steps < 1:
        raise ValueError("time_steps must be >= 1.")
    chaos_monkey = bool(chaos_monkey)
    sensor_dropout_prob = float(sensor_dropout_prob)
    sensor_noise_std = float(sensor_noise_std)
    if not np.isfinite(sensor_dropout_prob) or sensor_dropout_prob < 0.0 or sensor_dropout_prob > 1.0:
        raise ValueError("sensor_dropout_prob must be finite and in [0, 1].")
    if not np.isfinite(sensor_noise_std) or sensor_noise_std < 0.0:
        raise ValueError("sensor_noise_std must be finite and >= 0.")
    local_rng = _resolve_rng(seed=int(seed), rng=rng)
    if verbose:
        logger.info("--- SCPN 2D TOKAMAK DIGITAL TWIN + NEURAL CONTROL ---")
    
    topo = TokamakTopoloy()
    plasma = Plasma2D(topo, gyro_surrogate=gyro_surrogate)
    
    state_dim = GRID_SIZE
    brain = SimpleNeuralNet(state_dim, HIDDEN_SIZE, 1, rng=local_rng)
    
    history_rewards = []
    history_actions = []
    sensor_dropouts_total = 0
    
    if verbose:
        logger.info("Training Neural Network for %d steps...", steps)
    
    for t in range(steps):
        midplane_idx = GRID_SIZE // 2
        state_vector = np.asarray(plasma.T[midplane_idx, :], dtype=float).reshape(1, -1).copy()
        if chaos_monkey:
            if sensor_noise_std > 0.0:
                state_vector += local_rng.normal(
                    0.0, sensor_noise_std, size=state_vector.shape
                )
            if sensor_dropout_prob > 0.0:
                dropout_mask = local_rng.random(state_vector.shape[1]) < sensor_dropout_prob
                dropped = int(np.sum(dropout_mask, dtype=np.int64))
                if dropped > 0:
                    state_vector[0, dropout_mask] = 0.0
                    sensor_dropouts_total += dropped
            state_vector = np.nan_to_num(
                state_vector, nan=0.0, posinf=100.0, neginf=0.0
            )
        
        noise = float(local_rng.normal(0.0, 0.2))
        raw_action = brain.forward(state_vector)
        action = float(np.clip(raw_action + noise, -1.0, 1.0)[0, 0])

        _, avg_temp = plasma.step(action)

        islands_area = np.sum(topo.get_rational_surfaces())
        reward = avg_temp - (islands_area * 0.05)

        baseline = np.mean(history_rewards[-50:]) if len(history_rewards) > 50 else 0
        # Derivative-free policy gradient estimator
        advantage = (reward - baseline) * noise
        
        loss = brain.train_step(state_vector, None, advantage)
        
        history_rewards.append(reward)
        history_actions.append(action)
        
        if verbose and t % 500 == 0:
            logger.info(
                "Digital twin simulation progress",
                extra={"physics_context": {
                    "step": t,
                    "avg_temp": float(avg_temp),
                    "action": float(action),
                    "loss": float(loss),
                    "islands_px": int(np.sum(topo.get_rational_surfaces()))
                }}
            )

    plot_saved = False
    plot_error = None
    if save_plot:
        try:
            fig = plt.figure(figsize=(15, 6))
            ax1 = fig.add_subplot(1, 3, 1)
            im = ax1.imshow(plasma.T, cmap='inferno', origin='lower')
            ax1.set_title("Final Plasma Cross-Section (2D)")
            plt.colorbar(im, ax=ax1, label='Temperature (keV)')

            islands = topo.get_rational_surfaces()
            ax1.contour(islands, colors='cyan', levels=[0.5], linewidths=1, alpha=0.5)
            ax1.text(2, 2, "Cyan = q-Resonance (Islands)", color='cyan', fontsize=8)

            ax2 = fig.add_subplot(1, 3, 2)
            ax2.plot(history_rewards, color='orange', alpha=0.6)
            # Moving average
            if len(history_rewards) > 50:
                mov_avg = np.convolve(history_rewards, np.ones(50)/50, mode='valid')
                ax2.plot(range(len(mov_avg)), mov_avg, 'r-', linewidth=2, label='Moving Avg')
            ax2.set_title("Learning Curve")
            ax2.set_xlabel("Steps")
            ax2.set_ylabel("Reward (Confinement)")
            ax2.legend()

            ax3 = fig.add_subplot(1, 3, 3)
            r_axis = np.linspace(0, 1, GRID_SIZE//2)
            q_axis = topo.q_map[GRID_SIZE//2, GRID_SIZE//2:]
            ax3.plot(r_axis, q_axis, 'b-', linewidth=2, label='Safety Factor q(r)')

            for q_res in [1.5, 2.0, 2.5, 3.0]:
                ax3.axhline(q_res, color='red', linestyle='--', alpha=0.3)
                ax3.text(0.1, q_res, f"q={q_res}", color='red', fontsize=8)

            ax3.set_title("Final Safety Factor Profile")
            ax3.set_xlabel("Normalized Radius r/a")
            ax3.set_ylabel("q")
            ax3.legend()

            plt.tight_layout()
            plt.savefig(output_path)
            plt.close(fig)
            plot_saved = True
            if verbose:
                logger.info("Digital Twin Simulation Complete. Snapshot saved: %s", output_path)
        except Exception as exc:
            plot_error = str(exc)
            if verbose:
                logger.warning("Digital Twin completed without plot artifact: %s", exc)

    islands_final = int(np.sum(topo.get_rational_surfaces()))
    final_avg_temp = float(history_rewards[-1] + islands_final * 0.05) if history_rewards else 0.0
    summary = {
        "seed": int(seed),
        "steps": int(steps),
        "final_avg_temp": float(final_avg_temp),
        "final_reward": float(history_rewards[-1]) if history_rewards else 0.0,
        "final_action": float(history_actions[-1]) if history_actions else 0.0,
        "final_islands_px": islands_final,
        "reward_mean_last_50": float(np.mean(history_rewards[-50:])) if history_rewards else 0.0,
        "chaos_monkey": chaos_monkey,
        "sensor_dropout_prob": sensor_dropout_prob,
        "sensor_noise_std": sensor_noise_std,
        "sensor_dropouts_total": int(sensor_dropouts_total),
        "sensor_dropout_rate": float(sensor_dropouts_total / (steps * GRID_SIZE)),
        "plot_saved": bool(plot_saved),
        "plot_error": plot_error,
    }
    return summary


def run_digital_twin_ids(
    *,
    machine: str = "ITER",
    shot: int = 0,
    run: int = 0,
    **kwargs,
):
    """
    Run digital twin and return IDS-like equilibrium payload.
    """
    summary = run_digital_twin(**kwargs)
    return digital_twin_summary_to_ids(
        summary,
        machine=machine,
        shot=shot,
        run=run,
    )


def run_digital_twin_ids_history(
    history_steps: Sequence[int],
    *,
    machine: str = "ITER",
    shot: int = 0,
    run: int = 0,
    seed: int = 42,
    **kwargs,
):
    """
    Run digital twin at multiple horizons and return IDS-like payload sequence.
    """
    if "time_steps" in kwargs:
        raise ValueError("time_steps is controlled by history_steps in history mode.")
    if "seed" in kwargs:
        raise ValueError("seed is controlled by the seed argument in history mode.")
    if isinstance(history_steps, (str, bytes, bytearray)) or not isinstance(history_steps, Sequence):
        raise ValueError("history_steps must be a sequence of positive integers.")
    if len(history_steps) == 0:
        raise ValueError("history_steps must contain at least one step count.")

    snapshots = _run_digital_twin_history_snapshots(
        history_steps=history_steps,
        seed=seed,
        **kwargs,
    )

    return digital_twin_history_to_ids(
        snapshots,
        machine=machine,
        shot=shot,
        run=run,
    )


def run_digital_twin_ids_pulse(
    history_steps: Sequence[int],
    *,
    machine: str = "ITER",
    shot: int = 0,
    run: int = 0,
    seed: int = 42,
    **kwargs,
):
    """
    Run digital twin at multiple horizons and return pulse-style IDS container.
    """
    if "time_steps" in kwargs:
        raise ValueError("time_steps is controlled by history_steps in pulse mode.")
    if "seed" in kwargs:
        raise ValueError("seed is controlled by the seed argument in pulse mode.")

    snapshots = _run_digital_twin_history_snapshots(
        history_steps=history_steps,
        seed=seed,
        **kwargs,
    )
    return digital_twin_history_to_ids_pulse(
        snapshots,
        machine=machine,
        shot=shot,
        run=run,
    )


def _run_digital_twin_history_snapshots(
    *,
    history_steps: Sequence[int],
    seed: int,
    **kwargs,
) -> list[dict[str, object]]:
    if isinstance(history_steps, (str, bytes, bytearray)) or not isinstance(history_steps, Sequence):
        raise ValueError("history_steps must be a sequence of positive integers.")
    if len(history_steps) == 0:
        raise ValueError("history_steps must contain at least one step count.")

    snapshots: list[dict[str, object]] = []
    base_seed = int(seed)
    for idx, step in enumerate(history_steps):
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError(f"history_steps[{idx}] must be a positive integer.")
        if int(step) < 1:
            raise ValueError(f"history_steps[{idx}] must be >= 1.")
        summary = run_digital_twin(
            time_steps=int(step),
            seed=base_seed,
            **kwargs,
        )
        snapshots.append(summary)
    return snapshots

if __name__ == "__main__":
    run_digital_twin()
