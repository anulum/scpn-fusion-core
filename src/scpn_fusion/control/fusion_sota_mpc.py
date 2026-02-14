# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Fusion Sota MPC
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel

# --- SOTA PARAMETERS ---
PREDICTION_HORIZON = 10
SHOT_LENGTH = 100


def _normalize_bounds(bounds: Tuple[float, float], name: str) -> Tuple[float, float]:
    lo = float(bounds[0])
    hi = float(bounds[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError(f"{name} must be finite with lower < upper.")
    return lo, hi


class NeuralSurrogate:
    """
    Linearized surrogate model around current operating point.
    """

    def __init__(self, n_coils: int, n_state: int, verbose: bool = True) -> None:
        self.verbose = bool(verbose)
        self.A = np.eye(int(n_state), dtype=np.float64)
        self.B = np.zeros((int(n_state), int(n_coils)), dtype=np.float64)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def train_on_kernel(self, kernel: Any, perturbation: float = 1.0) -> None:
        self._log("[SOTA] Training Neural Surrogate on Physics Kernel...")
        kernel.solve_equilibrium()
        base_state = self.get_state(kernel)
        p = max(float(perturbation), 1e-9)

        for i in range(len(kernel.cfg["coils"])):
            old_i = float(kernel.cfg["coils"][i].get("current", 0.0))
            kernel.cfg["coils"][i]["current"] = old_i + p
            kernel.solve_equilibrium()
            new_state = self.get_state(kernel)
            self.B[:, i] = (new_state - base_state) / p
            kernel.cfg["coils"][i]["current"] = old_i

        kernel.solve_equilibrium()
        self._log("[SOTA] Surrogate Training Complete.")

    def get_state(self, kernel: Any) -> np.ndarray:
        idx_max = int(np.argmax(kernel.Psi))
        iz, ir = np.unravel_index(idx_max, kernel.Psi.shape)
        r_ax = float(kernel.R[ir])
        z_ax = float(kernel.Z[iz])
        xp_pos, _ = kernel.find_x_point(kernel.Psi)
        return np.array([r_ax, z_ax, float(xp_pos[0]), float(xp_pos[1])], dtype=np.float64)

    def predict(self, current_state: np.ndarray, action_delta: np.ndarray) -> np.ndarray:
        return np.asarray(current_state, dtype=np.float64) + (self.B @ np.asarray(action_delta, dtype=np.float64))


class ModelPredictiveController:
    """
    Gradient-based MPC planner over surrogate dynamics.
    """

    def __init__(
        self,
        surrogate: NeuralSurrogate,
        target_state: np.ndarray,
        *,
        prediction_horizon: int = PREDICTION_HORIZON,
        learning_rate: float = 0.5,
        iterations: int = 20,
        action_limit: float = 2.0,
        action_regularization: float = 0.1,
    ) -> None:
        self.model = surrogate
        self.target = np.asarray(target_state, dtype=np.float64).reshape(-1)
        self.horizon = max(int(prediction_horizon), 1)
        self.learning_rate = max(float(learning_rate), 1e-9)
        self.iterations = max(int(iterations), 1)
        self.action_limit = max(float(action_limit), 1e-9)
        self.action_regularization = max(float(action_regularization), 0.0)

    def plan_trajectory(self, current_state: np.ndarray) -> np.ndarray:
        n_coils = int(self.model.B.shape[1])
        planned_actions = np.zeros((self.horizon, n_coils), dtype=np.float64)
        state0 = np.asarray(current_state, dtype=np.float64).reshape(-1)

        for _ in range(self.iterations):
            temp_state = state0.copy()
            grads = np.zeros_like(planned_actions)
            for t in range(self.horizon):
                next_state = self.model.predict(temp_state, planned_actions[t])
                error = next_state - self.target
                grad_step = self.model.B.T @ error
                grad_step += self.action_regularization * planned_actions[t]
                grads[t] = grad_step
                temp_state = next_state
            planned_actions -= self.learning_rate * grads
            planned_actions = np.clip(
                planned_actions, -self.action_limit, self.action_limit
            )

        return np.asarray(planned_actions[0], dtype=np.float64)


def _plot_telemetry(
    h_r: np.ndarray,
    h_z: np.ndarray,
    h_xr: np.ndarray,
    h_xz: np.ndarray,
    target_vec: np.ndarray,
    output_path: str,
) -> Tuple[bool, Optional[str]]:
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.set_title("MPC Axis Tracking")
        ax1.plot(h_r, label="R Axis")
        ax1.plot(h_z, label="Z Axis")
        ax1.axhline(target_vec[0], color="blue", linestyle="--", alpha=0.5, label="Target R")
        ax1.axhline(target_vec[1], color="orange", linestyle="--", alpha=0.5, label="Target Z")
        ax1.legend()
        ax1.grid(True)

        ax2.set_title("MPC Divertor (X-Point) Stabilization")
        ax2.plot(h_xr, label="X-Point R")
        ax2.plot(h_xz, label="X-Point Z")
        ax2.axhline(target_vec[2], color="blue", linestyle="--", alpha=0.5, label="Target XR")
        ax2.axhline(target_vec[3], color="orange", linestyle="--", alpha=0.5, label="Target XZ")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        return True, None
    except Exception as exc:
        return False, str(exc)


def run_sota_simulation(
    config_file: Optional[str] = None,
    shot_length: int = SHOT_LENGTH,
    prediction_horizon: int = PREDICTION_HORIZON,
    target_vector: Optional[np.ndarray] = None,
    disturbance_start_step: int = 20,
    disturbance_per_step_ma: float = 0.1,
    current_target_bounds: Tuple[float, float] = (5.0, 16.0),
    action_limit: float = 2.0,
    coil_current_limits: Tuple[float, float] = (-40.0, 40.0),
    save_plot: bool = True,
    output_path: str = "SOTA_MPC_Results.png",
    verbose: bool = True,
    kernel_factory: Callable[[str], Any] = FusionKernel,
) -> Dict[str, Any]:
    if config_file is None:
        repo_root = Path(__file__).resolve().parents[3]
        config_file = str(repo_root / "iter_config.json")

    lo_ip, hi_ip = _normalize_bounds(current_target_bounds, "current_target_bounds")
    lo_i, hi_i = _normalize_bounds(coil_current_limits, "coil_current_limits")
    steps = max(int(shot_length), 1)
    drift_start = max(int(disturbance_start_step), 0)
    drift_per = float(disturbance_per_step_ma)
    if target_vector is None:
        target_vec = np.array([6.0, 0.0, 5.0, -3.5], dtype=np.float64)
    else:
        target_vec = np.asarray(target_vector, dtype=np.float64).reshape(4)

    if verbose:
        print("\n--- SCPN FUSION SOTA: Neural-MPC Hybrid Control ---")

    kernel = kernel_factory(str(config_file))
    surrogate = NeuralSurrogate(
        n_coils=len(kernel.cfg["coils"]),
        n_state=4,
        verbose=verbose,
    )
    surrogate.train_on_kernel(kernel)

    mpc = ModelPredictiveController(
        surrogate,
        target_vec,
        prediction_horizon=prediction_horizon,
        action_limit=action_limit,
    )

    h_r: list[float] = []
    h_z: list[float] = []
    h_xr: list[float] = []
    h_xz: list[float] = []
    h_error: list[float] = []
    h_action: list[float] = []
    h_coil_abs: list[float] = []

    physics_cfg = kernel.cfg.setdefault("physics", {})
    target_ip_ma = float(np.clip(physics_cfg.get("plasma_current_target", lo_ip), lo_ip, hi_ip))
    physics_cfg["plasma_current_target"] = target_ip_ma

    if verbose:
        print(f"Starting {steps} step simulation with MPC Horizon={prediction_horizon}...")
    start_time = time.time()
    for t in range(steps):
        curr_state = surrogate.get_state(kernel)
        best_action = mpc.plan_trajectory(curr_state)
        h_action.append(float(np.max(np.abs(best_action))) if best_action.size else 0.0)

        for i, delta in enumerate(best_action):
            old_i = float(kernel.cfg["coils"][i].get("current", 0.0))
            kernel.cfg["coils"][i]["current"] = float(np.clip(old_i + float(delta), lo_i, hi_i))

        if t >= drift_start:
            target_ip_ma = float(np.clip(target_ip_ma + drift_per, lo_ip, hi_ip))
            physics_cfg["plasma_current_target"] = target_ip_ma

        kernel.solve_equilibrium()
        h_coil_abs.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(
                            [float(c.get("current", 0.0)) for c in kernel.cfg["coils"]],
                            dtype=np.float64,
                        )
                    )
                )
            )
        )

        h_r.append(float(curr_state[0]))
        h_z.append(float(curr_state[1]))
        h_xr.append(float(curr_state[2]))
        h_xz.append(float(curr_state[3]))
        err = float(np.linalg.norm(curr_state - target_vec))
        h_error.append(err)

        if verbose and t % 10 == 0:
            print(
                f"Step {t}: R={curr_state[0]:.2f}, Z={curr_state[1]:.2f} | "
                f"X-Point=({curr_state[2]:.2f},{curr_state[3]:.2f}) | Err={err:.3f}"
            )

    runtime_s = float(time.time() - start_time)
    if verbose:
        print(f"Simulation finished in {runtime_s:.2f}s")

    plot_saved = False
    plot_error: Optional[str] = None
    if save_plot:
        plot_saved, plot_error = _plot_telemetry(
            np.asarray(h_r, dtype=np.float64),
            np.asarray(h_z, dtype=np.float64),
            np.asarray(h_xr, dtype=np.float64),
            np.asarray(h_xz, dtype=np.float64),
            target_vec,
            output_path,
        )
        if verbose and plot_saved:
            print(f"SOTA Analysis saved: {output_path}")

    return {
        "config_path": str(config_file),
        "steps": int(steps),
        "prediction_horizon": int(prediction_horizon),
        "runtime_seconds": runtime_s,
        "final_target_ip_ma": float(target_ip_ma),
        "final_r_axis": float(h_r[-1]) if h_r else 0.0,
        "final_z_axis": float(h_z[-1]) if h_z else 0.0,
        "final_xpoint_r": float(h_xr[-1]) if h_xr else 0.0,
        "final_xpoint_z": float(h_xz[-1]) if h_xz else 0.0,
        "mean_tracking_error": float(np.mean(np.asarray(h_error, dtype=np.float64)))
        if h_error
        else 0.0,
        "max_abs_action": float(np.max(np.asarray(h_action, dtype=np.float64)))
        if h_action
        else 0.0,
        "max_abs_coil_current": float(np.max(np.asarray(h_coil_abs, dtype=np.float64)))
        if h_coil_abs
        else 0.0,
        "plot_saved": bool(plot_saved),
        "plot_error": plot_error,
    }


if __name__ == "__main__":
    run_sota_simulation()
