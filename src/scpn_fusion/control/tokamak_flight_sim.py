# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Tokamak Flight Sim
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel

# --- FLIGHT PARAMETERS ---
SHOT_DURATION = 50 # Time steps
TARGET_R = 6.2     # Target Major Radius
TARGET_Z = 0.0     # Target Vertical Position
TARGET_ELONGATION = 1.7 


class FirstOrderActuator:
    """Discrete first-order actuator with rate limits, noise, and delay.

    Models a realistic coil power supply for tokamak control:
    - First-order lag: u_applied(s) = 1/(tau*s+1) * u_cmd
    - Coil current rate limit: |du/dt| <= rate_limit [A/s]
    - Sensor noise: additive Gaussian on measurement
    - Measurement delay: pure transport delay on feedback signal

    Parameters
    ----------
    tau_s : float
        Actuator time constant [s].
    dt_s : float
        Simulation timestep [s].
    u_min, u_max : float
        Saturation limits.
    rate_limit : float
        Maximum current rate of change [A/s]. Default 1e6 (1 MA/s, ITER PF spec).
    sensor_noise_std : float
        Standard deviation of additive sensor noise. Default 0.0 (disabled).
    delay_steps : int
        Number of timesteps of measurement delay. Default 0.
    rng_seed : int or None
        Random seed for reproducible noise (None = random).
    """

    def __init__(
        self,
        *,
        tau_s: float,
        dt_s: float,
        u_min: float = -1.0e9,
        u_max: float = 1.0e9,
        rate_limit: float = 1.0e6,
        sensor_noise_std: float = 0.0,
        delay_steps: int = 0,
        rng_seed: Optional[int] = None,
    ) -> None:
        tau_s = float(tau_s)
        dt_s = float(dt_s)
        if not np.isfinite(tau_s) or tau_s <= 0.0:
            raise ValueError("tau_s must be finite and > 0.")
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and > 0.")
        self.tau_s = tau_s
        self.dt_s = dt_s
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.rate_limit = float(rate_limit)
        self.sensor_noise_std = float(sensor_noise_std)
        self.delay_steps = int(delay_steps)
        self._rng = np.random.default_rng(rng_seed)
        self.state = 0.0
        self._delay_buffer: list[float] = [0.0] * max(self.delay_steps, 1)

    def step(self, command: float) -> float:
        """Apply command through actuator dynamics with rate limiting."""
        u_cmd = float(np.clip(command, self.u_min, self.u_max))
        alpha = self.dt_s / (self.tau_s + self.dt_s)
        u_new = self.state + alpha * (u_cmd - self.state)

        # Rate limiting (coil current slew rate)
        du = u_new - self.state
        max_du = self.rate_limit * self.dt_s
        if abs(du) > max_du:
            du = np.sign(du) * max_du
            u_new = self.state + du

        self.state = float(np.clip(u_new, self.u_min, self.u_max))

        # Update delay buffer
        self._delay_buffer.append(self.state)

        return self.state

    def get_measurement(self) -> float:
        """Return delayed, noisy measurement of actuator output."""
        idx = max(0, len(self._delay_buffer) - 1 - self.delay_steps)
        delayed = self._delay_buffer[idx]

        if self.sensor_noise_std > 0:
            noise = float(self._rng.normal(0.0, self.sensor_noise_std))
            return delayed + noise
        return delayed

class IsoFluxController:
    """
    Simulates the Plasma Control System (PCS).
    Uses PID loops to adjust Coil Currents to maintain plasma shape.
    """
    def __init__(
        self,
        config_file: str,
        kernel_factory: Callable[[str], Any] = FusionKernel,
        verbose: bool = True,
        actuator_tau_s: float = 0.06,
        heating_actuator_tau_s: Optional[float] = None,
        actuator_current_delta_limit: float = 1.0e9,
        heating_beta_max: float = 5.0,
        control_dt_s: float = 0.05,
    ) -> None:
        self.kernel = kernel_factory(config_file)
        self.verbose = bool(verbose)
        self.history = {
            't': [],
            'Ip': [],
            'R_axis': [],
            'Z_axis': [],
            'X_point': [],
            'ctrl_R_cmd': [],
            'ctrl_R_applied': [],
            'ctrl_Z_cmd': [],
            'ctrl_Z_applied': [],
            'beta_cmd': [],
            'beta_applied': [],
        }
        control_dt_s = float(control_dt_s)
        if not np.isfinite(control_dt_s) or control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be finite and > 0.")
        self.control_dt_s = control_dt_s
        actuator_current_delta_limit = float(actuator_current_delta_limit)
        if (
            not np.isfinite(actuator_current_delta_limit)
            or actuator_current_delta_limit <= 0.0
        ):
            raise ValueError("actuator_current_delta_limit must be finite and > 0.")
        heating_beta_max = float(heating_beta_max)
        if not np.isfinite(heating_beta_max) or heating_beta_max <= 1.0:
            raise ValueError("heating_beta_max must be finite and > 1.0.")
        if heating_actuator_tau_s is None:
            heating_actuator_tau_s = float(actuator_tau_s)
        heating_actuator_tau_s = float(heating_actuator_tau_s)
        if not np.isfinite(heating_actuator_tau_s) or heating_actuator_tau_s <= 0.0:
            raise ValueError("heating_actuator_tau_s must be finite and > 0.")
        
        # PID Gains for Position Control
        # Radial Control (Horizontal) -> Controlled by Outer Coils (PF2, PF3, PF4)
        self.pid_R = {'Kp': 2.0, 'Ki': 0.1, 'Kd': 0.5, 'err_sum': 0, 'last_err': 0}
        
        # Vertical Control (Z-pos) -> Controlled by Top/Bottom diff (PF1 vs PF5)
        self.pid_Z = {'Kp': 5.0, 'Ki': 0.2, 'Kd': 2.0, 'err_sum': 0, 'last_err': 0}

        self._act_radial = FirstOrderActuator(
            tau_s=actuator_tau_s,
            dt_s=self.control_dt_s,
            u_min=-actuator_current_delta_limit,
            u_max=actuator_current_delta_limit,
        )
        self._act_top = FirstOrderActuator(
            tau_s=actuator_tau_s,
            dt_s=self.control_dt_s,
            u_min=-actuator_current_delta_limit,
            u_max=actuator_current_delta_limit,
        )
        self._act_bottom = FirstOrderActuator(
            tau_s=actuator_tau_s,
            dt_s=self.control_dt_s,
            u_min=-actuator_current_delta_limit,
            u_max=actuator_current_delta_limit,
        )
        self._act_heating = FirstOrderActuator(
            tau_s=heating_actuator_tau_s,
            dt_s=self.control_dt_s,
            u_min=1.0,
            u_max=heating_beta_max,
        )

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def pid_step(self, pid: Dict[str, float], error: float) -> float:
        pid['err_sum'] += error
        d_err = error - pid['last_err']
        pid['last_err'] = error
        return (pid['Kp'] * error) + (pid['Ki'] * pid['err_sum']) + (pid['Kd'] * d_err)

    def _add_coil_current(self, coil_idx: int, delta: float) -> None:
        coils = self.kernel.cfg.get("coils", [])
        if 0 <= coil_idx < len(coils):
            current = float(coils[coil_idx].get("current", 0.0))
            coils[coil_idx]["current"] = current + float(delta)

    def run_shot(
        self,
        shot_duration: int = SHOT_DURATION,
        save_plot: bool = True,
        output_path: str = "Tokamak_Flight_Report.png",
    ) -> Dict[str, Any]:
        steps = int(shot_duration)
        if steps < 1:
            raise ValueError("shot_duration must be >= 1.")
        self._log("--- INITIATING TOKAMAK FLIGHT SIMULATOR ---")
        self._log("Scenario: Current Ramp-Up & Divertor Formation")
        
        # Initial Solve
        self.kernel.solve_equilibrium()
        
        # Physics Evolution Loop
        for t in range(steps):
            # 1. EVOLVE PHYSICS (Scenario)
            # Ramp up plasma current
            target_Ip = 5.0 + (10.0 * t / steps) # 5MA -> 15MA
            physics_cfg = self.kernel.cfg.setdefault("physics", {})
            physics_cfg['plasma_current_target'] = target_Ip
            
            # Increase Pressure (Heating) -> This pushes plasma outward (Shafranov Shift)
            # The controller must fight this drift!
            beta_cmd = 1.0 + (0.05 * t)
            beta_applied = self._act_heating.step(beta_cmd)
            physics_cfg['beta_scale'] = beta_applied
            
            # 2. MEASURE STATE (Diagnostics)
            # Find current axis
            idx_max = np.argmax(self.kernel.Psi)
            iz, ir = np.unravel_index(idx_max, self.kernel.Psi.shape)
            curr_R = self.kernel.R[ir]
            curr_Z = self.kernel.Z[iz]
            
            # Find X-point (Divertor)
            xp_pos, _ = self.kernel.find_x_point(self.kernel.Psi)
            
            # 3. COMPUTE CONTROL (Iso-Flux)
            err_R = TARGET_R - curr_R
            err_Z = TARGET_Z - curr_Z
            
            # Control Actions (Current Deltas)
            ctrl_radial_cmd = self.pid_step(self.pid_R, err_R)
            ctrl_vertical_cmd = self.pid_step(self.pid_Z, err_Z)

            # First-order actuator transfer layer (power-supply lag / inductance).
            ctrl_radial = self._act_radial.step(ctrl_radial_cmd)
            ctrl_vertical_top = self._act_top.step(-ctrl_vertical_cmd)
            ctrl_vertical_bottom = self._act_bottom.step(ctrl_vertical_cmd)
            ctrl_vertical_applied = 0.5 * (ctrl_vertical_bottom - ctrl_vertical_top)
            
            # 4. ACTUATE COILS (Map Control -> Coils)
            # Radial Correction: If R is too small (Inner), Push with Outer Coils
            # PF3 is the main pusher
            self._add_coil_current(2, ctrl_radial)
            
            # Vertical Correction: Differential pull
            self._add_coil_current(0, ctrl_vertical_top) # Top
            self._add_coil_current(4, ctrl_vertical_bottom) # Bottom
            
            # 5. SOLVE NEW EQUILIBRIUM
            # We use the previous solution as guess (hot start) -> Faster
            self.kernel.solve_equilibrium()
            
            # Log
            self.history['t'].append(t)
            self.history['Ip'].append(target_Ip)
            self.history['R_axis'].append(curr_R)
            self.history['Z_axis'].append(curr_Z)
            self.history['X_point'].append(xp_pos)
            self.history['ctrl_R_cmd'].append(ctrl_radial_cmd)
            self.history['ctrl_R_applied'].append(ctrl_radial)
            self.history['ctrl_Z_cmd'].append(ctrl_vertical_cmd)
            self.history['ctrl_Z_applied'].append(ctrl_vertical_applied)
            self.history['beta_cmd'].append(beta_cmd)
            self.history['beta_applied'].append(beta_applied)
            
            self._log(
                f"Time {t}: Ip={target_Ip:.1f}MA | "
                f"Axis=({curr_R:.2f}, {curr_Z:.2f}) | Ctrl_R={ctrl_radial:.2f}"
            )

        plot_saved = False
        plot_error = None
        if save_plot:
            plot_saved, plot_error = self.visualize_flight(output_path=output_path)

        final_axis_r = float(self.history['R_axis'][-1]) if self.history['R_axis'] else 0.0
        final_axis_z = float(self.history['Z_axis'][-1]) if self.history['Z_axis'] else 0.0
        final_ip_ma = float(self.history['Ip'][-1]) if self.history['Ip'] else 0.0
        mean_abs_r_error = (
            float(np.mean(np.abs(np.asarray(self.history['R_axis']) - TARGET_R)))
            if self.history['R_axis']
            else 0.0
        )
        mean_abs_z_error = (
            float(np.mean(np.abs(np.asarray(self.history['Z_axis']) - TARGET_Z)))
            if self.history['Z_axis']
            else 0.0
        )
        mean_abs_radial_actuator_lag = (
            float(
                np.mean(
                    np.abs(
                        np.asarray(self.history['ctrl_R_cmd'], dtype=np.float64)
                        - np.asarray(self.history['ctrl_R_applied'], dtype=np.float64)
                    )
                )
            )
            if self.history['ctrl_R_cmd']
            else 0.0
        )
        mean_abs_vertical_actuator_lag = (
            float(
                np.mean(
                    np.abs(
                        np.asarray(self.history['ctrl_Z_cmd'], dtype=np.float64)
                        - np.asarray(self.history['ctrl_Z_applied'], dtype=np.float64)
                    )
                )
            )
            if self.history['ctrl_Z_cmd']
            else 0.0
        )
        mean_abs_heating_actuator_lag = (
            float(
                np.mean(
                    np.abs(
                        np.asarray(self.history['beta_cmd'], dtype=np.float64)
                        - np.asarray(self.history['beta_applied'], dtype=np.float64)
                    )
                )
            )
            if self.history['beta_cmd']
            else 0.0
        )
        final_beta_scale = (
            float(self.history['beta_applied'][-1]) if self.history['beta_applied'] else 1.0
        )
        return {
            "steps": int(steps),
            "final_ip_ma": final_ip_ma,
            "final_axis_r": final_axis_r,
            "final_axis_z": final_axis_z,
            "final_beta_scale": final_beta_scale,
            "mean_abs_r_error": mean_abs_r_error,
            "mean_abs_z_error": mean_abs_z_error,
            "mean_abs_radial_actuator_lag": mean_abs_radial_actuator_lag,
            "mean_abs_vertical_actuator_lag": mean_abs_vertical_actuator_lag,
            "mean_abs_heating_actuator_lag": mean_abs_heating_actuator_lag,
            "plot_saved": bool(plot_saved),
            "plot_error": plot_error,
        }

    def visualize_flight(
        self,
        output_path: str = "Tokamak_Flight_Report.png",
    ) -> Tuple[bool, Optional[str]]:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            return False, f"matplotlib unavailable: {exc}"
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # 1. Trajectory Plot
            ax1.set_title("Plasma Trajectory Control")
            ax1.plot(self.history['t'], self.history['R_axis'], 'b-', label='R Axis (Radial)')
            ax1.plot(self.history['t'], self.history['Z_axis'], 'r-', label='Z Axis (Vertical)')

            # Targets
            ax1.axhline(TARGET_R, color='b', linestyle='--', alpha=0.5, label='Target R')
            ax1.axhline(TARGET_Z, color='r', linestyle='--', alpha=0.5, label='Target Z')

            ax1.set_xlabel("Shot Time (a.u.)")
            ax1.set_ylabel("Position (m)")
            ax1.legend()
            ax1.grid(True)

            # 2. X-Point Evolution (Divertor Stability)
            rx = [p[0] for p in self.history['X_point']]
            rz = [p[1] for p in self.history['X_point']]

            # Filter out 0,0 (Limiter phase)
            valid_idx = [i for i, x in enumerate(rx) if x > 1.0]
            if valid_idx:
                ax2.plot([rx[i] for i in valid_idx], [rz[i] for i in valid_idx], 'g-o', markersize=4)
                ax2.set_title("Divertor X-Point Movement")
                ax2.set_xlabel("R (m)")
                ax2.set_ylabel("Z (m)")
                ax2.grid(True)

                # Draw final shape if available from kernel implementation.
                if hasattr(self.kernel, "RR") and hasattr(self.kernel, "ZZ"):
                    ax2.contour(
                        self.kernel.RR,
                        self.kernel.ZZ,
                        self.kernel.Psi,
                        levels=10,
                        colors='k',
                        alpha=0.2,
                    )
            else:
                ax2.text(0.5, 0.5, "Plasma Remained Limited (No Divertor)", ha='center')

            plt.tight_layout()
            plt.savefig(output_path)
            plt.close(fig)
            self._log(f"Flight Sim Complete. Report: {output_path}")
            return True, None
        except Exception as exc:
            return False, str(exc)


def run_flight_sim(
    config_file: Optional[str] = None,
    shot_duration: int = SHOT_DURATION,
    seed: int = 42,
    save_plot: bool = True,
    output_path: str = "Tokamak_Flight_Report.png",
    verbose: bool = True,
    actuator_tau_s: float = 0.06,
    heating_actuator_tau_s: Optional[float] = None,
    actuator_current_delta_limit: float = 1.0e9,
    heating_beta_max: float = 5.0,
    control_dt_s: float = 0.05,
    kernel_factory: Callable[[str], Any] = FusionKernel,
) -> Dict[str, Any]:
    """Run deterministic tokamak flight-sim control loop and return summary."""
    seed_int = int(seed)
    if config_file is None:
        repo_root = Path(__file__).resolve().parents[3]
        config_file = str(repo_root / "iter_config.json")

    sim = IsoFluxController(
        config_file=str(config_file),
        kernel_factory=kernel_factory,
        verbose=verbose,
        actuator_tau_s=actuator_tau_s,
        heating_actuator_tau_s=heating_actuator_tau_s,
        actuator_current_delta_limit=actuator_current_delta_limit,
        heating_beta_max=heating_beta_max,
        control_dt_s=control_dt_s,
    )
    summary = sim.run_shot(
        shot_duration=shot_duration,
        save_plot=save_plot,
        output_path=output_path,
    )
    summary["seed"] = seed_int
    summary["config_path"] = str(config_file)
    return summary

if __name__ == "__main__":
    run_flight_sim()
