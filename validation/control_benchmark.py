# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
"""Benchmark scenarios and metrics for vertical position control.

Six scenarios test different aspects of controller performance.
Metrics are computed identically for PID and SNN controllers.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class PlantConfig:
    """Vertical stability plant parameters."""
    gamma: float = 500.0       # Growth rate [1/s]
    gain: float = 1e6          # Control effectiveness [m/s²/A]
    dt: float = 1e-4           # Time step [s] = 100 µs
    noise_std: float = 1e-3    # Noise std dev [m/s²]
    u_max: float = 10.0        # Actuator saturation [A]
    z_max: float = 0.1         # Safety limit [m]
    sensor_noise_std: float = 1e-4  # Measurement noise [m]
    sensor_delay_steps: int = 0     # Measurement delay [steps]


@dataclass
class Scenario:
    """A single benchmark scenario definition."""
    name: str
    description: str
    plant_config: PlantConfig
    z0: float = 0.005          # Initial displacement [m]
    dz0: float = 0.0           # Initial velocity [m/s]
    duration_s: float = 0.1    # 100ms simulation
    disturbance_fn: str = "none"  # "none", "ramp", "random", "dropout"


# ---------------------------------------------------------------------------
# Six canonical benchmark scenarios
# ---------------------------------------------------------------------------
SCENARIOS: list[Scenario] = [
    Scenario(
        name="step_5mm",
        description="5mm instantaneous displacement, nominal plant",
        plant_config=PlantConfig(gamma=500, noise_std=0),
        z0=0.005,
    ),
    Scenario(
        name="step_noisy",
        description="5mm displacement with process and sensor noise",
        plant_config=PlantConfig(gamma=500, noise_std=1e-3, sensor_noise_std=1e-4),
        z0=0.005,
    ),
    Scenario(
        name="ramp_disturbance",
        description="Ramp force disturbance (linearly increasing external force)",
        plant_config=PlantConfig(gamma=500, noise_std=0),
        z0=0.0,
        disturbance_fn="ramp",
    ),
    Scenario(
        name="random_perturbation",
        description="Continuous random Gaussian perturbations (10mm std)",
        plant_config=PlantConfig(gamma=500, noise_std=0.01),
        z0=0.0,
        duration_s=0.5,
    ),
    Scenario(
        name="plant_uncertainty",
        description="±20% gamma, ±30% gain uncertainty",
        plant_config=PlantConfig(gamma=600, gain=0.7e6, noise_std=1e-3),
        z0=0.005,
    ),
    Scenario(
        name="sensor_dropout",
        description="50ms sensor measurement dropout (stuck at last value)",
        plant_config=PlantConfig(gamma=500, noise_std=1e-3, sensor_delay_steps=500),
        z0=0.005,
    ),
]


@dataclass
class BenchmarkResult:
    """Metrics for a single controller on a single scenario."""
    controller_name: str
    scenario_name: str
    settling_time_ms: float
    max_overshoot_mm: float
    steady_state_error_mm: float
    rms_control_effort: float
    peak_control_effort: float
    disrupted: bool
    trajectory_z: list[float] = field(default_factory=list)
    trajectory_u: list[float] = field(default_factory=list)
    wall_time_us: float = 0.0

    def to_dict(self) -> dict:
        """Serialise result to a flat dictionary (excludes raw trajectories)."""
        return {
            "controller": self.controller_name,
            "scenario": self.scenario_name,
            "settling_time_ms": self.settling_time_ms,
            "max_overshoot_mm": self.max_overshoot_mm,
            "steady_state_error_mm": self.steady_state_error_mm,
            "rms_control_effort": self.rms_control_effort,
            "peak_control_effort": self.peak_control_effort,
            "disrupted": self.disrupted,
            "wall_time_us_per_step": self.wall_time_us,
        }


def compute_metrics(
    z_traj: np.ndarray,
    u_traj: np.ndarray,
    dt: float,
    z_max: float,
) -> dict:
    """Compute benchmark metrics from trajectory arrays.

    Parameters
    ----------
    z_traj : np.ndarray
        Vertical displacement trajectory [m], shape ``(N,)``.
    u_traj : np.ndarray
        Control signal trajectory [A], shape ``(N,)``.
    dt : float
        Time step [s].
    z_max : float
        Safety displacement limit [m].

    Returns
    -------
    dict
        settling_time_ms : float
            Time in milliseconds at which ``|z|`` first drops below 1 mm
            and remains below 1 mm for the rest of the simulation.
            If settling never occurs, returns the full duration.
        max_overshoot_mm : float
            Maximum ``|z|`` over the entire simulation, in millimetres.
        steady_state_error_mm : float
            Mean ``|z|`` in the last 20 % of the simulation, in millimetres.
        rms_control_effort : float
            RMS of the control signal over the simulation.
        peak_control_effort : float
            Peak ``|u|`` over the simulation.
        disrupted : bool
            ``True`` if ``|z|`` exceeded *z_max* at any point.
    """
    z_traj = np.asarray(z_traj, dtype=np.float64)
    u_traj = np.asarray(u_traj, dtype=np.float64)

    n = len(z_traj)

    # --- disrupted ---------------------------------------------------------
    abs_z = np.abs(z_traj)
    disrupted = bool(np.any(abs_z > z_max))

    # --- max overshoot [mm] ------------------------------------------------
    max_overshoot_mm = float(np.max(abs_z) * 1000.0)

    # --- settling time [ms] ------------------------------------------------
    settling_threshold = 1e-3  # 1 mm
    settled_mask = abs_z < settling_threshold

    # Find the first index from which the trajectory stays below threshold
    # until the end.  Walk backwards to find the last index that is NOT
    # settled, then settling index = that + 1.
    if n == 0:
        settling_time_ms = 0.0
    elif np.all(settled_mask):
        # Already settled from the very start
        settling_time_ms = 0.0
    else:
        # Indices where abs_z >= threshold
        unsettled_indices = np.where(~settled_mask)[0]
        if len(unsettled_indices) == 0:
            settling_time_ms = 0.0
        else:
            last_unsettled = unsettled_indices[-1]
            if last_unsettled >= n - 1:
                # Never settled (last sample still above threshold)
                settling_time_ms = float(n * dt * 1000.0)
            else:
                settling_index = last_unsettled + 1
                settling_time_ms = float(settling_index * dt * 1000.0)

    # --- steady-state error [mm] -------------------------------------------
    tail_len = max(1, n // 5)  # last 20 %
    steady_state_error_mm = float(np.mean(abs_z[-tail_len:]) * 1000.0)

    # --- control effort ----------------------------------------------------
    if len(u_traj) == 0:
        rms_control_effort = 0.0
        peak_control_effort = 0.0
    else:
        rms_control_effort = float(np.sqrt(np.mean(u_traj ** 2)))
        peak_control_effort = float(np.max(np.abs(u_traj)))

    return {
        "settling_time_ms": settling_time_ms,
        "max_overshoot_mm": max_overshoot_mm,
        "steady_state_error_mm": steady_state_error_mm,
        "rms_control_effort": rms_control_effort,
        "peak_control_effort": peak_control_effort,
        "disrupted": disrupted,
    }
