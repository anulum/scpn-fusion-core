# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
"""Linearized vertical position dynamics for elongated tokamak.

Physics: An elongated tokamak plasma is vertically unstable — a small
upward displacement creates a net upward force (positive feedback).
The control system must actively counteract this with external coils.

Model:
    m_eff * Z'' = -K_s * Z + K_ctrl * ΔI_ctrl + F_noise

Nondimensionalized:
    Z'' = γ² * Z + g * u + noise
    Where u = ΔI_ctrl (normalized), g = K_ctrl / m_eff
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class PlantConfig:
    """Vertical stability plant parameters."""
    gamma: float = 500.0       # Growth rate [1/s] (DIII-D-like: 200-800)
    gain: float = 1e6          # Control effectiveness [m/s²/A]
    dt: float = 1e-4           # Time step [s] = 100 µs
    noise_std: float = 1e-3    # Noise std dev [m/s²]
    u_max: float = 10.0        # Actuator saturation [A]
    z_max: float = 0.1         # Safety limit [m] — disruption if exceeded
    sensor_noise_std: float = 1e-4  # Measurement noise [m]
    sensor_delay_steps: int = 0     # Measurement delay [steps]


class VerticalStabilityPlant:
    """Linearized vertical position dynamics."""

    def __init__(self, config: PlantConfig | None = None):
        self.cfg = config or PlantConfig()
        self.z = 0.0
        self.dz = 0.0
        self.t = 0.0
        self.step_count = 0
        self._rng = np.random.default_rng(42)
        self._sensor_buffer: list[float] = []

    def reset(self, z0: float = 0.0, dz0: float = 0.0) -> None:
        self.z = z0
        self.dz = dz0
        self.t = 0.0
        self.step_count = 0
        self._sensor_buffer.clear()

    def step(self, u: float) -> tuple[float, float]:
        """Advance one timestep. Returns (z_measured, dz_measured)."""
        u_clamped = np.clip(u, -self.cfg.u_max, self.cfg.u_max)
        noise = self._rng.normal(0.0, self.cfg.noise_std)
        ddz = self.cfg.gamma**2 * self.z + self.cfg.gain * u_clamped + noise
        # Symplectic Euler
        self.dz += ddz * self.cfg.dt
        self.z += self.dz * self.cfg.dt
        self.t += self.cfg.dt
        self.step_count += 1
        # Sensor with noise
        z_meas = self.z + self._rng.normal(0.0, self.cfg.sensor_noise_std)
        dz_meas = self.dz + self._rng.normal(0.0, self.cfg.sensor_noise_std * 10)
        # Optional delay
        if self.cfg.sensor_delay_steps > 0:
            self._sensor_buffer.append(z_meas)
            if len(self._sensor_buffer) > self.cfg.sensor_delay_steps:
                z_meas = self._sensor_buffer.pop(0)
            else:
                z_meas = 0.0
        return z_meas, dz_meas

    @property
    def disrupted(self) -> bool:
        return abs(self.z) > self.cfg.z_max

    @property
    def growth_time_ms(self) -> float:
        return 1000.0 / self.cfg.gamma
