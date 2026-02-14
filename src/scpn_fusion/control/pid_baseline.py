# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
"""PID controller for vertical position stabilization.

This is the baseline controller to benchmark against the SNN controller.
PID gains can be set manually or auto-tuned via Ziegler-Nichols.

Reference gains for DIII-D-like plant (gamma=500, gain=1e6, dt=1e-4):
    Kp ~ -0.5, Ki ~ -0.05, Kd ~ -7e-4
    (negative because upward Z needs downward force)

Gain sizing rationale:
    The plant obeys Z'' = gamma^2 * Z + gain * u.  To place the
    closed-loop natural frequency at omega_cl ~ 500 rad/s (well
    resolved by dt = 100 us) with damping ratio zeta ~ 0.7:
        gain * |Kp| ~ gamma^2 + omega_cl^2   =>  Kp ~ -0.5
        gain * |Kd| ~ 2 * zeta * omega_cl    =>  Kd ~ -7e-4
        Ki chosen small to eliminate steady-state bias without
        introducing overshoot.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class PIDConfig:
    kp: float = -0.5       # Proportional gain [A/m]
    ki: float = -0.05      # Integral gain [A/(m*s)]
    kd: float = -7e-4      # Derivative gain [A*s/m]
    dt: float = 1e-4       # Time step [s]
    u_max: float = 10.0    # Output saturation [A]
    anti_windup: bool = True


class PIDController:
    """Discrete PID with anti-windup for vertical position control."""

    def __init__(self, config: PIDConfig | None = None):
        self.cfg = config or PIDConfig()
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0

    def compute(self, z_measured: float, dz_measured: float) -> float:
        """Compute control output. Target is Z = 0."""
        error = z_measured
        p_term = self.cfg.kp * error
        self.integral += error * self.cfg.dt
        i_term = self.cfg.ki * self.integral
        d_term = self.cfg.kd * dz_measured
        u = p_term + i_term + d_term
        u_sat = np.clip(u, -self.cfg.u_max, self.cfg.u_max)
        if self.cfg.anti_windup and u != u_sat:
            self.integral -= error * self.cfg.dt
        self.prev_error = error
        self.prev_output = u_sat
        return float(u_sat)


def tune_ziegler_nichols(gamma: float, gain: float, dt: float) -> PIDConfig:
    """Auto-tune PID via pole-placement inspired by the Ziegler-Nichols method.

    For the unstable plant ``Z'' = gamma^2 Z + gain u``, the classical
    Z-N relay test does not produce a limit cycle (the system diverges
    rather than oscillates).  Instead we use a pole-placement strategy
    that targets a closed-loop natural frequency ``omega_cl`` safely
    below the Nyquist rate ``pi / dt`` and a damping ratio ``zeta = 0.7``
    (quarter-decay per cycle, the Z-N design intent).

    Closed-loop characteristic equation (with PD action, no integral):
        s^2 + (-gain Kd) s + (-gain Kp - gamma^2) = 0
    Desired: s^2 + 2 zeta omega_cl s + omega_cl^2 = 0

    Matching coefficients:
        Kp = -(gamma^2 + omega_cl^2) / gain
        Kd = -2 zeta omega_cl / gain
        Ki = small fraction of Kp / T to remove steady-state bias
    """
    # Target closed-loop frequency: 10x the growth rate or 1/20 of
    # Nyquist, whichever is smaller, so the discrete integrator stays
    # well-resolved.
    nyquist = np.pi / dt
    omega_cl = min(10.0 * gamma, nyquist / 20.0)
    zeta = 0.7

    kp = -(gamma**2 + omega_cl**2) / gain
    kd = -2.0 * zeta * omega_cl / gain
    # Integral gain: remove steady-state bias on a time scale of ~5
    # natural periods.
    ki = kp / (5.0 * 2.0 * np.pi / omega_cl)
    return PIDConfig(kp=kp, ki=ki, kd=kd, dt=dt)
