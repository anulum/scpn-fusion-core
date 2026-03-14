# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Resistive Wall Mode Feedback
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations


import numpy as np


class RWMPhysics:
    """
    Resistive Wall Mode (RWM) growth rate model.
    """

    def __init__(self, beta_n: float, beta_n_nowall: float, beta_n_wall: float, tau_wall: float):
        self.beta_n = beta_n
        self.beta_n_nowall = beta_n_nowall
        self.beta_n_wall = beta_n_wall
        self.tau_wall = tau_wall

    def is_unstable(self) -> bool:
        """True if beta_N > beta_N_nowall (between limits)."""
        return self.beta_n_nowall < self.beta_n < self.beta_n_wall

    def growth_rate(self) -> float:
        """
        RWM growth rate [s^-1].
        gamma_wall = 1/tau_wall * (beta_n - beta_n_nowall) / (beta_n_wall - beta_n)
        If beta_n > beta_n_wall, it's an ideal kink, growth rate is very large.
        If beta_n < beta_n_nowall, it's stable, gamma = 0.
        """
        if self.tau_wall <= 0.0:
            # No wall -> instantly unstable if above no-wall limit
            return 1e6 if self.beta_n > self.beta_n_nowall else 0.0

        if self.beta_n >= self.beta_n_wall:
            return 1e6  # Ideal kink

        if self.beta_n <= self.beta_n_nowall:
            return 0.0  # Stable

        gamma = (
            (1.0 / self.tau_wall)
            * (self.beta_n - self.beta_n_nowall)
            / (self.beta_n_wall - self.beta_n)
        )
        return gamma


class RWMFeedbackController:
    """
    Sensor-coil feedback for RWM stabilization.
    """

    def __init__(
        self,
        n_sensors: int,
        n_coils: int,
        G_p: float,
        G_d: float,
        tau_controller: float = 1e-4,
        M_coil: float = 1.0,
    ):
        self.n_sensors = n_sensors
        self.n_coils = n_coils
        self.G_p = G_p
        self.G_d = G_d
        self.tau_controller = tau_controller
        self.M_coil = M_coil
        self.prev_B_r = np.zeros(n_sensors)

    def step(self, B_r_sensors: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute coil currents for feedback.
        """
        if dt <= 0.0:
            dB_dt = np.zeros_like(B_r_sensors)
        else:
            dB_dt = (B_r_sensors - self.prev_B_r) / dt

        self.prev_B_r = B_r_sensors.copy()

        # Simple mapping from sensors to coils (assume 1:1 or broadcast)
        if self.n_sensors == self.n_coils:
            I_coil = self.G_p * B_r_sensors + self.G_d * dB_dt
        else:
            # Average or project if different numbers
            I_mean = np.mean(self.G_p * B_r_sensors + self.G_d * dB_dt)
            I_coil = np.full(self.n_coils, I_mean)

        return I_coil

    def effective_growth_rate(self, rwm: RWMPhysics) -> float:
        """
        Compute closed-loop growth rate.
        gamma_eff = gamma_wall - G_p * M_coil * gamma_wall / (1 + gamma_wall * tau_controller)
        """
        gamma_wall = rwm.growth_rate()
        if gamma_wall == 0.0:
            return 0.0
        if gamma_wall >= 1e6:
            return gamma_wall  # Cannot stabilize ideal kink

        stabilization = (
            self.G_p * self.M_coil * gamma_wall / (1.0 + gamma_wall * self.tau_controller)
        )
        return gamma_wall - stabilization

    def is_stabilized(self, rwm: RWMPhysics) -> bool:
        """True if the effective growth rate is < 0."""
        return self.effective_growth_rate(rwm) < 0.0


class RWMStabilityAnalysis:
    """
    Analyze stability window and required gains.
    """

    @staticmethod
    def required_feedback_gain(
        beta_n: float,
        beta_n_nowall: float,
        beta_n_wall: float,
        tau_wall: float,
        tau_controller: float,
        M_coil: float = 1.0,
    ) -> float:
        """
        Find minimum G_p for stabilization.
        We need gamma_eff < 0 => G_p * M_coil / (1 + gamma_wall * tau_controller) > 1
        => G_p > (1 + gamma_wall * tau_controller) / M_coil
        """
        rwm = RWMPhysics(beta_n, beta_n_nowall, beta_n_wall, tau_wall)
        gamma_wall = rwm.growth_rate()
        if gamma_wall == 0.0:
            return 0.0
        if gamma_wall >= 1e6:
            return float("inf")

        min_G_p = (1.0 + gamma_wall * tau_controller) / M_coil
        return min_G_p
