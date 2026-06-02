# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Resistive Wall Mode Feedback
"""Resistive-wall-mode stability and feedback-control utilities.

The module exposes a compact physics/controls surface for stability-window tests:
RWM boundary growth estimation, sensor-feedback command generation, and required
gain calculations for closed-loop stabilization.
"""

from __future__ import annotations


import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class RWMPhysics:
    """
    Resistive Wall Mode (RWM) growth rate model.
    """

    def __init__(self, beta_n: float, beta_n_nowall: float, beta_n_wall: float, tau_wall: float):
        """Initialize normalized-beta limits and the resistive-wall time constant."""
        self.beta_n = beta_n
        self.beta_n_nowall = beta_n_nowall
        self.beta_n_wall = beta_n_wall
        self.tau_wall = tau_wall

    def is_unstable(self) -> bool:
        """Return whether the current normalized beta is between nowall and wall limits.

        Returns:
            True when ``beta_n_nowall < beta_n < beta_n_wall``.
        """
        return self.beta_n_nowall < self.beta_n < self.beta_n_wall

    def growth_rate(self) -> float:
        """Return the linear RWM growth-rate estimate in s^-1.

        The model returns a large sentinel value for ideal-kink conditions and
        zero for already-stable cases.

        Returns:
            float: Estimated wall-mode growth rate in hertz-equivalent units.
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
        """Initialize sensor/coil dimensions and proportional-derivative gains."""
        self.n_sensors = n_sensors
        self.n_coils = n_coils
        self.G_p = G_p
        self.G_d = G_d
        self.tau_controller = tau_controller
        self.M_coil = M_coil
        self.prev_B_r: FloatArray = np.zeros(n_sensors, dtype=np.float64)

    def step(self, B_r_sensors: FloatArray, dt: float) -> FloatArray:
        """
        Compute coil currents for feedback from radial-field samples.

        Args:
            B_r_sensors: Radial magnetic perturbation samples at each sensor.
            dt: Sample interval in seconds.

        Returns:
            Current commands for each control coil.
        """
        if dt <= 0.0:
            dB_dt: FloatArray = np.zeros_like(B_r_sensors, dtype=np.float64)
        else:
            dB_dt = np.asarray((B_r_sensors - self.prev_B_r) / dt, dtype=np.float64)

        self.prev_B_r = B_r_sensors.copy()

        # Simple mapping from sensors to coils (assume 1:1 or broadcast)
        if self.n_sensors == self.n_coils:
            I_coil: FloatArray = np.asarray(
                self.G_p * B_r_sensors + self.G_d * dB_dt, dtype=np.float64
            )
        else:
            # Average or project if different numbers
            I_mean = np.mean(self.G_p * B_r_sensors + self.G_d * dB_dt)
            I_coil = np.full(self.n_coils, I_mean, dtype=np.float64)

        return I_coil

    def effective_growth_rate(self, rwm: RWMPhysics) -> float:
        """Compute the closed-loop RWM growth rate with proportional feedback.

        Args:
            rwm: Physics state object that provides open-loop growth.

        Returns:
            Effective growth rate after feedback injection.
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
        """Calculate the minimum proportional gain required for stabilization.

        Args:
            beta_n: Plasma normalized beta.
            beta_n_nowall: No-wall marginal normalized beta.
            beta_n_wall: Ideal wall stability limit.
            tau_wall: Resistive wall time constant in seconds.
            tau_controller: Controller lag in seconds.
            M_coil: Effective coil coupling factor.

        Returns:
            Minimum ``G_p`` that makes ``gamma_eff < 0`` under this control
            contract.

        Raises:
            ValueError: Through the underlying :class:`RWMPhysics` calculations
                if input parameters are invalid.
        """
        rwm = RWMPhysics(beta_n, beta_n_nowall, beta_n_wall, tau_wall)
        gamma_wall = rwm.growth_rate()
        if gamma_wall == 0.0:
            return 0.0
        if gamma_wall >= 1e6:
            return float("inf")

        min_G_p = (1.0 + gamma_wall * tau_controller) / M_coil
        return min_G_p
