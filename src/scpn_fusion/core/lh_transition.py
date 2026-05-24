# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Zonal Flow Predator-Prey Model for L-H Transition
"""Predator-prey L-H transition model, threshold estimates, and control helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PredatorPreyResult:
    """Time traces and classified confinement regime from a predator-prey run."""

    epsilon_trace: np.ndarray
    V_ZF_trace: np.ndarray
    p_trace: np.ndarray
    time: np.ndarray
    regime: str


class PredatorPreyModel:
    """Zonal-flow predator-prey model for L-H transition dynamics."""

    def __init__(
        self,
        gamma_L: float = 5e4,
        alpha1: float = 1e-4,
        alpha2: float = 2e-8,
        alpha3: float = 1e-8,
        gamma_damp: float = 1e3,
        p0: float = 10.0,
        drive_gain: float = 100.0,
    ):
        """Initialize turbulence, zonal-flow, damping, and heating-drive parameters."""
        self.gamma_L = gamma_L
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.gamma_damp = gamma_damp
        self.p0 = float(p0)
        self.drive_gain = float(drive_gain)
        if not np.isfinite(self.p0) or self.p0 <= 0.0:
            raise ValueError("p0 must be finite and > 0.")
        if not np.isfinite(self.drive_gain) or self.drive_gain < 0.0:
            raise ValueError("drive_gain must be finite and >= 0.")

    def turbulence_drive(self, p_edge: float, Q_heating: float) -> float:
        """Return non-negative background turbulence forcing S_drive."""
        p = float(p_edge)
        qh = float(Q_heating)
        if not np.isfinite(p) or p < 0.0:
            raise ValueError("p_edge must be finite and non-negative.")
        if not np.isfinite(qh) or qh < 0.0:
            raise ValueError("Q_heating must be finite and non-negative.")
        pressure_factor = p / (self.p0 + p)
        heating_factor = qh / (1.0 + qh)
        return float(self.drive_gain * pressure_factor * heating_factor)

    def confinement_time(self, epsilon: float) -> float:
        """Return confinement time as turbulence energy suppresses transport."""
        # tau_E = tau_E0 / (1 + C * epsilon)
        tau_E0 = 1.0  # 1 second baseline
        C = 1e-4
        return tau_E0 / (1.0 + C * max(0.0, epsilon))

    def step(self, state: np.ndarray, dt: float, Q_heating: float) -> np.ndarray:
        """Advance turbulence, zonal-flow, and edge-pressure state by one step."""
        # state = [epsilon, V_ZF, p_edge]
        state = np.asarray(state, dtype=float)
        if state.shape != (3,) or not np.all(np.isfinite(state)):
            raise ValueError("state must be a finite vector [epsilon, V_ZF, p_edge]")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not np.isfinite(Q_heating) or Q_heating < 0.0:
            raise ValueError("Q_heating must be finite and non-negative")

        eps, V, p = state
        eps = max(0.0, eps)
        V = max(0.0, V)
        p = max(0.0, p)

        # d_eps/dt = gamma_L * (p/p0) * eps - alpha1 eps^2 - alpha2 eps V^2 + S_drive
        S_drive = self.turbulence_drive(p, Q_heating)
        d_eps = (
            self.gamma_L * (p / self.p0) * eps
            - self.alpha1 * eps**2
            - self.alpha2 * eps * V**2
            + S_drive
        )

        # dV/dt = alpha3 eps V - gamma_damp V
        d_V = self.alpha3 * eps * V - self.gamma_damp * V

        # dp/dt = Q - p / tau_E(eps)
        tau_e = self.confinement_time(eps)
        d_p = Q_heating - p / tau_e

        new_state = np.maximum(state + np.array([d_eps, d_V, d_p]) * dt, 0.0)
        return np.asarray(new_state)

    def evolve(
        self, Q_heating: float, t_span: tuple[float, float], dt: float
    ) -> PredatorPreyResult:
        """Integrate the predator-prey model and classify the resulting regime."""
        n_steps = int((t_span[1] - t_span[0]) / dt)
        t_arr = np.linspace(t_span[0], t_span[1], n_steps)

        state = np.array([1e4, 1.0, 1.0])  # L-mode initial state with small V_ZF seed

        eps_trace = np.zeros(n_steps)
        V_trace = np.zeros(n_steps)
        p_trace = np.zeros(n_steps)

        for i in range(n_steps):
            state = self.step(state, dt, Q_heating)
            eps_trace[i] = state[0]
            V_trace[i] = state[1]
            p_trace[i] = state[2]

        # Classify regime based on final state
        eps_final = state[0]
        V_final = state[1]

        if V_final > 100.0 and eps_final < 5e4:
            regime = "H_MODE"
        else:
            regime = "L_MODE"

        return PredatorPreyResult(eps_trace, V_trace, p_trace, t_arr, regime)


class LHTrigger:
    """Threshold search helper for L-to-H bifurcation scans."""

    def __init__(self, model: PredatorPreyModel):
        """Bind a predator-prey model used for threshold scans."""
        self.model = model

    def find_threshold(self, Q_range: np.ndarray) -> float:
        """Find Q at which L->H bifurcation occurs."""
        for Q in Q_range:
            res = self.model.evolve(Q, (0.0, 1.0), 0.001)
            if res.regime == "H_MODE":
                return float(Q)
        return float(Q_range[-1])


class MartinThreshold:
    """Martin scaling-law estimate for L-H power threshold."""

    @staticmethod
    def power_threshold_MW(ne_19: float, B_T: float, S_m2: float) -> float:
        """
        P_LH [MW] = 0.0488 * ne^0.717 * B^0.803 * S^0.941
        """
        if ne_19 <= 0 or B_T <= 0 or S_m2 <= 0:
            return 0.0
        return float(0.0488 * (ne_19**0.717) * (B_T**0.803) * (S_m2**0.941))


class IPhaseDetector:
    """Detect I-phase oscillations from recent turbulence-energy history."""

    def __init__(
        self,
        window_size: int = 100,
        *,
        relative_std_threshold: float = 0.1,
        min_absolute_std: float = 1.0e-6,
    ):
        """Initialize rolling-window and oscillation sensitivity parameters."""
        if not isinstance(window_size, int) or window_size < 2:
            raise ValueError("window_size must be an integer >= 2.")
        self.window_size = window_size
        self.relative_std_threshold = float(relative_std_threshold)
        self.min_absolute_std = float(min_absolute_std)
        if not np.isfinite(self.relative_std_threshold) or self.relative_std_threshold <= 0.0:
            raise ValueError("relative_std_threshold must be finite and > 0.")
        if not np.isfinite(self.min_absolute_std) or self.min_absolute_std < 0.0:
            raise ValueError("min_absolute_std must be finite and >= 0.")

    def detect(self, epsilon_trace: np.ndarray) -> bool:
        """True if limit cycle oscillations detected."""
        arr = np.asarray(epsilon_trace, dtype=float).reshape(-1)
        if arr.size < self.window_size:
            return False
        if not np.all(np.isfinite(arr)):
            raise ValueError("epsilon_trace must contain only finite values.")

        recent = arr[-self.window_size :]
        mean_val = np.mean(recent)
        std_val = np.std(recent)
        if std_val < self.min_absolute_std:
            return False

        return bool(mean_val > 0 and std_val / mean_val > self.relative_std_threshold)


class LHTransitionController:
    """Heating-command controller for H-mode access while avoiding I-phase."""

    def __init__(
        self,
        model: PredatorPreyModel,
        Q_target: float,
        *,
        epsilon_hmode_threshold: float = 5.0e4,
        q_ramp_rate: float = 10.0,
    ):
        """Initialize target heating, H-mode threshold, and ramp-rate limits."""
        if not np.isfinite(Q_target) or Q_target < 0.0:
            raise ValueError("Q_target must be finite and non-negative")
        if not np.isfinite(epsilon_hmode_threshold) or epsilon_hmode_threshold <= 0.0:
            raise ValueError("epsilon_hmode_threshold must be finite and > 0.")
        if not np.isfinite(q_ramp_rate) or q_ramp_rate <= 0.0:
            raise ValueError("q_ramp_rate must be finite and > 0.")
        self.model = model
        self.Q_target = Q_target
        self.epsilon_hmode_threshold = float(epsilon_hmode_threshold)
        self.q_ramp_rate = float(q_ramp_rate)
        self.detector = IPhaseDetector()
        self._epsilon_history: list[float] = []

    def step(self, epsilon_measured: float, Q_current: float, dt: float) -> float:
        """Ramp Q slowly until H-mode confirmed, avoid I-phase."""
        if not np.isfinite(epsilon_measured) or epsilon_measured < 0.0:
            raise ValueError("epsilon_measured must be finite and non-negative")
        if not np.isfinite(Q_current) or Q_current < 0.0:
            raise ValueError("Q_current must be finite and non-negative")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")

        self._epsilon_history.append(float(epsilon_measured))
        if len(self._epsilon_history) > self.detector.window_size:
            self._epsilon_history = self._epsilon_history[-self.detector.window_size :]

        if self.detector.detect(np.asarray(self._epsilon_history, dtype=float)):
            return min(float(Q_current), self.Q_target)

        if epsilon_measured < self.epsilon_hmode_threshold:
            return self.Q_target

        return min(float(Q_current + self.q_ramp_rate * dt), self.Q_target)
