# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Zonal Flow Predator-Prey Model for L-H Transition
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PredatorPreyResult:
    epsilon_trace: np.ndarray
    V_ZF_trace: np.ndarray
    p_trace: np.ndarray
    time: np.ndarray
    regime: str


class PredatorPreyModel:
    def __init__(
        self,
        gamma_L: float = 5e4,
        alpha1: float = 1e-4,
        alpha2: float = 2e-8,
        alpha3: float = 1e-8,
        gamma_damp: float = 1e3,
    ):
        self.gamma_L = gamma_L
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.gamma_damp = gamma_damp

    def confinement_time(self, epsilon: float) -> float:
        # tau_E = tau_E0 / (1 + C * epsilon)
        tau_E0 = 1.0  # 1 second baseline
        C = 1e-4
        return tau_E0 / (1.0 + C * max(0.0, epsilon))

    def step(self, state: np.ndarray, dt: float, Q_heating: float) -> np.ndarray:
        # state = [epsilon, V_ZF, p_edge]
        eps, V, p = state
        eps = max(0.0, eps)
        V = max(0.0, V)
        p = max(0.0, p)

        # Drive scales with pressure gradient, simplified here to scale with p
        # Actually gamma_L should scale with p. We use a threshold model.
        # d_eps/dt = gamma_L * (p/p0) * eps - alpha1 eps^2 - alpha2 eps V^2 + S_drive
        p0 = 10.0
        S_drive = 1e2  # Background noise drive
        d_eps = (
            self.gamma_L * (p / p0) * eps
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
    def __init__(self, model: PredatorPreyModel):
        self.model = model

    def find_threshold(self, Q_range: np.ndarray) -> float:
        """Find Q at which L->H bifurcation occurs."""
        for Q in Q_range:
            res = self.model.evolve(Q, (0.0, 1.0), 0.001)
            if res.regime == "H_MODE":
                return float(Q)
        return float(Q_range[-1])


class MartinThreshold:
    @staticmethod
    def power_threshold_MW(ne_19: float, B_T: float, S_m2: float) -> float:
        """
        P_LH [MW] = 0.0488 * ne^0.717 * B^0.803 * S^0.941
        """
        if ne_19 <= 0 or B_T <= 0 or S_m2 <= 0:
            return 0.0
        return float(0.0488 * (ne_19**0.717) * (B_T**0.803) * (S_m2**0.941))


class IPhaseDetector:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size

    def detect(self, epsilon_trace: np.ndarray) -> bool:
        """True if limit cycle oscillations detected."""
        if len(epsilon_trace) < self.window_size:
            return False

        recent = epsilon_trace[-self.window_size :]
        # Simple heuristic: strong oscillation -> std > 10% of mean
        mean_val = np.mean(recent)
        std_val = np.std(recent)

        return bool(mean_val > 0 and std_val / mean_val > 0.1)


class LHTransitionController:
    def __init__(self, model: PredatorPreyModel, Q_target: float):
        self.model = model
        self.Q_target = Q_target
        self.detector = IPhaseDetector()

    def step(self, epsilon_measured: float, Q_current: float, dt: float) -> float:
        """Ramp Q slowly until H-mode confirmed, avoid I-phase."""
        # Highly simplified heuristic logic
        if epsilon_measured < 5e4:  # Assumed H-mode
            return self.Q_target

        # Still in L-mode or I-phase, keep ramping
        return Q_current + 10.0 * dt
