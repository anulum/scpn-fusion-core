# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Divertor Detachment Control
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from scpn_fusion.core.sol_model import TwoPointSOL


class DetachmentState(Enum):
    ATTACHED = auto()
    PARTIALLY_DETACHED = auto()
    FULLY_DETACHED = auto()
    XPOINT_MARFE = auto()


class RadiationFrontModel:
    def __init__(self, impurity: str, R0: float, a: float, q95: float):
        self.impurity = impurity
        self.R0 = R0
        self.a = a
        self.q95 = q95

    def radiation_temperature(self, impurity: str) -> float:
        """Peak radiation temperature [eV]."""
        temps = {"N2": 10.0, "Ne": 30.0, "Ar": 100.0}
        return temps.get(impurity, 10.0)

    def front_position(self, P_SOL_MW: float, n_u_19: float, seeding_rate: float) -> float:
        """
        Location of the radiation front: 0 = target, 1 = X-point.
        Higher seeding -> moves towards X-point.
        Higher P_SOL -> pushes front back to target.
        """
        if P_SOL_MW <= 0.0:
            return 1.0

        # Simplistic empirical mapping for front movement
        drive = seeding_rate * n_u_19 / P_SOL_MW

        # Sigmoid shape
        rho_front = 1.0 - np.exp(-drive * 2.0)
        return float(np.clip(rho_front, 0.0, 1.0))

    def degree_of_detachment(self, T_target_eV: float, n_target: float, n_u: float) -> float:
        """
        DOD = Gamma_t_attached / Gamma_t_actual
        We approximate this via T_target using the classic two-point rollover.
        """
        if T_target_eV > 5.0:
            return 1.0
        elif T_target_eV <= 0.1:
            return 10.0
        else:
            # Below 5 eV, DOD grows rapidly
            return 1.0 + 5.0 * (1.0 - T_target_eV / 5.0)


class DetachmentController:
    def __init__(self, impurity: str = "N2", target_DOD: float = 3.0, target_T_t_eV: float = 3.0):
        self.impurity = impurity
        self.target_DOD = target_DOD
        self.target_T_t = target_T_t_eV

        self.Kp = 50.0  # Pa m^3/s per eV error
        self.Ki = 10.0  # Pa m^3/s per eV*s

        self.integral_e = 0.0
        self.last_cmd = 0.0
        self.state = DetachmentState.ATTACHED

    def _determine_state(self, T_t: float, rho_front: float) -> DetachmentState:
        if rho_front > 0.8:
            return DetachmentState.XPOINT_MARFE
        if T_t > 30.0:
            return DetachmentState.ATTACHED
        if T_t > 5.0:
            return DetachmentState.PARTIALLY_DETACHED
        return DetachmentState.FULLY_DETACHED

    def step(
        self,
        T_t_measured: float,
        n_t_measured: float,
        P_rad_measured: float,
        rho_front: float,
        dt: float,
    ) -> float:
        self.state = self._determine_state(T_t_measured, rho_front)

        # If X-point MARFE imminent, hard drop seeding
        if self.state == DetachmentState.XPOINT_MARFE:
            self.last_cmd *= 0.5
            self.integral_e *= 0.5
            return self.last_cmd

        error = T_t_measured - self.target_T_t
        self.integral_e += error * dt

        cmd = self.Kp * error + self.Ki * self.integral_e

        cmd = max(0.0, float(cmd))
        self.last_cmd = cmd
        return cmd


@dataclass
class DetachmentPoint:
    seeding_rate: float
    T_target: float
    n_target: float
    DOD: float
    P_rad_frac: float
    state: DetachmentState


class DetachmentBifurcation:
    def __init__(self, sol: TwoPointSOL, impurity: str):
        self.sol = sol
        self.impurity = impurity
        self.front_model = RadiationFrontModel(impurity, sol.R0, sol.a, sol.q95)

    def _steady_state_target(
        self, seeding_rate: float, P_SOL_MW: float, n_u_19: float
    ) -> DetachmentPoint:
        # Seeding rate determines f_rad
        f_rad = min(0.95, seeding_rate * 0.1)  # Mock scaling

        res = self.sol.solve(P_SOL_MW, n_u_19, f_rad=f_rad)

        rho_front = self.front_model.front_position(P_SOL_MW, n_u_19, seeding_rate)

        # If deeply detached, T_t drops rapidly below the conduction limit solver
        T_t = res.T_target_eV
        if f_rad > 0.8:
            T_t = max(0.5, T_t * math.exp(-(f_rad - 0.8) * 20.0))

        dod = self.front_model.degree_of_detachment(T_t, res.n_target_19, n_u_19)

        # State
        if rho_front > 0.8:
            state = DetachmentState.XPOINT_MARFE
        elif T_t > 30.0:
            state = DetachmentState.ATTACHED
        elif T_t > 5.0:
            state = DetachmentState.PARTIALLY_DETACHED
        else:
            state = DetachmentState.FULLY_DETACHED

        return DetachmentPoint(seeding_rate, T_t, res.n_target_19, dod, f_rad, state)

    def scan_seeding(
        self, seeding_range: np.ndarray, P_SOL_MW: float, n_u_19: float
    ) -> list[DetachmentPoint]:
        return [self._steady_state_target(sr, P_SOL_MW, n_u_19) for sr in seeding_range]

    def find_rollover_point(self, P_SOL_MW: float, n_u_19: float) -> float:
        """Find seeding rate where ion flux (n_t * sqrt(T_t)) peaks."""
        sr_scan = np.linspace(0.0, 10.0, 100)
        fluxes = []
        for sr in sr_scan:
            pt = self._steady_state_target(sr, P_SOL_MW, n_u_19)
            # Gamma ~ n_t * c_s ~ n_t * sqrt(T_t)
            flux = pt.n_target * math.sqrt(pt.T_target)
            fluxes.append(flux)

        max_idx = np.argmax(fluxes)
        return float(sr_scan[max_idx])


class MultiImpuritySeeding:
    def __init__(self, impurities: list[str], controllers: dict[str, DetachmentController]):
        self.impurities = impurities
        self.controllers = controllers

    def step(self, diagnostics: dict[str, float], dt: float) -> dict[str, float]:
        T_t = diagnostics.get("T_target_eV", 20.0)
        n_t = diagnostics.get("n_target_19", 10.0)
        P_rad = diagnostics.get("P_rad_MW", 10.0)
        rho_front = diagnostics.get("rho_front", 0.1)

        rates = {}
        for imp in self.impurities:
            if imp in self.controllers:
                rates[imp] = self.controllers[imp].step(T_t, n_t, P_rad, rho_front, dt)
            else:
                rates[imp] = 0.0

        return rates
