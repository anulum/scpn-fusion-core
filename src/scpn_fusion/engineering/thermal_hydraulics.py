# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Thermal Hydraulics
"""Thermal-hydraulic helper utilities for flow resistance and pump power."""

from __future__ import annotations

import logging
from typing import TypedDict

import numpy as np

logger = logging.getLogger(__name__)


class CoolantProperties(TypedDict):
    """Thermophysical coolant properties used by the lumped loop model."""

    rho: float
    mu: float
    cp: float


class PumpingPowerResult(TypedDict):
    """Computed coolant-loop pumping-power diagnostics."""

    mdot_kg_s: float
    velocity_m_s: float
    Re: float
    dP_Pa: float
    P_pump_MW: float


def churchill_friction_factor(Re: float, epsilon_d: float = 1e-4) -> float:
    """Churchill Correlation for Darcy Friction Factor (f).

    Valid for all flow regimes (laminar, transition, turbulent).
    """
    if Re <= 0.0:
        raise ValueError("Reynolds number must be positive.")
    if Re < 1e-3:
        return float(64.0 / 1e-3)  # Limit

    A = (2.457 * np.log(1.0 / ((7.0 / Re) ** 0.9 + 0.27 * epsilon_d))) ** 16
    B = (37530.0 / Re) ** 16

    f = 8.0 * ((8.0 / Re) ** 12 + 1.0 / (A + B) ** 1.5) ** (1.0 / 12.0)
    return float(f)


class CoolantLoop:
    """Calculate pressure drop and pumping power for reactor cooling.

    Supports Water, Helium, and Liquid Metal (LiPb).
    """

    def __init__(self, coolant_type: str = "water") -> None:
        # Properties at 300C (Approx)
        props: dict[str, CoolantProperties] = {
            "water": {"rho": 700.0, "mu": 1e-4, "cp": 5000.0},
            "helium": {"rho": 5.0, "mu": 3e-5, "cp": 5190.0},
            "lipb": {"rho": 9000.0, "mu": 1e-3, "cp": 190.0},
        }
        self.p = props.get(coolant_type, props["water"])

    def calculate_pumping_power(
        self,
        Q_thermal_MW: float,
        delta_T: float = 50.0,
        L: float = 100.0,
        D: float = 0.05,
    ) -> PumpingPowerResult:
        """Estimate pumping power needed to exhaust Q_thermal.

        Q_thermal: MW
        delta_T: Temperature rise (K)
        L: Total pipe length (m)
        D: Pipe diameter (m).
        """
        if Q_thermal_MW < 0.0:
            raise ValueError("Q_thermal_MW must be non-negative.")
        if delta_T <= 0.0:
            raise ValueError("delta_T must be > 0.")
        if L <= 0.0:
            raise ValueError("L must be > 0.")
        if D <= 0.0:
            raise ValueError("D must be > 0.")

        # 1. Mass flow rate (mdot = Q / (cp * dT))
        mdot = float((Q_thermal_MW * 1e6) / (self.p["cp"] * delta_T))

        # 2. Velocity (v = mdot / (rho * Area))
        area = float(np.pi * (D / 2) ** 2)
        v = float(mdot / (self.p["rho"] * area))

        # 3. Reynolds Number
        Re = float((self.p["rho"] * v * D) / self.p["mu"])

        # 4. Friction Factor (Churchill)
        f = churchill_friction_factor(Re)

        # 5. Pressure Drop (Darcy-Weisbach)
        dP = float(f * (L / D) * (self.p["rho"] * v**2 / 2.0))

        # 6. Pumping Power (W)
        eta_pump = 0.8
        vol_flow = float(mdot / self.p["rho"])
        P_pump_W = float((dP * vol_flow) / eta_pump)

        return {
            "mdot_kg_s": mdot,
            "velocity_m_s": v,
            "Re": Re,
            "dP_Pa": dP,
            "P_pump_MW": P_pump_W / 1e6,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loop = CoolantLoop("water")
    res = loop.calculate_pumping_power(Q_thermal_MW=500.0)
    logger.info("--- Thermal Hydraulics (Water) ---")
    logger.info("Mass Flow: %.1f kg/s", res["mdot_kg_s"])
    logger.info("Pressure Drop: %.2f bar", res["dP_Pa"] / 1e5)
    logger.info("Pumping Power: %.2f MW", res["P_pump_MW"])
