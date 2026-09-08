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
import math
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
    channel_mass_flow_kg_s: float
    parallel_channels: int
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
        """Select fixed approximate coolant properties near 300 degrees Celsius.

        Parameters
        ----------
        coolant_type : str
            One of ``water``, ``helium`` or ``lipb``. Properties do not vary with
            pressure, temperature or phase during this lumped calculation.

        Raises
        ------
        ValueError
            If the coolant identifier is unknown; no substitute is selected.
        """
        # Properties at 300C (Approx)
        props: dict[str, CoolantProperties] = {
            "water": {"rho": 700.0, "mu": 1e-4, "cp": 5000.0},
            "helium": {"rho": 5.0, "mu": 3e-5, "cp": 5190.0},
            "lipb": {"rho": 9000.0, "mu": 1e-3, "cp": 190.0},
        }
        if coolant_type not in props:
            raise ValueError(f"Unknown coolant type: {coolant_type}")
        self.p = props[coolant_type]

    def calculate_pumping_power(
        self,
        Q_thermal_MW: float,
        delta_T: float = 50.0,
        L: float = 100.0,
        D: float = 0.05,
        *,
        parallel_channels: int = 1,
    ) -> PumpingPowerResult:
        """Calculate total pumping power for identical parallel coolant channels.

        Parameters
        ----------
        Q_thermal_MW : float
            Total thermal load across all channels, in MW; zero is permitted.
        delta_T : float
            Coolant temperature rise in each channel, in kelvin.
        L : float
            Length of each hydraulic path in metres, not summed channel length.
        D : float
            Internal diameter of each circular channel in metres.
        parallel_channels : int
            Positive number of equal-flow parallel channels. Default one retains
            the historical single-pipe calculation; it is not a reactor layout.

        Returns
        -------
        PumpingPowerResult
            Total and per-channel mass flows, per-channel velocity, Reynolds
            number and pressure drop, and total electrical pump power in MW.
            Pump efficiency is fixed at 0.8. Zero load yields zero flow and power.

        Raises
        ------
        ValueError
            For nonfinite loads/geometry, negative load or nonpositive geometry
            and channel count, or unrepresentable intermediate/output values.
            Channels must be an integer, not a boolean.

        Notes
        -----
        This constant-property incompressible model omits headers, minor losses,
        boiling, compressibility and flow maldistribution. Selecting channel
        geometry does not certify the model's physical applicability.
        """
        for name, value in (
            ("Q_thermal_MW", Q_thermal_MW),
            ("delta_T", delta_T),
            ("L", L),
            ("D", D),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            if value < 0 or (value == 0 and name != "Q_thermal_MW"):
                raise ValueError(f"{name} is outside its allowed domain.")
        if type(parallel_channels) is not int or parallel_channels <= 0:
            raise ValueError("parallel_channels must be a positive integer.")
        if Q_thermal_MW == 0:
            return {
                "mdot_kg_s": 0.0,
                "channel_mass_flow_kg_s": 0.0,
                "parallel_channels": parallel_channels,
                "velocity_m_s": 0.0,
                "Re": 0.0,
                "dP_Pa": 0.0,
                "P_pump_MW": 0.0,
            }

        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                mdot = float((Q_thermal_MW * 1e6) / (self.p["cp"] * delta_T))
                area = float(np.pi * (D / 2) ** 2)
                channel_mdot = mdot / parallel_channels
                v = float(channel_mdot / (self.p["rho"] * area))
                Re = float((self.p["rho"] * v * D) / self.p["mu"])
                f = churchill_friction_factor(Re)
                dP = float(f * (L / D) * (self.p["rho"] * v**2 / 2.0))
                vol_flow = float(mdot / self.p["rho"])
                P_pump_MW = float((dP * vol_flow) / 0.8 / 1e6)
        except ArithmeticError as error:
            raise ValueError("Cooling calculation is outside the representable domain.") from error
        diagnostics = (mdot, area, channel_mdot, v, Re, f, dP, vol_flow, P_pump_MW)
        if any(not math.isfinite(value) or value <= 0 for value in diagnostics):
            raise ValueError("Cooling calculation is outside the representable domain.")
        return {
            "mdot_kg_s": mdot,
            "channel_mass_flow_kg_s": channel_mdot,
            "parallel_channels": parallel_channels,
            "velocity_m_s": v,
            "Re": Re,
            "dP_Pa": dP,
            "P_pump_MW": P_pump_MW,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loop = CoolantLoop("water")
    res = loop.calculate_pumping_power(Q_thermal_MW=500.0)
    logger.info("--- Thermal Hydraulics (Water) ---")
    logger.info("Mass Flow: %.1f kg/s", res["mdot_kg_s"])
    logger.info("Pressure Drop: %.2f bar", res["dP_Pa"] / 1e5)
    logger.info("Pumping Power: %.2f MW", res["P_pump_MW"])
