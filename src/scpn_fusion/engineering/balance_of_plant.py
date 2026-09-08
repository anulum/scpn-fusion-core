# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Balance Of Plant
"""Balance-of-plant performance model for conversion efficiency and parasitic loads."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import matplotlib.pyplot as plt

from scpn_fusion.engineering.thermal_hydraulics import CoolantLoop, PumpingPowerResult

logger = logging.getLogger(__name__)


class PlantPowerBreakdown(TypedDict):
    """Parasitic-load breakdown for balance-of-plant accounting."""

    Cryo: float
    Pumps: float
    Heating_Plug: float
    Misc: float


class PlantPerformance(TypedDict):
    """Balance-of-plant scalar performance metrics."""

    P_fusion: float
    P_thermal: float
    P_gross: float
    P_recirc: float
    P_net: float
    Q_plasma: float
    Q_eng: float
    breakdown: PlantPowerBreakdown
    hydraulics: PumpingPowerResult


class PowerPlantModel:
    """SCPN Balance of Plant (BOP) Simulator.

    Calculates the conversion of Fusion Energy to Grid Electricity.
    Includes parasitic loads (Magnets, Heating, Pumping).
    """

    def __init__(self, coolant_type: str = "water") -> None:
        # Efficiency Parameters
        self.eta_thermal = 0.35  # Rankine Cycle efficiency (Steam turbines)
        self.eta_direct = 0.60  # Direct Conversion (if advanced fuels) - unlikely for DT
        self.eta_heating = 0.40  # Wall-plug efficiency of NBI/ECRH systems

        # Parasitic Loads (Base load in MW)
        self.P_cryo = 30.0  # Cryogenics for Superconductors
        self.P_bop_misc = 15.0  # Control rooms, lights, HVAC

        # Thermal Hydraulics
        self.coolant = CoolantLoop(coolant_type)

    def calculate_plant_performance(
        self,
        P_fusion_MW: float,
        P_aux_absorbed_MW: float,
        *,
        coolant_parallel_channels: int = 1,
        coolant_length_m: float = 100.0,
        coolant_diameter_m: float = 0.05,
        coolant_temperature_rise_k: float = 50.0,
    ) -> PlantPerformance:
        """Calculate instantaneous plant power with explicitly configured cooling.

        Parameters
        ----------
        P_fusion_MW : float
            Nonnegative finite total alpha plus neutron fusion power in MW.
        P_aux_absorbed_MW : float
            Nonnegative finite heating power absorbed by the plasma in MW.
        coolant_parallel_channels : int
            Number of identical equal-flow cooling paths, default one.
        coolant_length_m : float
            Length of each cooling path in metres.
        coolant_diameter_m : float
            Internal diameter of each cooling path in metres.
        coolant_temperature_rise_k : float
            Coolant temperature rise per path in kelvin.

        Returns
        -------
        PlantPerformance
            Thermal, gross, recirculating and net powers in MW, gain ratios,
            component electrical loads and explicit hydraulic diagnostics.
            Negative net power is preserved. Historical zero-denominator gain
            convention returns zero; it does not establish a finite physical gain.

        Raises
        ------
        ValueError
            For invalid input powers or invalid cooling configuration.

        Notes
        -----
        Default cooling is one 5 cm pipe, not a plant-scale layout. Supply actual
        geometry and channel count; do not calibrate them solely to target net
        electricity. CoolantLoop's constant-property limitations apply.
        """
        for name, value in (("P_fusion_MW", P_fusion_MW), ("P_aux_absorbed_MW", P_aux_absorbed_MW)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        # 1. Thermal Power Generation
        P_neutron = 0.8 * P_fusion_MW
        P_alpha = 0.2 * P_fusion_MW
        P_aux_thermal = P_aux_absorbed_MW

        # Exothermic Lithium Reactions in Blanket (Energy Multiplication)
        M_blanket = 1.15
        P_thermal_blanket = P_neutron * M_blanket
        P_thermal_total = P_thermal_blanket + P_alpha + P_aux_thermal

        # 2. Gross Electrical Generation
        P_gross_electric = P_thermal_total * self.eta_thermal

        # 3. Parasitic Consumption (Recirculating Power)
        # Physics-based pumping power
        pump_res = self.coolant.calculate_pumping_power(
            P_thermal_total,
            delta_T=coolant_temperature_rise_k,
            L=coolant_length_m,
            D=coolant_diameter_m,
            parallel_channels=coolant_parallel_channels,
        )
        P_pump = float(pump_res["P_pump_MW"])

        # Power needed to drive the Aux Heating systems (Efficiency loss)
        P_aux_wall_plug = P_aux_absorbed_MW / self.eta_heating

        # Total House Load
        P_recirculating = self.P_cryo + P_pump + self.P_bop_misc + P_aux_wall_plug

        # 4. Net Electricity
        P_net_electric = P_gross_electric - P_recirculating

        # 5. Metrics
        Q_plasma = P_fusion_MW / P_aux_absorbed_MW if P_aux_absorbed_MW > 0 else 0
        Q_engineering = P_gross_electric / P_recirculating if P_recirculating > 0 else 0

        return {
            "hydraulics": pump_res,
            "P_fusion": P_fusion_MW,
            "P_thermal": P_thermal_total,
            "P_gross": P_gross_electric,
            "P_recirc": P_recirculating,
            "P_net": P_net_electric,
            "Q_plasma": Q_plasma,
            "Q_eng": Q_engineering,
            "breakdown": {
                "Cryo": self.P_cryo,
                "Pumps": P_pump,
                "Heating_Plug": P_aux_wall_plug,
                "Misc": self.P_bop_misc,
            },
        }

    def plot_sankey_diagram(self, metrics: PlantPerformance) -> Any:
        """Visualise power flow as a text-based or simple bar summary."""
        labels = ["Fusion Power", "Thermal Total", "Gross Elec", "Recirculating", "NET TO GRID"]
        values = [
            metrics["P_fusion"],
            metrics["P_thermal"],
            metrics["P_gross"],
            metrics["P_recirc"],
            metrics["P_net"],
        ]
        colors = ["red", "orange", "blue", "gray", "green"]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(labels, values, color=colors)
        ax.set_ylabel("Power (MW)")
        ax.set_title(f"Plant Power Balance (Net: {metrics['P_net']:.1f} MWe)")

        # Threshold line
        ax.axhline(0, color="black", linewidth=1)

        return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test Run
    plant = PowerPlantModel()
    # Scenario: 500MW Fusion, 50MW Heating
    res = plant.calculate_plant_performance(500.0, 50.0)
    logger.info("--- FUSION PLANT STATUS ---")
    logger.info("Fusion Power: %.1f MW", res["P_fusion"])
    logger.info("Gross Electric: %.1f MW", res["P_gross"])
    logger.info("House Load: %.1f MW", res["P_recirc"])
    logger.info("NET TO GRID: %.1f MW", res["P_net"])
    logger.info("Q_eng: %.2f", res["Q_eng"])

    logger.info("Screening calculation only; net power does not establish commercial viability.")
