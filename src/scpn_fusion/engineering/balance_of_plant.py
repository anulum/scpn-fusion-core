# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Balance Of Plant
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt

class PowerPlantModel:
    """
    SCPN Balance of Plant (BOP) Simulator.
    Calculates the conversion of Fusion Energy to Grid Electricity.
    Includes parasitic loads (Magnets, Heating, Pumping).
    """
    def __init__(self):
        # Efficiency Parameters
        self.eta_thermal = 0.35 # Rankine Cycle efficiency (Steam turbines)
        self.eta_direct = 0.60  # Direct Conversion (if advanced fuels) - unlikely for DT
        self.eta_heating = 0.40 # Wall-plug efficiency of NBI/ECRH systems
        
        # Parasitic Loads (Base load in MW)
        self.P_cryo = 30.0      # Cryogenics for Superconductors
        self.P_pump = 10.0      # Vacuum & Fuel pumps
        self.P_bop_misc = 15.0  # Control rooms, lights, HVAC
        
    def calculate_plant_performance(self, P_fusion_MW, P_aux_absorbed_MW):
        """
        Takes Plasma Physics output and computes Electrical Output.
        P_fusion: Total fusion power (Alpha + Neutrons).
        P_aux_absorbed: Power absorbed by plasma (heating).
        """
        # 1. Thermal Power Generation
        # Neutron Power (80% of Fusion) -> Heat in Blanket
        P_neutron = 0.8 * P_fusion_MW
        
        # Alpha Power (20% of Fusion) -> Heat in Divertor
        P_alpha = 0.2 * P_fusion_MW
        
        # Aux Power -> Heat in Plasma -> Heat in Divertor
        P_aux_thermal = P_aux_absorbed_MW
        
        # Exothermic Lithium Reactions in Blanket (Energy Multiplication)
        # M ~ 1.1 to 1.2
        M_blanket = 1.15 
        P_thermal_blanket = P_neutron * M_blanket
        
        # Total Thermal Power (Heat available for steam)
        P_thermal_total = P_thermal_blanket + P_alpha + P_aux_thermal
        
        # 2. Gross Electrical Generation
        P_gross_electric = P_thermal_total * self.eta_thermal
        
        # 3. Parasitic Consumption (Recirculating Power)
        # Power needed to drive the Aux Heating systems (Efficiency loss)
        P_aux_wall_plug = P_aux_absorbed_MW / self.eta_heating
        
        # Total House Load
        P_recirculating = self.P_cryo + self.P_pump + self.P_bop_misc + P_aux_wall_plug
        
        # 4. Net Electricity
        P_net_electric = P_gross_electric - P_recirculating
        
        # 5. Metrics
        Q_plasma = P_fusion_MW / P_aux_absorbed_MW if P_aux_absorbed_MW > 0 else 0
        Q_engineering = P_gross_electric / P_recirculating if P_recirculating > 0 else 0
        
        return {
            'P_fusion': P_fusion_MW,
            'P_thermal': P_thermal_total,
            'P_gross': P_gross_electric,
            'P_recirc': P_recirculating,
            'P_net': P_net_electric,
            'Q_plasma': Q_plasma,
            'Q_eng': Q_engineering,
            'breakdown': {
                'Cryo': self.P_cryo,
                'Pumps': self.P_pump,
                'Heating_Plug': P_aux_wall_plug,
                'Misc': self.P_bop_misc
            }
        }

    def plot_sankey_diagram(self, metrics):
        """
        Simple visualization of power flow (Text-based or simple bar for now).
        """
        labels = ['Fusion Power', 'Thermal Total', 'Gross Elec', 'Recirculating', 'NET TO GRID']
        values = [metrics['P_fusion'], metrics['P_thermal'], metrics['P_gross'], 
                  metrics['P_recirc'], metrics['P_net']]
        colors = ['red', 'orange', 'blue', 'gray', 'green']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(labels, values, color=colors)
        ax.set_ylabel("Power (MW)")
        ax.set_title(f"Plant Power Balance (Net: {metrics['P_net']:.1f} MWe)")
        
        # Threshold line
        ax.axhline(0, color='black', linewidth=1)
        
        return fig

if __name__ == "__main__":
    # Test Run
    plant = PowerPlantModel()
    # Scenario: 500MW Fusion, 50MW Heating
    res = plant.calculate_plant_performance(500.0, 50.0)
    print(f"--- FUSION PLANT STATUS ---")
    print(f"Fusion Power: {res['P_fusion']:.1f} MW")
    print(f"Gross Electric: {res['P_gross']:.1f} MW")
    print(f"House Load: {res['P_recirc']:.1f} MW")
    print(f"NET TO GRID: {res['P_net']:.1f} MW")
    print(f"Q_eng: {res['Q_eng']:.2f}")
    
    if res['P_net'] > 0:
        print("SUCCESS: Reactor is commercially viable.")
    else:
        print("FAIL: Reactor consumes more than it produces.")
