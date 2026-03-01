# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Thermal Hydraulics
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ──────────────────────────────────────────────────────────────────────
import numpy as np

def churchill_friction_factor(Re, epsilon_d=1e-4):
    """
    Churchill Correlation for Darcy Friction Factor (f).
    Valid for all flow regimes (laminar, transition, turbulent).
    """
    if Re <= 0.0:
        raise ValueError("Reynolds number must be positive.")
    if Re < 1e-3:
        return 64.0 / 1e-3  # Limit
    
    A = (2.457 * np.log(1.0 / ((7.0 / Re)**0.9 + 0.27 * epsilon_d)))**16
    B = (37530.0 / Re)**16
    
    f = 8.0 * ((8.0 / Re)**12 + 1.0 / (A + B)**1.5)**(1.0/12.0)
    return f

class CoolantLoop:
    """
    Calculates pressure drop and pumping power for reactor cooling.
    Supports Water, Helium, and Liquid Metal (LiPb).
    """
    def __init__(self, coolant_type='water'):
        # Properties at 300C (Approx)
        props = {
            'water':  {'rho': 700.0, 'mu': 1e-4, 'cp': 5000.0},
            'helium': {'rho': 5.0,   'mu': 3e-5, 'cp': 5190.0},
            'lipb':   {'rho': 9000.0, 'mu': 1e-3, 'cp': 190.0}
        }
        self.p = props.get(coolant_type, props['water'])
        
    def calculate_pumping_power(self, Q_thermal_MW, delta_T=50.0, L=100.0, D=0.05):
        """
        Estimates pumping power needed to exhaust Q_thermal.
        Q_thermal: MW
        delta_T: Temperature rise (K)
        L: Total pipe length (m)
        D: Pipe diameter (m)
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
        mdot = (Q_thermal_MW * 1e6) / (self.p['cp'] * delta_T)
        
        # 2. Velocity (v = mdot / (rho * Area))
        area = np.pi * (D/2)**2
        v = mdot / (self.p['rho'] * area)
        
        # 3. Reynolds Number
        Re = (self.p['rho'] * v * D) / self.p['mu']
        
        # 4. Friction Factor (Churchill)
        f = churchill_friction_factor(Re)
        
        # 5. Pressure Drop (Darcy-Weisbach)
        dP = f * (L/D) * (self.p['rho'] * v**2 / 2.0)
        
        # 6. Pumping Power (W)
        eta_pump = 0.8
        vol_flow = mdot / self.p['rho']
        P_pump_W = (dP * vol_flow) / eta_pump
        
        return {
            'mdot_kg_s': mdot,
            'velocity_m_s': v,
            'Re': Re,
            'dP_Pa': dP,
            'P_pump_MW': P_pump_W / 1e6
        }

if __name__ == "__main__":
    loop = CoolantLoop('water')
    res = loop.calculate_pumping_power(Q_thermal_MW=500.0)
    print(f"--- Thermal Hydraulics (Water) ---")
    print(f"Mass Flow: {res['mdot_kg_s']:.1f} kg/s")
    print(f"Pressure Drop: {res['dP_Pa']/1e5:.2f} bar")
    print(f"Pumping Power: {res['P_pump_MW']:.2f} MW")
