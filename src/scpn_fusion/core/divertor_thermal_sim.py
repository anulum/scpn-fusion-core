import numpy as np
import matplotlib.pyplot as plt
import sys

class DivertorLab:
    """
    Simulates Heat Exhaust in a Compact Fusion Reactor.
    Compares Solid Tungsten vs. Liquid Lithium Vapor Shielding.
    """
    def __init__(self, P_sol_MW=50.0, R_major=2.1, B_pol=2.0):
        self.P_sol = P_sol_MW # Power entering Scrape-Off Layer
        self.R = R_major
        self.B_pol = B_pol
        
        # Eich Scaling for Heat Flux Width (lambda_q)
        # lambda_q (mm) = 0.63 * B_pol^(-1.19)
        self.lambda_q_mm = 0.63 * (B_pol**(-1.19))
        self.lambda_q = self.lambda_q_mm / 1000.0 # meters
        
        print(f"--- DIVERTOR PHYSICS ---")
        print(f"Power to Divertor: {self.P_sol} MW")
        print(f"Eich Width (lambda_q): {self.lambda_q_mm:.3f} mm")
        
    def calculate_heat_load(self, expansion_factor=10.0):
        """
        Calculates Peak Heat Flux on the target plate.
        expansion_factor: Magnetic flux expansion (fx) * Target tilt (sin theta).
        Typically 10-20 for advanced divertors (Super-X).
        """
        # Parallel Heat Flux (q_par)
        # P_sol = 2 * pi * R * lambda_q * q_par (approx for single null)
        # q_par = P_sol / (2 * pi * R * lambda_q)
        
        # Note: 4*pi*R for Double Null. Assuming Single Null (DN is better but harder).
        self.q_parallel = (self.P_sol * 1e6) / (2 * np.pi * self.R * self.lambda_q)
        
        # Target Heat Flux (q_target) = q_par / expansion_factor
        self.q_target_solid = self.q_parallel / expansion_factor
        
        print(f"Parallel Heat Flux: {self.q_parallel/1e9:.1f} GW/m2")
        print(f"Unmitigated Target Flux: {self.q_target_solid/1e6:.1f} MW/m2")
        
        return self.q_target_solid

    def simulate_tungsten(self):
        """
        1D Thermal limit of Tungsten Monoblock.
        Simple conduction model: T_surf = q * d / k + T_coolant
        """
        k_W = 100.0 # W/mK (Thermal conductivity)
        d_block = 0.01 # 1 cm to cooling channel
        T_coolant = 100.0 # C (Water)
        
        q = self.q_target_solid
        
        delta_T = (q * d_block) / k_W
        T_surf = T_coolant + delta_T
        
        status = "MELTED" if T_surf > 3422 else "OK"
        return T_surf, status

    def simulate_lithium_vapor(self):
        """
        Vapor Shielding Physics (Simplified).
        Lithium evaporates -> Cloud radiates energy -> q_target drops.
        
        Self-Regulating Model:
        q_target = q_incident * exp(-f_rad * T_surf) 
        (Heuristic for non-coronal radiation cooling)
        """
        # Iterative solution for self-consistent State
        T_surf = 500.0 # Initial guess (C)
        T_boil = 1342.0 # C
        
        # Lithium properties
        L_vap = 147.0 # kJ/mol (Heat of vaporization) - not used directly in simple model
        
        history_q = []
        history_T = []
        
        for i in range(50):
            # Evaporation Rate (Hertz-Knudsen) ~ P_sat(T)
            # Simplified: Radiative fraction increases with T (more vapor)
            # f_rad = 1 - (q_surface / q_incident)
            
            # Radiated Power Fraction (f_rad)
            # T < 500C: 0% radiation
            # T > 1000C: 99% radiation (strong shielding)
            if T_surf < 400:
                f_rad = 0.0
            else:
                # Sigmoid function for shielding onset
                f_rad = 0.95 / (1 + np.exp(-(T_surf - 700)/100))
            
            # New q at surface
            q_surf = self.q_target_solid * (1.0 - f_rad)
            
            # Surface Temp (Liquid metal layer + substrate)
            # Liquid Li is thin, assume conduction to substrate dominates or Li convection
            # Effective k_Li_eff (Convection) is high
            k_eff = 200.0 
            d = 0.005 # 5mm layer
            T_new = 300.0 + (q_surf * d) / k_eff
            
            # Relaxation
            T_surf = 0.5*T_surf + 0.5*T_new
            
            history_q.append(q_surf)
            history_T.append(T_surf)
            
        return T_surf, q_surf, f_rad

def run_divertor_sim():
    print("\n--- SCPN HEAT EXHAUST: The Lithium Solution ---")
    
    lab = DivertorLab(P_sol_MW=80.0, R_major=2.1, B_pol=2.5) # Compact Pilot parameters
    
    # 1. Unmitigated Load
    q_solid = lab.calculate_heat_load(expansion_factor=15.0) # Advanced geometry
    
    # 2. Tungsten Test
    Tw, status_w = lab.simulate_tungsten()
    print(f"Tungsten Divertor: {Tw:.0f} degC -> {status_w}!")
    
    # 3. Lithium Test
    Tli, q_li, shielding = lab.simulate_lithium_vapor()
    print(f"Liquid Li Divertor: {Tli:.0f} degC (Shielding: {shielding*100:.1f}%)")
    
    # --- VISUALIZATION ---
    fig, ax = plt.subplots(figsize=(8, 6))
    materials = ['Tungsten (Solid)', 'Lithium (Vapor Shield)']
    temps = [Tw, Tli]
    limits = [3422, 1342] # Melting / Boiling
    colors = ['gray', 'purple']
    
    bars = ax.bar(materials, temps, color=colors)
    ax.axhline(3422, color='red', linestyle='--', label='W Melting Point')
    ax.set_ylabel("Surface Temperature (°C)")
    ax.set_title("Divertor Material Performance")
    
    # Add Heat Flux annotations
    ax.text(0, Tw/2, f"Flux: {q_solid/1e6:.0f} MW/m2", ha='center', color='white', fontweight='bold')
    ax.text(1, Tli/2, f"Flux: {q_li/1e6:.1f} MW/m2", ha='center', color='white', fontweight='bold')
    
    ax.legend()
    plt.savefig("Divertor_Solution.png")
    print("Saved: Divertor_Solution.png")

if __name__ == "__main__":
    run_divertor_sim()
