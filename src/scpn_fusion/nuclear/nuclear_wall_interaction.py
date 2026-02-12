import numpy as np
import matplotlib.pyplot as plt
from scpn_fusion.core.fusion_ignition_sim import FusionBurnPhysics
import sys

# --- MATERIALS DATABASE ---
# Thresholds for neutron damage before replacement is needed
MATERIALS = {
    'Tungsten (W)': {'dpa_limit': 50.0, 'sigma_dpa': 1000}, # Divertor armor
    'Eurofer (Steel)': {'dpa_limit': 150.0, 'sigma_dpa': 500}, # Structural blanket
    'Beryllium (Be)': {'dpa_limit': 10.0, 'sigma_dpa': 200}, # First wall (old design)
}

class NuclearEngineeringLab(FusionBurnPhysics):
    """
    Simulates the nuclear interaction between Plasma and the Reactor Vessel.
    1. Helium Ash accumulation.
    2. Neutron Flux distribution on the First Wall.
    3. Material Damage (DPA).
    """
    def __init__(self, config_path):
        super().__init__(config_path)
        
    def generate_first_wall(self):
        """
        Defines the geometry of the reactor wall (Vacuum Vessel).
        Approximated as a D-shaped contour surrounding the plasma.
        """
        theta = np.linspace(0, 2*np.pi, 200)
        # Wall Parameters
        R0, a, kappa, delta = 5.0, 3.0, 1.9, 0.4 
        
        # Parametric Wall
        R_wall = R0 + a * np.cos(theta + np.arcsin(delta)*np.sin(theta))
        Z_wall = kappa * a * np.sin(theta)
        
        return R_wall, Z_wall

    def simulate_ash_poisoning(self, burn_time_sec=1000, tau_He_ratio=5.0):
        """
        Simulates the drop in fusion power due to Helium buildup.
        tau_He_ratio: Ratio of Helium particle confinement to Energy confinement (tau_He / tau_E).
        If ratio > 10, the reactor chokes.
        """
        print(f"[Nuclear] Simulating Ash Poisoning (tau_He*/tau_E = {tau_He_ratio})...")
        
        # 1. Get Base Plasma State
        self.solve_equilibrium()
        
        # Initial Conditions
        n_e_target = 1.0e20 # Electron density (Greenwald Limit constant)
        f_He = 0.0 # Helium fraction
        dt = 1.0 # Second
        
        history = {'time': [], 'P_fus': [], 'f_He': [], 'Q': []}
        
        # Volume (Approximation)
        Vol = 800.0 # m^3
        
        for t in range(int(burn_time_sec)):
            # A. Composition (Quasi-neutrality constraint)
            # n_e = n_D + n_T + 2*n_He + Z_imp*n_imp
            # Assume n_D = n_T
            # n_fuel = n_e - 2*n_He
            
            n_He = f_He * n_e_target
            n_fuel = n_e_target - 2*n_He
            n_D = 0.5 * n_fuel
            n_T = 0.5 * n_fuel
            
            if n_fuel < 0:
                print("  -> Plasma Quenched (Dilution Limit)")
                break
                
            # B. Reaction Rate
            T_keV = 20.0 # Keep temp constant for this isolation study
            sigmav = self.bosch_hale_dt(T_keV)
            
            # Reaction rate per volume
            R_fus = n_D * n_T * sigmav
            
            # C. Ash Dynamics (0D equation)
            # dn_He/dt = Source(Fusion) - Sink(Transport/Pump)
            tau_E = 3.0
            tau_He = tau_He_ratio * tau_E
            
            dn_He = R_fus - (n_He / tau_He)
            
            # Update State
            n_He += dn_He * dt
            f_He = n_He / n_e_target
            
            # D. Power Output
            E_fus = 17.6 * 1.602e-13
            P_fus_MW = (R_fus * E_fus * Vol) / 1e6
            
            history['time'].append(t)
            history['P_fus'].append(P_fus_MW)
            history['f_He'].append(f_He)
            
        return history

    def calculate_neutron_wall_loading(self):
        """
        Ray-Tracing calculation of 14.1 MeV neutrons hitting the wall.
        """
        print("[Nuclear] Calculating Neutron Wall Loading (NWL)...")
        
        # 1. Source: Plasma Grid (Fusion Power Density)
        # We reuse the thermodynamics calculation to get local emissivity
        # Recalculate power density profile
        idx_max = np.argmax(self.Psi)
        iz_ax, ir_ax = np.unravel_index(idx_max, self.Psi.shape)
        Psi_axis = self.Psi[iz_ax, ir_ax]
        xp, psi_x = self.find_x_point(self.Psi)
        Psi_edge = psi_x
        if abs(Psi_edge - Psi_axis) < 1.0: Psi_edge = np.min(self.Psi)
        
        Psi_norm = (self.Psi - Psi_axis) / (Psi_edge - Psi_axis)
        Psi_norm = np.clip(Psi_norm, 0, 1)
        mask = (Psi_norm < 1.0)
        
        # Emissivity Profile (Neutrons/s/m^3)
        # S_n ~ n^2 * sigmav
        # We simplify: Profile follows Pressure^2 ~ (1-psi)^2
        S_peak = 1e18 # Neutrons/m^3/s (Approx for 500MW)
        Source_Map = np.zeros_like(self.Psi)
        Source_Map[mask] = S_peak * (1.0 - Psi_norm[mask])**2
        
        # 2. Target: First Wall Segments
        Rw, Zw = self.generate_first_wall()
        wall_flux = np.zeros(len(Rw))
        
        # 3. Ray Tracing Integration (Line-of-Sight)
        # For every point on wall, sum contrib from every point in plasma
        # Flux = Sum( Source_i * dV_i / (4*pi*r^2) ) * cos(incidence)
        
        # Optimization: Downsample plasma grid for ray tracing
        step = 4
        RR_sub = self.RR[::step, ::step]
        ZZ_sub = self.ZZ[::step, ::step]
        S_sub = Source_Map[::step, ::step]
        dV = (self.dR * step) * (self.dZ * step) * 2 * np.pi * RR_sub
        
        # Flatten sources
        src_r = RR_sub.flatten()
        src_z = ZZ_sub.flatten()
        src_S = S_sub.flatten() * dV.flatten()
        
        # Filter only active plasma points
        active_idx = src_S > 0
        src_r = src_r[active_idx]
        src_z = src_z[active_idx]
        src_S = src_S[active_idx]
        
        print(f"  Ray-tracing from {len(src_r)} plasma elements to {len(Rw)} wall segments...")
        
        for i in range(len(Rw)):
            # Target point
            wx, wz = Rw[i], Zw[i]
            
            # Normal vector of wall (approx)
            if i < len(Rw)-1:
                dx, dz = Rw[i+1]-Rw[i], Zw[i+1]-Zw[i]
            else:
                dx, dz = Rw[0]-Rw[i], Zw[0]-Zw[i]
            normal = np.array([-dz, dx])
            normal /= np.linalg.norm(normal)
            
            # Vector from source to target
            vec_r = wx - src_r
            vec_z = wz - src_z
            dist_sq = vec_r**2 + vec_z**2
            dist = np.sqrt(dist_sq)
            
            # Cosine incidence (Dot product with normal)
            # We assume toroidal symmetry, so we treat it as line source mostly
            # Simplified flux: Flux ~ Source / Distance (Cylindrical decay approx)
            # Correct Spherical: Source / Distance^2
            
            flux_contrib = src_S / (4 * np.pi * dist_sq)
            
            # Sum up
            wall_flux[i] = np.sum(flux_contrib)
            
        return Rw, Zw, wall_flux

    def analyze_materials(self, wall_flux):
        """
        Calculates lifespan for different materials.
        """
        # Convert Flux (n/m2/s) to MW/m2 (14 MeV per neutron)
        # 1 MeV = 1.6e-13 J
        # Flux * 14 * 1.6e-13 = Watts/m2
        MW_m2 = wall_flux * 14.1 * 1.602e-13 / 1e6 # MW/m2
        
        peak_load = np.max(MW_m2)
        avg_load = np.mean(MW_m2)
        
        print(f"[Nuclear] Wall Loading: Peak={peak_load:.2f} MW/m2, Avg={avg_load:.2f} MW/m2")
        
        results = {}
        for mat_name, props in MATERIALS.items():
            # DPA accumulation per year
            # Rule of thumb: 1 MW/m2 ~ 10 DPA/fpy (Full Power Year)
            dpa_per_year = peak_load * 10.0 
            
            lifespan = props['dpa_limit'] / dpa_per_year if dpa_per_year > 0 else 999.0
            results[mat_name] = lifespan
            
        return results, MW_m2

def run_nuclear_sim():
    print("--- SCPN NUCLEAR ENGINEERING: Ash & Materials ---")
    config_path = "03_CODE/SCPN-Fusion-Core/iter_config.json"
    lab = NuclearEngineeringLab(config_path)
    
    # 1. Ash Simulation
    # Simulate two scenarios: Good Pumping (tau=5) vs Bad Pumping (tau=15)
    hist_good = lab.simulate_ash_poisoning(tau_He_ratio=5.0)
    hist_bad = lab.simulate_ash_poisoning(tau_He_ratio=15.0)
    
    # 2. Neutron Wall Load
    Rw, Zw, neutron_flux = lab.calculate_neutron_wall_loading()
    
    # 3. Material Analysis
    lifespans, mw_load = lab.analyze_materials(neutron_flux)
    
    # --- VISUALIZATION ---
    fig = plt.figure(figsize=(15, 10))
    
    # Plot A: Ash Poisoning
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(hist_good['time'], hist_good['P_fus'], 'g-', label='Good Pumping (Tau*=5)')
    ax1.plot(hist_bad['time'], hist_bad['P_fus'], 'r--', label='Bad Pumping (Tau*=15)')
    ax1.set_title("Fusion Power Evolution (Helium Poisoning)")
    ax1.set_xlabel("Burn Time (s)")
    ax1.set_ylabel("Power (MW)")
    ax1.legend()
    ax1.grid(True)
    
    # Plot B: Helium Fraction
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(hist_good['time'], np.array(hist_good['f_He'])*100, 'g-', label='He % (Good)')
    ax2.plot(hist_bad['time'], np.array(hist_bad['f_He'])*100, 'r--', label='He % (Bad)')
    ax2.axhline(10.0, color='k', linestyle=':', label='Dilution Limit (10%)')
    ax2.set_title("Helium Ash Accumulation")
    ax2.set_ylabel("He Concentration (%)")
    ax2.legend()
    ax2.grid(True)
    
    # Plot C: Neutron Wall Load (Heatmap on Wall)
    ax3 = fig.add_subplot(2, 2, 3)
    # Plot Plasma Core
    ax3.contour(lab.RR, lab.ZZ, lab.Psi, levels=10, colors='gray', alpha=0.3)
    # Plot Wall colored by Load
    sc = ax3.scatter(Rw, Zw, c=mw_load, cmap='inferno', s=20)
    plt.colorbar(sc, ax=ax3, label='Neutron Load (MW/m2)')
    ax3.set_title("Neutron Flux Distribution (2D)")
    ax3.axis('equal')
    
    # Plot D: Component Lifespan
    ax4 = fig.add_subplot(2, 2, 4)
    mats = list(lifespans.keys())
    years = list(lifespans.values())
    colors = ['gray', 'orange', 'green']
    ax4.bar(mats, years, color=colors)
    ax4.set_title("First Wall Component Lifespan (at Peak Flux)")
    ax4.set_ylabel("Full Power Years (FPY)")
    ax4.axhline(5.0, color='r', linestyle='--', label='Maintenance Cycle (5y)')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig("Nuclear_Engineering_Report.png")
    print("\nResults saved: Nuclear_Engineering_Report.png")
    print("Material Lifespan Estimates:")
    for m, y in lifespans.items():
        print(f"  {m}: {y:.1f} years")

if __name__ == "__main__":
    run_nuclear_sim()
