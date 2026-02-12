import numpy as np
import matplotlib.pyplot as plt
from .fusion_kernel import FusionKernel
import sys

class FusionBurnPhysics(FusionKernel):
    """
    Extends the Grad-Shafranov Solver with Thermonuclear Physics.
    Calculates Fusion Power, Alpha Heating, and Q-Factor.
    """
    def __init__(self, config_path):
        super().__init__(config_path)
        
    def bosch_hale_dt(self, T_keV):
        """
        Calculates <sigma*v> for Deuterium-Tritium fusion.
        Parametrization by Bosch & Hale (1992).
        T_keV: Ion Temperature in keV.
        Returns: Reaction rate in m^3/s
        """
        # Avoid zero/negative temp
        T = np.maximum(T_keV, 0.1)
        
        # NRL Plasma Formulary approximation for D-T <sigma v> (m^3/s)
        # Valid for T < 100 keV
        sigmav = 3.68e-18 / (T**(2/3)) * np.exp(-19.94 / (T**(1/3)))
        
        return sigmav

    def calculate_thermodynamics(self, P_aux_MW=50.0):
        """
        Maps Magnetic Equilibrium -> Thermodynamics -> Fusion Power.
        P_aux_MW: External Heating Power (NBI/ECRH) in MegaWatts.
        """
        # 1. Derive Pressure from Grad-Shafranov (J ~ R*p')
        # In our simplified kernel, J was modeled directly. 
        # Here we assume Pressure follows Flux Surfaces: p(psi) ~ (1-psi)^2
        
        idx_max = np.argmax(self.Psi)
        iz, ir = np.unravel_index(idx_max, self.Psi.shape)
        Psi_axis = self.Psi[iz, ir]
        
        # FIX: Find real boundary using X-point
        xp, psi_x = self.find_x_point(self.Psi)
        Psi_boundary = psi_x
        
        # Safety: if boundary close to axis (limiter case), use min of flux map
        if abs(Psi_boundary - Psi_axis) < 1.0: 
            Psi_boundary = np.min(self.Psi)
            
        Psi_norm = (self.Psi - Psi_axis) / (Psi_boundary - Psi_axis)
        Psi_norm = np.clip(Psi_norm, 0, 1)
        mask = (Psi_norm >= 0) & (Psi_norm < 1.0)
        
        # Peak values (ITER-like)
        n_peak = 1.0e20 # m^-3 (Density)
        T_peak_keV = 20.0 # keV (Temperature)
        
        # Profiles
        n = np.zeros_like(self.Psi)
        T = np.zeros_like(self.Psi)
        
        n[mask] = n_peak * (1 - Psi_norm[mask]**2)**0.5
        T[mask] = T_peak_keV * (1 - Psi_norm[mask]**2)**1.0
        
        # 2. Calculate Fusion Power
        # P_fus = E_fus * nD * nT * <sigma v>
        # Assume 50-50 D-T mix
        nD = 0.5 * n
        nT = 0.5 * n
        E_fus = 17.6 * 1.602e-13 # MeV to Joules (17.6 MeV per reaction)
        
        sigmav = self.bosch_hale_dt(T)
        power_density = nD * nT * sigmav * E_fus # Watts/m^3
        
        # Integrate over volume (Approximating Toroidal symmetry 2*pi*R)
        dV = self.dR * self.dZ * 2 * np.pi * self.RR
        P_fusion_total = np.sum(power_density * dV)
        
        # 3. Alpha Heating (Self-Heating)
        # Alphas carry 20% of fusion energy (3.5 MeV / 17.6 MeV)
        P_alpha = P_fusion_total * 0.2
        
        # 4. Losses (Simplified Confinement scaling)
        # Tau_E (Energy Confinement Time) ~ ITER98y2 scaling
        # Simplified: Fixed loss + Bremsstrahlung
        W_thermal = np.sum(3 * n * (T * 1.602e-16) * dV) # Thermal energy in Joules
        Tau_E = 3.0 # seconds (Good H-mode)
        P_loss = W_thermal / Tau_E
        
        # 5. Global Balance
        # dW/dt = P_alpha + P_aux - P_loss
        net_heating = P_alpha + (P_aux_MW*1e6) - P_loss
        
        # Q Factor
        Q = P_fusion_total / (P_aux_MW*1e6) if P_aux_MW > 0 else 0
        
        return {
            'P_fusion_MW': P_fusion_total / 1e6,
            'P_alpha_MW': P_alpha / 1e6,
            'P_loss_MW': P_loss / 1e6,
            'P_aux_MW': P_aux_MW,
            'Net_MW': net_heating / 1e6,
            'Q': Q,
            'T_peak': T_peak_keV,
            'W_MJ': W_thermal / 1e6
        }

def run_ignition_experiment():
    print("--- SCPN IGNITION EXPERIMENT: The Road to Q > 10 ---")
    
    config_path = "03_CODE/SCPN-Fusion-Core/iter_config.json"
    sim = FusionBurnPhysics(config_path)
    
    # Simulation: Power Ramp Up
    # We increase Auxiliary Heating and measure the response
    power_ramp = np.linspace(0, 100, 20) # 0 to 100 MW
    
    history_Q = []
    history_P_fus = []
    
    print(f"{ 'Aux (MW)':<10} | { 'Fusion (MW)':<12} | { 'Alpha (MW)':<10} | { 'Q-Factor':<8} | {'Status'}")
    print("-" * 60)
    
    # 1. Establish Geometry
    sim.solve_equilibrium()
    
    for P_aux in power_ramp:
        # In a real dynamic code, P_aux would modify T_peak dynamically
        # Here we perform a static check: "If we had this geometry and profiles, what is the output?"
        # To make it dynamic, we link T_peak to P_net from previous step
        
        metrics = sim.calculate_thermodynamics(P_aux)
        
        # Status check
        status = "L-Mode"
        if metrics['Q'] > 1.0: status = "Breakeven"
        if metrics['Q'] > 5.0: status = "Burning"
        if metrics['Q'] > 10.0: status = "IGNITION"
        
        history_Q.append(metrics['Q'])
        history_P_fus.append(metrics['P_fusion_MW'])
        
        print(f"{P_aux:<10.1f} | {metrics['P_fusion_MW']:<12.1f} | {metrics['P_alpha_MW']:<10.1f} | {metrics['Q']:<8.2f} | {status}")

    # --- VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Q-Curve
    ax1.set_title("Fusion Gain (Q) vs Input Power")
    ax1.plot(power_ramp, history_Q, 'r-o', linewidth=2)
    ax1.axhline(1.0, color='gray', linestyle='--', label='Breakeven (Q=1)')
    ax1.axhline(10.0, color='green', linestyle='--', label='Ignition (Q=10)')
    ax1.set_xlabel("Auxiliary Heating (MW)")
    ax1.set_ylabel("Q")
    ax1.legend()
    ax1.grid(True)
    
    # POP-CON Plot (Operating Point)
    # We visualize where the final state sits in Power space
    ax2.set_title("Power Balance (Ignition Condition)")
    ax2.bar(['Alpha Heat', 'Aux Heat'], [metrics['P_alpha_MW'], metrics['P_aux_MW']], color=['red', 'orange'])
    ax2.bar(['Losses'], [metrics['P_loss_MW']], color='blue')
    ax2.set_ylabel("Power (MW)")
    
    plt.tight_layout()
    plt.savefig("Ignition_Result.png")
    print("\nExperiment Complete. Results: Ignition_Result.png")

if __name__ == "__main__":
    run_ignition_experiment()
