# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — SPI Mitigation
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt

class ShatteredPelletInjection:
    """
    Simulates the mitigation of a Major Disruption using Neon SPI.
    Physics:
    1. Thermal Quench (TQ): Radiation cooling by impurities.
    2. Current Quench (CQ): Current decay due to increased resistivity.
    """
    def __init__(self, Plasma_Energy_MJ=300.0, Plasma_Current_MA=15.0):
        self.W_th = Plasma_Energy_MJ * 1e6 # Joules
        self.Ip = Plasma_Current_MA * 1e6 # Amps
        self.Te = 20.0 # keV initial
        
    def trigger_mitigation(self, neon_quantity_mol=0.1):
        print(f"--- DISRUPTION DETECTED! TRIGGERING SPI ({neon_quantity_mol} mol Neon) ---")
        
        # 1. Assimilation
        # Time for pellet to shatter and mix
        t_mix = 0.002 # 2 ms
        
        # 2. Radiation Phase (Thermal Quench)
        # Radiated Power ~ n_e * n_imp * L_z(Te)
        # Neon cooling rate is massive at 10-100 eV
        
        dt = 1e-5 # 10 us
        time_axis = []
        history_W = []
        history_I = []
        history_T = []
        
        # Physics Loop
        t = 0
        phase = "Thermal Quench"
        
        while t < 0.05: # Simulate 50 ms
            # Radiation Loss
            if t > t_mix:
                # Cooling Power (Simplified Model)
                # P_rad ~ 1 GW for massive injection
                P_rad = 1e9 * (self.Te / 1.0)**0.5 # Depends on T
                
                # Energy drop
                dW = -P_rad * dt
                self.W_th += dW
                
                # Temperature drop (W ~ n T)
                self.Te = max(0.01, self.Te * (self.W_th / (self.W_th - dW)))
                
                if self.Te < 0.1 and phase == "Thermal Quench":
                    phase = "Current Quench"
                    print(f"  [t={t*1000:.1f}ms] Thermal Quench Complete. Plasma Cold.")
            
            # Current Decay (Current Quench)
            if phase == "Current Quench":
                # Cold plasma = High Resistivity (Spitzer)
                # eta ~ T^(-3/2)
                R_plasma = 1e-6 * (1.0 / self.Te)**1.5
                L_plasma = 1e-6 # Inductance
                
                # L dI/dt + R I = 0
                dI = -(R_plasma / L_plasma) * self.Ip * dt
                self.Ip += dI
                
            history_W.append(self.W_th / 1e6)
            history_I.append(self.Ip / 1e6)
            history_T.append(self.Te)
            time_axis.append(t * 1000)
            t += dt
            
        return time_axis, history_W, history_I

def run_spi_test():
    spi = ShatteredPelletInjection()
    t, W, I = spi.trigger_mitigation()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(t, W, 'r-', linewidth=2)
    ax1.set_title("Thermal Energy (Thermal Quench)")
    ax1.set_ylabel("Energy (MJ)")
    ax1.grid(True)
    
    ax2.plot(t, I, 'b-', linewidth=2)
    ax2.set_title("Plasma Current (Current Quench)")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Current (MA)")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("SPI_Mitigation_Result.png")
    print("Saved: SPI_Mitigation_Result.png")

if __name__ == "__main__":
    run_spi_test()
