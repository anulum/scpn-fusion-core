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
        self.Z_eff = 1.0
        self.last_tau_cq_s = 0.02

    @staticmethod
    def estimate_z_eff(neon_quantity_mol):
        """
        Reduced impurity-dilution model for SPI-induced effective charge.

        Maps injected neon quantity to an approximate post-mixing Zeff used by
        the current-quench model. Calibrated for deterministic campaign studies,
        not first-principles impurity transport.
        """
        neon = max(float(neon_quantity_mol), 0.0)
        # Map pellet quantity to impurity fraction in a bounded way.
        impurity_fraction = np.clip((neon / 0.12) * 0.015, 0.0, 0.12)
        z_imp = 10.0  # Reduced effective charge state for cold neon mix.
        zeff = (1.0 - impurity_fraction) * 1.0 + impurity_fraction * (z_imp ** 2)
        return float(np.clip(zeff, 1.0, 12.0))

    @staticmethod
    def estimate_tau_cq(te_keV, z_eff):
        """
        Reduced CQ time-constant model (seconds).

        Targets ITER-like 10-30 ms CQ windows under high-impurity cold-plasma
        conditions while preserving monotonic trends:
        - higher Zeff -> faster quench (smaller tau)
        - colder Te   -> faster quench (smaller tau)
        """
        te = max(float(te_keV), 0.01)
        zeff = max(float(z_eff), 1.0)
        tau = 0.02 * (2.0 / zeff) * ((te / 0.1) ** 0.25)
        return float(np.clip(tau, 0.002, 0.05))
        
    def trigger_mitigation(self, neon_quantity_mol=0.1, return_diagnostics=False):
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
        history_tau_cq = []
        
        # Physics Loop
        t = 0
        phase = "Thermal Quench"
        self.Z_eff = 1.0
        self.last_tau_cq_s = 0.02
        
        while t < 0.05: # Simulate 50 ms
            # Radiation Loss
            if t > t_mix:
                self.Z_eff = self.estimate_z_eff(neon_quantity_mol)
                # Cooling Power (Simplified Model)
                # P_rad ~ 1 GW for massive injection
                P_rad = 1e9 * (self.Z_eff ** 0.5) * (self.Te / 1.0)**0.5
                
                # Energy drop
                dW = -P_rad * dt
                self.W_th += dW
                
                # Temperature drop (W ~ n T)
                self.Te = max(0.01, self.Te * (self.W_th / (self.W_th - dW)))
                
                if self.Te < 5.0 and phase == "Thermal Quench":
                    phase = "Current Quench"
                    print(
                        f"  [t={t*1000:.1f}ms] Thermal Quench Complete. Entering Current Quench."
                    )
            
            # Current Decay (Current Quench)
            if phase == "Current Quench":
                tau_cq_s = self.estimate_tau_cq(self.Te, self.Z_eff)
                self.last_tau_cq_s = tau_cq_s
                # dI/dt = -I/tau
                dI = -(self.Ip / tau_cq_s) * dt
                self.Ip += dI
                history_tau_cq.append(tau_cq_s * 1000.0)
            else:
                history_tau_cq.append(self.last_tau_cq_s * 1000.0)
                
            history_W.append(self.W_th / 1e6)
            history_I.append(self.Ip / 1e6)
            history_T.append(self.Te)
            time_axis.append(t * 1000)
            t += dt

        if return_diagnostics:
            diagnostics = {
                "z_eff": float(self.Z_eff),
                "tau_cq_ms_mean": float(np.mean(history_tau_cq)) if history_tau_cq else 0.0,
                "tau_cq_ms_p95": float(np.percentile(history_tau_cq, 95)) if history_tau_cq else 0.0,
                "final_current_MA": float(self.Ip / 1e6),
            }
            return time_axis, history_W, history_I, diagnostics
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
