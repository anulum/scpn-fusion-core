# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — PWI Erosion
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt

class SputteringPhysics:
    """
    Simulates Plasma-Wall Interaction (PWI).
    Calculates Sputtering Yield (Atoms ejected per incident ion)
    and Wall Erosion Rate.
    Based on Eckstein semi-empirical formulas.
    """
    def __init__(self, material="Tungsten", redeposition_factor=0.95):
        self.material = material
        # Physics Parameters for Tungsten (W) impacting by Deuterium (D)
        if material == "Tungsten":
            self.E_th = 200.0 # Threshold energy (eV)
            self.Q = 0.03     # Yield factor
            self.atomic_mass = 183.84 # amu
            self.density = 19.25 # g/cm3
        else:
            # Fallback (Carbon-like)
            self.E_th = 30.0
            self.Q = 0.1
            self.atomic_mass = 12.0
            self.density = 2.2
        self.redeposition_factor = float(np.clip(redeposition_factor, 0.0, 0.999))

    def calculate_yield(self, E_ion_eV, angle_deg=45.0):
        """
        Calculates Sputtering Yield (Y).
        Y = ejected_atoms / incident_ion
        """
        # Reduced Bohdansky-style yield:
        # Y ~ Q * S_n(eps) * (1 - (Eth/E)^(2/3)) * (1 - Eth/E)^2 * f(theta)
        # with bounded angular enhancement.
        E = float(E_ion_eV)
        if E <= self.E_th:
            return 0.0

        eps = E / self.E_th
        eth_ratio = self.E_th / E

        # Nuclear stopping surrogate (monotone in eps and bounded for stability).
        s_n = np.log1p(1.2288 * eps) / (1.0 + np.sqrt(eps))

        # Bohdansky threshold correction.
        threshold_term = (1.0 - eth_ratio ** (2.0 / 3.0)) * (1.0 - eth_ratio) ** 2
        threshold_term = max(0.0, threshold_term)

        # Angular dependence: stronger sputtering near grazing incidence.
        theta = np.deg2rad(np.clip(angle_deg, 0.0, 89.0))
        f_alpha = min(5.0, 1.0 / max(np.cos(theta), 1e-3))

        Y = self.Q * s_n * threshold_term * f_alpha
        return max(0.0, Y)

    def calculate_erosion_rate(self, flux_particles_m2_s, T_ion_eV, angle_deg=45.0):
        """
        Calculates macroscopic erosion rate (mm/year).
        """
        # 1. Sheath Acceleration
        # Ions fall through the sheath potential (~3 * Te)
        # Impact Energy approx 3 * T_ion + Thermal Energy (2 * T_ion) = 5 * T_ion
        E_impact = 5.0 * T_ion_eV
        
        # 2. Yield
        Y = self.calculate_yield(E_impact, angle_deg=angle_deg)
        
        # 3. Gross Erosion Flux (atoms/m2/s)
        Flux_erosion = flux_particles_m2_s * Y
        
        # 4. Redeposition (Crucial!)
        # In magnetic field, 90-99% of sputtered atoms ionize and fly back to wall.
        Flux_net = Flux_erosion * (1.0 - self.redeposition_factor)
        
        # 5. Convert to thickness (mm/year)
        # Mass loss rate = Flux_net * Atomic_Mass / Avogadro
        # Volume loss rate = Mass_rate / Density
        
        # m/s
        recession_speed = (Flux_net * (self.atomic_mass * 1.66e-27)) / (self.density * 1000) 
        
        # mm/year
        seconds_per_year = 3600 * 24 * 365
        mm_year = recession_speed * 1000.0 * seconds_per_year
        
        return {
            'Yield': Y,
            'E_impact': E_impact,
            'Net_Flux': Flux_net,
            'Redeposition': self.redeposition_factor,
            'Erosion_mm_year': mm_year,
            'Impurity_Source': Flux_net # Particles entering plasma
        }

def run_pwi_demo():
    print("--- SCPN PLASMA-WALL INTERACTION: SPUTTERING ---")
    
    pwi = SputteringPhysics("Tungsten")
    
    # Divertor Conditions (High Flux, Low Temp is desired)
    flux = 1e24 # particles/m2/s (Very high!)
    
    temps = np.linspace(10, 100, 50) # Ion Temp at Divertor (eV)
    erosion_rates = []
    yields = []
    
    print(f"{'T_ion (eV)':<10} | {'Impact (eV)':<12} | {'Yield':<10} | {'Erosion (mm/y)':<15}")
    print("-" * 55)
    
    for T in temps:
        res = pwi.calculate_erosion_rate(flux, T)
        erosion_rates.append(res['Erosion_mm_year'])
        yields.append(res['Yield'])
        
        if int(T) % 20 == 0:
            print(f"{T:<10.1f} | {res['E_impact']:<12.1f} | {res['Yield']:<10.4f} | {res['Erosion_mm_year']:<15.2f}")

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Divertor Ion Temperature (eV)')
    ax1.set_ylabel('Sputtering Yield (Y)', color=color)
    ax1.plot(temps, yields, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('Erosion Rate (mm/year)', color=color)
    ax2.plot(temps, erosion_rates, color=color, linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Limit line
    ax2.axhline(5.0, color='k', linestyle=':', label='Max limit (5mm/y)')
    
    plt.title("Tungsten Divertor Erosion vs Plasma Temperature")
    plt.tight_layout()
    plt.savefig("PWI_Erosion_Result.png")
    print("Saved: PWI_Erosion_Result.png")

if __name__ == "__main__":
    run_pwi_demo()
