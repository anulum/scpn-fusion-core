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
    def __init__(self, material="Tungsten"):
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

    def calculate_yield(self, E_ion_eV, angle_deg=45.0):
        """
        Calculates Sputtering Yield (Y).
        Y = ejected_atoms / incident_ion
        """
        # Eckstein Bohdansky Formula (Simplified)
        # Y depends heavily on Energy E
        
        if E_ion_eV < self.E_th:
            return 0.0
        
        reduced_E = E_ion_eV / self.E_th
        f_E = (np.log(reduced_E) * (reduced_E - 1)) / (reduced_E**2 + 1)
        
        # Angular dependence (Grazing incidence increases sputtering)
        # But for rough surfaces, it averages out. We use a factor.
        f_alpha = 1.0 + (angle_deg / 90.0)**2 
        
        Y = self.Q * f_E * f_alpha
        return max(0.0, Y)

    def calculate_erosion_rate(self, flux_particles_m2_s, T_ion_eV):
        """
        Calculates macroscopic erosion rate (mm/year).
        """
        # 1. Sheath Acceleration
        # Ions fall through the sheath potential (~3 * Te)
        # Impact Energy approx 3 * T_ion + Thermal Energy (2 * T_ion) = 5 * T_ion
        E_impact = 5.0 * T_ion_eV
        
        # 2. Yield
        Y = self.calculate_yield(E_impact)
        
        # 3. Gross Erosion Flux (atoms/m2/s)
        Flux_erosion = flux_particles_m2_s * Y
        
        # 4. Redeposition (Crucial!)
        # In magnetic field, 90-99% of sputtered atoms ionize and fly back to wall.
        R_redep = 0.95 # 95% redeposition efficiency
        Flux_net = Flux_erosion * (1.0 - R_redep)
        
        # 5. Convert to thickness (mm/year)
        # Mass loss rate = Flux_net * Atomic_Mass / Avogadro
        # Volume loss rate = Mass_rate / Density
        
        Avogadro = 6.022e23
        mass_per_atom_g = self.atomic_mass / Avogadro
        
        # g/m2/s
        mass_loss_rate = Flux_net * mass_per_atom_g
        
        # cm3/m2/s = (g/m2/s) / (g/cm3) -> needs unit conversion
        # density is g/cm3 = 1e6 g/m3
        density_SI = self.density * 1e6
        
        # m/s
        recession_speed = (Flux_net * (self.atomic_mass * 1.66e-27)) / (self.density * 1000) 
        
        # mm/year
        seconds_per_year = 3600 * 24 * 365
        mm_year = recession_speed * 1000.0 * seconds_per_year
        
        return {
            'Yield': Y,
            'E_impact': E_impact,
            'Net_Flux': Flux_net,
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
