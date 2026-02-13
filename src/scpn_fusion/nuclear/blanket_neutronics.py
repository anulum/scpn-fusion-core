# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Blanket Neutronics
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class VolumetricBlanketReport:
    """Reduced 3D blanket surrogate summary."""

    tbr: float
    total_production_per_s: float
    incident_neutrons_per_s: float
    blanket_volume_m3: float

class BreedingBlanket:
    """
    1D Neutronics Transport Code for Tritium Breeding Ratio (TBR) calculation.
    Simulates neutron attenuation and Li-6 capture in a Liquid Metal Blanket (LiPb).
    """
    def __init__(self, thickness_cm=100, li6_enrichment=1.0):
        self.thickness = thickness_cm
        self.points = 100
        self.x = np.linspace(0, thickness_cm, self.points)
        self.dx = self.x[1] - self.x[0]
        self.li6_enrichment = float(np.clip(li6_enrichment, 0.0, 1.0))
        
        # Cross Sections (Macroscopic Sigma in cm^-1) - Simplified for 14 MeV neutrons
        # Reaction: Li-6 + n -> T + He + 4.8 MeV
        # ENRICHED BLANKET (90% Li-6 + Beryllium Multiplier)
        self.Sigma_capture_Li6 = 0.15 * self.li6_enrichment
        self.Sigma_scatter = 0.2      
        self.Sigma_parasitic = 0.02   
        self.Sigma_multiply = 0.08    # High Multiplication (Beryllium)
        
        # Multiplier gain (neutrons per (n,2n) reaction)
        self.multiplier_gain = 1.8 

    def solve_transport(self, incident_flux=1e14):
        """
        Solves steady-state diffusion-reaction equation for neutron flux Phi(x).
        -D * d2Phi/dx2 + Sigma_abs * Phi = Source
        """
        # Diffusion Coefficient (D = 1 / 3*Sigma_total)
        Sigma_total = self.Sigma_capture_Li6 + self.Sigma_scatter + self.Sigma_parasitic + self.Sigma_multiply
        D = 1.0 / (3.0 * Sigma_total)
        
        # Finite Difference Matrix
        N = self.points
        A = np.zeros((N, N))
        b = np.zeros(N)
        
        # Absorption term (Capture + Parasitic - Multiplication impact)
        # Effective removal cross section. (n,2n) acts as a negative removal (source) proportional to flux
        Sigma_removal = self.Sigma_capture_Li6 + self.Sigma_parasitic - (self.Sigma_multiply * (self.multiplier_gain - 1.0))
        
        coeff = D / (self.dx**2)
        
        for i in range(1, N-1):
            A[i, i-1] = -coeff
            A[i, i]   = 2*coeff + Sigma_removal
            A[i, i+1] = -coeff
            b[i] = 0 # No internal volume source, only boundary
            
        # Boundary Conditions
        # x=0 (First Wall): Flux imposed by plasma
        A[0, 0] = 1.0
        b[0] = incident_flux
        
        # x=L (Shield): Vacuum/Reflective (Simplified: Flux -> 0)
        A[-1, -1] = 1.0
        b[-1] = 0.0
        
        # Solve
        phi = np.linalg.solve(A, b)
        
        return phi

    def calculate_tbr(self, phi):
        """
        Integrates Tritium production over the blanket volume.
        TBR = (Rate of Tritium Production) / (Rate of Incoming Neutrons)
        """
        # Production Rate density: R(x) = Sigma_Li6 * Phi(x)
        production_rate = self.Sigma_capture_Li6 * phi
        
        # Integrate over thickness (trapezoidal)
        total_production = np.trapz(production_rate, self.x)
        
        # Incoming Current (Approx D * dPhi/dx at boundary, or simplified Incident Flux)
        # TBR is defined relative to 1 source neutron entering.
        # Incident Flux is a density boundary condition.
        # Total neutrons entering per cm2 = Current J_in. 
        # In diffusion approx, J_in ~ Phi[0]/4 + D/2 * dPhi/dx
        # Simplified: We normalize by the source that sustains Phi[0].
        # Flux at boundary is Phi_0. Current into slab is roughly Phi_0/2.
        
        incident_current = phi[0] / 2.0 
        
        # TBR calculation
        TBR = total_production / incident_current
        
        return TBR, production_rate

    def calculate_volumetric_tbr(
        self,
        major_radius_m=6.2,
        minor_radius_m=2.0,
        elongation=1.7,
        radial_cells=24,
        poloidal_cells=72,
        toroidal_cells=48,
        incident_flux=1e14,
    ):
        """
        Reduced 3D blanket-volume surrogate built on top of the 1D transport profile.

        Assumptions:
        - 1D depth attenuation from `solve_transport` is reused as the radial blanket profile.
        - Blanket shell is toroidal with shaped poloidal section via `elongation`.
        - Incident-angle weighting captures first-order poloidal asymmetry.
        """
        major_radius_m = max(float(major_radius_m), 0.1)
        minor_radius_m = max(float(minor_radius_m), 0.05)
        elongation = max(float(elongation), 0.1)
        radial_cells = max(int(radial_cells), 2)
        poloidal_cells = max(int(poloidal_cells), 8)
        toroidal_cells = max(int(toroidal_cells), 8)
        incident_flux = max(float(incident_flux), 1.0)

        # Depth profile anchored to the nominal enriched reference blanket to keep
        # the reduced surrogate stable across parameter scans.
        transport_profile = BreedingBlanket(thickness_cm=80.0, li6_enrichment=0.9)
        phi_1d = transport_profile.solve_transport(incident_flux=incident_flux)
        x_norm = transport_profile.x / max(transport_profile.thickness, 1e-9)

        thickness_m = max(self.thickness * 0.01, 1e-6)
        dr = thickness_m / radial_cells
        dtheta = 2.0 * np.pi / poloidal_cells
        dphi = 2.0 * np.pi / toroidal_cells

        total_production = 0.0
        blanket_volume_m3 = 0.0

        for i in range(radial_cells):
            depth_m = (i + 0.5) * dr
            depth_norm = depth_m / thickness_m
            base_flux = float(np.interp(depth_norm, x_norm, phi_1d))
            shell_r = minor_radius_m + depth_m

            for j in range(poloidal_cells):
                theta = (j + 0.5) * dtheta
                incidence_factor = max(0.2, 0.6 + 0.4 * np.cos(theta) ** 2)
                major_local = max(0.1, major_radius_m + shell_r * np.cos(theta))

                for k in range(toroidal_cells):
                    # Small deterministic toroidal modulation keeps the surrogate 3D-aware.
                    toroidal_factor = 1.0 + 0.05 * np.cos((k + 0.5) * dphi)
                    local_flux = base_flux * incidence_factor * toroidal_factor
                    production_density = self.Sigma_capture_Li6 * local_flux  # [1/cm^3/s]

                    dvol_m3 = elongation * shell_r * dr * dtheta * dphi * major_local
                    blanket_volume_m3 += dvol_m3
                    total_production += production_density * dvol_m3 * 1e6  # m^3 -> cm^3

        first_wall_area_m2 = 4.0 * np.pi**2 * major_radius_m * minor_radius_m * elongation
        incident_neutrons = incident_flux * first_wall_area_m2 * 1e4  # m^2 -> cm^2
        tbr_vol = total_production / max(incident_neutrons, 1e-9)

        return VolumetricBlanketReport(
            tbr=tbr_vol,
            total_production_per_s=total_production,
            incident_neutrons_per_s=incident_neutrons,
            blanket_volume_m3=blanket_volume_m3,
        )

def run_breeding_sim():
    print("--- SCPN FUEL CYCLE: Tritium Breeding Ratio (TBR) ---")
    
    blanket = BreedingBlanket(thickness_cm=80)
    phi = blanket.solve_transport()
    tbr, prod_profile = blanket.calculate_tbr(phi)
    
    print(f"Design Thickness: {blanket.thickness} cm")
    print(f"Calculated TBR: {tbr:.3f}")
    
    status = "SUSTAINABLE" if tbr > 1.05 else "DYING REACTOR"
    print(f"Status: {status}")
    
    # Visuals
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_title(f"Neutron Flux & Tritium Production (TBR={tbr:.2f})")
    ax1.set_xlabel("Distance from First Wall (cm)")
    ax1.set_ylabel("Neutron Flux (n/cm2/s)", color='blue')
    ax1.plot(blanket.x, phi, 'b-', label='Neutron Flux')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Tritium Production Rate", color='green')
    ax2.plot(blanket.x, prod_profile, 'g--', label='T-Production')
    ax2.fill_between(blanket.x, 0, prod_profile, color='green', alpha=0.1)
    ax2.tick_params(axis='y', labelcolor='green')
    
    plt.tight_layout()
    plt.savefig("Tritium_Breeding_Result.png")
    print("Saved: Tritium_Breeding_Result.png")

if __name__ == "__main__":
    run_breeding_sim()
