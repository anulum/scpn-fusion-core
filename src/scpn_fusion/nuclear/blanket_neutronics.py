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

    def solve_transport(self, incident_flux=1e14, rear_albedo=0.0):
        """
        Solves steady-state diffusion-reaction equation for neutron flux Phi(x).
        -D * d2Phi/dx2 + Sigma_abs * Phi = Source
        """
        incident_flux = float(incident_flux)
        if (not np.isfinite(incident_flux)) or incident_flux <= 0.0:
            raise ValueError("incident_flux must be finite and > 0")

        rear_albedo = float(rear_albedo)
        if rear_albedo < 0.0 or rear_albedo >= 1.0:
            raise ValueError("rear_albedo must satisfy 0.0 <= rear_albedo < 1.0")

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
        
        # x=L (Shield): albedo reflection relation phi[L] = A * phi[L-dx].
        # A=0.0 -> vacuum-like sink (legacy behavior), A->1.0 -> strong reflection.
        A[-1, -1] = 1.0
        A[-1, -2] = -rear_albedo
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
        if hasattr(np, "trapezoid"):
            total_production = np.trapezoid(production_rate, self.x)
        else:  # pragma: no cover - legacy NumPy fallback
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
        major_radius_m = float(major_radius_m)
        if (not np.isfinite(major_radius_m)) or major_radius_m < 0.1:
            raise ValueError("major_radius_m must be finite and >= 0.1.")
        minor_radius_m = float(minor_radius_m)
        if (not np.isfinite(minor_radius_m)) or minor_radius_m < 0.05:
            raise ValueError("minor_radius_m must be finite and >= 0.05.")
        elongation = float(elongation)
        if (not np.isfinite(elongation)) or elongation < 0.1:
            raise ValueError("elongation must be finite and >= 0.1.")
        radial_cells = int(radial_cells)
        if radial_cells < 2:
            raise ValueError("radial_cells must be >= 2.")
        poloidal_cells = int(poloidal_cells)
        if poloidal_cells < 8:
            raise ValueError("poloidal_cells must be >= 8.")
        toroidal_cells = int(toroidal_cells)
        if toroidal_cells < 8:
            raise ValueError("toroidal_cells must be >= 8.")
        incident_flux = float(incident_flux)
        if (not np.isfinite(incident_flux)) or incident_flux < 1.0:
            raise ValueError("incident_flux must be finite and >= 1.0.")

        # Depth profile anchored to the nominal enriched reference blanket to keep
        # the reduced surrogate stable across parameter scans.
        transport_profile = BreedingBlanket(thickness_cm=80.0, li6_enrichment=0.9)
        phi_1d = transport_profile.solve_transport(
            incident_flux=incident_flux,
            rear_albedo=0.5,
        )
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

def run_breeding_sim(
    *,
    thickness_cm=80.0,
    li6_enrichment=0.9,
    incident_flux=1e14,
    rear_albedo=0.0,
    save_plot=True,
    output_path="Tritium_Breeding_Result.png",
    verbose=True,
):
    """Run deterministic blanket breeding simulation and return summary metrics."""
    thickness_cm = float(thickness_cm)
    if (not np.isfinite(thickness_cm)) or thickness_cm <= 0.0:
        raise ValueError("thickness_cm must be finite and > 0")

    li6_enrichment = float(li6_enrichment)
    if (not np.isfinite(li6_enrichment)) or li6_enrichment < 0.0 or li6_enrichment > 1.0:
        raise ValueError("li6_enrichment must satisfy 0.0 <= li6_enrichment <= 1.0")

    if verbose:
        print("--- SCPN FUEL CYCLE: Tritium Breeding Ratio (TBR) ---")

    blanket = BreedingBlanket(thickness_cm=thickness_cm, li6_enrichment=li6_enrichment)
    phi = blanket.solve_transport(incident_flux=incident_flux, rear_albedo=rear_albedo)
    tbr, prod_profile = blanket.calculate_tbr(phi)

    status = "SUSTAINABLE" if tbr > 1.05 else "DYING REACTOR"
    if verbose:
        print(f"Design Thickness: {blanket.thickness} cm")
        print(f"Calculated TBR: {tbr:.3f}")
        print(f"Status: {status}")

    plot_saved = False
    plot_error = None
    if save_plot:
        try:
            fig, ax1 = plt.subplots(figsize=(10, 6))

            ax1.set_title(f"Neutron Flux & Tritium Production (TBR={tbr:.2f})")
            ax1.set_xlabel("Distance from First Wall (cm)")
            ax1.set_ylabel("Neutron Flux (n/cm2/s)", color="blue")
            ax1.plot(blanket.x, phi, "b-", label="Neutron Flux")
            ax1.tick_params(axis="y", labelcolor="blue")

            ax2 = ax1.twinx()
            ax2.set_ylabel("Tritium Production Rate", color="green")
            ax2.plot(blanket.x, prod_profile, "g--", label="T-Production")
            ax2.fill_between(blanket.x, 0, prod_profile, color="green", alpha=0.1)
            ax2.tick_params(axis="y", labelcolor="green")

            plt.tight_layout()
            plt.savefig(output_path)
            plt.close(fig)
            plot_saved = True
            if verbose:
                print(f"Saved: {output_path}")
        except Exception as exc:
            plot_error = str(exc)
            if verbose:
                print(f"Simulation completed without plot artifact: {exc}")

    summary = {
        "thickness_cm": float(blanket.thickness),
        "li6_enrichment": float(blanket.li6_enrichment),
        "incident_flux": float(incident_flux),
        "rear_albedo": float(rear_albedo),
        "tbr": float(tbr),
        "status": status,
        "flux_peak": float(np.max(phi)),
        "flux_mean": float(np.mean(phi)),
        "production_peak": float(np.max(prod_profile)),
        "production_mean": float(np.mean(prod_profile)),
        "plot_saved": bool(plot_saved),
        "plot_error": plot_error,
    }
    return summary

if __name__ == "__main__":
    run_breeding_sim()
