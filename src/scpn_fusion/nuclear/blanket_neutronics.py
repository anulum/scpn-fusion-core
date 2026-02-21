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
from numpy.typing import NDArray


@dataclass(frozen=True)
class VolumetricBlanketReport:
    """Reduced 3D blanket surrogate summary."""

    tbr: float
    total_production_per_s: float
    incident_neutrons_per_s: float
    blanket_volume_m3: float
    tbr_ideal: float = 0.0


def _require_finite_float(
    name: str,
    value: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    if min_value is not None and out < min_value:
        raise ValueError(f"{name} must be >= {min_value}.")
    if max_value is not None and out > max_value:
        raise ValueError(f"{name} must be <= {max_value}.")
    return out


def _require_int(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    out = int(value)
    if out < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return out


class BreedingBlanket:
    """
    1D Cylindrical Neutronics Transport Code for TBR calculation.
    Simulates neutron attenuation in a blanket annulus (r_inner to r_outer).
    """
    def __init__(self, thickness_cm: float = 100, li6_enrichment: float = 1.0, r_inner_cm: float = 200.0) -> None:
        self.thickness = _require_finite_float("thickness_cm", thickness_cm, min_value=0.1)
        self.r_inner = _require_finite_float("r_inner_cm", r_inner_cm, min_value=10.0)
        self.li6_enrichment = _require_finite_float(
            "li6_enrichment",
            li6_enrichment,
            min_value=0.0,
            max_value=1.0,
        )
        self.points = 100
        # Radial grid from r_inner to r_outer
        self.r = np.linspace(self.r_inner, self.r_inner + self.thickness, self.points)
        self.dr = self.r[1] - self.r[0]
        # For legacy compatibility, alias x to distance from first wall
        self.x = self.r - self.r_inner
        
        # Cross Sections (Macroscopic Sigma in cm^-1) - Simplified for 14 MeV neutrons
        # ENRICHED BLANKET (90% Li-6 + Beryllium Multiplier)
        self.Sigma_capture_Li6 = 0.15 * self.li6_enrichment
        self.Sigma_scatter = 0.2      
        self.Sigma_parasitic = 0.02   
        self.Sigma_multiply = 0.08    # High Multiplication (Beryllium)
        
        # Multiplier gain (neutrons per (n,2n) reaction)
        self.multiplier_gain = 1.8 

    def solve_transport(self, incident_flux: float = 1e14, rear_albedo: float = 0.0) -> NDArray[np.float64]:
        """
        Solves steady-state cylindrical diffusion-reaction equation for neutron flux Phi(r).
        -D * (1/r * d/dr(r * dPhi/dr)) + Sigma_rem * Phi = 0
        """
        incident_flux = float(incident_flux)
        if (not np.isfinite(incident_flux)) or incident_flux <= 0.0:
            raise ValueError("incident_flux must be finite and > 0")

        rear_albedo = float(rear_albedo)
        if rear_albedo < 0.0 or rear_albedo >= 1.0:
            raise ValueError("rear_albedo must satisfy 0.0 <= rear_albedo < 1.0")

        # Diffusion Coefficient
        Sigma_total = self.Sigma_capture_Li6 + self.Sigma_scatter + self.Sigma_parasitic + self.Sigma_multiply
        D = 1.0 / (3.0 * Sigma_total)
        
        # Finite Difference Matrix for Cylindrical Laplacian
        # 1/r * d/dr (r dPhi/dr) ~ (r_{i+1/2}(phi_{i+1}-phi_i) - r_{i-1/2}(phi_i-phi_{i-1})) / (r_i * dr^2)
        N = self.points
        A = np.zeros((N, N))
        b = np.zeros(N)
        
        # Effective removal
        Sigma_removal = self.Sigma_capture_Li6 + self.Sigma_parasitic - (self.Sigma_multiply * (self.multiplier_gain - 1.0))
        
        dr = self.dr
        
        for i in range(1, N-1):
            r_i = self.r[i]
            r_plus = r_i + 0.5*dr
            r_minus = r_i - 0.5*dr
            
            # Coefficients from discretization:
            # -D/r_i * [ (r_plus/dr^2) * (phi_{i+1}-phi_i) - (r_minus/dr^2) * (phi_i-phi_{i-1}) ] + Sigma * phi_i = 0
            
            c_plus = (D * r_plus) / (r_i * dr**2)
            c_minus = (D * r_minus) / (r_i * dr**2)
            c_center = c_plus + c_minus + Sigma_removal
            
            A[i, i-1] = -c_minus
            A[i, i]   = c_center
            A[i, i+1] = -c_plus
            b[i] = 0
            
        # Boundary Conditions
        # r=r_inner (First Wall): Flux imposed
        A[0, 0] = 1.0
        b[0] = incident_flux
        
        # r=r_outer (Shield): Albedo
        A[-1, -1] = 1.0
        A[-1, -2] = -rear_albedo
        b[-1] = 0.0
        
        # Solve
        phi = np.linalg.solve(A, b)
        
        return phi  # type: ignore[return-value,unused-ignore]

    def calculate_tbr(self, phi: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        """
        Integrates Tritium production over the blanket volume (Cylindrical).
        TBR = (Rate of Tritium Production) / (Rate of Incoming Neutrons)
        """
        # Production Rate density: R(r) = Sigma_Li6 * Phi(r)
        production_rate = self.Sigma_capture_Li6 * phi
        
        # Integrate over cylindrical volume (per unit length)
        # Integral P(r) * 2*pi*r * dr
        integrand = production_rate * 2.0 * np.pi * self.r
        
        if hasattr(np, "trapezoid"):
            total_production = np.trapezoid(integrand, self.r)
        else:  # pragma: no cover - legacy NumPy fallback
            total_production = np.trapz(integrand, self.r)  # type: ignore[attr-defined,unused-ignore]
        
        # Incoming Current (per unit length)
        # Total neutrons entering cylinder surface = J_in * Area
        # Area = 2*pi*r_inner * 1
        # J_in ~ Phi[0]/4 (isotropic)
        
        incident_current = (phi[0] / 4.0) * (2.0 * np.pi * self.r_inner)
        
        # TBR calculation
        TBR = total_production / max(incident_current, 1e-12)
        
        return TBR, production_rate

    def calculate_volumetric_tbr(
        self,
        major_radius_m: float = 6.2,
        minor_radius_m: float = 2.0,
        elongation: float = 1.7,
        radial_cells: int = 24,
        poloidal_cells: int = 72,
        toroidal_cells: int = 48,
        incident_flux: float = 1e14,
        port_coverage_factor: float = 0.80,
        streaming_factor: float = 0.85,
        blanket_fill_factor: float = 1.0,
    ) -> VolumetricBlanketReport:
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

        port_coverage_factor = float(port_coverage_factor)
        if not (0.0 < port_coverage_factor <= 1.0):
            raise ValueError("port_coverage_factor must be in (0, 1].")
        streaming_factor = float(streaming_factor)
        if not (0.0 < streaming_factor <= 1.0):
            raise ValueError("streaming_factor must be in (0, 1].")
        blanket_fill_factor = float(blanket_fill_factor)
        if not (0.0 < blanket_fill_factor <= 1.0):
            raise ValueError("blanket_fill_factor must be in (0, 1].")

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
        tbr_ideal = total_production / max(incident_neutrons, 1e-9)

        # Apply 3D correction factors (Fischer et al. 2015, DEMO blanket studies):
        # - port_coverage_factor: fraction of first wall covered by blanket modules
        #   (~80% for ITER/DEMO; rest is heating, diagnostic, maintenance ports)
        # - streaming_factor: neutron streaming losses through inter-module gaps
        # - blanket_fill_factor: optional breeding material packing fraction
        tbr_vol = tbr_ideal * port_coverage_factor * streaming_factor * blanket_fill_factor

        return VolumetricBlanketReport(
            tbr=tbr_vol,
            total_production_per_s=total_production,
            incident_neutrons_per_s=incident_neutrons,
            blanket_volume_m3=blanket_volume_m3,
            tbr_ideal=tbr_ideal,
        )

def run_breeding_sim(
    *,
    thickness_cm: float = 80.0,
    li6_enrichment: float = 0.9,
    incident_flux: float = 1e14,
    rear_albedo: float = 0.0,
    save_plot: bool = True,
    output_path: str = "Tritium_Breeding_Result.png",
    verbose: bool = True,
) -> dict[str, object]:
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

class MultiGroupBlanket:
    """3-group neutron transport for tritium breeding ratio calculation.

    Energy groups:
        Group 1 (fast):       E > 1 MeV  (source: 14.1 MeV D-T neutrons)
        Group 2 (epithermal): 1 eV < E < 1 MeV  (down-scattered)
        Group 3 (thermal):    E < 1 eV  (thermalised, main Li-6 capture)

    Includes:
        - Energy-dependent cross sections per group
        - Down-scatter from fast → epithermal → thermal
        - Beryllium (n,2n) multiplication in fast group
        - Li-6(n,t) capture in all groups (dominant in thermal)

    This is a significant upgrade over the single-group BreedingBlanket above.
    """

    def __init__(
        self,
        thickness_cm: float = 80.0,
        li6_enrichment: float = 0.9,
        n_cells: int = 100,
        r_inner_cm: float = 200.0,
    ) -> None:
        self.thickness = _require_finite_float("thickness_cm", thickness_cm, min_value=0.1)
        self.r_inner = _require_finite_float("r_inner_cm", r_inner_cm, min_value=10.0)
        self.li6_enrich = _require_finite_float(
            "li6_enrichment",
            li6_enrichment,
            min_value=0.0,
            max_value=1.0,
        )
        self.n_cells = max(_require_int("n_cells", n_cells, 3), int(self.thickness * 2.5))
        self.r = np.linspace(self.r_inner, self.r_inner + self.thickness, self.n_cells)
        self.dx = self.r[1] - self.r[0]
        # For compatibility
        self.x = self.r - self.r_inner

        # ── Cross sections (cm^-1) per group ─────────────────────────
        # Group 1: fast (14 MeV)
        # Li-6(n,t) at 14 MeV is small (~25 mb); Be(n,2n) threshold ~1.8 MeV.
        self.sigma_capture_g1 = 0.005 * self.li6_enrich  # Li-6 capture at 14 MeV (small)
        self.sigma_scatter_g1 = 0.20  # elastic scatter
        self.sigma_multiply_g1 = 0.10  # Be (n,2n) at 14 MeV
        self.sigma_downscatter_12 = 0.20  # fast → epithermal (inelastic)
        self.sigma_parasitic_g1 = 0.005  # structural parasitic

        # Group 2: epithermal (keV–MeV)
        # Li-6 has resonance capture in the keV range.
        self.sigma_capture_g2 = 0.05 * self.li6_enrich  # Li-6 resonance capture
        self.sigma_scatter_g2 = 0.15
        self.sigma_downscatter_23 = 0.18  # epithermal → thermal (moderation)
        self.sigma_parasitic_g2 = 0.01

        # Group 3: thermal (< 1 eV)
        # Li-6(n,t) at thermal: ~940 barns, dominant capture pathway.
        # LiPb atom density × micro-sigma → macro ~0.8 cm^-1 at 90% enrichment.
        self.sigma_capture_g3 = 0.80 * self.li6_enrich  # Li-6 dominant at thermal
        self.sigma_scatter_g3 = 0.05
        self.sigma_parasitic_g3 = 0.01

        self.multiplier_gain = 1.8  # Be(n,2n) neutron gain

    def _solve_cylindrical_group(
        self, D: float, sigma_rem: float, source: NDArray[np.float64], 
        bc_left: tuple[str, float], bc_right: tuple[str, float]
    ) -> NDArray[np.float64]:
        """Solve 1D cylindrical diffusion for a single group."""
        N = self.n_cells
        dr = self.dx
        A = np.zeros((N, N))
        b = source.copy()
        
        for i in range(1, N-1):
            r_i = self.r[i]
            r_p = r_i + 0.5*dr
            r_m = r_i - 0.5*dr
            c_p = (D * r_p) / (r_i * dr**2)
            c_m = (D * r_m) / (r_i * dr**2)
            A[i, i-1] = -c_m
            A[i, i]   = c_p + c_m + sigma_rem
            A[i, i+1] = -c_p
            
        # Left BC (r = r_inner)
        if bc_left[0] == "dirichlet":
            A[0, 0] = 1.0
            b[0] = bc_left[1]
        elif bc_left[0] == "neumann":
            A[0, 0] = 1.0
            A[0, 1] = -1.0
            b[0] = bc_left[1] * dr
            
        # Right BC (r = r_outer)
        if bc_right[0] == "dirichlet":
            A[-1, -1] = 1.0
            b[-1] = bc_right[1]
        elif bc_right[0] == "neumann":
            A[-1, -1] = 1.0
            A[-1, -2] = -1.0
            b[-1] = bc_right[1] * dr
            
        return np.linalg.solve(A, b)

    def solve_transport(
        self,
        incident_flux: float = 1e14,
        port_coverage_factor: float = 0.80,
        streaming_factor: float = 0.85,
    ) -> dict[str, object]:
        """Solve 3-group steady-state cylindrical neutron diffusion."""
        incident_flux = _require_finite_float("incident_flux", incident_flux, min_value=1.0)
        
        # --- Group 1 (fast) ---
        sigma_tot_1 = (self.sigma_capture_g1 + self.sigma_scatter_g1 + 
                       self.sigma_multiply_g1 + self.sigma_downscatter_12 + self.sigma_parasitic_g1)
        D1 = 1.0 / (3.0 * sigma_tot_1)
        sigma_rem_1 = (self.sigma_capture_g1 + self.sigma_downscatter_12 + 
                       self.sigma_parasitic_g1 - self.sigma_multiply_g1 * (self.multiplier_gain - 1.0))
        
        phi_g1 = self._solve_cylindrical_group(
            D1, sigma_rem_1, np.zeros(self.n_cells), 
            ("dirichlet", incident_flux), ("dirichlet", 0.0)
        )
        phi_g1 = np.maximum(phi_g1, 0.0)

        # --- Group 2 (epithermal) ---
        sigma_tot_2 = (self.sigma_capture_g2 + self.sigma_scatter_g2 + 
                       self.sigma_downscatter_23 + self.sigma_parasitic_g2)
        D2 = 1.0 / (3.0 * sigma_tot_2)
        sigma_rem_2 = (self.sigma_capture_g2 + self.sigma_downscatter_23 + self.sigma_parasitic_g2)
        
        source2 = self.sigma_downscatter_12 * phi_g1
        phi_g2 = self._solve_cylindrical_group(
            D2, sigma_rem_2, source2, 
            ("neumann", 0.0), ("dirichlet", 0.0)
        )
        phi_g2 = np.maximum(phi_g2, 0.0)

        # --- Group 3 (thermal) ---
        sigma_tot_3 = (self.sigma_capture_g3 + self.sigma_scatter_g3 + self.sigma_parasitic_g3)
        D3 = 1.0 / (3.0 * sigma_tot_3)
        sigma_rem_3 = self.sigma_capture_g3 + self.sigma_parasitic_g3
        
        source3 = self.sigma_downscatter_23 * phi_g2
        phi_g3 = self._solve_cylindrical_group(
            D3, sigma_rem_3, source3, 
            ("neumann", 0.0), ("dirichlet", 0.0)
        )
        phi_g3 = np.maximum(phi_g3, 0.0)

        # --- TBR and Integration (Cylindrical Volume) ---
        prod_g1 = self.sigma_capture_g1 * phi_g1
        prod_g2 = self.sigma_capture_g2 * phi_g2
        prod_g3 = self.sigma_capture_g3 * phi_g3
        total_prod = prod_g1 + prod_g2 + prod_g3

        trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        total_tritium = trap(total_prod * 2.0 * np.pi * self.r, self.r)
        
        # Incident current (per unit length): J+ * Area_inner
        incident_current_total = (phi_g1[0] / 4.0) * (2.0 * np.pi * self.r_inner)
        tbr_ideal = total_tritium / max(incident_current_total, 1e-12)
        tbr = tbr_ideal * port_coverage_factor * streaming_factor

        return {
            "phi_g1": phi_g1, "phi_g2": phi_g2, "phi_g3": phi_g3,
            "total_production": total_prod,
            "tbr": float(tbr), "tbr_ideal": float(tbr_ideal),
            "incident_current_total": float(incident_current_total),
        }


if __name__ == "__main__":
    run_breeding_sim()
