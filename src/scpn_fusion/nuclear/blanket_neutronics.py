# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Blanket Neutronics
"""Reduced-order neutronics surrogate models for blanket studies.

This module provides lightweight, deterministic utilities used by engineering
workflows and tests:

- :class:`BreedingBlanket` for 1D radial transport and tritium breeding estimates.
- :class:`VolumetricBlanketReport` for compact aggregate material-balance outputs.

The implementation intentionally omits Monte-Carlo detail and is suitable for
fast iteration, benchmarking, and regression checks in the native solver pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ._blanket_validators import _require_finite_float
from .multigroup_blanket import MultiGroupBlanket

logger = logging.getLogger(__name__)

__all__ = [
    "BreedingBlanket",
    "MultiGroupBlanket",
    "VolumetricBlanketReport",
    "run_breeding_sim",
]


@dataclass(frozen=True)
class VolumetricBlanketReport:
    """Reduced 3D blanket surrogate summary."""

    tbr: float
    total_production_per_s: float
    incident_neutrons_per_s: float
    blanket_volume_m3: float
    tbr_ideal: float = 0.0


class BreedingBlanket:
    """1D Cylindrical Neutronics Transport Code for TBR calculation.

    Simulates neutron attenuation in a blanket annulus (r_inner to r_outer).
    """

    def __init__(
        self, thickness_cm: float = 100, li6_enrichment: float = 1.0, r_inner_cm: float = 200.0
    ) -> None:
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

        # Cross sections (macroscopic Sigma in cm^-1) - reduced-order 14 MeV closure.
        # ENRICHED BLANKET (90% Li-6 + Beryllium Multiplier)
        self.Sigma_capture_Li6 = 0.15 * self.li6_enrichment
        self.Sigma_scatter = 0.2
        self.Sigma_parasitic = 0.02
        self.Sigma_multiply = 0.08  # High Multiplication (Beryllium)

        # Multiplier gain (neutrons per (n,2n) reaction)
        self.multiplier_gain = 1.8

    def solve_transport(
        self, incident_flux: float = 1e14, rear_albedo: float = 0.0
    ) -> NDArray[np.float64]:
        """Solve steady-state cylindrical diffusion-reaction equation for neutron flux Phi(r).

        -D * (1/r * d/dr(r * dPhi/dr)) + Sigma_rem * Phi = 0.
        """
        incident_flux = float(incident_flux)
        if (not np.isfinite(incident_flux)) or incident_flux <= 0.0:
            raise ValueError("incident_flux must be finite and > 0")

        rear_albedo = float(rear_albedo)
        if rear_albedo < 0.0 or rear_albedo >= 1.0:
            raise ValueError("rear_albedo must satisfy 0.0 <= rear_albedo < 1.0")

        # Diffusion Coefficient
        Sigma_total = (
            self.Sigma_capture_Li6 + self.Sigma_scatter + self.Sigma_parasitic + self.Sigma_multiply
        )
        D = 1.0 / (3.0 * Sigma_total)

        # Finite Difference Matrix for Cylindrical Laplacian
        # 1/r * d/dr (r dPhi/dr) ~ (r_{i+1/2}(phi_{i+1}-phi_i) - r_{i-1/2}(phi_i-phi_{i-1})) / (r_i * dr^2)
        N = self.points
        A = np.zeros((N, N))
        b = np.zeros(N)

        # Effective removal = absorption minus the net (n,2n) multiplication source.
        Sigma_removal = (
            self.Sigma_capture_Li6
            + self.Sigma_parasitic
            - (self.Sigma_multiply * (self.multiplier_gain - 1.0))
        )

        # A breeding blanket must be subcritical: a non-positive net removal makes
        # the diffusion problem supercritical (the flux grows away from the wall
        # and the breeding ratio comes out unphysically negative). This happens
        # when the Li-6 enrichment is too low for the configured Be multiplier.
        if Sigma_removal <= 0.0:
            raise ValueError(
                "Supercritical blanket: net removal "
                f"{Sigma_removal:.4f} <= 0 (Li-6 enrichment {self.li6_enrichment:.3f} "
                "too low for the configured multiplier); increase enrichment or "
                "reduce the multiplier."
            )

        dr = self.dr

        for i in range(1, N - 1):
            r_i = self.r[i]
            r_plus = r_i + 0.5 * dr
            r_minus = r_i - 0.5 * dr

            # Coefficients from discretization:
            # -D/r_i * [ (r_plus/dr^2) * (phi_{i+1}-phi_i) - (r_minus/dr^2) * (phi_i-phi_{i-1}) ] + Sigma * phi_i = 0

            c_plus = (D * r_plus) / (r_i * dr**2)
            c_minus = (D * r_minus) / (r_i * dr**2)
            c_center = c_plus + c_minus + Sigma_removal

            A[i, i - 1] = -c_minus
            A[i, i] = c_center
            A[i, i + 1] = -c_plus
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

        return phi

    def calculate_tbr(self, phi: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        """Integrate Tritium production over the blanket volume (Cylindrical).

        TBR = (Rate of Tritium Production) / (Rate of Incoming Neutrons).
        """
        # Production Rate density: R(r) = Sigma_Li6 * Phi(r)
        production_rate = self.Sigma_capture_Li6 * phi

        # Integrate over cylindrical volume (per unit length)
        # Integral P(r) * 2*pi*r * dr
        integrand = production_rate * 2.0 * np.pi * self.r

        if hasattr(np, "trapezoid"):
            total_production = np.trapezoid(
                integrand, self.r
            )  # pragma: no cover - numpy>=2.0 trapezoid path
        else:
            total_production = np.trapz(integrand, self.r)

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
        logger.info("--- SCPN FUEL CYCLE: Tritium Breeding Ratio (TBR) ---")

    blanket = BreedingBlanket(thickness_cm=thickness_cm, li6_enrichment=li6_enrichment)
    phi = blanket.solve_transport(incident_flux=incident_flux, rear_albedo=rear_albedo)
    tbr, prod_profile = blanket.calculate_tbr(phi)

    status = "SUSTAINABLE" if tbr > 1.05 else "DYING REACTOR"
    if verbose:
        logger.info("Design Thickness: %s cm", blanket.thickness)
        logger.info("Calculated TBR: %.3f", tbr)
        logger.info("Status: %s", status)

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
                logger.info("Saved: %s", output_path)
        except Exception as exc:
            plot_error = str(exc)
            if verbose:
                logger.info("Simulation completed without plot artifact: %s", exc)

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
