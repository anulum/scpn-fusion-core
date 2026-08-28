# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Coupled Transport Contracts
"""Typed inputs and evidence records for coupled transport stepping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CoupledTransportInputs:
    """One prescribed coupled-transport timestep.

    All powers and rates are totals over the named circular torus. Source
    profiles are normalized against that geometry before they enter a state
    equation, so the recorded budgets can be checked independently.
    """

    time_s: float
    dt_s: float
    major_radius_m: float
    minor_radius_m: float
    magnetic_field_t: float
    effective_charge: float
    ion_heat_diffusivity_m2_s: float
    electron_heat_diffusivity_m2_s: float
    electron_particle_diffusivity_m2_s: float
    heat_power_w: float
    electron_heat_fraction: float
    heat_center_rho: float
    heat_width_rho: float
    particle_rate_s: float
    particle_center_rho: float
    particle_width_rho: float
    driven_current_a: float
    current_center_rho: float
    current_width_rho: float
    ion_electron_exchange_rate_s: float
    ion_temperature_edge_kev: float
    electron_temperature_edge_kev: float
    electron_density_edge_1e19_m3: float
    resistivity_multiplier: float = 1.0

    def __post_init__(self) -> None:
        """Reject malformed or physically inadmissible deck values."""
        finite_values = {name: float(value) for name, value in vars(self).items()}
        for name, value in finite_values.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        positive = (
            "dt_s",
            "major_radius_m",
            "minor_radius_m",
            "magnetic_field_t",
            "effective_charge",
            "ion_heat_diffusivity_m2_s",
            "electron_heat_diffusivity_m2_s",
            "electron_particle_diffusivity_m2_s",
            "heat_width_rho",
            "particle_width_rho",
            "current_width_rho",
            "ion_temperature_edge_kev",
            "electron_temperature_edge_kev",
            "electron_density_edge_1e19_m3",
            "resistivity_multiplier",
        )
        for name in positive:
            if finite_values[name] <= 0.0:
                raise ValueError(f"{name} must be > 0")
        non_negative = (
            "time_s",
            "heat_power_w",
            "particle_rate_s",
            "driven_current_a",
            "ion_electron_exchange_rate_s",
        )
        for name in non_negative:
            if finite_values[name] < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if not 0.0 <= self.electron_heat_fraction <= 1.0:
            raise ValueError("electron_heat_fraction must be in [0, 1]")
        for name in ("heat_center_rho", "particle_center_rho", "current_center_rho"):
            value = finite_values[name]
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class CoupledTransportBudget:
    """Machine-readable state, source, exchange, and solve accounting."""

    thermal_energy_before_j: float
    thermal_energy_after_j: float
    heat_energy_injected_j: float
    ion_exchange_energy_j: float
    electron_exchange_energy_j: float
    thermal_boundary_and_diffusion_j: float
    particle_inventory_before: float
    particle_inventory_after: float
    particles_injected: float
    particle_boundary_and_diffusion: float
    flux_l2_before: float
    flux_l2_after: float
    driven_current_target_a: float
    driven_current_reconstructed_a: float
    ion_heat_source_reconstructed_w: float
    electron_heat_source_reconstructed_w: float
    particle_source_reconstructed_s: float
    ion_temperature_linear_residual_linf: float
    electron_temperature_linear_residual_linf: float
    electron_density_linear_residual_linf: float
    current_linear_residual_linf: float
    ion_electron_exchange_closure_j: float


@dataclass(frozen=True)
class CoupledTransportStepResult:
    """Accepted coupled state and its independently inspectable budget."""

    time_s: float
    rho: FloatArray
    ion_temperature_kev: FloatArray
    electron_temperature_kev: FloatArray
    electron_density_1e19_m3: FloatArray
    poloidal_flux_wb_per_rad: FloatArray
    budget: CoupledTransportBudget
    converged: bool


__all__ = [
    "CoupledTransportBudget",
    "CoupledTransportInputs",
    "CoupledTransportStepResult",
]
