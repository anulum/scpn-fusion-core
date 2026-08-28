# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Coupled Integrated Transport Runtime
"""Audited coupled temperature, density, and current transport stepping.

The ordinary :class:`~scpn_fusion.core.integrated_transport_solver.TransportSolver`
runtime remains the production nonlinear transport path.  This module adds a
prescribed-coefficient model-intersection surface for independent code-to-code
comparison.  One call advances ion temperature, electron temperature, electron
density, and poloidal flux from the same beginning-of-step state while preserving
explicit source, boundary, exchange, and linear-solve accounting.

The model-intersection surface is deliberately narrow: circular geometry,
constant radial diffusivities, Gaussian heat/particle/current sources, symmetric
ion-electron exchange, and prescribed edge values.  Those assumptions are all
caller-visible and are never promoted as a full-device transport model.
"""

from __future__ import annotations

import numpy as np

from scpn_fusion.core._integrated_transport_solver_coupled_numerics import (
    crank_nicolson_step,
    cylindrical_operator,
    normalised_gaussian,
    thermal_energy_j,
    validate_uniform_rho,
)
from scpn_fusion.core._integrated_transport_solver_base import TransportSolverState
from scpn_fusion.core.current_diffusion import CurrentDiffusionSolver
from scpn_fusion.core.integrated_transport_solver_coupled_contracts import (
    CoupledTransportBudget,
    CoupledTransportInputs,
    CoupledTransportStepResult,
    FloatArray,
)

_KEV_J = 1.602176634e-16


class CoupledTransportRuntimeMixin(TransportSolverState):
    """Public prescribed-coefficient coupled-transport runtime surface."""

    _coupled_current_solver: CurrentDiffusionSolver | None
    _coupled_current_geometry: tuple[float, float, float] | None

    def set_coupled_flux_profile(
        self,
        poloidal_flux_wb_per_rad: FloatArray,
        *,
        major_radius_m: float,
        minor_radius_m: float,
        magnetic_field_t: float,
    ) -> None:
        """Initialize the coupled current-diffusion state from a caller profile."""
        rho, _ = validate_uniform_rho(self.rho)
        flux = np.asarray(poloidal_flux_wb_per_rad, dtype=np.float64)
        if flux.shape != rho.shape or not np.all(np.isfinite(flux)):
            raise ValueError("poloidal flux must be finite and match rho")
        current_solver = CurrentDiffusionSolver(
            rho,
            R0=float(major_radius_m),
            a=float(minor_radius_m),
            B0=float(magnetic_field_t),
        )
        current_solver.psi = flux.copy()
        self._coupled_current_solver = current_solver
        self._coupled_current_geometry = (
            float(major_radius_m),
            float(minor_radius_m),
            float(magnetic_field_t),
        )

    def evolve_coupled_transport(
        self,
        inputs: CoupledTransportInputs,
    ) -> CoupledTransportStepResult:
        """Advance the public four-state model-intersection transport path."""
        rho, spacing = validate_uniform_rho(self.rho)
        for name, values in (
            ("Ti", self.Ti),
            ("Te", self.Te),
            ("ne", self.ne),
        ):
            profile = np.asarray(values, dtype=np.float64)
            if profile.shape != rho.shape or not np.all(np.isfinite(profile)):
                raise ValueError(f"{name} must be finite and match rho")

        geometry = (
            inputs.major_radius_m,
            inputs.minor_radius_m,
            inputs.magnetic_field_t,
        )
        if getattr(self, "_coupled_current_geometry", None) != geometry:
            self._coupled_current_solver = CurrentDiffusionSolver(
                rho,
                R0=inputs.major_radius_m,
                a=inputs.minor_radius_m,
                B0=inputs.magnetic_field_t,
            )
            self._coupled_current_geometry = geometry
        current_solver = self._coupled_current_solver
        if current_solver is None:
            raise RuntimeError("coupled current solver was not initialized")

        ti_before = np.asarray(self.Ti, dtype=np.float64).copy()
        te_before = np.asarray(self.Te, dtype=np.float64).copy()
        ne_before = np.asarray(self.ne, dtype=np.float64).copy()
        psi_before = current_solver.psi.copy()

        volume_derivative = 4.0 * np.pi**2 * inputs.major_radius_m * inputs.minor_radius_m**2 * rho
        area_derivative = 2.0 * np.pi * inputs.minor_radius_m**2 * rho
        heat_shape = normalised_gaussian(
            rho,
            center=inputs.heat_center_rho,
            width=inputs.heat_width_rho,
            measure=volume_derivative,
        )
        particle_shape = normalised_gaussian(
            rho,
            center=inputs.particle_center_rho,
            width=inputs.particle_width_rho,
            measure=volume_derivative,
        )
        current_shape = normalised_gaussian(
            rho,
            center=inputs.current_center_rho,
            width=inputs.current_width_rho,
            measure=area_derivative,
        )

        heat_density_w_m3 = inputs.heat_power_w * heat_shape
        ion_heat_density = heat_density_w_m3 * (1.0 - inputs.electron_heat_fraction)
        electron_heat_density = heat_density_w_m3 * inputs.electron_heat_fraction
        density_floor = np.maximum(ne_before, 1.0e-6) * 1.0e19
        ion_heat_source = ion_heat_density / (1.5 * density_floor * _KEV_J)
        electron_heat_source = electron_heat_density / (1.5 * density_floor * _KEV_J)
        particle_source = inputs.particle_rate_s * particle_shape / 1.0e19

        ion_operator = cylindrical_operator(
            rho,
            spacing,
            inputs.ion_heat_diffusivity_m2_s,
            inputs.minor_radius_m,
        )
        electron_operator = cylindrical_operator(
            rho,
            spacing,
            inputs.electron_heat_diffusivity_m2_s,
            inputs.minor_radius_m,
        )
        particle_operator = cylindrical_operator(
            rho,
            spacing,
            inputs.electron_particle_diffusivity_m2_s,
            inputs.minor_radius_m,
        )
        ti_trial, ti_residual = crank_nicolson_step(
            ti_before,
            operator=ion_operator,
            source=ion_heat_source,
            dt_s=inputs.dt_s,
            edge_value=inputs.ion_temperature_edge_kev,
        )
        te_trial, te_residual = crank_nicolson_step(
            te_before,
            operator=electron_operator,
            source=electron_heat_source,
            dt_s=inputs.dt_s,
            edge_value=inputs.electron_temperature_edge_kev,
        )
        ne_after, ne_residual = crank_nicolson_step(
            ne_before,
            operator=particle_operator,
            source=particle_source,
            dt_s=inputs.dt_s,
            edge_value=inputs.electron_density_edge_1e19_m3,
        )

        thermal_before_exchange = thermal_energy_j(
            rho,
            volume_derivative,
            ne_after,
            ti_trial,
            te_trial,
        )
        mean_temperature = 0.5 * (ti_trial + te_trial)
        difference = (ti_trial - te_trial) * np.exp(
            -2.0 * inputs.ion_electron_exchange_rate_s * inputs.dt_s
        )
        ti_after = mean_temperature + 0.5 * difference
        te_after = mean_temperature - 0.5 * difference
        ti_after[-1] = inputs.ion_temperature_edge_kev
        te_after[-1] = inputs.electron_temperature_edge_kev

        ion_exchange_energy = float(
            np.trapz(
                1.5 * ne_after * 1.0e19 * _KEV_J * (ti_after - ti_trial) * volume_derivative,
                rho,
            )
        )
        electron_exchange_energy = float(
            np.trapz(
                1.5 * ne_after * 1.0e19 * _KEV_J * (te_after - te_trial) * volume_derivative,
                rho,
            )
        )

        driven_current_density = inputs.driven_current_a * current_shape
        psi_after = current_solver.step(
            inputs.dt_s,
            te_after,
            ne_after,
            inputs.effective_charge,
            np.zeros_like(rho),
            np.zeros_like(rho),
            driven_current_density,
            resistivity_multiplier=inputs.resistivity_multiplier,
        ).copy()
        current_budget = current_solver.last_step_budget
        if current_budget is None:
            raise RuntimeError("current diffusion step did not produce a budget")

        self.Ti = np.maximum(ti_after, 1.0e-6)
        self.Te = np.maximum(te_after, 1.0e-6)
        self.ne = np.maximum(ne_after, 1.0e-9)

        thermal_before = thermal_energy_j(
            rho,
            volume_derivative,
            ne_before,
            ti_before,
            te_before,
        )
        thermal_after = thermal_energy_j(
            rho,
            volume_derivative,
            self.ne,
            self.Ti,
            self.Te,
        )
        particle_before = float(np.trapz(ne_before * 1.0e19 * volume_derivative, rho))
        particle_after = float(np.trapz(self.ne * 1.0e19 * volume_derivative, rho))
        reconstructed_ion_heat = float(np.trapz(ion_heat_density * volume_derivative, rho))
        reconstructed_electron_heat = float(
            np.trapz(electron_heat_density * volume_derivative, rho)
        )
        reconstructed_particles = float(
            np.trapz(
                particle_source * 1.0e19 * volume_derivative,
                rho,
            )
        )
        reconstructed_current = float(np.trapz(driven_current_density * area_derivative, rho))
        heat_injected = inputs.heat_power_w * inputs.dt_s
        particles_injected = inputs.particle_rate_s * inputs.dt_s
        thermal_boundary_and_diffusion = (
            thermal_after
            - thermal_before
            - heat_injected
            - ion_exchange_energy
            - electron_exchange_energy
        )
        particle_boundary_and_diffusion = particle_after - particle_before - particles_injected

        budget = CoupledTransportBudget(
            thermal_energy_before_j=thermal_before,
            thermal_energy_after_j=thermal_after,
            heat_energy_injected_j=heat_injected,
            ion_exchange_energy_j=ion_exchange_energy,
            electron_exchange_energy_j=electron_exchange_energy,
            thermal_boundary_and_diffusion_j=thermal_boundary_and_diffusion,
            particle_inventory_before=particle_before,
            particle_inventory_after=particle_after,
            particles_injected=particles_injected,
            particle_boundary_and_diffusion=particle_boundary_and_diffusion,
            flux_l2_before=float(np.linalg.norm(psi_before)),
            flux_l2_after=float(np.linalg.norm(psi_after)),
            driven_current_target_a=inputs.driven_current_a,
            driven_current_reconstructed_a=reconstructed_current,
            ion_heat_source_reconstructed_w=reconstructed_ion_heat,
            electron_heat_source_reconstructed_w=reconstructed_electron_heat,
            particle_source_reconstructed_s=reconstructed_particles,
            ion_temperature_linear_residual_linf=ti_residual,
            electron_temperature_linear_residual_linf=te_residual,
            electron_density_linear_residual_linf=ne_residual,
            current_linear_residual_linf=current_budget.linear_residual_linf,
            ion_electron_exchange_closure_j=ion_exchange_energy + electron_exchange_energy,
        )
        residuals = (
            ti_residual,
            te_residual,
            ne_residual,
            current_budget.linear_residual_linf,
        )
        converged = bool(
            all(np.isfinite(value) and value <= 1.0e-8 for value in residuals)
            and abs(budget.ion_electron_exchange_closure_j)
            <= 1.0e-6 * max(abs(thermal_before_exchange), 1.0)
        )
        return CoupledTransportStepResult(
            time_s=inputs.time_s + inputs.dt_s,
            rho=rho.copy(),
            ion_temperature_kev=self.Ti.copy(),
            electron_temperature_kev=self.Te.copy(),
            electron_density_1e19_m3=self.ne.copy(),
            poloidal_flux_wb_per_rad=psi_after,
            budget=budget,
            converged=converged,
        )


__all__ = [
    "CoupledTransportBudget",
    "CoupledTransportInputs",
    "CoupledTransportRuntimeMixin",
    "CoupledTransportStepResult",
]
