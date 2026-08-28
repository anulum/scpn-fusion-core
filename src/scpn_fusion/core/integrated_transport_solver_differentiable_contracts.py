# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Differentiable Coupled Transport Contracts
"""Public controls, targets, and evidence records for transport autodiff."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scpn_fusion.core.integrated_transport_solver_coupled_contracts import FloatArray


@dataclass(frozen=True)
class CoupledTransportControls:
    """Dimensionless multipliers for heat, particles, and driven current."""

    heat_power_scale: float = 1.0
    particle_rate_scale: float = 1.0
    driven_current_scale: float = 1.0

    def __post_init__(self) -> None:
        """Require finite positive controls."""
        for name, value in vars(self).items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")

    def as_array(self) -> FloatArray:
        """Return controls in the stable public order."""
        return np.asarray(
            [self.heat_power_scale, self.particle_rate_scale, self.driven_current_scale],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, values: FloatArray) -> CoupledTransportControls:
        """Construct controls from the stable public order."""
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (3,):
            raise ValueError("control array must have shape (3,)")
        return cls(*map(float, array))


@dataclass(frozen=True)
class CoupledTransportTarget:
    """Four final-state profiles and non-negative objective weights."""

    ion_temperature_kev: FloatArray
    electron_temperature_kev: FloatArray
    electron_density_1e19_m3: FloatArray
    poloidal_flux_wb_per_rad: FloatArray
    state_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

    def validated(self, radial_points: int) -> CoupledTransportTarget:
        """Return a defensive, validated target copy."""
        arrays: list[FloatArray] = []
        for name in (
            "ion_temperature_kev",
            "electron_temperature_kev",
            "electron_density_1e19_m3",
            "poloidal_flux_wb_per_rad",
        ):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != (radial_points,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite and match rho")
            arrays.append(values.copy())
        if len(self.state_weights) != 4:
            raise ValueError("state_weights must contain four finite non-negative values")
        weights = (
            float(self.state_weights[0]),
            float(self.state_weights[1]),
            float(self.state_weights[2]),
            float(self.state_weights[3]),
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in weights):
            raise ValueError("state_weights must contain four finite non-negative values")
        if not any(value > 0.0 for value in weights):
            raise ValueError("at least one state weight must be positive")
        return CoupledTransportTarget(
            ion_temperature_kev=arrays[0],
            electron_temperature_kev=arrays[1],
            electron_density_1e19_m3=arrays[2],
            poloidal_flux_wb_per_rad=arrays[3],
            state_weights=weights,
        )


@dataclass(frozen=True)
class DifferentiableCoupledTransportResult:
    """One objective/gradient evaluation and its final differentiable state."""

    controls: CoupledTransportControls
    objective: float
    gradient: FloatArray
    ion_temperature_kev: FloatArray
    electron_temperature_kev: FloatArray
    electron_density_1e19_m3: FloatArray
    poloidal_flux_wb_per_rad: FloatArray


@dataclass(frozen=True)
class CoupledTransportOptimisationResult:
    """Deterministic bounded Adam trajectory for three source controls."""

    initial_controls: CoupledTransportControls
    final_controls: CoupledTransportControls
    initial_objective: float
    final_objective: float
    objective_history: FloatArray
    control_history: FloatArray
    final_gradient: FloatArray
    iterations: int


__all__ = [
    "CoupledTransportControls",
    "CoupledTransportOptimisationResult",
    "CoupledTransportTarget",
    "DifferentiableCoupledTransportResult",
]
