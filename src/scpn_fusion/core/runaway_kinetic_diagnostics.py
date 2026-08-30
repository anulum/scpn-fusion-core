# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Runaway Kinetic Diagnostics
"""Conservation and residual diagnostics for full kinetic trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scpn_fusion.core.runaway_kinetic_grid import FloatArray
from scpn_fusion.core.runaway_kinetic_operator import (
    RunawayKineticGeometry,
    RunawayKineticTendencies,
)


@dataclass(frozen=True)
class RunawayKineticBudget:
    """Phase-space-integrated rate from every declared operator component."""

    radial_transport: float
    electric_acceleration: float
    collisional_drag_diffusion: float
    pitch_scattering: float
    cross_diffusion: float
    synchrotron_loss: float
    bremsstrahlung_loss: float
    avalanche_generation: float
    external_source: float
    total: float


def weighted_relative_l2(
    actual: FloatArray,
    expected: FloatArray,
    weight: FloatArray,
    *,
    floor: float = 1.0,
) -> float:
    """Return a finite weighted relative L2 error with an explicit floor."""
    lhs = np.asarray(actual, dtype=np.float64)
    rhs = np.asarray(expected, dtype=np.float64)
    weights = np.asarray(weight, dtype=np.float64)
    if lhs.shape != rhs.shape or lhs.shape != weights.shape:
        raise ValueError("actual, expected and weight must have identical shapes")
    if floor <= 0.0 or not np.isfinite(floor):
        raise ValueError("floor must be finite and positive")
    numerator = np.sqrt(np.sum(weights * (lhs - rhs) ** 2))
    denominator = max(float(np.sqrt(np.sum(weights * rhs**2))), floor)
    return float(numerator / denominator)


def integrated_budget(
    tendencies: RunawayKineticTendencies,
    geometry: RunawayKineticGeometry,
) -> RunawayKineticBudget:
    """Integrate every tendency without hiding canceling contributions."""
    weight = geometry.cell_measure

    def integrate(values: FloatArray) -> float:
        return float(np.sum(values * weight))

    return RunawayKineticBudget(
        radial_transport=integrate(tendencies.radial_transport),
        electric_acceleration=integrate(tendencies.electric_acceleration),
        collisional_drag_diffusion=integrate(tendencies.collisional_drag_diffusion),
        pitch_scattering=integrate(tendencies.pitch_scattering),
        cross_diffusion=integrate(tendencies.cross_diffusion),
        synchrotron_loss=integrate(tendencies.synchrotron_loss),
        bremsstrahlung_loss=integrate(tendencies.bremsstrahlung_loss),
        avalanche_generation=integrate(tendencies.avalanche_generation),
        external_source=integrate(tendencies.external_source),
        total=integrate(tendencies.total),
    )


def interval_residual(
    previous: FloatArray,
    current: FloatArray,
    tendency_at_current: FloatArray,
    geometry: RunawayKineticGeometry,
    dt_s: float,
) -> float:
    """Check a backward-Euler/implicit interval against its full tendency."""
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    finite_difference = (np.asarray(current) - np.asarray(previous)) / dt_s
    return weighted_relative_l2(
        np.asarray(tendency_at_current),
        np.asarray(finite_difference),
        geometry.cell_measure,
    )


__all__ = [
    "RunawayKineticBudget",
    "integrated_budget",
    "interval_residual",
    "weighted_relative_l2",
]
