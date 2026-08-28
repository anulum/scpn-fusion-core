# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Coupled Transport Numerics
"""Conservative numerical helpers for coupled transport stepping."""

from __future__ import annotations

import numpy as np

from scpn_fusion.core.integrated_transport_solver_coupled_contracts import FloatArray

_KEV_J = 1.602176634e-16


def validate_uniform_rho(rho: FloatArray) -> tuple[FloatArray, float]:
    """Validate the model-intersection radial grid and return its spacing."""
    grid = np.asarray(rho, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 4:
        raise ValueError("rho must be a one-dimensional grid with at least four points")
    if not np.all(np.isfinite(grid)) or not np.all(np.diff(grid) > 0.0):
        raise ValueError("rho must be finite and strictly increasing")
    spacing = np.diff(grid)
    if not np.allclose(spacing, spacing[0], rtol=0.0, atol=1e-12):
        raise ValueError("coupled transport requires a uniform rho grid")
    if not np.isclose(grid[0], 0.0) or not np.isclose(grid[-1], 1.0):
        raise ValueError("coupled transport rho must span [0, 1]")
    return grid, float(spacing[0])


def cylindrical_operator(
    rho: FloatArray,
    spacing: float,
    diffusivity: float,
    minor_radius_m: float,
) -> FloatArray:
    """Return the normalized-radius cylindrical diffusion operator."""
    n_points = rho.size
    operator = np.zeros((n_points, n_points), dtype=np.float64)
    scale = diffusivity / (minor_radius_m * minor_radius_m)
    operator[0, 0] = -4.0 * scale / spacing**2
    operator[0, 1] = 4.0 * scale / spacing**2
    for index in range(1, n_points - 1):
        radius = float(rho[index])
        lower = scale * (1.0 / spacing**2 - 1.0 / (2.0 * radius * spacing))
        upper = scale * (1.0 / spacing**2 + 1.0 / (2.0 * radius * spacing))
        operator[index, index - 1] = lower
        operator[index, index] = -2.0 * scale / spacing**2
        operator[index, index + 1] = upper
    return operator


def crank_nicolson_step(
    profile: FloatArray,
    *,
    operator: FloatArray,
    source: FloatArray,
    dt_s: float,
    edge_value: float,
) -> tuple[FloatArray, float]:
    """Advance one linear diffusion/source equation with a Dirichlet edge."""
    identity = np.eye(profile.size, dtype=np.float64)
    lhs = identity - 0.5 * dt_s * operator
    rhs_matrix = identity + 0.5 * dt_s * operator
    rhs = rhs_matrix @ profile + dt_s * source
    lhs[-1, :] = 0.0
    lhs[-1, -1] = 1.0
    rhs[-1] = edge_value
    result = np.asarray(np.linalg.solve(lhs, rhs), dtype=np.float64)
    residual = float(np.max(np.abs(lhs @ result - rhs)))
    return result, residual


def normalised_gaussian(
    rho: FloatArray,
    *,
    center: float,
    width: float,
    measure: FloatArray,
) -> FloatArray:
    """Normalize a Gaussian profile against a caller-supplied measure."""
    shape = np.exp(-0.5 * ((rho - center) / width) ** 2)
    integral = float(np.trapz(shape * measure, rho))
    if not np.isfinite(integral) or integral <= 0.0:
        raise ValueError("source profile has a non-positive normalization integral")
    return np.asarray(shape / integral, dtype=np.float64)


def thermal_energy_j(
    rho: FloatArray,
    volume_derivative: FloatArray,
    density_1e19_m3: FloatArray,
    ion_temperature_kev: FloatArray,
    electron_temperature_kev: FloatArray,
) -> float:
    """Integrate two-temperature thermal energy over the circular torus."""
    energy_density = (
        1.5 * density_1e19_m3 * 1.0e19 * _KEV_J * (ion_temperature_kev + electron_temperature_kev)
    )
    return float(np.trapz(energy_density * volume_derivative, rho))


__all__ = [
    "crank_nicolson_step",
    "cylindrical_operator",
    "normalised_gaussian",
    "thermal_energy_j",
    "validate_uniform_rho",
]
