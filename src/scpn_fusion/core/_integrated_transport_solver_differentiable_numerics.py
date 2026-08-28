# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Differentiable Coupled Transport Numerics
"""JAX-native four-state numerics shared by transport autodiff surfaces."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from scpn_fusion.core.current_diffusion import MU_0
from scpn_fusion.core.integrated_transport_solver_coupled_contracts import (
    CoupledTransportInputs,
)

_KEV_J = 1.602176634e-16
JaxState = tuple[jax.Array, jax.Array, jax.Array, jax.Array]


def _operator(rho: jax.Array, diffusivity: jax.Array, minor_radius_m: float) -> jax.Array:
    spacing = rho[1] - rho[0]
    scale = diffusivity / minor_radius_m**2
    matrix = jnp.zeros((rho.size, rho.size), dtype=jnp.float64)
    matrix = matrix.at[0, 0].set(-4.0 * scale[0] / spacing**2)
    matrix = matrix.at[0, 1].set(4.0 * scale[0] / spacing**2)
    indices = jnp.arange(1, rho.size - 1)
    radius = rho[indices]
    lower = scale[indices] * (1.0 / spacing**2 - 1.0 / (2.0 * radius * spacing))
    upper = scale[indices] * (1.0 / spacing**2 + 1.0 / (2.0 * radius * spacing))
    matrix = matrix.at[indices, indices - 1].set(lower)
    matrix = matrix.at[indices, indices].set(-2.0 * scale[indices] / spacing**2)
    return matrix.at[indices, indices + 1].set(upper)


def _thomas(
    lower: jax.Array,
    diagonal: jax.Array,
    upper: jax.Array,
    rhs: jax.Array,
) -> jax.Array:
    """Differentiable tridiagonal solve without a host BLAS dependency."""
    size = rhs.shape[0]

    def forward(
        carry: tuple[jax.Array, jax.Array], index: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        upper_previous, rhs_previous = carry
        lower_value = jnp.where(index > 0, lower[index - 1], 0.0)
        pivot = diagonal[index] - lower_value * upper_previous
        pivot = jnp.where(jnp.abs(pivot) < 1.0e-30, 1.0e-30, pivot)
        rhs_value = (rhs[index] - lower_value * rhs_previous) / pivot
        upper_value = jnp.where(index < size - 1, upper[index] / pivot, 0.0)
        return (upper_value, rhs_value), (upper_value, rhs_value)

    initial = (jnp.asarray(0.0, dtype=jnp.float64), jnp.asarray(0.0, dtype=jnp.float64))
    _, stacked = jax.lax.scan(forward, initial, jnp.arange(size))
    upper_values, rhs_values = stacked

    def backward(next_value: jax.Array, index: jax.Array) -> tuple[jax.Array, jax.Array]:
        value = rhs_values[index] - upper_values[index] * next_value
        return value, value

    _, reversed_values = jax.lax.scan(
        backward,
        rhs_values[-1],
        jnp.arange(size - 2, -1, -1),
    )
    return jnp.concatenate((jnp.flip(reversed_values), rhs_values[-1:]))


def _cn_step(
    profile: jax.Array,
    operator: jax.Array,
    source: jax.Array,
    dt_s: float,
    edge_value: float,
) -> jax.Array:
    diagonal = jnp.diag(operator)
    lower = jnp.diag(operator, k=-1)
    upper = jnp.diag(operator, k=1)
    applied = diagonal * profile
    applied = applied.at[1:].add(lower * profile[:-1])
    applied = applied.at[:-1].add(upper * profile[1:])
    lhs_diagonal = 1.0 - 0.5 * dt_s * diagonal
    lhs_lower = -0.5 * dt_s * lower
    lhs_upper = -0.5 * dt_s * upper
    rhs = profile + 0.5 * dt_s * applied + dt_s * source
    lhs_diagonal = lhs_diagonal.at[-1].set(1.0)
    lhs_lower = lhs_lower.at[-1].set(0.0)
    rhs = rhs.at[-1].set(edge_value)
    return _thomas(lhs_lower, lhs_diagonal, lhs_upper, rhs)


def _gaussian(
    rho: jax.Array,
    center: float,
    width: float,
    measure: jax.Array,
) -> jax.Array:
    shape = jnp.exp(-0.5 * ((rho - center) / width) ** 2)
    return shape / jnp.trapezoid(shape * measure, rho)


def _resistivity(te_kev: jax.Array, effective_charge: float, epsilon: jax.Array) -> jax.Array:
    temperature = jnp.maximum(te_kev, 1.0e-3)
    inverse_aspect = jnp.maximum(epsilon, 1.0e-6)
    spitzer = 1.65e-9 * effective_charge * 17.0 / temperature**1.5
    trapped = 1.0 - (1.0 - inverse_aspect) ** 2 / (
        jnp.sqrt(1.0 - inverse_aspect**2) * (1.0 + 1.46 * jnp.sqrt(inverse_aspect))
    )
    trapped = jnp.clip(trapped, 0.0, 1.0)
    correction = (
        1.0 - (1.0 + 0.36 / effective_charge) * trapped + (0.59 / effective_charge) * trapped**2
    )
    neoclassical = spitzer / (1.0 - trapped) * correction
    return jnp.maximum(neoclassical, spitzer)


def _current_step(
    psi: jax.Array,
    te_kev: jax.Array,
    driven_current_density: jax.Array,
    rho: jax.Array,
    inputs: CoupledTransportInputs,
) -> jax.Array:
    epsilon = rho * inputs.minor_radius_m / inputs.major_radius_m
    eta = _resistivity(te_kev, inputs.effective_charge, epsilon) * inputs.resistivity_multiplier
    diffusivity = eta / (MU_0 * inputs.minor_radius_m**2)
    operator = _operator(
        rho,
        diffusivity * inputs.minor_radius_m**2,
        inputs.minor_radius_m,
    )
    source = inputs.major_radius_m * eta * driven_current_density
    return _cn_step(psi, operator, source, inputs.dt_s, cast(float, psi[-1]))


def one_step(
    state: JaxState,
    controls: jax.Array,
    rho: jax.Array,
    inputs: CoupledTransportInputs,
) -> JaxState:
    """Advance one coupled Ti/Te/ne/flux state with traceable controls."""
    ti, te, ne, psi = state
    volume_measure = 4.0 * jnp.pi**2 * inputs.major_radius_m * inputs.minor_radius_m**2 * rho
    area_measure = 2.0 * jnp.pi * inputs.minor_radius_m**2 * rho
    heat_shape = _gaussian(rho, inputs.heat_center_rho, inputs.heat_width_rho, volume_measure)
    particle_shape = _gaussian(
        rho, inputs.particle_center_rho, inputs.particle_width_rho, volume_measure
    )
    current_shape = _gaussian(
        rho, inputs.current_center_rho, inputs.current_width_rho, area_measure
    )
    heat_density = controls[0] * inputs.heat_power_w * heat_shape
    density = jnp.maximum(ne, 1.0e-6) * 1.0e19
    ion_source = heat_density * (1.0 - inputs.electron_heat_fraction) / (1.5 * density * _KEV_J)
    electron_source = heat_density * inputs.electron_heat_fraction / (1.5 * density * _KEV_J)
    particle_source = controls[1] * inputs.particle_rate_s * particle_shape / 1.0e19
    ones = jnp.ones_like(rho)
    ti_trial = _cn_step(
        ti,
        _operator(rho, ones * inputs.ion_heat_diffusivity_m2_s, inputs.minor_radius_m),
        ion_source,
        inputs.dt_s,
        inputs.ion_temperature_edge_kev,
    )
    te_trial = _cn_step(
        te,
        _operator(rho, ones * inputs.electron_heat_diffusivity_m2_s, inputs.minor_radius_m),
        electron_source,
        inputs.dt_s,
        inputs.electron_temperature_edge_kev,
    )
    ne_after = _cn_step(
        ne,
        _operator(rho, ones * inputs.electron_particle_diffusivity_m2_s, inputs.minor_radius_m),
        particle_source,
        inputs.dt_s,
        inputs.electron_density_edge_1e19_m3,
    )
    mean_temperature = 0.5 * (ti_trial + te_trial)
    difference = (ti_trial - te_trial) * jnp.exp(
        -2.0 * inputs.ion_electron_exchange_rate_s * inputs.dt_s
    )
    ti_after = (mean_temperature + 0.5 * difference).at[-1].set(inputs.ion_temperature_edge_kev)
    te_after = (
        (mean_temperature - 0.5 * difference).at[-1].set(inputs.electron_temperature_edge_kev)
    )
    driven_current = controls[2] * inputs.driven_current_a * current_shape
    psi_after = _current_step(psi, te_after, driven_current, rho, inputs)
    return (
        jnp.maximum(ti_after, 1.0e-6),
        jnp.maximum(te_after, 1.0e-6),
        jnp.maximum(ne_after, 1.0e-9),
        psi_after,
    )


def objective(
    states: JaxState,
    targets: JaxState,
    weights: tuple[float, float, float, float],
) -> jax.Array:
    """Return the weighted dimensionless four-state profile objective."""
    total = jnp.asarray(0.0, dtype=jnp.float64)
    for observed, target, weight in zip(states, targets, weights, strict=True):
        scale = jnp.maximum(jnp.linalg.norm(target) / jnp.sqrt(target.size), 1.0e-12)
        total = total + weight * jnp.mean(((observed - target) / scale) ** 2)
    return total / sum(weights)


__all__ = ["JaxState", "objective", "one_step"]
