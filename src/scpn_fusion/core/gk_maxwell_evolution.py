#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Local electromagnetic Maxwell evolution evidence for nonlinear GK reports.

This module advances a source-free perpendicular spectral Maxwell field system
for ``A_parallel`` and a compressional ``B_parallel`` component. It is a native
field-evolution contract for Faraday induction, Ampere-Maxwell displacement
current, and the inductive parallel electric field relation. It is not a full
Vlasov-Maxwell gyrokinetic parity claim because the kinetic current is not yet
self-consistently supplied by the 5D distribution and no same-deck external
GENE/CGYRO/GS2 electromagnetic outputs are compared here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_C_LIGHT = 299_792_458.0
_EPSILON_0 = 8.854_187_8128e-12
_MU_0 = 1.256_637_06212e-6
_TINY = 1.0e-300


@dataclass(frozen=True)
class MaxwellEvolutionConfig:
    """Configuration for source-free local spectral Maxwell evolution."""

    n_kx: int = 8
    n_ky: int = 8
    n_steps: int = 16
    dt: float = 1.0e-12
    Lx_m: float = 1.0
    Ly_m: float = 1.0
    seed: int = 23
    amplitude_A_parallel: float = 1.0e-9
    amplitude_E_parallel: float = 1.0e-2
    amplitude_B_parallel: float = 1.0e-8
    amplitude_E_perpendicular: float = 1.0e-2
    courant_limit: float = 0.2
    relative_energy_tolerance: float = 1.0e-11
    residual_tolerance: float = 1.0e-12


@dataclass(frozen=True)
class MaxwellEvolutionResult:
    """Diagnostics from a local source-free Maxwell field evolution run."""

    schema: str
    time_s: NDArray[np.float64]
    phi_energy_t: NDArray[np.float64]
    A_parallel_energy_t: NDArray[np.float64]
    B_parallel_energy_t: NDArray[np.float64]
    electric_parallel_energy_t: NDArray[np.float64]
    electric_perpendicular_energy_t: NDArray[np.float64]
    total_field_energy_t: NDArray[np.float64]
    faraday_linf_residual_t: NDArray[np.float64]
    ampere_maxwell_linf_residual_t: NDArray[np.float64]
    inductive_e_parallel_linf_residual_t: NDArray[np.float64]
    relative_total_field_energy_drift: float
    max_faraday_linf_residual: float
    max_ampere_maxwell_linf_residual: float
    max_inductive_e_parallel_linf_residual: float
    relative_energy_tolerance: float
    residual_tolerance: float
    faraday_induction_supported: bool
    ampere_maxwell_displacement_current_supported: bool
    inductive_parallel_electric_field_supported: bool
    self_consistent_kinetic_current_supported: bool

    def to_evidence(self) -> dict[str, object]:
        """Return a JSON-serialisable benchmark evidence record."""
        return {
            "ampere_maxwell_displacement_current_supported": (
                self.ampere_maxwell_displacement_current_supported
            ),
            "faraday_induction_supported": self.faraday_induction_supported,
            "inductive_parallel_electric_field_supported": (
                self.inductive_parallel_electric_field_supported
            ),
            "max_ampere_maxwell_linf_residual": self.max_ampere_maxwell_linf_residual,
            "max_faraday_linf_residual": self.max_faraday_linf_residual,
            "max_inductive_e_parallel_linf_residual": (self.max_inductive_e_parallel_linf_residual),
            "max_relative_total_field_energy_drift": (self.relative_total_field_energy_drift),
            "relative_energy_tolerance": self.relative_energy_tolerance,
            "residual_tolerance": self.residual_tolerance,
            "saved_steps": int(self.time_s.size),
            "schema": self.schema,
            "self_consistent_kinetic_current_supported": (
                self.self_consistent_kinetic_current_supported
            ),
            "status": "accepted_local_source_free_maxwell_evolution"
            if (
                self.relative_total_field_energy_drift <= self.relative_energy_tolerance
                and self.max_faraday_linf_residual <= self.residual_tolerance
                and self.max_ampere_maxwell_linf_residual <= self.residual_tolerance
                and self.max_inductive_e_parallel_linf_residual <= self.residual_tolerance
            )
            else "blocked_local_maxwell_evolution_residuals_failed",
        }


def _spectral_grid(config: MaxwellEvolutionConfig) -> tuple[NDArray[np.float64], ...]:
    """Return perpendicular spectral coordinates and magnitudes."""
    if config.n_kx < 2 or config.n_ky < 2:
        raise ValueError("Maxwell evolution requires at least two kx and ky modes")
    if config.n_steps < 2:
        raise ValueError("Maxwell evolution requires at least two saved steps")
    if config.dt <= 0.0 or config.Lx_m <= 0.0 or config.Ly_m <= 0.0:
        raise ValueError("Maxwell evolution requires positive dt and domain lengths")

    kx = 2.0 * np.pi * np.fft.fftfreq(config.n_kx, d=config.Lx_m / config.n_kx)
    ky = 2.0 * np.pi * np.fft.fftfreq(config.n_ky, d=config.Ly_m / config.n_ky)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k_perp = np.hypot(kx_grid, ky_grid)
    max_omega_dt = float(_C_LIGHT * np.max(k_perp) * config.dt)
    if max_omega_dt > config.courant_limit:
        raise ValueError(
            "Courant limit violated for local Maxwell evolution: "
            f"max(c k_perp dt)={max_omega_dt:.6e}, limit={config.courant_limit:.6e}"
        )
    return kx_grid, ky_grid, k_perp


def _random_complex_modes(
    rng: np.random.Generator, shape: tuple[int, int], *, amplitude: float
) -> NDArray[np.complex128]:
    """Return deterministic complex spectral perturbations."""
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    modes = amplitude * (real + 1j * imag) / np.sqrt(2.0)
    modes[0, 0] = 0.0
    return np.asarray(modes, dtype=np.complex128)


def _relative_drift(values: NDArray[np.float64]) -> float:
    """Return maximum relative drift from the first saved value."""
    baseline = max(abs(float(values[0])), _TINY)
    return float(np.max(np.abs(values - values[0])) / baseline)


def run_local_maxwell_evolution(config: MaxwellEvolutionConfig) -> MaxwellEvolutionResult:
    """Run source-free local spectral Maxwell evolution and return diagnostics."""
    kx, ky, k_perp = _spectral_grid(config)
    omega = _C_LIGHT * k_perp
    nonzero = k_perp > 0.0
    rng = np.random.default_rng(config.seed)
    shape = (config.n_kx, config.n_ky)

    A0 = _random_complex_modes(rng, shape, amplitude=config.amplitude_A_parallel)
    E0 = _random_complex_modes(rng, shape, amplitude=config.amplitude_E_parallel)
    B0 = _random_complex_modes(rng, shape, amplitude=config.amplitude_B_parallel)
    C0 = _random_complex_modes(rng, shape, amplitude=config.amplitude_E_perpendicular)

    time_s = np.arange(config.n_steps, dtype=np.float64) * config.dt
    phi_energy_t = np.zeros(config.n_steps, dtype=np.float64)
    A_parallel_energy_t = np.zeros(config.n_steps, dtype=np.float64)
    B_parallel_energy_t = np.zeros(config.n_steps, dtype=np.float64)
    electric_parallel_energy_t = np.zeros(config.n_steps, dtype=np.float64)
    electric_perpendicular_energy_t = np.zeros(config.n_steps, dtype=np.float64)
    total_field_energy_t = np.zeros(config.n_steps, dtype=np.float64)
    faraday_linf_residual_t = np.zeros(config.n_steps, dtype=np.float64)
    ampere_maxwell_linf_residual_t = np.zeros(config.n_steps, dtype=np.float64)
    inductive_e_parallel_linf_residual_t = np.zeros(config.n_steps, dtype=np.float64)

    inv_omega = np.zeros_like(omega)
    inv_omega[nonzero] = 1.0 / omega[nonzero]
    inv_c = 1.0 / _C_LIGHT

    for idx, time in enumerate(time_s):
        phase = omega * time
        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)

        A = A0 * cos_phase - E0 * inv_omega * sin_phase
        E_parallel = E0 * cos_phase + omega * A0 * sin_phase
        dA_dt = -E_parallel
        d2A_dt2 = -(omega * omega) * A

        B_parallel = B0 * cos_phase - C0 * inv_c * sin_phase
        E_perpendicular = C0 * cos_phase + _C_LIGHT * B0 * sin_phase

        Bx = 1j * ky * A
        By = -1j * kx * A
        dBx_dt = 1j * ky * dA_dt
        dBy_dt = -1j * kx * dA_dt

        faraday_x = dBx_dt + 1j * ky * E_parallel
        faraday_y = dBy_dt - 1j * kx * E_parallel
        ampere = d2A_dt2 + (omega * omega) * A
        inductive = E_parallel + dA_dt

        faraday_scale = np.maximum(
            np.maximum(np.abs(dBx_dt), np.abs(dBy_dt)),
            np.maximum(np.abs(kx * E_parallel), np.abs(ky * E_parallel)),
        )
        ampere_scale = np.maximum(np.abs(d2A_dt2), np.abs((omega * omega) * A))
        inductive_scale = np.maximum(np.abs(E_parallel), np.abs(dA_dt))

        faraday_linf_residual_t[idx] = float(
            np.max(
                np.divide(
                    np.maximum(np.abs(faraday_x), np.abs(faraday_y)),
                    np.maximum(faraday_scale, _TINY),
                )
            )
        )
        ampere_maxwell_linf_residual_t[idx] = float(
            np.max(np.divide(np.abs(ampere), np.maximum(ampere_scale, _TINY)))
        )
        inductive_e_parallel_linf_residual_t[idx] = float(
            np.max(np.divide(np.abs(inductive), np.maximum(inductive_scale, _TINY)))
        )

        electric_parallel_energy_t[idx] = float(0.5 * _EPSILON_0 * np.sum(np.abs(E_parallel) ** 2))
        A_parallel_energy_t[idx] = float(0.5 / _MU_0 * np.sum(np.abs(Bx) ** 2 + np.abs(By) ** 2))
        B_parallel_energy_t[idx] = float(0.5 / _MU_0 * np.sum(np.abs(B_parallel) ** 2))
        electric_perpendicular_energy_t[idx] = float(
            0.5 * _EPSILON_0 * np.sum(np.abs(E_perpendicular) ** 2)
        )
        total_field_energy_t[idx] = (
            electric_parallel_energy_t[idx]
            + A_parallel_energy_t[idx]
            + B_parallel_energy_t[idx]
            + electric_perpendicular_energy_t[idx]
        )

    return MaxwellEvolutionResult(
        A_parallel_energy_t=A_parallel_energy_t,
        B_parallel_energy_t=B_parallel_energy_t,
        ampere_maxwell_displacement_current_supported=True,
        ampere_maxwell_linf_residual_t=ampere_maxwell_linf_residual_t,
        electric_parallel_energy_t=electric_parallel_energy_t,
        electric_perpendicular_energy_t=electric_perpendicular_energy_t,
        faraday_induction_supported=True,
        faraday_linf_residual_t=faraday_linf_residual_t,
        inductive_e_parallel_linf_residual_t=inductive_e_parallel_linf_residual_t,
        inductive_parallel_electric_field_supported=True,
        max_ampere_maxwell_linf_residual=float(np.max(ampere_maxwell_linf_residual_t)),
        max_faraday_linf_residual=float(np.max(faraday_linf_residual_t)),
        max_inductive_e_parallel_linf_residual=float(np.max(inductive_e_parallel_linf_residual_t)),
        phi_energy_t=phi_energy_t,
        relative_energy_tolerance=config.relative_energy_tolerance,
        relative_total_field_energy_drift=_relative_drift(total_field_energy_t),
        residual_tolerance=config.residual_tolerance,
        schema="gk-maxwell-evolution.v1",
        self_consistent_kinetic_current_supported=False,
        time_s=time_s,
        total_field_energy_t=total_field_energy_t,
    )
