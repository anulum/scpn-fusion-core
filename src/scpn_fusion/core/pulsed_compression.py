# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — MIF/FRC Pulsed Compression
"""Pulsed MIF/FRC compression dynamics over an accepted FRC equilibrium.

This module implements the explicit work-lane contract for FUS-C.6:

* external coil current maps to a uniform compression field,
* radial motion follows the pressure-imbalance force,
* temperature and density evolve through adiabatic compression,
* energy accounting is carried with an explicit residual, and
* flux evolution is wired through the existing Ono non-adiabatic carrier.

It does not fabricate the missing Slough 2011 Fig. 5 comparison. Reports that
need that public digitised trajectory must carry a blocked evidence row until
the reference trajectory and compression-work sidecar exist.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from .current_diffusion import solve_flux_evolution_nonadiabatic
from .frc_rigid_rotor import ELEMENTARY_CHARGE_C, FRCEquilibriumState, MU_0

FloatArray: TypeAlias = NDArray[np.float64]
ProfileCallable: TypeAlias = Callable[[float, FloatArray], FloatArray | float]
ScalarCallable: TypeAlias = Callable[[float], float]


@dataclass(frozen=True)
class CoilGeometry:
    """Uniform-solenoid approximation for the pulsed compression coil."""

    N_turns: int
    L_coil_m: float
    R_coil_m: float
    L_inductance_H: float
    R_resistance_ohm: float
    bank_voltage_max_V: float


@dataclass(frozen=True)
class PulsedCompressionConfig:
    """Configuration for a supplied-current FRC pulsed-compression run."""

    equilibrium: FRCEquilibriumState
    coil: CoilGeometry
    coil_current_t: ScalarCallable
    plasma_mass_kg: float
    ion_temperature_eV: float
    electron_temperature_eV: float
    plasma_length_m: float = 1.0
    gamma: float = 5.0 / 3.0
    radial_loss_time_s: float | None = None
    tau_psi_s: float = np.inf
    E_theta_t: ProfileCallable | None = None
    J_theta_t: ProfileCallable | None = None
    Z_eff: float = 1.0
    ln_lambda: float = 17.0
    min_radius_m: float = 1.0e-4


@dataclass(frozen=True)
class PulsedCompressionState:
    """State of the supplied-current pulsed-compression trajectory."""

    t_s: float
    R_s_m: float
    dR_s_dt_m_s: float
    T_i_eV: float
    T_e_eV: float
    density_m3: float
    beta: float
    B_ext_T: float
    internal_pressure_Pa: float
    external_magnetic_pressure_Pa: float
    thermal_energy_J: float
    compression_work_J: float
    radiated_loss_J: float
    energy_balance_residual: float
    flux_psi: FloatArray
    flux_psi_checksum: float
    flux_coupling_status: str


def coil_field_t(coil: CoilGeometry, coil_current_a: float) -> float:
    """Return ``B_ext = mu0*N*I/L`` for the uniform-solenoid approximation."""

    checked = _validate_coil(coil)
    current = _require_finite("coil_current_a", coil_current_a)
    return float(MU_0 * checked.N_turns * current / checked.L_coil_m)


def magnetic_pressure_pa(B_ext_T: float) -> float:
    """Return external magnetic pressure in pascal."""

    field = _require_finite("B_ext_T", B_ext_T)
    return float(field * field / (2.0 * MU_0))


def plasma_volume_m3(R_s_m: float, plasma_length_m: float) -> float:
    """Return cylindrical FRC volume for the supplied separatrix radius."""

    radius = _require_positive("R_s_m", R_s_m)
    length = _require_positive("plasma_length_m", plasma_length_m)
    return float(np.pi * radius * radius * length)


def thermal_pressure_pa(density_m3: float, T_i_eV: float, T_e_eV: float) -> float:
    """Return ion-plus-electron scalar pressure in pascal."""

    density = _require_positive("density_m3", density_m3)
    ion_temperature = _require_positive("T_i_eV", T_i_eV)
    electron_temperature = _require_positive("T_e_eV", T_e_eV)
    return float(density * (ion_temperature + electron_temperature) * ELEMENTARY_CHARGE_C)


def spitzer_resistivity_ohm_m(
    T_e_eV: FloatArray | float,
    *,
    Z_eff: float = 1.0,
    ln_lambda: float = 17.0,
) -> FloatArray:
    """Return NRL-formulary Spitzer resistivity in ohm metre."""

    temperature = np.asarray(T_e_eV, dtype=np.float64)
    if not np.all(np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("T_e_eV must contain positive finite values")
    z_eff = _require_positive("Z_eff", Z_eff)
    coulomb_log = _require_positive("ln_lambda", ln_lambda)
    return cast(FloatArray, 1.65e-9 * z_eff * coulomb_log / np.power(temperature, 1.5))


def adiabatic_temperature_update_eV(
    temperature_eV: float,
    old_volume_m3: float,
    new_volume_m3: float,
    gamma: float = 5.0 / 3.0,
) -> float:
    """Return temperature after ideal adiabatic compression."""

    temperature = _require_positive("temperature_eV", temperature_eV)
    old_volume = _require_positive("old_volume_m3", old_volume_m3)
    new_volume = _require_positive("new_volume_m3", new_volume_m3)
    gamma_value = _require_positive("gamma", gamma)
    if gamma_value <= 1.0:
        raise ValueError("gamma must be greater than one")
    return float(temperature * (old_volume / new_volume) ** (gamma_value - 1.0))


def initial_pulsed_compression_state(config: PulsedCompressionConfig) -> PulsedCompressionState:
    """Construct the initial compression state from the accepted FRC equilibrium."""

    cfg = _validate_config(config)
    radius = float(cfg.equilibrium.target_separatrix_radius_m)
    density = float(cfg.equilibrium.density_peak_m3)
    field = coil_field_t(cfg.coil, cfg.coil_current_t(0.0))
    volume = plasma_volume_m3(radius, cfg.plasma_length_m)
    pressure = thermal_pressure_pa(density, cfg.ion_temperature_eV, cfg.electron_temperature_eV)
    thermal_energy = _thermal_energy_j(
        density, volume, cfg.ion_temperature_eV, cfg.electron_temperature_eV
    )
    return PulsedCompressionState(
        t_s=0.0,
        R_s_m=radius,
        dR_s_dt_m_s=0.0,
        T_i_eV=float(cfg.ion_temperature_eV),
        T_e_eV=float(cfg.electron_temperature_eV),
        density_m3=density,
        beta=_beta(pressure, field),
        B_ext_T=field,
        internal_pressure_Pa=pressure,
        external_magnetic_pressure_Pa=magnetic_pressure_pa(field),
        thermal_energy_J=thermal_energy,
        compression_work_J=0.0,
        radiated_loss_J=0.0,
        energy_balance_residual=0.0,
        flux_psi=cfg.equilibrium.psi.copy(),
        flux_psi_checksum=float(np.sum(cfg.equilibrium.psi, dtype=np.float64)),
        flux_coupling_status="initialised_from_frc_equilibrium",
    )


def step_pulsed_compression(
    state: PulsedCompressionState,
    config: PulsedCompressionConfig,
    dt_s: float,
) -> PulsedCompressionState:
    """Advance one pressure-driven pulsed-compression step."""

    cfg = _validate_config(config)
    dt = _require_positive("dt_s", dt_s)
    _validate_state(state, cfg)

    old_volume = plasma_volume_m3(state.R_s_m, cfg.plasma_length_m)
    old_pressure = thermal_pressure_pa(state.density_m3, state.T_i_eV, state.T_e_eV)
    field = coil_field_t(cfg.coil, cfg.coil_current_t(state.t_s))
    external_pressure = magnetic_pressure_pa(field)
    force_area = 2.0 * np.pi * state.R_s_m * cfg.plasma_length_m
    radial_acceleration = (old_pressure - external_pressure) * force_area / cfg.plasma_mass_kg
    dR_s_dt = state.dR_s_dt_m_s + radial_acceleration * dt
    radius = max(cfg.min_radius_m, state.R_s_m + dR_s_dt * dt)

    new_volume = plasma_volume_m3(radius, cfg.plasma_length_m)
    density = state.density_m3 * old_volume / new_volume
    T_i_adiabatic = adiabatic_temperature_update_eV(state.T_i_eV, old_volume, new_volume, cfg.gamma)
    T_e_adiabatic = adiabatic_temperature_update_eV(state.T_e_eV, old_volume, new_volume, cfg.gamma)
    thermal_adiabatic = _thermal_energy_j(density, new_volume, T_i_adiabatic, T_e_adiabatic)
    loss_factor = 1.0
    if cfg.radial_loss_time_s is not None:
        loss_factor = float(
            np.exp(-dt / _require_positive("radial_loss_time_s", cfg.radial_loss_time_s))
        )
    T_i = T_i_adiabatic * loss_factor
    T_e = T_e_adiabatic * loss_factor
    pressure = thermal_pressure_pa(density, T_i, T_e)
    thermal_energy = _thermal_energy_j(density, new_volume, T_i, T_e)
    compression_work = state.compression_work_J + (thermal_adiabatic - state.thermal_energy_J)
    radiated_loss = state.radiated_loss_J + (thermal_adiabatic - thermal_energy)
    balance_residual = (
        thermal_energy
        - state.thermal_energy_J
        - (compression_work - state.compression_work_J)
        + (radiated_loss - state.radiated_loss_J)
    )
    balance_scale = max(
        abs(thermal_energy),
        abs(state.thermal_energy_J),
        abs(compression_work),
        1.0e-30,
    )

    rho = cfg.equilibrium.rho
    e_theta = cfg.E_theta_t if cfg.E_theta_t is not None else _zero_profile
    j_theta = cfg.J_theta_t if cfg.J_theta_t is not None else _zero_profile
    flux = solve_flux_evolution_nonadiabatic(
        rho,
        state.flux_psi,
        tau_psi_fn=lambda _time, _rho: cfg.tau_psi_s,
        R_null_t=lambda _time: radius,
        E_theta_t=e_theta,
        eta_spitzer_fn=lambda grid: spitzer_resistivity_ohm_m(
            np.full_like(grid, T_e),
            Z_eff=cfg.Z_eff,
            ln_lambda=cfg.ln_lambda,
        ),
        J_theta_t=j_theta,
        dt=dt,
        n_steps=1,
    )
    psi = flux.psi[-1].copy()

    return PulsedCompressionState(
        t_s=state.t_s + dt,
        R_s_m=radius,
        dR_s_dt_m_s=float(dR_s_dt),
        T_i_eV=T_i,
        T_e_eV=T_e,
        density_m3=density,
        beta=_beta(pressure, field),
        B_ext_T=field,
        internal_pressure_Pa=pressure,
        external_magnetic_pressure_Pa=external_pressure,
        thermal_energy_J=thermal_energy,
        compression_work_J=compression_work,
        radiated_loss_J=radiated_loss,
        energy_balance_residual=float(balance_residual / balance_scale),
        flux_psi=psi,
        flux_psi_checksum=float(np.sum(psi, dtype=np.float64)),
        flux_coupling_status="ono_nonadiabatic_flux_carrier",
    )


def run_pulsed_compression(
    initial: PulsedCompressionState,
    config: PulsedCompressionConfig,
    dt_s: float,
    n_steps: int,
) -> tuple[PulsedCompressionState, ...]:
    """Run a supplied-current pulsed-compression trajectory."""

    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    states = [initial]
    state = initial
    for _ in range(n_steps):
        state = step_pulsed_compression(state, config, dt_s)
        states.append(state)
    return tuple(states)


def slough_fig5_acceptance_status() -> dict[str, str]:
    """Return the current fail-closed status for the external Slough comparison."""

    return {
        "case": "slough_2011_fig5",
        "status": "blocked_missing_public_digitised_reference",
        "required_artifact": "digitised radius/temperature/field trajectory with provenance and checksum",
    }


def _validate_coil(coil: CoilGeometry) -> CoilGeometry:
    if isinstance(coil.N_turns, bool) or coil.N_turns <= 0:
        raise ValueError("coil.N_turns must be a positive integer")
    _require_positive("coil.L_coil_m", coil.L_coil_m)
    _require_positive("coil.R_coil_m", coil.R_coil_m)
    _require_positive("coil.L_inductance_H", coil.L_inductance_H)
    _require_positive("coil.R_resistance_ohm", coil.R_resistance_ohm)
    _require_positive("coil.bank_voltage_max_V", coil.bank_voltage_max_V)
    return coil


def _validate_config(config: PulsedCompressionConfig) -> PulsedCompressionConfig:
    _validate_coil(config.coil)
    _require_positive("plasma_mass_kg", config.plasma_mass_kg)
    _require_positive("ion_temperature_eV", config.ion_temperature_eV)
    _require_positive("electron_temperature_eV", config.electron_temperature_eV)
    _require_positive("plasma_length_m", config.plasma_length_m)
    if config.gamma <= 1.0:
        raise ValueError("gamma must be greater than one")
    _require_positive("gamma", config.gamma)
    if config.radial_loss_time_s is not None:
        _require_positive("radial_loss_time_s", config.radial_loss_time_s)
    if config.tau_psi_s != np.inf:
        _require_positive("tau_psi_s", config.tau_psi_s)
    _require_positive("Z_eff", config.Z_eff)
    _require_positive("ln_lambda", config.ln_lambda)
    _require_positive("min_radius_m", config.min_radius_m)
    return config


def _validate_state(state: PulsedCompressionState, config: PulsedCompressionConfig) -> None:
    _require_finite("state.t_s", state.t_s)
    _require_positive("state.R_s_m", state.R_s_m)
    _require_finite("state.dR_s_dt_m_s", state.dR_s_dt_m_s)
    _require_positive("state.T_i_eV", state.T_i_eV)
    _require_positive("state.T_e_eV", state.T_e_eV)
    _require_positive("state.density_m3", state.density_m3)
    psi = np.asarray(state.flux_psi, dtype=np.float64)
    if psi.shape != config.equilibrium.rho.shape or not np.all(np.isfinite(psi)):
        raise ValueError("state.flux_psi must match the equilibrium radial grid")


def _thermal_energy_j(density_m3: float, volume_m3: float, T_i_eV: float, T_e_eV: float) -> float:
    particles = density_m3 * volume_m3
    return float(1.5 * particles * (T_i_eV + T_e_eV) * ELEMENTARY_CHARGE_C)


def _beta(pressure_pa: float, field_t: float) -> float:
    pressure = _require_positive("pressure_pa", pressure_pa)
    field = abs(_require_finite("field_t", field_t))
    if field == 0.0:
        return np.inf
    return float(2.0 * MU_0 * pressure / (field * field))


def _zero_profile(_: float, rho: FloatArray) -> FloatArray:
    return np.zeros_like(rho)


def _require_finite(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_positive(name: str, value: float) -> float:
    result = _require_finite(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result
