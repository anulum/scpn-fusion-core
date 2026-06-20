# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — MIF/FRC Pulsed Hall-MHD
"""Axisymmetric pulsed Hall-MHD flux carrier for the MIF/FRC lane.

The accepted contract evolves the flattened Ono Eq. 8 carrier

``dpsi/dt = -psi/tau_psi + R_null E_theta - eta_spitzer J_theta``

on the accepted FRC radial grid. It is a production flux-evolution surface for
MIF trigger/replay work, not a claim of full 2D two-fluid Hall-MHD or Gkeyll
same-case parity.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid

from .frc_rigid_rotor import FRCEquilibriumState, MU_0

FloatArray: TypeAlias = NDArray[np.float64]
ScalarCallable: TypeAlias = Callable[[float], float]
ProfileCallable: TypeAlias = Callable[[float, FloatArray], FloatArray | float]
IntegratorName: TypeAlias = Literal["implicit_be", "imex_rk2"]
ClosureName: TypeAlias = Literal["axisymmetric_ono_flux", "adiabatic_electron"]


@dataclass(frozen=True)
class HallMHDPulsedConfig:
    """Configuration for the accepted axisymmetric pulsed Hall-MHD carrier."""

    equilibrium: FRCEquilibriumState
    B_ext_t: ScalarCallable
    tau_psi_s: float
    electron_temperature_eV: float
    E_theta_t: ProfileCallable | None = None
    J_theta_t: ProfileCallable | None = None
    Z_eff: float = 1.0
    ln_lambda: float = 17.0
    hall_scale: float = 1.0
    closure: ClosureName = "axisymmetric_ono_flux"
    time_integrator: IntegratorName = "implicit_be"


@dataclass(frozen=True)
class HallMHDPulsedState:
    """State of the axisymmetric pulsed Hall-MHD flux carrier."""

    t_s: float
    psi: FloatArray
    B_z: FloatArray
    E_theta: FloatArray
    J_theta: FloatArray
    R_null_m: float
    energy_proxy_J_m: float
    hall_drive_l2: float
    resistive_sink_l2: float
    damping_sink_l2: float
    source_residual_linf: float
    closure_status: str
    external_parity_status: str


def spitzer_resistivity_ohm_m(
    temperature_eV: FloatArray | float,
    *,
    Z_eff: float = 1.0,
    ln_lambda: float = 17.0,
) -> FloatArray:
    """Return NRL-style Spitzer resistivity ``1.65e-9 Z lnLambda / T_e^1.5``."""
    temperature = np.asarray(temperature_eV, dtype=np.float64)
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("temperature_eV must be positive and finite")
    z_eff = _require_positive("Z_eff", Z_eff)
    coulomb_log = _require_positive("ln_lambda", ln_lambda)
    return np.asarray(1.65e-9 * z_eff * coulomb_log / np.power(temperature, 1.5), dtype=np.float64)


def faraday_e_theta_from_b_ramp(
    rho_m: FloatArray,
    B_ext_t: ScalarCallable,
    t_s: float,
    *,
    derivative_dt_s: float = 1.0e-9,
) -> FloatArray:
    """Return circular-loop Faraday drive ``E_theta = -r/2 dB_ext/dt``."""
    rho = _validate_grid(rho_m)
    dt = _require_positive("derivative_dt_s", derivative_dt_s)
    time = _require_non_negative("t_s", t_s)
    if time >= dt:
        dB_dt = (B_ext_t(time + dt) - B_ext_t(time - dt)) / (2.0 * dt)
    else:
        dB_dt = (B_ext_t(time + dt) - B_ext_t(time)) / dt
    dB_dt = _require_finite("dB_ext_dt", dB_dt)
    return np.asarray(-0.5 * rho * dB_dt, dtype=np.float64)


def axial_field_from_flux(rho_m: FloatArray, psi: FloatArray) -> FloatArray:
    """Return axisymmetric ``B_z = (1/r) dpsi/dr`` with finite-axis handling."""
    rho = _validate_grid(rho_m)
    flux = _validate_profile("psi", psi, rho.size)
    dpsi_dr = np.gradient(flux, rho, edge_order=2)
    field = np.empty_like(flux)
    field[1:] = dpsi_dr[1:] / rho[1:]
    field[0] = field[1]
    return np.asarray(field, dtype=np.float64)


def initial_hall_mhd_pulsed_state(config: HallMHDPulsedConfig) -> HallMHDPulsedState:
    """Construct the initial Hall-MHD carrier state from an FRC equilibrium."""
    cfg = _validate_config(config)
    rho = cfg.equilibrium.rho
    psi = np.asarray(cfg.equilibrium.psi, dtype=np.float64).copy()
    e_theta = _resolve_e_theta(cfg, 0.0)
    j_theta = _resolve_j_theta(cfg, 0.0)
    return HallMHDPulsedState(
        t_s=0.0,
        psi=psi,
        B_z=axial_field_from_flux(rho, psi),
        E_theta=e_theta,
        J_theta=j_theta,
        R_null_m=float(cfg.equilibrium.R_null),
        energy_proxy_J_m=_magnetic_energy_proxy(rho, psi),
        hall_drive_l2=_l2(rho, cfg.hall_scale * cfg.equilibrium.R_null * e_theta),
        resistive_sink_l2=_l2(rho, _eta_profile(cfg, rho) * j_theta),
        damping_sink_l2=_l2(rho, psi / cfg.tau_psi_s),
        source_residual_linf=0.0,
        closure_status=cfg.closure,
        external_parity_status=gkeyll_small_hall_acceptance_status()["status"],
    )


def step_hall_mhd_pulsed(
    state: HallMHDPulsedState,
    config: HallMHDPulsedConfig,
    dt_s: float,
) -> HallMHDPulsedState:
    """Advance one accepted axisymmetric Ono Eq. 8 carrier step."""
    cfg = _validate_config(config)
    dt = _require_positive("dt_s", dt_s)
    _validate_state(state, cfg)
    if cfg.time_integrator == "implicit_be":
        next_psi, source = _implicit_be_step(state.psi, state.t_s, cfg, dt)
    elif cfg.time_integrator == "imex_rk2":
        next_psi, source = _imex_rk2_step(state.psi, state.t_s, cfg, dt)
    else:
        raise ValueError("time_integrator must be 'implicit_be' or 'imex_rk2'")
    rho = cfg.equilibrium.rho
    e_theta = _resolve_e_theta(cfg, state.t_s + dt)
    j_theta = _resolve_j_theta(cfg, state.t_s + dt)
    residual = (next_psi - state.psi) / dt - source + next_psi / cfg.tau_psi_s
    scale = max(float(np.max(np.abs(source))), float(np.max(np.abs(next_psi / cfg.tau_psi_s))), 1.0)
    return HallMHDPulsedState(
        t_s=state.t_s + dt,
        psi=next_psi,
        B_z=axial_field_from_flux(rho, next_psi),
        E_theta=e_theta,
        J_theta=j_theta,
        R_null_m=float(cfg.equilibrium.R_null),
        energy_proxy_J_m=_magnetic_energy_proxy(rho, next_psi),
        hall_drive_l2=_l2(rho, cfg.hall_scale * cfg.equilibrium.R_null * e_theta),
        resistive_sink_l2=_l2(rho, _eta_profile(cfg, rho) * j_theta),
        damping_sink_l2=_l2(rho, next_psi / cfg.tau_psi_s),
        source_residual_linf=float(np.max(np.abs(residual)) / scale),
        closure_status=cfg.closure,
        external_parity_status=gkeyll_small_hall_acceptance_status()["status"],
    )


def run_hall_mhd_pulsed(
    initial: HallMHDPulsedState,
    config: HallMHDPulsedConfig,
    dt_s: float,
    n_steps: int,
) -> tuple[HallMHDPulsedState, ...]:
    """Run a supplied-current Hall-MHD carrier trajectory."""
    if isinstance(n_steps, bool) or n_steps <= 0:
        raise ValueError("n_steps must be positive")
    states = [initial]
    state = initial
    for _ in range(n_steps):
        state = step_hall_mhd_pulsed(state, config, dt_s)
        states.append(state)
    return tuple(states)


def gkeyll_small_hall_acceptance_status() -> dict[str, str]:
    """Return the current fail-closed status for 2D external-code parity."""
    return {
        "case": "gkeyll_axisymmetric_small_hall",
        "status": "blocked_missing_public_same_case_reference",
        "required_artifact": "version-pinned Gkeyll/BOUT++ same-case output with provenance and checksum",
    }


def ono_fig4_acceptance_status() -> dict[str, str]:
    """Return the current fail-closed status for Ono figure reproduction."""
    return {
        "case": "ono_1997_fig4_flux_decay",
        "status": "blocked_missing_public_digitised_reference",
        "required_artifact": "digitised Ono flux-decay curve with deck metadata and checksum",
    }


def _implicit_be_step(
    psi: FloatArray,
    t_s: float,
    cfg: HallMHDPulsedConfig,
    dt_s: float,
) -> tuple[FloatArray, FloatArray]:
    rho = cfg.equilibrium.rho
    source = _ono_source(cfg, t_s + dt_s, rho)
    return np.asarray(
        (psi + dt_s * source) / (1.0 + dt_s / cfg.tau_psi_s), dtype=np.float64
    ), source


def _imex_rk2_step(
    psi: FloatArray,
    t_s: float,
    cfg: HallMHDPulsedConfig,
    dt_s: float,
) -> tuple[FloatArray, FloatArray]:
    rho = cfg.equilibrium.rho
    source_1 = _ono_source(cfg, t_s, rho)
    half_psi = (psi + 0.5 * dt_s * source_1) / (1.0 + 0.5 * dt_s / cfg.tau_psi_s)
    source_2 = _ono_source(cfg, t_s + 0.5 * dt_s, rho)
    del half_psi
    return np.asarray(
        (psi + dt_s * source_2) / (1.0 + dt_s / cfg.tau_psi_s), dtype=np.float64
    ), source_2


def _ono_source(cfg: HallMHDPulsedConfig, t_s: float, rho: FloatArray) -> FloatArray:
    e_theta = _resolve_e_theta(cfg, t_s)
    j_theta = _resolve_j_theta(cfg, t_s)
    return np.asarray(
        cfg.hall_scale * cfg.equilibrium.R_null * e_theta - _eta_profile(cfg, rho) * j_theta,
        dtype=np.float64,
    )


def _eta_profile(cfg: HallMHDPulsedConfig, rho: FloatArray) -> FloatArray:
    return spitzer_resistivity_ohm_m(
        np.full_like(rho, cfg.electron_temperature_eV),
        Z_eff=cfg.Z_eff,
        ln_lambda=cfg.ln_lambda,
    )


def _resolve_e_theta(cfg: HallMHDPulsedConfig, t_s: float) -> FloatArray:
    rho = cfg.equilibrium.rho
    if cfg.E_theta_t is None:
        return faraday_e_theta_from_b_ramp(rho, cfg.B_ext_t, t_s)
    return _coerce_profile("E_theta", cfg.E_theta_t(t_s, rho), rho.size)


def _resolve_j_theta(cfg: HallMHDPulsedConfig, t_s: float) -> FloatArray:
    rho = cfg.equilibrium.rho
    if cfg.J_theta_t is None:
        return np.asarray(cfg.equilibrium.J_theta, dtype=np.float64)
    return _coerce_profile("J_theta", cfg.J_theta_t(t_s, rho), rho.size)


def _magnetic_energy_proxy(rho: FloatArray, psi: FloatArray) -> float:
    return float(trapezoid(0.5 * psi * psi * 2.0 * np.pi * rho / MU_0, rho))


def _l2(rho: FloatArray, profile: FloatArray) -> float:
    return float(np.sqrt(max(trapezoid(profile * profile * 2.0 * np.pi * rho, rho), 0.0)))


def _validate_config(config: HallMHDPulsedConfig) -> HallMHDPulsedConfig:
    _validate_grid(config.equilibrium.rho)
    _validate_profile("equilibrium.psi", config.equilibrium.psi, config.equilibrium.rho.size)
    _require_positive("tau_psi_s", config.tau_psi_s)
    _require_positive("electron_temperature_eV", config.electron_temperature_eV)
    _require_positive("Z_eff", config.Z_eff)
    _require_positive("ln_lambda", config.ln_lambda)
    _require_finite("hall_scale", config.hall_scale)
    if config.closure not in {"axisymmetric_ono_flux", "adiabatic_electron"}:
        raise ValueError("closure must be 'axisymmetric_ono_flux' or 'adiabatic_electron'")
    if config.time_integrator not in {"implicit_be", "imex_rk2"}:
        raise ValueError("time_integrator must be 'implicit_be' or 'imex_rk2'")
    _require_finite("B_ext_t(0)", config.B_ext_t(0.0))
    return config


def _validate_state(state: HallMHDPulsedState, cfg: HallMHDPulsedConfig) -> None:
    _require_non_negative("state.t_s", state.t_s)
    _validate_profile("state.psi", state.psi, cfg.equilibrium.rho.size)
    _validate_profile("state.B_z", state.B_z, cfg.equilibrium.rho.size)


def _validate_grid(rho_m: FloatArray) -> FloatArray:
    rho = np.asarray(rho_m, dtype=np.float64)
    if rho.ndim != 1 or rho.size < 3:
        raise ValueError("rho_m must be a one-dimensional grid with at least three points")
    if np.any(~np.isfinite(rho)):
        raise ValueError("rho_m must be finite")
    if rho[0] < 0.0:
        raise ValueError("rho_m must start at a non-negative radius")
    if np.any(np.diff(rho) <= 0.0):
        raise ValueError("rho_m must be strictly increasing")
    return rho


def _validate_profile(name: str, values: FloatArray, expected_size: int) -> FloatArray:
    profile = np.asarray(values, dtype=np.float64)
    if profile.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},)")
    if np.any(~np.isfinite(profile)):
        raise ValueError(f"{name} must be finite")
    return profile


def _coerce_profile(name: str, values: FloatArray | float, expected_size: int) -> FloatArray:
    profile = np.asarray(values, dtype=np.float64)
    if profile.shape == ():
        return np.full(expected_size, float(profile), dtype=np.float64)
    return _validate_profile(name, profile, expected_size)


def _require_positive(name: str, value: float) -> float:
    checked = _require_finite(name, value)
    if checked <= 0.0:
        raise ValueError(f"{name} must be positive")
    return checked


def _require_non_negative(name: str, value: float) -> float:
    checked = _require_finite(name, value)
    if checked < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return checked


def _require_finite(name: str, value: float) -> float:
    checked = float(value)
    if not np.isfinite(checked):
        raise ValueError(f"{name} must be finite")
    return checked
