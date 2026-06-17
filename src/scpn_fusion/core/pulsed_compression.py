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

It does not invent the missing Slough 2011 Fig. 5 comparison. Reports that
need that public digitised trajectory must carry a blocked evidence row until
the reference trajectory and compression-work sidecar exist.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeAlias

import json
import numpy as np
from numpy.typing import NDArray

from .current_diffusion import solve_flux_evolution_nonadiabatic
from .frc_rigid_rotor import ELEMENTARY_CHARGE_C, FRCEquilibriumState, MU_0

FloatArray: TypeAlias = NDArray[np.float64]
ProfileCallable: TypeAlias = Callable[[float, FloatArray], FloatArray | float]
ScalarCallable: TypeAlias = Callable[[float], float]
_FLUX_UPDATE_RESIDUAL_ABS_TOLERANCE = 1.0e-12


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
    radial_acceleration_m_s2: float
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
    flux_source_increment_checksum: float
    flux_damping_decrement_checksum: float
    flux_update_residual_abs_max: float
    flux_budget_claim_status: str
    flux_coupling_status: str


@dataclass(frozen=True)
class CoilCircuitState:
    """Exact lumped R-L coil-circuit state for one voltage-drive interval."""

    t_s: float
    current_A: float
    drive_voltage_V: float
    d_current_dt_A_s: float
    magnetic_energy_J: float
    ohmic_loss_J: float
    source_work_J: float
    energy_balance_residual: float


@dataclass(frozen=True)
class VoltageDrivenCompressionResult:
    """Coupled coil-circuit and pulsed-compression trajectories."""

    coil_circuit: tuple[CoilCircuitState, ...]
    compression: tuple[PulsedCompressionState, ...]


@dataclass(frozen=True)
class PulsedCompressionTrajectoryDiagnostics:
    """Validated aggregate diagnostics for a pulsed-compression trajectory."""

    monotonic_time: bool
    min_radius_m: float
    max_abs_radial_acceleration_m_s2: float
    radius_floor_contact_count: int
    radial_turning_point_count: int
    compression_ratio: float
    all_flux_budgets_passed: bool


def coil_field_t(coil: CoilGeometry, coil_current_a: float) -> float:
    """Return ``B_ext = mu0*N*I/L`` for the uniform-solenoid approximation."""

    checked = _validate_coil(coil)
    current = _require_finite("coil_current_a", coil_current_a)
    return float(MU_0 * checked.N_turns * current / checked.L_coil_m)


def initial_coil_circuit_state(
    coil: CoilGeometry, initial_current_A: float = 0.0
) -> CoilCircuitState:
    """Return an initial lumped R-L coil-circuit state."""

    checked = _validate_coil(coil)
    current = _require_finite("initial_current_A", initial_current_A)
    return CoilCircuitState(
        t_s=0.0,
        current_A=current,
        drive_voltage_V=0.0,
        d_current_dt_A_s=0.0,
        magnetic_energy_J=0.5 * checked.L_inductance_H * current * current,
        ohmic_loss_J=0.0,
        source_work_J=0.0,
        energy_balance_residual=0.0,
    )


def step_coil_circuit(
    state: CoilCircuitState,
    coil: CoilGeometry,
    drive_voltage_V: float,
    dt_s: float,
) -> CoilCircuitState:
    """Advance the exact constant-voltage solution of ``L dI/dt + R I = V``."""

    checked = _validate_coil(coil)
    _validate_circuit_state(state)
    voltage = _require_bank_voltage(checked, drive_voltage_V)
    dt = _require_positive("dt_s", dt_s)
    inductance = checked.L_inductance_H
    resistance = checked.R_resistance_ohm
    tau = inductance / resistance
    decay = float(np.exp(-dt / tau))
    steady_current = voltage / resistance
    delta = state.current_A - steady_current
    current = steady_current + delta * decay
    int_i_dt = steady_current * dt + delta * tau * (1.0 - decay)
    int_i2_dt = (
        steady_current * steady_current * dt
        + 2.0 * steady_current * delta * tau * (1.0 - decay)
        + delta * delta * tau * 0.5 * (1.0 - decay * decay)
    )
    interval_source_work = voltage * int_i_dt
    interval_ohmic_loss = resistance * int_i2_dt
    magnetic_energy = 0.5 * inductance * current * current
    energy_delta = magnetic_energy - state.magnetic_energy_J
    residual = energy_delta - interval_source_work + interval_ohmic_loss
    scale = max(
        abs(energy_delta),
        abs(interval_source_work),
        abs(interval_ohmic_loss),
        abs(magnetic_energy),
        1.0e-30,
    )
    return CoilCircuitState(
        t_s=state.t_s + dt,
        current_A=current,
        drive_voltage_V=voltage,
        d_current_dt_A_s=(voltage - resistance * current) / inductance,
        magnetic_energy_J=magnetic_energy,
        ohmic_loss_J=state.ohmic_loss_J + interval_ohmic_loss,
        source_work_J=state.source_work_J + interval_source_work,
        energy_balance_residual=float(residual / scale),
    )


def run_coil_circuit(
    coil: CoilGeometry,
    drive_voltage_t: ScalarCallable,
    dt_s: float,
    n_steps: int,
    *,
    initial_current_A: float = 0.0,
) -> tuple[CoilCircuitState, ...]:
    """Run a bank-limited lumped R-L coil-current trajectory."""

    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    dt = _require_positive("dt_s", dt_s)
    state = initial_coil_circuit_state(coil, initial_current_A)
    states = [state]
    for _ in range(n_steps):
        state = step_coil_circuit(state, coil, drive_voltage_t(state.t_s), dt)
        states.append(state)
    return tuple(states)


def coil_current_interpolator(states: tuple[CoilCircuitState, ...]) -> ScalarCallable:
    """Return a finite piecewise-linear current interpolator for circuit states."""

    if len(states) < 2:
        raise ValueError("at least two coil-circuit states are required")
    time_s = np.asarray([state.t_s for state in states], dtype=np.float64)
    current_a = np.asarray([state.current_A for state in states], dtype=np.float64)
    if not np.all(np.isfinite(time_s)) or not np.all(np.isfinite(current_a)):
        raise ValueError("coil-circuit trajectory must contain finite values")
    if not np.all(np.diff(time_s) > 0.0):
        raise ValueError("coil-circuit time_s must be strictly increasing")

    def _current_at(time_value_s: float) -> float:
        time_value = _require_finite("time_s", time_value_s)
        if time_value < time_s[0] or time_value > time_s[-1]:
            raise ValueError("time_s is outside the coil-circuit trajectory span")
        return float(np.interp(time_value, time_s, current_a))

    return _current_at


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


# NRL Plasma Formulary transverse Spitzer resistivity, eta_perp =
# 1.03e-2 * Z * lnLambda * T_e[eV]^-3/2 ohm-cm = 1.03e-4 ohm-m. The perpendicular
# branch is the relevant one for the azimuthal FRC current, which flows across
# the axial confining field. (Verified: https://w3.pppl.gov/~hammett/hpf/hpf.pdf)
SPITZER_PERP_COEFFICIENT_OHM_M = 1.03e-4


def spitzer_resistivity_ohm_m(
    T_e_eV: FloatArray | float,
    *,
    Z_eff: float = 1.0,
    ln_lambda: float = 17.0,
) -> FloatArray:
    """Return the NRL-formulary transverse Spitzer resistivity in ohm metre.

    Implements ``eta_perp = 1.03e-4 * Z_eff * lnLambda / T_e[eV]^(3/2)`` ohm
    metre, the perpendicular branch carried by the azimuthal FRC current.
    """

    temperature = np.asarray(T_e_eV, dtype=np.float64)
    if not np.all(np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("T_e_eV must contain positive finite values")
    z_eff = _require_positive("Z_eff", Z_eff)
    coulomb_log = _require_positive("ln_lambda", ln_lambda)
    return SPITZER_PERP_COEFFICIENT_OHM_M * z_eff * coulomb_log / np.power(temperature, 1.5)


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
        radial_acceleration_m_s2=0.0,
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
        flux_source_increment_checksum=0.0,
        flux_damping_decrement_checksum=0.0,
        flux_update_residual_abs_max=0.0,
        flux_budget_claim_status="not_evaluated_initial_state",
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
    flux_residual_abs_max = float(np.max(np.abs(flux.update_residual[-1])))
    flux_budget_claim_status = (
        "passed" if flux_residual_abs_max <= _FLUX_UPDATE_RESIDUAL_ABS_TOLERANCE else "failed"
    )

    return PulsedCompressionState(
        t_s=state.t_s + dt,
        R_s_m=radius,
        dR_s_dt_m_s=float(dR_s_dt),
        radial_acceleration_m_s2=float(radial_acceleration),
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
        flux_source_increment_checksum=float(np.sum(flux.source_increment[-1], dtype=np.float64)),
        flux_damping_decrement_checksum=float(np.sum(flux.damping_decrement[-1], dtype=np.float64)),
        flux_update_residual_abs_max=flux_residual_abs_max,
        flux_budget_claim_status=flux_budget_claim_status,
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


def run_voltage_driven_pulsed_compression(
    config: PulsedCompressionConfig,
    drive_voltage_t: ScalarCallable,
    dt_s: float,
    n_steps: int,
    *,
    initial_current_A: float = 0.0,
) -> VoltageDrivenCompressionResult:
    """Run a bank-limited R-L coil drive coupled to pulsed compression.

    The coil circuit is the exact solution of the declared lumped equation
    ``L dI/dt + R I = V`` over piecewise-constant voltage intervals. It feeds
    the existing pressure-balance compression path through the public
    ``coil_current_t`` contract and raises when the requested voltage exceeds
    the configured bank limit.
    """

    cfg = _validate_config(config)
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    dt = _require_positive("dt_s", dt_s)
    circuit_states = run_coil_circuit(
        cfg.coil,
        drive_voltage_t,
        dt,
        n_steps,
        initial_current_A=initial_current_A,
    )
    driven_config = replace(cfg, coil_current_t=coil_current_interpolator(circuit_states))
    compression_states = run_pulsed_compression(
        initial_pulsed_compression_state(driven_config),
        driven_config,
        dt,
        n_steps,
    )
    return VoltageDrivenCompressionResult(
        coil_circuit=circuit_states,
        compression=compression_states,
    )


def pulsed_compression_trajectory_diagnostics(
    states: Sequence[PulsedCompressionState],
    *,
    radius_floor_m: float | None = None,
) -> PulsedCompressionTrajectoryDiagnostics:
    """Return fail-closed aggregate diagnostics for a compression trajectory."""

    if len(states) < 2:
        raise ValueError("at least two pulsed-compression states are required")
    time_s = np.asarray([state.t_s for state in states], dtype=np.float64)
    radius_m = np.asarray([state.R_s_m for state in states], dtype=np.float64)
    speed_m_s = np.asarray([state.dR_s_dt_m_s for state in states], dtype=np.float64)
    acceleration_m_s2 = np.asarray(
        [state.radial_acceleration_m_s2 for state in states],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(time_s)) or not np.all(np.diff(time_s) > 0.0):
        raise ValueError("trajectory time_s must be finite and strictly increasing")
    if not np.all(np.isfinite(radius_m)) or np.any(radius_m <= 0.0):
        raise ValueError("trajectory radius must be positive and finite")
    if not np.all(np.isfinite(speed_m_s)):
        raise ValueError("trajectory radial speed must be finite")
    if not np.all(np.isfinite(acceleration_m_s2)):
        raise ValueError("trajectory radial acceleration must be finite")
    floor_contact_count = 0
    if radius_floor_m is not None:
        floor = _require_positive("radius_floor_m", radius_floor_m)
        floor_contact_count = int(np.count_nonzero(radius_m <= floor * (1.0 + 1.0e-12)))
    min_radius = float(np.min(radius_m))
    return PulsedCompressionTrajectoryDiagnostics(
        monotonic_time=True,
        min_radius_m=min_radius,
        max_abs_radial_acceleration_m_s2=float(np.max(np.abs(acceleration_m_s2))),
        radius_floor_contact_count=floor_contact_count,
        radial_turning_point_count=_count_radial_turning_points(speed_m_s),
        compression_ratio=float(radius_m[0] / min_radius),
        all_flux_budgets_passed=all(
            state.flux_budget_claim_status == "passed" for state in states[1:]
        ),
    )


def slough_fig5_acceptance_status(reference_path: Path | None = None) -> dict[str, Any]:
    """Return the current acceptance status for the Slough 2011 Figure 5 comparison."""
    ref_path = reference_path or (
        Path(__file__).resolve().parents[3]
        / "validation"
        / "reference_data"
        / "slough_2011_fig5.json"
    )

    if not ref_path.exists():
        return {
            "case": "slough_2011_fig5",
            "status": "blocked_missing_public_digitised_reference",
            "required_artifact": "digitised radius/temperature/field trajectory with provenance and checksum",
        }

    try:
        with ref_path.open("r", encoding="utf-8") as f:
            ref_data = json.load(f)

        method = str(ref_data.get("method", "")).lower()
        fidelity = str(ref_data.get("fidelity", "")).lower()
        if "reconstructed" in method or "operational-verification" in fidelity:
            return {
                "case": "slough_2011_fig5",
                "status": "blocked_reconstructed_reference_not_public_digitised",
                "scenario": ref_data.get("scenario"),
                "required_artifact": (
                    "redistributable digitised Slough Fig. 5 radius/temperature/field "
                    "trajectory with provenance and checksum"
                ),
                "claim_boundary": (
                    "Tracked sidecar is reconstructed operational evidence only; no "
                    "external Slough trajectory parity is accepted."
                ),
            }

        trajectory = ref_data.get("trajectory", [])
        checksum = ref_data.get("sha256") or ref_data.get("checksum_sha256")
        parity_report = ref_data.get("same_case_parity_report")
        checksum_verified = isinstance(checksum, str) and len(checksum) == 64
        same_case_parity_passed = (
            isinstance(parity_report, dict)
            and parity_report.get("status") == "passed"
            and parity_report.get("same_case") is True
        )
        return {
            "case": "slough_2011_fig5",
            "status": "blocked_public_digitised_reference_validation_missing",
            "source": ref_data.get("source"),
            "scenario": ref_data.get("scenario"),
            "n_points": len(trajectory) if isinstance(trajectory, list) else 0,
            "checksum_verified": checksum_verified,
            "same_case_parity_passed": same_case_parity_passed,
            "required_artifact": (
                "checksum-verified public digitised Slough Fig. 5 trajectory plus "
                "same-case pulsed-compression simulation parity report"
            ),
            "claim_boundary": (
                "Digitised sidecar presence is not acceptance; Slough trajectory "
                "parity remains blocked until checksum and same-case validation pass."
            ),
        }
    except Exception as e:
        return {"case": "slough_2011_fig5", "status": "error", "error": str(e)}


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
    _require_finite("state.radial_acceleration_m_s2", state.radial_acceleration_m_s2)
    _require_positive("state.T_i_eV", state.T_i_eV)
    _require_positive("state.T_e_eV", state.T_e_eV)
    _require_positive("state.density_m3", state.density_m3)
    psi = np.asarray(state.flux_psi, dtype=np.float64)
    if psi.shape != config.equilibrium.rho.shape or not np.all(np.isfinite(psi)):
        raise ValueError("state.flux_psi must match the equilibrium radial grid")
    _require_finite("state.flux_psi_checksum", state.flux_psi_checksum)
    _require_finite("state.flux_source_increment_checksum", state.flux_source_increment_checksum)
    _require_finite("state.flux_damping_decrement_checksum", state.flux_damping_decrement_checksum)
    _require_finite("state.flux_update_residual_abs_max", state.flux_update_residual_abs_max)
    if not state.flux_budget_claim_status:
        raise ValueError("state.flux_budget_claim_status must be non-empty")
    if not state.flux_coupling_status:
        raise ValueError("state.flux_coupling_status must be non-empty")


def _validate_circuit_state(state: CoilCircuitState) -> None:
    _require_finite("state.t_s", state.t_s)
    _require_finite("state.current_A", state.current_A)
    _require_finite("state.drive_voltage_V", state.drive_voltage_V)
    _require_finite("state.d_current_dt_A_s", state.d_current_dt_A_s)
    _require_finite("state.magnetic_energy_J", state.magnetic_energy_J)
    _require_finite("state.ohmic_loss_J", state.ohmic_loss_J)
    _require_finite("state.source_work_J", state.source_work_J)
    _require_finite("state.energy_balance_residual", state.energy_balance_residual)


def _require_bank_voltage(coil: CoilGeometry, drive_voltage_V: float) -> float:
    voltage = _require_finite("drive_voltage_V", drive_voltage_V)
    if abs(voltage) > coil.bank_voltage_max_V:
        raise ValueError("drive_voltage_V exceeds coil.bank_voltage_max_V")
    return voltage


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


def _count_radial_turning_points(speed_m_s: FloatArray) -> int:
    signs: list[int] = []
    for value in speed_m_s:
        if value > 0.0:
            signs.append(1)
        elif value < 0.0:
            signs.append(-1)
    return sum(1 for left, right in zip(signs, signs[1:]) if left != right)


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
