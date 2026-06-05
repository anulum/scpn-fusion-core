// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — MIF/FRC Pulsed Compression
//! Supplied-current pulsed compression dynamics.

use super::coil_geometry::CoilGeometry;

const MU_0: f64 = 4.0e-7 * std::f64::consts::PI;
const ELEMENTARY_CHARGE_C: f64 = 1.602_176_634e-19;

#[derive(Debug, Clone, PartialEq)]
pub struct PulsedCompressionConfig {
    pub coil: CoilGeometry,
    pub plasma_mass_kg: f64,
    pub plasma_length_m: f64,
    pub gamma: f64,
    pub radial_loss_time_s: Option<f64>,
    pub z_eff: f64,
    pub ln_lambda: f64,
    pub min_radius_m: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PulsedCompressionState {
    pub t_s: f64,
    pub r_s_m: f64,
    pub d_r_s_dt_m_s: f64,
    pub t_i_ev: f64,
    pub t_e_ev: f64,
    pub density_m3: f64,
    pub beta: f64,
    pub b_ext_t: f64,
    pub internal_pressure_pa: f64,
    pub external_magnetic_pressure_pa: f64,
    pub thermal_energy_j: f64,
    pub compression_work_j: f64,
    pub radiated_loss_j: f64,
    pub energy_balance_residual: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoilCircuitState {
    pub t_s: f64,
    pub current_a: f64,
    pub drive_voltage_v: f64,
    pub d_current_dt_a_s: f64,
    pub magnetic_energy_j: f64,
    pub ohmic_loss_j: f64,
    pub source_work_j: f64,
    pub energy_balance_residual: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VoltageDrivenPulsedCompressionResult {
    pub coil_circuit: Vec<CoilCircuitState>,
    pub compression: Vec<PulsedCompressionState>,
}

pub fn coil_field_t(coil: &CoilGeometry, coil_current_a: f64) -> Result<f64, String> {
    coil.validate()?;
    let current = require_finite("coil_current_a", coil_current_a)?;
    Ok(MU_0 * coil.n_turns as f64 * current / coil.l_coil_m)
}

pub fn initial_coil_circuit_state(
    coil: &CoilGeometry,
    initial_current_a: f64,
) -> Result<CoilCircuitState, String> {
    coil.validate()?;
    let current = require_finite("initial_current_a", initial_current_a)?;
    Ok(CoilCircuitState {
        t_s: 0.0,
        current_a: current,
        drive_voltage_v: 0.0,
        d_current_dt_a_s: 0.0,
        magnetic_energy_j: 0.5 * coil.l_inductance_h * current * current,
        ohmic_loss_j: 0.0,
        source_work_j: 0.0,
        energy_balance_residual: 0.0,
    })
}

pub fn step_coil_circuit(
    state: &CoilCircuitState,
    coil: &CoilGeometry,
    drive_voltage_v: f64,
    dt_s: f64,
) -> Result<CoilCircuitState, String> {
    coil.validate()?;
    validate_circuit_state(state)?;
    let voltage = require_bank_voltage(coil, drive_voltage_v)?;
    let dt = require_positive("dt_s", dt_s)?;
    let inductance = coil.l_inductance_h;
    let resistance = coil.r_resistance_ohm;
    let tau = inductance / resistance;
    let decay = (-dt / tau).exp();
    let steady_current = voltage / resistance;
    let delta = state.current_a - steady_current;
    let current = steady_current + delta * decay;
    let int_i_dt = steady_current * dt + delta * tau * (1.0 - decay);
    let int_i2_dt = steady_current * steady_current * dt
        + 2.0 * steady_current * delta * tau * (1.0 - decay)
        + delta * delta * tau * 0.5 * (1.0 - decay * decay);
    let interval_source_work = voltage * int_i_dt;
    let interval_ohmic_loss = resistance * int_i2_dt;
    let magnetic_energy = 0.5 * inductance * current * current;
    let energy_delta = magnetic_energy - state.magnetic_energy_j;
    let residual = energy_delta - interval_source_work + interval_ohmic_loss;
    let scale = energy_delta
        .abs()
        .max(interval_source_work.abs())
        .max(interval_ohmic_loss.abs())
        .max(magnetic_energy.abs())
        .max(1.0e-30);
    Ok(CoilCircuitState {
        t_s: state.t_s + dt,
        current_a: current,
        drive_voltage_v: voltage,
        d_current_dt_a_s: (voltage - resistance * current) / inductance,
        magnetic_energy_j: magnetic_energy,
        ohmic_loss_j: state.ohmic_loss_j + interval_ohmic_loss,
        source_work_j: state.source_work_j + interval_source_work,
        energy_balance_residual: residual / scale,
    })
}

pub fn run_coil_circuit(
    coil: &CoilGeometry,
    drive_voltage_v: f64,
    dt_s: f64,
    n_steps: usize,
    initial_current_a: f64,
) -> Result<Vec<CoilCircuitState>, String> {
    if n_steps == 0 {
        return Err("n_steps must be positive".to_string());
    }
    let mut states = Vec::with_capacity(n_steps + 1);
    states.push(initial_coil_circuit_state(coil, initial_current_a)?);
    for _ in 0..n_steps {
        let next = step_coil_circuit(
            states.last().expect("state exists"),
            coil,
            drive_voltage_v,
            dt_s,
        )?;
        states.push(next);
    }
    Ok(states)
}

pub fn plasma_volume_m3(r_s_m: f64, plasma_length_m: f64) -> Result<f64, String> {
    let radius = require_positive("r_s_m", r_s_m)?;
    let length = require_positive("plasma_length_m", plasma_length_m)?;
    Ok(std::f64::consts::PI * radius * radius * length)
}

pub fn spitzer_resistivity_ohm_m(t_e_ev: f64, z_eff: f64, ln_lambda: f64) -> Result<f64, String> {
    let temperature = require_positive("t_e_ev", t_e_ev)?;
    let charge = require_positive("z_eff", z_eff)?;
    let coulomb_log = require_positive("ln_lambda", ln_lambda)?;
    Ok(1.65e-9 * charge * coulomb_log / temperature.powf(1.5))
}

pub fn adiabatic_temperature_update_ev(
    temperature_ev: f64,
    old_volume_m3: f64,
    new_volume_m3: f64,
    gamma: f64,
) -> Result<f64, String> {
    let temperature = require_positive("temperature_ev", temperature_ev)?;
    let old_volume = require_positive("old_volume_m3", old_volume_m3)?;
    let new_volume = require_positive("new_volume_m3", new_volume_m3)?;
    let gamma_value = require_positive("gamma", gamma)?;
    if gamma_value <= 1.0 {
        return Err("gamma must be greater than one".to_string());
    }
    Ok(temperature * (old_volume / new_volume).powf(gamma_value - 1.0))
}

pub fn step_pulsed_compression(
    state: &PulsedCompressionState,
    config: &PulsedCompressionConfig,
    coil_current_a: f64,
    dt_s: f64,
) -> Result<PulsedCompressionState, String> {
    validate_config(config)?;
    validate_state(state)?;
    let dt = require_positive("dt_s", dt_s)?;
    let old_volume = plasma_volume_m3(state.r_s_m, config.plasma_length_m)?;
    let old_pressure = thermal_pressure_pa(state.density_m3, state.t_i_ev, state.t_e_ev)?;
    let field = coil_field_t(&config.coil, coil_current_a)?;
    let external_pressure = magnetic_pressure_pa(field);
    let force_area = 2.0 * std::f64::consts::PI * state.r_s_m * config.plasma_length_m;
    let acceleration = (old_pressure - external_pressure) * force_area / config.plasma_mass_kg;
    let speed = state.d_r_s_dt_m_s + acceleration * dt;
    let radius = (state.r_s_m + speed * dt).max(config.min_radius_m);
    let new_volume = plasma_volume_m3(radius, config.plasma_length_m)?;
    let density = state.density_m3 * old_volume / new_volume;
    let t_i_adiabatic =
        adiabatic_temperature_update_ev(state.t_i_ev, old_volume, new_volume, config.gamma)?;
    let t_e_adiabatic =
        adiabatic_temperature_update_ev(state.t_e_ev, old_volume, new_volume, config.gamma)?;
    let thermal_adiabatic = thermal_energy_j(density, new_volume, t_i_adiabatic, t_e_adiabatic);
    let loss_factor = match config.radial_loss_time_s {
        None => 1.0,
        Some(value) => (-dt / require_positive("radial_loss_time_s", value)?).exp(),
    };
    let t_i = t_i_adiabatic * loss_factor;
    let t_e = t_e_adiabatic * loss_factor;
    let pressure = thermal_pressure_pa(density, t_i, t_e)?;
    let thermal_energy = thermal_energy_j(density, new_volume, t_i, t_e);
    let compression_work = state.compression_work_j + (thermal_adiabatic - state.thermal_energy_j);
    let radiated_loss = state.radiated_loss_j + (thermal_adiabatic - thermal_energy);
    let residual =
        thermal_energy - state.thermal_energy_j - (compression_work - state.compression_work_j)
            + (radiated_loss - state.radiated_loss_j);
    let scale = thermal_energy
        .abs()
        .max(state.thermal_energy_j.abs())
        .max(compression_work.abs())
        .max(1.0e-30);

    Ok(PulsedCompressionState {
        t_s: state.t_s + dt,
        r_s_m: radius,
        d_r_s_dt_m_s: speed,
        t_i_ev: t_i,
        t_e_ev: t_e,
        density_m3: density,
        beta: beta(pressure, field),
        b_ext_t: field,
        internal_pressure_pa: pressure,
        external_magnetic_pressure_pa: external_pressure,
        thermal_energy_j: thermal_energy,
        compression_work_j: compression_work,
        radiated_loss_j: radiated_loss,
        energy_balance_residual: residual / scale,
    })
}

pub fn run_pulsed_compression(
    initial: PulsedCompressionState,
    config: &PulsedCompressionConfig,
    coil_current_a: f64,
    dt_s: f64,
    n_steps: usize,
) -> Result<Vec<PulsedCompressionState>, String> {
    if n_steps == 0 {
        return Err("n_steps must be positive".to_string());
    }
    let mut states = Vec::with_capacity(n_steps + 1);
    states.push(initial);
    for _ in 0..n_steps {
        let next = step_pulsed_compression(
            states.last().expect("state exists"),
            config,
            coil_current_a,
            dt_s,
        )?;
        states.push(next);
    }
    Ok(states)
}

pub fn run_voltage_driven_pulsed_compression(
    initial: PulsedCompressionState,
    config: &PulsedCompressionConfig,
    drive_voltage_v: f64,
    dt_s: f64,
    n_steps: usize,
    initial_current_a: f64,
) -> Result<VoltageDrivenPulsedCompressionResult, String> {
    validate_config(config)?;
    if n_steps == 0 {
        return Err("n_steps must be positive".to_string());
    }
    let coil_circuit = run_coil_circuit(
        &config.coil,
        drive_voltage_v,
        dt_s,
        n_steps,
        initial_current_a,
    )?;
    let mut compression = Vec::with_capacity(n_steps + 1);
    compression.push(initial);
    for circuit_state in coil_circuit.iter().take(n_steps) {
        let next = step_pulsed_compression(
            compression.last().expect("state exists"),
            config,
            circuit_state.current_a,
            dt_s,
        )?;
        compression.push(next);
    }
    Ok(VoltageDrivenPulsedCompressionResult {
        coil_circuit,
        compression,
    })
}

fn validate_config(config: &PulsedCompressionConfig) -> Result<(), String> {
    config.coil.validate()?;
    require_positive("plasma_mass_kg", config.plasma_mass_kg)?;
    require_positive("plasma_length_m", config.plasma_length_m)?;
    if require_positive("gamma", config.gamma)? <= 1.0 {
        return Err("gamma must be greater than one".to_string());
    }
    if let Some(value) = config.radial_loss_time_s {
        require_positive("radial_loss_time_s", value)?;
    }
    require_positive("z_eff", config.z_eff)?;
    require_positive("ln_lambda", config.ln_lambda)?;
    require_positive("min_radius_m", config.min_radius_m)?;
    Ok(())
}

fn validate_state(state: &PulsedCompressionState) -> Result<(), String> {
    require_finite("t_s", state.t_s)?;
    require_positive("r_s_m", state.r_s_m)?;
    require_finite("d_r_s_dt_m_s", state.d_r_s_dt_m_s)?;
    require_positive("t_i_ev", state.t_i_ev)?;
    require_positive("t_e_ev", state.t_e_ev)?;
    require_positive("density_m3", state.density_m3)?;
    Ok(())
}

fn validate_circuit_state(state: &CoilCircuitState) -> Result<(), String> {
    require_finite("t_s", state.t_s)?;
    require_finite("current_a", state.current_a)?;
    require_finite("drive_voltage_v", state.drive_voltage_v)?;
    require_finite("d_current_dt_a_s", state.d_current_dt_a_s)?;
    require_finite("magnetic_energy_j", state.magnetic_energy_j)?;
    require_finite("ohmic_loss_j", state.ohmic_loss_j)?;
    require_finite("source_work_j", state.source_work_j)?;
    require_finite("energy_balance_residual", state.energy_balance_residual)?;
    Ok(())
}

fn require_bank_voltage(coil: &CoilGeometry, drive_voltage_v: f64) -> Result<f64, String> {
    let voltage = require_finite("drive_voltage_v", drive_voltage_v)?;
    if voltage.abs() > coil.bank_voltage_max_v {
        return Err("drive_voltage_v exceeds coil.bank_voltage_max_v".to_string());
    }
    Ok(voltage)
}

fn magnetic_pressure_pa(field_t: f64) -> f64 {
    field_t * field_t / (2.0 * MU_0)
}

fn thermal_pressure_pa(density_m3: f64, t_i_ev: f64, t_e_ev: f64) -> Result<f64, String> {
    Ok(require_positive("density_m3", density_m3)?
        * (require_positive("t_i_ev", t_i_ev)? + require_positive("t_e_ev", t_e_ev)?)
        * ELEMENTARY_CHARGE_C)
}

fn thermal_energy_j(density_m3: f64, volume_m3: f64, t_i_ev: f64, t_e_ev: f64) -> f64 {
    1.5 * density_m3 * volume_m3 * (t_i_ev + t_e_ev) * ELEMENTARY_CHARGE_C
}

fn beta(pressure_pa: f64, field_t: f64) -> f64 {
    if field_t == 0.0 {
        f64::INFINITY
    } else {
        2.0 * MU_0 * pressure_pa / (field_t * field_t)
    }
}

fn require_finite(name: &str, value: f64) -> Result<f64, String> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("{name} must be finite"))
    }
}

fn require_positive(name: &str, value: f64) -> Result<f64, String> {
    let checked = require_finite(name, value)?;
    if checked > 0.0 {
        Ok(checked)
    } else {
        Err(format!("{name} must be positive"))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        adiabatic_temperature_update_ev, coil_field_t, initial_coil_circuit_state,
        plasma_volume_m3, run_coil_circuit, run_pulsed_compression,
        run_voltage_driven_pulsed_compression, spitzer_resistivity_ohm_m, step_coil_circuit,
        step_pulsed_compression, PulsedCompressionConfig, PulsedCompressionState,
        ELEMENTARY_CHARGE_C, MU_0,
    };
    use crate::compression::CoilGeometry;

    fn coil() -> CoilGeometry {
        CoilGeometry {
            n_turns: 80,
            l_coil_m: 1.0,
            r_coil_m: 0.35,
            l_inductance_h: 2.0e-6,
            r_resistance_ohm: 0.02,
            bank_voltage_max_v: 20_000.0,
        }
    }

    fn config() -> PulsedCompressionConfig {
        PulsedCompressionConfig {
            coil: coil(),
            plasma_mass_kg: 2.0e-5,
            plasma_length_m: 1.0,
            gamma: 5.0 / 3.0,
            radial_loss_time_s: None,
            z_eff: 1.0,
            ln_lambda: 17.0,
            min_radius_m: 1.0e-4,
        }
    }

    fn state() -> PulsedCompressionState {
        let density = 2.0e21;
        let volume = plasma_volume_m3(0.2, 1.0).unwrap();
        let t_i = 10_000.0;
        let t_e = 5_000.0;
        let pressure = density * (t_i + t_e) * ELEMENTARY_CHARGE_C;
        let field = coil_field_t(&coil(), 2.0e5).unwrap();
        PulsedCompressionState {
            t_s: 0.0,
            r_s_m: 0.2,
            d_r_s_dt_m_s: 0.0,
            t_i_ev: t_i,
            t_e_ev: t_e,
            density_m3: density,
            beta: 2.0 * MU_0 * pressure / (field * field),
            b_ext_t: field,
            internal_pressure_pa: pressure,
            external_magnetic_pressure_pa: field * field / (2.0 * MU_0),
            thermal_energy_j: 1.5 * density * volume * (t_i + t_e) * ELEMENTARY_CHARGE_C,
            compression_work_j: 0.0,
            radiated_loss_j: 0.0,
            energy_balance_residual: 0.0,
        }
    }

    #[test]
    fn coil_field_matches_uniform_solenoid() {
        let field = coil_field_t(&coil(), 2.0e5).unwrap();
        let expected = MU_0 * 80.0 * 2.0e5;
        assert!((field - expected).abs() / expected < 1.0e-15);
    }

    #[test]
    fn coil_circuit_matches_exact_rl_solution() {
        let voltage = 12_000.0;
        let dt = 25.0e-9;
        let current = 2.5e5;
        let initial = initial_coil_circuit_state(&coil(), current).expect("valid initial");
        let next = step_coil_circuit(&initial, &coil(), voltage, dt).expect("valid step");
        let tau = coil().l_inductance_h / coil().r_resistance_ohm;
        let steady = voltage / coil().r_resistance_ohm;
        let expected = steady + (current - steady) * (-dt / tau).exp();
        assert!((next.current_a - expected).abs() / expected.abs() < 1.0e-14);
        assert!(next.magnetic_energy_j > initial.magnetic_energy_j);
        assert!(next.ohmic_loss_j > 0.0);
        assert!(next.source_work_j > 0.0);
        assert!(next.energy_balance_residual.abs() < 1.0e-12);
    }

    #[test]
    fn adiabatic_invariant_is_preserved() {
        let old_volume = plasma_volume_m3(0.2, 1.0).unwrap();
        let new_volume = plasma_volume_m3(0.15, 1.0).unwrap();
        let gamma = 5.0 / 3.0;
        let updated =
            adiabatic_temperature_update_ev(1000.0, old_volume, new_volume, gamma).unwrap();
        let lhs = updated * new_volume.powf(gamma - 1.0);
        let rhs = 1000.0 * old_volume.powf(gamma - 1.0);
        assert!((lhs - rhs).abs() / rhs < 1.0e-14);
    }

    #[test]
    fn external_compression_heats_and_shrinks_plasma() {
        let next = step_pulsed_compression(&state(), &config(), 5.0e5, 2.0e-8).unwrap();
        assert!(next.r_s_m < 0.2);
        assert!(next.t_i_ev > 10_000.0);
        assert!(next.compression_work_j > 0.0);
        assert!(next.energy_balance_residual.abs() < 1.0e-12);
    }

    #[test]
    fn run_returns_requested_history() {
        let states = run_pulsed_compression(state(), &config(), 5.0e5, 1.0e-8, 4).unwrap();
        assert_eq!(states.len(), 5);
        assert!((states[4].t_s - 4.0e-8).abs() < 1.0e-20);
    }

    #[test]
    fn voltage_driven_circuit_feeds_compression() {
        let result =
            run_voltage_driven_pulsed_compression(state(), &config(), 20_000.0, 1.0e-9, 32, 5.0e5)
                .expect("valid voltage-driven run");
        assert_eq!(result.coil_circuit.len(), 33);
        assert_eq!(result.compression.len(), 33);
        assert!(result.coil_circuit.last().unwrap().current_a > result.coil_circuit[0].current_a);
        assert!(result.compression.last().unwrap().r_s_m < result.compression[0].r_s_m);
        assert!(result.compression.last().unwrap().compression_work_j > 0.0);
        assert!(
            result
                .coil_circuit
                .last()
                .unwrap()
                .energy_balance_residual
                .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn spitzer_scaling_and_invalid_inputs() {
        let cold = spitzer_resistivity_ohm_m(100.0, 1.0, 17.0).unwrap();
        let hot = spitzer_resistivity_ohm_m(400.0, 1.0, 17.0).unwrap();
        assert!((cold / hot - 8.0).abs() < 1.0e-12);
        assert!(coil_field_t(&coil(), f64::INFINITY).is_err());
        assert!(run_pulsed_compression(state(), &config(), 1.0, 1.0e-8, 0).is_err());
        assert!(run_coil_circuit(&coil(), 20_001.0, 1.0e-9, 1, 0.0).is_err());
    }
}
