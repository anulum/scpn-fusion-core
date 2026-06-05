// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — FRC Faraday Recovery
//! Classical Faraday back-EMF and recovery-energy contract for MIF/FRC.

use crate::compression::{PulsedCompressionState, VoltageDrivenPulsedCompressionResult};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaradayRecoveryTrajectoryPoint {
    pub t_s: f64,
    pub separatrix_radius_m: f64,
    pub b_ext_t: f64,
    pub d_radius_dt_m_s: Option<f64>,
    pub d_b_ext_dt_t_s: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FaradayRecoverySample {
    pub t_s: f64,
    pub separatrix_radius_m: f64,
    pub b_ext_t: f64,
    pub d_radius_dt_m_s: f64,
    pub d_b_ext_dt_t_s: f64,
    pub magnetic_flux_wb: f64,
    pub back_emf_v: f64,
    pub load_current_a: f64,
    pub load_power_w: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FaradayCompressionFluxBudget {
    pub source_increment_checksum: f64,
    pub damping_decrement_checksum: f64,
    pub update_residual_abs_max: f64,
    pub budget_claim_status: String,
    pub coupling_status: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FaradayRecoveryReport {
    pub samples: Vec<FaradayRecoverySample>,
    pub n_turns: u32,
    pub coil_resistance_ohm: f64,
    pub recovered_energy_j: f64,
    pub flux_initial_wb: f64,
    pub flux_final_wb: f64,
    pub max_abs_back_emf_v: f64,
    pub max_abs_load_current_a: f64,
    pub compression_work_j: Option<f64>,
    pub energy_budget_relative_error: Option<f64>,
    pub energy_budget_passed: Option<bool>,
    pub budget_claim_status: String,
    pub coil_source_work_j: Option<f64>,
    pub source_energy_budget_relative_error: Option<f64>,
    pub source_energy_budget_passed: Option<bool>,
    pub source_budget_claim_status: String,
    pub compression_flux_budget: Option<FaradayCompressionFluxBudget>,
    pub compression_flux_budget_passed: Option<bool>,
    pub compression_flux_budget_claim_status: String,
}

struct BudgetEvaluation {
    work_j: Option<f64>,
    relative_error: Option<f64>,
    passed: Option<bool>,
    status: String,
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

fn require_positive_turns(n_turns: u32) -> Result<u32, String> {
    if n_turns == 0 {
        Err("N_turns must be a positive integer".to_string())
    } else {
        Ok(n_turns)
    }
}

pub fn magnetic_flux_wb(separatrix_radius_m: f64, b_ext_t: f64) -> Result<f64, String> {
    let radius = require_positive("separatrix_radius_m", separatrix_radius_m)?;
    let b_ext = require_finite("b_ext_t", b_ext_t)?;
    Ok(b_ext * std::f64::consts::PI * radius * radius)
}

pub fn faraday_back_emf_from_values(
    separatrix_radius_m: f64,
    b_ext_t: f64,
    d_radius_dt_m_s: f64,
    d_b_ext_dt_t_s: f64,
    n_turns: u32,
) -> Result<f64, String> {
    let turns = require_positive_turns(n_turns)? as f64;
    let radius = require_positive("separatrix_radius_m", separatrix_radius_m)?;
    let b_ext = require_finite("b_ext_t", b_ext_t)?;
    let d_radius_dt = require_finite("d_radius_dt_m_s", d_radius_dt_m_s)?;
    let d_b_ext_dt = require_finite("d_b_ext_dt_t_s", d_b_ext_dt_t_s)?;
    Ok(-turns
        * std::f64::consts::PI
        * (radius * radius * d_b_ext_dt + 2.0 * b_ext * radius * d_radius_dt))
}

pub fn integrated_recovery_energy(
    trajectory: &[FaradayRecoveryTrajectoryPoint],
    n_turns: u32,
    coil_resistance_ohm: f64,
    compression_work_j: Option<f64>,
    coil_source_work_j: Option<f64>,
    compression_flux_budget: Option<FaradayCompressionFluxBudget>,
    budget_tolerance: f64,
) -> Result<FaradayRecoveryReport, String> {
    let turns = require_positive_turns(n_turns)?;
    let resistance = require_positive("coil_resistance_ohm", coil_resistance_ohm)?;
    let tolerance = require_positive("budget_tolerance", budget_tolerance)?;
    if trajectory.len() < 2 {
        return Err("trajectory must contain at least two samples".to_string());
    }
    validate_trajectory(trajectory)?;

    let time = trajectory.iter().map(|point| point.t_s).collect::<Vec<_>>();
    let radius = trajectory
        .iter()
        .map(|point| point.separatrix_radius_m)
        .collect::<Vec<_>>();
    let field = trajectory
        .iter()
        .map(|point| point.b_ext_t)
        .collect::<Vec<_>>();
    let d_radius_dt = trajectory_derivative(trajectory, true, &time, &radius)?;
    let d_b_ext_dt = trajectory_derivative(trajectory, false, &time, &field)?;

    let mut samples = Vec::with_capacity(trajectory.len());
    for index in 0..trajectory.len() {
        let flux = magnetic_flux_wb(radius[index], field[index])?;
        let emf = faraday_back_emf_from_values(
            radius[index],
            field[index],
            d_radius_dt[index],
            d_b_ext_dt[index],
            turns,
        )?;
        let current = emf / resistance;
        let power = emf * emf / resistance;
        samples.push(FaradayRecoverySample {
            t_s: time[index],
            separatrix_radius_m: radius[index],
            b_ext_t: field[index],
            d_radius_dt_m_s: d_radius_dt[index],
            d_b_ext_dt_t_s: d_b_ext_dt[index],
            magnetic_flux_wb: flux,
            back_emf_v: emf,
            load_current_a: current,
            load_power_w: power,
        });
    }

    let powers = samples
        .iter()
        .map(|sample| sample.load_power_w)
        .collect::<Vec<_>>();
    let recovered_energy_j = trapezoid(&time, &powers);
    let compression_budget = evaluate_budget(
        recovered_energy_j,
        compression_work_j,
        tolerance,
        "compression_work_j",
        "blocked_missing_compression_work",
    )?;
    let source_budget = evaluate_budget(
        recovered_energy_j,
        coil_source_work_j,
        tolerance,
        "coil_source_work_j",
        "blocked_missing_coil_source_work",
    )?;
    let flux_budget = evaluate_compression_flux_budget(compression_flux_budget)?;

    let flux_initial_wb = samples[0].magnetic_flux_wb;
    let flux_final_wb = samples[samples.len() - 1].magnetic_flux_wb;
    let max_abs_back_emf_v = samples
        .iter()
        .map(|sample| sample.back_emf_v.abs())
        .fold(0.0_f64, f64::max);
    let max_abs_load_current_a = samples
        .iter()
        .map(|sample| sample.load_current_a.abs())
        .fold(0.0_f64, f64::max);

    Ok(FaradayRecoveryReport {
        samples,
        n_turns: turns,
        coil_resistance_ohm: resistance,
        recovered_energy_j,
        flux_initial_wb,
        flux_final_wb,
        max_abs_back_emf_v,
        max_abs_load_current_a,
        compression_work_j: compression_budget.work_j,
        energy_budget_relative_error: compression_budget.relative_error,
        energy_budget_passed: compression_budget.passed,
        budget_claim_status: compression_budget.status,
        coil_source_work_j: source_budget.work_j,
        source_energy_budget_relative_error: source_budget.relative_error,
        source_energy_budget_passed: source_budget.passed,
        source_budget_claim_status: source_budget.status,
        compression_flux_budget: flux_budget.budget,
        compression_flux_budget_passed: flux_budget.passed,
        compression_flux_budget_claim_status: flux_budget.status,
    })
}

pub fn faraday_trajectory_from_pulsed_compression(
    states: &[PulsedCompressionState],
) -> Result<Vec<FaradayRecoveryTrajectoryPoint>, String> {
    if states.len() < 2 {
        return Err("pulsed-compression trajectory must contain at least two states".to_string());
    }
    states
        .iter()
        .map(|state| {
            require_finite("state.t_s", state.t_s)?;
            require_positive("state.r_s_m", state.r_s_m)?;
            require_finite("state.b_ext_t", state.b_ext_t)?;
            require_finite("state.d_r_s_dt_m_s", state.d_r_s_dt_m_s)?;
            Ok(FaradayRecoveryTrajectoryPoint {
                t_s: state.t_s,
                separatrix_radius_m: state.r_s_m,
                b_ext_t: state.b_ext_t,
                d_radius_dt_m_s: Some(state.d_r_s_dt_m_s),
                d_b_ext_dt_t_s: None,
            })
        })
        .collect()
}

pub fn compression_work_from_pulsed_compression(
    states: &[PulsedCompressionState],
) -> Result<f64, String> {
    if states.len() < 2 {
        return Err("pulsed-compression trajectory must contain at least two states".to_string());
    }
    let work = states
        .last()
        .map(|state| state.compression_work_j)
        .unwrap_or(f64::NAN);
    require_positive("compression_work_j", work)
}

pub fn compression_flux_budget_from_pulsed_compression(
    states: &[PulsedCompressionState],
) -> Result<FaradayCompressionFluxBudget, String> {
    if states.len() < 2 {
        return Err("pulsed-compression trajectory must contain at least two states".to_string());
    }
    let post_initial_states = &states[1..];
    let source_increment_checksum = post_initial_states
        .iter()
        .map(|state| state.flux_state.source_increment_checksum)
        .sum::<f64>();
    let damping_decrement_checksum = post_initial_states
        .iter()
        .map(|state| state.flux_state.damping_decrement_checksum)
        .sum::<f64>();
    let update_residual_abs_max = post_initial_states
        .iter()
        .map(|state| state.flux_state.update_residual_abs_max)
        .fold(0.0_f64, f64::max);
    let budget_claim_status = if post_initial_states
        .iter()
        .all(|state| state.flux_state.budget_claim_status == "passed")
    {
        "passed"
    } else {
        "failed"
    };
    let first_coupling_status = post_initial_states[0].flux_state.coupling_status.as_str();
    let coupling_status = if post_initial_states
        .iter()
        .all(|state| state.flux_state.coupling_status == first_coupling_status)
    {
        first_coupling_status
    } else {
        "mixed_flux_coupling_status"
    };
    let budget = FaradayCompressionFluxBudget {
        source_increment_checksum,
        damping_decrement_checksum,
        update_residual_abs_max,
        budget_claim_status: budget_claim_status.to_string(),
        coupling_status: coupling_status.to_string(),
    };
    validate_compression_flux_budget(&budget)?;
    Ok(budget)
}

pub fn faraday_trajectory_from_voltage_driven_compression(
    result: &VoltageDrivenPulsedCompressionResult,
) -> Result<Vec<FaradayRecoveryTrajectoryPoint>, String> {
    faraday_trajectory_from_pulsed_compression(&result.compression)
}

pub fn compression_work_from_voltage_driven_compression(
    result: &VoltageDrivenPulsedCompressionResult,
) -> Result<f64, String> {
    compression_work_from_pulsed_compression(&result.compression)
}

pub fn compression_flux_budget_from_voltage_driven_compression(
    result: &VoltageDrivenPulsedCompressionResult,
) -> Result<FaradayCompressionFluxBudget, String> {
    compression_flux_budget_from_pulsed_compression(&result.compression)
}

pub fn coil_source_work_from_voltage_driven_compression(
    result: &VoltageDrivenPulsedCompressionResult,
) -> Result<f64, String> {
    if result.coil_circuit.len() < 2 {
        return Err("voltage-driven coil circuit must contain at least two samples".to_string());
    }
    let work = result
        .coil_circuit
        .last()
        .map(|state| state.source_work_j)
        .unwrap_or(f64::NAN);
    require_positive("coil_source_work_j", work)
}

struct FluxBudgetEvaluation {
    budget: Option<FaradayCompressionFluxBudget>,
    passed: Option<bool>,
    status: String,
}

fn evaluate_budget(
    recovered_energy_j: f64,
    supplied_work_j: Option<f64>,
    tolerance: f64,
    work_name: &str,
    missing_status: &str,
) -> Result<BudgetEvaluation, String> {
    match supplied_work_j {
        None => Ok(BudgetEvaluation {
            work_j: None,
            relative_error: None,
            passed: None,
            status: missing_status.to_string(),
        }),
        Some(work_value) => {
            let work = require_positive(work_name, work_value)?;
            let scale = work.abs().max(recovered_energy_j.abs()).max(f64::EPSILON);
            let error = (recovered_energy_j - work).abs() / scale;
            let passed = error <= tolerance;
            Ok(BudgetEvaluation {
                work_j: Some(work),
                relative_error: Some(error),
                passed: Some(passed),
                status: if passed {
                    "passed".to_string()
                } else {
                    "failed".to_string()
                },
            })
        }
    }
}

fn evaluate_compression_flux_budget(
    budget: Option<FaradayCompressionFluxBudget>,
) -> Result<FluxBudgetEvaluation, String> {
    match budget {
        None => Ok(FluxBudgetEvaluation {
            budget: None,
            passed: None,
            status: "blocked_missing_compression_flux_budget".to_string(),
        }),
        Some(value) => {
            validate_compression_flux_budget(&value)?;
            let passed = value.budget_claim_status == "passed";
            let status = value.budget_claim_status.clone();
            Ok(FluxBudgetEvaluation {
                budget: Some(value),
                passed: Some(passed),
                status,
            })
        }
    }
}

fn validate_compression_flux_budget(budget: &FaradayCompressionFluxBudget) -> Result<(), String> {
    require_finite(
        "compression_flux_source_increment_checksum",
        budget.source_increment_checksum,
    )?;
    require_finite(
        "compression_flux_damping_decrement_checksum",
        budget.damping_decrement_checksum,
    )?;
    require_finite(
        "compression_flux_update_residual_abs_max",
        budget.update_residual_abs_max,
    )?;
    if budget.budget_claim_status.is_empty() {
        return Err("compression_flux_budget.budget_claim_status must be non-empty".to_string());
    }
    if budget.coupling_status.is_empty() {
        return Err("compression_flux_budget.coupling_status must be non-empty".to_string());
    }
    Ok(())
}

fn validate_trajectory(trajectory: &[FaradayRecoveryTrajectoryPoint]) -> Result<(), String> {
    for point in trajectory {
        require_finite("t_s", point.t_s)?;
        require_positive("separatrix_radius_m", point.separatrix_radius_m)?;
        require_finite("b_ext_t", point.b_ext_t)?;
        if let Some(value) = point.d_radius_dt_m_s {
            require_finite("d_radius_dt_m_s", value)?;
        }
        if let Some(value) = point.d_b_ext_dt_t_s {
            require_finite("d_b_ext_dt_t_s", value)?;
        }
    }
    for pair in trajectory.windows(2) {
        if pair[1].t_s <= pair[0].t_s {
            return Err("trajectory time samples must be strictly increasing".to_string());
        }
    }
    Ok(())
}

fn trajectory_derivative(
    trajectory: &[FaradayRecoveryTrajectoryPoint],
    radius_field: bool,
    time: &[f64],
    values: &[f64],
) -> Result<Vec<f64>, String> {
    let supplied = trajectory
        .iter()
        .map(|point| {
            if radius_field {
                point.d_radius_dt_m_s
            } else {
                point.d_b_ext_dt_t_s
            }
        })
        .collect::<Vec<_>>();
    let supplied_count = supplied.iter().filter(|value| value.is_some()).count();
    if supplied_count > 0 && supplied_count != supplied.len() {
        return Err(
            "trajectory derivative samples must be all supplied or all omitted".to_string(),
        );
    }
    if supplied_count == supplied.len() {
        return supplied
            .into_iter()
            .map(|value| require_finite("trajectory derivative", value.unwrap_or(f64::NAN)))
            .collect();
    }

    let n = values.len();
    let mut derivative = vec![0.0; n];
    if n == 2 {
        let slope = (values[1] - values[0]) / (time[1] - time[0]);
        derivative[0] = slope;
        derivative[1] = slope;
        return Ok(derivative);
    }
    derivative[0] = (values[1] - values[0]) / (time[1] - time[0]);
    derivative[n - 1] = (values[n - 1] - values[n - 2]) / (time[n - 1] - time[n - 2]);
    for index in 1..(n - 1) {
        derivative[index] =
            (values[index + 1] - values[index - 1]) / (time[index + 1] - time[index - 1]);
    }
    Ok(derivative)
}

fn trapezoid(x: &[f64], y: &[f64]) -> f64 {
    x.windows(2)
        .zip(y.windows(2))
        .map(|(x_pair, y_pair)| 0.5 * (y_pair[0] + y_pair[1]) * (x_pair[1] - x_pair[0]))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::{
        coil_source_work_from_voltage_driven_compression,
        compression_flux_budget_from_pulsed_compression,
        compression_flux_budget_from_voltage_driven_compression,
        compression_work_from_pulsed_compression, compression_work_from_voltage_driven_compression,
        faraday_back_emf_from_values, faraday_trajectory_from_pulsed_compression,
        faraday_trajectory_from_voltage_driven_compression, integrated_recovery_energy,
        magnetic_flux_wb, FaradayRecoveryTrajectoryPoint,
    };
    use crate::compression::{
        plasma_volume_m3, run_pulsed_compression, run_voltage_driven_pulsed_compression,
        CoilGeometry, PulsedCompressionConfig, PulsedCompressionState,
    };

    #[test]
    fn constant_radius_and_field_have_zero_emf() {
        let emf = faraday_back_emf_from_values(0.2, 20.0, 0.0, 0.0, 12).expect("valid emf");
        assert_eq!(emf, 0.0);
    }

    #[test]
    fn closed_form_constant_field_expansion_matches_formula() {
        let emf = faraday_back_emf_from_values(0.2, 20.0, 1.5e4, 0.0, 8).expect("valid emf");
        let expected = -8.0 * std::f64::consts::PI * (2.0 * 20.0 * 0.2 * 1.5e4);
        assert!((emf - expected).abs() / expected.abs() < 1.0e-15);
    }

    #[test]
    fn integrated_energy_matches_linear_radius_case() {
        let turns = 6;
        let resistance = 0.08;
        let b_ext = 20.0;
        let radius_0 = 0.15;
        let speed = 4.0e3;
        let duration = 1.0e-6;
        let trajectory = (0..257)
            .map(|index| {
                let t_s = duration * index as f64 / 256.0;
                FaradayRecoveryTrajectoryPoint {
                    t_s,
                    separatrix_radius_m: radius_0 + speed * t_s,
                    b_ext_t: b_ext,
                    d_radius_dt_m_s: None,
                    d_b_ext_dt_t_s: None,
                }
            })
            .collect::<Vec<_>>();
        let report =
            integrated_recovery_energy(&trajectory, turns, resistance, None, None, None, 0.01)
                .expect("valid report");
        let coefficient = turns as f64 * std::f64::consts::PI * 2.0 * b_ext * speed;
        let expected = coefficient * coefficient / resistance
            * ((radius_0 + speed * duration).powi(3) - radius_0.powi(3))
            / (3.0 * speed);
        assert!((report.recovered_energy_j - expected).abs() / expected < 2.0e-5);
        assert_eq!(
            report.budget_claim_status,
            "blocked_missing_compression_work"
        );
        assert_eq!(report.energy_budget_passed, None);
        assert_eq!(
            report.source_budget_claim_status,
            "blocked_missing_coil_source_work"
        );
        assert_eq!(report.source_energy_budget_passed, None);
        assert_eq!(
            report.compression_flux_budget_claim_status,
            "blocked_missing_compression_flux_budget"
        );
        assert_eq!(report.compression_flux_budget_passed, None);
    }

    #[test]
    fn budget_status_reports_failure_when_supplied_work_does_not_match() {
        let trajectory = [
            FaradayRecoveryTrajectoryPoint {
                t_s: 0.0,
                separatrix_radius_m: 0.2,
                b_ext_t: 20.0,
                d_radius_dt_m_s: Some(0.0),
                d_b_ext_dt_t_s: Some(0.0),
            },
            FaradayRecoveryTrajectoryPoint {
                t_s: 1.0e-6,
                separatrix_radius_m: 0.2,
                b_ext_t: 20.0,
                d_radius_dt_m_s: Some(0.0),
                d_b_ext_dt_t_s: Some(0.0),
            },
        ];
        let report =
            integrated_recovery_energy(&trajectory, 4, 0.1, Some(1.0e-12), None, None, 0.01)
                .expect("valid report");
        assert_eq!(report.recovered_energy_j, 0.0);
        assert_eq!(report.energy_budget_passed, Some(false));
        assert_eq!(report.budget_claim_status, "failed");
        assert_eq!(
            report.source_budget_claim_status,
            "blocked_missing_coil_source_work"
        );
        assert_eq!(
            report.compression_flux_budget_claim_status,
            "blocked_missing_compression_flux_budget"
        );
    }

    #[test]
    fn invalid_inputs_fail_closed() {
        assert!(faraday_back_emf_from_values(0.2, 20.0, 0.0, 0.0, 0).is_err());
        assert!(magnetic_flux_wb(0.0, 20.0).is_err());
        let one = [FaradayRecoveryTrajectoryPoint {
            t_s: 0.0,
            separatrix_radius_m: 0.2,
            b_ext_t: 20.0,
            d_radius_dt_m_s: None,
            d_b_ext_dt_t_s: None,
        }];
        assert!(integrated_recovery_energy(&one, 2, 0.1, None, None, None, 0.01).is_err());
    }

    fn compression_coil() -> CoilGeometry {
        CoilGeometry {
            n_turns: 80,
            l_coil_m: 1.0,
            r_coil_m: 0.35,
            l_inductance_h: 2.0e-6,
            r_resistance_ohm: 0.02,
            bank_voltage_max_v: 20_000.0,
        }
    }

    fn compression_config() -> PulsedCompressionConfig {
        PulsedCompressionConfig {
            coil: compression_coil(),
            plasma_mass_kg: 2.0e-5,
            plasma_length_m: 1.0,
            gamma: 5.0 / 3.0,
            radial_loss_time_s: None,
            tau_psi_s: f64::INFINITY,
            e_theta_v_m: None,
            j_theta_a_m2: None,
            z_eff: 1.0,
            ln_lambda: 17.0,
            min_radius_m: 1.0e-4,
        }
    }

    fn compression_initial() -> PulsedCompressionState {
        let density = 2.0e21;
        let t_i = 10_000.0;
        let t_e = 5_000.0;
        let radius = 0.2;
        let field = crate::compression::coil_field_t(&compression_coil(), 2.0e5).unwrap();
        let volume = plasma_volume_m3(radius, 1.0).unwrap();
        let thermal_energy_j = 1.5 * density * volume * (t_i + t_e) * 1.602_176_634e-19;
        PulsedCompressionState {
            t_s: 0.0,
            r_s_m: radius,
            d_r_s_dt_m_s: 0.0,
            t_i_ev: t_i,
            t_e_ev: t_e,
            density_m3: density,
            beta: 0.0,
            b_ext_t: field,
            internal_pressure_pa: density * (t_i + t_e) * 1.602_176_634e-19,
            external_magnetic_pressure_pa: field * field / (2.0 * 4.0e-7 * std::f64::consts::PI),
            thermal_energy_j,
            compression_work_j: 0.0,
            radiated_loss_j: 0.0,
            energy_balance_residual: 0.0,
            flux_state: Default::default(),
        }
    }

    #[test]
    fn pulsed_compression_sidecar_evaluates_budget_status() {
        let states = run_pulsed_compression(
            compression_initial(),
            &compression_config(),
            5.0e5,
            1.0e-9,
            16,
        )
        .expect("valid compression states");
        let trajectory =
            faraday_trajectory_from_pulsed_compression(&states).expect("valid trajectory");
        let compression_work =
            compression_work_from_pulsed_compression(&states).expect("positive compression work");
        let compression_flux_budget =
            compression_flux_budget_from_pulsed_compression(&states).expect("valid flux budget");
        let report = integrated_recovery_energy(
            &trajectory,
            80,
            0.02,
            Some(compression_work),
            None,
            Some(compression_flux_budget.clone()),
            0.01,
        )
        .expect("valid recovery report");

        assert_eq!(trajectory.len(), states.len());
        assert_eq!(
            trajectory.last().unwrap().separatrix_radius_m,
            states.last().unwrap().r_s_m
        );
        assert_eq!(
            trajectory.last().unwrap().d_radius_dt_m_s,
            Some(states.last().unwrap().d_r_s_dt_m_s)
        );
        assert!(compression_work > 0.0);
        assert_eq!(report.compression_work_j, Some(compression_work));
        assert!(report
            .energy_budget_relative_error
            .unwrap_or(f64::NAN)
            .is_finite());
        assert!(matches!(
            report.budget_claim_status.as_str(),
            "passed" | "failed"
        ));
        assert_eq!(
            report.source_budget_claim_status,
            "blocked_missing_coil_source_work"
        );
        assert_eq!(compression_flux_budget.budget_claim_status, "passed");
        assert!(compression_flux_budget.update_residual_abs_max <= 1.0e-12);
        assert_eq!(
            report.compression_flux_budget,
            Some(compression_flux_budget)
        );
        assert_eq!(report.compression_flux_budget_passed, Some(true));
        assert_eq!(report.compression_flux_budget_claim_status, "passed");
    }

    #[test]
    fn voltage_driven_sidecars_evaluate_source_budget_status() {
        let result = run_voltage_driven_pulsed_compression(
            compression_initial(),
            &compression_config(),
            20_000.0,
            1.0e-9,
            32,
            5.0e5,
        )
        .expect("valid voltage-driven result");
        let trajectory = faraday_trajectory_from_voltage_driven_compression(&result)
            .expect("valid voltage-driven trajectory");
        let compression_work = compression_work_from_voltage_driven_compression(&result)
            .expect("positive compression work");
        let compression_flux_budget =
            compression_flux_budget_from_voltage_driven_compression(&result)
                .expect("valid flux budget");
        let source_work = coil_source_work_from_voltage_driven_compression(&result)
            .expect("positive source work");
        let report = integrated_recovery_energy(
            &trajectory,
            80,
            0.02,
            Some(compression_work),
            Some(source_work),
            Some(compression_flux_budget),
            0.01,
        )
        .expect("valid recovery report");

        assert_eq!(trajectory.len(), result.compression.len());
        assert_eq!(
            compression_work,
            result.compression.last().unwrap().compression_work_j
        );
        assert_eq!(
            source_work,
            result.coil_circuit.last().unwrap().source_work_j
        );
        assert!(report
            .source_energy_budget_relative_error
            .unwrap_or(f64::NAN)
            .is_finite());
        assert!(matches!(
            report.source_budget_claim_status.as_str(),
            "passed" | "failed"
        ));
        assert_eq!(report.compression_flux_budget_passed, Some(true));
        assert_eq!(report.compression_flux_budget_claim_status, "passed");
    }

    #[test]
    fn failed_compression_flux_budget_is_propagated() {
        let mut states = run_pulsed_compression(
            compression_initial(),
            &compression_config(),
            5.0e5,
            1.0e-9,
            4,
        )
        .expect("valid compression states");
        let final_state = states.last_mut().expect("state exists");
        final_state.flux_state.budget_claim_status = "failed".to_string();
        final_state.flux_state.update_residual_abs_max = 1.0e-3;
        let trajectory =
            faraday_trajectory_from_pulsed_compression(&states).expect("valid trajectory");
        let compression_work =
            compression_work_from_pulsed_compression(&states).expect("positive compression work");
        let compression_flux_budget =
            compression_flux_budget_from_pulsed_compression(&states).expect("valid flux budget");
        let report = integrated_recovery_energy(
            &trajectory,
            80,
            0.02,
            Some(compression_work),
            None,
            Some(compression_flux_budget),
            0.01,
        )
        .expect("valid recovery report");

        assert_eq!(report.compression_flux_budget_passed, Some(false));
        assert_eq!(report.compression_flux_budget_claim_status, "failed");
    }
}
