// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — FRC Faraday Recovery
//! Classical Faraday back-EMF and recovery-energy contract for MIF/FRC.

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
    let (energy_budget_relative_error, energy_budget_passed, budget_claim_status) =
        match compression_work_j {
            None => (None, None, "blocked_missing_compression_work".to_string()),
            Some(work_value) => {
                let work = require_positive("compression_work_j", work_value)?;
                let scale = work.abs().max(recovered_energy_j.abs()).max(f64::EPSILON);
                let error = (recovered_energy_j - work).abs() / scale;
                let passed = error <= tolerance;
                (
                    Some(error),
                    Some(passed),
                    if passed {
                        "passed".to_string()
                    } else {
                        "failed".to_string()
                    },
                )
            }
        };

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
        compression_work_j,
        energy_budget_relative_error,
        energy_budget_passed,
        budget_claim_status,
    })
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
        faraday_back_emf_from_values, integrated_recovery_energy, magnetic_flux_wb,
        FaradayRecoveryTrajectoryPoint,
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
        let report = integrated_recovery_energy(&trajectory, turns, resistance, None, 0.01)
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
        let report = integrated_recovery_energy(&trajectory, 4, 0.1, Some(1.0e-12), 0.01)
            .expect("valid report");
        assert_eq!(report.recovered_energy_j, 0.0);
        assert_eq!(report.energy_budget_passed, Some(false));
        assert_eq!(report.budget_claim_status, "failed");
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
        assert!(integrated_recovery_energy(&one, 2, 0.1, None, 0.01).is_err());
    }
}
