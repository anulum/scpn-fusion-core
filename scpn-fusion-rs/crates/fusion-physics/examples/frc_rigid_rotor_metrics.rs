// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — FRC Rigid Rotor Metrics
#![recursion_limit = "256"]
//! Emit deterministic FRC rigid-rotor benchmark metrics as JSON.

use fusion_physics::frc::{solve_frc_equilibrium, FrcEquilibriumState, RigidRotorFrcInputs};
use ndarray::Array1;
use serde_json::{json, Value};
use std::time::Instant;

const ELEMENTARY_CHARGE_C: f64 = 1.602_176_634e-19;
const MU_0: f64 = 4.0e-7 * std::f64::consts::PI;
const T_I_EV: f64 = 10_000.0;
const T_E_EV: f64 = 5_000.0;
const TOLERANCE: f64 = 1.0e-10;

#[derive(Clone, Copy)]
struct ParityCase {
    delta: Option<f64>,
    grid_points: usize,
    b_ext: f64,
    r_s: f64,
}

const PARITY_CASES: [ParityCase; 16] = [
    ParityCase {
        delta: Some(0.012),
        grid_points: 33,
        b_ext: 2.75,
        r_s: 0.16,
    },
    ParityCase {
        delta: Some(0.014),
        grid_points: 65,
        b_ext: 3.25,
        r_s: 0.18,
    },
    ParityCase {
        delta: Some(0.016),
        grid_points: 129,
        b_ext: 3.75,
        r_s: 0.19,
    },
    ParityCase {
        delta: Some(0.018),
        grid_points: 257,
        b_ext: 4.25,
        r_s: 0.20,
    },
    ParityCase {
        delta: Some(0.020),
        grid_points: 33,
        b_ext: 4.75,
        r_s: 0.21,
    },
    ParityCase {
        delta: Some(0.022),
        grid_points: 65,
        b_ext: 5.25,
        r_s: 0.22,
    },
    ParityCase {
        delta: Some(0.024),
        grid_points: 129,
        b_ext: 5.75,
        r_s: 0.23,
    },
    ParityCase {
        delta: Some(0.026),
        grid_points: 257,
        b_ext: 6.25,
        r_s: 0.24,
    },
    ParityCase {
        delta: Some(0.028),
        grid_points: 33,
        b_ext: 6.75,
        r_s: 0.25,
    },
    ParityCase {
        delta: Some(0.030),
        grid_points: 65,
        b_ext: 7.25,
        r_s: 0.26,
    },
    ParityCase {
        delta: Some(0.032),
        grid_points: 129,
        b_ext: 7.75,
        r_s: 0.27,
    },
    ParityCase {
        delta: Some(0.034),
        grid_points: 257,
        b_ext: 8.25,
        r_s: 0.28,
    },
    ParityCase {
        delta: None,
        grid_points: 33,
        b_ext: 3.50,
        r_s: 0.17,
    },
    ParityCase {
        delta: None,
        grid_points: 65,
        b_ext: 5.00,
        r_s: 0.20,
    },
    ParityCase {
        delta: None,
        grid_points: 129,
        b_ext: 6.50,
        r_s: 0.24,
    },
    ParityCase {
        delta: None,
        grid_points: 257,
        b_ext: 8.00,
        r_s: 0.28,
    },
];

fn pressure_matched_density_m3(b_ext: f64) -> f64 {
    b_ext.powi(2) / (2.0 * MU_0) / ((T_I_EV + T_E_EV) * ELEMENTARY_CHARGE_C)
}

fn inputs(delta: Option<f64>, b_ext: f64, r_s: f64) -> RigidRotorFrcInputs {
    RigidRotorFrcInputs {
        n0: pressure_matched_density_m3(b_ext),
        t_i_ev: T_I_EV,
        t_e_ev: T_E_EV,
        theta_dot: 0.0,
        r_s,
        b_ext,
        delta,
    }
}

fn checksum(values: &Array1<f64>) -> f64 {
    values.iter().sum::<f64>()
}

fn rho_grid(grid_points: usize, r_s: f64) -> Array1<f64> {
    Array1::linspace(0.0, 2.0 * r_s, grid_points)
}

fn metric_row(
    label: &str,
    grid_points: usize,
    elapsed_s: f64,
    state: &FrcEquilibriumState,
    case: Option<&ParityCase>,
) -> Value {
    json!({
        "case_label": label,
        "delta_m": state.delta,
        "input_delta_m": case.and_then(|item| item.delta),
        "input_b_ext_t": case.map(|item| item.b_ext),
        "input_separatrix_radius_m": case.map(|item| item.r_s),
        "grid_points": grid_points,
        "wall_time_s": elapsed_s,
        "r_null_m": state.r_null,
        "target_separatrix_radius_m": state.target_separatrix_radius_m,
        "separatrix_radius_error_m": state.separatrix_radius_error_m,
        "separatrix_index": state.separatrix_index,
        "field_reversal_passed": state.field_reversal_passed,
        "s_parameter": state.s_parameter,
        "energy_j_per_m": state.energy_j,
        "converged": state.converged,
        "residual": state.residual,
        "psi_axis_wb": state.psi_axis_wb,
        "psi_separatrix_wb": state.psi_separatrix_wb,
        "psi_normalized_axis_error": state.psi_normalized_axis_error,
        "psi_normalized_separatrix": state.psi_normalized_separatrix,
        "psi_normalized_separatrix_error": state.psi_normalized_separatrix_error,
        "psi_normalized_residual_linf": state.psi_normalized_residual_linf,
        "psi_normalized_monotonic_passed": state.psi_normalized_monotonic_passed,
        "psi_normalized_bounds_passed": state.psi_normalized_bounds_passed,
        "pressure_balance_ratio": state.pressure_balance_ratio,
        "pressure_balance_residual_linf": state.pressure_balance_residual_linf,
        "pressure_balance_residual_l2": state.pressure_balance_residual_l2,
        "pressure_gradient_residual_linf": state.pressure_gradient_residual_linf,
        "pressure_gradient_residual_l2": state.pressure_gradient_residual_l2,
        "peak_pressure_pa": state.peak_pressure_pa,
        "density_peak_m3": state.density_peak_m3,
        "input_density_m3": state.input_density_m3,
        "central_density_residual_m3": state.central_density_residual_m3,
        "central_density_relative_error": state.central_density_relative_error,
        "beta_peak": state.beta_peak,
        "beta_separatrix_average": state.beta_separatrix_average,
        "particle_line_density_m1": state.particle_line_density_m1,
        "separatrix_pressure_energy_j_m": state.separatrix_pressure_energy_j_m,
        "separatrix_magnetic_deficit_energy_j_m": state.separatrix_magnetic_deficit_energy_j_m,
        "separatrix_energy_closure_relative_error": state.separatrix_energy_closure_relative_error,
        "input_thermal_pressure_pa": state.input_thermal_pressure_pa,
        "thermal_pressure_ratio": state.thermal_pressure_ratio,
        "flux_derivative_residual_linf": state.flux_derivative_residual_linf,
        "flux_derivative_residual_l2": state.flux_derivative_residual_l2,
        "ampere_residual_linf": state.ampere_residual_linf,
        "ampere_residual_l2": state.ampere_residual_l2,
        "peak_j_theta_a_m2": state.peak_j_theta_a_m2,
        "separatrix_bz_gradient_t_m": state.separatrix_bz_gradient_t_m,
        "separatrix_expected_bz_gradient_t_m": state.separatrix_expected_bz_gradient_t_m,
        "separatrix_gradient_relative_error": state.separatrix_gradient_relative_error,
        "separatrix_current_density_a_m2": state.separatrix_current_density_a_m2,
        "separatrix_expected_current_density_a_m2": state.separatrix_expected_current_density_a_m2,
        "separatrix_current_density_relative_error": state.separatrix_current_density_relative_error,
        "sheet_current_integral_a_m": state.sheet_current_integral_a_m,
        "expected_sheet_current_integral_a_m": state.expected_sheet_current_integral_a_m,
        "sheet_current_integral_relative_error": state.sheet_current_integral_relative_error,
        "force_balance_residual_linf": state.force_balance_residual_linf,
        "force_balance_residual_l2": state.force_balance_residual_l2,
        "psi_checksum": checksum(&state.psi),
        "psi_normalized_checksum": checksum(&state.psi_normalized),
        "bz_checksum": checksum(&state.b_z),
        "jtheta_checksum": checksum(&state.j_theta),
        "pressure_checksum": checksum(&state.p),
        "density_checksum": checksum(&state.density_m3),
    })
}

fn grid_rows() -> Vec<Value> {
    [65_usize, 129, 257, 513]
        .into_iter()
        .map(|grid_points| {
            let case = ParityCase {
                delta: Some(0.02),
                grid_points,
                b_ext: 5.0,
                r_s: 0.20,
            };
            let rho = rho_grid(case.grid_points, case.r_s);
            let start = Instant::now();
            let case_inputs = inputs(case.delta, case.b_ext, case.r_s);
            let state = solve_frc_equilibrium(&case_inputs, &rho, TOLERANCE)
                .expect("FRC grid benchmark case must solve");
            metric_row(
                "grid_convergence_pressure_matched_no_rotation",
                case.grid_points,
                start.elapsed().as_secs_f64(),
                &state,
                Some(&case),
            )
        })
        .collect()
}

fn parameter_case_rows() -> Vec<Value> {
    PARITY_CASES
        .iter()
        .enumerate()
        .map(|(index, case)| {
            let rho = rho_grid(case.grid_points, case.r_s);
            let start = Instant::now();
            let case_inputs = inputs(case.delta, case.b_ext, case.r_s);
            let state = solve_frc_equilibrium(&case_inputs, &rho, TOLERANCE)
                .expect("FRC deterministic parameter case must solve");
            metric_row(
                &format!("mif_frc_no_rotation_parity_{index:02}"),
                case.grid_points,
                start.elapsed().as_secs_f64(),
                &state,
                Some(case),
            )
        })
        .collect()
}

fn main() {
    let payload = json!({
        "surface": "rust_fusion_physics",
        "contract": "Steinhauer no-rotation FRC analytical equilibrium",
        "grids": grid_rows(),
        "parameter_cases": parameter_case_rows(),
    });
    println!("{}", payload);
}
