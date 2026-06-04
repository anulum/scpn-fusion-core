// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — FRC Rigid-Rotor Metrics
//! Emit benchmark metrics for the Rust FRC analytical solver.

use fusion_physics::frc::{solve_frc_equilibrium, RigidRotorFrcInputs};
use ndarray::Array1;
use std::time::Instant;

fn linspace(start: f64, end: f64, n: usize) -> Array1<f64> {
    let step = (end - start) / (n as f64 - 1.0);
    Array1::from_iter((0..n).map(|idx| start + idx as f64 * step))
}

fn checksum(values: &Array1<f64>) -> f64 {
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx as f64 + 1.0) * value)
        .sum()
}

fn main() {
    let grids: Vec<usize> = std::env::args()
        .skip(1)
        .map(|value| value.parse::<usize>().expect("grid size must be usize"))
        .collect();
    let grids = if grids.is_empty() {
        vec![64, 256, 1024]
    } else {
        grids
    };
    let inputs = RigidRotorFrcInputs {
        n0: 25.0 / (2.0 * 1.2566370614359173e-6) / (15_000.0 * 1.602176634e-19),
        t_i_ev: 10_000.0,
        t_e_ev: 5_000.0,
        theta_dot: 0.0,
        r_s: 0.20,
        b_ext: 5.0,
        delta: Some(0.02),
    };

    print!("{{\"language\":\"rust\",\"solver\":\"fusion-physics-frc\",\"grids\":[");
    for (idx, n) in grids.iter().enumerate() {
        let rho = linspace(0.0, 0.4, *n);
        let start = Instant::now();
        let state = solve_frc_equilibrium(&inputs, &rho, 1.0e-10).expect("valid FRC state");
        let wall_time_s = start.elapsed().as_secs_f64();
        if idx > 0 {
            print!(",");
        }
        print!(
            "{{\"grid_points\":{},\"wall_time_s\":{:.12e},\"r_null_m\":{:.17e},\"target_separatrix_radius_m\":{:.17e},\"separatrix_radius_error_m\":{:.17e},\"field_reversal_passed\":{},\"s_parameter\":{:.17e},\"energy_j_per_m\":{:.17e},\"pressure_balance_ratio\":{:.17e},\"pressure_balance_residual_linf\":{:.17e},\"peak_pressure_pa\":{:.17e},\"density_peak_m3\":{:.17e},\"input_density_m3\":{:.17e},\"central_density_residual_m3\":{:.17e},\"central_density_relative_error\":{:.17e},\"beta_peak\":{:.17e},\"beta_separatrix_average\":{:.17e},\"particle_line_density_m1\":{:.17e},\"input_thermal_pressure_pa\":{:.17e},\"thermal_pressure_ratio\":{:.17e},\"flux_derivative_residual_linf\":{:.17e},\"peak_j_theta_a_m2\":{:.17e},\"ampere_residual_linf\":{:.17e},\"force_balance_residual_linf\":{:.17e},\"b_z_checksum\":{:.17e},\"j_theta_checksum\":{:.17e},\"psi_checksum\":{:.17e},\"p_checksum\":{:.17e},\"density_checksum\":{:.17e},\"beta_checksum\":{:.17e}}}",
            n,
            wall_time_s,
            state.r_null,
            state.target_separatrix_radius_m,
            state.separatrix_radius_error_m,
            state.field_reversal_passed,
            state.s_parameter,
            state.energy_j,
            state.pressure_balance_ratio,
            state.pressure_balance_residual_linf,
            state.peak_pressure_pa,
            state.density_peak_m3,
            state.input_density_m3,
            state.central_density_residual_m3,
            state.central_density_relative_error,
            state.beta_peak,
            state.beta_separatrix_average,
            state.particle_line_density_m1,
            state.input_thermal_pressure_pa,
            state.thermal_pressure_ratio,
            state.flux_derivative_residual_linf,
            state.peak_j_theta_a_m2,
            state.ampere_residual_linf,
            state.force_balance_residual_linf,
            checksum(&state.b_z),
            checksum(&state.j_theta),
            checksum(&state.psi),
            checksum(&state.p),
            checksum(&state.density_m3),
            checksum(&state.beta)
        );
    }
    println!("]}}");
}
