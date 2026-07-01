// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — FRC Rotating Rigid-Rotor Status Probe

use fusion_physics::frc::{
    rotating_frc_bvp_acceptance_status, solve_frc_equilibrium, solve_rotating_frc_equilibrium,
    RigidRotorFrcInputs,
};
use ndarray::Array1;

const MU_0: f64 = 4.0 * std::f64::consts::PI * 1.0e-7;
const ELEMENTARY_CHARGE_C: f64 = 1.602_176_634e-19;

fn json_string(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

fn json_string_array(values: &[&str]) -> String {
    let quoted: Vec<String> = values.iter().map(|value| json_string(value)).collect();
    format!("[{}]", quoted.join(","))
}

fn main() {
    let t_i_ev = 10_000.0;
    let t_e_ev = 5_000.0;
    let b_ext = 5.0;
    let r_s = 0.20;
    let n0 = b_ext * b_ext / (2.0 * MU_0) / ((t_i_ev + t_e_ev) * ELEMENTARY_CHARGE_C);
    let rho = Array1::linspace(0.0, 0.4, 129);
    let base_inputs = RigidRotorFrcInputs {
        n0,
        t_i_ev,
        t_e_ev,
        theta_dot: 0.0,
        r_s,
        b_ext,
        delta: Some(0.02),
    };
    let rotating_inputs = RigidRotorFrcInputs {
        theta_dot: 3.0e5,
        ..base_inputs
    };
    let small_rotation_inputs = RigidRotorFrcInputs {
        theta_dot: 1.0e3,
        ..base_inputs
    };

    let status = rotating_frc_bvp_acceptance_status();
    let no_rotation =
        solve_frc_equilibrium(&base_inputs, &rho, 1.0e-10).expect("no-rotation solve failed");
    let rotating = solve_rotating_frc_equilibrium(&rotating_inputs, &rho, 1.0e-10)
        .expect("rotating solve failed");
    let small_rotation = solve_rotating_frc_equilibrium(&small_rotation_inputs, &rho, 1.0e-10)
        .expect("small-rotation solve failed");

    let no_rotation_converged = no_rotation.converged && no_rotation.field_reversal_passed;
    let rotating_pressure_non_negative = rotating.p.iter().all(|&pressure| pressure >= 0.0);
    // Reduction: at small omega the rotating pressure returns to the no-rotation
    // contract and the rigid-rotor field is byte-identical.
    let peak = no_rotation.p.iter().cloned().fold(f64::MIN, f64::max);
    let reduction_deviation = small_rotation
        .p
        .iter()
        .zip(no_rotation.p.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
        / peak;
    let field_bit_exact = small_rotation
        .b_z
        .iter()
        .zip(no_rotation.b_z.iter())
        .all(|(a, b)| a == b);
    let no_rotation_reduction_passed = reduction_deviation < 1.0e-4 && field_bit_exact;

    println!(
        "{{\"schema\":\"frc-rotating-bvp-rust-status.v2\",\
         \"status\":{},\
         \"accepted_contract\":{},\
         \"rotating_bvp_implemented\":{},\
         \"rotating_closure_reference\":{},\
         \"solver_action\":{},\
         \"required_reference\":{},\
         \"reduces_to_no_rotation_contract\":{},\
         \"steinhauer_figure3_parity_claimed\":{},\
         \"non_closing_references\":{},\
         \"rotating_model\":{},\
         \"rotation_mach_number\":{},\
         \"rotation_force_balance_residual_linf\":{},\
         \"no_rotation_converged\":{},\
         \"no_rotation_residual\":{},\
         \"rotating_pressure_non_negative\":{},\
         \"no_rotation_reduction_passed\":{}}}",
        json_string(status.status),
        json_string(status.accepted_contract),
        status.rotating_bvp_implemented,
        json_string(status.rotating_closure_reference),
        json_string(status.solver_action),
        json_string(status.required_reference),
        status.reduces_to_no_rotation_contract,
        status.steinhauer_figure3_parity_claimed,
        json_string_array(status.non_closing_references),
        json_string(rotating.model),
        rotating.rotation_mach_number,
        rotating.rotation_force_balance_residual_linf,
        no_rotation_converged,
        no_rotation.residual,
        rotating_pressure_non_negative,
        no_rotation_reduction_passed
    );
}
