// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — FRC Rotating BVP Status Probe

use fusion_physics::frc::{
    rotating_frc_bvp_acceptance_status, solve_frc_equilibrium, solve_rotating_frc_equilibrium,
    FrcSolverError, RigidRotorFrcInputs,
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
        theta_dot: 1.0,
        ..base_inputs
    };

    let status = rotating_frc_bvp_acceptance_status();
    let no_rotation = solve_frc_equilibrium(&base_inputs, &rho, 1.0e-10);
    let rotating = solve_rotating_frc_equilibrium(&rotating_inputs, &rho, 1.0e-10);
    let rotating_fail_closed = matches!(rotating, Err(FrcSolverError::RotatingBvpNotImplemented));
    let no_rotation_converged = no_rotation
        .as_ref()
        .map(|state| state.converged && state.field_reversal_passed)
        .unwrap_or(false);
    let no_rotation_residual = no_rotation
        .as_ref()
        .map(|state| state.residual)
        .unwrap_or(f64::NAN);

    println!(
        "{{\"schema\":\"frc-rotating-bvp-rust-status.v1\",\
         \"status\":{},\
         \"accepted_contract\":{},\
         \"rotating_bvp_implemented\":{},\
         \"solver_action\":{},\
         \"required_reference\":{},\
         \"non_closing_references\":{},\
         \"no_rotation_converged\":{},\
         \"no_rotation_residual\":{},\
         \"nonzero_rotation_fail_closed\":{}}}",
        json_string(status.status),
        json_string(status.accepted_contract),
        status.rotating_bvp_implemented,
        json_string(status.solver_action),
        json_string(status.required_reference),
        json_string_array(status.non_closing_references),
        no_rotation_converged,
        no_rotation_residual,
        rotating_fail_closed
    );
}
