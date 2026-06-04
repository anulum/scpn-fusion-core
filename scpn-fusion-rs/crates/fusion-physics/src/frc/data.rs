// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — FRC Data Contracts
//! Data contracts for the FRC rigid-rotor analytical solver.

use ndarray::Array1;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Physical inputs for the Steinhauer no-rotation FRC analytical limit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigidRotorFrcInputs {
    pub n0: f64,
    pub t_i_ev: f64,
    pub t_e_ev: f64,
    pub theta_dot: f64,
    pub r_s: f64,
    pub b_ext: f64,
    pub delta: Option<f64>,
}

/// Radial FRC equilibrium state returned by the Rust analytical solver.
#[derive(Debug, Clone)]
pub struct FrcEquilibriumState {
    pub rho: Array1<f64>,
    pub psi: Array1<f64>,
    pub b_z: Array1<f64>,
    pub b_theta: Array1<f64>,
    pub j_theta: Array1<f64>,
    pub p: Array1<f64>,
    pub r_null: f64,
    pub separatrix_index: usize,
    pub s_parameter: f64,
    pub energy_j: f64,
    pub converged: bool,
    pub residual: f64,
    pub delta: f64,
    pub pressure_balance_ratio: f64,
    pub ampere_residual: Array1<f64>,
    pub ampere_residual_linf: f64,
    pub ampere_residual_l2: f64,
    pub peak_j_theta_a_m2: f64,
    pub force_balance_residual: Array1<f64>,
    pub force_balance_residual_linf: f64,
    pub force_balance_residual_l2: f64,
    pub model: &'static str,
}

/// Strict solver errors for invalid input or unimplemented physics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrcSolverError {
    InvalidInput(&'static str),
    RotatingBvpNotImplemented,
}

impl Display for FrcSolverError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            FrcSolverError::InvalidInput(message) => write!(f, "{message}"),
            FrcSolverError::RotatingBvpNotImplemented => {
                write!(f, "rotating rigid-rotor BVP support is not implemented yet")
            }
        }
    }
}

impl Error for FrcSolverError {}
