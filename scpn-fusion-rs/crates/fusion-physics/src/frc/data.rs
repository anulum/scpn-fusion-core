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
    /// Central particle number density per cubic metre.
    pub n0: f64,
    /// Ion temperature in electronvolts.
    pub t_i_ev: f64,
    /// Electron temperature in electronvolts.
    pub t_e_ev: f64,
    /// Rigid-rotor angular frequency in radians per second.
    pub theta_dot: f64,
    /// Target separatrix radius in metres.
    pub r_s: f64,
    /// External axial magnetic-field strength in tesla.
    pub b_ext: f64,
    /// Optional delta value.
    pub delta: Option<f64>,
}

/// Radial FRC equilibrium state returned by the Rust analytical solver.
#[derive(Debug, Clone)]
pub struct FrcEquilibriumState {
    /// Normalized radial-grid coordinates.
    pub rho: Array1<f64>,
    /// Poloidal-flux samples.
    pub psi: Array1<f64>,
    /// Normalized poloidal-flux samples.
    pub psi_normalized: Array1<f64>,
    /// Axial magnetic-field samples in tesla.
    pub b_z: Array1<f64>,
    /// Azimuthal magnetic-field samples in tesla.
    pub b_theta: Array1<f64>,
    /// Azimuthal current-density samples in amperes per square metre.
    pub j_theta: Array1<f64>,
    /// Plasma-pressure samples in pascals.
    pub p: Array1<f64>,
    /// Particle number density per cubic metre.
    pub density_m3: Array1<f64>,
    /// Ratio of plasma pressure to magnetic pressure.
    pub beta: Array1<f64>,
    /// Radius of the axial-field null in metres.
    pub r_null: f64,
    /// Target separatrix radius in metres.
    pub target_separatrix_radius_m: f64,
    /// Separatrix radius error in metres.
    pub separatrix_radius_error_m: f64,
    /// Index of the separatrix sample.
    pub separatrix_index: usize,
    /// Whether the field reversal check passed.
    pub field_reversal_passed: bool,
    /// FRC kinetic-stability `s` parameter.
    pub s_parameter: f64,
    /// Energy in joules.
    pub energy_j: f64,
    /// Whether the numerical solve converged.
    pub converged: bool,
    /// Final numerical residual.
    pub residual: f64,
    /// Profile-shape exponent used by the analytical closure.
    pub delta: f64,
    /// Magnetic flux on axis in webers.
    pub psi_axis_wb: f64,
    /// Magnetic flux at the separatrix in webers.
    pub psi_separatrix_wb: f64,
    /// Absolute normalized-flux error on axis.
    pub psi_normalized_axis_error: f64,
    /// Normalized flux at the separatrix.
    pub psi_normalized_separatrix: f64,
    /// Absolute normalized-flux error at the separatrix.
    pub psi_normalized_separatrix_error: f64,
    /// Infinity norm of the psi normalized residual.
    pub psi_normalized_residual_linf: f64,
    /// Whether the psi normalized monotonic check passed.
    pub psi_normalized_monotonic_passed: bool,
    /// Whether the psi normalized bounds check passed.
    pub psi_normalized_bounds_passed: bool,
    /// Pressure balance ratio.
    pub pressure_balance_ratio: f64,
    /// Samples of pressure balance residual.
    pub pressure_balance_residual: Array1<f64>,
    /// Infinity norm of the pressure balance residual.
    pub pressure_balance_residual_linf: f64,
    /// Euclidean norm of the pressure balance residual.
    pub pressure_balance_residual_l2: f64,
    /// Analytical pressure-gradient samples in pascals per metre.
    pub pressure_gradient_analytic_pa_m: Array1<f64>,
    /// Samples of pressure gradient residual.
    pub pressure_gradient_residual: Array1<f64>,
    /// Infinity norm of the pressure gradient residual.
    pub pressure_gradient_residual_linf: f64,
    /// Euclidean norm of the pressure gradient residual.
    pub pressure_gradient_residual_l2: f64,
    /// Peak pressure in pascals.
    pub peak_pressure_pa: f64,
    /// Density peak in particles per cubic metre.
    pub density_peak_m3: f64,
    /// Input density in particles per cubic metre.
    pub input_density_m3: f64,
    /// Central density residual in particles per cubic metre.
    pub central_density_residual_m3: f64,
    /// Central density relative error.
    pub central_density_relative_error: f64,
    /// Beta peak.
    pub beta_peak: f64,
    /// Beta separatrix average.
    pub beta_separatrix_average: f64,
    /// Radially integrated particle line density per metre.
    pub particle_line_density_m1: f64,
    /// Pressure-energy integral inside the separatrix in joules per metre.
    pub separatrix_pressure_energy_j_m: f64,
    /// Magnetic-deficit energy integral inside the separatrix in joules per metre.
    pub separatrix_magnetic_deficit_energy_j_m: f64,
    /// Separatrix energy closure relative error.
    pub separatrix_energy_closure_relative_error: f64,
    /// Input thermal pressure in pascals.
    pub input_thermal_pressure_pa: f64,
    /// Thermal pressure ratio.
    pub thermal_pressure_ratio: f64,
    /// Samples of flux derivative residual.
    pub flux_derivative_residual: Array1<f64>,
    /// Infinity norm of the flux derivative residual.
    pub flux_derivative_residual_linf: f64,
    /// Euclidean norm of the flux derivative residual.
    pub flux_derivative_residual_l2: f64,
    /// Ampère-law residual samples in amperes per square metre.
    pub ampere_residual: Array1<f64>,
    /// Infinity norm of the ampere residual.
    pub ampere_residual_linf: f64,
    /// Euclidean norm of the ampere residual.
    pub ampere_residual_l2: f64,
    /// Peak azimuthal current density in amperes per square metre.
    pub peak_j_theta_a_m2: f64,
    /// Axial-field gradient at the separatrix in tesla per metre.
    pub separatrix_bz_gradient_t_m: f64,
    /// Expected axial-field gradient at the separatrix in tesla per metre.
    pub separatrix_expected_bz_gradient_t_m: f64,
    /// Separatrix gradient relative error.
    pub separatrix_gradient_relative_error: f64,
    /// Separatrix current density in amperes per square metre.
    pub separatrix_current_density_a_m2: f64,
    /// Separatrix expected current density in amperes per square metre.
    pub separatrix_expected_current_density_a_m2: f64,
    /// Separatrix current density relative error.
    pub separatrix_current_density_relative_error: f64,
    /// Integrated sheet current in amperes per metre.
    pub sheet_current_integral_a_m: f64,
    /// Ampère-law sheet-current target in amperes per metre.
    pub expected_sheet_current_integral_a_m: f64,
    /// Sheet current integral relative error.
    pub sheet_current_integral_relative_error: f64,
    /// Samples of force balance residual.
    pub force_balance_residual: Array1<f64>,
    /// Infinity norm of the force balance residual.
    pub force_balance_residual_linf: f64,
    /// Euclidean norm of the force balance residual.
    pub force_balance_residual_l2: f64,
    /// Stable identifier for the analytical model used.
    pub model: &'static str,
    // Rotating rigid-rotor (Rostoker & Qerushi 2002) diagnostics. For
    // `theta_dot == 0` these carry the trivial no-rotation values so the accepted
    // contract is byte-unchanged.
    /// Rigid-rotor angular frequency in radians per second.
    pub theta_dot: f64,
    /// Literature reference for the rotating closure.
    pub rotation_reference: &'static str,
    /// Centrifugal pressure-gradient source in pascals per metre.
    pub centrifugal_source_pa_m: Array1<f64>,
    /// Samples of rotation force balance residual.
    pub rotation_force_balance_residual: Array1<f64>,
    /// Infinity norm of the rotation force balance residual.
    pub rotation_force_balance_residual_linf: f64,
    /// Euclidean norm of the rotation force balance residual.
    pub rotation_force_balance_residual_l2: f64,
    /// Peak rotation Mach number.
    pub rotation_mach_number: f64,
    /// Rotation pressure peak radius in metres.
    pub rotation_pressure_peak_radius_m: f64,
    /// Fraction of pressure samples clipped to maintain physical bounds.
    pub pressure_clipped_fraction: f64,
}

/// Strict solver errors for invalid input or unimplemented physics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrcSolverError {
    /// A named input violated the solver contract.
    InvalidInput(&'static str),
    /// The requested rotating boundary-value problem is not implemented.
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
