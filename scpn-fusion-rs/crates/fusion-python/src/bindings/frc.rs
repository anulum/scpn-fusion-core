// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
//! PyO3 bindings for the field-reversed-configuration (FRC) lane (`fusion-physics`).
//!
//! Exposes the rigid-rotor FRC equilibrium solvers — the no-rotation and
//! rotating rigid-rotor rings plus the rotating-BVP acceptance status — to
//! Python, each returning a rich diagnostics dict mirroring the NumPy tier.

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use fusion_physics::frc::{
    rotating_frc_bvp_acceptance_status, solve_frc_equilibrium as solve_frc_equilibrium_rust,
    solve_rotating_frc_equilibrium as solve_rotating_frc_equilibrium_rust, RigidRotorFrcInputs,
};

/// Python-accessible Steinhauer no-rotation FRC analytical solver.

#[pyfunction]
#[pyo3(name = "py_rotating_frc_bvp_acceptance_status")]
pub(crate) fn py_rotating_frc_bvp_acceptance_status<'py>(
    py: pyo3::Python<'py>,
) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
    let status = rotating_frc_bvp_acceptance_status();
    let out = pyo3::types::PyDict::new(py);
    out.set_item("status", status.status)?;
    out.set_item("accepted_contract", status.accepted_contract)?;
    out.set_item("rotating_bvp_implemented", status.rotating_bvp_implemented)?;
    out.set_item(
        "rotating_closure_reference",
        status.rotating_closure_reference,
    )?;
    out.set_item("solver_action", status.solver_action)?;
    out.set_item("required_reference", status.required_reference)?;
    out.set_item(
        "reduces_to_no_rotation_contract",
        status.reduces_to_no_rotation_contract,
    )?;
    out.set_item(
        "steinhauer_figure3_parity_claimed",
        status.steinhauer_figure3_parity_claimed,
    )?;
    out.set_item("non_closing_references", status.non_closing_references)?;
    out.set_item("claim_boundary", status.claim_boundary)?;
    Ok(out)
}

#[pyfunction]
#[pyo3(
    signature = (n0, t_i_ev, t_e_ev, theta_dot, r_s, b_ext, rho_grid, tolerance=1.0e-10, delta=None),
    text_signature = "(n0, t_i_ev, t_e_ev, theta_dot, r_s, b_ext, rho_grid, tolerance=1e-10, delta=None)",
    name = "solve_rotating_frc_equilibrium_rust"
)]
// Arguments mirror the named Python keyword parameters of the binding; bundling
// them into a struct would break the public Python call signature.
#[allow(clippy::too_many_arguments)]
pub(crate) fn py_solve_rotating_frc_equilibrium<'py>(
    py: pyo3::Python<'py>,
    n0: f64,
    t_i_ev: f64,
    t_e_ev: f64,
    theta_dot: f64,
    r_s: f64,
    b_ext: f64,
    rho_grid: numpy::PyReadonlyArray1<'py, f64>,
    tolerance: f64,
    delta: Option<f64>,
) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
    let rho_grid_owned = rho_grid.as_array().to_owned();
    let inputs = RigidRotorFrcInputs {
        n0,
        t_i_ev,
        t_e_ev,
        theta_dot,
        r_s,
        b_ext,
        delta,
    };
    let state = solve_rotating_frc_equilibrium_rust(&inputs, &rho_grid_owned, tolerance)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let out = pyo3::types::PyDict::new(py);
    out.set_item("rho", numpy::IntoPyArray::into_pyarray(state.rho, py))?;
    out.set_item("psi", numpy::IntoPyArray::into_pyarray(state.psi, py))?;
    out.set_item(
        "psi_normalized",
        numpy::IntoPyArray::into_pyarray(state.psi_normalized, py),
    )?;
    out.set_item("B_z", numpy::IntoPyArray::into_pyarray(state.b_z, py))?;
    out.set_item(
        "B_theta",
        numpy::IntoPyArray::into_pyarray(state.b_theta, py),
    )?;
    out.set_item(
        "J_theta",
        numpy::IntoPyArray::into_pyarray(state.j_theta, py),
    )?;
    out.set_item("p", numpy::IntoPyArray::into_pyarray(state.p, py))?;
    out.set_item(
        "density_m3",
        numpy::IntoPyArray::into_pyarray(state.density_m3, py),
    )?;
    out.set_item("beta", numpy::IntoPyArray::into_pyarray(state.beta, py))?;
    out.set_item("R_null", state.r_null)?;
    out.set_item(
        "target_separatrix_radius_m",
        state.target_separatrix_radius_m,
    )?;
    out.set_item("separatrix_radius_error_m", state.separatrix_radius_error_m)?;
    out.set_item("separatrix_index", state.separatrix_index)?;
    out.set_item("field_reversal_passed", state.field_reversal_passed)?;
    out.set_item("s_parameter", state.s_parameter)?;
    out.set_item("energy_J", state.energy_j)?;
    out.set_item("converged", state.converged)?;
    out.set_item("residual", state.residual)?;
    out.set_item("delta", state.delta)?;
    out.set_item("psi_axis_Wb", state.psi_axis_wb)?;
    out.set_item("psi_separatrix_Wb", state.psi_separatrix_wb)?;
    out.set_item("psi_normalized_axis_error", state.psi_normalized_axis_error)?;
    out.set_item("psi_normalized_separatrix", state.psi_normalized_separatrix)?;
    out.set_item(
        "psi_normalized_separatrix_error",
        state.psi_normalized_separatrix_error,
    )?;
    out.set_item(
        "psi_normalized_residual_linf",
        state.psi_normalized_residual_linf,
    )?;
    out.set_item(
        "psi_normalized_monotonic_passed",
        state.psi_normalized_monotonic_passed,
    )?;
    out.set_item(
        "psi_normalized_bounds_passed",
        state.psi_normalized_bounds_passed,
    )?;
    out.set_item("pressure_balance_ratio", state.pressure_balance_ratio)?;
    out.set_item(
        "pressure_balance_residual",
        numpy::IntoPyArray::into_pyarray(state.pressure_balance_residual, py),
    )?;
    out.set_item(
        "pressure_balance_residual_linf",
        state.pressure_balance_residual_linf,
    )?;
    out.set_item(
        "pressure_balance_residual_l2",
        state.pressure_balance_residual_l2,
    )?;
    out.set_item(
        "pressure_gradient_analytic_Pa_m",
        numpy::IntoPyArray::into_pyarray(state.pressure_gradient_analytic_pa_m, py),
    )?;
    out.set_item(
        "pressure_gradient_residual",
        numpy::IntoPyArray::into_pyarray(state.pressure_gradient_residual, py),
    )?;
    out.set_item(
        "pressure_gradient_residual_linf",
        state.pressure_gradient_residual_linf,
    )?;
    out.set_item(
        "pressure_gradient_residual_l2",
        state.pressure_gradient_residual_l2,
    )?;
    out.set_item("peak_pressure_pa", state.peak_pressure_pa)?;
    out.set_item("density_peak_m3", state.density_peak_m3)?;
    out.set_item("input_density_m3", state.input_density_m3)?;
    out.set_item(
        "central_density_residual_m3",
        state.central_density_residual_m3,
    )?;
    out.set_item(
        "central_density_relative_error",
        state.central_density_relative_error,
    )?;
    out.set_item("beta_peak", state.beta_peak)?;
    out.set_item("beta_separatrix_average", state.beta_separatrix_average)?;
    out.set_item("particle_line_density_m1", state.particle_line_density_m1)?;
    out.set_item(
        "separatrix_pressure_energy_J_m",
        state.separatrix_pressure_energy_j_m,
    )?;
    out.set_item(
        "separatrix_magnetic_deficit_energy_J_m",
        state.separatrix_magnetic_deficit_energy_j_m,
    )?;
    out.set_item(
        "separatrix_energy_closure_relative_error",
        state.separatrix_energy_closure_relative_error,
    )?;
    out.set_item("input_thermal_pressure_pa", state.input_thermal_pressure_pa)?;
    out.set_item("thermal_pressure_ratio", state.thermal_pressure_ratio)?;
    out.set_item(
        "flux_derivative_residual",
        numpy::IntoPyArray::into_pyarray(state.flux_derivative_residual, py),
    )?;
    out.set_item(
        "flux_derivative_residual_linf",
        state.flux_derivative_residual_linf,
    )?;
    out.set_item(
        "flux_derivative_residual_l2",
        state.flux_derivative_residual_l2,
    )?;
    out.set_item(
        "ampere_residual",
        numpy::IntoPyArray::into_pyarray(state.ampere_residual, py),
    )?;
    out.set_item("ampere_residual_linf", state.ampere_residual_linf)?;
    out.set_item("ampere_residual_l2", state.ampere_residual_l2)?;
    out.set_item("peak_j_theta_A_m2", state.peak_j_theta_a_m2)?;
    out.set_item(
        "separatrix_bz_gradient_T_m",
        state.separatrix_bz_gradient_t_m,
    )?;
    out.set_item(
        "separatrix_expected_bz_gradient_T_m",
        state.separatrix_expected_bz_gradient_t_m,
    )?;
    out.set_item(
        "separatrix_gradient_relative_error",
        state.separatrix_gradient_relative_error,
    )?;
    out.set_item(
        "separatrix_current_density_A_m2",
        state.separatrix_current_density_a_m2,
    )?;
    out.set_item(
        "separatrix_expected_current_density_A_m2",
        state.separatrix_expected_current_density_a_m2,
    )?;
    out.set_item(
        "separatrix_current_density_relative_error",
        state.separatrix_current_density_relative_error,
    )?;
    out.set_item(
        "sheet_current_integral_A_m",
        state.sheet_current_integral_a_m,
    )?;
    out.set_item(
        "expected_sheet_current_integral_A_m",
        state.expected_sheet_current_integral_a_m,
    )?;
    out.set_item(
        "sheet_current_integral_relative_error",
        state.sheet_current_integral_relative_error,
    )?;
    out.set_item(
        "force_balance_residual",
        numpy::IntoPyArray::into_pyarray(state.force_balance_residual, py),
    )?;
    out.set_item(
        "force_balance_residual_linf",
        state.force_balance_residual_linf,
    )?;
    out.set_item("force_balance_residual_l2", state.force_balance_residual_l2)?;
    out.set_item("model", state.model)?;
    out.set_item("theta_dot", state.theta_dot)?;
    out.set_item("rotation_reference", state.rotation_reference)?;
    out.set_item(
        "centrifugal_source_Pa_m",
        numpy::IntoPyArray::into_pyarray(state.centrifugal_source_pa_m, py),
    )?;
    out.set_item(
        "rotation_force_balance_residual",
        numpy::IntoPyArray::into_pyarray(state.rotation_force_balance_residual, py),
    )?;
    out.set_item(
        "rotation_force_balance_residual_linf",
        state.rotation_force_balance_residual_linf,
    )?;
    out.set_item(
        "rotation_force_balance_residual_l2",
        state.rotation_force_balance_residual_l2,
    )?;
    out.set_item("rotation_mach_number", state.rotation_mach_number)?;
    out.set_item(
        "rotation_pressure_peak_radius_m",
        state.rotation_pressure_peak_radius_m,
    )?;
    out.set_item("pressure_clipped_fraction", state.pressure_clipped_fraction)?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (rho, n0, t_i_ev, t_e_ev, theta_dot, r_s, b_ext, delta=None, tolerance=1.0e-10))]
#[expect(
    clippy::too_many_arguments,
    reason = "PyO3 binding preserves the stable Python keyword API; inputs are assembled into RigidRotorFrcInputs immediately."
)]
pub(crate) fn py_solve_frc_equilibrium<'py>(
    py: Python<'py>,
    rho: PyReadonlyArray1<'py, f64>,
    n0: f64,
    t_i_ev: f64,
    t_e_ev: f64,
    theta_dot: f64,
    r_s: f64,
    b_ext: f64,
    delta: Option<f64>,
    tolerance: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let inputs = RigidRotorFrcInputs {
        n0,
        t_i_ev,
        t_e_ev,
        theta_dot,
        r_s,
        b_ext,
        delta,
    };
    let rho_grid = rho.as_array().to_owned();
    let state = solve_frc_equilibrium_rust(&inputs, &rho_grid, tolerance)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let out = PyDict::new(py);
    out.set_item("rho", state.rho.into_pyarray(py))?;
    out.set_item("psi", state.psi.into_pyarray(py))?;
    out.set_item("psi_normalized", state.psi_normalized.into_pyarray(py))?;
    out.set_item("B_z", state.b_z.into_pyarray(py))?;
    out.set_item("B_theta", state.b_theta.into_pyarray(py))?;
    out.set_item("J_theta", state.j_theta.into_pyarray(py))?;
    out.set_item("p", state.p.into_pyarray(py))?;
    out.set_item("density_m3", state.density_m3.into_pyarray(py))?;
    out.set_item("beta", state.beta.into_pyarray(py))?;
    out.set_item("R_null", state.r_null)?;
    out.set_item(
        "target_separatrix_radius_m",
        state.target_separatrix_radius_m,
    )?;
    out.set_item("separatrix_radius_error_m", state.separatrix_radius_error_m)?;
    out.set_item("separatrix_index", state.separatrix_index)?;
    out.set_item("field_reversal_passed", state.field_reversal_passed)?;
    out.set_item("s_parameter", state.s_parameter)?;
    out.set_item("energy_J", state.energy_j)?;
    out.set_item("converged", state.converged)?;
    out.set_item("residual", state.residual)?;
    out.set_item("delta", state.delta)?;
    out.set_item("psi_axis_Wb", state.psi_axis_wb)?;
    out.set_item("psi_separatrix_Wb", state.psi_separatrix_wb)?;
    out.set_item("psi_normalized_axis_error", state.psi_normalized_axis_error)?;
    out.set_item("psi_normalized_separatrix", state.psi_normalized_separatrix)?;
    out.set_item(
        "psi_normalized_separatrix_error",
        state.psi_normalized_separatrix_error,
    )?;
    out.set_item(
        "psi_normalized_residual_linf",
        state.psi_normalized_residual_linf,
    )?;
    out.set_item(
        "psi_normalized_monotonic_passed",
        state.psi_normalized_monotonic_passed,
    )?;
    out.set_item(
        "psi_normalized_bounds_passed",
        state.psi_normalized_bounds_passed,
    )?;
    out.set_item("pressure_balance_ratio", state.pressure_balance_ratio)?;
    out.set_item(
        "pressure_balance_residual",
        state.pressure_balance_residual.into_pyarray(py),
    )?;
    out.set_item(
        "pressure_balance_residual_linf",
        state.pressure_balance_residual_linf,
    )?;
    out.set_item(
        "pressure_balance_residual_l2",
        state.pressure_balance_residual_l2,
    )?;
    out.set_item(
        "pressure_gradient_analytic_Pa_m",
        state.pressure_gradient_analytic_pa_m.into_pyarray(py),
    )?;
    out.set_item(
        "pressure_gradient_residual",
        state.pressure_gradient_residual.into_pyarray(py),
    )?;
    out.set_item(
        "pressure_gradient_residual_linf",
        state.pressure_gradient_residual_linf,
    )?;
    out.set_item(
        "pressure_gradient_residual_l2",
        state.pressure_gradient_residual_l2,
    )?;
    out.set_item("peak_pressure_pa", state.peak_pressure_pa)?;
    out.set_item("density_peak_m3", state.density_peak_m3)?;
    out.set_item("input_density_m3", state.input_density_m3)?;
    out.set_item(
        "central_density_residual_m3",
        state.central_density_residual_m3,
    )?;
    out.set_item(
        "central_density_relative_error",
        state.central_density_relative_error,
    )?;
    out.set_item("beta_peak", state.beta_peak)?;
    out.set_item("beta_separatrix_average", state.beta_separatrix_average)?;
    out.set_item("particle_line_density_m1", state.particle_line_density_m1)?;
    out.set_item(
        "separatrix_pressure_energy_J_m",
        state.separatrix_pressure_energy_j_m,
    )?;
    out.set_item(
        "separatrix_magnetic_deficit_energy_J_m",
        state.separatrix_magnetic_deficit_energy_j_m,
    )?;
    out.set_item(
        "separatrix_energy_closure_relative_error",
        state.separatrix_energy_closure_relative_error,
    )?;
    out.set_item("input_thermal_pressure_pa", state.input_thermal_pressure_pa)?;
    out.set_item("thermal_pressure_ratio", state.thermal_pressure_ratio)?;
    out.set_item(
        "flux_derivative_residual",
        state.flux_derivative_residual.into_pyarray(py),
    )?;
    out.set_item(
        "flux_derivative_residual_linf",
        state.flux_derivative_residual_linf,
    )?;
    out.set_item(
        "flux_derivative_residual_l2",
        state.flux_derivative_residual_l2,
    )?;
    out.set_item("ampere_residual", state.ampere_residual.into_pyarray(py))?;
    out.set_item("ampere_residual_linf", state.ampere_residual_linf)?;
    out.set_item("ampere_residual_l2", state.ampere_residual_l2)?;
    out.set_item("peak_j_theta_A_m2", state.peak_j_theta_a_m2)?;
    out.set_item(
        "separatrix_bz_gradient_T_m",
        state.separatrix_bz_gradient_t_m,
    )?;
    out.set_item(
        "separatrix_expected_bz_gradient_T_m",
        state.separatrix_expected_bz_gradient_t_m,
    )?;
    out.set_item(
        "separatrix_gradient_relative_error",
        state.separatrix_gradient_relative_error,
    )?;
    out.set_item(
        "separatrix_current_density_A_m2",
        state.separatrix_current_density_a_m2,
    )?;
    out.set_item(
        "separatrix_expected_current_density_A_m2",
        state.separatrix_expected_current_density_a_m2,
    )?;
    out.set_item(
        "separatrix_current_density_relative_error",
        state.separatrix_current_density_relative_error,
    )?;
    out.set_item(
        "sheet_current_integral_A_m",
        state.sheet_current_integral_a_m,
    )?;
    out.set_item(
        "expected_sheet_current_integral_A_m",
        state.expected_sheet_current_integral_a_m,
    )?;
    out.set_item(
        "sheet_current_integral_relative_error",
        state.sheet_current_integral_relative_error,
    )?;
    out.set_item(
        "force_balance_residual",
        state.force_balance_residual.into_pyarray(py),
    )?;
    out.set_item(
        "force_balance_residual_linf",
        state.force_balance_residual_linf,
    )?;
    out.set_item("force_balance_residual_l2", state.force_balance_residual_l2)?;
    out.set_item("model", state.model)?;
    out.set_item("theta_dot", state.theta_dot)?;
    out.set_item("rotation_reference", state.rotation_reference)?;
    out.set_item(
        "centrifugal_source_Pa_m",
        numpy::IntoPyArray::into_pyarray(state.centrifugal_source_pa_m, py),
    )?;
    out.set_item(
        "rotation_force_balance_residual",
        numpy::IntoPyArray::into_pyarray(state.rotation_force_balance_residual, py),
    )?;
    out.set_item(
        "rotation_force_balance_residual_linf",
        state.rotation_force_balance_residual_linf,
    )?;
    out.set_item(
        "rotation_force_balance_residual_l2",
        state.rotation_force_balance_residual_l2,
    )?;
    out.set_item("rotation_mach_number", state.rotation_mach_number)?;
    out.set_item(
        "rotation_pressure_peak_radius_m",
        state.rotation_pressure_peak_radius_m,
    )?;
    out.set_item("pressure_clipped_fraction", state.pressure_clipped_fraction)?;
    Ok(out)
}
