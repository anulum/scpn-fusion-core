// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
//! PyO3 Python bindings for SCPN Fusion Core.
//!
//! Stage 10: Exposes Grad-Shafranov solver, thermodynamics, control,
//! diagnostics, and ML modules to Python via PyO3 + numpy.

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use fusion_physics::frc::{
    rotating_frc_bvp_acceptance_status, solve_frc_equilibrium as solve_frc_equilibrium_rust,
    solve_rotating_frc_equilibrium as solve_rotating_frc_equilibrium_rust, RigidRotorFrcInputs,
};

mod bindings;
use bindings::control::{PyFnoController, PyMpcController};
use bindings::diagnostics::PyTomography;
use bindings::equilibrium::{
    measure_magnetics, multigrid_vcycle, shafranov_bv, solve_coil_currents, PyEquilibriumResult,
    PyFusionKernel, PyInverseResult, PyInverseSolver, PyThermodynamicsResult,
};
use bindings::flight::{PyFlightState, PyRustFlightSim, PySimulationReport, PyStepMetrics};
#[cfg(feature = "gpu")]
use bindings::gpu as gpu_bindings;
use bindings::mhd::{rutherford_island_growth, simulate_tearing_mode, PyHallMHD, PyReducedMHD};
use bindings::ml::PyNeuralTransport;
use bindings::neural::{
    scpn_dense_activations, scpn_marking_update, scpn_sample_firing, PySnnController, PySnnPool,
};
use bindings::nuclear::PyBreedingBlanket;
use bindings::particles::{
    py_advance_boris, py_get_heating_profile, py_particle_population_summary,
    py_seed_alpha_particles, PyParticle, PyPopulationSummary,
};
use bindings::phase::{py_kuramoto_run, py_kuramoto_step, py_upde_run, py_upde_tick};
use bindings::plant::PyPlantModel;
use bindings::rmf::{PyPacingMode, PyRmfAotCertificate, PyRmfConfig, PyRmfController};
use bindings::transport::{
    py_evaluate_design, py_run_design_scan, PyDriftWave, PyFokkerPlanckSolver, PyPlasma2D,
    PySpiAblationSolver, PyTransportSolver,
};

// ─── Module registration ───

use fusion_physics::gk_nonlinear::{NonlinearGKConfig, NonlinearGKSolver, NonlinearGKState};
/// SCPN Fusion Core — Rust-accelerated plasma physics.
use ndarray::{Ix3, Ix6};
use num_complex::Complex64;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};

#[pyclass]
pub struct PyNonlinearGKSolver {
    inner: NonlinearGKSolver,
}

type PyComplexFieldPair<'py> = (
    Bound<'py, PyArrayDyn<Complex64>>,
    Bound<'py, PyArrayDyn<Complex64>>,
);

#[pymethods]
impl PyNonlinearGKSolver {
    #[new]
    fn new() -> PyResult<Self> {
        let cfg = NonlinearGKConfig::default();
        Ok(PyNonlinearGKSolver {
            inner: NonlinearGKSolver::new(cfg),
        })
    }

    fn step<'py>(
        &self,
        py: Python<'py>,
        f_py: PyReadonlyArrayDyn<'py, Complex64>,
        phi_py: PyReadonlyArrayDyn<'py, Complex64>,
        time: f64,
        dt: f64,
    ) -> PyResult<PyComplexFieldPair<'py>> {
        let f = f_py
            .as_array()
            .into_dimensionality::<Ix6>()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .to_owned();
        let phi = phi_py
            .as_array()
            .into_dimensionality::<Ix3>()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .to_owned();
        let state = NonlinearGKState {
            f,
            phi,
            time,
            a_par: None,
        };
        let new_state = self.inner.rk4_step(&state, dt);
        let f_out = new_state.f.into_dyn().into_pyarray(py);
        let phi_out = new_state.phi.into_dyn().into_pyarray(py);
        Ok((f_out, phi_out))
    }
}

/// Python-accessible Steinhauer no-rotation FRC analytical solver.

#[pyfunction]
#[pyo3(name = "py_rotating_frc_bvp_acceptance_status")]
fn py_rotating_frc_bvp_acceptance_status<'py>(
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
fn py_solve_rotating_frc_equilibrium<'py>(
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
fn py_solve_frc_equilibrium<'py>(
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

#[pymodule]
fn scpn_fusion_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFusionKernel>()?;
    m.add_class::<PyEquilibriumResult>()?;
    m.add_class::<PyThermodynamicsResult>()?;
    m.add_class::<PyNeuralTransport>()?;
    m.add_class::<PyInverseSolver>()?;
    m.add_class::<PyInverseResult>()?;
    m.add_class::<PyPlantModel>()?;
    m.add_class::<PyRustFlightSim>()?;
    m.add_class::<PySimulationReport>()?;
    m.add_class::<PyStepMetrics>()?;
    m.add_class::<PyFlightState>()?;
    m.add_function(wrap_pyfunction!(shafranov_bv, m)?)?;
    m.add_function(wrap_pyfunction!(solve_coil_currents, m)?)?;
    m.add_function(wrap_pyfunction!(measure_magnetics, m)?)?;
    m.add_function(wrap_pyfunction!(multigrid_vcycle, m)?)?;
    m.add_function(wrap_pyfunction!(scpn_dense_activations, m)?)?;
    m.add_function(wrap_pyfunction!(scpn_marking_update, m)?)?;
    m.add_function(wrap_pyfunction!(scpn_sample_firing, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_tearing_mode, m)?)?;
    m.add_function(wrap_pyfunction!(rutherford_island_growth, m)?)?;
    // Particle / Boris integrator bridge
    m.add_class::<PyParticle>()?;
    m.add_class::<PyPopulationSummary>()?;
    m.add_function(wrap_pyfunction!(py_seed_alpha_particles, m)?)?;
    m.add_function(wrap_pyfunction!(py_advance_boris, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_heating_profile, m)?)?;
    m.add_function(wrap_pyfunction!(py_particle_population_summary, m)?)?;
    // SNN controller bridge
    m.add_class::<PySnnPool>()?;
    m.add_class::<PySnnController>()?;
    // Extended PyO3 bridges
    m.add_class::<PyHallMHD>()?;
    m.add_class::<PyFnoController>()?;
    m.add_class::<PyMpcController>()?;
    m.add_class::<PyTomography>()?;
    m.add_class::<PyBreedingBlanket>()?;
    m.add_class::<PyPlasma2D>()?;
    m.add_function(wrap_pyfunction!(py_evaluate_design, m)?)?;
    m.add_function(wrap_pyfunction!(py_run_design_scan, m)?)?;
    m.add_class::<PyDriftWave>()?;
    m.add_class::<PyTransportSolver>()?;
    m.add_class::<PyFokkerPlanckSolver>()?;
    m.add_class::<PySpiAblationSolver>()?;
    m.add_class::<PyReducedMHD>()?;
    #[cfg(feature = "gpu")]
    {
        m.add_class::<gpu_bindings::PyGpuSolver>()?;
        m.add_function(wrap_pyfunction!(gpu_bindings::py_gpu_available, m)?)?;
        m.add_function(wrap_pyfunction!(gpu_bindings::py_gpu_info, m)?)?;
    }
    m.add_class::<PyNonlinearGKSolver>()?;
    m.add_function(wrap_pyfunction!(py_kuramoto_step, m)?)?;
    m.add_function(wrap_pyfunction!(py_kuramoto_run, m)?)?;
    m.add_function(wrap_pyfunction!(py_upde_tick, m)?)?;
    m.add_function(wrap_pyfunction!(py_upde_run, m)?)?;
    m.add_function(wrap_pyfunction!(py_solve_frc_equilibrium, m)?)?;
    m.add_function(wrap_pyfunction!(py_rotating_frc_bvp_acceptance_status, m)?)?;
    m.add_function(wrap_pyfunction!(py_solve_rotating_frc_equilibrium, m)?)?;
    m.add_class::<PyRmfConfig>()?;
    m.add_class::<PyRmfController>()?;
    m.add_class::<PyPacingMode>()?;
    m.add_class::<PyRmfAotCertificate>()?;
    Ok(())
}
