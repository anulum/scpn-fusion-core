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

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use fusion_control::digital_twin::Plasma2D;
use fusion_control::flight_sim::RustFlightSim;
use fusion_control::mpc::{MPController, NeuralSurrogate};
use fusion_control::snn::{NeuroCyberneticController, SpikingControllerPool};
use fusion_control::spi_ablation::SpiAblationSolver;
use fusion_core::ignition::calculate_thermodynamics;
use fusion_core::inverse::{reconstruct_equilibrium, InverseConfig, JacobianMode};
use fusion_core::kernel::FusionKernel;
use fusion_core::particles::{
    advance_particles_boris, estimate_alpha_heating_profile, seed_alpha_test_particles,
    summarize_particle_population, ChargedParticle,
};
use fusion_core::source::ProfileParams;
use fusion_core::transport::{self, NeoclassicalParams, TransportSolver};
use fusion_engineering::blanket::neutron_wall_loading;
use fusion_engineering::layout::{
    aries_cost_scaling, cost_of_electricity as engineering_coe, scan_major_radius,
};
use fusion_engineering::tritium::tritium_breeding_ratio;
use fusion_ml::neural_transport::NeuralTransportModel;
use fusion_physics::design_scanner;
use fusion_physics::fno::FnoController;
use fusion_physics::fokker_planck::FokkerPlanckSolver;
use fusion_physics::frc::{
    rotating_frc_bvp_acceptance_status, solve_frc_equilibrium as solve_frc_equilibrium_rust,
    solve_rotating_frc_equilibrium as solve_rotating_frc_equilibrium_rust, RigidRotorFrcInputs,
};
use fusion_physics::hall_mhd::HallMHD;
use fusion_physics::rmf_control::{RmfConfig, RmfPhaseLockController};
use fusion_physics::sawtooth::ReducedMHD;
use fusion_physics::turbulence::DriftWavePhysics;
use fusion_types::state::Grid2D;
// ReactorConfig used internally by FusionKernel::from_file

mod bindings;
use bindings::diagnostics::PyTomography;
use bindings::nuclear::PyBreedingBlanket;
use bindings::phase::{py_kuramoto_run, py_kuramoto_step, py_upde_run, py_upde_tick};

// ─── Equilibrium solver ───

/// Python-accessible Grad-Shafranov equilibrium solver.
#[pyclass]
struct PyFusionKernel {
    inner: FusionKernel,
}

#[pymethods]
impl PyFusionKernel {
    /// Load reactor configuration from JSON file.
    #[new]
    fn new(config_path: &str) -> PyResult<Self> {
        let inner = FusionKernel::from_file(config_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(PyFusionKernel { inner })
    }

    /// Solve Grad-Shafranov equilibrium. Returns PyEquilibriumResult.
    fn solve_equilibrium(&mut self) -> PyResult<PyEquilibriumResult> {
        let result = self
            .inner
            .solve_equilibrium()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyEquilibriumResult {
            converged: result.converged,
            iterations: result.iterations,
            residual: result.residual,
            axis_r: result.axis_position.0,
            axis_z: result.axis_position.1,
            x_point_r: result.x_point_position.0,
            x_point_z: result.x_point_position.1,
            psi_axis: result.psi_axis,
            psi_boundary: result.psi_boundary,
            solve_time_ms: result.solve_time_ms,
        })
    }

    /// Calculate thermodynamics for given auxiliary power [MW].
    fn calculate_thermodynamics(&self, p_aux_mw: f64) -> PyResult<PyThermodynamicsResult> {
        let result = calculate_thermodynamics(&self.inner, p_aux_mw)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyThermodynamicsResult {
            p_fusion_mw: result.p_fusion_mw,
            p_alpha_mw: result.p_alpha_mw,
            p_loss_mw: result.p_loss_mw,
            p_aux_mw: result.p_aux_mw,
            net_mw: result.net_mw,
            q_factor: result.q_factor,
            t_peak_kev: result.t_peak_kev,
            w_thermal_mj: result.w_thermal_mj,
        })
    }

    /// Get Psi (magnetic flux) as numpy array [nz, nr].
    fn get_psi<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.psi().clone().into_pyarray(py)
    }

    /// Get toroidal current density as numpy array [nz, nr].
    fn get_j_phi<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.j_phi().clone().into_pyarray(py)
    }

    /// Get grid R coordinates as numpy array.
    fn get_r<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.grid().r.clone().into_pyarray(py)
    }

    /// Get grid Z coordinates as numpy array.
    fn get_z<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.grid().z.clone().into_pyarray(py)
    }

    /// Grid dimensions: (nr, nz).
    fn grid_shape(&self) -> (usize, usize) {
        (self.inner.grid().nr, self.inner.grid().nz)
    }

    /// Set inner linear solver: "sor" (default) or "multigrid".
    fn set_solver_method(&mut self, method: &str) -> PyResult<()> {
        let m = match method.to_ascii_lowercase().as_str() {
            "sor" | "picard_sor" => fusion_core::kernel::SolverMethod::PicardSor,
            "multigrid" | "picard_multigrid" | "mg" => {
                fusion_core::kernel::SolverMethod::PicardMultigrid
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown solver method '{method}'. Use 'sor' or 'multigrid'."
                )))
            }
        };
        self.inner.set_solver_method(m);
        Ok(())
    }

    /// Get current solver method name.
    fn solver_method(&self) -> &str {
        match self.inner.solver_method() {
            fusion_core::kernel::SolverMethod::PicardSor => "sor",
            fusion_core::kernel::SolverMethod::PicardMultigrid => "multigrid",
        }
    }
}

// ─── Neural transport ───

/// Python wrapper for neural transport surrogate (10 -> 64 -> 32 -> 3).
#[pyclass]
struct PyNeuralTransport {
    inner: NeuralTransportModel,
}

#[pymethods]
impl PyNeuralTransport {
    #[new]
    fn new() -> Self {
        Self {
            inner: NeuralTransportModel::new(),
        }
    }

    #[staticmethod]
    fn from_npz(path: &str) -> PyResult<Self> {
        let inner = NeuralTransportModel::from_npz(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn predict<'py>(&self, py: Python<'py>, input: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
        let output = self.inner.predict(&Array1::from_vec(input));
        output.into_pyarray(py)
    }

    fn predict_profile<'py>(
        &self,
        py: Python<'py>,
        inputs: PyReadonlyArray2<'py, f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let input_arr = inputs.as_array().to_owned();
        let output = self.inner.predict_profile(&input_arr);
        output.into_pyarray(py)
    }

    fn is_neural(&self) -> bool {
        self.inner.is_neural()
    }
}

// ─── High-speed Flight Simulator ───

#[pyclass]
struct PySimulationReport {
    #[pyo3(get)]
    pub steps: usize,
    #[pyo3(get)]
    pub duration_s: f64,
    #[pyo3(get)]
    pub wall_time_ms: f64,
    #[pyo3(get)]
    pub max_step_time_us: f64,
    #[pyo3(get)]
    pub mean_abs_r_error: f64,
    #[pyo3(get)]
    pub mean_abs_z_error: f64,
    #[pyo3(get)]
    pub final_beta: f64,
    #[pyo3(get)]
    pub final_heating_mw: f64,
    #[pyo3(get)]
    pub max_beta: f64,
    #[pyo3(get)]
    pub max_heating_mw: f64,
    #[pyo3(get)]
    pub vessel_contact_events: usize,
    #[pyo3(get)]
    pub pf_constraint_events: usize,
    #[pyo3(get)]
    pub heating_constraint_events: usize,
    #[pyo3(get)]
    pub retained_steps: usize,
    #[pyo3(get)]
    pub history_truncated: bool,
    #[pyo3(get)]
    pub disrupted: bool,
    #[pyo3(get)]
    pub r_history: Vec<f64>,
    #[pyo3(get)]
    pub z_history: Vec<f64>,
    #[pyo3(get)]
    pub ip_history: Vec<f64>,
}

#[pyclass]
struct PyStepMetrics {
    #[pyo3(get)]
    pub r_error: f64,
    #[pyo3(get)]
    pub z_error: f64,
    #[pyo3(get)]
    pub disrupted: bool,
    #[pyo3(get)]
    pub step_time_us: f64,
    #[pyo3(get)]
    pub beta: f64,
    #[pyo3(get)]
    pub heating_mw: f64,
    #[pyo3(get)]
    pub vessel_contact: bool,
    #[pyo3(get)]
    pub pf_constraint_active: bool,
    #[pyo3(get)]
    pub heating_constraint_active: bool,
}

#[pyclass]
struct PyFlightState {
    #[pyo3(get)]
    pub r: f64,
    #[pyo3(get)]
    pub z: f64,
    #[pyo3(get)]
    pub ip_ma: f64,
    #[pyo3(get)]
    pub beta: f64,
    #[pyo3(get)]
    pub heating_mw: f64,
}

fn to_py_simulation_report(
    report: fusion_control::flight_sim::SimulationReport,
) -> PySimulationReport {
    PySimulationReport {
        steps: report.steps,
        duration_s: report.duration_s,
        wall_time_ms: report.wall_time_ms,
        max_step_time_us: report.max_step_time_us,
        mean_abs_r_error: report.mean_abs_r_error,
        mean_abs_z_error: report.mean_abs_z_error,
        final_beta: report.final_beta,
        final_heating_mw: report.final_heating_mw,
        max_beta: report.max_beta,
        max_heating_mw: report.max_heating_mw,
        vessel_contact_events: report.vessel_contact_events,
        pf_constraint_events: report.pf_constraint_events,
        heating_constraint_events: report.heating_constraint_events,
        retained_steps: report.retained_steps,
        history_truncated: report.history_truncated,
        disrupted: report.disrupted,
        r_history: report.r_history,
        z_history: report.z_history,
        ip_history: report.ip_history,
    }
}

fn to_py_step_metrics(step: fusion_control::flight_sim::StepMetrics) -> PyStepMetrics {
    PyStepMetrics {
        r_error: step.r_error,
        z_error: step.z_error,
        disrupted: step.disrupted,
        step_time_us: step.step_time_us,
        beta: step.beta,
        heating_mw: step.heating_mw,
        vessel_contact: step.vessel_contact,
        pf_constraint_active: step.pf_constraint_active,
        heating_constraint_active: step.heating_constraint_active,
    }
}

fn to_py_flight_state(state: (f64, f64, f64, f64, f64)) -> PyFlightState {
    PyFlightState {
        r: state.0,
        z: state.1,
        ip_ma: state.2,
        beta: state.3,
        heating_mw: state.4,
    }
}

#[pyclass]
struct PyRustFlightSim {
    inner: RustFlightSim,
}

#[pymethods]
impl PyRustFlightSim {
    #[new]
    #[pyo3(signature = (target_r=6.2, target_z=0.0, control_hz=10000.0))]
    fn new(target_r: f64, target_z: f64, control_hz: f64) -> PyResult<Self> {
        let inner = RustFlightSim::new(target_r, target_z, control_hz)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (shot_duration_s, deterministic=false))]
    fn run_shot(
        &mut self,
        shot_duration_s: f64,
        deterministic: bool,
    ) -> PyResult<PySimulationReport> {
        let report = self
            .inner
            .run_shot(shot_duration_s, deterministic)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(to_py_simulation_report(report))
    }

    fn prepare_shot(&mut self, shot_duration_s: f64) -> PyResult<usize> {
        self.inner
            .prepare_shot(shot_duration_s)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn reset_for_shot(&mut self) {
        self.inner.reset_for_shot();
    }

    fn reset_plasma_state(&mut self) {
        self.inner.reset_plasma_state();
    }

    fn plasma_state(&self) -> PyFlightState {
        to_py_flight_state(self.inner.plasma_state())
    }

    fn step_once(&mut self, step_index: usize, shot_duration_s: f64) -> PyResult<PyStepMetrics> {
        let step = self
            .inner
            .step_once(step_index, shot_duration_s)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(to_py_step_metrics(step))
    }
}

// ─── Result types ───

/// Equilibrium solve result.
#[pyclass(from_py_object)]
#[derive(Clone)]
struct PyEquilibriumResult {
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    residual: f64,
    #[pyo3(get)]
    axis_r: f64,
    #[pyo3(get)]
    axis_z: f64,
    #[pyo3(get)]
    x_point_r: f64,
    #[pyo3(get)]
    x_point_z: f64,
    #[pyo3(get)]
    psi_axis: f64,
    #[pyo3(get)]
    psi_boundary: f64,
    #[pyo3(get)]
    solve_time_ms: f64,
}

#[pymethods]
impl PyEquilibriumResult {
    fn __repr__(&self) -> String {
        format!(
            "EquilibriumResult(converged={}, iters={}, residual={:.2e})",
            self.converged, self.iterations, self.residual
        )
    }
}

/// Thermodynamics result.
#[pyclass(from_py_object)]
#[derive(Clone)]
struct PyThermodynamicsResult {
    #[pyo3(get)]
    p_fusion_mw: f64,
    #[pyo3(get)]
    p_alpha_mw: f64,
    #[pyo3(get)]
    p_loss_mw: f64,
    #[pyo3(get)]
    p_aux_mw: f64,
    #[pyo3(get)]
    net_mw: f64,
    #[pyo3(get)]
    q_factor: f64,
    #[pyo3(get)]
    t_peak_kev: f64,
    #[pyo3(get)]
    w_thermal_mj: f64,
}

#[pymethods]
impl PyThermodynamicsResult {
    fn __repr__(&self) -> String {
        format!(
            "ThermodynamicsResult(Q={:.2}, P_fusion={:.1} MW)",
            self.q_factor, self.p_fusion_mw
        )
    }
}

// ─── Inverse solver ───

#[pyclass(from_py_object)]
#[derive(Clone)]
struct PyInverseResult {
    #[pyo3(get)]
    params_p: Vec<f64>,
    #[pyo3(get)]
    params_ff: Vec<f64>,
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    residual: f64,
}

#[pymethods]
impl PyInverseResult {
    fn __repr__(&self) -> String {
        format!(
            "InverseResult(converged={}, iters={}, residual={:.3e})",
            self.converged, self.iterations, self.residual
        )
    }
}

#[pyclass]
struct PyInverseSolver;

fn parse_jacobian_mode(mode: &str) -> PyResult<JacobianMode> {
    match mode.to_ascii_lowercase().as_str() {
        "analytical" => Ok(JacobianMode::Analytical),
        "fd" | "finite_difference" | "finite-difference" => Ok(JacobianMode::FiniteDifference),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown jacobian_mode '{mode}'. Use 'analytical' or 'finite_difference'."
        ))),
    }
}

fn parse_profile(values: Option<Vec<f64>>, default: ProfileParams) -> PyResult<ProfileParams> {
    let Some(v) = values else {
        return Ok(default);
    };
    if v.len() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Profile parameter vector must have 4 values [ped_height, ped_top, ped_width, core_alpha], got {}",
            v.len()
        )));
    }
    Ok(ProfileParams {
        ped_height: v[0],
        ped_top: v[1],
        ped_width: v[2],
        core_alpha: v[3],
    })
}

#[pymethods]
impl PyInverseSolver {
    #[new]
    fn new() -> Self {
        Self
    }

    #[pyo3(signature=(probes, measurements, jacobian_mode="analytical", initial_p=None, initial_ff=None))]
    fn reconstruct(
        &self,
        probes: Vec<f64>,
        measurements: Vec<f64>,
        jacobian_mode: &str,
        initial_p: Option<Vec<f64>>,
        initial_ff: Option<Vec<f64>>,
    ) -> PyResult<PyInverseResult> {
        let mode = parse_jacobian_mode(jacobian_mode)?;
        let init_p = parse_profile(initial_p, ProfileParams::default())?;
        let init_ff = parse_profile(initial_ff, ProfileParams::default())?;

        let config = InverseConfig {
            jacobian_mode: mode,
            ..Default::default()
        };

        let result = reconstruct_equilibrium(&probes, &measurements, init_p, init_ff, &config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyInverseResult {
            params_p: vec![
                result.params_p.ped_height,
                result.params_p.ped_top,
                result.params_p.ped_width,
                result.params_p.core_alpha,
            ],
            params_ff: vec![
                result.params_ff.ped_height,
                result.params_ff.ped_top,
                result.params_ff.ped_width,
                result.params_ff.core_alpha,
            ],
            converged: result.converged,
            iterations: result.iterations,
            residual: result.residual,
        })
    }
}

// ─── Plant model ───

#[pyclass]
struct PyPlantModel;

#[pymethods]
impl PyPlantModel {
    #[new]
    fn new() -> Self {
        Self
    }

    fn tritium_breeding_ratio(
        &self,
        n_li6: f64,
        sigma_li6: f64,
        neutron_flux: f64,
        blanket_vol: f64,
    ) -> f64 {
        tritium_breeding_ratio(n_li6, sigma_li6, neutron_flux, blanket_vol)
    }

    fn wall_loading(&self, p_neutron: f64, r: f64, a: f64, kappa: f64) -> f64 {
        neutron_wall_loading(p_neutron, r, a, kappa)
    }

    fn aries_cost_scaling(&self, c0: f64, r: f64, b: f64) -> f64 {
        aries_cost_scaling(c0, r, b)
    }

    fn cost_of_electricity(&self, capital_annuity: f64, o_and_m: f64, p_net: f64, cf: f64) -> f64 {
        engineering_coe(capital_annuity, o_and_m, p_net, cf)
    }

    fn scan_radius(
        &self,
        r_min: f64,
        r_max: f64,
        steps: usize,
    ) -> Vec<(f64, f64, f64, f64, f64, f64)> {
        scan_major_radius(r_min, r_max, steps)
            .into_iter()
            .map(|d| {
                (
                    d.r_major,
                    d.b_field,
                    d.p_net,
                    d.capacity_factor,
                    d.capital_cost,
                    d.coe,
                )
            })
            .collect()
    }
}

// ─── Control systems ───

/// Shafranov equilibrium calculator.
///
/// Returns ``(bv_required, term_log, term_physics)``. ``bv_required`` is the
/// canonical dispatch output, bit-exact with the NumPy tier
/// (``scpn_fusion.control.analytic_solver.shafranov_bv``); the other two are
/// diagnostic terms of the force-balance formula.
#[pyfunction]
#[pyo3(signature = (
    r_geo,
    a_min,
    ip_ma,
    beta_p = fusion_control::analytic::BETA_P,
    li = fusion_control::analytic::LI,
))]
fn shafranov_bv(
    r_geo: f64,
    a_min: f64,
    ip_ma: f64,
    beta_p: f64,
    li: f64,
) -> PyResult<(f64, f64, f64)> {
    let result = fusion_control::analytic::shafranov_bv(r_geo, a_min, ip_ma, beta_p, li)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok((result.bv_required, result.term_log, result.term_physics))
}

/// Minimum-norm coil current solver.
///
/// `ridge_lambda` (default 0.0) adds Tikhonov regularisation; the result is
/// bit-exact with the NumPy tier
/// (`scpn_fusion.control.analytic_solver.solve_coil_currents`).
#[pyfunction]
#[pyo3(signature = (green_func, target_bv, ridge_lambda = 0.0))]
fn solve_coil_currents(
    green_func: Vec<f64>,
    target_bv: f64,
    ridge_lambda: f64,
) -> PyResult<Vec<f64>> {
    fusion_control::analytic::solve_coil_currents(&green_func, target_bv, ridge_lambda)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

// ─── Diagnostics ───

/// Create sensor suite and measure magnetics from Psi array.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn measure_magnetics<'py>(
    py: Python<'py>,
    psi: PyReadonlyArray2<'py, f64>,
    nr: usize,
    nz: usize,
    r_min: f64,
    r_max: f64,
    z_min: f64,
    z_max: f64,
) -> Bound<'py, PyArray1<f64>> {
    let psi_arr: Array2<f64> = psi.as_array().to_owned();
    let suite = fusion_diagnostics::sensors::SensorSuite::new(nr, nz, r_min, r_max, z_min, z_max);
    let measurements = suite.measure_magnetics(&psi_arr);
    ndarray::Array1::from_vec(measurements).into_pyarray(py)
}

// ─── Equilibrium solvers ───

/// Standalone geometric multigrid solve of the Grad-Shafranov GS* operator.
///
/// Relaxes `L*[psi] = source` on an `nr x nz` R-Z grid starting from the
/// boundary-valued `psi_bc`, returning `(psi, residual, n_cycles, converged)`.
/// `residual` is the final L-infinity residual. Algorithm-parity with the NumPy
/// tier (`scpn_fusion.core.multigrid_solve.multigrid_solve`): both relax the
/// identical toroidal GS* operator to the same fixed point within tolerance.
#[pyfunction]
#[pyo3(signature = (source, psi_bc, r_min, r_max, z_min, z_max, nr, nz, tol = 1e-6, max_cycles = 500))]
#[allow(clippy::too_many_arguments)]
fn multigrid_vcycle<'py>(
    py: Python<'py>,
    source: PyReadonlyArray2<'py, f64>,
    psi_bc: PyReadonlyArray2<'py, f64>,
    r_min: f64,
    r_max: f64,
    z_min: f64,
    z_max: f64,
    nr: usize,
    nz: usize,
    tol: f64,
    max_cycles: usize,
) -> (Bound<'py, PyArray2<f64>>, f64, usize, bool) {
    let grid = Grid2D::new(nr, nz, r_min, r_max, z_min, z_max);
    let source_arr: Array2<f64> = source.as_array().to_owned();
    let mut psi: Array2<f64> = psi_bc.as_array().to_owned();
    let result = fusion_math::multigrid::multigrid_solve(
        &mut psi,
        &source_arr,
        &grid,
        &fusion_math::multigrid::MultigridConfig::default(),
        max_cycles,
        tol,
    );
    (
        psi.into_pyarray(py),
        result.residual,
        result.cycles,
        result.converged,
    )
}

// ─── SCPN runtime kernels ───

/// Dense activation kernel for SCPN controller path.
#[pyfunction]
fn scpn_dense_activations<'py>(
    py: Python<'py>,
    arg1: &Bound<'py, PyAny>,
    arg2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // Canonical signature: (w_in, marking). Legacy parity tests use (marking, w_in).
    let (w, m) = if let (Ok(w), Ok(m)) = (
        arg1.extract::<PyReadonlyArray2<'py, f64>>(),
        arg2.extract::<PyReadonlyArray1<'py, f64>>(),
    ) {
        (w, m)
    } else if let (Ok(m), Ok(w)) = (
        arg1.extract::<PyReadonlyArray1<'py, f64>>(),
        arg2.extract::<PyReadonlyArray2<'py, f64>>(),
    ) {
        (w, m)
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "scpn_dense_activations expects (matrix, vector) or (vector, matrix)",
        ));
    };

    let w = w.as_array();
    let m = m.as_array();
    if w.ncols() != m.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "shape mismatch: w_in has {} columns, marking has length {}",
            w.ncols(),
            m.len()
        )));
    }
    Ok(w.dot(&m).into_pyarray(py))
}

/// Dense marking update kernel: clip(m - W_in^T f + W_out f, 0, 1).
#[pyfunction]
fn scpn_marking_update<'py>(
    py: Python<'py>,
    marking: PyReadonlyArray1<'py, f64>,
    w_in: &Bound<'py, PyAny>,
    w_out: &Bound<'py, PyAny>,
    firing: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let m = marking.as_array();
    let f = firing.as_array();

    // Canonical path used by controller: matrices W_in, W_out.
    if let (Ok(wi_2d), Ok(wo_2d)) = (
        w_in.extract::<PyReadonlyArray2<'py, f64>>(),
        w_out.extract::<PyReadonlyArray2<'py, f64>>(),
    ) {
        let wi = wi_2d.as_array();
        let wo = wo_2d.as_array();

        if wi.nrows() != f.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "shape mismatch: w_in rows {} != firing length {}",
                wi.nrows(),
                f.len()
            )));
        }
        if wo.ncols() != f.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "shape mismatch: w_out cols {} != firing length {}",
                wo.ncols(),
                f.len()
            )));
        }
        if wi.ncols() != m.len() || wo.nrows() != m.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "shape mismatch: marking length {} expects w_in cols {} and w_out rows {}",
                m.len(),
                wi.ncols(),
                wo.nrows()
            )));
        }

        let cons = wi.t().dot(&f);
        let prod = wo.dot(&f);
        let out = (&m - &cons + &prod).mapv(|x| x.clamp(0.0, 1.0));
        return Ok(out.into_pyarray(py));
    }

    // Legacy path used in parity tests: vectors pre, post.
    if let (Ok(pre_1d), Ok(post_1d)) = (
        w_in.extract::<PyReadonlyArray1<'py, f64>>(),
        w_out.extract::<PyReadonlyArray1<'py, f64>>(),
    ) {
        let pre = pre_1d.as_array();
        let post = post_1d.as_array();
        if pre.len() != m.len() || post.len() != m.len() || f.len() != m.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "vector mode length mismatch: marking={}, pre={}, post={}, firing={}",
                m.len(),
                pre.len(),
                post.len(),
                f.len()
            )));
        }
        let out = &m - &(&pre * &f) + &(&post * &f);
        return Ok(out.into_pyarray(py));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "scpn_marking_update expects matrix or vector pre/post arguments",
    ))
}

/// Stochastic firing estimator for SCPN transitions.
///
/// Returns mean Bernoulli activation over `n_passes` samples with optional
/// antithetic pairing.
#[pyfunction]
fn scpn_sample_firing<'py>(
    py: Python<'py>,
    p_fire: PyReadonlyArray1<'py, f64>,
    n_passes: usize,
    seed: u64,
    antithetic: bool,
) -> Bound<'py, PyArray1<f64>> {
    let probs = p_fire.as_array();
    let n_t = probs.len();
    let mut counts = vec![0_u32; n_t];
    let mut rng = StdRng::seed_from_u64(seed);

    if antithetic && n_passes >= 2 {
        let n_pairs = n_passes.div_ceil(2);
        let odd_trim = !n_passes.is_multiple_of(2);
        for pair_idx in 0..n_pairs {
            let use_mirror = !odd_trim || (pair_idx + 1 < n_pairs);
            for (i, p_raw) in probs.iter().enumerate() {
                let p = p_raw.clamp(0.0, 1.0);
                let u: f64 = rng.gen();
                if u < p {
                    counts[i] += 1;
                }
                if use_mirror && (1.0 - u) < p {
                    counts[i] += 1;
                }
            }
        }
    } else {
        for _ in 0..n_passes {
            for (i, p_raw) in probs.iter().enumerate() {
                let p = p_raw.clamp(0.0, 1.0);
                let u: f64 = rng.gen();
                if u < p {
                    counts[i] += 1;
                }
            }
        }
    }

    let denom = n_passes.max(1) as f64;
    let out = Array1::from_iter(counts.into_iter().map(|c| (c as f64) / denom));
    out.into_pyarray(py)
}

// ─── ML ───

/// Simulate a tearing mode plasma shot (full Modified Rutherford physics).
///
/// `seed` makes the trajectory reproducible; `beta_p`/`w_crit` parametrise the
/// bootstrap drive. Returns `(signal, label, time_to_disruption)`. The
/// deterministic per-step physics is bit-exact with the NumPy tier
/// (`scpn_fusion.control.disruption_risk_runtime.simulate_tearing_mode`); the
/// stochastic trajectory is statistically equivalent (independent RNG streams).
#[pyfunction]
#[pyo3(signature = (
    steps,
    seed = None,
    beta_p = fusion_ml::disruption::DEFAULT_BETA_P,
    w_crit = fusion_ml::disruption::DEFAULT_W_CRIT,
))]
fn simulate_tearing_mode(
    steps: usize,
    seed: Option<u64>,
    beta_p: f64,
    w_crit: f64,
) -> (Vec<f64>, u8, i64) {
    let shot = fusion_ml::disruption::simulate_tearing_mode(steps, seed, beta_p, w_crit);
    (shot.signal, shot.label, shot.time_to_disruption)
}

/// Deterministic Modified Rutherford island-width increment for one step.
///
/// `dw = (delta_prime + beta_p·w/(w² + w_crit²))·(1 - w/w_sat)·dt`. Bit-exact
/// with the NumPy tier's `rutherford_island_growth`.
#[pyfunction]
fn rutherford_island_growth(w: f64, delta_prime: f64, beta_p: f64, w_crit: f64, dt: f64) -> f64 {
    fusion_ml::disruption::rutherford_island_growth(w, delta_prime, beta_p, w_crit, dt)
}

// ─── Particle / Boris integrator ───

/// Python-accessible charged particle.
#[pyclass(from_py_object)]
#[derive(Clone)]
struct PyParticle {
    #[pyo3(get, set)]
    x_m: f64,
    #[pyo3(get, set)]
    y_m: f64,
    #[pyo3(get, set)]
    z_m: f64,
    #[pyo3(get, set)]
    vx_m_s: f64,
    #[pyo3(get, set)]
    vy_m_s: f64,
    #[pyo3(get, set)]
    vz_m_s: f64,
    #[pyo3(get, set)]
    charge_c: f64,
    #[pyo3(get, set)]
    mass_kg: f64,
    #[pyo3(get, set)]
    weight: f64,
}

#[pymethods]
impl PyParticle {
    fn kinetic_energy_mev(&self) -> f64 {
        self.to_internal().kinetic_energy_mev()
    }

    fn cylindrical_radius_m(&self) -> f64 {
        self.to_internal().cylindrical_radius_m()
    }

    fn __repr__(&self) -> String {
        format!(
            "Particle(E={:.2} MeV, R={:.3} m)",
            self.kinetic_energy_mev(),
            self.cylindrical_radius_m()
        )
    }
}

impl PyParticle {
    fn to_internal(&self) -> ChargedParticle {
        ChargedParticle {
            x_m: self.x_m,
            y_m: self.y_m,
            z_m: self.z_m,
            vx_m_s: self.vx_m_s,
            vy_m_s: self.vy_m_s,
            vz_m_s: self.vz_m_s,
            charge_c: self.charge_c,
            mass_kg: self.mass_kg,
            weight: self.weight,
        }
    }

    fn from_internal(p: &ChargedParticle) -> Self {
        PyParticle {
            x_m: p.x_m,
            y_m: p.y_m,
            z_m: p.z_m,
            vx_m_s: p.vx_m_s,
            vy_m_s: p.vy_m_s,
            vz_m_s: p.vz_m_s,
            charge_c: p.charge_c,
            mass_kg: p.mass_kg,
            weight: p.weight,
        }
    }
}

/// Population summary result.
#[pyclass(from_py_object)]
#[derive(Clone)]
struct PyPopulationSummary {
    #[pyo3(get)]
    count: usize,
    #[pyo3(get)]
    mean_energy_mev: f64,
    #[pyo3(get)]
    p95_energy_mev: f64,
    #[pyo3(get)]
    max_energy_mev: f64,
    #[pyo3(get)]
    runaway_fraction: f64,
}

#[pymethods]
impl PyPopulationSummary {
    fn __repr__(&self) -> String {
        format!(
            "PopulationSummary(n={}, E_mean={:.2} MeV)",
            self.count, self.mean_energy_mev
        )
    }
}

/// Seed alpha test particles.
#[pyfunction]
fn py_seed_alpha_particles(
    n: usize,
    r0: f64,
    z0: f64,
    energy_mev: f64,
    pitch: f64,
    weight: f64,
) -> PyResult<Vec<PyParticle>> {
    let particles = seed_alpha_test_particles(n, r0, z0, energy_mev, pitch, weight)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(particles.iter().map(PyParticle::from_internal).collect())
}

/// Advance particles using Boris integrator.
#[pyfunction]
fn py_advance_boris(
    particles: Vec<PyParticle>,
    e_field: [f64; 3],
    b_field: [f64; 3],
    dt: f64,
    steps: usize,
) -> PyResult<Vec<PyParticle>> {
    let mut internal: Vec<ChargedParticle> = particles.iter().map(|p| p.to_internal()).collect();
    advance_particles_boris(&mut internal, e_field, b_field, dt, steps)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(internal.iter().map(PyParticle::from_internal).collect())
}

/// Get alpha heating power density profile on R-Z grid.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn py_get_heating_profile<'py>(
    py: Python<'py>,
    particles: Vec<PyParticle>,
    nr: usize,
    nz: usize,
    r_min: f64,
    r_max: f64,
    z_min: f64,
    z_max: f64,
    confinement_tau_s: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let internal: Vec<ChargedParticle> = particles.iter().map(|p| p.to_internal()).collect();
    let grid = Grid2D::new(nr, nz, r_min, r_max, z_min, z_max);
    let profile = estimate_alpha_heating_profile(&internal, &grid, confinement_tau_s)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(profile.into_pyarray(py))
}

/// Summarize particle population statistics.
#[pyfunction]
fn py_particle_population_summary(
    particles: Vec<PyParticle>,
    threshold_mev: f64,
) -> PyResult<PyPopulationSummary> {
    let internal: Vec<ChargedParticle> = particles.iter().map(|p| p.to_internal()).collect();
    let summary = summarize_particle_population(&internal, threshold_mev)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPopulationSummary {
        count: summary.count,
        mean_energy_mev: summary.mean_energy_mev,
        p95_energy_mev: summary.p95_energy_mev,
        max_energy_mev: summary.max_energy_mev,
        runaway_fraction: summary.runaway_fraction,
    })
}

// ─── SNN controller ───

/// Python-accessible spiking controller pool (LIF neuron population).
#[pyclass]
struct PySnnPool {
    inner: SpikingControllerPool,
}

#[pymethods]
impl PySnnPool {
    #[new]
    #[pyo3(signature = (n_neurons=50, gain=10.0, window_size=20))]
    fn new(n_neurons: usize, gain: f64, window_size: usize) -> PyResult<Self> {
        let inner = SpikingControllerPool::new(n_neurons, gain, window_size)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PySnnPool { inner })
    }

    /// Process error signal through SNN population. Returns control output.
    fn step(&mut self, error: f64) -> PyResult<f64> {
        self.inner
            .step(error)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn n_neurons(&self) -> usize {
        self.inner.n_neurons
    }

    #[getter]
    fn gain(&self) -> f64 {
        self.inner.gain
    }

    fn __repr__(&self) -> String {
        format!(
            "PySnnPool(n_neurons={}, gain={:.1})",
            self.inner.n_neurons, self.inner.gain
        )
    }
}

/// Python-accessible neuro-cybernetic controller (dual R+Z SNN pools).
#[pyclass]
struct PySnnController {
    inner: NeuroCyberneticController,
}

#[pymethods]
impl PySnnController {
    #[new]
    fn new(target_r: f64, target_z: f64) -> PyResult<Self> {
        let inner = NeuroCyberneticController::new(target_r, target_z)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PySnnController { inner })
    }

    /// Process measured (R, Z) position. Returns (ctrl_R, ctrl_Z).
    fn step(&mut self, measured_r: f64, measured_z: f64) -> PyResult<(f64, f64)> {
        self.inner
            .step(measured_r, measured_z)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn target_r(&self) -> f64 {
        self.inner.target_r
    }

    #[getter]
    fn target_z(&self) -> f64 {
        self.inner.target_z
    }

    fn __repr__(&self) -> String {
        format!(
            "PySnnController(target_r={:.3}, target_z={:.3})",
            self.inner.target_r, self.inner.target_z
        )
    }
}

// ─── Extended PyO3 bridges (Rust extension plan) ───

#[pyclass]
struct PyHallMHD {
    inner: HallMHD,
}

#[pymethods]
impl PyHallMHD {
    #[new]
    #[pyo3(signature = (n=64, eta=None, nu=None, seed=None, background_amplitude=0.0))]
    fn new(
        n: usize,
        eta: Option<f64>,
        nu: Option<f64>,
        seed: Option<u64>,
        background_amplitude: f64,
    ) -> Self {
        Self {
            inner: HallMHD::configure(
                n,
                eta.unwrap_or(1.0e-4),
                nu.unwrap_or(1.0e-4),
                seed,
                background_amplitude,
            ),
        }
    }

    fn step(&mut self) -> (f64, f64) {
        self.inner.step()
    }

    fn run(&mut self, n_steps: usize) -> Vec<(f64, f64)> {
        self.inner.run(n_steps)
    }

    #[getter]
    fn energy_history<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.energy_history.clone()).into_pyarray(py)
    }

    #[getter]
    fn zonal_history<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.zonal_history.clone()).into_pyarray(py)
    }

    #[getter]
    fn grid_size(&self) -> usize {
        self.inner.n
    }

    #[getter]
    fn eta(&self) -> f64 {
        self.inner.eta
    }

    #[getter]
    fn nu(&self) -> f64 {
        self.inner.nu
    }
}

#[pyclass]
struct PyFnoController {
    inner: FnoController,
}

#[pymethods]
impl PyFnoController {
    #[new]
    fn new() -> Self {
        Self {
            inner: FnoController::new(),
        }
    }

    #[staticmethod]
    fn from_npz(path: &str) -> PyResult<Self> {
        let inner = FnoController::load_weights_npz(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        field: PyReadonlyArray2<'py, f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let out = self.inner.predict(&field.as_array().to_owned());
        out.into_pyarray(py)
    }

    fn predict_and_suppress<'py>(
        &self,
        py: Python<'py>,
        field: PyReadonlyArray2<'py, f64>,
    ) -> (f64, Bound<'py, PyArray2<f64>>) {
        let (suppression, prediction) = self
            .inner
            .predict_and_suppress(&field.as_array().to_owned());
        (suppression, prediction.into_pyarray(py))
    }
}

#[pyclass]
struct PyMpcController {
    inner: MPController,
}

#[pymethods]
impl PyMpcController {
    #[new]
    fn new(
        b_matrix: PyReadonlyArray2<'_, f64>,
        target: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let surrogate = NeuralSurrogate::new(b_matrix.as_array().to_owned());
        let inner = MPController::new(surrogate, target.as_array().to_owned())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn plan<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let result = self
            .inner
            .plan(&state.as_array().to_owned())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }
}

#[pyclass]
struct PyPlasma2D {
    inner: Plasma2D,
}

#[pymethods]
impl PyPlasma2D {
    #[new]
    fn new() -> Self {
        Self {
            inner: Plasma2D::new(),
        }
    }

    fn step(&mut self, action: f64) -> PyResult<(f64, f64)> {
        self.inner
            .step(action)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn measure_core_temp(&self, noise: f64) -> PyResult<f64> {
        self.inner
            .measure_core_temp(noise)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

/// Build a Python dict mirroring `GlobalDesignExplorer.evaluate_design` output.
fn design_result_to_dict<'py>(
    py: pyo3::Python<'py>,
    d: &design_scanner::DesignResult,
) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
    let out = pyo3::types::PyDict::new(py);
    out.set_item("R", d.r_major)?;
    out.set_item("B", d.b_field)?;
    out.set_item("Ip", d.i_plasma)?;
    out.set_item("Model_Regime", "physics_scaling_surrogate")?;
    out.set_item("P_fus", d.p_fusion)?;
    out.set_item("Q", d.q_engineering)?;
    out.set_item("Wall_Load", d.wall_load)?;
    out.set_item("Div_Load_Baseline", d.div_load_baseline)?;
    out.set_item("Shadow_Fraction", d.shadow_fraction)?;
    out.set_item("Div_Load_Optimized", d.div_load_optimized)?;
    out.set_item("Div_Load", d.div_load_optimized)?;
    out.set_item("B_peak_HTS_T", d.b_peak_hts_t)?;
    out.set_item("Zeff_Est", d.zeff_est)?;
    out.set_item("Constraint_OK", d.constraint_ok)?;
    out.set_item("beta_N_eff", d.beta_n_eff)?;
    out.set_item("Cost", d.cost)?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (r, b, i_p, divertor_flux_cap_mw_m2=45.0, zeff_cap=0.4, hts_peak_cap_t=21.0))]
fn py_evaluate_design<'py>(
    py: pyo3::Python<'py>,
    r: f64,
    b: f64,
    i_p: f64,
    divertor_flux_cap_mw_m2: f64,
    zeff_cap: f64,
    hts_peak_cap_t: f64,
) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
    let d = design_scanner::evaluate_design_with_caps(
        r,
        b,
        i_p,
        divertor_flux_cap_mw_m2,
        zeff_cap,
        hts_peak_cap_t,
    );
    design_result_to_dict(py, &d)
}

#[pyfunction]
fn py_run_design_scan<'py>(
    py: pyo3::Python<'py>,
    n_samples: usize,
) -> pyo3::PyResult<Vec<pyo3::Bound<'py, pyo3::types::PyDict>>> {
    design_scanner::run_scan(n_samples)
        .iter()
        .map(|d| design_result_to_dict(py, d))
        .collect()
}

#[pyclass]
struct PyDriftWave {
    inner: DriftWavePhysics,
}

#[pymethods]
impl PyDriftWave {
    #[new]
    #[pyo3(signature = (n=64, seed=None))]
    fn new(n: usize, seed: Option<u64>) -> Self {
        let inner = match seed {
            Some(seed) => DriftWavePhysics::with_seed(n, seed),
            None => DriftWavePhysics::new(n),
        };
        Self { inner }
    }

    fn step<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.step()).into_pyarray(py)
    }
}

// ─── Fokker-Planck RE solver ───

#[pyclass]
struct PyFokkerPlanckSolver {
    inner: FokkerPlanckSolver,
}

#[pymethods]
impl PyFokkerPlanckSolver {
    #[new]
    #[pyo3(signature = (np_grid=200, p_max=100.0))]
    fn new(np_grid: usize, p_max: f64) -> Self {
        Self {
            inner: FokkerPlanckSolver::new(np_grid, p_max),
        }
    }

    fn step(&mut self, dt: f64, e_field: f64, n_e: f64, t_e_ev: f64, z_eff: f64) -> (f64, f64) {
        let state = self.inner.step(dt, e_field, n_e, t_e_ev, z_eff);
        (state.n_re, state.current_re)
    }

    fn run(
        &mut self,
        n_steps: usize,
        dt: f64,
        e_field: f64,
        n_e: f64,
        t_e_ev: f64,
        z_eff: f64,
    ) -> Vec<(f64, f64)> {
        self.inner
            .run(n_steps, dt, e_field, n_e, t_e_ev, z_eff)
            .into_iter()
            .map(|s| (s.n_re, s.current_re))
            .collect()
    }

    fn get_f<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.f.clone()).into_pyarray(py)
    }

    fn set_f(&mut self, values: Vec<f64>) {
        self.inner.f = values;
    }

    fn get_p<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.p.clone()).into_pyarray(py)
    }

    fn get_dp<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.dp.clone()).into_pyarray(py)
    }

    #[getter]
    fn time(&self) -> f64 {
        self.inner.time
    }
}

// ─── SPI Fragment Ablation ───

#[pyclass]
struct PySpiAblationSolver {
    inner: SpiAblationSolver,
}

#[pymethods]
impl PySpiAblationSolver {
    #[new]
    #[pyo3(signature = (n_fragments=100, total_mass_kg=0.01, velocity_mps=200.0, dispersion=0.1))]
    fn new(
        n_fragments: usize,
        total_mass_kg: f64,
        velocity_mps: f64,
        dispersion: f64,
    ) -> PyResult<Self> {
        let inner = SpiAblationSolver::new(n_fragments, total_mass_kg, velocity_mps, dispersion)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        dt: f64,
        plasma_ne: Vec<f64>,
        plasma_te: Vec<f64>,
        r_grid: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let dep = self
            .inner
            .step(dt, &plasma_ne, &plasma_te, &r_grid)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Array1::from_vec(dep).into_pyarray(py))
    }

    fn n_active(&self) -> usize {
        self.inner.n_active()
    }

    fn total_mass(&self) -> f64 {
        self.inner.total_mass()
    }
}

// ─── Reduced MHD Sawtooth ───

#[pyclass]
struct PyReducedMHD {
    inner: ReducedMHD,
}

#[pymethods]
impl PyReducedMHD {
    #[new]
    fn new() -> Self {
        Self {
            inner: ReducedMHD::new(),
        }
    }

    fn step(&mut self, dt: f64) -> (f64, bool) {
        self.inner.step(dt)
    }

    fn run(&mut self, n_steps: usize) -> Vec<(f64, bool)> {
        self.inner.run(n_steps)
    }

    #[getter]
    fn crash_count(&self) -> usize {
        self.inner.crash_count
    }

    fn amplitude_history<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.amplitude_history.clone()).into_pyarray(py)
    }
}

fn ensure_matching_length(name: &str, expected: usize, actual: usize) -> PyResult<()> {
    if expected != actual {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} length mismatch: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

#[pyclass]
struct PyTransportSolver {
    inner: TransportSolver,
}

#[pymethods]
impl PyTransportSolver {
    #[new]
    fn new() -> Self {
        Self {
            inner: TransportSolver::new(),
        }
    }

    fn evolve_profiles(&mut self, p_aux_mw: f64) -> PyResult<()> {
        self.inner
            .evolve_profiles(p_aux_mw)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn transport_step(&mut self, p_aux_mw: f64, dt: f64) -> PyResult<()> {
        transport::transport_step(&mut self.inner, p_aux_mw, dt)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn chang_hinton_chi_profile<'py>(
        &self,
        py: Python<'py>,
        rho: PyReadonlyArray1<'py, f64>,
        t_i_kev: PyReadonlyArray1<'py, f64>,
        n_e_19: PyReadonlyArray1<'py, f64>,
        q: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let rho_arr = rho.as_array().to_owned();
        let t_i_arr = t_i_kev.as_array().to_owned();
        let n_e_arr = n_e_19.as_array().to_owned();
        let q_arr = q.as_array().to_owned();
        let n = rho_arr.len();
        ensure_matching_length("t_i_kev", n, t_i_arr.len())?;
        ensure_matching_length("n_e_19", n, n_e_arr.len())?;
        ensure_matching_length("q", n, q_arr.len())?;

        let params = NeoclassicalParams {
            q_profile: q_arr.clone(),
            ..NeoclassicalParams::default()
        };
        let chi =
            transport::chang_hinton_chi_profile(&rho_arr, &t_i_arr, &n_e_arr, &q_arr, &params);
        Ok(chi.into_pyarray(py))
    }

    #[allow(clippy::too_many_arguments)]
    fn sauter_bootstrap_profile<'py>(
        &self,
        py: Python<'py>,
        rho: PyReadonlyArray1<'py, f64>,
        t_e_kev: PyReadonlyArray1<'py, f64>,
        t_i_kev: PyReadonlyArray1<'py, f64>,
        n_e_19: PyReadonlyArray1<'py, f64>,
        q: PyReadonlyArray1<'py, f64>,
        epsilon: PyReadonlyArray1<'py, f64>,
        b_field: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let rho_arr = rho.as_array().to_owned();
        let t_e_arr = t_e_kev.as_array().to_owned();
        let t_i_arr = t_i_kev.as_array().to_owned();
        let n_e_arr = n_e_19.as_array().to_owned();
        let q_arr = q.as_array().to_owned();
        let eps_arr = epsilon.as_array().to_owned();
        let n = rho_arr.len();
        ensure_matching_length("t_e_kev", n, t_e_arr.len())?;
        ensure_matching_length("t_i_kev", n, t_i_arr.len())?;
        ensure_matching_length("n_e_19", n, n_e_arr.len())?;
        ensure_matching_length("q", n, q_arr.len())?;
        ensure_matching_length("epsilon", n, eps_arr.len())?;

        let j_bs = transport::sauter_bootstrap_current_profile(
            &rho_arr, &t_e_arr, &t_i_arr, &n_e_arr, &q_arr, &eps_arr, b_field,
        );
        Ok(j_bs.into_pyarray(py))
    }
}

#[cfg(feature = "gpu")]
mod gpu_bindings {
    use super::*;
    use fusion_gpu::GpuGsSolver;

    #[pyclass]
    pub struct PyGpuSolver {
        inner: GpuGsSolver,
    }

    #[pymethods]
    impl PyGpuSolver {
        #[new]
        fn new(
            nr: usize,
            nz: usize,
            r_left: f64,
            r_right: f64,
            z_bottom: f64,
            z_top: f64,
        ) -> PyResult<Self> {
            let inner = GpuGsSolver::new(nr, nz, r_left, r_right, z_bottom, z_top)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self { inner })
        }

        fn solve<'py>(
            &self,
            py: Python<'py>,
            psi: Vec<f32>,
            source: Vec<f32>,
            iterations: usize,
            omega: f32,
        ) -> PyResult<Bound<'py, PyArray1<f32>>> {
            let result = self
                .inner
                .solve_full(&psi, &source, iterations, omega)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(Array1::from_vec(result).into_pyarray(py))
        }

        fn grid_shape(&self) -> (usize, usize) {
            self.inner.grid_shape()
        }

        fn vcycle<'py>(
            &self,
            py: Python<'py>,
            psi: Vec<f32>,
            source: Vec<f32>,
            pre_sweeps: usize,
            post_sweeps: usize,
            omega: f32,
        ) -> PyResult<Bound<'py, PyArray1<f32>>> {
            self.inner
                .upload(&psi, &source)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            self.inner
                .vcycle(pre_sweeps, post_sweeps, omega)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let result = self
                .inner
                .download()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(Array1::from_vec(result).into_pyarray(py))
        }
    }

    #[pyfunction]
    pub fn py_gpu_available() -> bool {
        fusion_gpu::gpu_available()
    }

    #[pyfunction]
    pub fn py_gpu_info() -> Option<String> {
        fusion_gpu::gpu_info()
    }
}

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

#[pyclass(from_py_object)]
#[derive(Clone, Copy)]
enum PyPacingMode {
    Sleep,
    Spin,
}

#[pyclass(from_py_object)]
#[derive(Clone, Copy)]
struct PyRmfAotCertificate {
    #[pyo3(get, set)]
    pub max_freq_hz: f64,
    #[pyo3(get, set)]
    pub min_freq_hz: f64,
    #[pyo3(get, set)]
    pub max_phase_error: f64,
}

#[pymethods]
impl PyRmfAotCertificate {
    #[new]
    #[pyo3(signature = (max_freq_hz=5.0e6, min_freq_hz=1.0e5, max_phase_error=std::f64::consts::FRAC_PI_2))]
    fn new(max_freq_hz: f64, min_freq_hz: f64, max_phase_error: f64) -> Self {
        Self {
            max_freq_hz,
            min_freq_hz,
            max_phase_error,
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Copy)]
struct PyRmfConfig {
    #[pyo3(get, set)]
    pub f_rmf_nom_hz: f64,
    #[pyo3(get, set)]
    pub f_sampling_hz: f64,
    #[pyo3(get, set)]
    pub k_p: f64,
    #[pyo3(get, set)]
    pub k_d: f64,
    #[pyo3(get, set)]
    pub n_neurons: usize,
    #[pyo3(get, set)]
    pub aot_safety: PyRmfAotCertificate,
}

#[pymethods]
impl PyRmfConfig {
    #[new]
    #[pyo3(signature = (f_rmf_nom_hz=1.0e6, f_sampling_hz=10.0e6, k_p=0.5, k_d=0.01, n_neurons=128, aot_safety=PyRmfAotCertificate::new(5.0e6, 1.0e5, std::f64::consts::FRAC_PI_2)))]
    fn new(
        f_rmf_nom_hz: f64,
        f_sampling_hz: f64,
        k_p: f64,
        k_d: f64,
        n_neurons: usize,
        aot_safety: PyRmfAotCertificate,
    ) -> Self {
        Self {
            f_rmf_nom_hz,
            f_sampling_hz,
            k_p,
            k_d,
            n_neurons,
            aot_safety,
        }
    }
}

#[pyclass]
struct PyRmfController {
    inner: RmfPhaseLockController,
}

#[pymethods]
impl PyRmfController {
    #[new]
    fn new(config: PyRmfConfig) -> Self {
        use fusion_physics::rmf_control::RmfAotCertificate;
        let cfg = RmfConfig {
            f_rmf_nom_hz: config.f_rmf_nom_hz,
            f_sampling_hz: config.f_sampling_hz,
            k_p: config.k_p,
            k_d: config.k_d,
            n_neurons: config.n_neurons,
            aot_safety: RmfAotCertificate {
                max_freq_hz: config.aot_safety.max_freq_hz,
                min_freq_hz: config.aot_safety.min_freq_hz,
                max_phase_error: config.aot_safety.max_phase_error,
            },
        };
        Self {
            inner: RmfPhaseLockController::new(cfg),
        }
    }

    fn enable_pacing(&mut self, mode: PyPacingMode) {
        use fusion_physics::precision_pacer::PacingMode;
        let m = match mode {
            PyPacingMode::Sleep => PacingMode::Sleep,
            PyPacingMode::Spin => PacingMode::Spin,
        };
        self.inner.enable_pacing(m);
    }

    #[getter]
    fn safety_violations(&self) -> u64 {
        self.inner.safety_violations
    }

    fn step(&mut self, phi_plasma: f64) -> f64 {
        self.inner.step(phi_plasma)
    }

    fn step_horizon<'py>(
        &mut self,
        py: Python<'py>,
        phi_plasma_traj: PyReadonlyArray1<'py, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let traj = phi_plasma_traj.as_array().to_owned();
        let out = self.inner.step_horizon(traj.as_slice().unwrap());
        Array1::from_vec(out).into_pyarray(py)
    }

    #[getter]
    fn phi_ant(&self) -> f64 {
        self.inner.phi_ant
    }

    #[getter]
    fn omega_rmf(&self) -> f64 {
        self.inner.omega_rmf
    }
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
