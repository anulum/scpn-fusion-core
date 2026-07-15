// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
//! PyO3 bindings for the Grad-Shafranov equilibrium lane.
//!
//! Exposes the forward equilibrium solver (`PyFusionKernel`) and its result
//! records (`PyEquilibriumResult`, `PyThermodynamicsResult`), the inverse
//! equilibrium reconstructor (`PyInverseSolver`, `PyInverseResult`), the
//! analytic Shafranov / coil-current helpers, magnetics sensing, and the
//! standalone geometric-multigrid GS* solve to Python, mirroring the NumPy tier.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use fusion_core::ignition::calculate_thermodynamics;
use fusion_core::inverse::{reconstruct_equilibrium, InverseConfig, JacobianMode};
use fusion_core::kernel::FusionKernel;
use fusion_core::source::ProfileParams;
use fusion_types::state::Grid2D;

/// Python-accessible Grad-Shafranov equilibrium solver.
#[pyclass]
pub(crate) struct PyFusionKernel {
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

/// Equilibrium solve result.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub(crate) struct PyEquilibriumResult {
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
pub(crate) struct PyThermodynamicsResult {
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

#[pyclass(from_py_object)]
#[derive(Clone)]
pub(crate) struct PyInverseResult {
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
pub(crate) struct PyInverseSolver;

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
pub(crate) fn shafranov_bv(
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
pub(crate) fn solve_coil_currents(
    green_func: Vec<f64>,
    target_bv: f64,
    ridge_lambda: f64,
) -> PyResult<Vec<f64>> {
    fusion_control::analytic::solve_coil_currents(&green_func, target_bv, ridge_lambda)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Create sensor suite and measure magnetics from Psi array.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn measure_magnetics<'py>(
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
pub(crate) fn multigrid_vcycle<'py>(
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
