// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Fusion Python
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! PyO3 Python bindings for SCPN Fusion Core.
//!
//! Stage 10: Exposes Grad-Shafranov solver, thermodynamics, control,
//! diagnostics, and ML modules to Python via PyO3 + numpy.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use fusion_core::ignition::calculate_thermodynamics;
use fusion_core::kernel::FusionKernel;
// ReactorConfig used internally by FusionKernel::from_file

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
    fn calculate_thermodynamics(&self, p_aux_mw: f64) -> PyThermodynamicsResult {
        let result = calculate_thermodynamics(&self.inner, p_aux_mw);
        PyThermodynamicsResult {
            p_fusion_mw: result.p_fusion_mw,
            p_alpha_mw: result.p_alpha_mw,
            p_loss_mw: result.p_loss_mw,
            p_aux_mw: result.p_aux_mw,
            net_mw: result.net_mw,
            q_factor: result.q_factor,
            t_peak_kev: result.t_peak_kev,
            w_thermal_mj: result.w_thermal_mj,
        }
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
}

// ─── Result types ───

/// Equilibrium solve result.
#[pyclass]
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
#[pyclass]
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

// ─── Control systems ───

/// Shafranov equilibrium calculator.
#[pyfunction]
fn shafranov_bv(r_geo: f64, a_min: f64, ip_ma: f64) -> (f64, f64, f64) {
    let result = fusion_control::analytic::shafranov_bv(r_geo, a_min, ip_ma);
    (result.bv_required, result.term_log, result.term_physics)
}

/// Minimum-norm coil current solver.
#[pyfunction]
fn solve_coil_currents(green_func: Vec<f64>, target_bv: f64) -> Vec<f64> {
    fusion_control::analytic::solve_coil_currents(&green_func, target_bv)
}

// ─── Diagnostics ───

/// Create sensor suite and measure magnetics from Psi array.
#[pyfunction]
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

// ─── ML ───

/// Simulate a tearing mode plasma shot.
#[pyfunction]
fn simulate_tearing_mode(steps: usize) -> (Vec<f64>, u8, i64) {
    let shot = fusion_ml::disruption::simulate_tearing_mode(steps);
    (shot.signal, shot.label, shot.time_to_disruption)
}

// ─── Module registration ───

/// SCPN Fusion Core — Rust-accelerated plasma physics.
#[pymodule]
fn scpn_fusion_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFusionKernel>()?;
    m.add_class::<PyEquilibriumResult>()?;
    m.add_class::<PyThermodynamicsResult>()?;
    m.add_function(wrap_pyfunction!(shafranov_bv, m)?)?;
    m.add_function(wrap_pyfunction!(solve_coil_currents, m)?)?;
    m.add_function(wrap_pyfunction!(measure_magnetics, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_tearing_mode, m)?)?;
    Ok(())
}
