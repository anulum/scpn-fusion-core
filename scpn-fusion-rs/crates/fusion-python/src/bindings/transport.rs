// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
//! PyO3 bindings for the transport lane.
//!
//! Exposes the 2-D digital-twin plasma (`PyPlasma2D`), the global design
//! explorer (`py_evaluate_design`, `py_run_design_scan`), drift-wave turbulence
//! (`PyDriftWave`), the Fokker-Planck runaway-electron solver
//! (`PyFokkerPlanckSolver`), the SPI fragment-ablation solver
//! (`PySpiAblationSolver`), and the neoclassical transport solver
//! (`PyTransportSolver`) to Python, mirroring the NumPy tier's APIs.

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use fusion_control::digital_twin::Plasma2D;
use fusion_control::spi_ablation::SpiAblationSolver;
use fusion_core::transport::{self, NeoclassicalParams, TransportSolver};
use fusion_physics::design_scanner;
use fusion_physics::fokker_planck::FokkerPlanckSolver;
use fusion_physics::turbulence::DriftWavePhysics;

#[pyclass]
pub(crate) struct PyPlasma2D {
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
pub(crate) fn py_evaluate_design<'py>(
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
pub(crate) fn py_run_design_scan<'py>(
    py: pyo3::Python<'py>,
    n_samples: usize,
) -> pyo3::PyResult<Vec<pyo3::Bound<'py, pyo3::types::PyDict>>> {
    design_scanner::run_scan(n_samples)
        .iter()
        .map(|d| design_result_to_dict(py, d))
        .collect()
}

#[pyclass]
pub(crate) struct PyDriftWave {
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

#[pyclass]
pub(crate) struct PyFokkerPlanckSolver {
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

#[pyclass]
pub(crate) struct PySpiAblationSolver {
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

fn ensure_matching_length(name: &str, expected: usize, actual: usize) -> PyResult<()> {
    if expected != actual {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} length mismatch: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

#[pyclass]
pub(crate) struct PyTransportSolver {
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
