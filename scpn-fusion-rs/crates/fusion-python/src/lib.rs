//! PyO3 Python bindings for SCPN Fusion Core.
//!
//! Stage 10: Exposes Grad-Shafranov solver, thermodynamics, control,
//! diagnostics, and ML modules to Python via PyO3 + numpy.

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use fusion_core::ignition::calculate_thermodynamics;
use fusion_core::inverse::{reconstruct_equilibrium, InverseConfig, JacobianMode};
use fusion_core::kernel::FusionKernel;
use fusion_core::particles::{
    advance_particles_boris, estimate_alpha_heating_profile, seed_alpha_test_particles,
    summarize_particle_population, ChargedParticle,
};
use fusion_core::source::ProfileParams;
use fusion_engineering::blanket::neutron_wall_loading;
use fusion_engineering::layout::{
    aries_cost_scaling, cost_of_electricity as engineering_coe, scan_major_radius,
};
use fusion_engineering::tritium::tritium_breeding_ratio;
use fusion_ml::neural_transport::NeuralTransportModel;
use fusion_types::state::Grid2D;
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

// ─── Inverse solver ───

#[pyclass]
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
#[pyfunction]
fn shafranov_bv(r_geo: f64, a_min: f64, ip_ma: f64) -> PyResult<(f64, f64, f64)> {
    let result = fusion_control::analytic::shafranov_bv(r_geo, a_min, ip_ma)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok((result.bv_required, result.term_log, result.term_physics))
}

/// Minimum-norm coil current solver.
#[pyfunction]
fn solve_coil_currents(green_func: Vec<f64>, target_bv: f64) -> PyResult<Vec<f64>> {
    fusion_control::analytic::solve_coil_currents(&green_func, target_bv)
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

// ─── SCPN runtime kernels ───

/// Dense activation kernel for SCPN controller path.
#[pyfunction]
fn scpn_dense_activations<'py>(
    py: Python<'py>,
    w_in: PyReadonlyArray2<'py, f64>,
    marking: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let w = w_in.as_array();
    let m = marking.as_array();
    let out = w.dot(&m);
    out.into_pyarray(py)
}

/// Dense marking update kernel: clip(m - W_in^T f + W_out f, 0, 1).
#[pyfunction]
fn scpn_marking_update<'py>(
    py: Python<'py>,
    marking: PyReadonlyArray1<'py, f64>,
    w_in: PyReadonlyArray2<'py, f64>,
    w_out: PyReadonlyArray2<'py, f64>,
    firing: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let m = marking.as_array();
    let wi = w_in.as_array();
    let wo = w_out.as_array();
    let f = firing.as_array();

    let cons = wi.t().dot(&f);
    let prod = wo.dot(&f);
    let out = (&m - &cons + &prod).mapv(|x| x.clamp(0.0, 1.0));
    out.into_pyarray(py)
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

/// Simulate a tearing mode plasma shot.
#[pyfunction]
fn simulate_tearing_mode(steps: usize) -> (Vec<f64>, u8, i64) {
    let shot = fusion_ml::disruption::simulate_tearing_mode(steps);
    (shot.signal, shot.label, shot.time_to_disruption)
}

// ─── Particle / Boris integrator ───

/// Python-accessible charged particle.
#[pyclass]
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
#[pyclass]
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

// ─── Module registration ───

/// SCPN Fusion Core — Rust-accelerated plasma physics.
#[pymodule]
fn scpn_fusion_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFusionKernel>()?;
    m.add_class::<PyEquilibriumResult>()?;
    m.add_class::<PyThermodynamicsResult>()?;
    m.add_class::<PyNeuralTransport>()?;
    m.add_class::<PyInverseSolver>()?;
    m.add_class::<PyInverseResult>()?;
    m.add_class::<PyPlantModel>()?;
    m.add_function(wrap_pyfunction!(shafranov_bv, m)?)?;
    m.add_function(wrap_pyfunction!(solve_coil_currents, m)?)?;
    m.add_function(wrap_pyfunction!(measure_magnetics, m)?)?;
    m.add_function(wrap_pyfunction!(scpn_dense_activations, m)?)?;
    m.add_function(wrap_pyfunction!(scpn_marking_update, m)?)?;
    m.add_function(wrap_pyfunction!(scpn_sample_firing, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_tearing_mode, m)?)?;
    // Particle / Boris integrator bridge
    m.add_class::<PyParticle>()?;
    m.add_class::<PyPopulationSummary>()?;
    m.add_function(wrap_pyfunction!(py_seed_alpha_particles, m)?)?;
    m.add_function(wrap_pyfunction!(py_advance_boris, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_heating_profile, m)?)?;
    m.add_function(wrap_pyfunction!(py_particle_population_summary, m)?)?;
    Ok(())
}
