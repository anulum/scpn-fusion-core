// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
//! PyO3 bindings for the MHD lane.
//!
//! Exposes tearing-mode disruption physics (Modified Rutherford island growth
//! and a full tearing-mode shot), the Hall-MHD turbulence solver, and the
//! reduced-MHD sawtooth model to Python, mirroring the NumPy tier's APIs.

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use fusion_physics::hall_mhd::HallMHD;
use fusion_physics::sawtooth::ReducedMHD;

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
pub(crate) fn simulate_tearing_mode(
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
pub(crate) fn rutherford_island_growth(
    w: f64,
    delta_prime: f64,
    beta_p: f64,
    w_crit: f64,
    dt: f64,
) -> f64 {
    fusion_ml::disruption::rutherford_island_growth(w, delta_prime, beta_p, w_crit, dt)
}

#[pyclass]
pub(crate) struct PyHallMHD {
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
pub(crate) struct PyReducedMHD {
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
