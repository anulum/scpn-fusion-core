// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
//! PyO3 bindings for the rotating-magnetic-field control lane (`fusion-physics`).
//!
//! Exposes the RMF phase-lock controller (`PyRmfController`) with its
//! configuration (`PyRmfConfig`), ahead-of-time safety certificate
//! (`PyRmfAotCertificate`), and precision-pacing mode (`PyPacingMode`) to
//! Python, mirroring the NumPy tier's control API.

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use fusion_physics::rmf_control::{RmfConfig, RmfPhaseLockController};

#[pyclass(from_py_object)]
#[derive(Clone, Copy)]
pub(crate) enum PyPacingMode {
    Sleep,
    Spin,
}

#[pyclass(from_py_object)]
#[derive(Clone, Copy)]
pub(crate) struct PyRmfAotCertificate {
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
pub(crate) struct PyRmfConfig {
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
pub(crate) struct PyRmfController {
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
