// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
//! PyO3 bindings for the GPU lane (`fusion-gpu`, `gpu` feature).
//!
//! Exposes the GPU-accelerated Grad-Shafranov solver (`PyGpuSolver`) and the
//! GPU availability/info probes to Python. Compiled only under the `gpu`
//! feature; the crate root gates registration on the same feature.

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

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
