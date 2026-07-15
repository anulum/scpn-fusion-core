// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
//! PyO3 bindings for the control lane.
//!
//! Exposes the Fourier neural-operator disruption controller and the
//! model-predictive controller (over a neural surrogate) to Python, mirroring
//! the array keyword APIs of the Rust `fusion-control`/`fusion-physics` tiers.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use fusion_control::mpc::{MPController, NeuralSurrogate};
use fusion_physics::fno::FnoController;

#[pyclass]
pub(crate) struct PyFnoController {
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
pub(crate) struct PyMpcController {
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
