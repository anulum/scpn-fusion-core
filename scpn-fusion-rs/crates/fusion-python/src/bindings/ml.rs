// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
//! PyO3 bindings for the machine-learning surrogate lane (`fusion-ml`).
//!
//! Exposes the neural transport surrogate (`PyNeuralTransport`, a
//! 10 → 64 → 32 → 3 network) to Python, mirroring the NumPy tier's predict API.

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use fusion_ml::neural_transport::NeuralTransportModel;

/// Python wrapper for neural transport surrogate (10 -> 64 -> 32 -> 3).
#[pyclass]
pub(crate) struct PyNeuralTransport {
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
