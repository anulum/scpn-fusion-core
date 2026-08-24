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

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use fusion_ml::deeponet_equilibrium::{DeepOnetEquilibrium, DeepOnetEquilibriumConfig, DenseLayer};
use fusion_ml::neural_transport::NeuralTransportModel;

/// Python wrapper for the fixed-grid native equilibrium DeepONet runtime.
#[pyclass]
pub(crate) struct PyDeepOnetEquilibrium {
    inner: DeepOnetEquilibrium,
}

#[pymethods]
impl PyDeepOnetEquilibrium {
    #[new]
    fn new(
        branch_payload: (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>),
        trunk_payload: (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>),
        input_normalisation: (Vec<f64>, Vec<f64>),
        coordinate_contract: (Vec<Vec<f64>>, Vec<f64>, Vec<f64>),
        field_contract: (Vec<f64>, f64, usize, usize, usize),
    ) -> PyResult<Self> {
        let (branch_weights, branch_biases) = branch_payload;
        let (trunk_weights, trunk_biases) = trunk_payload;
        let (input_mean, input_std) = input_normalisation;
        let (coordinates_rz_m, coordinate_mean, coordinate_std) = coordinate_contract;
        let (field_mean, field_scale, basis_width, grid_nh, grid_nw) = field_contract;
        let branch = dense_layers(branch_weights, branch_biases, "branch")?;
        let trunk = dense_layers(trunk_weights, trunk_biases, "trunk")?;
        let coordinates = rectangular_array(coordinates_rz_m, "coordinates_rz_m")?;
        let inner = DeepOnetEquilibrium::new(DeepOnetEquilibriumConfig {
            branch,
            trunk,
            input_mean: Array1::from_vec(input_mean),
            input_std: Array1::from_vec(input_std),
            coordinates_rz_m: coordinates,
            coordinate_mean: Array1::from_vec(coordinate_mean),
            coordinate_std: Array1::from_vec(coordinate_std),
            field_mean: Array1::from_vec(field_mean),
            field_scale,
            basis_width,
            grid_shape: (grid_nh, grid_nw),
        })
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }

    fn predict_batch<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let output = self
            .inner
            .predict_batch(&features.as_array().to_owned())
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(output.into_pyarray(py))
    }

    #[getter]
    fn grid_shape(&self) -> (usize, usize) {
        self.inner.grid_shape()
    }
}

fn dense_layers(
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<f64>>,
    name: &str,
) -> PyResult<Vec<DenseLayer>> {
    if weights.len() != biases.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "DeepONet {name} weight/bias layer counts differ"
        )));
    }
    weights
        .into_iter()
        .zip(biases)
        .enumerate()
        .map(|(index, (weight, bias))| {
            Ok(DenseLayer {
                weights: rectangular_array(weight, &format!("{name}_{index}_W"))?,
                bias: Array1::from_vec(bias),
            })
        })
        .collect()
}

fn rectangular_array(rows: Vec<Vec<f64>>, name: &str) -> PyResult<Array2<f64>> {
    let n_rows = rows.len();
    let n_cols = rows.first().map(Vec::len).unwrap_or(0);
    if n_rows == 0 || n_cols == 0 || rows.iter().any(|row| row.len() != n_cols) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "DeepONet {name} must be a non-empty rectangular matrix"
        )));
    }
    let values = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((n_rows, n_cols), values)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

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
