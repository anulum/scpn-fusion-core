// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
//! PyO3 bindings for the diagnostics lane (`fusion-diagnostics`).
//!
//! Exposes the regularised plasma tomography reconstructor to Python, mirroring
//! the chord-geometry keyword API of the Rust `PlasmaTomography` solver.

use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

use fusion_diagnostics::tomography::PlasmaTomography;

#[pyclass]
pub(crate) struct PyTomography {
    inner: PlasmaTomography,
}

#[pymethods]
impl PyTomography {
    #[new]
    #[pyo3(signature = (chords, r_range, z_range, res, lambda_reg=0.1))]
    fn new(
        chords: Vec<((f64, f64), (f64, f64))>,
        r_range: (f64, f64),
        z_range: (f64, f64),
        res: usize,
        lambda_reg: f64,
    ) -> Self {
        Self {
            inner: PlasmaTomography::with_lambda(&chords, r_range, z_range, res, lambda_reg),
        }
    }

    fn reconstruct<'py>(&self, py: Python<'py>, signals: Vec<f64>) -> Bound<'py, PyArray2<f64>> {
        self.inner.reconstruct_2d(&signals).into_pyarray(py)
    }

    #[getter]
    fn lambda_reg(&self) -> f64 {
        self.inner.lambda_reg
    }

    #[getter]
    fn res(&self) -> usize {
        self.inner.res
    }
}
