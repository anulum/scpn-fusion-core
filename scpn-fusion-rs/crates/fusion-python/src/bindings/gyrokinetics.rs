// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
//! PyO3 bindings for the nonlinear gyrokinetics lane (`fusion-physics`).
//!
//! Exposes the nonlinear gyrokinetic solver (`PyNonlinearGKSolver`) — an RK4
//! step over the 6-D distribution function and 3-D potential — to Python,
//! mirroring the NumPy tier's complex-field API.

use ndarray::{Ix3, Ix6};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

use fusion_physics::gk_nonlinear::{NonlinearGKConfig, NonlinearGKSolver, NonlinearGKState};

#[pyclass]
pub struct PyNonlinearGKSolver {
    inner: NonlinearGKSolver,
}

type PyComplexFieldPair<'py> = (
    Bound<'py, PyArrayDyn<Complex64>>,
    Bound<'py, PyArrayDyn<Complex64>>,
);

#[pymethods]
impl PyNonlinearGKSolver {
    #[new]
    fn new() -> PyResult<Self> {
        let cfg = NonlinearGKConfig::default();
        Ok(PyNonlinearGKSolver {
            inner: NonlinearGKSolver::new(cfg),
        })
    }

    fn step<'py>(
        &self,
        py: Python<'py>,
        f_py: PyReadonlyArrayDyn<'py, Complex64>,
        phi_py: PyReadonlyArrayDyn<'py, Complex64>,
        time: f64,
        dt: f64,
    ) -> PyResult<PyComplexFieldPair<'py>> {
        let f = f_py
            .as_array()
            .into_dimensionality::<Ix6>()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .to_owned();
        let phi = phi_py
            .as_array()
            .into_dimensionality::<Ix3>()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .to_owned();
        let state = NonlinearGKState {
            f,
            phi,
            time,
            a_par: None,
        };
        let new_state = self.inner.rk4_step(&state, dt);
        let f_out = new_state.f.into_dyn().into_pyarray(py);
        let phi_out = new_state.phi.into_dyn().into_pyarray(py);
        Ok((f_out, phi_out))
    }
}
