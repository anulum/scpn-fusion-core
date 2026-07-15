// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
//! PyO3 bindings for the SCPN neural / spiking-controller lane.
//!
//! Exposes the SCPN runtime kernels (dense activations, marking update,
//! stochastic firing estimator) and the spiking-neural controller pools
//! (`PySnnPool`, `PySnnController`) to Python, mirroring the NumPy tier's APIs.

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use fusion_control::snn::{NeuroCyberneticController, SpikingControllerPool};

/// Dense activation kernel for SCPN controller path.
#[pyfunction]
pub(crate) fn scpn_dense_activations<'py>(
    py: Python<'py>,
    arg1: &Bound<'py, PyAny>,
    arg2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // Canonical signature: (w_in, marking). Legacy parity tests use (marking, w_in).
    let (w, m) = if let (Ok(w), Ok(m)) = (
        arg1.extract::<PyReadonlyArray2<'py, f64>>(),
        arg2.extract::<PyReadonlyArray1<'py, f64>>(),
    ) {
        (w, m)
    } else if let (Ok(m), Ok(w)) = (
        arg1.extract::<PyReadonlyArray1<'py, f64>>(),
        arg2.extract::<PyReadonlyArray2<'py, f64>>(),
    ) {
        (w, m)
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "scpn_dense_activations expects (matrix, vector) or (vector, matrix)",
        ));
    };

    let w = w.as_array();
    let m = m.as_array();
    if w.ncols() != m.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "shape mismatch: w_in has {} columns, marking has length {}",
            w.ncols(),
            m.len()
        )));
    }
    Ok(w.dot(&m).into_pyarray(py))
}

/// Dense marking update kernel: clip(m - W_in^T f + W_out f, 0, 1).
#[pyfunction]
pub(crate) fn scpn_marking_update<'py>(
    py: Python<'py>,
    marking: PyReadonlyArray1<'py, f64>,
    w_in: &Bound<'py, PyAny>,
    w_out: &Bound<'py, PyAny>,
    firing: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let m = marking.as_array();
    let f = firing.as_array();

    // Canonical path used by controller: matrices W_in, W_out.
    if let (Ok(wi_2d), Ok(wo_2d)) = (
        w_in.extract::<PyReadonlyArray2<'py, f64>>(),
        w_out.extract::<PyReadonlyArray2<'py, f64>>(),
    ) {
        let wi = wi_2d.as_array();
        let wo = wo_2d.as_array();

        if wi.nrows() != f.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "shape mismatch: w_in rows {} != firing length {}",
                wi.nrows(),
                f.len()
            )));
        }
        if wo.ncols() != f.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "shape mismatch: w_out cols {} != firing length {}",
                wo.ncols(),
                f.len()
            )));
        }
        if wi.ncols() != m.len() || wo.nrows() != m.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "shape mismatch: marking length {} expects w_in cols {} and w_out rows {}",
                m.len(),
                wi.ncols(),
                wo.nrows()
            )));
        }

        let cons = wi.t().dot(&f);
        let prod = wo.dot(&f);
        let out = (&m - &cons + &prod).mapv(|x| x.clamp(0.0, 1.0));
        return Ok(out.into_pyarray(py));
    }

    // Legacy path used in parity tests: vectors pre, post.
    if let (Ok(pre_1d), Ok(post_1d)) = (
        w_in.extract::<PyReadonlyArray1<'py, f64>>(),
        w_out.extract::<PyReadonlyArray1<'py, f64>>(),
    ) {
        let pre = pre_1d.as_array();
        let post = post_1d.as_array();
        if pre.len() != m.len() || post.len() != m.len() || f.len() != m.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "vector mode length mismatch: marking={}, pre={}, post={}, firing={}",
                m.len(),
                pre.len(),
                post.len(),
                f.len()
            )));
        }
        let out = &m - &(&pre * &f) + &(&post * &f);
        return Ok(out.into_pyarray(py));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "scpn_marking_update expects matrix or vector pre/post arguments",
    ))
}

/// Stochastic firing estimator for SCPN transitions.
///
/// Returns mean Bernoulli activation over `n_passes` samples with optional
/// antithetic pairing.
#[pyfunction]
pub(crate) fn scpn_sample_firing<'py>(
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

/// Python-accessible spiking controller pool (LIF neuron population).
#[pyclass]
pub(crate) struct PySnnPool {
    inner: SpikingControllerPool,
}

#[pymethods]
impl PySnnPool {
    #[new]
    #[pyo3(signature = (n_neurons=50, gain=10.0, window_size=20))]
    fn new(n_neurons: usize, gain: f64, window_size: usize) -> PyResult<Self> {
        let inner = SpikingControllerPool::new(n_neurons, gain, window_size)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PySnnPool { inner })
    }

    /// Process error signal through SNN population. Returns control output.
    fn step(&mut self, error: f64) -> PyResult<f64> {
        self.inner
            .step(error)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn n_neurons(&self) -> usize {
        self.inner.n_neurons
    }

    #[getter]
    fn gain(&self) -> f64 {
        self.inner.gain
    }

    fn __repr__(&self) -> String {
        format!(
            "PySnnPool(n_neurons={}, gain={:.1})",
            self.inner.n_neurons, self.inner.gain
        )
    }
}

/// Python-accessible neuro-cybernetic controller (dual R+Z SNN pools).
#[pyclass]
pub(crate) struct PySnnController {
    inner: NeuroCyberneticController,
}

#[pymethods]
impl PySnnController {
    #[new]
    fn new(target_r: f64, target_z: f64) -> PyResult<Self> {
        let inner = NeuroCyberneticController::new(target_r, target_z)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PySnnController { inner })
    }

    /// Process measured (R, Z) position. Returns (ctrl_R, ctrl_Z).
    fn step(&mut self, measured_r: f64, measured_z: f64) -> PyResult<(f64, f64)> {
        self.inner
            .step(measured_r, measured_z)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn target_r(&self) -> f64 {
        self.inner.target_r
    }

    #[getter]
    fn target_z(&self) -> f64 {
        self.inner.target_z
    }

    fn __repr__(&self) -> String {
        format!(
            "PySnnController(target_r={:.3}, target_z={:.3})",
            self.inner.target_r, self.inner.target_z
        )
    }
}
