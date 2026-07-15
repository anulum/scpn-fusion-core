// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
//! PyO3 bindings for the phase-dynamics lane (`fusion-phase`).
//!
//! Exposes the mean-field Kuramoto-Sakaguchi step and its batched run, plus the
//! multi-layer UPDE tick and its batched run, to Python for fastest-first
//! dispatch. Each binding mirrors the NumPy reference tier's keyword API.

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
#[pyo3(signature = (theta, omega, dt, k, alpha=0.0, zeta=0.0, psi=0.0, wrap=true))]
#[expect(
    clippy::too_many_arguments,
    reason = "PyO3 binding preserves the stable Python keyword API of the NumPy tier."
)]
pub(crate) fn py_kuramoto_step<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'py, f64>,
    omega: PyReadonlyArray1<'py, f64>,
    dt: f64,
    k: f64,
    alpha: f64,
    zeta: f64,
    psi: f64,
    wrap: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let theta_slice = theta.as_slice()?;
    let omega_slice = omega.as_slice()?;
    let out = fusion_phase::kuramoto_step(theta_slice, omega_slice, dt, k, alpha, zeta, psi, wrap)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let result = PyDict::new(py);
    result.set_item("theta1", Array1::from_vec(out.theta1).into_pyarray(py))?;
    result.set_item("dtheta", Array1::from_vec(out.dtheta).into_pyarray(py))?;
    result.set_item("R", out.r)?;
    result.set_item("Psi_r", out.psi_r)?;
    result.set_item("Psi", psi)?;
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (theta, omega, n_steps, dt, k, alpha=0.0, zeta=0.0, psi=0.0, wrap=true))]
#[expect(
    clippy::too_many_arguments,
    reason = "PyO3 binding preserves the stable Python keyword API of the NumPy tier."
)]
pub(crate) fn py_kuramoto_run<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'py, f64>,
    omega: PyReadonlyArray1<'py, f64>,
    n_steps: usize,
    dt: f64,
    k: f64,
    alpha: f64,
    zeta: f64,
    psi: f64,
    wrap: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let theta_slice = theta.as_slice()?;
    let omega_slice = omega.as_slice()?;
    let out = fusion_phase::kuramoto_run(
        theta_slice,
        omega_slice,
        n_steps,
        dt,
        k,
        alpha,
        zeta,
        psi,
        wrap,
    )
    .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let result = PyDict::new(py);
    result.set_item(
        "theta_final",
        Array1::from_vec(out.theta_final).into_pyarray(py),
    )?;
    result.set_item("R_hist", Array1::from_vec(out.r_hist).into_pyarray(py))?;
    result.set_item(
        "Psi_r_hist",
        Array1::from_vec(out.psi_r_hist).into_pyarray(py),
    )?;
    result.set_item("Psi", psi)?;
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (theta, omega, offsets, k, alpha, zeta, dt, psi_global, actuation_gain=1.0, pac_gamma=0.0, wrap=true))]
#[expect(
    clippy::too_many_arguments,
    reason = "PyO3 binding preserves the stable Python keyword API of the NumPy tier."
)]
pub(crate) fn py_upde_tick<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'py, f64>,
    omega: PyReadonlyArray1<'py, f64>,
    offsets: Vec<usize>,
    k: PyReadonlyArray2<'py, f64>,
    alpha: PyReadonlyArray2<'py, f64>,
    zeta: PyReadonlyArray1<'py, f64>,
    dt: f64,
    psi_global: f64,
    actuation_gain: f64,
    pac_gamma: f64,
    wrap: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let theta_slice = theta.as_slice()?;
    let omega_slice = omega.as_slice()?;
    let k_owned = k.as_array().to_owned();
    let alpha_owned = alpha.as_array().to_owned();
    let k_flat: Vec<f64> = k_owned.iter().copied().collect();
    let alpha_flat: Vec<f64> = alpha_owned.iter().copied().collect();
    let zeta_slice = zeta.as_slice()?;
    let out = fusion_phase::upde_tick(
        theta_slice,
        omega_slice,
        &offsets,
        &k_flat,
        &alpha_flat,
        zeta_slice,
        dt,
        psi_global,
        actuation_gain,
        pac_gamma,
        wrap,
    )
    .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let result = PyDict::new(py);
    result.set_item("theta1", Array1::from_vec(out.theta1).into_pyarray(py))?;
    result.set_item("dtheta", Array1::from_vec(out.dtheta).into_pyarray(py))?;
    result.set_item("R_layer", Array1::from_vec(out.r_layer).into_pyarray(py))?;
    result.set_item(
        "Psi_layer",
        Array1::from_vec(out.psi_layer).into_pyarray(py),
    )?;
    result.set_item("R_global", out.r_global)?;
    result.set_item("Psi_r_global", out.psi_r_global)?;
    result.set_item("V_layer", Array1::from_vec(out.v_layer).into_pyarray(py))?;
    result.set_item("V_global", out.v_global)?;
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (theta, omega, offsets, k, alpha, zeta, n_steps, dt, psi_global, actuation_gain=1.0, pac_gamma=0.0, wrap=true))]
#[expect(
    clippy::too_many_arguments,
    reason = "PyO3 binding preserves the stable Python keyword API of the NumPy tier."
)]
pub(crate) fn py_upde_run<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'py, f64>,
    omega: PyReadonlyArray1<'py, f64>,
    offsets: Vec<usize>,
    k: PyReadonlyArray2<'py, f64>,
    alpha: PyReadonlyArray2<'py, f64>,
    zeta: PyReadonlyArray1<'py, f64>,
    n_steps: usize,
    dt: f64,
    psi_global: f64,
    actuation_gain: f64,
    pac_gamma: f64,
    wrap: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let theta_slice = theta.as_slice()?;
    let omega_slice = omega.as_slice()?;
    let k_owned = k.as_array().to_owned();
    let alpha_owned = alpha.as_array().to_owned();
    let k_flat: Vec<f64> = k_owned.iter().copied().collect();
    let alpha_flat: Vec<f64> = alpha_owned.iter().copied().collect();
    let zeta_slice = zeta.as_slice()?;
    let l = offsets.len().saturating_sub(1);
    let out = fusion_phase::upde_run(
        theta_slice,
        omega_slice,
        &offsets,
        &k_flat,
        &alpha_flat,
        zeta_slice,
        n_steps,
        dt,
        psi_global,
        actuation_gain,
        pac_gamma,
        wrap,
    )
    .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let result = PyDict::new(py);
    result.set_item(
        "theta_final",
        Array1::from_vec(out.theta_final).into_pyarray(py),
    )?;
    result.set_item(
        "R_layer_hist",
        Array2::from_shape_vec((n_steps, l), out.r_layer_hist)
            .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?
            .into_pyarray(py),
    )?;
    result.set_item(
        "R_global_hist",
        Array1::from_vec(out.r_global_hist).into_pyarray(py),
    )?;
    result.set_item(
        "V_layer_hist",
        Array2::from_shape_vec((n_steps, l), out.v_layer_hist)
            .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?
            .into_pyarray(py),
    )?;
    result.set_item(
        "V_global_hist",
        Array1::from_vec(out.v_global_hist).into_pyarray(py),
    )?;
    Ok(result)
}
