// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Full Runaway Kinetic PyO3 Bridge
//! PyO3 bridge for full radius-pitch-momentum runaway kinetics.

use fusion_physics::runaway_kinetic::{
    RunawayKineticCoefficients, RunawayKineticGeometry, RunawayKineticGrid, RunawayKineticOperator,
    RunawayKineticSolver,
};
use ndarray::{Array1, ArrayD, IxDyn};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn value_error(message: String) -> PyErr {
    PyValueError::new_err(message)
}

fn dict_array(mapping: &Bound<'_, PyDict>, name: &str) -> PyResult<Vec<f64>> {
    let item = mapping
        .get_item(name)?
        .ok_or_else(|| PyValueError::new_err(format!("missing required array '{name}'")))?;
    let array = item.extract::<PyReadonlyArrayDyn<'_, f64>>()?;
    Ok(array.as_array().iter().copied().collect())
}

fn shaped<'py>(
    py: Python<'py>,
    values: Vec<f64>,
    shape: &[usize],
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f64>>> {
    let array = ArrayD::from_shape_vec(IxDyn(shape), values)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(py))
}

/// Solve the full Rust kinetic system from explicit NumPy arrays.
#[pyfunction]
#[pyo3(
    signature = (
        radius_faces_m,
        pitch_faces,
        momentum_faces_mc,
        coefficients,
        geometry,
        initial_distribution,
        times_s,
        initial_runaway_density_m3=None,
        maximum_step_s=1.0e-7,
        negativity_tolerance=1.0e-12
    ),
    name = "runaway_kinetic_solve_rust"
)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn py_runaway_kinetic_solve<'py>(
    py: Python<'py>,
    radius_faces_m: PyReadonlyArray1<'py, f64>,
    pitch_faces: PyReadonlyArray1<'py, f64>,
    momentum_faces_mc: PyReadonlyArray1<'py, f64>,
    coefficients: &Bound<'py, PyDict>,
    geometry: &Bound<'py, PyDict>,
    initial_distribution: PyReadonlyArrayDyn<'py, f64>,
    times_s: PyReadonlyArray1<'py, f64>,
    initial_runaway_density_m3: Option<PyReadonlyArray1<'py, f64>>,
    maximum_step_s: f64,
    negativity_tolerance: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let grid = RunawayKineticGrid::new(
        radius_faces_m.as_slice()?.to_vec(),
        pitch_faces.as_slice()?.to_vec(),
        momentum_faces_mc.as_slice()?.to_vec(),
    )
    .map_err(value_error)?;
    let coefficient_bundle = RunawayKineticCoefficients::new(
        &grid,
        dict_array(coefficients, "radial_advection")?,
        dict_array(coefficients, "momentum_electric_advection")?,
        dict_array(coefficients, "momentum_collision_advection")?,
        dict_array(coefficients, "momentum_synchrotron_advection")?,
        dict_array(coefficients, "momentum_bremsstrahlung_advection")?,
        dict_array(coefficients, "pitch_electric_advection")?,
        dict_array(coefficients, "pitch_synchrotron_advection")?,
        dict_array(coefficients, "radial_diffusion")?,
        dict_array(coefficients, "momentum_diffusion")?,
        dict_array(coefficients, "pitch_diffusion")?,
        dict_array(coefficients, "momentum_pitch_diffusion")?,
        dict_array(coefficients, "pitch_momentum_diffusion")?,
        dict_array(coefficients, "avalanche_source_kernel")?,
        dict_array(coefficients, "total_electron_density_m3")?,
        dict_array(coefficients, "total_density_avalanche_rate_s_inv")?,
        dict_array(coefficients, "total_density_external_source_m3_s")?,
        dict_array(coefficients, "external_source")?,
    )
    .map_err(value_error)?;
    let geometry_bundle = RunawayKineticGeometry::checked(
        &grid,
        dict_array(geometry, "cell_measure")?,
        dict_array(geometry, "density_cell_measure")?,
        dict_array(geometry, "radial_face_measure")?,
        dict_array(geometry, "momentum_face_measure")?,
        dict_array(geometry, "pitch_face_measure")?,
    )
    .map_err(value_error)?;
    let state: Vec<f64> = initial_distribution.as_array().iter().copied().collect();
    let times = times_s.as_slice()?.to_vec();
    let density = initial_runaway_density_m3
        .map(|values| values.as_slice().map(<[f64]>::to_vec))
        .transpose()?;
    let nr = grid.nr();
    let nxi = grid.nxi();
    let np = grid.np();
    let solver = RunawayKineticSolver::new(
        RunawayKineticOperator::with_geometry(grid, coefficient_bundle, geometry_bundle),
        maximum_step_s,
        negativity_tolerance,
    )
    .map_err(value_error)?;
    let trajectory = py
        .detach(move || solver.solve(&state, &times, density.as_deref()))
        .map_err(value_error)?;
    let nt = trajectory.times_s.len();
    let result = PyDict::new(py);
    result.set_item(
        "times_s",
        Array1::from_vec(trajectory.times_s).into_pyarray(py),
    )?;
    result.set_item(
        "distribution",
        shaped(py, trajectory.distribution, &[nt, nr, nxi, np])?,
    )?;
    result.set_item(
        "radial_transport",
        shaped(py, trajectory.radial_transport, &[nt, nr, nxi, np])?,
    )?;
    result.set_item(
        "electric_acceleration",
        shaped(py, trajectory.electric_acceleration, &[nt, nr, nxi, np])?,
    )?;
    result.set_item(
        "collisional_drag_diffusion",
        shaped(
            py,
            trajectory.collisional_drag_diffusion,
            &[nt, nr, nxi, np],
        )?,
    )?;
    result.set_item(
        "pitch_scattering",
        shaped(py, trajectory.pitch_scattering, &[nt, nr, nxi, np])?,
    )?;
    result.set_item(
        "cross_diffusion",
        shaped(py, trajectory.cross_diffusion, &[nt, nr, nxi, np])?,
    )?;
    result.set_item(
        "synchrotron_loss",
        shaped(py, trajectory.synchrotron_loss, &[nt, nr, nxi, np])?,
    )?;
    result.set_item(
        "bremsstrahlung_loss",
        shaped(py, trajectory.bremsstrahlung_loss, &[nt, nr, nxi, np])?,
    )?;
    result.set_item(
        "avalanche_generation",
        shaped(py, trajectory.avalanche_generation, &[nt, nr, nxi, np])?,
    )?;
    result.set_item(
        "external_source",
        shaped(py, trajectory.external_source, &[nt, nr, nxi, np])?,
    )?;
    result.set_item(
        "total_tendency",
        shaped(py, trajectory.total_tendency, &[nt, nr, nxi, np])?,
    )?;
    result.set_item(
        "runaway_density_m3",
        shaped(py, trajectory.runaway_density_m3, &[nt, nr])?,
    )?;
    result.set_item(
        "runaway_density_radial_transport_m3_s",
        shaped(
            py,
            trajectory.runaway_density_radial_transport_m3_s,
            &[nt, nr],
        )?,
    )?;
    result.set_item(
        "runaway_density_avalanche_generation_m3_s",
        shaped(
            py,
            trajectory.runaway_density_avalanche_generation_m3_s,
            &[nt, nr],
        )?,
    )?;
    result.set_item(
        "runaway_density_external_source_m3_s",
        shaped(
            py,
            trajectory.runaway_density_external_source_m3_s,
            &[nt, nr],
        )?,
    )?;
    result.set_item(
        "runaway_density_tendency_m3_s",
        shaped(py, trajectory.runaway_density_tendency_m3_s, &[nt, nr])?,
    )?;
    result.set_item(
        "density_m3",
        shaped(py, trajectory.moments.density_m3, &[nt, nr])?,
    )?;
    result.set_item(
        "current_density_a_m2",
        shaped(py, trajectory.moments.current_density_a_m2, &[nt, nr])?,
    )?;
    result.set_item(
        "kinetic_energy_density_j_m3",
        shaped(
            py,
            trajectory.moments.kinetic_energy_density_j_m3,
            &[nt, nr],
        )?,
    )?;
    result.set_item("internal_steps", trajectory.internal_steps)?;
    result.set_item("minimum_distribution", trajectory.minimum_distribution)?;
    Ok(result)
}
