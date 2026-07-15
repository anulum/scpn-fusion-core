// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
//! PyO3 bindings for the particle / Boris-integrator lane (`fusion-core`).
//!
//! Exposes charged test particles, alpha-particle seeding, the Boris pusher,
//! alpha heating-profile deposition on an R-Z grid, and population statistics to
//! Python, mirroring the NumPy tier's keyword APIs.

use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

use fusion_core::particles::{
    advance_particles_boris, estimate_alpha_heating_profile, seed_alpha_test_particles,
    summarize_particle_population, ChargedParticle,
};
use fusion_types::state::Grid2D;

/// Python-accessible charged particle.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub(crate) struct PyParticle {
    #[pyo3(get, set)]
    x_m: f64,
    #[pyo3(get, set)]
    y_m: f64,
    #[pyo3(get, set)]
    z_m: f64,
    #[pyo3(get, set)]
    vx_m_s: f64,
    #[pyo3(get, set)]
    vy_m_s: f64,
    #[pyo3(get, set)]
    vz_m_s: f64,
    #[pyo3(get, set)]
    charge_c: f64,
    #[pyo3(get, set)]
    mass_kg: f64,
    #[pyo3(get, set)]
    weight: f64,
}

#[pymethods]
impl PyParticle {
    fn kinetic_energy_mev(&self) -> f64 {
        self.to_internal().kinetic_energy_mev()
    }

    fn cylindrical_radius_m(&self) -> f64 {
        self.to_internal().cylindrical_radius_m()
    }

    fn __repr__(&self) -> String {
        format!(
            "Particle(E={:.2} MeV, R={:.3} m)",
            self.kinetic_energy_mev(),
            self.cylindrical_radius_m()
        )
    }
}

impl PyParticle {
    fn to_internal(&self) -> ChargedParticle {
        ChargedParticle {
            x_m: self.x_m,
            y_m: self.y_m,
            z_m: self.z_m,
            vx_m_s: self.vx_m_s,
            vy_m_s: self.vy_m_s,
            vz_m_s: self.vz_m_s,
            charge_c: self.charge_c,
            mass_kg: self.mass_kg,
            weight: self.weight,
        }
    }

    fn from_internal(p: &ChargedParticle) -> Self {
        PyParticle {
            x_m: p.x_m,
            y_m: p.y_m,
            z_m: p.z_m,
            vx_m_s: p.vx_m_s,
            vy_m_s: p.vy_m_s,
            vz_m_s: p.vz_m_s,
            charge_c: p.charge_c,
            mass_kg: p.mass_kg,
            weight: p.weight,
        }
    }
}

/// Population summary result.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub(crate) struct PyPopulationSummary {
    #[pyo3(get)]
    count: usize,
    #[pyo3(get)]
    mean_energy_mev: f64,
    #[pyo3(get)]
    p95_energy_mev: f64,
    #[pyo3(get)]
    max_energy_mev: f64,
    #[pyo3(get)]
    runaway_fraction: f64,
}

#[pymethods]
impl PyPopulationSummary {
    fn __repr__(&self) -> String {
        format!(
            "PopulationSummary(n={}, E_mean={:.2} MeV)",
            self.count, self.mean_energy_mev
        )
    }
}

/// Seed alpha test particles.
#[pyfunction]
pub(crate) fn py_seed_alpha_particles(
    n: usize,
    r0: f64,
    z0: f64,
    energy_mev: f64,
    pitch: f64,
    weight: f64,
) -> PyResult<Vec<PyParticle>> {
    let particles = seed_alpha_test_particles(n, r0, z0, energy_mev, pitch, weight)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(particles.iter().map(PyParticle::from_internal).collect())
}

/// Advance particles using Boris integrator.
#[pyfunction]
pub(crate) fn py_advance_boris(
    particles: Vec<PyParticle>,
    e_field: [f64; 3],
    b_field: [f64; 3],
    dt: f64,
    steps: usize,
) -> PyResult<Vec<PyParticle>> {
    let mut internal: Vec<ChargedParticle> = particles.iter().map(|p| p.to_internal()).collect();
    advance_particles_boris(&mut internal, e_field, b_field, dt, steps)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(internal.iter().map(PyParticle::from_internal).collect())
}

/// Get alpha heating power density profile on R-Z grid.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub(crate) fn py_get_heating_profile<'py>(
    py: Python<'py>,
    particles: Vec<PyParticle>,
    nr: usize,
    nz: usize,
    r_min: f64,
    r_max: f64,
    z_min: f64,
    z_max: f64,
    confinement_tau_s: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let internal: Vec<ChargedParticle> = particles.iter().map(|p| p.to_internal()).collect();
    let grid = Grid2D::new(nr, nz, r_min, r_max, z_min, z_max);
    let profile = estimate_alpha_heating_profile(&internal, &grid, confinement_tau_s)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(profile.into_pyarray(py))
}

/// Summarize particle population statistics.
#[pyfunction]
pub(crate) fn py_particle_population_summary(
    particles: Vec<PyParticle>,
    threshold_mev: f64,
) -> PyResult<PyPopulationSummary> {
    let internal: Vec<ChargedParticle> = particles.iter().map(|p| p.to_internal()).collect();
    let summary = summarize_particle_population(&internal, threshold_mev)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPopulationSummary {
        count: summary.count,
        mean_energy_mev: summary.mean_energy_mev,
        p95_energy_mev: summary.p95_energy_mev,
        max_energy_mev: summary.max_energy_mev,
        runaway_fraction: summary.runaway_fraction,
    })
}
