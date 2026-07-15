// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
//! PyO3 Python bindings for SCPN Fusion Core.
//!
//! Stage 10: Exposes Grad-Shafranov solver, thermodynamics, control,
//! diagnostics, and ML modules to Python via PyO3 + numpy.

use pyo3::prelude::*;

mod bindings;
use bindings::control::{PyFnoController, PyMpcController};
use bindings::diagnostics::PyTomography;
use bindings::equilibrium::{
    measure_magnetics, multigrid_vcycle, shafranov_bv, solve_coil_currents, PyEquilibriumResult,
    PyFusionKernel, PyInverseResult, PyInverseSolver, PyThermodynamicsResult,
};
use bindings::flight::{PyFlightState, PyRustFlightSim, PySimulationReport, PyStepMetrics};
use bindings::frc::{
    py_rotating_frc_bvp_acceptance_status, py_solve_frc_equilibrium,
    py_solve_rotating_frc_equilibrium,
};
#[cfg(feature = "gpu")]
use bindings::gpu as gpu_bindings;
use bindings::gyrokinetics::PyNonlinearGKSolver;
use bindings::mhd::{rutherford_island_growth, simulate_tearing_mode, PyHallMHD, PyReducedMHD};
use bindings::ml::PyNeuralTransport;
use bindings::neural::{
    scpn_dense_activations, scpn_marking_update, scpn_sample_firing, PySnnController, PySnnPool,
};
use bindings::nuclear::PyBreedingBlanket;
use bindings::particles::{
    py_advance_boris, py_get_heating_profile, py_particle_population_summary,
    py_seed_alpha_particles, PyParticle, PyPopulationSummary,
};
use bindings::phase::{py_kuramoto_run, py_kuramoto_step, py_upde_run, py_upde_tick};
use bindings::plant::PyPlantModel;
use bindings::rmf::{PyPacingMode, PyRmfAotCertificate, PyRmfConfig, PyRmfController};
use bindings::transport::{
    py_evaluate_design, py_run_design_scan, PyDriftWave, PyFokkerPlanckSolver, PyPlasma2D,
    PySpiAblationSolver, PyTransportSolver,
};

// ─── Module registration ───

#[pymodule]
fn scpn_fusion_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFusionKernel>()?;
    m.add_class::<PyEquilibriumResult>()?;
    m.add_class::<PyThermodynamicsResult>()?;
    m.add_class::<PyNeuralTransport>()?;
    m.add_class::<PyInverseSolver>()?;
    m.add_class::<PyInverseResult>()?;
    m.add_class::<PyPlantModel>()?;
    m.add_class::<PyRustFlightSim>()?;
    m.add_class::<PySimulationReport>()?;
    m.add_class::<PyStepMetrics>()?;
    m.add_class::<PyFlightState>()?;
    m.add_function(wrap_pyfunction!(shafranov_bv, m)?)?;
    m.add_function(wrap_pyfunction!(solve_coil_currents, m)?)?;
    m.add_function(wrap_pyfunction!(measure_magnetics, m)?)?;
    m.add_function(wrap_pyfunction!(multigrid_vcycle, m)?)?;
    m.add_function(wrap_pyfunction!(scpn_dense_activations, m)?)?;
    m.add_function(wrap_pyfunction!(scpn_marking_update, m)?)?;
    m.add_function(wrap_pyfunction!(scpn_sample_firing, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_tearing_mode, m)?)?;
    m.add_function(wrap_pyfunction!(rutherford_island_growth, m)?)?;
    // Particle / Boris integrator bridge
    m.add_class::<PyParticle>()?;
    m.add_class::<PyPopulationSummary>()?;
    m.add_function(wrap_pyfunction!(py_seed_alpha_particles, m)?)?;
    m.add_function(wrap_pyfunction!(py_advance_boris, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_heating_profile, m)?)?;
    m.add_function(wrap_pyfunction!(py_particle_population_summary, m)?)?;
    // SNN controller bridge
    m.add_class::<PySnnPool>()?;
    m.add_class::<PySnnController>()?;
    // Extended PyO3 bridges
    m.add_class::<PyHallMHD>()?;
    m.add_class::<PyFnoController>()?;
    m.add_class::<PyMpcController>()?;
    m.add_class::<PyTomography>()?;
    m.add_class::<PyBreedingBlanket>()?;
    m.add_class::<PyPlasma2D>()?;
    m.add_function(wrap_pyfunction!(py_evaluate_design, m)?)?;
    m.add_function(wrap_pyfunction!(py_run_design_scan, m)?)?;
    m.add_class::<PyDriftWave>()?;
    m.add_class::<PyTransportSolver>()?;
    m.add_class::<PyFokkerPlanckSolver>()?;
    m.add_class::<PySpiAblationSolver>()?;
    m.add_class::<PyReducedMHD>()?;
    #[cfg(feature = "gpu")]
    {
        m.add_class::<gpu_bindings::PyGpuSolver>()?;
        m.add_function(wrap_pyfunction!(gpu_bindings::py_gpu_available, m)?)?;
        m.add_function(wrap_pyfunction!(gpu_bindings::py_gpu_info, m)?)?;
    }
    m.add_class::<PyNonlinearGKSolver>()?;
    m.add_function(wrap_pyfunction!(py_kuramoto_step, m)?)?;
    m.add_function(wrap_pyfunction!(py_kuramoto_run, m)?)?;
    m.add_function(wrap_pyfunction!(py_upde_tick, m)?)?;
    m.add_function(wrap_pyfunction!(py_upde_run, m)?)?;
    m.add_function(wrap_pyfunction!(py_solve_frc_equilibrium, m)?)?;
    m.add_function(wrap_pyfunction!(py_rotating_frc_bvp_acceptance_status, m)?)?;
    m.add_function(wrap_pyfunction!(py_solve_rotating_frc_equilibrium, m)?)?;
    m.add_class::<PyRmfConfig>()?;
    m.add_class::<PyRmfController>()?;
    m.add_class::<PyPacingMode>()?;
    m.add_class::<PyRmfAotCertificate>()?;
    Ok(())
}
