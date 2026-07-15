// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
//! PyO3 bindings for the high-speed flight-simulator lane (`fusion-control`).
//!
//! Exposes the real-time flight simulator (`PyRustFlightSim`) and its result
//! records (`PySimulationReport`, `PyStepMetrics`, `PyFlightState`) to Python,
//! mirroring the NumPy tier's shot-runner API.

use pyo3::prelude::*;

use fusion_control::flight_sim::RustFlightSim;

#[pyclass]
pub(crate) struct PySimulationReport {
    #[pyo3(get)]
    pub steps: usize,
    #[pyo3(get)]
    pub duration_s: f64,
    #[pyo3(get)]
    pub wall_time_ms: f64,
    #[pyo3(get)]
    pub max_step_time_us: f64,
    #[pyo3(get)]
    pub mean_abs_r_error: f64,
    #[pyo3(get)]
    pub mean_abs_z_error: f64,
    #[pyo3(get)]
    pub final_beta: f64,
    #[pyo3(get)]
    pub final_heating_mw: f64,
    #[pyo3(get)]
    pub max_beta: f64,
    #[pyo3(get)]
    pub max_heating_mw: f64,
    #[pyo3(get)]
    pub vessel_contact_events: usize,
    #[pyo3(get)]
    pub pf_constraint_events: usize,
    #[pyo3(get)]
    pub heating_constraint_events: usize,
    #[pyo3(get)]
    pub retained_steps: usize,
    #[pyo3(get)]
    pub history_truncated: bool,
    #[pyo3(get)]
    pub disrupted: bool,
    #[pyo3(get)]
    pub r_history: Vec<f64>,
    #[pyo3(get)]
    pub z_history: Vec<f64>,
    #[pyo3(get)]
    pub ip_history: Vec<f64>,
}

#[pyclass]
pub(crate) struct PyStepMetrics {
    #[pyo3(get)]
    pub r_error: f64,
    #[pyo3(get)]
    pub z_error: f64,
    #[pyo3(get)]
    pub disrupted: bool,
    #[pyo3(get)]
    pub step_time_us: f64,
    #[pyo3(get)]
    pub beta: f64,
    #[pyo3(get)]
    pub heating_mw: f64,
    #[pyo3(get)]
    pub vessel_contact: bool,
    #[pyo3(get)]
    pub pf_constraint_active: bool,
    #[pyo3(get)]
    pub heating_constraint_active: bool,
}

#[pyclass]
pub(crate) struct PyFlightState {
    #[pyo3(get)]
    pub r: f64,
    #[pyo3(get)]
    pub z: f64,
    #[pyo3(get)]
    pub ip_ma: f64,
    #[pyo3(get)]
    pub beta: f64,
    #[pyo3(get)]
    pub heating_mw: f64,
}

fn to_py_simulation_report(
    report: fusion_control::flight_sim::SimulationReport,
) -> PySimulationReport {
    PySimulationReport {
        steps: report.steps,
        duration_s: report.duration_s,
        wall_time_ms: report.wall_time_ms,
        max_step_time_us: report.max_step_time_us,
        mean_abs_r_error: report.mean_abs_r_error,
        mean_abs_z_error: report.mean_abs_z_error,
        final_beta: report.final_beta,
        final_heating_mw: report.final_heating_mw,
        max_beta: report.max_beta,
        max_heating_mw: report.max_heating_mw,
        vessel_contact_events: report.vessel_contact_events,
        pf_constraint_events: report.pf_constraint_events,
        heating_constraint_events: report.heating_constraint_events,
        retained_steps: report.retained_steps,
        history_truncated: report.history_truncated,
        disrupted: report.disrupted,
        r_history: report.r_history,
        z_history: report.z_history,
        ip_history: report.ip_history,
    }
}

fn to_py_step_metrics(step: fusion_control::flight_sim::StepMetrics) -> PyStepMetrics {
    PyStepMetrics {
        r_error: step.r_error,
        z_error: step.z_error,
        disrupted: step.disrupted,
        step_time_us: step.step_time_us,
        beta: step.beta,
        heating_mw: step.heating_mw,
        vessel_contact: step.vessel_contact,
        pf_constraint_active: step.pf_constraint_active,
        heating_constraint_active: step.heating_constraint_active,
    }
}

fn to_py_flight_state(state: (f64, f64, f64, f64, f64)) -> PyFlightState {
    PyFlightState {
        r: state.0,
        z: state.1,
        ip_ma: state.2,
        beta: state.3,
        heating_mw: state.4,
    }
}

#[pyclass]
pub(crate) struct PyRustFlightSim {
    inner: RustFlightSim,
}

#[pymethods]
impl PyRustFlightSim {
    #[new]
    #[pyo3(signature = (target_r=6.2, target_z=0.0, control_hz=10000.0))]
    fn new(target_r: f64, target_z: f64, control_hz: f64) -> PyResult<Self> {
        let inner = RustFlightSim::new(target_r, target_z, control_hz)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (shot_duration_s, deterministic=false))]
    fn run_shot(
        &mut self,
        shot_duration_s: f64,
        deterministic: bool,
    ) -> PyResult<PySimulationReport> {
        let report = self
            .inner
            .run_shot(shot_duration_s, deterministic)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(to_py_simulation_report(report))
    }

    fn prepare_shot(&mut self, shot_duration_s: f64) -> PyResult<usize> {
        self.inner
            .prepare_shot(shot_duration_s)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn reset_for_shot(&mut self) {
        self.inner.reset_for_shot();
    }

    fn reset_plasma_state(&mut self) {
        self.inner.reset_plasma_state();
    }

    fn plasma_state(&self) -> PyFlightState {
        to_py_flight_state(self.inner.plasma_state())
    }

    fn step_once(&mut self, step_index: usize, shot_duration_s: f64) -> PyResult<PyStepMetrics> {
        let step = self
            .inner
            .step_once(step_index, shot_duration_s)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(to_py_step_metrics(step))
    }
}
