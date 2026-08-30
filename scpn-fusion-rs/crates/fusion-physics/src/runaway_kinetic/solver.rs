// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Full Runaway Kinetic Solver
//! Deterministic SSPRK3 evolution of the complete three-axis operator.

use super::grid::validate_array;
use super::operator::{RunawayKineticOperator, RunawayKineticTendencies};

const ELECTRON_CHARGE_C: f64 = 1.602_176_634e-19;
const SPEED_OF_LIGHT_M_PER_S: f64 = 299_792_458.0;
const ELECTRON_REST_ENERGY_J: f64 = 8.187_105_776_9e-14;

/// Radially resolved kinetic moments for every requested output time.
#[derive(Debug, Clone)]
pub struct RunawayKineticMoments {
    /// Number density with shape `[time, radius]`.
    pub density_m3: Vec<f64>,
    /// Parallel current density with shape `[time, radius]`.
    pub current_density_a_m2: Vec<f64>,
    /// Kinetic energy density with shape `[time, radius]`.
    pub kinetic_energy_density_j_m3: Vec<f64>,
}

/// Full unprojected state, tendency, density, and moment history.
#[derive(Debug, Clone)]
pub struct RunawayKineticTrajectory {
    /// Requested output times.
    pub times_s: Vec<f64>,
    /// Distribution history with shape `[time, radius, pitch, momentum]`.
    pub distribution: Vec<f64>,
    /// Radial transport tendency history.
    pub radial_transport: Vec<f64>,
    /// Electric-acceleration tendency history.
    pub electric_acceleration: Vec<f64>,
    /// Collisional drag-diffusion tendency history.
    pub collisional_drag_diffusion: Vec<f64>,
    /// Pitch-scattering tendency history.
    pub pitch_scattering: Vec<f64>,
    /// Cross-diffusion tendency history.
    pub cross_diffusion: Vec<f64>,
    /// Synchrotron-loss tendency history.
    pub synchrotron_loss: Vec<f64>,
    /// Bremsstrahlung-loss tendency history.
    pub bremsstrahlung_loss: Vec<f64>,
    /// Avalanche-generation tendency history.
    pub avalanche_generation: Vec<f64>,
    /// External kinetic source history.
    pub external_source: Vec<f64>,
    /// Complete kinetic tendency history.
    pub total_tendency: Vec<f64>,
    /// Independently evolved runaway density with shape `[time, radius]`.
    pub runaway_density_m3: Vec<f64>,
    /// Density radial-transport history.
    pub runaway_density_radial_transport_m3_s: Vec<f64>,
    /// Density avalanche-generation history.
    pub runaway_density_avalanche_generation_m3_s: Vec<f64>,
    /// Density external-source history.
    pub runaway_density_external_source_m3_s: Vec<f64>,
    /// Complete density tendency history.
    pub runaway_density_tendency_m3_s: Vec<f64>,
    /// Radially resolved moments.
    pub moments: RunawayKineticMoments,
    /// Number of internal SSPRK3 steps.
    pub internal_steps: usize,
    /// Minimum distribution value over the full trajectory.
    pub minimum_distribution: f64,
}

/// Public deterministic full-kinetic time integrator.
#[derive(Debug, Clone)]
pub struct RunawayKineticSolver {
    /// Complete kinetic operator.
    pub operator: RunawayKineticOperator,
    /// Maximum internal time step in seconds.
    pub maximum_step_s: f64,
    /// Allowed negative undershoot relative to the initial scale.
    pub negativity_tolerance: f64,
}

impl RunawayKineticSolver {
    /// Construct after validating explicit numerical controls.
    pub fn new(
        operator: RunawayKineticOperator,
        maximum_step_s: f64,
        negativity_tolerance: f64,
    ) -> Result<Self, String> {
        if !maximum_step_s.is_finite() || maximum_step_s <= 0.0 {
            return Err("maximum_step_s must be finite and positive".to_string());
        }
        if !negativity_tolerance.is_finite() || negativity_tolerance < 0.0 {
            return Err("negativity_tolerance must be finite and non-negative".to_string());
        }
        Ok(Self {
            operator,
            maximum_step_s,
            negativity_tolerance,
        })
    }

    fn rhs(&self, state: &[f64], density: &[f64]) -> Result<(Vec<f64>, Vec<f64>), String> {
        let tendency = self.operator.evaluate(state, Some(density))?;
        Ok((tendency.total(), tendency.runaway_density_tendency_m3_s))
    }

    fn step(
        &self,
        state: &[f64],
        density: &[f64],
        dt: f64,
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        let (rhs0, density_rhs0) = self.rhs(state, density)?;
        let first: Vec<f64> = state
            .iter()
            .zip(rhs0)
            .map(|(value, rhs)| value + dt * rhs)
            .collect();
        let density_first: Vec<f64> = density
            .iter()
            .zip(density_rhs0)
            .map(|(value, rhs)| value + dt * rhs)
            .collect();
        let (rhs1, density_rhs1) = self.rhs(&first, &density_first)?;
        let second: Vec<f64> = state
            .iter()
            .zip(first.iter().zip(rhs1))
            .map(|(initial, (stage, rhs))| 0.75 * initial + 0.25 * (stage + dt * rhs))
            .collect();
        let density_second: Vec<f64> = density
            .iter()
            .zip(density_first.iter().zip(density_rhs1))
            .map(|(initial, (stage, rhs))| 0.75 * initial + 0.25 * (stage + dt * rhs))
            .collect();
        let (rhs2, density_rhs2) = self.rhs(&second, &density_second)?;
        let result = state
            .iter()
            .zip(second.iter().zip(rhs2))
            .map(|(initial, (stage, rhs))| (1.0 / 3.0) * initial + (2.0 / 3.0) * (stage + dt * rhs))
            .collect();
        let density_result = density
            .iter()
            .zip(density_second.iter().zip(density_rhs2))
            .map(|(initial, (stage, rhs))| (1.0 / 3.0) * initial + (2.0 / 3.0) * (stage + dt * rhs))
            .collect();
        Ok((result, density_result))
    }

    /// Evolve and return the unprojected state at every requested time.
    pub fn solve(
        &self,
        initial_distribution: &[f64],
        times_s: &[f64],
        initial_runaway_density_m3: Option<&[f64]>,
    ) -> Result<RunawayKineticTrajectory, String> {
        if times_s.len() < 2 {
            return Err("times_s must contain at least two entries".to_string());
        }
        if times_s.iter().any(|value| !value.is_finite()) || times_s[0] != 0.0 {
            return Err("times_s must be finite and start exactly at zero".to_string());
        }
        if times_s.windows(2).any(|pair| pair[1] <= pair[0]) {
            return Err("times_s must be strictly increasing".to_string());
        }
        let mut state = self
            .operator
            .grid
            .require_state("initial_distribution", initial_distribution)?;
        let mut density = if let Some(values) = initial_runaway_density_m3 {
            validate_array(
                "initial_runaway_density_m3",
                values,
                self.operator.grid.nr(),
                true,
            )?;
            values.to_vec()
        } else {
            self.operator.integrated_density(&state)
        };
        let scale = state
            .iter()
            .fold(1.0_f64, |current, value| current.max(value.abs()));
        let cells = self.operator.grid.cell_count();
        let nr = self.operator.grid.nr();
        let mut distribution = Vec::with_capacity(times_s.len() * cells);
        let mut density_history = Vec::with_capacity(times_s.len() * nr);
        distribution.extend_from_slice(&state);
        density_history.extend_from_slice(&density);
        let mut internal_steps = 0;

        for interval in times_s.windows(2) {
            let duration = interval[1] - interval[0];
            let count = ((duration / self.maximum_step_s).ceil() as usize).max(1);
            let dt = duration / count as f64;
            for _ in 0..count {
                (state, density) = self.step(&state, &density, dt)?;
                internal_steps += 1;
                if state.iter().any(|value| !value.is_finite()) {
                    return Err("kinetic evolution produced a non-finite state".to_string());
                }
                let minimum = state.iter().copied().fold(f64::INFINITY, f64::min);
                if minimum < -self.negativity_tolerance * scale {
                    return Err(format!(
                        "kinetic evolution violated the declared negativity tolerance: {minimum}"
                    ));
                }
                if density
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
                {
                    return Err("runaway-density evolution produced an invalid state".to_string());
                }
            }
            distribution.extend_from_slice(&state);
            density_history.extend_from_slice(&density);
        }

        let mut tendencies = Vec::with_capacity(times_s.len());
        for time_index in 0..times_s.len() {
            tendencies.push(self.operator.evaluate(
                &distribution[time_index * cells..(time_index + 1) * cells],
                Some(&density_history[time_index * nr..(time_index + 1) * nr]),
            )?);
        }
        let moments = self.moments(&distribution, times_s.len());
        let minimum_distribution = distribution.iter().copied().fold(f64::INFINITY, f64::min);
        Ok(RunawayKineticTrajectory {
            times_s: times_s.to_vec(),
            distribution,
            radial_transport: stack_cells(&tendencies, |item| &item.radial_transport),
            electric_acceleration: stack_cells(&tendencies, |item| &item.electric_acceleration),
            collisional_drag_diffusion: stack_cells(&tendencies, |item| {
                &item.collisional_drag_diffusion
            }),
            pitch_scattering: stack_cells(&tendencies, |item| &item.pitch_scattering),
            cross_diffusion: stack_cells(&tendencies, |item| &item.cross_diffusion),
            synchrotron_loss: stack_cells(&tendencies, |item| &item.synchrotron_loss),
            bremsstrahlung_loss: stack_cells(&tendencies, |item| &item.bremsstrahlung_loss),
            avalanche_generation: stack_cells(&tendencies, |item| &item.avalanche_generation),
            external_source: stack_cells(&tendencies, |item| &item.external_source),
            total_tendency: tendencies.iter().flat_map(|item| item.total()).collect(),
            runaway_density_m3: density_history,
            runaway_density_radial_transport_m3_s: stack_cells(&tendencies, |item| {
                &item.runaway_density_radial_transport_m3_s
            }),
            runaway_density_avalanche_generation_m3_s: stack_cells(&tendencies, |item| {
                &item.runaway_density_avalanche_generation_m3_s
            }),
            runaway_density_external_source_m3_s: stack_cells(&tendencies, |item| {
                &item.runaway_density_external_source_m3_s
            }),
            runaway_density_tendency_m3_s: stack_cells(&tendencies, |item| {
                &item.runaway_density_tendency_m3_s
            }),
            moments,
            internal_steps,
            minimum_distribution,
        })
    }

    fn moments(&self, history: &[f64], nt: usize) -> RunawayKineticMoments {
        let grid = &self.operator.grid;
        let weight = &self.operator.geometry.density_cell_measure;
        let momentum = grid.momentum_mc();
        let pitch = grid.pitch();
        let cells = grid.cell_count();
        let mut density_m3 = vec![0.0; nt * grid.nr()];
        let mut current_density_a_m2 = vec![0.0; nt * grid.nr()];
        let mut kinetic_energy_density_j_m3 = vec![0.0; nt * grid.nr()];
        for it in 0..nt {
            for ir in 0..grid.nr() {
                for (ixi, pitch_value) in pitch.iter().enumerate() {
                    for (ip, momentum_value) in momentum.iter().enumerate() {
                        let cell = grid.cell_index(ir, ixi, ip);
                        let value = history[it * cells + cell];
                        let gamma = (1.0 + momentum_value * momentum_value).sqrt();
                        let parallel_speed =
                            SPEED_OF_LIGHT_M_PER_S * momentum_value * pitch_value / gamma;
                        let weighted = value * weight[cell];
                        density_m3[it * grid.nr() + ir] += weighted;
                        current_density_a_m2[it * grid.nr() + ir] +=
                            ELECTRON_CHARGE_C * weighted * parallel_speed;
                        kinetic_energy_density_j_m3[it * grid.nr() + ir] +=
                            weighted * (gamma - 1.0) * ELECTRON_REST_ENERGY_J;
                    }
                }
            }
        }
        RunawayKineticMoments {
            density_m3,
            current_density_a_m2,
            kinetic_energy_density_j_m3,
        }
    }
}

fn stack_cells<F>(tendencies: &[RunawayKineticTendencies], select: F) -> Vec<f64>
where
    F: Fn(&RunawayKineticTendencies) -> &[f64],
{
    tendencies
        .iter()
        .flat_map(|item| select(item).iter().copied())
        .collect()
}
