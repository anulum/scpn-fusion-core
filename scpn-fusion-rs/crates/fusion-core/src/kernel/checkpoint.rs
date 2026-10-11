// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Origin-bound complete native kernel checkpoint
//! Restore the original complete native state only to its producing kernel.

use super::{FusionKernel, SolverMethod};
use crate::source::ProfileParams;
use fusion_types::config::ReactorConfig;
use fusion_types::error::{FusionError, FusionResult};
use fusion_types::state::{Grid2D, PlasmaState};
use ndarray::Array2;
use std::sync::Arc;

#[derive(Clone)]
struct SavedKernelState {
    config: ReactorConfig,
    grid: Grid2D,
    state: PlasmaState,
    external_profile_mode: bool,
    profile_params_p: Option<ProfileParams>,
    profile_params_ff: Option<ProfileParams>,
    particle_current_feedback: Option<Array2<f64>>,
    particle_feedback_coupling: f64,
    solver_method: SolverMethod,
}

/// Opaque original state issued by one native kernel.
///
/// No constructor, state setters or byte deserializer are supplied. A checkpoint
/// preserves configuration, grid, all plasma fields, external profiles, particle
/// feedback and method selection. It grants in-process restoration only, never
/// equilibrium acceptance, detached installation or physical authority.
pub struct KernelCheckpoint {
    origin: Arc<()>,
    saved: SavedKernelState,
}

impl FusionKernel {
    /// Capture the complete actual kernel state before an attempted operation.
    ///
    /// Array values are independent copies; no caller-owned replacement state is
    /// consumed. Snapshot allocation and work still require the caller's resource
    /// supervisor. Exhaustive field matching makes newly added kernel fields a
    /// compile error until their checkpoint behaviour is specified.
    pub fn snapshot_state(&self) -> KernelCheckpoint {
        let Self {
            checkpoint_origin,
            config,
            grid,
            state,
            external_profile_mode,
            profile_params_p,
            profile_params_ff,
            particle_current_feedback,
            particle_feedback_coupling,
            solver_method,
        } = self;
        KernelCheckpoint {
            origin: Arc::clone(checkpoint_origin),
            saved: SavedKernelState {
                config: config.clone(),
                grid: grid.clone(),
                state: state.clone(),
                external_profile_mode: *external_profile_mode,
                profile_params_p: *profile_params_p,
                profile_params_ff: *profile_params_ff,
                particle_current_feedback: particle_current_feedback.clone(),
                particle_feedback_coupling: *particle_feedback_coupling,
                solver_method: *solver_method,
            },
        }
    }

    /// Restore only an original checkpoint issued by this exact kernel.
    ///
    /// Recipient mismatch is refused before copying or mutating state. All
    /// independent saved values are copied before the first replacement, so a
    /// failed copy cannot leave a partially installed kernel in this process.
    /// No numerical operation, normalisation or equilibrium retry is performed.
    pub fn restore_state(&mut self, checkpoint: &KernelCheckpoint) -> FusionResult<()> {
        if !Arc::ptr_eq(&self.checkpoint_origin, &checkpoint.origin) {
            return Err(FusionError::ConfigError(
                "Native checkpoint belongs to another kernel".to_string(),
            ));
        }
        let saved = checkpoint.saved.clone();
        let Self {
            checkpoint_origin: _,
            config,
            grid,
            state,
            external_profile_mode,
            profile_params_p,
            profile_params_ff,
            particle_current_feedback,
            particle_feedback_coupling,
            solver_method,
        } = self;
        *config = saved.config;
        *grid = saved.grid;
        *state = saved.state;
        *external_profile_mode = saved.external_profile_mode;
        *profile_params_p = saved.profile_params_p;
        *profile_params_ff = saved.profile_params_ff;
        *particle_current_feedback = saved.particle_current_feedback;
        *particle_feedback_coupling = saved.particle_feedback_coupling;
        *solver_method = saved.solver_method;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn config() -> ReactorConfig {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../validation/iter_config.json");
        let mut config = ReactorConfig::from_file(path.to_str().unwrap()).unwrap();
        config.grid_resolution = [17, 17];
        config.solver.max_iterations = 2;
        config.solver.convergence_threshold = 1e-12;
        config
    }

    fn state_bits(kernel: &FusionKernel) -> Vec<u64> {
        let state = kernel.state();
        let mut bits = Vec::new();
        for field in [&state.psi, &state.j_phi] {
            bits.extend(field.iter().map(|value| value.to_bits()));
        }
        for field in [&state.b_r, &state.b_z] {
            bits.push(u64::from(field.is_some()));
            if let Some(field) = field {
                bits.extend(field.iter().map(|value| value.to_bits()));
            }
        }
        for point in [state.axis, state.x_point] {
            bits.push(u64::from(point.is_some()));
            if let Some((r, z)) = point {
                bits.extend([r.to_bits(), z.to_bits()]);
            }
        }
        bits.extend([state.psi_axis.to_bits(), state.psi_boundary.to_bits()]);
        bits
    }

    #[test]
    fn original_cold_checkpoint_restores_all_fields_and_presence() {
        let mut kernel = FusionKernel::new(config());
        let before = state_bits(&kernel);
        let checkpoint = kernel.snapshot_state();
        kernel.set_solver_method(SolverMethod::PicardMultigrid);
        kernel.solve_equilibrium().unwrap();
        assert_ne!(state_bits(&kernel), before);
        kernel.restore_state(&checkpoint).unwrap();
        assert_eq!(state_bits(&kernel), before);
        assert_eq!(kernel.solver_method(), SolverMethod::PicardSor);
        kernel.restore_state(&checkpoint).unwrap();
        assert_eq!(state_bits(&kernel), before);
    }

    #[test]
    fn warm_checkpoint_restores_profile_feedback_and_deterministic_replay() {
        let mut kernel = FusionKernel::new(config());
        let profile = ProfileParams {
            ped_height: 1.3,
            ..ProfileParams::default()
        };
        kernel.set_external_profiles(profile, ProfileParams::default());
        kernel
            .set_particle_current_feedback(Array2::from_elem((17, 17), 1000.0), 0.2)
            .unwrap();
        kernel.solve_equilibrium().unwrap();
        let before = state_bits(&kernel);
        let checkpoint = kernel.snapshot_state();
        kernel.clear_external_profiles();
        kernel.clear_particle_current_feedback();
        kernel.set_solver_method(SolverMethod::PicardMultigrid);
        kernel.solve_equilibrium().unwrap();
        assert_ne!(state_bits(&kernel), before);
        kernel.restore_state(&checkpoint).unwrap();
        assert_eq!(state_bits(&kernel), before);
        assert_eq!(kernel.solver_method(), SolverMethod::PicardSor);
        kernel.solve_equilibrium().unwrap();
        assert_eq!(state_bits(&kernel), before);
    }

    #[test]
    fn foreign_checkpoint_refuses_identical_configuration_before_mutation() {
        let config = config();
        let origin = FusionKernel::new(config.clone());
        let checkpoint = origin.snapshot_state();
        let mut recipient = FusionKernel::new(config);
        recipient.set_solver_method(SolverMethod::PicardMultigrid);
        recipient.solve_equilibrium().unwrap();
        let before = state_bits(&recipient);
        assert!(recipient.restore_state(&checkpoint).is_err());
        assert_eq!(state_bits(&recipient), before);
        assert_eq!(recipient.solver_method(), SolverMethod::PicardMultigrid);
    }

    #[test]
    fn original_checkpoint_restores_after_actual_profile_validation_failure() {
        let mut kernel = FusionKernel::new(config());
        kernel.solve_equilibrium().unwrap();
        let before = state_bits(&kernel);
        let checkpoint = kernel.snapshot_state();
        kernel.set_external_profiles(
            ProfileParams {
                ped_width: 0.0,
                ..ProfileParams::default()
            },
            ProfileParams::default(),
        );
        assert!(kernel.solve_equilibrium().is_err());
        kernel.restore_state(&checkpoint).unwrap();
        assert_eq!(state_bits(&kernel), before);
        kernel.solve_equilibrium().unwrap();
        assert_eq!(state_bits(&kernel), before);
    }
}
