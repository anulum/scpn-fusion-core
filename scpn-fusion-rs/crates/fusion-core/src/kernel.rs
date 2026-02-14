//! FusionKernel — The Grad-Shafranov equilibrium solver.
//!
//! Port of fusion_kernel.py `FusionKernel` class (lines 173-313).
//! Implements the Picard iteration loop with nonlinear source term,
//! X-point detection, and convergence tracking.

use crate::bfield::compute_b_field;
use crate::particles::{
    blend_particle_current, deposit_toroidal_current_density, summarize_particle_population,
    ChargedParticle, ParticlePopulationSummary,
};
use crate::source::{
    update_plasma_source_nonlinear, update_plasma_source_with_profiles, ProfileParams,
    SourceProfileContext,
};
use crate::vacuum::calculate_vacuum_field;
use crate::xpoint::find_x_point;
use fusion_types::config::ReactorConfig;
use fusion_types::error::{FusionError, FusionResult};
use fusion_types::state::{EquilibriumResult, Grid2D, PlasmaState};
use ndarray::{Array1, Array2};

/// Picard relaxation factor (Python line 180: `alpha = 0.1`).
const DEFAULT_PICARD_RELAXATION: f64 = 0.1;

/// Gaussian sigma for seed current distribution (Python line 193: `sigma = 1.0`).
const SEED_GAUSSIAN_SIGMA: f64 = 1.0;

/// Number of Jacobi iterations for initial elliptic solve (Python line 204: `range(50)`).
const INITIAL_JACOBI_ITERS: usize = 50;

/// Number of inner SOR iterations per Picard step.
const INNER_SOLVE_ITERS: usize = 50;

/// Minimum Ψ_axis to avoid normalization singularity (Python line 218: `1e-6`).
const MIN_PSI_AXIS: f64 = 1e-6;

/// Minimum separation between Ψ_axis and Ψ_boundary (Python line 225: `0.1`).
const MIN_PSI_SEPARATION: f64 = 0.1;

/// Fallback factor when axis/boundary too close (Python line 226: `0.1`).
const LIMITER_FALLBACK_FACTOR: f64 = 0.1;

/// The Grad-Shafranov equilibrium solver.
pub struct FusionKernel {
    config: ReactorConfig,
    grid: Grid2D,
    state: PlasmaState,
    external_profile_mode: bool,
    profile_params_p: Option<ProfileParams>,
    profile_params_ff: Option<ProfileParams>,
    particle_current_feedback: Option<Array2<f64>>,
    particle_feedback_coupling: f64,
}

impl FusionKernel {
    /// Create a new kernel from configuration.
    pub fn new(config: ReactorConfig) -> Self {
        let grid = config.create_grid();
        let state = PlasmaState::new(grid.nz, grid.nr);

        FusionKernel {
            config,
            grid,
            state,
            external_profile_mode: false,
            profile_params_p: None,
            profile_params_ff: None,
            particle_current_feedback: None,
            particle_feedback_coupling: 0.0,
        }
    }

    /// Create a new kernel from a JSON config file.
    pub fn from_file(path: &str) -> FusionResult<Self> {
        let config = ReactorConfig::from_file(path)?;
        Ok(Self::new(config))
    }

    /// Main equilibrium solver. Port of solve_equilibrium().
    ///
    /// Algorithm:
    /// 1. Compute vacuum field from coils
    /// 2. Seed Gaussian current distribution
    /// 3. Initial Jacobi solve
    /// 4. Picard iteration loop:
    ///    a. Find O-point (axis) and X-point
    ///    b. Update nonlinear source term
    ///    c. Jacobi elliptic solve
    ///    d. Apply vacuum boundary conditions
    ///    e. Relax: Ψ = (1-α)Ψ_old + α Ψ_new
    ///    f. Check convergence
    pub fn solve_equilibrium(&mut self) -> FusionResult<EquilibriumResult> {
        let start = std::time::Instant::now();
        let mu0 = self.config.physics.vacuum_permeability;
        let i_target = self.config.physics.plasma_current_target;
        let max_iter = self.config.solver.max_iterations;
        let tol = self.config.solver.convergence_threshold;
        let alpha = DEFAULT_PICARD_RELAXATION;

        let nz = self.grid.nz;
        let nr = self.grid.nr;
        let dr = self.grid.dr;
        let dz = self.grid.dz;

        // 1. Compute vacuum field
        let psi_vac = calculate_vacuum_field(&self.grid, &self.config.coils, mu0);
        self.state.psi = psi_vac.clone();
        let psi_vac_boundary = psi_vac;

        // 2. Seed Gaussian current
        let r_center = (self.config.dimensions.r_min + self.config.dimensions.r_max) / 2.0;
        let z_center = 0.0;

        for iz in 0..nz {
            for ir in 0..nr {
                let r = self.grid.rr[[iz, ir]];
                let z = self.grid.zz[[iz, ir]];
                let dist_sq = (r - r_center).powi(2) + (z - z_center).powi(2);
                self.state.j_phi[[iz, ir]] =
                    (-dist_sq / (2.0 * SEED_GAUSSIAN_SIGMA * SEED_GAUSSIAN_SIGMA)).exp();
            }
        }

        // Normalize seed current
        let i_seed: f64 = self.state.j_phi.iter().sum::<f64>() * dr * dz;
        if i_seed > 0.0 {
            let scale = i_target / i_seed;
            self.state.j_phi.mapv_inplace(|v| v * scale);
        }

        // 3. Source = -μ₀ R J_phi
        let mut source = Array2::zeros((nz, nr));
        for iz in 0..nz {
            for ir in 0..nr {
                source[[iz, ir]] = -mu0 * self.grid.rr[[iz, ir]] * self.state.j_phi[[iz, ir]];
            }
        }

        // Initial Jacobi solve
        let dr_sq = dr * dr;
        for _ in 0..INITIAL_JACOBI_ITERS {
            for iz in 1..nz - 1 {
                for ir in 1..nr - 1 {
                    self.state.psi[[iz, ir]] = 0.25
                        * (self.state.psi[[iz - 1, ir]]
                            + self.state.psi[[iz + 1, ir]]
                            + self.state.psi[[iz, ir - 1]]
                            + self.state.psi[[iz, ir + 1]]
                            - dr_sq * source[[iz, ir]]);
                }
            }
        }

        // 4. Picard iteration
        let mut x_point_pos = (0.0_f64, 0.0_f64);
        let mut psi_best = self.state.psi.clone();
        let mut diff_best = f64::MAX;
        let mut converged = false;
        let mut final_iter = 0;
        let mut final_residual = f64::MAX;
        let mut psi_axis_val = 0.0_f64;
        let mut psi_boundary_val = 0.0_f64;
        let mut axis_position = (0.0_f64, 0.0_f64);

        for k in 0..max_iter {
            // 4a. Find O-point (axis) — maximum of Ψ
            let mut max_psi = f64::NEG_INFINITY;
            let mut ir_ax = 0;
            let mut iz_ax = 0;
            for iz in 0..nz {
                for ir in 0..nr {
                    if self.state.psi[[iz, ir]] > max_psi {
                        max_psi = self.state.psi[[iz, ir]];
                        iz_ax = iz;
                        ir_ax = ir;
                    }
                }
            }
            psi_axis_val = max_psi;
            if psi_axis_val.abs() < MIN_PSI_AXIS {
                psi_axis_val = MIN_PSI_AXIS;
            }
            axis_position = (self.grid.r[ir_ax], self.grid.z[iz_ax]);

            // 4a. Find X-point
            let ((r_x, z_x), psi_x) =
                find_x_point(&self.state.psi, &self.grid, self.config.dimensions.z_min);
            x_point_pos = (r_x, z_x);
            psi_boundary_val = psi_x;

            // Safety: if axis and boundary too close, use limiter mode
            if (psi_axis_val - psi_boundary_val).abs() < MIN_PSI_SEPARATION {
                psi_boundary_val = psi_axis_val * LIMITER_FALLBACK_FACTOR;
            }

            // 4b. Update nonlinear source
            if self.external_profile_mode {
                if let (Some(params_p), Some(params_ff)) = (
                    self.profile_params_p.as_ref(),
                    self.profile_params_ff.as_ref(),
                ) {
                    self.state.j_phi = update_plasma_source_with_profiles(
                        SourceProfileContext {
                            psi: &self.state.psi,
                            grid: &self.grid,
                            psi_axis: psi_axis_val,
                            psi_boundary: psi_boundary_val,
                            mu0,
                            i_target,
                        },
                        params_p,
                        params_ff,
                    );
                }
            } else {
                self.state.j_phi = update_plasma_source_nonlinear(
                    &self.state.psi,
                    &self.grid,
                    psi_axis_val,
                    psi_boundary_val,
                    mu0,
                    i_target,
                );
            }
            if let Some(particle_j_phi) = self.particle_current_feedback.as_ref() {
                self.state.j_phi = blend_particle_current(
                    &self.state.j_phi,
                    particle_j_phi,
                    &self.grid,
                    i_target,
                    self.particle_feedback_coupling,
                )?;
            }

            // Source = -μ₀ R J_phi
            for iz in 0..nz {
                for ir in 0..nr {
                    source[[iz, ir]] = -mu0 * self.grid.rr[[iz, ir]] * self.state.j_phi[[iz, ir]];
                }
            }

            // 4c. Jacobi elliptic solve
            let mut psi_new = self.state.psi.clone();
            for _ in 0..INNER_SOLVE_ITERS {
                for iz in 1..nz - 1 {
                    for ir in 1..nr - 1 {
                        psi_new[[iz, ir]] = 0.25
                            * (psi_new[[iz - 1, ir]]
                                + psi_new[[iz + 1, ir]]
                                + psi_new[[iz, ir - 1]]
                                + psi_new[[iz, ir + 1]]
                                - dr_sq * source[[iz, ir]]);
                    }
                }
            }

            // 4d. Apply vacuum boundary conditions
            for ir in 0..nr {
                psi_new[[0, ir]] = psi_vac_boundary[[0, ir]];
                psi_new[[nz - 1, ir]] = psi_vac_boundary[[nz - 1, ir]];
            }
            for iz in 0..nz {
                psi_new[[iz, 0]] = psi_vac_boundary[[iz, 0]];
                psi_new[[iz, nr - 1]] = psi_vac_boundary[[iz, nr - 1]];
            }

            // Robustness check
            if psi_new.iter().any(|v| v.is_nan() || v.is_infinite()) {
                self.state.psi = psi_best;
                return Err(FusionError::SolverDiverged {
                    iteration: k,
                    message: "NaN or Inf detected in solution".to_string(),
                });
            }

            // 4e. Relax
            let mut diff = 0.0_f64;
            let mut count = 0;
            for iz in 0..nz {
                for ir in 0..nr {
                    diff += (psi_new[[iz, ir]] - self.state.psi[[iz, ir]]).abs();
                    count += 1;
                    self.state.psi[[iz, ir]] =
                        (1.0 - alpha) * self.state.psi[[iz, ir]] + alpha * psi_new[[iz, ir]];
                }
            }
            diff /= count as f64;

            // Save best
            if diff < diff_best {
                diff_best = diff;
                psi_best = self.state.psi.clone();
            }

            final_iter = k;
            final_residual = diff;

            // 4f. Check convergence
            if diff < tol {
                converged = true;
                break;
            }
        }

        // Finalize: compute B-field
        let (b_r, b_z) = compute_b_field(&self.state.psi, &self.grid);
        self.state.b_r = Some(b_r);
        self.state.b_z = Some(b_z);
        self.state.axis = Some(axis_position);
        self.state.x_point = Some(x_point_pos);
        self.state.psi_axis = psi_axis_val;
        self.state.psi_boundary = psi_boundary_val;

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        Ok(EquilibriumResult {
            converged,
            iterations: final_iter + 1,
            residual: final_residual,
            axis_position,
            x_point_position: x_point_pos,
            psi_axis: psi_axis_val,
            psi_boundary: psi_boundary_val,
            solve_time_ms: elapsed,
        })
    }

    /// Read-only access to flux.
    pub fn psi(&self) -> &Array2<f64> {
        &self.state.psi
    }

    /// Read-only access to current density.
    pub fn j_phi(&self) -> &Array2<f64> {
        &self.state.j_phi
    }

    /// Read-only access to the plasma state.
    pub fn state(&self) -> &PlasmaState {
        &self.state
    }

    /// Read-only access to the grid.
    pub fn grid(&self) -> &Grid2D {
        &self.grid
    }

    /// Read-only access to the configuration.
    pub fn config(&self) -> &ReactorConfig {
        &self.config
    }

    /// Enable externally specified mTanh profiles for source update.
    pub fn set_external_profiles(&mut self, params_p: ProfileParams, params_ff: ProfileParams) {
        self.external_profile_mode = true;
        self.profile_params_p = Some(params_p);
        self.profile_params_ff = Some(params_ff);
    }

    /// Disable external profile mode and revert to default nonlinear source update.
    pub fn clear_external_profiles(&mut self) {
        self.external_profile_mode = false;
        self.profile_params_p = None;
        self.profile_params_ff = None;
    }

    /// Inject a static particle-current map blended into fluid J_phi each Picard step.
    pub fn set_particle_current_feedback(
        &mut self,
        particle_j_phi: Array2<f64>,
        coupling: f64,
    ) -> FusionResult<()> {
        if particle_j_phi.dim() != (self.grid.nz, self.grid.nr) {
            return Err(FusionError::PhysicsViolation(format!(
                "Particle feedback shape mismatch: expected ({}, {}), got {:?}",
                self.grid.nz,
                self.grid.nr,
                particle_j_phi.dim(),
            )));
        }
        self.particle_current_feedback = Some(particle_j_phi);
        self.particle_feedback_coupling = coupling.clamp(0.0, 1.0);
        Ok(())
    }

    /// Disable particle-current feedback for subsequent solves.
    pub fn clear_particle_current_feedback(&mut self) {
        self.particle_current_feedback = None;
        self.particle_feedback_coupling = 0.0;
    }

    /// Build particle feedback map from macro-particle population and enable coupling.
    pub fn set_particle_feedback_from_population(
        &mut self,
        particles: &[ChargedParticle],
        coupling: f64,
        runaway_threshold_mev: f64,
    ) -> FusionResult<ParticlePopulationSummary> {
        let summary = summarize_particle_population(particles, runaway_threshold_mev);
        let particle_j_phi = deposit_toroidal_current_density(particles, &self.grid);
        self.set_particle_current_feedback(particle_j_phi, coupling)?;
        Ok(summary)
    }

    /// Convenience wrapper to solve with temporary external profile parameters.
    pub fn solve_equilibrium_with_profiles(
        &mut self,
        params_p: ProfileParams,
        params_ff: ProfileParams,
    ) -> FusionResult<EquilibriumResult> {
        self.set_external_profiles(params_p, params_ff);
        self.solve_equilibrium()
    }

    fn nearest_index(axis: &Array1<f64>, value: f64) -> usize {
        let mut best_idx = 0usize;
        let mut best_dist = f64::INFINITY;
        for (idx, &x) in axis.iter().enumerate() {
            let d = (x - value).abs();
            if d < best_dist {
                best_dist = d;
                best_idx = idx;
            }
        }
        best_idx
    }

    /// Sample solved flux at nearest grid point to (R, Z).
    pub fn sample_psi_at(&self, r: f64, z: f64) -> f64 {
        let ir = Self::nearest_index(&self.grid.r, r);
        let iz = Self::nearest_index(&self.grid.z, z);
        self.state.psi[[iz, ir]]
    }

    /// Sample solved flux at multiple probe coordinates.
    pub fn sample_psi_at_probes(&self, probes: &[(f64, f64)]) -> Vec<f64> {
        probes
            .iter()
            .map(|&(r, z)| self.sample_psi_at(r, z))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use std::path::PathBuf;

    fn project_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
    }

    fn config_path(relative: &str) -> String {
        project_root().join(relative).to_string_lossy().to_string()
    }

    #[test]
    fn test_kernel_creation() {
        let kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        assert_eq!(kernel.grid().nr, 128);
        assert_eq!(kernel.grid().nz, 128);
        assert_eq!(kernel.config().coils.len(), 7);

        // Can access state
        assert_eq!(kernel.psi().shape(), &[128, 128]);
    }

    #[test]
    fn test_full_equilibrium_iter_config() {
        let mut kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        let result = kernel.solve_equilibrium().unwrap();

        // Should run through iterations (may or may not converge in 1000 iters)
        assert!(
            result.converged || result.iterations == 1000,
            "Should either converge or exhaust iterations"
        );

        // No NaN in solution
        assert!(
            !kernel.psi().iter().any(|v| v.is_nan()),
            "Ψ contains NaN after solve"
        );

        // X-point should be in lower half (Z < 0)
        assert!(
            result.x_point_position.1 < 0.0,
            "X-point Z={} should be negative",
            result.x_point_position.1
        );
    }

    #[test]
    fn test_b_field_computed_after_solve() {
        let mut kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        kernel.solve_equilibrium().unwrap();

        assert!(kernel.state().b_r.is_some(), "B_R should be computed");
        assert!(kernel.state().b_z.is_some(), "B_Z should be computed");

        // B-field should have no NaN
        let b_r = kernel.state().b_r.as_ref().unwrap();
        let b_z = kernel.state().b_z.as_ref().unwrap();
        assert!(!b_r.iter().any(|v| v.is_nan()), "B_R contains NaN");
        assert!(!b_z.iter().any(|v| v.is_nan()), "B_Z contains NaN");
    }

    #[test]
    fn test_validated_config_equilibrium() {
        let mut kernel =
            FusionKernel::from_file(&config_path("validation/iter_validated_config.json")).unwrap();
        let result = kernel.solve_equilibrium().unwrap();

        // 65x65 grid should be faster
        assert!(result.iterations > 0, "Should run at least 1 iteration");
        assert!(
            !kernel.psi().iter().any(|v| v.is_nan()),
            "No NaN in validated config solve"
        );
    }

    #[test]
    fn test_particle_feedback_shape_guard() {
        let mut kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        let bad_shape = Array2::zeros((8, 8));
        let err = kernel
            .set_particle_current_feedback(bad_shape, 0.4)
            .unwrap_err();
        match err {
            FusionError::PhysicsViolation(msg) => {
                assert!(msg.contains("shape mismatch"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_particle_feedback_set_and_clear() {
        let mut kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        let feedback = Array2::from_elem((kernel.grid().nz, kernel.grid().nr), 1.0);
        kernel
            .set_particle_current_feedback(feedback, 1.5)
            .expect("feedback set should succeed");
        assert!(kernel.particle_current_feedback.is_some());
        assert!((kernel.particle_feedback_coupling - 1.0).abs() < 1e-12);
        kernel.clear_particle_current_feedback();
        assert!(kernel.particle_current_feedback.is_none());
        assert_eq!(kernel.particle_feedback_coupling, 0.0);
    }

    #[test]
    fn test_particle_feedback_from_population_builds_summary() {
        let mut kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        let particles = vec![ChargedParticle {
            x_m: 6.1,
            y_m: 0.0,
            z_m: 0.0,
            vx_m_s: 0.0,
            vy_m_s: 2.0e7,
            vz_m_s: 0.0,
            charge_c: 1.602_176_634e-19,
            mass_kg: 1.672_621_923_69e-27,
            weight: 3.0e16,
        }];

        let summary = kernel
            .set_particle_feedback_from_population(&particles, 0.35, 0.5)
            .expect("population feedback should be set");
        assert_eq!(summary.count, 1);
        assert!(summary.max_energy_mev > 0.5);
        assert!(kernel.particle_current_feedback.is_some());
        assert!((kernel.particle_feedback_coupling - 0.35).abs() < 1e-12);
    }
}
