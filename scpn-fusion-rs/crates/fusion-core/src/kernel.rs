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
const MIN_CURRENT_INTEGRAL: f64 = 1e-9;

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

        if !mu0.is_finite() || mu0 <= 0.0 {
            return Err(FusionError::ConfigError(
                "physics.vacuum_permeability must be finite and > 0".to_string(),
            ));
        }
        if !i_target.is_finite() {
            return Err(FusionError::PhysicsViolation(
                "physics.plasma_current_target must be finite".to_string(),
            ));
        }
        if max_iter == 0 {
            return Err(FusionError::ConfigError(
                "solver.max_iterations must be >= 1".to_string(),
            ));
        }
        if !tol.is_finite() || tol <= 0.0 {
            return Err(FusionError::ConfigError(
                "solver.convergence_threshold must be finite and > 0".to_string(),
            ));
        }
        if nz < 2 || nr < 2 {
            return Err(FusionError::ConfigError(format!(
                "grid dimensions must be >= 2 in both axes, got nz={nz}, nr={nr}"
            )));
        }
        if !dr.is_finite() || !dz.is_finite() || dr == 0.0 || dz == 0.0 {
            return Err(FusionError::ConfigError(format!(
                "grid spacing must be finite and non-zero, got dr={dr}, dz={dz}"
            )));
        }
        let external_profiles: Option<(ProfileParams, ProfileParams)> =
            if self.external_profile_mode {
                let params_p = self.profile_params_p.ok_or_else(|| {
                    FusionError::ConfigError(
                        "external profile mode requires params_p to be set".to_string(),
                    )
                })?;
                let params_ff = self.profile_params_ff.ok_or_else(|| {
                    FusionError::ConfigError(
                        "external profile mode requires params_ff to be set".to_string(),
                    )
                })?;
                Self::validate_profile_params(&params_p, "params_p")?;
                Self::validate_profile_params(&params_ff, "params_ff")?;
                Some((params_p, params_ff))
            } else {
                None
            };

        // 1. Compute vacuum field
        let psi_vac = calculate_vacuum_field(&self.grid, &self.config.coils, mu0)?;
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
                find_x_point(&self.state.psi, &self.grid, self.config.dimensions.z_min)?;
            x_point_pos = (r_x, z_x);
            psi_boundary_val = psi_x;

            // Safety: if axis and boundary too close, use limiter mode
            if (psi_axis_val - psi_boundary_val).abs() < MIN_PSI_SEPARATION {
                psi_boundary_val = psi_axis_val * LIMITER_FALLBACK_FACTOR;
            }

            // 4b. Update nonlinear source
            if let Some((params_p, params_ff)) = external_profiles {
                self.state.j_phi = update_plasma_source_with_profiles(
                    SourceProfileContext {
                        psi: &self.state.psi,
                        grid: &self.grid,
                        psi_axis: psi_axis_val,
                        psi_boundary: psi_boundary_val,
                        mu0,
                        i_target,
                    },
                    &params_p,
                    &params_ff,
                )?;
            } else {
                self.state.j_phi = update_plasma_source_nonlinear(
                    &self.state.psi,
                    &self.grid,
                    psi_axis_val,
                    psi_boundary_val,
                    mu0,
                    i_target,
                )?;
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
        let (b_r, b_z) = compute_b_field(&self.state.psi, &self.grid)?;
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
        let i_target = self.config.physics.plasma_current_target;
        if !i_target.is_finite() {
            return Err(FusionError::ConfigError(
                "particle feedback requires finite plasma_current_target".to_string(),
            ));
        }
        if !self.grid.dr.is_finite()
            || !self.grid.dz.is_finite()
            || self.grid.dr == 0.0
            || self.grid.dz == 0.0
        {
            return Err(FusionError::ConfigError(format!(
                "particle feedback requires finite non-zero grid spacing, got dr={}, dz={}",
                self.grid.dr, self.grid.dz
            )));
        }
        if particle_j_phi.dim() != (self.grid.nz, self.grid.nr) {
            return Err(FusionError::PhysicsViolation(format!(
                "Particle feedback shape mismatch: expected ({}, {}), got {:?}",
                self.grid.nz,
                self.grid.nr,
                particle_j_phi.dim(),
            )));
        }
        if !coupling.is_finite() || !(0.0..=1.0).contains(&coupling) {
            return Err(FusionError::PhysicsViolation(
                "particle feedback coupling must be finite and in [0, 1]".to_string(),
            ));
        }
        if particle_j_phi.iter().any(|v| !v.is_finite()) {
            return Err(FusionError::PhysicsViolation(
                "particle feedback map must contain only finite values".to_string(),
            ));
        }
        let particle_integral = particle_j_phi.iter().sum::<f64>() * self.grid.dr * self.grid.dz;
        if !particle_integral.is_finite() {
            return Err(FusionError::PhysicsViolation(
                "particle feedback integral became non-finite".to_string(),
            ));
        }
        if coupling >= 1.0 - f64::EPSILON
            && i_target.abs() > MIN_CURRENT_INTEGRAL
            && particle_integral.abs() <= MIN_CURRENT_INTEGRAL
        {
            return Err(FusionError::PhysicsViolation(
                "particle feedback integral is near zero for coupling=1 with non-zero plasma current target".to_string(),
            ));
        }
        self.particle_current_feedback = Some(particle_j_phi);
        self.particle_feedback_coupling = coupling;
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
        if particles.is_empty() {
            return Err(FusionError::PhysicsViolation(
                "particle population must be non-empty".to_string(),
            ));
        }
        let summary = summarize_particle_population(particles, runaway_threshold_mev).map_err(
            |err| match err {
                FusionError::PhysicsViolation(msg) => FusionError::PhysicsViolation(format!(
                    "particle population summary failed: {msg}"
                )),
                FusionError::ConfigError(msg) => {
                    FusionError::ConfigError(format!("particle population summary failed: {msg}"))
                }
                other => other,
            },
        )?;
        let particle_j_phi =
            deposit_toroidal_current_density(particles, &self.grid).map_err(|err| match err {
                FusionError::PhysicsViolation(msg) => FusionError::PhysicsViolation(format!(
                    "particle current deposition failed: {msg}"
                )),
                FusionError::ConfigError(msg) => {
                    FusionError::ConfigError(format!("particle current deposition failed: {msg}"))
                }
                other => other,
            })?;
        self.set_particle_current_feedback(particle_j_phi, coupling)
            .map_err(|err| match err {
                FusionError::PhysicsViolation(msg) => FusionError::PhysicsViolation(format!(
                    "particle population feedback setup failed: {msg}"
                )),
                FusionError::ConfigError(msg) => FusionError::ConfigError(format!(
                    "particle population feedback setup failed: {msg}"
                )),
                other => other,
            })?;
        Ok(summary)
    }

    /// Convenience wrapper to solve with temporary external profile parameters.
    pub fn solve_equilibrium_with_profiles(
        &mut self,
        params_p: ProfileParams,
        params_ff: ProfileParams,
    ) -> FusionResult<EquilibriumResult> {
        Self::validate_profile_params(&params_p, "params_p")?;
        Self::validate_profile_params(&params_ff, "params_ff")?;
        let prev_mode = self.external_profile_mode;
        let prev_params_p = self.profile_params_p;
        let prev_params_ff = self.profile_params_ff;
        self.set_external_profiles(params_p, params_ff);
        let result = self.solve_equilibrium();
        self.external_profile_mode = prev_mode;
        self.profile_params_p = prev_params_p;
        self.profile_params_ff = prev_params_ff;
        result
    }

    fn nearest_index(axis: &Array1<f64>, value: f64) -> FusionResult<usize> {
        if axis.is_empty() {
            return Err(FusionError::ConfigError(
                "sample axis must contain at least one coordinate".to_string(),
            ));
        }
        if !value.is_finite() {
            return Err(FusionError::ConfigError(format!(
                "sample coordinate must be finite, got {value}"
            )));
        }

        let mut best_idx = 0usize;
        let mut best_dist = f64::INFINITY;
        for (idx, &x) in axis.iter().enumerate() {
            if !x.is_finite() {
                return Err(FusionError::ConfigError(format!(
                    "sample axis contains non-finite coordinate at index {idx}"
                )));
            }
            let d = (x - value).abs();
            if d < best_dist {
                best_dist = d;
                best_idx = idx;
            }
        }
        Ok(best_idx)
    }

    fn axis_bounds(axis: &Array1<f64>, label: &str) -> FusionResult<(f64, f64)> {
        if axis.is_empty() {
            return Err(FusionError::ConfigError(format!(
                "{label} axis must contain at least one coordinate"
            )));
        }
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        for (idx, &x) in axis.iter().enumerate() {
            if !x.is_finite() {
                return Err(FusionError::ConfigError(format!(
                    "{label} axis contains non-finite coordinate at index {idx}"
                )));
            }
            min_x = min_x.min(x);
            max_x = max_x.max(x);
        }
        Ok((min_x, max_x))
    }

    fn validate_profile_params(params: &ProfileParams, label: &str) -> FusionResult<()> {
        if !params.ped_top.is_finite() || params.ped_top <= 0.0 {
            return Err(FusionError::ConfigError(format!(
                "{label}.ped_top must be finite and > 0, got {}",
                params.ped_top
            )));
        }
        if !params.ped_width.is_finite() || params.ped_width <= 0.0 {
            return Err(FusionError::ConfigError(format!(
                "{label}.ped_width must be finite and > 0, got {}",
                params.ped_width
            )));
        }
        if !params.ped_height.is_finite() || !params.core_alpha.is_finite() {
            return Err(FusionError::ConfigError(format!(
                "{label}.ped_height/core_alpha must be finite"
            )));
        }
        Ok(())
    }

    /// Sample solved flux at nearest grid point to (R, Z).
    pub fn sample_psi_at(&self, r: f64, z: f64) -> FusionResult<f64> {
        if self.state.psi.dim() != (self.grid.nz, self.grid.nr) {
            return Err(FusionError::ConfigError(format!(
                "sample psi grid/state shape mismatch: psi={:?}, grid=({}, {})",
                self.state.psi.dim(),
                self.grid.nz,
                self.grid.nr
            )));
        }
        let (r_min, r_max) = Self::axis_bounds(&self.grid.r, "sample R")?;
        let (z_min, z_max) = Self::axis_bounds(&self.grid.z, "sample Z")?;
        if r < r_min || r > r_max || z < z_min || z > z_max {
            return Err(FusionError::ConfigError(format!(
                "sample coordinate outside grid domain: (R={r}, Z={z}) not in R[{r_min}, {r_max}] Z[{z_min}, {z_max}]"
            )));
        }
        let ir = Self::nearest_index(&self.grid.r, r)?;
        let iz = Self::nearest_index(&self.grid.z, z)?;
        let psi = self.state.psi[[iz, ir]];
        if !psi.is_finite() {
            return Err(FusionError::ConfigError(
                "sampled psi value is non-finite".to_string(),
            ));
        }
        Ok(psi)
    }

    /// Sample solved flux at multiple probe coordinates.
    pub fn sample_psi_at_probes(&self, probes: &[(f64, f64)]) -> FusionResult<Vec<f64>> {
        if probes.is_empty() {
            return Err(FusionError::ConfigError(
                "sample probes list must be non-empty".to_string(),
            ));
        }
        probes
            .iter()
            .enumerate()
            .map(|(idx, &(r, z))| {
                self.sample_psi_at(r, z).map_err(|err| match err {
                    FusionError::ConfigError(msg) => {
                        FusionError::ConfigError(format!("probe[{idx}] {msg}"))
                    }
                    other => other,
                })
            })
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
    fn test_sample_psi_rejects_non_finite_probe_coordinates() {
        let kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        assert!(kernel.sample_psi_at(f64::NAN, 0.0).is_err());
        assert!(kernel.sample_psi_at(6.2, f64::INFINITY).is_err());
        assert!(kernel
            .sample_psi_at_probes(&[(6.2, 0.0), (f64::NAN, 0.1)])
            .is_err());
    }

    #[test]
    fn test_sample_psi_rejects_non_finite_axis_coordinates() {
        let mut kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        kernel.grid.r[10] = f64::NAN;
        let err = kernel
            .sample_psi_at(6.2, 0.0)
            .expect_err("non-finite axis coordinate must fail");
        match err {
            FusionError::ConfigError(msg) => {
                assert!(msg.contains("non-finite coordinate"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_sample_psi_rejects_out_of_domain_probe_coordinates() {
        let kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        let r_min = kernel.grid().r[0].min(kernel.grid().r[kernel.grid().nr - 1]);
        let r_max = kernel.grid().r[0].max(kernel.grid().r[kernel.grid().nr - 1]);
        let z_min = kernel.grid().z[0].min(kernel.grid().z[kernel.grid().nz - 1]);
        let z_max = kernel.grid().z[0].max(kernel.grid().z[kernel.grid().nz - 1]);

        assert!(kernel.sample_psi_at(r_min - 1.0e-6, 0.0).is_err());
        assert!(kernel.sample_psi_at(r_max + 1.0e-6, 0.0).is_err());
        assert!(kernel.sample_psi_at(6.2, z_min - 1.0e-6).is_err());
        assert!(kernel.sample_psi_at(6.2, z_max + 1.0e-6).is_err());
        assert!(kernel
            .sample_psi_at_probes(&[(6.2, 0.0), (r_max + 1.0e-6, 0.1)])
            .is_err());
    }

    #[test]
    fn test_sample_psi_probe_errors_include_probe_index() {
        let kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        let r_max = kernel.grid().r[0].max(kernel.grid().r[kernel.grid().nr - 1]);
        let err = kernel
            .sample_psi_at_probes(&[(6.2, 0.0), (r_max + 1.0e-6, 0.1)])
            .expect_err("out-of-domain probe list must fail");
        match err {
            FusionError::ConfigError(msg) => {
                assert!(msg.contains("probe[1]"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_sample_psi_rejects_empty_probe_list() {
        let kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        let err = kernel
            .sample_psi_at_probes(&[])
            .expect_err("empty probe list must error");
        match err {
            FusionError::ConfigError(msg) => {
                assert!(msg.contains("non-empty"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_sample_psi_rejects_state_shape_mismatch() {
        let mut kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        kernel.state.psi = Array2::zeros((kernel.grid().nz - 1, kernel.grid().nr));
        let err = kernel
            .sample_psi_at(6.2, 0.0)
            .expect_err("mismatched psi shape must error");
        match err {
            FusionError::ConfigError(msg) => {
                assert!(msg.contains("shape mismatch"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
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
            .set_particle_current_feedback(feedback, 1.0)
            .expect("feedback set should succeed");
        assert!(kernel.particle_current_feedback.is_some());
        assert!((kernel.particle_feedback_coupling - 1.0).abs() < 1e-12);
        kernel.clear_particle_current_feedback();
        assert!(kernel.particle_current_feedback.is_none());
        assert_eq!(kernel.particle_feedback_coupling, 0.0);
    }

    #[test]
    fn test_particle_feedback_rejects_invalid_coupling() {
        let mut kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        let feedback = Array2::from_elem((kernel.grid().nz, kernel.grid().nr), 1.0);
        for coupling in [f64::NAN, -0.2, 1.2] {
            let err = kernel
                .set_particle_current_feedback(feedback.clone(), coupling)
                .expect_err("invalid coupling must error");
            match err {
                FusionError::PhysicsViolation(msg) => {
                    assert!(msg.contains("coupling"));
                }
                other => panic!("Unexpected error: {other:?}"),
            }
        }
    }

    #[test]
    fn test_particle_feedback_rejects_non_finite_map() {
        let mut kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        let mut feedback = Array2::from_elem((kernel.grid().nz, kernel.grid().nr), 1.0);
        feedback[[0, 0]] = f64::NAN;
        let err = kernel
            .set_particle_current_feedback(feedback, 0.2)
            .expect_err("non-finite feedback map must error");
        match err {
            FusionError::PhysicsViolation(msg) => {
                assert!(msg.contains("finite values"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_particle_feedback_rejects_invalid_grid_spacing() {
        let mut kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        let feedback = Array2::from_elem((kernel.grid().nz, kernel.grid().nr), 1.0);
        kernel.grid.dr = 0.0;
        let err = kernel
            .set_particle_current_feedback(feedback, 0.2)
            .expect_err("invalid grid spacing must fail");
        match err {
            FusionError::ConfigError(msg) => {
                assert!(msg.contains("grid spacing"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_particle_feedback_rejects_non_finite_plasma_current_target() {
        let mut kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        let feedback = Array2::from_elem((kernel.grid().nz, kernel.grid().nr), 1.0);
        kernel.config.physics.plasma_current_target = f64::NAN;
        let err = kernel
            .set_particle_current_feedback(feedback, 0.3)
            .expect_err("non-finite plasma_current_target must fail");
        match err {
            FusionError::ConfigError(msg) => {
                assert!(msg.contains("plasma_current_target"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_particle_feedback_rejects_zero_integral_with_full_coupling() {
        let mut kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        let zero_feedback = Array2::zeros((kernel.grid().nz, kernel.grid().nr));
        let err = kernel
            .set_particle_current_feedback(zero_feedback.clone(), 1.0)
            .expect_err("zero-integral full-coupling feedback must fail");
        match err {
            FusionError::PhysicsViolation(msg) => {
                assert!(msg.contains("coupling=1") || msg.contains("near zero"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }

        kernel
            .set_particle_current_feedback(zero_feedback, 0.5)
            .expect("partial-coupling zero feedback should remain valid");
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

    #[test]
    fn test_particle_feedback_from_population_rejects_invalid_threshold() {
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
        let err = kernel
            .set_particle_feedback_from_population(&particles, 0.2, -0.5)
            .expect_err("invalid runaway threshold must error");
        match err {
            FusionError::PhysicsViolation(msg) => {
                assert!(msg.contains("summary failed"));
                assert!(msg.contains("runaway_threshold_mev"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_particle_feedback_from_population_rejects_empty_population() {
        let mut kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        let err = kernel
            .set_particle_feedback_from_population(&[], 0.2, 0.5)
            .expect_err("empty particle population must error");
        match err {
            FusionError::PhysicsViolation(msg) => {
                assert!(msg.contains("non-empty"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_particle_feedback_from_population_rejects_out_of_domain_full_coupling() {
        let mut kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        let particles = vec![ChargedParticle {
            x_m: 100.0,
            y_m: 0.0,
            z_m: 100.0,
            vx_m_s: 0.0,
            vy_m_s: 2.0e7,
            vz_m_s: 0.0,
            charge_c: 1.602_176_634e-19,
            mass_kg: 1.672_621_923_69e-27,
            weight: 3.0e16,
        }];
        let err = kernel
            .set_particle_feedback_from_population(&particles, 1.0, 0.5)
            .expect_err("out-of-domain population with full coupling must fail");
        match err {
            FusionError::PhysicsViolation(msg) => {
                assert!(msg.contains("feedback setup failed"));
                assert!(msg.contains("coupling=1") || msg.contains("near zero"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_particle_feedback_from_population_wraps_deposition_errors() {
        let mut kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        kernel.grid.dr = 0.0;
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
        let err = kernel
            .set_particle_feedback_from_population(&particles, 0.3, 0.5)
            .expect_err("invalid grid spacing must fail during deposition");
        match err {
            FusionError::PhysicsViolation(msg) => {
                assert!(msg.contains("deposition failed"));
                assert!(msg.contains("grid spacing"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_particle_feedback_from_population_wraps_setup_config_errors() {
        let mut kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        kernel.config.physics.plasma_current_target = f64::NAN;
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
        let err = kernel
            .set_particle_feedback_from_population(&particles, 0.3, 0.5)
            .expect_err("non-finite plasma_current_target must fail during setup");
        match err {
            FusionError::ConfigError(msg) => {
                assert!(msg.contains("feedback setup failed"));
                assert!(msg.contains("plasma_current_target"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_solve_equilibrium_rejects_invalid_runtime_controls() {
        let mut kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        kernel.config.solver.max_iterations = 0;
        let err = kernel
            .solve_equilibrium()
            .expect_err("zero max_iterations must fail");
        match err {
            FusionError::ConfigError(msg) => {
                assert!(msg.contains("max_iterations"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }

        let mut kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        kernel.config.solver.convergence_threshold = 0.0;
        let err = kernel
            .solve_equilibrium()
            .expect_err("non-positive convergence_threshold must fail");
        match err {
            FusionError::ConfigError(msg) => {
                assert!(msg.contains("convergence_threshold"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }

        let mut kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        kernel.grid.dr = 0.0;
        let err = kernel
            .solve_equilibrium()
            .expect_err("zero grid spacing must fail");
        match err {
            FusionError::ConfigError(msg) => {
                assert!(msg.contains("grid spacing"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_solve_equilibrium_rejects_external_profile_mode_without_params() {
        let mut kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        kernel.external_profile_mode = true;
        kernel.profile_params_p = None;
        kernel.profile_params_ff = None;
        let err = kernel
            .solve_equilibrium()
            .expect_err("external profile mode without params must fail");
        match err {
            FusionError::ConfigError(msg) => {
                assert!(msg.contains("requires params_p"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_solve_equilibrium_rejects_invalid_external_profile_params() {
        let mut kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        let bad_p = ProfileParams {
            ped_top: 0.9,
            ped_width: 0.0,
            ped_height: 1.0,
            core_alpha: 0.2,
        };
        kernel.set_external_profiles(bad_p, ProfileParams::default());
        let err = kernel
            .solve_equilibrium()
            .expect_err("invalid external profile params must fail");
        match err {
            FusionError::ConfigError(msg) => {
                assert!(msg.contains("ped_width"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_solve_with_profiles_restores_profile_state_on_error() {
        let mut kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        kernel.config.solver.max_iterations = 0;
        assert!(!kernel.external_profile_mode);
        assert!(kernel.profile_params_p.is_none());
        assert!(kernel.profile_params_ff.is_none());

        let err = kernel
            .solve_equilibrium_with_profiles(ProfileParams::default(), ProfileParams::default())
            .expect_err("invalid runtime control must fail");
        match err {
            FusionError::ConfigError(msg) => {
                assert!(msg.contains("max_iterations"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }

        assert!(!kernel.external_profile_mode);
        assert!(kernel.profile_params_p.is_none());
        assert!(kernel.profile_params_ff.is_none());
    }

    #[test]
    fn test_solve_with_profiles_restores_previous_external_profile_state() {
        let mut kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        let prev_p = ProfileParams {
            ped_top: 0.85,
            ped_width: 0.07,
            ped_height: 0.95,
            core_alpha: 0.22,
        };
        let prev_ff = ProfileParams {
            ped_top: 0.92,
            ped_width: 0.09,
            ped_height: 1.05,
            core_alpha: 0.18,
        };
        kernel.set_external_profiles(prev_p, prev_ff);
        kernel.config.solver.max_iterations = 0;

        let next_p = ProfileParams {
            ped_top: 0.80,
            ped_width: 0.06,
            ped_height: 1.10,
            core_alpha: 0.25,
        };
        let next_ff = ProfileParams {
            ped_top: 0.95,
            ped_width: 0.10,
            ped_height: 0.90,
            core_alpha: 0.15,
        };
        let err = kernel
            .solve_equilibrium_with_profiles(next_p, next_ff)
            .expect_err("invalid runtime control must fail");
        match err {
            FusionError::ConfigError(msg) => {
                assert!(msg.contains("max_iterations"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }

        assert!(kernel.external_profile_mode);
        assert_eq!(kernel.profile_params_p, Some(prev_p));
        assert_eq!(kernel.profile_params_ff, Some(prev_ff));
    }

    #[test]
    fn test_solve_with_profiles_rejects_invalid_params_without_state_mutation() {
        let mut kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        let prev_p = ProfileParams {
            ped_top: 0.85,
            ped_width: 0.07,
            ped_height: 0.95,
            core_alpha: 0.22,
        };
        let prev_ff = ProfileParams {
            ped_top: 0.92,
            ped_width: 0.09,
            ped_height: 1.05,
            core_alpha: 0.18,
        };
        kernel.set_external_profiles(prev_p, prev_ff);

        let bad_p = ProfileParams {
            ped_top: 0.8,
            ped_width: 0.0,
            ped_height: 1.1,
            core_alpha: 0.25,
        };
        let err = kernel
            .solve_equilibrium_with_profiles(bad_p, ProfileParams::default())
            .expect_err("invalid profile params must fail before state mutation");
        match err {
            FusionError::ConfigError(msg) => {
                assert!(msg.contains("ped_width"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }

        assert!(kernel.external_profile_mode);
        assert_eq!(kernel.profile_params_p, Some(prev_p));
        assert_eq!(kernel.profile_params_ff, Some(prev_ff));
    }
}
