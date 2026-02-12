// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Kernel
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! FusionKernel — The Grad-Shafranov equilibrium solver.
//!
//! Port of fusion_kernel.py `FusionKernel` class (lines 173-313).
//! Implements the Picard iteration loop with nonlinear source term,
//! X-point detection, and convergence tracking.

use crate::bfield::compute_b_field;
use crate::source::{update_plasma_source_with_profiles, ProfileMode, ProfileParams};
use crate::vacuum::calculate_vacuum_field;
use crate::xpoint::find_x_point;
use fusion_math::multigrid::{multigrid_solve, MultigridConfig};
use fusion_types::config::ReactorConfig;
use fusion_types::error::{FusionError, FusionResult};
use fusion_types::state::{EquilibriumResult, Grid2D, PlasmaState};
use ndarray::Array2;

/// Picard relaxation factor (Python line 180: `alpha = 0.1`).
const DEFAULT_PICARD_RELAXATION: f64 = 0.1;

/// Gaussian sigma for seed current distribution (Python line 193: `sigma = 1.0`).
const SEED_GAUSSIAN_SIGMA: f64 = 1.0;

/// V-cycles for initial multigrid solve (replaces 50 Jacobi iterations).
const INITIAL_MG_CYCLES: usize = 10;

/// V-cycles per inner Picard step.
const INNER_MG_CYCLES: usize = 5;

/// Multigrid convergence tolerance for inner solves.
const MG_TOL: f64 = 1e-6;

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
    profile_mode: ProfileMode,
    p_prime_params: ProfileParams,
    ff_prime_params: ProfileParams,
}

impl FusionKernel {
    /// Create a new kernel from configuration.
    pub fn new(config: ReactorConfig) -> Self {
        let grid = config.create_grid();
        let state = PlasmaState::new(grid.nz, grid.nr);

        // Read profile config if present
        let (profile_mode, p_prime_params, ff_prime_params) =
            if let Some(ref pc) = config.physics.profiles {
                let mode = match pc.mode.as_str() {
                    "h-mode" | "H-mode" | "hmode" => ProfileMode::HMode,
                    _ => ProfileMode::LMode,
                };
                let p = ProfileParams {
                    ped_top: pc.p_prime.ped_top,
                    ped_width: pc.p_prime.ped_width,
                    ped_height: pc.p_prime.ped_height,
                    core_alpha: pc.p_prime.core_alpha,
                };
                let ff = ProfileParams {
                    ped_top: pc.ff_prime.ped_top,
                    ped_width: pc.ff_prime.ped_width,
                    ped_height: pc.ff_prime.ped_height,
                    core_alpha: pc.ff_prime.core_alpha,
                };
                (mode, p, ff)
            } else {
                (
                    ProfileMode::LMode,
                    ProfileParams::default(),
                    ProfileParams::default(),
                )
            };

        FusionKernel {
            config,
            grid,
            state,
            external_profile_mode: false,
            profile_mode,
            p_prime_params,
            ff_prime_params,
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
    /// 3. Initial multigrid solve (V-cycle with Red-Black SOR smoother)
    /// 4. Picard iteration loop:
    ///    a. Find O-point (axis) and X-point
    ///    b. Update nonlinear source term
    ///    c. Multigrid elliptic solve (correct GS stencil with 1/R terms)
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

        // Initial multigrid solve (replaces plain Jacobi — fixes missing 1/R terms)
        let mg_config = MultigridConfig::default();
        multigrid_solve(
            &mut self.state.psi,
            &source,
            &self.grid,
            &mg_config,
            INITIAL_MG_CYCLES,
            MG_TOL,
        );

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
            if !self.external_profile_mode {
                self.state.j_phi = update_plasma_source_with_profiles(
                    &self.state.psi,
                    &self.grid,
                    psi_axis_val,
                    psi_boundary_val,
                    mu0,
                    i_target,
                    self.profile_mode,
                    &self.p_prime_params,
                    &self.ff_prime_params,
                );
            }

            // Source = -μ₀ R J_phi
            for iz in 0..nz {
                for ir in 0..nr {
                    source[[iz, ir]] = -mu0 * self.grid.rr[[iz, ir]] * self.state.j_phi[[iz, ir]];
                }
            }

            // 4c. Multigrid elliptic solve (replaces Jacobi — correct GS stencil with 1/R terms)
            let mut psi_new = self.state.psi.clone();
            multigrid_solve(
                &mut psi_new,
                &source,
                &self.grid,
                &mg_config,
                INNER_MG_CYCLES,
                MG_TOL,
            );

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

    /// Set profile mode and pedestal parameters for p' and FF'.
    pub fn set_profile_mode(
        &mut self,
        mode: ProfileMode,
        p_params: ProfileParams,
        ff_params: ProfileParams,
    ) {
        self.profile_mode = mode;
        self.p_prime_params = p_params;
        self.ff_prime_params = ff_params;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn test_hmode_equilibrium_no_crash() {
        use crate::source::{ProfileMode, ProfileParams};

        let mut kernel =
            FusionKernel::from_file(&config_path("validation/iter_validated_config.json")).unwrap();

        // Switch to H-mode pedestal profiles
        kernel.set_profile_mode(
            ProfileMode::HMode,
            ProfileParams {
                ped_top: 0.92,
                ped_width: 0.05,
                ped_height: 1.0,
                core_alpha: 0.3,
            },
            ProfileParams {
                ped_top: 0.90,
                ped_width: 0.06,
                ped_height: 0.8,
                core_alpha: 0.2,
            },
        );

        let result = kernel.solve_equilibrium().unwrap();

        // Must not NaN
        assert!(
            !kernel.psi().iter().any(|v| v.is_nan()),
            "H-mode solve produced NaN"
        );
        assert!(
            !kernel.psi().iter().any(|v| v.is_infinite()),
            "H-mode solve produced Inf"
        );
        assert!(result.iterations > 0, "H-mode should run at least 1 iteration");
    }
}
