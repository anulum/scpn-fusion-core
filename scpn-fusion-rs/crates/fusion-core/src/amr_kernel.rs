// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — AMR-Aware Kernel Solver
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! AMR-assisted equilibrium solve wrapper.

use fusion_math::amr::{estimate_error_field, AmrHierarchy};
use fusion_math::sor::sor_solve;
use fusion_types::error::{FusionError, FusionResult};
use fusion_types::state::Grid2D;
use ndarray::Array2;

#[derive(Debug, Clone)]
pub struct AmrKernelConfig {
    pub max_levels: usize,
    pub refinement_threshold: f64,
    pub omega: f64,
    pub coarse_iters: usize,
    pub patch_iters: usize,
    pub blend: f64,
}

impl Default for AmrKernelConfig {
    fn default() -> Self {
        Self {
            max_levels: 2,
            refinement_threshold: 0.1,
            omega: 1.8,
            coarse_iters: 400,
            patch_iters: 300,
            blend: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AmrKernelSolver {
    pub config: AmrKernelConfig,
}

impl AmrKernelSolver {
    pub fn new(config: AmrKernelConfig) -> FusionResult<Self> {
        if config.max_levels == 0 {
            return Err(FusionError::ConfigError(
                "AMR max_levels must be >= 1".to_string(),
            ));
        }
        if config.refinement_threshold.is_nan() || config.refinement_threshold.is_sign_negative() {
            return Err(FusionError::ConfigError(
                "AMR refinement_threshold must be >= 0 (or +inf)".to_string(),
            ));
        }
        if !config.blend.is_finite() || !(0.0..=1.0).contains(&config.blend) {
            return Err(FusionError::ConfigError(
                "AMR blend must be finite and in [0, 1]".to_string(),
            ));
        }
        if !config.omega.is_finite() || config.omega <= 0.0 {
            return Err(FusionError::ConfigError(
                "AMR omega must be finite and > 0".to_string(),
            ));
        }
        if config.coarse_iters == 0 || config.patch_iters == 0 {
            return Err(FusionError::ConfigError(
                "AMR iteration counts must be >= 1".to_string(),
            ));
        }
        Ok(Self { config })
    }

    fn validate_source_inputs(&self, base_grid: &Grid2D, source: &Array2<f64>) -> FusionResult<()> {
        if source.nrows() != base_grid.nz || source.ncols() != base_grid.nr {
            return Err(FusionError::ConfigError(format!(
                "AMR source shape mismatch: expected ({}, {}), got ({}, {})",
                base_grid.nz,
                base_grid.nr,
                source.nrows(),
                source.ncols()
            )));
        }
        if source.iter().any(|v| !v.is_finite()) {
            return Err(FusionError::ConfigError(
                "AMR source must contain only finite values".to_string(),
            ));
        }
        Ok(())
    }

    pub fn solve(&self, base_grid: &Grid2D, source: &Array2<f64>) -> FusionResult<Array2<f64>> {
        self.solve_with_hierarchy(base_grid, source)
            .map(|(psi, _)| psi)
    }

    pub fn solve_with_hierarchy(
        &self,
        base_grid: &Grid2D,
        source: &Array2<f64>,
    ) -> FusionResult<(Array2<f64>, AmrHierarchy)> {
        self.validate_source_inputs(base_grid, source)?;
        let mut psi = Array2::zeros((base_grid.nz, base_grid.nr));
        sor_solve(
            &mut psi,
            source,
            base_grid,
            self.config.omega,
            self.config.coarse_iters,
        );
        if psi.iter().any(|v| !v.is_finite()) {
            return Err(FusionError::ConfigError(
                "AMR coarse solve produced non-finite psi values".to_string(),
            ));
        }

        let error = estimate_error_field(source, base_grid);
        let mut hierarchy = AmrHierarchy::new(
            base_grid.clone(),
            self.config.max_levels,
            self.config.refinement_threshold,
        );
        hierarchy.refine(&error);

        if hierarchy.patches.is_empty() {
            return Ok((psi, hierarchy));
        }

        fn refinement_scale(level: usize) -> usize {
            let max_shift = usize::BITS as usize - 1;
            let shift = level.min(max_shift) as u32;
            1usize << shift
        }

        for patch in &mut hierarchy.patches {
            let (iz_lo, _iz_hi, ir_lo, _ir_hi) = patch.bounds;
            let scale = refinement_scale(patch.level);

            for pz in 0..patch.grid.nz {
                for pr in 0..patch.grid.nr {
                    let base_iz = (iz_lo + pz / scale).min(base_grid.nz - 1);
                    let base_ir = (ir_lo + pr / scale).min(base_grid.nr - 1);
                    patch.psi[[pz, pr]] = psi[[base_iz, base_ir]];
                }
            }

            let patch_source = Array2::from_shape_fn((patch.grid.nz, patch.grid.nr), |(pz, pr)| {
                let base_iz = (iz_lo + pz / scale).min(base_grid.nz - 1);
                let base_ir = (ir_lo + pr / scale).min(base_grid.nr - 1);
                source[[base_iz, base_ir]]
            });

            sor_solve(
                &mut patch.psi,
                &patch_source,
                &patch.grid,
                self.config.omega,
                self.config.patch_iters,
            );
            if patch.psi.iter().any(|v| !v.is_finite()) {
                return Err(FusionError::ConfigError(
                    "AMR patch solve produced non-finite psi values".to_string(),
                ));
            }
        }

        let blend = self.config.blend;
        for patch in &hierarchy.patches {
            let scale = refinement_scale(patch.level);
            for iz in patch.bounds.0..=patch.bounds.1 {
                for ir in patch.bounds.2..=patch.bounds.3 {
                    let pz = (iz - patch.bounds.0) * scale;
                    let pr = (ir - patch.bounds.2) * scale;
                    if pz >= patch.grid.nz || pr >= patch.grid.nr {
                        continue;
                    }
                    let patch_val = patch.psi[[pz, pr]];
                    psi[[iz, ir]] = (1.0 - blend) * psi[[iz, ir]] + blend * patch_val;
                }
            }
        }

        if psi.iter().any(|v| !v.is_finite()) {
            return Err(FusionError::ConfigError(
                "AMR blended solve produced non-finite psi values".to_string(),
            ));
        }

        Ok((psi, hierarchy))
    }
}

impl Default for AmrKernelSolver {
    fn default() -> Self {
        Self::new(AmrKernelConfig::default()).expect("default AMR config must be valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian_source(grid: &Grid2D) -> Array2<f64> {
        let r0 = grid.r[grid.nr - 1] - 0.08 * (grid.r[grid.nr - 1] - grid.r[0]);
        let z0 = 0.0;
        Array2::from_shape_fn((grid.nz, grid.nr), |(iz, ir)| {
            let dr = grid.rr[[iz, ir]] - r0;
            let dz = grid.zz[[iz, ir]] - z0;
            -(-((dr * dr) / 0.02 + (dz * dz) / 0.04)).exp()
        })
    }

    fn rel_l2(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let num = (a - b).mapv(|v| v * v).sum().sqrt();
        let den = b.mapv(|v| v * v).sum().sqrt().max(1e-12);
        num / den
    }

    #[test]
    fn test_amr_kernel_runs_with_refinement() {
        let grid = Grid2D::new(33, 33, 1.0, 2.0, -1.0, 1.0);
        let source = gaussian_source(&grid);
        let solver = AmrKernelSolver::default();
        let (psi, hierarchy) = solver
            .solve_with_hierarchy(&grid, &source)
            .expect("valid AMR solve inputs");

        assert!(psi.iter().all(|v| v.is_finite()));
        assert!(
            !hierarchy.patches.is_empty(),
            "Expected at least one AMR patch for pedestal-weighted source"
        );
    }

    #[test]
    fn test_amr_kernel_no_refinement_matches_coarse() {
        let grid = Grid2D::new(33, 33, 1.0, 2.0, -1.0, 1.0);
        let source = gaussian_source(&grid);

        let coarse_cfg = AmrKernelConfig {
            refinement_threshold: f64::INFINITY,
            ..Default::default()
        };
        let solver = AmrKernelSolver::new(coarse_cfg.clone()).expect("valid AMR config");
        let (amr_off, hierarchy) = solver
            .solve_with_hierarchy(&grid, &source)
            .expect("valid AMR solve inputs");
        assert!(hierarchy.patches.is_empty());

        let mut coarse = Array2::zeros((grid.nz, grid.nr));
        sor_solve(
            &mut coarse,
            &source,
            &grid,
            coarse_cfg.omega,
            coarse_cfg.coarse_iters,
        );
        let err = rel_l2(&amr_off, &coarse);
        assert!(err < 1e-12, "No-refinement path should match coarse solve");
    }

    #[test]
    fn test_amr_kernel_multilevel_hierarchy_when_enabled() {
        let grid = Grid2D::new(33, 33, 1.0, 2.0, -1.0, 1.0);
        let source = gaussian_source(&grid);
        let solver = AmrKernelSolver::new(AmrKernelConfig {
            max_levels: 3,
            refinement_threshold: 0.05,
            ..Default::default()
        })
        .expect("valid AMR config");

        let (_psi, hierarchy) = solver
            .solve_with_hierarchy(&grid, &source)
            .expect("valid AMR solve inputs");
        assert!(
            hierarchy.patches.len() >= 2,
            "Expected multi-level AMR hierarchy for max_levels=3"
        );
        assert_eq!(
            hierarchy.patches.iter().map(|p| p.level).max().unwrap_or(0),
            2
        );
    }

    #[test]
    fn test_amr_kernel_rejects_invalid_constructor_config() {
        for bad_blend in [f64::NAN, -0.1, 1.1] {
            let err = AmrKernelSolver::new(AmrKernelConfig {
                blend: bad_blend,
                ..Default::default()
            })
            .expect_err("invalid blend must error");
            match err {
                FusionError::ConfigError(msg) => {
                    assert!(msg.contains("blend"));
                }
                other => panic!("Unexpected error: {other:?}"),
            }
        }
        let err = AmrKernelSolver::new(AmrKernelConfig {
            omega: 0.0,
            ..Default::default()
        })
        .expect_err("invalid omega must error");
        match err {
            FusionError::ConfigError(msg) => {
                assert!(msg.contains("omega"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }

        let err = AmrKernelSolver::new(AmrKernelConfig {
            max_levels: 0,
            ..Default::default()
        })
        .expect_err("invalid max_levels must error");
        match err {
            FusionError::ConfigError(msg) => {
                assert!(msg.contains("max_levels"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }

        for bad_threshold in [f64::NAN, -0.01, f64::NEG_INFINITY] {
            let err = AmrKernelSolver::new(AmrKernelConfig {
                refinement_threshold: bad_threshold,
                ..Default::default()
            })
            .expect_err("invalid refinement_threshold must error");
            match err {
                FusionError::ConfigError(msg) => {
                    assert!(msg.contains("refinement_threshold"));
                }
                other => panic!("Unexpected error: {other:?}"),
            }
        }
    }

    #[test]
    fn test_amr_kernel_rejects_invalid_source_shape_or_values() {
        let grid = Grid2D::new(17, 17, 1.0, 2.0, -1.0, 1.0);
        let solver = AmrKernelSolver::default();

        let bad_shape = Array2::zeros((16, 17));
        assert!(solver.solve_with_hierarchy(&grid, &bad_shape).is_err());

        let mut bad_values = Array2::zeros((17, 17));
        bad_values[[0, 0]] = f64::NAN;
        assert!(solver.solve_with_hierarchy(&grid, &bad_values).is_err());
    }
}
