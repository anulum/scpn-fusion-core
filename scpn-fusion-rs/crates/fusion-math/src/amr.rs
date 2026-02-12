// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Adaptive Mesh Refinement
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Patch-based AMR hierarchy (Berger-Oliger style, 2:1 refinement).

use fusion_types::state::Grid2D;
use ndarray::Array2;

#[derive(Debug, Clone)]
pub struct AmrPatch {
    pub grid: Grid2D,
    pub psi: Array2<f64>,
    pub level: usize,
    pub bounds: (usize, usize, usize, usize), // (iz_lo, iz_hi, ir_lo, ir_hi)
}

#[derive(Debug, Clone)]
pub struct AmrHierarchy {
    pub base: Grid2D,
    pub patches: Vec<AmrPatch>,
    pub max_levels: usize,
    pub refinement_threshold: f64,
}

impl AmrHierarchy {
    pub fn new(base: Grid2D, max_levels: usize, refinement_threshold: f64) -> Self {
        Self {
            base,
            patches: Vec::new(),
            max_levels: max_levels.max(1),
            refinement_threshold,
        }
    }

    /// Refine in pedestal-adjacent cells where |∇²J_phi| exceeds threshold.
    pub fn refine(&mut self, error_field: &Array2<f64>) {
        self.patches.clear();

        if self.max_levels < 2 || !self.refinement_threshold.is_finite() {
            return;
        }
        if error_field.dim() != (self.base.nz, self.base.nr) {
            return;
        }
        if self.base.nr < 5 || self.base.nz < 5 {
            return;
        }

        let r_span = (self.base.r[self.base.nr - 1] - self.base.r[0])
            .abs()
            .max(1e-12);
        let mut iz_lo = usize::MAX;
        let mut iz_hi = 0usize;
        let mut ir_lo = usize::MAX;
        let mut ir_hi = 0usize;

        for iz in 1..self.base.nz - 1 {
            for ir in 1..self.base.nr - 1 {
                let rho = (self.base.r[ir] - self.base.r[0]) / r_span;
                if rho < 0.9 {
                    continue;
                }
                if error_field[[iz, ir]] <= self.refinement_threshold {
                    continue;
                }
                iz_lo = iz_lo.min(iz);
                iz_hi = iz_hi.max(iz);
                ir_lo = ir_lo.min(ir);
                ir_hi = ir_hi.max(ir);
            }
        }

        if iz_lo == usize::MAX {
            return;
        }

        // Expand for stencil support.
        let pad = 1usize;
        iz_lo = iz_lo.saturating_sub(pad).max(1);
        ir_lo = ir_lo.saturating_sub(pad).max(1);
        iz_hi = (iz_hi + pad).min(self.base.nz - 2);
        ir_hi = (ir_hi + pad).min(self.base.nr - 2);

        if iz_hi <= iz_lo || ir_hi <= ir_lo {
            return;
        }

        let r_min = self.base.r[ir_lo];
        let r_max = self.base.r[ir_hi];
        let z_min = self.base.z[iz_lo];
        let z_max = self.base.z[iz_hi];
        let nr_patch = (ir_hi - ir_lo) * 2 + 1;
        let nz_patch = (iz_hi - iz_lo) * 2 + 1;
        let grid = Grid2D::new(nr_patch, nz_patch, r_min, r_max, z_min, z_max);
        let psi = Array2::zeros((nz_patch, nr_patch));

        self.patches.push(AmrPatch {
            grid,
            psi,
            level: 1,
            bounds: (iz_lo, iz_hi, ir_lo, ir_hi),
        });
    }

    pub fn coarsen(&mut self) {
        self.patches.clear();
    }

    /// Restrict refined patch values back to base-grid collocated points.
    pub fn interpolate_to_base(&self) -> Array2<f64> {
        let mut out = Array2::zeros((self.base.nz, self.base.nr));
        let mut hits: Array2<f64> = Array2::zeros((self.base.nz, self.base.nr));

        for patch in &self.patches {
            let (iz_lo, iz_hi, ir_lo, ir_hi) = patch.bounds;
            for iz in iz_lo..=iz_hi {
                for ir in ir_lo..=ir_hi {
                    let pz = (iz - iz_lo) * 2;
                    let pr = (ir - ir_lo) * 2;
                    out[[iz, ir]] += patch.psi[[pz, pr]];
                    hits[[iz, ir]] += 1.0;
                }
            }
        }

        for iz in 0..self.base.nz {
            for ir in 0..self.base.nr {
                if hits[[iz, ir]] > 0.0 {
                    out[[iz, ir]] /= hits[[iz, ir]];
                }
            }
        }
        out
    }
}

/// Error estimator |∇²(field)| on base grid.
pub fn estimate_error_field(field: &Array2<f64>, grid: &Grid2D) -> Array2<f64> {
    let mut err = Array2::zeros((grid.nz, grid.nr));
    let dr2 = (grid.dr * grid.dr).max(1e-12);
    let dz2 = (grid.dz * grid.dz).max(1e-12);

    for iz in 1..grid.nz - 1 {
        for ir in 1..grid.nr - 1 {
            let lap_r = (field[[iz, ir + 1]] - 2.0 * field[[iz, ir]] + field[[iz, ir - 1]]) / dr2;
            let lap_z = (field[[iz + 1, ir]] - 2.0 * field[[iz, ir]] + field[[iz - 1, ir]]) / dz2;
            err[[iz, ir]] = (lap_r + lap_z).abs();
        }
    }
    err
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sor::sor_solve;
    use ndarray::Array2;

    fn gaussian_source(grid: &Grid2D) -> Array2<f64> {
        let r0 = grid.r[grid.nr - 1] - 0.08 * (grid.r[grid.nr - 1] - grid.r[0]);
        let z0 = 0.0;
        Array2::from_shape_fn((grid.nz, grid.nr), |(iz, ir)| {
            let dr = grid.rr[[iz, ir]] - r0;
            let dz = grid.zz[[iz, ir]] - z0;
            -(-((dr * dr) / 0.02 + (dz * dz) / 0.04)).exp()
        })
    }

    fn solve_uniform(grid: &Grid2D, source: &Array2<f64>, iters: usize) -> Array2<f64> {
        let mut psi = Array2::zeros((grid.nz, grid.nr));
        sor_solve(&mut psi, source, grid, 1.8, iters);
        psi
    }

    fn apply_single_patch_amr(
        base_grid: &Grid2D,
        source: &Array2<f64>,
        threshold: f64,
    ) -> (Array2<f64>, AmrHierarchy) {
        let mut psi = solve_uniform(base_grid, source, 600);
        let error = estimate_error_field(source, base_grid);
        let mut hierarchy = AmrHierarchy::new(base_grid.clone(), 2, threshold);
        hierarchy.refine(&error);

        for patch in &mut hierarchy.patches {
            let (iz_lo, _iz_hi, ir_lo, _ir_hi) = patch.bounds;

            // Seed patch from collocated coarse values.
            for pz in 0..patch.grid.nz {
                for pr in 0..patch.grid.nr {
                    let base_iz = (iz_lo + pz / 2).min(base_grid.nz - 1);
                    let base_ir = (ir_lo + pr / 2).min(base_grid.nr - 1);
                    patch.psi[[pz, pr]] = psi[[base_iz, base_ir]];
                }
            }

            // Nearest-neighbor source prolongation from base to patch grid.
            let patch_source = Array2::from_shape_fn((patch.grid.nz, patch.grid.nr), |(pz, pr)| {
                let base_iz = (iz_lo + pz / 2).min(base_grid.nz - 1);
                let base_ir = (ir_lo + pr / 2).min(base_grid.nr - 1);
                source[[base_iz, base_ir]]
            });

            sor_solve(&mut patch.psi, &patch_source, &patch.grid, 1.8, 450);

            // Inject back (restricted collocation).
            for iz in patch.bounds.0..=patch.bounds.1 {
                for ir in patch.bounds.2..=patch.bounds.3 {
                    let pz = (iz - patch.bounds.0) * 2;
                    let pr = (ir - patch.bounds.2) * 2;
                    psi[[iz, ir]] = 0.5 * psi[[iz, ir]] + 0.5 * patch.psi[[pz, pr]];
                }
            }
        }

        (psi, hierarchy)
    }

    fn rel_l2(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let num = (a - b).mapv(|v| v * v).sum().sqrt();
        let den = b.mapv(|v| v * v).sum().sqrt().max(1e-12);
        num / den
    }

    fn downsample_2x(fine: &Array2<f64>, nz: usize, nr: usize) -> Array2<f64> {
        Array2::from_shape_fn((nz, nr), |(iz, ir)| fine[[iz * 2, ir * 2]])
    }

    #[test]
    fn test_amr_refines_pedestal_region() {
        let base = Grid2D::new(33, 33, 1.0, 2.0, -1.0, 1.0);
        let mut error = Array2::zeros((33, 33));
        for iz in 12..20 {
            for ir in 30..32 {
                error[[iz, ir]] = 10.0;
            }
        }

        let mut hierarchy = AmrHierarchy::new(base, 2, 1.0);
        hierarchy.refine(&error);
        assert!(
            !hierarchy.patches.is_empty(),
            "Expected AMR patch in pedestal zone"
        );

        let patch = &hierarchy.patches[0];
        let ir_mid = (patch.bounds.2 + patch.bounds.3) / 2;
        let rho = (hierarchy.base.r[ir_mid] - hierarchy.base.r[0])
            / (hierarchy.base.r[hierarchy.base.nr - 1] - hierarchy.base.r[0]);
        assert!(
            rho > 0.9,
            "Patch center rho should be in pedestal region: {rho}"
        );
    }

    #[test]
    fn test_amr_solution_matches_fine_grid() {
        let coarse_grid = Grid2D::new(33, 33, 1.0, 2.0, -1.0, 1.0);
        let fine_grid = Grid2D::new(65, 65, 1.0, 2.0, -1.0, 1.0);

        let coarse_source = gaussian_source(&coarse_grid);
        let fine_source = gaussian_source(&fine_grid);

        let (amr_solution, hierarchy) = apply_single_patch_amr(&coarse_grid, &coarse_source, 0.1);
        assert!(
            !hierarchy.patches.is_empty(),
            "Expected AMR to create a refinement patch"
        );

        let fine_solution = solve_uniform(&fine_grid, &fine_source, 1400);
        let fine_down = downsample_2x(&fine_solution, coarse_grid.nz, coarse_grid.nr);

        let err = rel_l2(&amr_solution, &fine_down);
        assert!(
            err < 0.01,
            "AMR solution should be within 1% of fine-grid reference, got {err:.4}"
        );
    }

    #[test]
    fn test_amr_no_refinement_matches_uniform() {
        let coarse_grid = Grid2D::new(33, 33, 1.0, 2.0, -1.0, 1.0);
        let source = gaussian_source(&coarse_grid);

        let coarse = solve_uniform(&coarse_grid, &source, 700);
        let (amr, hierarchy) = apply_single_patch_amr(&coarse_grid, &source, f64::INFINITY);
        assert!(
            hierarchy.patches.is_empty(),
            "No patches expected at inf threshold"
        );

        let err = rel_l2(&amr, &coarse);
        assert!(
            err < 1e-12,
            "No-refinement AMR should match uniform solve exactly, got {err:e}"
        );
    }
}
