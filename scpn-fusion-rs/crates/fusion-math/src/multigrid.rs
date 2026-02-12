// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Multigrid Solver
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Geometric multigrid V-cycle solver for the Grad-Shafranov equation.
//!
//! Implements a standard V-cycle with:
//! - **Restriction**: full-weighting (bilinear) from fine to coarse
//! - **Prolongation**: bilinear interpolation from coarse to fine
//! - **Smoother**: Red-Black SOR (from [`crate::sor`])
//!
//! The multigrid method achieves O(N) convergence versus O(N²) for plain SOR,
//! making it ~10-50× faster for large grids (256×256+).
//!
//! # Grid Size Requirements
//!
//! For correct coarsening, grid dimensions should satisfy: n = 2^k + 1
//! (e.g., 17, 33, 65, 129, 257). The solver handles other sizes but
//! may truncate to the nearest coarsenable dimension.

use fusion_types::state::Grid2D;
use ndarray::Array2;

use crate::sor::{sor_residual, sor_step};

/// Configuration for the multigrid V-cycle solver.
#[derive(Debug, Clone)]
pub struct MultigridConfig {
    /// Number of pre-smoothing SOR sweeps (default: 3)
    pub pre_smooth: usize,
    /// Number of post-smoothing SOR sweeps (default: 3)
    pub post_smooth: usize,
    /// SOR relaxation parameter (default: 1.8)
    pub omega: f64,
    /// Number of coarsest-level SOR sweeps (default: 50)
    pub coarse_iters: usize,
    /// Minimum grid dimension for coarsening (default: 5)
    pub min_grid_size: usize,
}

impl Default for MultigridConfig {
    fn default() -> Self {
        MultigridConfig {
            pre_smooth: 3,
            post_smooth: 3,
            omega: 1.8,
            coarse_iters: 50,
            min_grid_size: 5,
        }
    }
}

/// Result of a multigrid solve.
#[derive(Debug, Clone)]
pub struct MultigridResult {
    /// Number of V-cycles performed.
    pub cycles: usize,
    /// Final L-infinity residual.
    pub residual: f64,
    /// Whether convergence was achieved.
    pub converged: bool,
}

/// Full-weighting restriction: fine grid → coarse grid.
///
/// Standard 2D stencil:
///   1/16 [1 2 1; 2 4 2; 1 2 1]
///
/// Maps (2N-1) × (2M-1) → N × M.
fn restrict(fine: &Array2<f64>, coarse: &mut Array2<f64>) {
    let (cnz, cnr) = coarse.dim();

    for iz in 0..cnz {
        for ir in 0..cnr {
            let fiz = 2 * iz;
            let fir = 2 * ir;

            if iz == 0 || iz == cnz - 1 || ir == 0 || ir == cnr - 1 {
                // Boundary: direct injection
                coarse[[iz, ir]] = fine[[fiz, fir]];
            } else {
                // Interior: full-weighting stencil
                coarse[[iz, ir]] = (4.0 * fine[[fiz, fir]]
                    + 2.0
                        * (fine[[fiz - 1, fir]]
                            + fine[[fiz + 1, fir]]
                            + fine[[fiz, fir - 1]]
                            + fine[[fiz, fir + 1]])
                    + fine[[fiz - 1, fir - 1]]
                    + fine[[fiz - 1, fir + 1]]
                    + fine[[fiz + 1, fir - 1]]
                    + fine[[fiz + 1, fir + 1]])
                    / 16.0;
            }
        }
    }
}

/// Bilinear prolongation: coarse grid → fine grid.
///
/// Maps N × M → (2N-1) × (2M-1).
fn prolongate(coarse: &Array2<f64>, fine: &mut Array2<f64>) {
    let (cnz, cnr) = coarse.dim();

    for iz in 0..cnz {
        for ir in 0..cnr {
            let fiz = 2 * iz;
            let fir = 2 * ir;
            // Coincident points: direct copy
            fine[[fiz, fir]] += coarse[[iz, ir]];
        }
    }

    // Horizontal midpoints
    for iz in 0..cnz {
        for ir in 0..cnr - 1 {
            let fiz = 2 * iz;
            let fir = 2 * ir + 1;
            fine[[fiz, fir]] += 0.5 * (coarse[[iz, ir]] + coarse[[iz, ir + 1]]);
        }
    }

    // Vertical midpoints
    for iz in 0..cnz - 1 {
        for ir in 0..cnr {
            let fiz = 2 * iz + 1;
            let fir = 2 * ir;
            fine[[fiz, fir]] += 0.5 * (coarse[[iz, ir]] + coarse[[iz + 1, ir]]);
        }
    }

    // Center midpoints
    for iz in 0..cnz - 1 {
        for ir in 0..cnr - 1 {
            let fiz = 2 * iz + 1;
            let fir = 2 * ir + 1;
            fine[[fiz, fir]] += 0.25
                * (coarse[[iz, ir]]
                    + coarse[[iz, ir + 1]]
                    + coarse[[iz + 1, ir]]
                    + coarse[[iz + 1, ir + 1]]);
        }
    }
}

/// Compute the residual vector r = source - L[psi] on the given grid.
///
/// The operator L is the Grad-Shafranov 5-point stencil with toroidal 1/R terms.
fn compute_residual_vector(psi: &Array2<f64>, source: &Array2<f64>, grid: &Grid2D) -> Array2<f64> {
    let (nz, nr) = psi.dim();
    let dr = grid.dr;
    let dz = grid.dz;
    let dr_sq = dr * dr;
    let dz_sq = dz * dz;

    let mut residual = Array2::zeros((nz, nr));

    for iz in 1..nz - 1 {
        for ir in 1..nr - 1 {
            let r = grid.rr[[iz, ir]];

            let c_r_plus = 1.0 / dr_sq - 1.0 / (2.0 * r * dr);
            let c_r_minus = 1.0 / dr_sq + 1.0 / (2.0 * r * dr);
            let c_z = 1.0 / dz_sq;
            let center = 2.0 / dr_sq + 2.0 / dz_sq;

            let lhs = center * psi[[iz, ir]]
                - c_z * (psi[[iz + 1, ir]] + psi[[iz - 1, ir]])
                - c_r_plus * psi[[iz, ir + 1]]
                - c_r_minus * psi[[iz, ir - 1]];

            residual[[iz, ir]] = source[[iz, ir]] - lhs;
        }
    }

    residual
}

/// Compute coarse grid dimensions from fine grid.
fn coarse_size(n: usize) -> usize {
    n.div_ceil(2)
}

/// Build a coarsened Grid2D from a fine Grid2D.
fn coarsen_grid(fine_grid: &Grid2D) -> Grid2D {
    let cnr = coarse_size(fine_grid.nr);
    let cnz = coarse_size(fine_grid.nz);
    let r_min = fine_grid.r[0];
    let r_max = fine_grid.r[fine_grid.nr - 1];
    let z_min = fine_grid.z[0];
    let z_max = fine_grid.z[fine_grid.nz - 1];
    Grid2D::new(cnr, cnz, r_min, r_max, z_min, z_max)
}

/// Perform one multigrid V-cycle.
///
/// Recursive: smooths on the current level, restricts the residual to
/// a coarser grid, solves the coarse correction, prolongs it back, and
/// post-smooths.
fn v_cycle(psi: &mut Array2<f64>, source: &Array2<f64>, grid: &Grid2D, config: &MultigridConfig) {
    let (nz, nr) = psi.dim();

    // Base case: grid too small for further coarsening — solve directly
    if nr <= config.min_grid_size || nz <= config.min_grid_size {
        for _ in 0..config.coarse_iters {
            sor_step(psi, source, grid, config.omega);
        }
        return;
    }

    // 1. Pre-smoothing
    for _ in 0..config.pre_smooth {
        sor_step(psi, source, grid, config.omega);
    }

    // 2. Compute residual on fine grid
    let residual_fine = compute_residual_vector(psi, source, grid);

    // 3. Restrict residual to coarse grid
    let coarse_grid = coarsen_grid(grid);
    let cnz = coarse_grid.nz;
    let cnr = coarse_grid.nr;
    let mut residual_coarse = Array2::zeros((cnz, cnr));
    restrict(&residual_fine, &mut residual_coarse);

    // 4. Solve correction on coarse grid (e = 0 initially)
    let mut correction_coarse = Array2::zeros((cnz, cnr));
    v_cycle(
        &mut correction_coarse,
        &residual_coarse,
        &coarse_grid,
        config,
    );

    // 5. Prolongate correction to fine grid and add
    let mut correction_fine = Array2::zeros((nz, nr));
    prolongate(&correction_coarse, &mut correction_fine);

    // Add correction to solution
    for iz in 1..nz - 1 {
        for ir in 1..nr - 1 {
            psi[[iz, ir]] += correction_fine[[iz, ir]];
        }
    }

    // 6. Post-smoothing
    for _ in 0..config.post_smooth {
        sor_step(psi, source, grid, config.omega);
    }
}

/// Solve the Grad-Shafranov equation using multigrid V-cycles.
///
/// # Arguments
/// * `psi` — initial guess / solution array [nz, nr] (modified in place)
/// * `source` — source term [nz, nr]
/// * `grid` — computational grid
/// * `config` — multigrid parameters
/// * `max_cycles` — maximum number of V-cycles
/// * `tol` — convergence tolerance on L-infinity residual
///
/// # Returns
/// A [`MultigridResult`] with convergence information.
///
/// # Example
/// ```
/// use fusion_types::state::Grid2D;
/// use fusion_math::multigrid::{multigrid_solve, MultigridConfig};
/// use ndarray::Array2;
///
/// let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
/// let mut psi = Array2::zeros((33, 33));
/// let source = Array2::from_elem((33, 33), -1.0);
///
/// let result = multigrid_solve(
///     &mut psi, &source, &grid,
///     &MultigridConfig::default(), 20, 1e-8,
/// );
/// assert!(result.converged);
/// ```
pub fn multigrid_solve(
    psi: &mut Array2<f64>,
    source: &Array2<f64>,
    grid: &Grid2D,
    config: &MultigridConfig,
    max_cycles: usize,
    tol: f64,
) -> MultigridResult {
    let mut residual = sor_residual(psi, source, grid);

    for cycle in 1..=max_cycles {
        v_cycle(psi, source, grid, config);
        residual = sor_residual(psi, source, grid);

        if residual < tol {
            return MultigridResult {
                cycles: cycle,
                residual,
                converged: true,
            };
        }
    }

    MultigridResult {
        cycles: max_cycles,
        residual,
        converged: residual < tol,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multigrid_convergence_33() {
        let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
        let mut psi = Array2::zeros((33, 33));
        let source = Array2::from_elem((33, 33), -1.0);

        let result = multigrid_solve(
            &mut psi,
            &source,
            &grid,
            &MultigridConfig::default(),
            30,
            1e-8,
        );

        assert!(
            result.converged,
            "Should converge: residual = {}, cycles = {}",
            result.residual, result.cycles
        );
        assert!(
            result.cycles < 20,
            "Should converge in fewer than 20 cycles"
        );
    }

    #[test]
    fn test_multigrid_convergence_65() {
        let grid = Grid2D::new(65, 65, 1.0, 9.0, -5.0, 5.0);
        let mut psi = Array2::zeros((65, 65));
        let source = Array2::from_elem((65, 65), -1.0);

        let result = multigrid_solve(
            &mut psi,
            &source,
            &grid,
            &MultigridConfig::default(),
            30,
            1e-6,
        );

        assert!(
            result.converged,
            "65x65 should converge: residual = {}",
            result.residual
        );
    }

    #[test]
    fn test_multigrid_faster_than_sor() {
        // Multigrid with 10 cycles should achieve lower residual
        // than plain SOR with equivalent work
        let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
        let source = Array2::from_elem((33, 33), -1.0);

        // SOR: 200 iterations
        let mut psi_sor = Array2::zeros((33, 33));
        crate::sor::sor_solve(&mut psi_sor, &source, &grid, 1.8, 200);
        let res_sor = sor_residual(&psi_sor, &source, &grid);

        // Multigrid: 10 cycles (each ~ 20 SOR sweeps of work across levels)
        let mut psi_mg = Array2::zeros((33, 33));
        let result = multigrid_solve(
            &mut psi_mg,
            &source,
            &grid,
            &MultigridConfig::default(),
            10,
            1e-15,
        );

        assert!(
            result.residual < res_sor,
            "Multigrid (res={}) should beat SOR 200 iters (res={})",
            result.residual,
            res_sor
        );
    }

    #[test]
    fn test_multigrid_zero_source() {
        let grid = Grid2D::new(17, 17, 1.0, 9.0, -5.0, 5.0);
        let mut psi = Array2::zeros((17, 17));
        let source = Array2::zeros((17, 17));

        let result = multigrid_solve(
            &mut psi,
            &source,
            &grid,
            &MultigridConfig::default(),
            5,
            1e-12,
        );

        let max_val: f64 = psi.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
        assert!(
            max_val < 1e-14,
            "Zero source should give zero solution: max = {}",
            max_val
        );
        assert!(result.converged, "Zero source should converge immediately");
    }

    #[test]
    fn test_multigrid_boundary_preserved() {
        let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
        let mut psi = Array2::zeros((33, 33));
        let source = Array2::from_elem((33, 33), -1.0);

        multigrid_solve(
            &mut psi,
            &source,
            &grid,
            &MultigridConfig::default(),
            10,
            1e-8,
        );

        for ir in 0..33 {
            assert!(
                psi[[0, ir]].abs() < 1e-15,
                "Top boundary modified: {}",
                psi[[0, ir]]
            );
            assert!(
                psi[[32, ir]].abs() < 1e-15,
                "Bottom boundary modified: {}",
                psi[[32, ir]]
            );
        }
        for iz in 0..33 {
            assert!(
                psi[[iz, 0]].abs() < 1e-15,
                "Left boundary modified: {}",
                psi[[iz, 0]]
            );
            assert!(
                psi[[iz, 32]].abs() < 1e-15,
                "Right boundary modified: {}",
                psi[[iz, 32]]
            );
        }
    }

    #[test]
    fn test_restrict_constant() {
        // Restricting a constant field should give the same constant
        let fine = Array2::from_elem((9, 9), 7.0);
        let mut coarse = Array2::zeros((5, 5));
        restrict(&fine, &mut coarse);

        for &v in coarse.iter() {
            assert!(
                (v - 7.0).abs() < 1e-14,
                "Constant restriction failed: {}",
                v
            );
        }
    }

    #[test]
    fn test_prolongate_adds_to_existing() {
        // Prolongation adds to the fine grid (does not overwrite)
        let coarse = Array2::from_elem((3, 3), 1.0);
        let mut fine = Array2::from_elem((5, 5), 10.0);
        prolongate(&coarse, &mut fine);

        // All fine grid values should be > 10.0 (original + prolongated)
        for &v in fine.iter() {
            assert!(v > 10.0 - 1e-14, "Prolongation should add: {}", v);
        }
    }
}
