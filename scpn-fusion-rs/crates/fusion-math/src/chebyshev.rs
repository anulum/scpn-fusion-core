// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Chebyshev-Accelerated SOR
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Adaptive Chebyshev-accelerated SOR solver.
//!
//! This module wraps red-black SOR with a scalar Chebyshev semi-iterative
//! schedule for the relaxation factor:
//!   omega_0 = 1
//!   omega_1 = 1 / (1 - rho_j^2 / 2)
//!   omega_{k+1} = 1 / (1 - rho_j^2 * omega_k / 4)
//! where rho_j is an estimate of the Jacobi spectral radius from warmup
//! residual ratios.

use crate::sor::{sor_residual, sor_step};
use fusion_types::state::Grid2D;
use ndarray::Array2;

const OMEGA_MIN: f64 = 1.0;
const OMEGA_MAX: f64 = 1.98;

#[derive(Debug, Clone, Copy)]
pub struct ChebyshevConfig {
    pub warmup_iters: usize,
    pub max_iters: usize,
    pub tol: f64,
}

impl Default for ChebyshevConfig {
    fn default() -> Self {
        Self {
            warmup_iters: 3,
            max_iters: 800,
            tol: 1e-8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SolveResult {
    pub converged: bool,
    pub iterations: usize,
    pub residual: f64,
    pub rho_jacobi: f64,
    pub final_omega: f64,
}

fn estimate_rho_jacobi(
    psi: &Array2<f64>,
    source: &Array2<f64>,
    grid: &Grid2D,
    warmup_iters: usize,
) -> (f64, f64) {
    // Use a temporary field so the caller can decide how to continue.
    let mut tmp = psi.clone();
    let warmup_iters = warmup_iters.max(2);

    let mut prev_res = sor_residual(&tmp, source, grid);
    let mut ratios = Vec::with_capacity(warmup_iters.saturating_sub(1));

    for _ in 0..warmup_iters {
        sor_step(&mut tmp, source, grid, 1.0);
        let res = sor_residual(&tmp, source, grid);
        if prev_res > 0.0 && res.is_finite() {
            ratios.push((res / prev_res).clamp(0.0, 0.999));
        }
        prev_res = res;
    }

    let rho = if ratios.is_empty() {
        0.9
    } else {
        let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
        mean.clamp(0.0, 0.999)
    };

    (rho, prev_res)
}

/// Solve with adaptive Chebyshev-SOR schedule.
pub fn chebyshev_sor_solve(
    psi: &mut Array2<f64>,
    source: &Array2<f64>,
    grid: &Grid2D,
    config: ChebyshevConfig,
) -> SolveResult {
    if config.max_iters == 0 {
        return SolveResult {
            converged: false,
            iterations: 0,
            residual: sor_residual(psi, source, grid),
            rho_jacobi: 0.0,
            final_omega: 1.0,
        };
    }

    let warmup = config.warmup_iters.max(2).min(config.max_iters);
    let (rho_j, _) = estimate_rho_jacobi(psi, source, grid, warmup);
    let rho_sq = rho_j * rho_j;

    // Apply warmup in-place (omega=1).
    let mut residual = sor_residual(psi, source, grid);
    let mut iterations = 0usize;
    for _ in 0..warmup {
        sor_step(psi, source, grid, 1.0);
        iterations += 1;
        residual = sor_residual(psi, source, grid);
        if residual <= config.tol {
            return SolveResult {
                converged: true,
                iterations,
                residual,
                rho_jacobi: rho_j,
                final_omega: 1.0,
            };
        }
    }

    // First accelerated step.
    let mut omega = (1.0 / (1.0 - rho_sq / 2.0)).clamp(OMEGA_MIN, OMEGA_MAX);

    while iterations < config.max_iters {
        sor_step(psi, source, grid, omega);
        iterations += 1;
        residual = sor_residual(psi, source, grid);
        if residual <= config.tol {
            return SolveResult {
                converged: true,
                iterations,
                residual,
                rho_jacobi: rho_j,
                final_omega: omega,
            };
        }

        let denom = (1.0 - rho_sq * omega / 4.0).max(1e-9);
        omega = (1.0 / denom).clamp(OMEGA_MIN, OMEGA_MAX);
    }

    SolveResult {
        converged: false,
        iterations,
        residual,
        rho_jacobi: rho_j,
        final_omega: omega,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sor::sor_step;

    fn fixed_sor_iters_to_tol(
        psi: &mut Array2<f64>,
        source: &Array2<f64>,
        grid: &Grid2D,
        omega: f64,
        tol: f64,
        max_iters: usize,
    ) -> (usize, f64) {
        let mut residual = sor_residual(psi, source, grid);
        for iter in 0..max_iters {
            sor_step(psi, source, grid, omega);
            residual = sor_residual(psi, source, grid);
            if residual <= tol {
                return (iter + 1, residual);
            }
        }
        (max_iters, residual)
    }

    #[test]
    fn test_chebyshev_converges_faster_than_fixed_sor() {
        let grid = Grid2D::new(129, 129, 1.0, 9.0, -5.0, 5.0);
        let source = Array2::from_elem((129, 129), -1.0);
        let tol = 2e-3;
        let max_iters = 900;

        let mut psi_fixed = Array2::zeros((129, 129));
        let (iters_fixed, res_fixed) =
            fixed_sor_iters_to_tol(&mut psi_fixed, &source, &grid, 1.8, tol, max_iters);

        let mut psi_cheb = Array2::zeros((129, 129));
        let result = chebyshev_sor_solve(
            &mut psi_cheb,
            &source,
            &grid,
            ChebyshevConfig {
                warmup_iters: 3,
                max_iters,
                tol,
            },
        );

        assert!(
            result.residual <= tol,
            "Chebyshev-SOR did not reach tolerance: {} > {}",
            result.residual,
            tol
        );
        assert!(
            result.iterations <= iters_fixed,
            "Expected Chebyshev-SOR to be at least as fast as fixed SOR: cheb={} fixed={} (fixed residual={})",
            result.iterations,
            iters_fixed,
            res_fixed
        );
    }

    #[test]
    fn test_chebyshev_solver_finite_and_positive_omega() {
        let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
        let source = Array2::from_elem((33, 33), -1.0);
        let mut psi = Array2::zeros((33, 33));
        let result = chebyshev_sor_solve(
            &mut psi,
            &source,
            &grid,
            ChebyshevConfig {
                warmup_iters: 3,
                max_iters: 200,
                tol: 1e-4,
            },
        );

        assert!(result.rho_jacobi.is_finite());
        assert!((0.0..1.0).contains(&result.rho_jacobi));
        assert!(result.final_omega.is_finite());
        assert!(result.final_omega >= OMEGA_MIN && result.final_omega <= OMEGA_MAX);
        assert!(result.residual.is_finite());
        assert!(psi.iter().all(|v| v.is_finite()));
    }
}
