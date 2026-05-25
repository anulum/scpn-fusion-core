// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — SOR
//! Red-Black Successive Over-Relaxation (SOR) solver.
//!
//! Direct port of solver.cpp. Uses the 5-point stencil with toroidal 1/R
//! corrections for the Grad-Shafranov equation in (R, Z) coordinates.
//!
//! The Grad-Shafranov operator in cylindrical coordinates is:
//!   R d/dR(1/R dPsi/dR) + d²Psi/dZ² = source
//!
//! Discretized with the stencil from solver.cpp lines 70-86.

use fusion_types::state::Grid2D;
use ndarray::Array2;

fn validate_sor_omega(omega: f64) {
    assert!(
        omega.is_finite() && (1.0..2.0).contains(&omega),
        "omega must be finite and satisfy 1.0 <= omega < 2.0"
    );
}

/// Perform one Red-Black SOR iteration.
///
/// `psi`: mutable [nz, nr] flux array
/// `source`: [nz, nr] source term (= -mu0 * R * J_phi, precomputed)
/// `grid`: computational grid
/// `omega`: SOR relaxation factor (1.0 = Gauss-Seidel, 1.8-1.9 = over-relaxation)
///
/// Boundary rows/columns (first/last row and column) are NOT updated.
pub fn sor_step(psi: &mut Array2<f64>, source: &Array2<f64>, grid: &Grid2D, omega: f64) {
    validate_sor_omega(omega);
    let nz = grid.nz;
    let nr = grid.nr;
    let dr = grid.dr;
    let dz = grid.dz;
    let dr_sq = dr * dr;
    let dz_sq = dz * dz;

    // Red pass: (iz + ir) % 2 == 0
    for iz in 1..nz - 1 {
        for ir in 1..nr - 1 {
            if (iz + ir) % 2 == 0 {
                update_point(psi, source, grid, iz, ir, dr_sq, dz_sq, dr, omega);
            }
        }
    }

    // Black pass: (iz + ir) % 2 != 0
    for iz in 1..nz - 1 {
        for ir in 1..nr - 1 {
            if (iz + ir) % 2 != 0 {
                update_point(psi, source, grid, iz, ir, dr_sq, dz_sq, dr, omega);
            }
        }
    }
}

/// Run N SOR iterations.
pub fn sor_solve(
    psi: &mut Array2<f64>,
    source: &Array2<f64>,
    grid: &Grid2D,
    omega: f64,
    iterations: usize,
) {
    validate_sor_omega(omega);
    for _ in 0..iterations {
        sor_step(psi, source, grid, omega);
    }
}

/// Compute the L-infinity residual (max absolute change from one iteration).
pub fn sor_residual(psi: &Array2<f64>, source: &Array2<f64>, grid: &Grid2D) -> f64 {
    let nz = grid.nz;
    let nr = grid.nr;
    let dr = grid.dr;
    let dz = grid.dz;
    let dr_sq = dr * dr;
    let dz_sq = dz * dz;

    let mut max_res: f64 = 0.0;

    for iz in 1..nz - 1 {
        for ir in 1..nr - 1 {
            let r = grid.rr[[iz, ir]];

            let c_r_plus = 1.0 / dr_sq - 1.0 / (2.0 * r * dr);
            let c_r_minus = 1.0 / dr_sq + 1.0 / (2.0 * r * dr);
            let c_z = 1.0 / dz_sq;
            let center = 2.0 / dr_sq + 2.0 / dz_sq;

            let p_up = psi[[iz + 1, ir]];
            let p_down = psi[[iz - 1, ir]];
            let p_right = psi[[iz, ir + 1]];
            let p_left = psi[[iz, ir - 1]];

            let lhs = c_z * (p_up + p_down) + c_r_plus * p_right + c_r_minus * p_left
                - center * psi[[iz, ir]];

            let residual = (lhs - source[[iz, ir]]).abs();
            max_res = max_res.max(residual);
        }
    }

    max_res
}

/// Update a single grid point using the SOR stencil.
///
/// Exact port of solver.cpp update_point() lines 64-86.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn update_point(
    psi: &mut Array2<f64>,
    source: &Array2<f64>,
    grid: &Grid2D,
    iz: usize,
    ir: usize,
    dr_sq: f64,
    dz_sq: f64,
    dr: f64,
    omega: f64,
) {
    let r = grid.rr[[iz, ir]];

    // Elliptic operator stencil (5-point) with 1/R toroidal correction
    let c_r_plus = 1.0 / dr_sq - 1.0 / (2.0 * r * dr);
    let c_r_minus = 1.0 / dr_sq + 1.0 / (2.0 * r * dr);
    let c_z = 1.0 / dz_sq;
    let center = 2.0 / dr_sq + 2.0 / dz_sq;

    // Neighbors
    let p_up = psi[[iz + 1, ir]];
    let p_down = psi[[iz - 1, ir]];
    let p_right = psi[[iz, ir + 1]];
    let p_left = psi[[iz, ir - 1]];

    // Gauss-Seidel prediction
    let p_star = (c_z * (p_up + p_down) + c_r_plus * p_right + c_r_minus * p_left
        - source[[iz, ir]])
        / center;

    // SOR update
    psi[[iz, ir]] = (1.0 - omega) * psi[[iz, ir]] + omega * p_star;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_sor_convergence_poisson() {
        // Solve with the Grad-Shafranov stencil (includes toroidal 1/R terms)
        // Using a tokamak grid (R in [1,9])
        let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
        let mut psi = Array2::zeros((33, 33));
        let source = Array2::from_elem((33, 33), -1.0);

        sor_solve(&mut psi, &source, &grid, 1.8, 500);

        // Center value should be non-zero (solver actually does something)
        assert!(
            psi[[16, 16]].abs() > 1e-10,
            "Center should be non-zero after 500 iterations"
        );
        // Should not contain NaN
        assert!(!psi.iter().any(|v| v.is_nan()), "No NaN allowed");
    }

    #[test]
    fn test_sor_zero_source_stays_zero() {
        let grid = Grid2D::new(16, 16, 1.0, 9.0, -5.0, 5.0);
        let mut psi = Array2::zeros((16, 16));
        let source = Array2::zeros((16, 16));

        sor_solve(&mut psi, &source, &grid, 1.8, 100);

        // With zero source and zero boundary, solution should remain zero
        let max_val = psi.iter().cloned().fold(0.0_f64, f64::max);
        assert!(max_val.abs() < 1e-15, "Should stay zero with zero source");
    }

    #[test]
    fn test_sor_residual_decreases() {
        let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
        let mut psi = Array2::zeros((33, 33));
        let source = Array2::from_elem((33, 33), -1.0);

        let res_before = sor_residual(&psi, &source, &grid);
        sor_solve(&mut psi, &source, &grid, 1.5, 200);
        let res_after = sor_residual(&psi, &source, &grid);

        assert!(
            res_after < res_before,
            "Residual should decrease: {res_before} -> {res_after}"
        );
    }

    #[test]
    fn test_sor_boundary_preserved() {
        let grid = Grid2D::new(16, 16, 1.0, 9.0, -5.0, 5.0);
        let mut psi = Array2::zeros((16, 16));
        let source = Array2::from_elem((16, 16), -1.0);

        sor_solve(&mut psi, &source, &grid, 1.8, 100);

        // Boundary (first/last row/col) should remain zero
        for ir in 0..16 {
            assert!(psi[[0, ir]].abs() < 1e-15, "Top boundary should be 0");
            assert!(psi[[15, ir]].abs() < 1e-15, "Bottom boundary should be 0");
        }
        for iz in 0..16 {
            assert!(psi[[iz, 0]].abs() < 1e-15, "Left boundary should be 0");
            assert!(psi[[iz, 15]].abs() < 1e-15, "Right boundary should be 0");
        }
    }

    #[test]
    #[should_panic(expected = "omega must be finite and satisfy 1.0 <= omega < 2.0")]
    fn test_sor_step_rejects_unstable_omega() {
        let grid = Grid2D::new(8, 8, 1.0, 9.0, -5.0, 5.0);
        let mut psi = Array2::zeros((8, 8));
        let source = Array2::zeros((8, 8));

        sor_step(&mut psi, &source, &grid, 2.0);
    }

    #[test]
    #[should_panic(expected = "omega must be finite and satisfy 1.0 <= omega < 2.0")]
    fn test_sor_solve_rejects_nonfinite_omega() {
        let grid = Grid2D::new(8, 8, 1.0, 9.0, -5.0, 5.0);
        let mut psi = Array2::zeros((8, 8));
        let source = Array2::zeros((8, 8));

        sor_solve(&mut psi, &source, &grid, f64::NAN, 1);
    }

    fn manufactured_flux(grid: &Grid2D) -> Array2<f64> {
        let mut psi = Array2::zeros((grid.nz, grid.nr));
        for iz in 0..grid.nz {
            for ir in 0..grid.nr {
                let r = grid.rr[[iz, ir]];
                let z = grid.zz[[iz, ir]];
                psi[[iz, ir]] =
                    0.03125 * r.powi(4) - 0.125 * z.powi(2) + 0.05 * r.powi(2) * z.powi(2);
            }
        }
        psi
    }

    fn discrete_gs_source(psi: &Array2<f64>, grid: &Grid2D) -> Array2<f64> {
        let mut source = Array2::zeros((grid.nz, grid.nr));
        let dr_sq = grid.dr * grid.dr;
        let dz_sq = grid.dz * grid.dz;
        for iz in 1..grid.nz - 1 {
            for ir in 1..grid.nr - 1 {
                let r = grid.rr[[iz, ir]];
                let d2r = (psi[[iz, ir + 1]] - 2.0 * psi[[iz, ir]] + psi[[iz, ir - 1]]) / dr_sq;
                let d1r = (psi[[iz, ir + 1]] - psi[[iz, ir - 1]]) / (2.0 * grid.dr);
                let d2z = (psi[[iz + 1, ir]] - 2.0 * psi[[iz, ir]] + psi[[iz - 1, ir]]) / dz_sq;
                source[[iz, ir]] = d2r - d1r / r + d2z;
            }
        }
        source
    }

    #[test]
    fn test_sor_step_preserves_exact_discrete_grad_shafranov_fixed_point() {
        let grid = Grid2D::new(9, 9, 1.0, 2.0, -0.5, 0.5);
        let expected = manufactured_flux(&grid);
        let mut psi = expected.clone();
        let source = discrete_gs_source(&expected, &grid);

        sor_step(&mut psi, &source, &grid, 1.4);

        let max_error = psi
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_error < 1e-12,
            "exact GS fixed point changed by {max_error}"
        );
    }
}
