//! X-point (magnetic null) detection.
//!
//! Port of fusion_kernel.py `find_x_point()` (lines 95-116).
//! Locates the saddle point where B=0 in the divertor region.

use fusion_math::interp::gradient_2d;
use fusion_types::state::Grid2D;
use ndarray::Array2;

/// Find the X-point (magnetic null) in the lower divertor region.
///
/// Algorithm:
/// 1. Compute gradient magnitude |∇Ψ| using central differences
/// 2. Mask upper half (only search Z < Z_min * 0.5)
/// 3. Find argmin of |∇Ψ| in masked region
///
/// Returns `((R, Z), psi_value)` at the X-point.
///
/// NOTE: The Python code calls `np.gradient(Psi, self.dR, self.dZ)` which returns
/// gradients along axis 0 (with spacing dR) and axis 1 (with spacing dZ).
/// The Python variable names are swapped (dPsi_dR is actually along Z-axis,
/// dPsi_dZ along R-axis), but the gradient magnitude is the same either way.
/// In Rust we use correct naming via gradient_2d().
pub fn find_x_point(psi: &Array2<f64>, grid: &Grid2D, z_min: f64) -> ((f64, f64), f64) {
    // Compute gradient components (correct naming)
    let (dpsi_dz, dpsi_dr) = gradient_2d(psi, grid);

    let nz = grid.nz;
    let nr = grid.nr;

    // Gradient magnitude |∇Ψ|
    // Note: In Python the variable names are swapped but |∇Ψ| = sqrt(a² + b²) is the same
    let mut min_b_mag = f64::MAX;
    let mut best_iz = 0;
    let mut best_ir = 0;
    let mut found = false;

    let z_threshold = z_min * 0.5;

    for iz in 0..nz {
        for ir in 0..nr {
            let z = grid.zz[[iz, ir]];

            // Only search in divertor region (Z < Z_min * 0.5)
            if z < z_threshold {
                let b_mag = (dpsi_dz[[iz, ir]].powi(2) + dpsi_dr[[iz, ir]].powi(2)).sqrt();

                if b_mag < min_b_mag {
                    min_b_mag = b_mag;
                    best_iz = iz;
                    best_ir = ir;
                    found = true;
                }
            }
        }
    }

    if found {
        let r = grid.r[best_ir];
        let z = grid.z[best_iz];
        ((r, z), psi[[best_iz, best_ir]])
    } else {
        // Fallback: return minimum psi location
        let mut min_psi = f64::MAX;
        let mut min_iz = 0;
        let mut min_ir = 0;
        for iz in 0..nz {
            for ir in 0..nr {
                if psi[[iz, ir]] < min_psi {
                    min_psi = psi[[iz, ir]];
                    min_iz = iz;
                    min_ir = ir;
                }
            }
        }
        ((grid.r[min_ir], grid.z[min_iz]), min_psi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_x_point_basic() {
        let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
        // Create a simple field with a null at (5.0, -2.5)
        let psi = Array2::from_shape_fn((33, 33), |(iz, ir)| {
            let r = grid.rr[[iz, ir]];
            let z = grid.zz[[iz, ir]];
            // Saddle point-like field
            (r - 5.0).powi(2) - (z + 2.5).powi(2)
        });

        let ((r_x, z_x), _psi_x) = find_x_point(&psi, &grid, -5.0);

        // X-point should be near (5.0, -2.5) — the saddle point
        assert!(
            (r_x - 5.0).abs() < 1.0,
            "X-point R={r_x}, expected near 5.0"
        );
        assert!(
            (z_x + 2.5).abs() < 1.0,
            "X-point Z={z_x}, expected near -2.5"
        );
    }
}
