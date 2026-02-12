//! Magnetic field computation from flux function.
//!
//! Port of fusion_kernel.py `compute_b_field()` (lines 305-309).
//! In cylindrical tokamak coordinates:
//!   B_R = -(1/R) ∂Ψ/∂Z
//!   B_Z =  (1/R) ∂Ψ/∂R

use fusion_math::interp::gradient_2d;
use fusion_types::state::Grid2D;
use ndarray::Array2;

/// Minimum R value to avoid division by zero at the magnetic axis.
const R_SAFE_MIN: f64 = 1e-6;

/// Compute the poloidal magnetic field components from the flux function Ψ.
///
/// In tokamak cylindrical coordinates:
/// - B_R = -(1/R) ∂Ψ/∂Z
/// - B_Z =  (1/R) ∂Ψ/∂R
///
/// Returns `(B_R, B_Z)` as `[nz, nr]` arrays.
///
/// NOTE: The Python code has swapped variable names (`dPsi_dR` is actually gradient
/// along axis 0 = Z direction). The physics usage is correct because both assignments
/// swap consistently. This Rust port uses correct naming from the start.
pub fn compute_b_field(psi: &Array2<f64>, grid: &Grid2D) -> (Array2<f64>, Array2<f64>) {
    let (dpsi_dz, dpsi_dr) = gradient_2d(psi, grid);

    let nz = grid.nz;
    let nr = grid.nr;
    let mut b_r = Array2::zeros((nz, nr));
    let mut b_z = Array2::zeros((nz, nr));

    for iz in 0..nz {
        for ir in 0..nr {
            let r_safe = grid.rr[[iz, ir]].max(R_SAFE_MIN);
            b_r[[iz, ir]] = -(1.0 / r_safe) * dpsi_dz[[iz, ir]];
            b_z[[iz, ir]] = (1.0 / r_safe) * dpsi_dr[[iz, ir]];
        }
    }

    (b_r, b_z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_b_field_uniform_psi() {
        // Uniform Ψ → zero B-field
        let grid = Grid2D::new(16, 16, 1.0, 9.0, -5.0, 5.0);
        let psi = Array2::from_elem((16, 16), 1.0);
        let (b_r, b_z) = compute_b_field(&psi, &grid);

        for iz in 0..16 {
            for ir in 0..16 {
                assert!(
                    b_r[[iz, ir]].abs() < 1e-10,
                    "B_R should be zero for uniform Ψ"
                );
                assert!(
                    b_z[[iz, ir]].abs() < 1e-10,
                    "B_Z should be zero for uniform Ψ"
                );
            }
        }
    }

    #[test]
    fn test_b_field_no_nan() {
        let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
        // Gaussian-like flux
        let psi = Array2::from_shape_fn((33, 33), |(iz, ir)| {
            let r = grid.rr[[iz, ir]];
            let z = grid.zz[[iz, ir]];
            (-(((r - 5.0).powi(2) + z.powi(2)) / 4.0)).exp()
        });
        let (b_r, b_z) = compute_b_field(&psi, &grid);
        assert!(!b_r.iter().any(|v| v.is_nan()), "B_R contains NaN");
        assert!(!b_z.iter().any(|v| v.is_nan()), "B_Z contains NaN");
    }
}
