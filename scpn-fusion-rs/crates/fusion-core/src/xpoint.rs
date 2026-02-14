// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Xpoint
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! X-point (magnetic null) detection.
//!
//! Port of fusion_kernel.py `find_x_point()` (lines 95-116).
//! Locates the saddle point where B=0 in the divertor region.

use fusion_math::interp::gradient_2d;
use fusion_types::error::{FusionError, FusionResult};
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
pub fn find_x_point(
    psi: &Array2<f64>,
    grid: &Grid2D,
    z_min: f64,
) -> FusionResult<((f64, f64), f64)> {
    if !z_min.is_finite() {
        return Err(FusionError::ConfigError(format!(
            "x-point z_min must be finite, got {z_min}"
        )));
    }
    if grid.nz < 2 || grid.nr < 2 {
        return Err(FusionError::ConfigError(format!(
            "x-point grid requires nz,nr >= 2, got nz={} nr={}",
            grid.nz, grid.nr
        )));
    }
    if psi.nrows() != grid.nz || psi.ncols() != grid.nr {
        return Err(FusionError::ConfigError(format!(
            "x-point psi shape mismatch: expected ({}, {}), got ({}, {})",
            grid.nz,
            grid.nr,
            psi.nrows(),
            psi.ncols()
        )));
    }
    if psi.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::ConfigError(
            "x-point psi contains non-finite values".to_string(),
        ));
    }
    if grid.z.iter().any(|v| !v.is_finite()) || grid.r.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::ConfigError(
            "x-point grid axes must be finite".to_string(),
        ));
    }

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
                let b_mag_sq = dpsi_dz[[iz, ir]].powi(2) + dpsi_dr[[iz, ir]].powi(2);
                if !b_mag_sq.is_finite() || b_mag_sq < 0.0 {
                    return Err(FusionError::ConfigError(format!(
                        "x-point gradient magnitude became invalid at ({iz}, {ir})"
                    )));
                }
                let b_mag = b_mag_sq.sqrt();

                if b_mag < min_b_mag {
                    min_b_mag = b_mag;
                    best_iz = iz;
                    best_ir = ir;
                    found = true;
                }
            }
        }
    }

    if !found {
        return Err(FusionError::ConfigError(
            "x-point search region is empty for provided z_min threshold".to_string(),
        ));
    }

    let r = grid.r[best_ir];
    let z = grid.z[best_iz];
    let psi_x = psi[[best_iz, best_ir]];
    if !r.is_finite() || !z.is_finite() || !psi_x.is_finite() {
        return Err(FusionError::ConfigError(
            "x-point output contains non-finite values".to_string(),
        ));
    }
    Ok(((r, z), psi_x))
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

        let ((r_x, z_x), _psi_x) = find_x_point(&psi, &grid, -5.0).expect("valid x-point inputs");

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

    #[test]
    fn test_find_x_point_rejects_invalid_runtime_inputs() {
        let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
        let psi = Array2::from_shape_fn((33, 33), |(iz, ir)| {
            let r = grid.rr[[iz, ir]];
            let z = grid.zz[[iz, ir]];
            (r - 5.0).powi(2) - (z + 2.5).powi(2)
        });

        let err = find_x_point(&psi, &grid, f64::NAN).expect_err("non-finite z_min must fail");
        assert!(matches!(err, FusionError::ConfigError(_)));

        let err =
            find_x_point(&psi, &grid, -1000.0).expect_err("empty divertor search region must fail");
        assert!(matches!(err, FusionError::ConfigError(_)));
    }
}
