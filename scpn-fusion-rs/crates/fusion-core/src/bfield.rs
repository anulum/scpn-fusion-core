// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Bfield
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Magnetic field computation from flux function.
//!
//! Port of fusion_kernel.py `compute_b_field()` (lines 305-309).
//! In cylindrical tokamak coordinates:
//!   B_R = -(1/R) ∂Ψ/∂Z
//!   B_Z =  (1/R) ∂Ψ/∂R

use fusion_math::interp::gradient_2d;
use fusion_types::error::{FusionError, FusionResult};
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
pub fn compute_b_field(
    psi: &Array2<f64>,
    grid: &Grid2D,
) -> FusionResult<(Array2<f64>, Array2<f64>)> {
    if grid.nz < 2 || grid.nr < 2 {
        return Err(FusionError::ConfigError(format!(
            "b-field grid requires nz,nr >= 2, got nz={} nr={}",
            grid.nz, grid.nr
        )));
    }
    if !grid.dr.is_finite()
        || !grid.dz.is_finite()
        || grid.dr.abs() <= f64::EPSILON
        || grid.dz.abs() <= f64::EPSILON
    {
        return Err(FusionError::ConfigError(format!(
            "b-field grid spacing must be finite and non-zero, got dr={} dz={}",
            grid.dr, grid.dz
        )));
    }
    if psi.nrows() != grid.nz || psi.ncols() != grid.nr {
        return Err(FusionError::ConfigError(format!(
            "b-field psi shape mismatch: expected ({}, {}), got ({}, {})",
            grid.nz,
            grid.nr,
            psi.nrows(),
            psi.ncols()
        )));
    }
    if grid.rr.nrows() != grid.nz || grid.rr.ncols() != grid.nr {
        return Err(FusionError::ConfigError(format!(
            "b-field grid.rr shape mismatch: expected ({}, {}), got ({}, {})",
            grid.nz,
            grid.nr,
            grid.rr.nrows(),
            grid.rr.ncols()
        )));
    }
    if grid.zz.nrows() != grid.nz || grid.zz.ncols() != grid.nr {
        return Err(FusionError::ConfigError(format!(
            "b-field grid.zz shape mismatch: expected ({}, {}), got ({}, {})",
            grid.nz,
            grid.nr,
            grid.zz.nrows(),
            grid.zz.ncols()
        )));
    }
    if psi.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::ConfigError(
            "b-field psi contains non-finite values".to_string(),
        ));
    }

    let (dpsi_dz, dpsi_dr) = gradient_2d(psi, grid);

    let nz = grid.nz;
    let nr = grid.nr;
    let mut b_r = Array2::zeros((nz, nr));
    let mut b_z = Array2::zeros((nz, nr));

    for iz in 0..nz {
        for ir in 0..nr {
            let r = grid.rr[[iz, ir]];
            if !r.is_finite() || r <= 0.0 {
                return Err(FusionError::ConfigError(format!(
                    "b-field radius must be finite and > 0 at ({iz}, {ir}), got {r}"
                )));
            }
            let inv_r = 1.0 / r.max(R_SAFE_MIN);
            let br = -inv_r * dpsi_dz[[iz, ir]];
            let bz = inv_r * dpsi_dr[[iz, ir]];
            if !br.is_finite() || !bz.is_finite() {
                return Err(FusionError::ConfigError(format!(
                    "b-field output became non-finite at ({iz}, {ir})"
                )));
            }
            b_r[[iz, ir]] = br;
            b_z[[iz, ir]] = bz;
        }
    }

    Ok((b_r, b_z))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_b_field_uniform_psi() {
        // Uniform Ψ → zero B-field
        let grid = Grid2D::new(16, 16, 1.0, 9.0, -5.0, 5.0);
        let psi = Array2::from_elem((16, 16), 1.0);
        let (b_r, b_z) = compute_b_field(&psi, &grid).expect("valid b-field inputs");

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
        let (b_r, b_z) = compute_b_field(&psi, &grid).expect("valid b-field inputs");
        assert!(!b_r.iter().any(|v| v.is_nan()), "B_R contains NaN");
        assert!(!b_z.iter().any(|v| v.is_nan()), "B_Z contains NaN");
    }

    #[test]
    fn test_b_field_rejects_invalid_runtime_inputs() {
        let mut grid = Grid2D::new(16, 16, 1.0, 9.0, -5.0, 5.0);
        let psi = Array2::from_elem((16, 16), 1.0);

        grid.rr[[0, 0]] = 0.0;
        let err = compute_b_field(&psi, &grid).expect_err("non-positive radius must fail");
        assert!(matches!(err, FusionError::ConfigError(_)));

        let bad_shape = Array2::from_elem((15, 16), 0.0);
        let err = compute_b_field(&bad_shape, &Grid2D::new(16, 16, 1.0, 9.0, -5.0, 5.0))
            .expect_err("shape mismatch must fail");
        assert!(matches!(err, FusionError::ConfigError(_)));
    }
}
