//! Bilinear interpolation on Grid2D.
//!
//! Port of stability_analyzer.py get_vacuum_field_at() lines 28-31.

use fusion_types::state::Grid2D;
use ndarray::Array2;

/// Bilinear interpolation on a Grid2D.
///
/// Returns the interpolated value of `field` at position `(r, z)`.
/// Clamps to grid boundaries if outside.
pub fn interp2d(field: &Array2<f64>, grid: &Grid2D, r: f64, z: f64) -> f64 {
    // Map (r, z) to fractional grid indices
    let fr = (r - grid.r[0]) / grid.dr;
    let fz = (z - grid.z[0]) / grid.dz;

    // Clamp to valid range
    let ir0 = (fr.floor() as isize).clamp(0, (grid.nr as isize) - 2) as usize;
    let iz0 = (fz.floor() as isize).clamp(0, (grid.nz as isize) - 2) as usize;

    let ir1 = ir0 + 1;
    let iz1 = iz0 + 1;

    // Fractional position within cell
    let tr = (fr - ir0 as f64).clamp(0.0, 1.0);
    let tz = (fz - iz0 as f64).clamp(0.0, 1.0);

    // Bilinear interpolation
    let v00 = field[[iz0, ir0]];
    let v10 = field[[iz1, ir0]];
    let v01 = field[[iz0, ir1]];
    let v11 = field[[iz1, ir1]];

    (1.0 - tz) * ((1.0 - tr) * v00 + tr * v01) + tz * ((1.0 - tr) * v10 + tr * v11)
}

/// Compute gradient of a 2D field using central differences.
///
/// Returns (df_dz, df_dr) matching the axis convention:
/// - axis 0 = Z (rows) → df_dz
/// - axis 1 = R (cols) → df_dr
///
/// Uses forward/backward differences at boundaries.
pub fn gradient_2d(field: &Array2<f64>, grid: &Grid2D) -> (Array2<f64>, Array2<f64>) {
    let (nz, nr) = field.dim();
    let mut df_dz = Array2::zeros((nz, nr));
    let mut df_dr = Array2::zeros((nz, nr));

    // df/dZ (axis 0)
    for iz in 0..nz {
        for ir in 0..nr {
            if iz == 0 {
                df_dz[[iz, ir]] = (field[[1, ir]] - field[[0, ir]]) / grid.dz;
            } else if iz == nz - 1 {
                df_dz[[iz, ir]] = (field[[nz - 1, ir]] - field[[nz - 2, ir]]) / grid.dz;
            } else {
                df_dz[[iz, ir]] = (field[[iz + 1, ir]] - field[[iz - 1, ir]]) / (2.0 * grid.dz);
            }
        }
    }

    // df/dR (axis 1)
    for iz in 0..nz {
        for ir in 0..nr {
            if ir == 0 {
                df_dr[[iz, ir]] = (field[[iz, 1]] - field[[iz, 0]]) / grid.dr;
            } else if ir == nr - 1 {
                df_dr[[iz, ir]] = (field[[iz, nr - 1]] - field[[iz, nr - 2]]) / grid.dr;
            } else {
                df_dr[[iz, ir]] = (field[[iz, ir + 1]] - field[[iz, ir - 1]]) / (2.0 * grid.dr);
            }
        }
    }

    (df_dz, df_dr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interp2d_exact_gridpoint() {
        let grid = Grid2D::new(5, 5, 0.0, 4.0, 0.0, 4.0);
        let field = Array2::from_shape_fn((5, 5), |(iz, ir)| (iz * 10 + ir) as f64);

        // Value at exact grid point (R=2, Z=2) → ir=2, iz=2 → field[2,2] = 22
        let val = interp2d(&field, &grid, 2.0, 2.0);
        assert!((val - 22.0).abs() < 1e-10, "val = {val}, expected 22");
    }

    #[test]
    fn test_interp2d_midpoint() {
        let grid = Grid2D::new(5, 5, 0.0, 4.0, 0.0, 4.0);
        // Constant field
        let field = Array2::from_elem((5, 5), 7.0);
        let val = interp2d(&field, &grid, 1.5, 2.5);
        assert!((val - 7.0).abs() < 1e-10, "Constant field interpolation");
    }

    #[test]
    fn test_interp2d_linear() {
        let grid = Grid2D::new(11, 11, 0.0, 10.0, 0.0, 10.0);
        // f(R, Z) = R + Z (linear)
        let field =
            Array2::from_shape_fn((11, 11), |(iz, ir)| grid.rr[[iz, ir]] + grid.zz[[iz, ir]]);
        // At (R=3.5, Z=6.5), expected = 3.5 + 6.5 = 10.0
        let val = interp2d(&field, &grid, 3.5, 6.5);
        assert!((val - 10.0).abs() < 1e-10, "Linear interpolation: {val}");
    }

    #[test]
    fn test_gradient_2d_linear() {
        let grid = Grid2D::new(11, 11, 0.0, 10.0, 0.0, 10.0);
        // f(R, Z) = 2*R + 3*Z → df/dR = 2, df/dZ = 3
        let field = Array2::from_shape_fn((11, 11), |(iz, ir)| {
            2.0 * grid.rr[[iz, ir]] + 3.0 * grid.zz[[iz, ir]]
        });
        let (df_dz, df_dr) = gradient_2d(&field, &grid);

        // Check interior points (central differences are exact for linear)
        for iz in 1..10 {
            for ir in 1..10 {
                assert!(
                    (df_dr[[iz, ir]] - 2.0).abs() < 1e-10,
                    "df/dR at ({iz},{ir}) = {}",
                    df_dr[[iz, ir]]
                );
                assert!(
                    (df_dz[[iz, ir]] - 3.0).abs() < 1e-10,
                    "df/dZ at ({iz},{ir}) = {}",
                    df_dz[[iz, ir]]
                );
            }
        }
    }
}
