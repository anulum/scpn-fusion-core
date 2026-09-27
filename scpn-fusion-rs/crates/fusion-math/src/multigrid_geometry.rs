// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Multigrid geometry
//! Validate both independently generated native and restricted NumPy multigrid geometry.

use crate::multigrid::MultigridConfig;
use fusion_types::{array_storage::try_zeros, state::Grid2D};
use ndarray::{Array2, ArrayView1};

/// Check every actual coordinate before indexing or constructing a mesh.
fn axis(values: ArrayView1<'_, f64>) -> Result<(), String> {
    if !values.iter().all(|v| v.is_finite())
        || !values.iter().zip(values.iter().skip(1)).all(|(a, b)| b > a)
    {
        return Err("actual multigrid axes must be finite and strictly increasing".to_owned());
    }
    Ok(())
}

/// Validate identical denominators, reciprocals and signed off-diagonal coefficients.
fn level(radius: &Array2<f64>, dr: f64, dz: f64) -> Result<(), String> {
    let dr2 = dr * dr;
    let dz2 = dz * dz;
    let inv_r = 1.0 / dr2;
    let inv_z = 1.0 / dz2;
    let center = 2.0 * inv_r + 2.0 * inv_z;
    if ![dr, dz, dr2, dz2, inv_r, inv_z, 1.0 / (2.0 * dr), center]
        .iter()
        .all(|v| v.is_finite() && *v > 0.0)
    {
        return Err("multigrid spacing or inverse is not representable".to_owned());
    }
    let (nz, nr) = radius.dim();
    for iz in 1..nz - 1 {
        for ir in 1..nr - 1 {
            let r = radius[[iz, ir]];
            let denominator = 2.0 * r * dr;
            if ![r, 1.0 / r, denominator]
                .iter()
                .all(|v| v.is_finite() && *v > 0.0)
                || ![inv_r - 1.0 / denominator, inv_r + 1.0 / denominator]
                    .iter()
                    .all(|v| v.is_finite())
            {
                return Err("multigrid radial stencil is not representable".to_owned());
            }
        }
    }
    Ok(())
}

/// Check actual native fine/coarse generation, including the endpoints used at the next level.
pub fn validate_multigrid_geometry(grid: &Grid2D, config: &MultigridConfig) -> Result<(), String> {
    if grid.nr < 3 || grid.nz < 3 {
        return Err("fine grid must have at least three nodes per axis".to_owned());
    }
    let mut coarse: Option<Grid2D> = None;
    loop {
        let current = coarse.as_ref().unwrap_or(grid);
        if current.nr < 2
            || current.nz < 2
            || config.min_grid_size < 3
            || current.r.len() != current.nr
            || current.z.len() != current.nz
            || current.rr.dim() != (current.nz, current.nr)
        {
            return Err("invalid multigrid mesh dimensions".to_owned());
        }
        axis(current.r.view())?;
        axis(current.z.view())?;
        level(&current.rr, current.dr, current.dz)?;
        if current.nr <= config.min_grid_size || current.nz <= config.min_grid_size {
            return Ok(());
        }
        coarse = Some(Grid2D::try_new(
            current.nr.div_ceil(2),
            current.nz.div_ceil(2),
            current.r[0],
            current.r[current.nr - 1],
            current.z[0],
            current.z[current.nz - 1],
        )?);
    }
}

/// Reproduce the exact NumPy full-weight geometry restriction and boundary injection order.
fn restrict_geometry(fine: &Array2<f64>) -> Result<Array2<f64>, String> {
    let (nz, nr) = fine.dim();
    let (cnz, cnr) = (nz.div_ceil(2), nr.div_ceil(2));
    let mut coarse = try_zeros(cnz, cnr)?;
    for iz in 1..cnz - 1 {
        for ir in 1..cnr - 1 {
            let (i, j) = (2 * iz, 2 * ir);
            coarse[[iz, ir]] = (4.0 * fine[[i, j]]
                + 2.0
                    * (fine[[i - 1, j]] + fine[[i + 1, j]] + fine[[i, j - 1]] + fine[[i, j + 1]])
                + (fine[[i - 1, j - 1]]
                    + fine[[i - 1, j + 1]]
                    + fine[[i + 1, j - 1]]
                    + fine[[i + 1, j + 1]]))
                / 16.0;
        }
    }
    for ir in 0..cnr {
        coarse[[0, ir]] = fine[[0, 2 * ir]];
        coarse[[cnz - 1, ir]] = fine[[nz - 1, 2 * ir]];
    }
    for iz in 0..cnz {
        coarse[[iz, 0]] = fine[[2 * iz, 0]];
        coarse[[iz, cnr - 1]] = fine[[2 * iz, nr - 1]];
    }
    Ok(coarse)
}

/// Validate the other public provider's actual NumPy axes and doubled-spacing coarse sequence.
pub fn validate_numpy_multigrid_geometry(
    r: &[f64],
    z: &[f64],
    minimum: usize,
) -> Result<(), String> {
    if r.len() < 3 || z.len() < 3 || minimum < 3 {
        return Err("invalid NumPy multigrid dimensions".to_owned());
    }
    axis(ndarray::ArrayView1::from(r))?;
    axis(ndarray::ArrayView1::from(z))?;
    let mut mesh = try_zeros(z.len(), r.len())?;
    for iz in 0..z.len() {
        for ir in 0..r.len() {
            mesh[[iz, ir]] = r[ir];
        }
    }
    let (mut dr, mut dz) = (r[1] - r[0], z[1] - z[0]);
    loop {
        level(&mesh, dr, dz)?;
        if mesh.dim().0 <= minimum || mesh.dim().1 <= minimum {
            return Ok(());
        }
        mesh = restrict_geometry(&mesh)?;
        dr *= 2.0;
        dz *= 2.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multigrid::try_multigrid_solve;

    /// Preserve fixed-cycle tol0 native kernel calls while requiring truthful residual metadata.
    #[test]
    fn internal_zero_tolerance_is_a_fixed_budget() {
        let grid = Grid2D::try_new(7, 5, 1.0, 3.0, -1.0, 1.0).unwrap();
        let source = Array2::ones((5, 7));
        let mut flux = Array2::zeros((5, 7));
        let result = try_multigrid_solve(
            &mut flux,
            &source,
            &grid,
            &MultigridConfig::default(),
            1,
            0.0,
        )
        .unwrap();
        assert_eq!(result.cycles, 1);
        assert!(!result.converged);
        assert!(result.residual.is_finite());
        assert_eq!(result.residual_history.len(), 2);
    }

    /// Reject malformed public Grid2D values and nonrepresentable stencils before numerical work.
    #[test]
    fn public_geometry_refusals() {
        let config = MultigridConfig::default();
        let mut grid = Grid2D::try_new(7, 5, 1.0, 3.0, -1.0, 1.0).unwrap();
        grid.nr = 2;
        assert!(validate_multigrid_geometry(&grid, &config).is_err());
        grid.nr = 8;
        assert!(validate_multigrid_geometry(&grid, &config).is_err());
        grid.nr = 7;
        grid.r[2] = grid.r[1];
        assert!(validate_multigrid_geometry(&grid, &config).is_err());
        for (r_min, r_max, z_min, z_max) in [
            (1.0, 3.0, 0.0, 1e200),
            (1e-152, 1.06e-152, -1.0, 1.0),
            (1e-200, 2e-200, -1.0, 1.0),
        ] {
            let invalid = Grid2D::try_new(7, 5, r_min, r_max, z_min, z_max).unwrap();
            assert!(validate_multigrid_geometry(&invalid, &config).is_err());
        }
        assert!(validate_numpy_multigrid_geometry(&[1.0, 2.0], &[-1.0, 0.0, 1.0], 5).is_err());
        assert!(validate_numpy_multigrid_geometry(&[1.0, 1.0, 2.0], &[-1.0, 0.0, 1.0], 5).is_err());
    }

    /// Admit both generated paths on affordable odd/even rectangular meshes and multiple levels.
    #[test]
    fn public_geometry_rectangles() {
        for (nr, nz) in [(3, 3), (5, 3), (8, 6), (6, 8), (15, 13), (17, 17)] {
            let grid = Grid2D::try_new(nr, nz, 1.0, 3.0, -1.0, 1.0).unwrap();
            validate_multigrid_geometry(&grid, &MultigridConfig::default()).unwrap();
            validate_numpy_multigrid_geometry(
                grid.r.as_slice().unwrap(),
                grid.z.as_slice().unwrap(),
                5,
            )
            .unwrap();
        }
    }
}
