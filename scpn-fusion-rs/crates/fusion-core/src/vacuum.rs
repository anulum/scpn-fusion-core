// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Vacuum
//! Vacuum magnetic field from coils via toroidal Green's function.
//!
//! Port of fusion_kernel.py `calculate_vacuum_field()` (lines 59-93).
//! Uses complete elliptic integrals K(m), E(m) for the exact toroidal
//! single-coil flux function (Smythe/Jackson form).

use fusion_math::elliptic::{ellipe, ellipk};
use fusion_types::config::CoilConfig;
use fusion_types::error::{FusionError, FusionResult};
use fusion_types::state::Grid2D;
use nalgebra::{DMatrix, DVector};
use ndarray::Array2;

/// Evaluate the circular-filament poloidal-flux Green's function per ampere.
///
/// This external-coupling kernel deliberately excludes coil self-inductance:
/// an observation exactly on the source filament returns zero instead of a
/// clipped singular value.  Self-inductance belongs in the coil-circuit model,
/// not in the free-boundary vacuum-flux map.
pub fn circular_filament_green_function(
    r_src: f64,
    z_src: f64,
    r_obs: f64,
    z_obs: f64,
    mu0: f64,
) -> FusionResult<f64> {
    if !mu0.is_finite() || mu0 <= 0.0 {
        return Err(FusionError::ConfigError(format!(
            "vacuum permeability mu0 must be finite and > 0, got {mu0}"
        )));
    }
    if !r_src.is_finite() || r_src <= 0.0 || !r_obs.is_finite() || r_obs <= 0.0 {
        return Err(FusionError::ConfigError(
            "Green-function radii must be finite and > 0".to_string(),
        ));
    }
    if !z_src.is_finite() || !z_obs.is_finite() {
        return Err(FusionError::ConfigError(
            "Green-function vertical coordinates must be finite".to_string(),
        ));
    }

    circular_filament_green_function_unchecked(r_src, z_src, r_obs, z_obs, mu0)
}

fn circular_filament_green_function_unchecked(
    r_src: f64,
    z_src: f64,
    r_obs: f64,
    z_obs: f64,
    mu0: f64,
) -> FusionResult<f64> {
    let dz = z_obs - z_src;
    let distance_sq = (r_obs - r_src).powi(2) + dz * dz;
    if distance_sq < 1.0e-24 {
        return Ok(0.0);
    }

    let r_plus_rc = r_obs + r_src;
    let denom = r_plus_rc * r_plus_rc + dz * dz;
    if !denom.is_finite() || denom <= 0.0 {
        return Err(FusionError::ConfigError(
            "vacuum field denominator became non-finite or non-positive".to_string(),
        ));
    }

    let k2_raw = (4.0 * r_obs * r_src) / denom;
    if !k2_raw.is_finite() {
        return Err(FusionError::ConfigError(
            "vacuum field k^2 became non-finite".to_string(),
        ));
    }
    let k2 = k2_raw.clamp(1e-12, 1.0 - 1e-12);
    let k_val = ellipk(k2);
    let e_val = ellipe(k2);
    if !k_val.is_finite() || !e_val.is_finite() {
        return Err(FusionError::ConfigError(
            "vacuum field elliptic integrals became non-finite".to_string(),
        ));
    }

    let prefactor = mu0 / (2.0 * std::f64::consts::PI);
    let sqrt_term = (r_obs * r_src).sqrt();
    let k = k2.sqrt();
    let term = ((2.0 - k2) * k_val - 2.0 * e_val) / k;
    let flux = prefactor * sqrt_term * term;
    if !flux.is_finite() {
        return Err(FusionError::ConfigError(
            "vacuum field coil flux contribution became non-finite".to_string(),
        ));
    }
    Ok(flux)
}

/// Calculate vacuum magnetic flux from coils using the toroidal Green's function.
///
/// For each coil at `(Rc, Zc)` with current `I`, the flux contribution is:
///
///   Ψ = (μ₀ I / 2π) · √((R + Rc)² + (Z - Zc)²) · ((2 - k²)K(k²) - 2E(k²)) / k²
///
/// where k² = 4 R Rc / ((R + Rc)² + (Z - Zc)²), clipped to [1e-9, 0.999999].
///
/// Returns Ψ_vacuum `[nz, nr]`.
pub fn calculate_vacuum_field(
    grid: &Grid2D,
    coils: &[CoilConfig],
    mu0: f64,
) -> FusionResult<Array2<f64>> {
    if !mu0.is_finite() || mu0 <= 0.0 {
        return Err(FusionError::ConfigError(format!(
            "vacuum permeability mu0 must be finite and > 0, got {mu0}"
        )));
    }
    if grid.nz == 0 || grid.nr == 0 {
        return Err(FusionError::ConfigError(
            "vacuum field grid must have nz,nr >= 1".to_string(),
        ));
    }
    if grid.rr.nrows() != grid.nz || grid.rr.ncols() != grid.nr {
        return Err(FusionError::ConfigError(format!(
            "grid.rr shape mismatch: expected ({}, {}), got ({}, {})",
            grid.nz,
            grid.nr,
            grid.rr.nrows(),
            grid.rr.ncols()
        )));
    }
    if grid.zz.nrows() != grid.nz || grid.zz.ncols() != grid.nr {
        return Err(FusionError::ConfigError(format!(
            "grid.zz shape mismatch: expected ({}, {}), got ({}, {})",
            grid.nz,
            grid.nr,
            grid.zz.nrows(),
            grid.zz.ncols()
        )));
    }
    if grid.rr.iter().any(|v| !v.is_finite()) || grid.zz.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::ConfigError(
            "vacuum field grid coordinates must be finite".to_string(),
        ));
    }

    let nz = grid.nz;
    let nr = grid.nr;
    let mut psi_vac: Array2<f64> = Array2::zeros((nz, nr));

    for coil in coils {
        let rc = coil.r;
        let zc = coil.z;
        let current = coil.current;
        if !rc.is_finite() || rc <= 0.0 {
            return Err(FusionError::ConfigError(format!(
                "coil radius must be finite and > 0, got {}",
                coil.r
            )));
        }
        if !zc.is_finite() || !current.is_finite() {
            return Err(FusionError::ConfigError(
                "coil z/current must be finite".to_string(),
            ));
        }

        let prefactor = (mu0 * current) / (2.0 * std::f64::consts::PI);
        if !prefactor.is_finite() {
            return Err(FusionError::ConfigError(
                "coil prefactor became non-finite".to_string(),
            ));
        }

        for iz in 0..nz {
            for ir in 0..nr {
                let r = grid.rr[[iz, ir]];
                let z = grid.zz[[iz, ir]];

                let coil_flux =
                    current * circular_filament_green_function_unchecked(rc, zc, r, z, mu0)?;
                psi_vac[[iz, ir]] += coil_flux;
            }
        }
    }

    if psi_vac.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::ConfigError(
            "vacuum field result contains non-finite values".to_string(),
        ));
    }

    Ok(psi_vac)
}

/// Calculate vacuum poloidal flux at explicit boundary or limiter points.
///
/// This is the Rust free-boundary point-response counterpart to the full-grid
/// vacuum solve. It uses the same circular-filament Green's function and does
/// not replay fixed Dirichlet values.
pub fn calculate_vacuum_flux_at_points(
    points: &[(f64, f64)],
    coils: &[CoilConfig],
    mu0: f64,
) -> FusionResult<Vec<f64>> {
    if !mu0.is_finite() || mu0 <= 0.0 {
        return Err(FusionError::ConfigError(format!(
            "vacuum permeability mu0 must be finite and > 0, got {mu0}"
        )));
    }
    if points.is_empty() {
        return Err(FusionError::ConfigError(
            "vacuum flux point list must not be empty".to_string(),
        ));
    }
    let mut flux = vec![0.0; points.len()];
    for (idx, (r_obs, z_obs)) in points.iter().enumerate() {
        if !r_obs.is_finite() || *r_obs <= 0.0 || !z_obs.is_finite() {
            return Err(FusionError::ConfigError(
                "vacuum flux points must be finite with R > 0".to_string(),
            ));
        }
        let mut value = 0.0;
        for coil in coils {
            if !coil.r.is_finite() || coil.r <= 0.0 {
                return Err(FusionError::ConfigError(format!(
                    "coil radius must be finite and > 0, got {}",
                    coil.r
                )));
            }
            if !coil.z.is_finite() || !coil.current.is_finite() {
                return Err(FusionError::ConfigError(
                    "coil z/current must be finite".to_string(),
                ));
            }
            value += coil.current
                * circular_filament_green_function_unchecked(coil.r, coil.z, *r_obs, *z_obs, mu0)?;
        }
        if !value.is_finite() {
            return Err(FusionError::ConfigError(
                "vacuum point flux became non-finite".to_string(),
            ));
        }
        flux[idx] = value;
    }
    Ok(flux)
}

/// Diagnostics from reconstructing free-boundary contour flux from coils.
#[derive(Debug, Clone)]
pub struct BoundaryFluxReconstruction {
    pub reconstructed_flux: Vec<f64>,
    pub residual: Option<Vec<f64>>,
    pub rmse: Option<f64>,
    pub max_abs_error: Option<f64>,
    pub point_count: usize,
    pub coil_count: usize,
    pub limiter_flux: Vec<f64>,
    pub limiter_point_count: usize,
    pub min_limiter_distance_m: Option<f64>,
    pub boundary_containment_fraction: Option<f64>,
    pub boundary_containment_pass: Option<bool>,
    pub axis_flux: Option<f64>,
    pub x_point_flux: Vec<f64>,
    pub x_point_count: usize,
    pub x_point_flux_span: Option<f64>,
    pub x_point_pair_symmetry_abs_error: Option<f64>,
}

/// Diagnostics from reconstructing external coil currents from shape flux targets.
#[derive(Debug, Clone)]
pub struct ShapeCurrentReconstruction {
    pub coil_currents: Vec<f64>,
    pub reconstructed_flux: Vec<f64>,
    pub residual: Vec<f64>,
    pub residual_rmse: f64,
    pub relative_flux_rmse: f64,
    pub response_rank: usize,
    pub response_condition: f64,
    pub active_bounds: usize,
    pub point_count: usize,
    pub coil_count: usize,
}

/// Optional topology metadata for free-boundary contour reconstruction.
#[derive(Debug, Clone, Copy, Default)]
pub struct BoundaryFluxMetadata<'a> {
    pub limiter_points: Option<&'a [(f64, f64)]>,
    pub axis_point: Option<(f64, f64)>,
    pub x_points: Option<&'a [(f64, f64)]>,
}

fn point_inside_polygon(point: (f64, f64), polygon: &[(f64, f64)]) -> bool {
    let (r, z) = point;
    let mut inside = false;
    let mut previous = polygon.len() - 1;
    for current in 0..polygon.len() {
        let (ri, zi) = polygon[current];
        let (rj, zj) = polygon[previous];
        if (zi > z) != (zj > z) {
            let edge_r = (rj - ri) * (z - zi) / (zj - zi) + ri;
            if r < edge_r {
                inside = !inside;
            }
        }
        previous = current;
    }
    inside
}

fn boundary_containment_fraction(
    boundary_points: &[(f64, f64)],
    limiter_points: &[(f64, f64)],
) -> Option<f64> {
    if limiter_points.len() < 3 {
        return None;
    }
    let contained = boundary_points
        .iter()
        .filter(|point| point_inside_polygon(**point, limiter_points))
        .count();
    Some(contained as f64 / boundary_points.len() as f64)
}

fn validate_current_limits(current_limits: Option<&[f64]>, coil_count: usize) -> FusionResult<()> {
    if let Some(limits) = current_limits {
        if limits.len() != coil_count {
            return Err(FusionError::ConfigError(format!(
                "current_limits length must match coil count: got {}, expected {}",
                limits.len(),
                coil_count
            )));
        }
        if limits
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(FusionError::ConfigError(
                "current_limits must contain finite positive values only".to_string(),
            ));
        }
    }
    Ok(())
}

fn boundary_flux_response_matrix(
    boundary_points: &[(f64, f64)],
    coils: &[CoilConfig],
    mu0: f64,
) -> FusionResult<DMatrix<f64>> {
    if boundary_points.is_empty() {
        return Err(FusionError::ConfigError(
            "shape current reconstruction requires at least one boundary point".to_string(),
        ));
    }
    if coils.is_empty() {
        return Err(FusionError::ConfigError(
            "shape current reconstruction requires at least one external coil".to_string(),
        ));
    }

    let mut response = DMatrix::zeros(boundary_points.len(), coils.len());
    for (row, (r_obs, z_obs)) in boundary_points.iter().enumerate() {
        if !r_obs.is_finite() || *r_obs <= 0.0 || !z_obs.is_finite() {
            return Err(FusionError::ConfigError(
                "shape current boundary points must be finite with R > 0".to_string(),
            ));
        }
        for (col, coil) in coils.iter().enumerate() {
            if !coil.r.is_finite() || coil.r <= 0.0 {
                return Err(FusionError::ConfigError(format!(
                    "coil radius must be finite and > 0, got {}",
                    coil.r
                )));
            }
            if !coil.z.is_finite() || !coil.current.is_finite() {
                return Err(FusionError::ConfigError(
                    "coil z/current must be finite".to_string(),
                ));
            }
            response[(row, col)] =
                circular_filament_green_function_unchecked(coil.r, coil.z, *r_obs, *z_obs, mu0)?;
        }
    }
    if response.iter().any(|value| !value.is_finite()) {
        return Err(FusionError::ConfigError(
            "shape current response matrix contains non-finite entries".to_string(),
        ));
    }
    Ok(response)
}

fn matrix_rank_and_condition(response: &DMatrix<f64>) -> (usize, f64) {
    let singular = response.clone().svd(false, false).singular_values;
    let max_sigma = singular.iter().copied().fold(0.0, f64::max);
    let tol = (response.nrows().max(response.ncols()) as f64) * f64::EPSILON * max_sigma;
    let rank = singular.iter().filter(|value| **value > tol).count();
    let min_nonzero = singular
        .iter()
        .copied()
        .filter(|value| *value > tol)
        .fold(f64::INFINITY, f64::min);
    let condition = if rank == 0 || !min_nonzero.is_finite() {
        f64::INFINITY
    } else {
        max_sigma / min_nonzero
    };
    (rank, condition)
}

/// Reconstruct bounded coil currents from boundary/shape flux targets.
///
/// This is the Rust counterpart of the Python free-boundary shape inversion
/// contract. It solves the native Green-function response system directly; no
/// Python wrapper or fixed Dirichlet replay is involved. Bounds are enforced by
/// clipping the regularised least-squares solution, which is exact for the
/// benchmarked full-rank, inactive-bound case and fail-safe for saturated
/// actuator requests.
pub fn reconstruct_shape_currents_from_boundary_flux(
    boundary_points: &[(f64, f64)],
    coil_templates: &[CoilConfig],
    target_flux: &[f64],
    current_limits: Option<&[f64]>,
    tikhonov_alpha: f64,
    mu0: f64,
) -> FusionResult<ShapeCurrentReconstruction> {
    if target_flux.len() != boundary_points.len() {
        return Err(FusionError::ConfigError(format!(
            "target_flux length must match boundary point count: got {}, expected {}",
            target_flux.len(),
            boundary_points.len()
        )));
    }
    if target_flux.iter().any(|value| !value.is_finite()) {
        return Err(FusionError::ConfigError(
            "target_flux must contain finite values only".to_string(),
        ));
    }
    if !tikhonov_alpha.is_finite() || tikhonov_alpha < 0.0 {
        return Err(FusionError::ConfigError(
            "tikhonov_alpha must be finite and non-negative".to_string(),
        ));
    }
    validate_current_limits(current_limits, coil_templates.len())?;

    let response = boundary_flux_response_matrix(boundary_points, coil_templates, mu0)?;
    let target = DVector::from_column_slice(target_flux);
    let mut normal = response.transpose() * &response;
    for diagonal in 0..normal.nrows() {
        normal[(diagonal, diagonal)] += tikhonov_alpha;
    }
    let rhs = response.transpose() * target;
    let solved = normal.lu().solve(&rhs).ok_or_else(|| {
        FusionError::ConfigError(
            "shape current normal equations are singular; add independent control points or regularization"
                .to_string(),
        )
    })?;
    let mut currents: Vec<f64> = solved.iter().copied().collect();
    if currents.iter().any(|value| !value.is_finite()) {
        return Err(FusionError::ConfigError(
            "shape current reconstruction produced non-finite currents".to_string(),
        ));
    }
    if let Some(limits) = current_limits {
        for (current, limit) in currents.iter_mut().zip(limits.iter()) {
            *current = current.clamp(-limit.abs(), limit.abs());
        }
    }

    let current_vec = DVector::from_column_slice(&currents);
    let reconstructed = response.clone() * current_vec;
    let reconstructed_flux: Vec<f64> = reconstructed.iter().copied().collect();
    let residual: Vec<f64> = reconstructed_flux
        .iter()
        .zip(target_flux.iter())
        .map(|(observed, expected)| observed - expected)
        .collect();
    let residual_rmse =
        (residual.iter().map(|value| value * value).sum::<f64>() / residual.len() as f64).sqrt();
    let target_rmse = (target_flux.iter().map(|value| value * value).sum::<f64>()
        / target_flux.len() as f64)
        .sqrt();
    let relative_flux_rmse = residual_rmse / target_rmse.abs().max(1.0);
    let active_bounds = if let Some(limits) = current_limits {
        currents
            .iter()
            .zip(limits.iter())
            .filter(|(current, limit)| (current.abs() - limit.abs()).abs() <= 1.0e-9)
            .count()
    } else {
        0
    };
    let (response_rank, response_condition) = matrix_rank_and_condition(&response);

    Ok(ShapeCurrentReconstruction {
        coil_currents: currents,
        reconstructed_flux,
        residual,
        residual_rmse,
        relative_flux_rmse,
        response_rank,
        response_condition,
        active_bounds,
        point_count: boundary_points.len(),
        coil_count: coil_templates.len(),
    })
}

/// Reconstruct boundary-contour vacuum flux from coil Green functions.
///
/// This is the Rust counterpart of the Python free-boundary forward contract:
/// it evaluates external coil flux on explicit boundary or limiter points and,
/// when a target vector is supplied, reports residual diagnostics without
/// fitting a scale or replaying fixed Dirichlet values.
pub fn reconstruct_boundary_flux_from_coils(
    boundary_points: &[(f64, f64)],
    coils: &[CoilConfig],
    target_flux: Option<&[f64]>,
    mu0: f64,
) -> FusionResult<BoundaryFluxReconstruction> {
    reconstruct_boundary_flux_from_coils_with_metadata(
        boundary_points,
        coils,
        target_flux,
        mu0,
        BoundaryFluxMetadata::default(),
    )
}

/// Reconstruct boundary-contour flux and optional topology metadata.
pub fn reconstruct_boundary_flux_from_coils_with_metadata(
    boundary_points: &[(f64, f64)],
    coils: &[CoilConfig],
    target_flux: Option<&[f64]>,
    mu0: f64,
    metadata: BoundaryFluxMetadata<'_>,
) -> FusionResult<BoundaryFluxReconstruction> {
    if coils.is_empty() {
        return Err(FusionError::ConfigError(
            "free-boundary reconstruction requires at least one external coil".to_string(),
        ));
    }
    let reconstructed_flux = calculate_vacuum_flux_at_points(boundary_points, coils, mu0)?;
    let point_count = reconstructed_flux.len();
    let mut residual = None;
    let mut rmse = None;
    let mut max_abs_error = None;

    if let Some(target) = target_flux {
        if target.len() != point_count {
            return Err(FusionError::ConfigError(format!(
                "target_flux length must match boundary point count: got {}, expected {}",
                target.len(),
                point_count
            )));
        }
        if target.iter().any(|v| !v.is_finite()) {
            return Err(FusionError::ConfigError(
                "target_flux must contain finite values only".to_string(),
            ));
        }
        let values: Vec<f64> = reconstructed_flux
            .iter()
            .zip(target.iter())
            .map(|(observed, expected)| observed - expected)
            .collect();
        let mean_square = values.iter().map(|v| v * v).sum::<f64>() / point_count as f64;
        let max_error = values.iter().map(|v| v.abs()).fold(0.0, f64::max);
        residual = Some(values);
        rmse = Some(mean_square.sqrt());
        max_abs_error = Some(max_error);
    }

    let limiter_flux = if let Some(points) = metadata.limiter_points {
        calculate_vacuum_flux_at_points(points, coils, mu0)?
    } else {
        Vec::new()
    };
    let min_limiter_distance_m = metadata.limiter_points.map(|points| {
        points
            .iter()
            .flat_map(|(rl, zl)| {
                boundary_points
                    .iter()
                    .map(move |(rb, zb)| ((*rl - *rb).powi(2) + (*zl - *zb).powi(2)).sqrt())
            })
            .fold(f64::INFINITY, f64::min)
    });
    let boundary_containment_fraction = metadata
        .limiter_points
        .and_then(|points| boundary_containment_fraction(boundary_points, points));
    let boundary_containment_pass = boundary_containment_fraction.map(|fraction| fraction >= 1.0);
    let axis_flux = if let Some(axis) = metadata.axis_point {
        Some(calculate_vacuum_flux_at_points(&[axis], coils, mu0)?[0])
    } else {
        None
    };
    let x_point_flux = if let Some(points) = metadata.x_points {
        calculate_vacuum_flux_at_points(points, coils, mu0)?
    } else {
        Vec::new()
    };
    let x_point_flux_span = if x_point_flux.is_empty() {
        None
    } else {
        let min_flux = x_point_flux.iter().copied().fold(f64::INFINITY, f64::min);
        let max_flux = x_point_flux
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        Some(max_flux - min_flux)
    };
    let x_point_pair_symmetry_abs_error =
        if let (Some(axis), Some(points)) = (metadata.axis_point, metadata.x_points) {
            if points.len() == 2
                && (points[0].0 - points[1].0).abs() <= 1.0e-9
                && (points[0].1 + points[1].1 - 2.0 * axis.1).abs() <= 1.0e-9
            {
                Some((x_point_flux[0] - x_point_flux[1]).abs())
            } else {
                None
            }
        } else {
            None
        };

    Ok(BoundaryFluxReconstruction {
        reconstructed_flux,
        residual,
        rmse,
        max_abs_error,
        point_count,
        coil_count: coils.len(),
        limiter_point_count: limiter_flux.len(),
        limiter_flux,
        min_limiter_distance_m,
        boundary_containment_fraction,
        boundary_containment_pass,
        axis_flux,
        x_point_count: x_point_flux.len(),
        x_point_flux,
        x_point_flux_span,
        x_point_pair_symmetry_abs_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusion_types::config::ReactorConfig;
    use std::path::PathBuf;

    fn project_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
    }

    fn config_path(relative: &str) -> String {
        project_root().join(relative).to_string_lossy().to_string()
    }

    #[test]
    fn test_vacuum_field_not_nan() {
        let cfg = ReactorConfig::from_file(&config_path("validation/iter_config.json")).unwrap();
        let grid = cfg.create_grid();
        let psi_vac = calculate_vacuum_field(&grid, &cfg.coils, cfg.physics.vacuum_permeability)
            .expect("valid vacuum-field inputs");

        // No NaN values
        assert!(
            !psi_vac.iter().any(|v| v.is_nan()),
            "Vacuum field contains NaN"
        );

        // Should have non-zero values (coils produce flux)
        let max_val = psi_vac.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_val.abs() > 1e-10, "Vacuum field is all zeros");
    }

    #[test]
    fn test_vacuum_field_symmetry() {
        // Single coil at Z=0 should produce field symmetric about Z=0
        let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
        let coil = CoilConfig {
            name: "test".to_string(),
            r: 5.0,
            z: 0.0,
            current: 1.0,
        };
        let psi = calculate_vacuum_field(&grid, &[coil], 1.0).expect("valid vacuum-field inputs");

        // Check symmetry: psi[iz, ir] ≈ psi[nz-1-iz, ir]
        for ir in 0..33 {
            for iz in 0..16 {
                let diff = (psi[[iz, ir]] - psi[[32 - iz, ir]]).abs();
                assert!(
                    diff < 1e-10,
                    "Symmetry broken at iz={iz}, ir={ir}: {} vs {}",
                    psi[[iz, ir]],
                    psi[[32 - iz, ir]]
                );
            }
        }
    }

    #[test]
    fn test_vacuum_field_single_coil_shape() {
        // Flux from a single coil should peak near the coil position
        let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
        let coil = CoilConfig {
            name: "test".to_string(),
            r: 5.0,
            z: 0.0,
            current: 1.0,
        };
        let psi = calculate_vacuum_field(&grid, &[coil], 1.0).expect("valid vacuum-field inputs");

        // Find maximum
        let mut max_val = f64::NEG_INFINITY;
        let mut max_iz = 0;
        let mut max_ir = 0;
        for iz in 0..33 {
            for ir in 0..33 {
                if psi[[iz, ir]] > max_val {
                    max_val = psi[[iz, ir]];
                    max_iz = iz;
                    max_ir = ir;
                }
            }
        }

        // Maximum should be near Z=0 (middle row)
        assert!(
            (max_iz as f64 - 16.0).abs() < 3.0,
            "Peak not near Z=0: iz={max_iz}"
        );
        // Maximum should be near R=5.0 (middle of grid)
        assert!(
            (max_ir as f64 - 16.0).abs() < 3.0,
            "Peak not near R=5: ir={max_ir}"
        );
    }

    #[test]
    fn test_vacuum_field_rejects_invalid_runtime_inputs() {
        let grid = Grid2D::new(17, 17, 1.0, 9.0, -5.0, 5.0);
        let bad_coil = CoilConfig {
            name: "bad".to_string(),
            r: 0.0,
            z: 0.0,
            current: 1.0,
        };
        assert!(calculate_vacuum_field(&grid, &[bad_coil], 1.0).is_err());
        assert!(calculate_vacuum_field(&grid, &[], f64::NAN).is_err());
    }

    #[test]
    fn test_green_function_self_observation_is_regularised() {
        let flux = circular_filament_green_function(5.0, 0.0, 5.0, 0.0, 1.0)
            .expect("self-observation should regularise");
        assert_eq!(flux, 0.0);
    }

    #[test]
    fn test_vacuum_field_is_linear_in_current_and_self_regularised() {
        let grid = Grid2D::new(17, 17, 1.0, 9.0, -4.0, 4.0);
        let coil = CoilConfig {
            name: "self".to_string(),
            r: 5.0,
            z: 0.0,
            current: 2.0,
        };
        let mut scaled = coil.clone();
        scaled.current = 6.0;

        let psi = calculate_vacuum_field(&grid, &[coil], 1.0).expect("valid vacuum field");
        let psi_scaled =
            calculate_vacuum_field(&grid, &[scaled], 1.0).expect("valid scaled vacuum field");

        assert_eq!(psi[[8, 8]], 0.0);
        for iz in 0..17 {
            for ir in 0..17 {
                assert!((psi_scaled[[iz, ir]] - 3.0 * psi[[iz, ir]]).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn test_vacuum_flux_at_points_matches_grid_values() {
        let grid = Grid2D::new(17, 17, 1.0, 9.0, -4.0, 4.0);
        let coils = vec![
            CoilConfig {
                name: "upper".to_string(),
                r: 5.0,
                z: 1.5,
                current: 2.0,
            },
            CoilConfig {
                name: "lower".to_string(),
                r: 6.0,
                z: -1.0,
                current: -0.75,
            },
        ];
        let psi = calculate_vacuum_field(&grid, &coils, 1.0).expect("valid vacuum field");
        let points = vec![
            (grid.rr[[3, 4]], grid.zz[[3, 4]]),
            (grid.rr[[8, 8]], grid.zz[[8, 8]]),
            (grid.rr[[12, 11]], grid.zz[[12, 11]]),
        ];
        let point_flux =
            calculate_vacuum_flux_at_points(&points, &coils, 1.0).expect("valid point flux");

        assert!((point_flux[0] - psi[[3, 4]]).abs() < 1.0e-12);
        assert!((point_flux[1] - psi[[8, 8]]).abs() < 1.0e-12);
        assert!((point_flux[2] - psi[[12, 11]]).abs() < 1.0e-12);
    }

    #[test]
    fn test_boundary_flux_reconstruction_reports_residual_contract() {
        let coils = vec![
            CoilConfig {
                name: "upper".to_string(),
                r: 5.0,
                z: 1.5,
                current: 2.0,
            },
            CoilConfig {
                name: "lower".to_string(),
                r: 6.0,
                z: -1.0,
                current: -0.75,
            },
        ];
        let boundary_points = vec![(4.5, -1.0), (5.5, -1.25), (6.5, 0.25), (5.5, 1.25)];
        let target = calculate_vacuum_flux_at_points(&boundary_points, &coils, 1.0)
            .expect("valid target flux");

        let reconstruction =
            reconstruct_boundary_flux_from_coils(&boundary_points, &coils, Some(&target), 1.0)
                .expect("valid boundary reconstruction");

        assert_eq!(reconstruction.point_count, boundary_points.len());
        assert_eq!(reconstruction.coil_count, coils.len());
        assert!(reconstruction
            .residual
            .as_ref()
            .expect("target residual")
            .iter()
            .all(|v| v.abs() < 1.0e-12));
        assert!(reconstruction.rmse.expect("rmse") < 1.0e-12);
        assert!(reconstruction.max_abs_error.expect("max error") < 1.0e-12);
    }

    #[test]
    fn test_boundary_flux_reconstruction_rejects_empty_coil_set() {
        let boundary_points = vec![(4.5, -1.0), (5.5, -1.25), (6.5, 0.25), (5.5, 1.25)];

        let err = reconstruct_boundary_flux_from_coils(&boundary_points, &[], None, 1.0)
            .expect_err("free-boundary reconstruction requires at least one external coil");

        assert!(matches!(err, FusionError::ConfigError(_)));
    }

    #[test]
    fn test_boundary_flux_reconstruction_reports_topology_metadata() {
        let coils = vec![CoilConfig {
            name: "pf".to_string(),
            r: 5.0,
            z: 0.5,
            current: 2.0,
        }];
        let boundary_points = vec![(4.5, -1.0), (5.5, -1.25), (6.5, 0.25), (5.5, 1.25)];
        let limiter_points = vec![(4.25, -1.5), (6.75, -1.5), (6.75, 1.5), (4.25, 1.5)];
        let x_points = vec![(6.0, -0.75), (6.0, 0.75)];

        let reconstruction = reconstruct_boundary_flux_from_coils_with_metadata(
            &boundary_points,
            &coils,
            None,
            1.0,
            BoundaryFluxMetadata {
                limiter_points: Some(&limiter_points),
                axis_point: Some((5.5, 0.0)),
                x_points: Some(&x_points),
            },
        )
        .expect("valid topology reconstruction");

        assert_eq!(reconstruction.limiter_point_count, limiter_points.len());
        assert_eq!(reconstruction.x_point_count, x_points.len());
        assert!(reconstruction.axis_flux.expect("axis flux").is_finite());
        assert_eq!(reconstruction.limiter_flux.len(), limiter_points.len());
        assert_eq!(reconstruction.x_point_flux.len(), x_points.len());
        assert!(
            reconstruction
                .min_limiter_distance_m
                .expect("limiter clearance")
                > 0.0
        );
    }

    #[test]
    fn test_boundary_flux_reconstruction_reports_limiter_containment_fraction() {
        let coils = vec![CoilConfig {
            name: "pf".to_string(),
            r: 5.0,
            z: 0.0,
            current: 2.0,
        }];
        let boundary_points = vec![(4.5, -0.5), (5.5, -0.5), (6.5, 0.5), (8.0, 0.5)];
        let limiter_points = vec![(4.0, -1.0), (7.0, -1.0), (7.0, 1.0), (4.0, 1.0)];

        let reconstruction = reconstruct_boundary_flux_from_coils_with_metadata(
            &boundary_points,
            &coils,
            None,
            1.0,
            BoundaryFluxMetadata {
                limiter_points: Some(&limiter_points),
                axis_point: None,
                x_points: None,
            },
        )
        .expect("valid limiter containment reconstruction");

        assert_eq!(
            reconstruction
                .boundary_containment_fraction
                .expect("containment fraction"),
            0.75
        );
        assert!(!reconstruction
            .boundary_containment_pass
            .expect("containment pass"));
    }

    #[test]
    fn test_boundary_flux_reconstruction_reports_x_point_topology_residual() {
        let coils = vec![CoilConfig {
            name: "pf".to_string(),
            r: 5.0,
            z: 0.0,
            current: 2.0,
        }];
        let boundary_points = vec![(4.5, -1.0), (5.5, -1.25), (6.5, 0.25), (5.5, 1.25)];
        let x_points = vec![(6.0, -0.75), (6.0, 0.75)];

        let reconstruction = reconstruct_boundary_flux_from_coils_with_metadata(
            &boundary_points,
            &coils,
            None,
            1.0,
            BoundaryFluxMetadata {
                limiter_points: None,
                axis_point: Some((5.0, 0.0)),
                x_points: Some(&x_points),
            },
        )
        .expect("valid topology reconstruction");

        assert!(
            reconstruction.x_point_flux_span.expect("X-point span") < 1.0e-14,
            "symmetric X-point pair should have equal vacuum flux"
        );
        assert!(
            reconstruction
                .x_point_pair_symmetry_abs_error
                .expect("X-point pair symmetry")
                < 1.0e-14,
            "symmetric X-point pair should report a bounded topology residual"
        );
    }

    #[test]
    fn test_shape_current_reconstruction_recovers_boundary_flux_currents() {
        let coil_templates = vec![
            CoilConfig {
                name: "inner".to_string(),
                r: 0.8,
                z: 0.0,
                current: 0.0,
            },
            CoilConfig {
                name: "upper".to_string(),
                r: 1.85,
                z: 1.15,
                current: 0.0,
            },
            CoilConfig {
                name: "lower".to_string(),
                r: 1.85,
                z: -1.15,
                current: 0.0,
            },
        ];
        let boundary_points = vec![
            (0.75, -0.95),
            (1.25, -1.20),
            (2.15, -0.25),
            (2.15, 0.85),
            (1.20, 1.20),
        ];
        let true_currents = [0.85e6, -0.45e6, 0.30e6];
        let mut driven_coils = coil_templates.clone();
        for (coil, current) in driven_coils.iter_mut().zip(true_currents.iter()) {
            coil.current = *current;
        }
        let target_flux = calculate_vacuum_flux_at_points(&boundary_points, &driven_coils, 1.0)
            .expect("valid target flux");
        let limits = [1.2e6, 1.2e6, 1.2e6];

        let reconstruction = reconstruct_shape_currents_from_boundary_flux(
            &boundary_points,
            &coil_templates,
            &target_flux,
            Some(&limits),
            0.0,
            1.0,
        )
        .expect("full-rank shape current reconstruction should solve");

        assert_eq!(reconstruction.point_count, boundary_points.len());
        assert_eq!(reconstruction.coil_count, coil_templates.len());
        assert_eq!(reconstruction.response_rank, coil_templates.len());
        assert!(reconstruction.response_condition.is_finite());
        assert_eq!(reconstruction.active_bounds, 0);
        assert!(reconstruction.relative_flux_rmse < 1.0e-12);
        assert!(reconstruction.residual_rmse < 1.0e-9);
        for (observed, expected) in reconstruction
            .coil_currents
            .iter()
            .zip(true_currents.iter())
        {
            assert!(((observed - expected) / expected).abs() < 1.0e-9);
        }
    }

    #[test]
    fn test_shape_current_reconstruction_enforces_current_limits() {
        let coil_templates = vec![CoilConfig {
            name: "limited".to_string(),
            r: 5.0,
            z: 0.0,
            current: 0.0,
        }];
        let boundary_points = vec![(5.5, 0.25)];
        let mut driven_coils = coil_templates.clone();
        driven_coils[0].current = 10.0;
        let target_flux = calculate_vacuum_flux_at_points(&boundary_points, &driven_coils, 1.0)
            .expect("valid target flux");
        let limits = [0.5];

        let reconstruction = reconstruct_shape_currents_from_boundary_flux(
            &boundary_points,
            &coil_templates,
            &target_flux,
            Some(&limits),
            0.0,
            1.0,
        )
        .expect("bounded shape current reconstruction should return saturated current");

        assert!(reconstruction.coil_currents[0].abs() <= limits[0] + 1.0e-12);
        assert_eq!(reconstruction.active_bounds, 1);
        assert!(reconstruction.relative_flux_rmse > 0.0);
    }
}
