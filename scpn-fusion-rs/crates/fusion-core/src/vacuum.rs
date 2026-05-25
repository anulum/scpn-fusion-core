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

    Ok(BoundaryFluxReconstruction {
        reconstructed_flux,
        residual,
        rmse,
        max_abs_error,
        point_count,
        coil_count: coils.len(),
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
        let cfg = ReactorConfig::from_file(&config_path("iter_config.json")).unwrap();
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
}
