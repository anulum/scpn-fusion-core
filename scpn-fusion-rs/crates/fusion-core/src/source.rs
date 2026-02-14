//! Nonlinear plasma source term for the Grad-Shafranov equation.
//!
//! Port of fusion_kernel.py `update_plasma_source_nonlinear()` (lines 118-171).
//! Computes J_phi using:
//!   J_phi = R · p'(ψ_norm) + (1/(μ₀R)) · FF'(ψ_norm)
//! where ψ_norm = (Ψ - Ψ_axis) / (Ψ_boundary - Ψ_axis).

use fusion_types::error::{FusionError, FusionResult};
use fusion_types::state::Grid2D;
use ndarray::Array2;

/// Default pressure/current mixing ratio (0=pure poloidal, 1=pure pressure).
/// Python line 158: `beta_mix = 0.5`
const DEFAULT_BETA_MIX: f64 = 0.5;

/// Minimum denominator for flux normalization (avoid div-by-zero).
/// Python line 130: `if abs(denom) < 1e-9: denom = 1e-9`
const MIN_FLUX_DENOMINATOR: f64 = 1e-9;

/// Minimum current integral threshold for renormalization.
/// Python line 165: `if abs(I_current) > 1e-9:`
const MIN_CURRENT_INTEGRAL: f64 = 1e-9;

/// mTanh profile parameters used by inverse reconstruction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfileParams {
    pub ped_top: f64,
    pub ped_width: f64,
    pub ped_height: f64,
    pub core_alpha: f64,
}

impl Default for ProfileParams {
    fn default() -> Self {
        Self {
            ped_top: 0.9,
            ped_width: 0.08,
            ped_height: 1.0,
            core_alpha: 0.2,
        }
    }
}

/// Context passed to profile-driven source update.
#[derive(Debug, Clone, Copy)]
pub struct SourceProfileContext<'a> {
    pub psi: &'a Array2<f64>,
    pub grid: &'a Grid2D,
    pub psi_axis: f64,
    pub psi_boundary: f64,
    pub mu0: f64,
    pub i_target: f64,
}

fn validate_source_inputs(
    psi: &Array2<f64>,
    grid: &Grid2D,
    psi_axis: f64,
    psi_boundary: f64,
    mu0: f64,
    i_target: f64,
) -> FusionResult<()> {
    if grid.nz == 0 || grid.nr == 0 {
        return Err(FusionError::ConfigError(
            "source update grid requires nz,nr >= 1".to_string(),
        ));
    }
    if !grid.dr.is_finite()
        || !grid.dz.is_finite()
        || grid.dr.abs() <= f64::EPSILON
        || grid.dz.abs() <= f64::EPSILON
    {
        return Err(FusionError::ConfigError(format!(
            "source update requires finite non-zero grid spacing, got dr={} dz={}",
            grid.dr, grid.dz
        )));
    }
    if psi.nrows() != grid.nz || psi.ncols() != grid.nr {
        return Err(FusionError::ConfigError(format!(
            "source update psi shape mismatch: expected ({}, {}), got ({}, {})",
            grid.nz,
            grid.nr,
            psi.nrows(),
            psi.ncols()
        )));
    }
    if grid.rr.nrows() != grid.nz || grid.rr.ncols() != grid.nr {
        return Err(FusionError::ConfigError(format!(
            "source update grid.rr shape mismatch: expected ({}, {}), got ({}, {})",
            grid.nz,
            grid.nr,
            grid.rr.nrows(),
            grid.rr.ncols()
        )));
    }
    if psi.iter().any(|v| !v.is_finite()) || grid.rr.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::ConfigError(
            "source update inputs must be finite".to_string(),
        ));
    }
    if !psi_axis.is_finite() || !psi_boundary.is_finite() {
        return Err(FusionError::ConfigError(
            "source update psi_axis/psi_boundary must be finite".to_string(),
        ));
    }
    if !mu0.is_finite() || mu0 <= 0.0 {
        return Err(FusionError::ConfigError(format!(
            "source update requires finite mu0 > 0, got {mu0}"
        )));
    }
    if !i_target.is_finite() {
        return Err(FusionError::ConfigError(
            "source update target current must be finite".to_string(),
        ));
    }
    let denom = psi_boundary - psi_axis;
    if !denom.is_finite() || denom.abs() < MIN_FLUX_DENOMINATOR {
        return Err(FusionError::ConfigError(format!(
            "source update flux denominator must satisfy |psi_boundary-psi_axis| >= {MIN_FLUX_DENOMINATOR}, got {}",
            denom
        )));
    }
    Ok(())
}

fn validate_profile_params(params: &ProfileParams, label: &str) -> FusionResult<()> {
    if !params.ped_top.is_finite() || params.ped_top <= 0.0 {
        return Err(FusionError::ConfigError(format!(
            "{label}.ped_top must be finite and > 0, got {}",
            params.ped_top
        )));
    }
    if !params.ped_width.is_finite() || params.ped_width <= 0.0 {
        return Err(FusionError::ConfigError(format!(
            "{label}.ped_width must be finite and > 0, got {}",
            params.ped_width
        )));
    }
    if !params.ped_height.is_finite() || !params.core_alpha.is_finite() {
        return Err(FusionError::ConfigError(format!(
            "{label}.ped_height/core_alpha must be finite"
        )));
    }
    Ok(())
}

/// mTanh profile:
/// f(psi_n) = 0.5 * h * (1 + tanh(y)) + alpha * core(psi_n)
/// y = (ped_top - psi_n) / ped_width
/// core(psi_n) = max(0, 1 - (psi_n/ped_top)^2)
pub fn mtanh_profile(psi_norm: f64, params: &ProfileParams) -> f64 {
    let w = params.ped_width.abs().max(1e-8);
    let ped_top = params.ped_top.abs().max(1e-8);
    let y = (params.ped_top - psi_norm) / w;
    let tanh_y = y.tanh();
    let core = (1.0 - (psi_norm / ped_top).powi(2)).max(0.0);
    0.5 * params.ped_height * (1.0 + tanh_y) + params.core_alpha * core
}

/// Analytical derivatives of mTanh profile with respect to:
/// [ped_height, ped_top, ped_width, core_alpha].
pub fn mtanh_profile_derivatives(psi_norm: f64, params: &ProfileParams) -> [f64; 4] {
    let w = params.ped_width.abs().max(1e-8);
    let ped_top = params.ped_top.abs().max(1e-8);
    let y = (params.ped_top - psi_norm) / w;
    let tanh_y = y.tanh();
    let sech2 = 1.0 - tanh_y * tanh_y;
    let core = (1.0 - (psi_norm / ped_top).powi(2)).max(0.0);

    let d_core_d_ped_top = if psi_norm.abs() < ped_top {
        2.0 * psi_norm.powi(2) / ped_top.powi(3)
    } else {
        0.0
    };

    let d_ped_height = 0.5 * (1.0 + tanh_y);
    let d_ped_top = 0.5 * params.ped_height * sech2 / w + params.core_alpha * d_core_d_ped_top;
    let d_ped_width = -0.5 * params.ped_height * sech2 * y / w;
    let d_core_alpha = core;

    [d_ped_height, d_ped_top, d_ped_width, d_core_alpha]
}

/// Update the toroidal current density J_phi using the full Grad-Shafranov source term.
///
/// Algorithm:
/// 1. Normalize flux: ψ_norm = (Ψ - Ψ_axis) / (Ψ_boundary - Ψ_axis)
/// 2. Define profile shape: f(ψ_norm) = (1 - ψ_norm) inside plasma (0 ≤ ψ_norm < 1)
/// 3. Pressure term: J_p = R · f(ψ_norm)
/// 4. Current term: J_f = (1/(μ₀R)) · f(ψ_norm)
/// 5. Mix: J_raw = β_mix · J_p + (1 - β_mix) · J_f
/// 6. Renormalize: scale J_raw so that ∫J_phi dR dZ = I_target
///
/// Returns the updated J_phi `[nz, nr]`.
pub fn update_plasma_source_nonlinear(
    psi: &Array2<f64>,
    grid: &Grid2D,
    psi_axis: f64,
    psi_boundary: f64,
    mu0: f64,
    i_target: f64,
) -> FusionResult<Array2<f64>> {
    validate_source_inputs(psi, grid, psi_axis, psi_boundary, mu0, i_target)?;
    let nz = grid.nz;
    let nr = grid.nr;

    // Normalize flux
    let denom = psi_boundary - psi_axis;

    let mut j_phi = Array2::zeros((nz, nr));

    for iz in 0..nz {
        for ir in 0..nr {
            let psi_norm = (psi[[iz, ir]] - psi_axis) / denom;
            if !psi_norm.is_finite() {
                return Err(FusionError::ConfigError(format!(
                    "source update produced non-finite psi_norm at ({iz}, {ir})"
                )));
            }

            // Only inside plasma (0 ≤ ψ_norm < 1)
            if (0.0..1.0).contains(&psi_norm) {
                let profile = 1.0 - psi_norm;

                let r = grid.rr[[iz, ir]];
                if r <= 0.0 {
                    return Err(FusionError::ConfigError(format!(
                        "source update requires R > 0 inside plasma at ({iz}, {ir}), got {r}"
                    )));
                }

                // Pressure-driven current (dominates at large R)
                let j_p = r * profile;

                // Poloidal field current (dominates at small R)
                let j_f = (1.0 / (mu0 * r)) * profile;
                if !j_p.is_finite() || !j_f.is_finite() {
                    return Err(FusionError::ConfigError(format!(
                        "source update produced non-finite current components at ({iz}, {ir})"
                    )));
                }

                // Mix
                j_phi[[iz, ir]] = DEFAULT_BETA_MIX * j_p + (1.0 - DEFAULT_BETA_MIX) * j_f;
            }
        }
    }

    // Renormalize to match target current
    let i_current: f64 = j_phi.iter().sum::<f64>() * grid.dr * grid.dz;
    if !i_current.is_finite() {
        return Err(FusionError::ConfigError(
            "source update current integral became non-finite".to_string(),
        ));
    }

    if i_current.abs() > MIN_CURRENT_INTEGRAL {
        let scale = i_target / i_current;
        if !scale.is_finite() {
            return Err(FusionError::ConfigError(
                "source update renormalization scale became non-finite".to_string(),
            ));
        }
        j_phi.mapv_inplace(|v| v * scale);
    } else {
        j_phi.fill(0.0);
    }

    if j_phi.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::ConfigError(
            "source update output contains non-finite values".to_string(),
        ));
    }
    Ok(j_phi)
}

/// Update toroidal current density using externally provided profile parameters.
///
/// This is the kernel-facing hook used by inverse reconstruction when profile
/// parameters are being estimated from measurements.
pub fn update_plasma_source_with_profiles(
    ctx: SourceProfileContext<'_>,
    params_p: &ProfileParams,
    params_ff: &ProfileParams,
) -> FusionResult<Array2<f64>> {
    let SourceProfileContext {
        psi,
        grid,
        psi_axis,
        psi_boundary,
        mu0,
        i_target,
    } = ctx;
    validate_source_inputs(psi, grid, psi_axis, psi_boundary, mu0, i_target)?;
    validate_profile_params(params_p, "params_p")?;
    validate_profile_params(params_ff, "params_ff")?;
    let nz = grid.nz;
    let nr = grid.nr;

    let denom = psi_boundary - psi_axis;

    let mut j_phi = Array2::zeros((nz, nr));
    for iz in 0..nz {
        for ir in 0..nr {
            let psi_norm = (psi[[iz, ir]] - psi_axis) / denom;
            if !psi_norm.is_finite() {
                return Err(FusionError::ConfigError(format!(
                    "profile source update produced non-finite psi_norm at ({iz}, {ir})"
                )));
            }
            if (0.0..1.0).contains(&psi_norm) {
                let r = grid.rr[[iz, ir]];
                if r <= 0.0 {
                    return Err(FusionError::ConfigError(format!(
                        "profile source update requires R > 0 inside plasma at ({iz}, {ir}), got {r}"
                    )));
                }
                let p_profile = mtanh_profile(psi_norm, params_p);
                let ff_profile = mtanh_profile(psi_norm, params_ff);
                if !p_profile.is_finite() || !ff_profile.is_finite() {
                    return Err(FusionError::ConfigError(format!(
                        "profile source update produced non-finite profile values at ({iz}, {ir})"
                    )));
                }

                let j_p = r * p_profile;
                let j_f = (1.0 / (mu0 * r)) * ff_profile;
                if !j_p.is_finite() || !j_f.is_finite() {
                    return Err(FusionError::ConfigError(format!(
                        "profile source update produced non-finite current components at ({iz}, {ir})"
                    )));
                }
                j_phi[[iz, ir]] = DEFAULT_BETA_MIX * j_p + (1.0 - DEFAULT_BETA_MIX) * j_f;
            }
        }
    }

    let i_current: f64 = j_phi.iter().sum::<f64>() * grid.dr * grid.dz;
    if !i_current.is_finite() {
        return Err(FusionError::ConfigError(
            "profile source update current integral became non-finite".to_string(),
        ));
    }
    if i_current.abs() > MIN_CURRENT_INTEGRAL {
        let scale = i_target / i_current;
        if !scale.is_finite() {
            return Err(FusionError::ConfigError(
                "profile source update renormalization scale became non-finite".to_string(),
            ));
        }
        j_phi.mapv_inplace(|v| v * scale);
    } else {
        j_phi.fill(0.0);
    }
    if j_phi.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::ConfigError(
            "profile source update output contains non-finite values".to_string(),
        ));
    }
    Ok(j_phi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_zero_outside_plasma() {
        let grid = Grid2D::new(16, 16, 1.0, 9.0, -5.0, 5.0);
        // Ψ everywhere = 0, axis = 1.0, boundary = 0.0
        // ψ_norm = (0 - 1) / (0 - 1) = 1.0 → outside plasma
        let psi = Array2::zeros((16, 16));
        let j = update_plasma_source_nonlinear(&psi, &grid, 1.0, 0.0, 1.0, 1e6)
            .expect("valid source-update inputs");

        // Everything should be zero (all ψ_norm = 1.0, exactly at boundary)
        let max_j = j.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
        assert!(max_j < 1e-15, "Should be zero outside plasma: {max_j}");
    }

    #[test]
    fn test_source_renormalization() {
        let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
        // Create Gaussian-like Ψ peaked at center
        let psi = Array2::from_shape_fn((33, 33), |(iz, ir)| {
            let r = grid.rr[[iz, ir]];
            let z = grid.zz[[iz, ir]];
            (-(((r - 5.0).powi(2) + z.powi(2)) / 4.0)).exp()
        });

        let psi_axis = 1.0; // peak
        let psi_boundary = 0.0; // edge
        let i_target = 15e6; // 15 MA

        let j = update_plasma_source_nonlinear(&psi, &grid, psi_axis, psi_boundary, 1.0, i_target)
            .expect("valid source-update inputs");

        // Check integral matches target
        let i_actual: f64 = j.iter().sum::<f64>() * grid.dr * grid.dz;
        let rel_error = ((i_actual - i_target) / i_target).abs();
        assert!(
            rel_error < 1e-10,
            "Current integral {i_actual} should match target {i_target}"
        );
    }

    #[test]
    fn test_mtanh_derivatives_match_finite_difference() {
        let params = ProfileParams {
            ped_top: 0.92,
            ped_width: 0.07,
            ped_height: 1.2,
            core_alpha: 0.3,
        };
        let psi = 0.35;
        let analytic = mtanh_profile_derivatives(psi, &params);
        let eps = 1e-6;

        let mut p = params;
        p.ped_height += eps;
        let fd_h = (mtanh_profile(psi, &p) - mtanh_profile(psi, &params)) / eps;

        p = params;
        p.ped_top += eps;
        let fd_top = (mtanh_profile(psi, &p) - mtanh_profile(psi, &params)) / eps;

        p = params;
        p.ped_width += eps;
        let fd_w = (mtanh_profile(psi, &p) - mtanh_profile(psi, &params)) / eps;

        p = params;
        p.core_alpha += eps;
        let fd_a = (mtanh_profile(psi, &p) - mtanh_profile(psi, &params)) / eps;

        let fd = [fd_h, fd_top, fd_w, fd_a];
        for i in 0..4 {
            let denom = fd[i].abs().max(1e-8);
            let rel = (analytic[i] - fd[i]).abs() / denom;
            assert!(
                rel < 1e-3,
                "Derivative mismatch at index {i}: analytic={}, fd={}, rel={}",
                analytic[i],
                fd[i],
                rel
            );
        }
    }

    #[test]
    fn test_source_with_profiles_finite() {
        let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
        let psi = Array2::from_shape_fn((33, 33), |(iz, ir)| {
            let r = grid.rr[[iz, ir]];
            let z = grid.zz[[iz, ir]];
            (-(((r - 5.0).powi(2) + z.powi(2)) / 4.0)).exp()
        });

        let params_p = ProfileParams {
            ped_top: 0.9,
            ped_width: 0.08,
            ped_height: 1.1,
            core_alpha: 0.25,
        };
        let params_ff = ProfileParams {
            ped_top: 0.85,
            ped_width: 0.06,
            ped_height: 0.95,
            core_alpha: 0.1,
        };

        let j = update_plasma_source_with_profiles(
            SourceProfileContext {
                psi: &psi,
                grid: &grid,
                psi_axis: 1.0,
                psi_boundary: 0.0,
                mu0: 1.0,
                i_target: 15e6,
            },
            &params_p,
            &params_ff,
        )
        .expect("valid profile-source-update inputs");
        assert!(
            j.iter().all(|v| v.is_finite()),
            "Profile source contains non-finite values"
        );
        let i_actual: f64 = j.iter().sum::<f64>() * grid.dr * grid.dz;
        let rel_error = ((i_actual - 15e6) / 15e6).abs();
        assert!(rel_error < 1e-10, "Current mismatch after renormalization");
    }

    #[test]
    fn test_source_rejects_invalid_runtime_inputs() {
        let mut grid = Grid2D::new(16, 16, 1.0, 9.0, -5.0, 5.0);
        let psi = Array2::zeros((16, 16));

        let err = update_plasma_source_nonlinear(&psi, &grid, 1.0, 1.0, 1.0, 1.0)
            .expect_err("degenerate flux normalization must fail");
        assert!(matches!(err, FusionError::ConfigError(_)));

        grid.rr[[3, 3]] = 0.0;
        let psi_inside = Array2::from_elem((16, 16), 0.5);
        let err = update_plasma_source_nonlinear(&psi_inside, &grid, 1.0, 0.0, 1.0, 1.0)
            .expect_err("non-positive radius inside plasma must fail");
        assert!(matches!(err, FusionError::ConfigError(_)));

        let params_bad = ProfileParams {
            ped_top: 0.9,
            ped_width: 0.0,
            ped_height: 1.0,
            core_alpha: 0.2,
        };
        let err = update_plasma_source_with_profiles(
            SourceProfileContext {
                psi: &Array2::from_elem((16, 16), 0.5),
                grid: &Grid2D::new(16, 16, 1.0, 9.0, -5.0, 5.0),
                psi_axis: 1.0,
                psi_boundary: 0.0,
                mu0: 1.0,
                i_target: 1.0,
            },
            &params_bad,
            &ProfileParams::default(),
        )
        .expect_err("invalid profile params must fail");
        assert!(matches!(err, FusionError::ConfigError(_)));
    }
}
