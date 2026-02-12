//! Nonlinear plasma source term for the Grad-Shafranov equation.
//!
//! Port of fusion_kernel.py `update_plasma_source_nonlinear()` (lines 118-171).
//! Computes J_phi using:
//!   J_phi = R · p'(ψ_norm) + (1/(μ₀R)) · FF'(ψ_norm)
//! where ψ_norm = (Ψ - Ψ_axis) / (Ψ_boundary - Ψ_axis).

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
) -> Array2<f64> {
    let nz = grid.nz;
    let nr = grid.nr;

    // Normalize flux
    let mut denom = psi_boundary - psi_axis;
    if denom.abs() < MIN_FLUX_DENOMINATOR {
        denom = MIN_FLUX_DENOMINATOR;
    }

    let mut j_phi = Array2::zeros((nz, nr));

    for iz in 0..nz {
        for ir in 0..nr {
            let psi_norm = (psi[[iz, ir]] - psi_axis) / denom;

            // Only inside plasma (0 ≤ ψ_norm < 1)
            if (0.0..1.0).contains(&psi_norm) {
                let profile = 1.0 - psi_norm;

                let r = grid.rr[[iz, ir]];

                // Pressure-driven current (dominates at large R)
                let j_p = r * profile;

                // Poloidal field current (dominates at small R)
                let j_f = (1.0 / (mu0 * r)) * profile;

                // Mix
                j_phi[[iz, ir]] = DEFAULT_BETA_MIX * j_p + (1.0 - DEFAULT_BETA_MIX) * j_f;
            }
        }
    }

    // Renormalize to match target current
    let i_current: f64 = j_phi.iter().sum::<f64>() * grid.dr * grid.dz;

    if i_current.abs() > MIN_CURRENT_INTEGRAL {
        let scale = i_target / i_current;
        j_phi.mapv_inplace(|v| v * scale);
    } else {
        j_phi.fill(0.0);
    }

    j_phi
}

/// Update toroidal current density using externally provided profile parameters.
///
/// This is the kernel-facing hook used by inverse reconstruction when profile
/// parameters are being estimated from measurements.
pub fn update_plasma_source_with_profiles(
    psi: &Array2<f64>,
    grid: &Grid2D,
    psi_axis: f64,
    psi_boundary: f64,
    mu0: f64,
    i_target: f64,
    params_p: &ProfileParams,
    params_ff: &ProfileParams,
) -> Array2<f64> {
    let nz = grid.nz;
    let nr = grid.nr;

    let mut denom = psi_boundary - psi_axis;
    if denom.abs() < MIN_FLUX_DENOMINATOR {
        denom = MIN_FLUX_DENOMINATOR;
    }

    let mut j_phi = Array2::zeros((nz, nr));
    for iz in 0..nz {
        for ir in 0..nr {
            let psi_norm = (psi[[iz, ir]] - psi_axis) / denom;
            if (0.0..1.0).contains(&psi_norm) {
                let r = grid.rr[[iz, ir]].abs().max(1e-9);
                let p_profile = mtanh_profile(psi_norm, params_p);
                let ff_profile = mtanh_profile(psi_norm, params_ff);

                let j_p = r * p_profile;
                let j_f = (1.0 / (mu0 * r)) * ff_profile;
                j_phi[[iz, ir]] = DEFAULT_BETA_MIX * j_p + (1.0 - DEFAULT_BETA_MIX) * j_f;
            }
        }
    }

    let i_current: f64 = j_phi.iter().sum::<f64>() * grid.dr * grid.dz;
    if i_current.abs() > MIN_CURRENT_INTEGRAL {
        let scale = i_target / i_current;
        j_phi.mapv_inplace(|v| v * scale);
    } else {
        j_phi.fill(0.0);
    }
    j_phi
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
        let j = update_plasma_source_nonlinear(&psi, &grid, 1.0, 0.0, 1.0, 1e6);

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

        let j = update_plasma_source_nonlinear(&psi, &grid, psi_axis, psi_boundary, 1.0, i_target);

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
            &psi, &grid, 1.0, 0.0, 1.0, 15e6, &params_p, &params_ff,
        );
        assert!(
            j.iter().all(|v| v.is_finite()),
            "Profile source contains non-finite values"
        );
        let i_actual: f64 = j.iter().sum::<f64>() * grid.dr * grid.dz;
        let rel_error = ((i_actual - 15e6) / 15e6).abs();
        assert!(rel_error < 1e-10, "Current mismatch after renormalization");
    }
}
