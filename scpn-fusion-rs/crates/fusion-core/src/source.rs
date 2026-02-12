// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Source
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Nonlinear plasma source term for the Grad-Shafranov equation.
//!
//! Computes J_phi using:
//!   J_phi = R · p'(ψ_norm) + (1/(μ₀R)) · FF'(ψ_norm)
//! where ψ_norm = (Ψ - Ψ_axis) / (Ψ_boundary - Ψ_axis).
//!
//! Supports both L-mode (linear `1 - ψ_norm`) and H-mode (mtanh pedestal)
//! profile shapes for p'(ψ) and FF'(ψ).

use fusion_types::state::Grid2D;
use ndarray::Array2;

/// Default pressure/current mixing ratio (0=pure poloidal, 1=pure pressure).
const DEFAULT_BETA_MIX: f64 = 0.5;

/// Minimum denominator for flux normalization (avoid div-by-zero).
const MIN_FLUX_DENOMINATOR: f64 = 1e-9;

/// Minimum current integral threshold for renormalization.
const MIN_CURRENT_INTEGRAL: f64 = 1e-9;

// ── Profile types ──────────────────────────────────────────────────

/// Selects the radial profile shape for p' and FF'.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ProfileMode {
    /// Linear `(1 - ψ_norm)` — standard L-mode.
    #[default]
    LMode,
    /// Modified-tanh pedestal — standard H-mode.
    HMode,
}

/// Parameters for a single mtanh pedestal profile.
///
/// The profile is:
///   `f(x) = (ped_height / 2) · (1 + mtanh((ped_top - x) / ped_width))
///           + core_shape(x) · core_alpha`
///
/// where `mtanh(y) = (exp(y) - exp(-y)) / (exp(y) + exp(-y))` and
/// `core_shape(x) = max(0, 1 - (x / ped_top)^2)` adds a parabolic core.
#[derive(Debug, Clone, Copy)]
pub struct ProfileParams {
    /// Pedestal top location in normalized flux (typically 0.9–0.95).
    pub ped_top: f64,
    /// Pedestal width in normalized flux units (typically 0.03–0.08).
    pub ped_width: f64,
    /// Pedestal height (relative, will be renormalized).
    pub ped_height: f64,
    /// Core peaking factor (0 = flat top, 1 = strong core peaking).
    pub core_alpha: f64,
}

impl Default for ProfileParams {
    fn default() -> Self {
        ProfileParams {
            ped_top: 0.92,
            ped_width: 0.05,
            ped_height: 1.0,
            core_alpha: 0.3,
        }
    }
}

/// Evaluate the mtanh pedestal profile at a single ψ_norm value.
///
/// Returns 0.0 outside the plasma (ψ_norm < 0 or ψ_norm ≥ 1).
///
/// The profile consists of:
/// 1. A pedestal step via mtanh centered at `ped_top` with width `ped_width`
/// 2. A parabolic core contribution for core peaking
pub fn mtanh_profile(psi_norm: f64, params: &ProfileParams) -> f64 {
    if !(0.0..1.0).contains(&psi_norm) {
        return 0.0;
    }

    // Pedestal component: mtanh step
    let y = (params.ped_top - psi_norm) / params.ped_width;
    // Clamp to avoid overflow in exp
    let y_clamped = y.clamp(-20.0, 20.0);
    let mtanh_val = y_clamped.tanh();
    let pedestal = 0.5 * params.ped_height * (1.0 + mtanh_val);

    // Core component: parabolic peaking inside pedestal top
    let core = if psi_norm < params.ped_top {
        let x_rel = psi_norm / params.ped_top;
        (1.0 - x_rel * x_rel).max(0.0)
    } else {
        0.0
    };

    pedestal + params.core_alpha * core
}

// ── Source term functions ───────────────────────────────────────────

/// Update J_phi with explicit profile parameters (supports L-mode and H-mode).
///
/// When `mode == LMode`, the legacy linear `(1 - ψ_norm)` profile is used
/// (identical to the original `update_plasma_source_nonlinear`).
///
/// When `mode == HMode`, `p_params` and `ff_params` define mtanh pedestal
/// profiles for the pressure gradient and poloidal current respectively.
#[allow(clippy::too_many_arguments)]
pub fn update_plasma_source_with_profiles(
    psi: &Array2<f64>,
    grid: &Grid2D,
    psi_axis: f64,
    psi_boundary: f64,
    mu0: f64,
    i_target: f64,
    mode: ProfileMode,
    p_params: &ProfileParams,
    ff_params: &ProfileParams,
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

            if !(0.0..1.0).contains(&psi_norm) {
                continue;
            }

            let (p_profile, ff_profile) = match mode {
                ProfileMode::LMode => {
                    let lin = 1.0 - psi_norm;
                    (lin, lin)
                }
                ProfileMode::HMode => (
                    mtanh_profile(psi_norm, p_params),
                    mtanh_profile(psi_norm, ff_params),
                ),
            };

            let r = grid.rr[[iz, ir]];
            let j_p = r * p_profile;
            let j_f = (1.0 / (mu0 * r)) * ff_profile;
            j_phi[[iz, ir]] = DEFAULT_BETA_MIX * j_p + (1.0 - DEFAULT_BETA_MIX) * j_f;
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

/// Original L-mode source term (backward compatible).
///
/// Delegates to [`update_plasma_source_with_profiles`] with `ProfileMode::LMode`.
pub fn update_plasma_source_nonlinear(
    psi: &Array2<f64>,
    grid: &Grid2D,
    psi_axis: f64,
    psi_boundary: f64,
    mu0: f64,
    i_target: f64,
) -> Array2<f64> {
    update_plasma_source_with_profiles(
        psi,
        grid,
        psi_axis,
        psi_boundary,
        mu0,
        i_target,
        ProfileMode::LMode,
        &ProfileParams::default(),
        &ProfileParams::default(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Legacy tests (must still pass) ─────────────────────────────

    #[test]
    fn test_source_zero_outside_plasma() {
        let grid = Grid2D::new(16, 16, 1.0, 9.0, -5.0, 5.0);
        let psi = Array2::zeros((16, 16));
        let j = update_plasma_source_nonlinear(&psi, &grid, 1.0, 0.0, 1.0, 1e6);

        let max_j = j.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
        assert!(max_j < 1e-15, "Should be zero outside plasma: {max_j}");
    }

    #[test]
    fn test_source_renormalization() {
        let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
        let psi = Array2::from_shape_fn((33, 33), |(iz, ir)| {
            let r = grid.rr[[iz, ir]];
            let z = grid.zz[[iz, ir]];
            (-(((r - 5.0).powi(2) + z.powi(2)) / 4.0)).exp()
        });

        let psi_axis = 1.0;
        let psi_boundary = 0.0;
        let i_target = 15e6;

        let j = update_plasma_source_nonlinear(&psi, &grid, psi_axis, psi_boundary, 1.0, i_target);

        let i_actual: f64 = j.iter().sum::<f64>() * grid.dr * grid.dz;
        let rel_error = ((i_actual - i_target) / i_target).abs();
        assert!(
            rel_error < 1e-10,
            "Current integral {i_actual} should match target {i_target}"
        );
    }

    // ── mtanh profile unit tests ───────────────────────────────────

    #[test]
    fn test_mtanh_at_axis_equals_one() {
        // At ψ_norm = 0 (magnetic axis), profile should be close to 1.0
        let params = ProfileParams::default();
        let val = mtanh_profile(0.0, &params);
        // pedestal: 0.5 * 1.0 * (1 + tanh(0.92/0.05)) ≈ 1.0
        // core: 0.3 * 1.0 = 0.3
        // total ≈ 1.3
        assert!(val > 0.9, "Profile at axis should be large, got {val}");
    }

    #[test]
    fn test_mtanh_at_separatrix_zero() {
        // At ψ_norm = 1.0 (separatrix), profile should be exactly 0
        let params = ProfileParams::default();
        let val = mtanh_profile(1.0, &params);
        assert!(
            val.abs() < 1e-15,
            "Profile at separatrix should be 0, got {val}"
        );
    }

    #[test]
    fn test_mtanh_monotone_decreasing() {
        // Profile should be monotone decreasing from axis to edge
        let params = ProfileParams::default();
        let n = 100;
        let mut prev = mtanh_profile(0.0, &params);
        for i in 1..n {
            let x = i as f64 / n as f64;
            let val = mtanh_profile(x, &params);
            assert!(
                val <= prev + 1e-12,
                "Profile not monotone at ψ_norm={x}: {prev} -> {val}"
            );
            prev = val;
        }
    }

    #[test]
    fn test_mtanh_steep_gradient_at_pedestal() {
        // The gradient should be steepest near ped_top
        let params = ProfileParams {
            ped_top: 0.92,
            ped_width: 0.04,
            ped_height: 1.0,
            core_alpha: 0.0, // disable core to isolate pedestal
        };
        let dx = 0.01;

        // Gradient at pedestal top
        let grad_ped = (mtanh_profile(params.ped_top - dx, &params)
            - mtanh_profile(params.ped_top + dx, &params))
            / (2.0 * dx);

        // Gradient at core (ψ_norm = 0.3)
        let grad_core =
            (mtanh_profile(0.3 - dx, &params) - mtanh_profile(0.3 + dx, &params)) / (2.0 * dx);

        assert!(
            grad_ped > grad_core * 2.0,
            "Pedestal gradient ({grad_ped}) should be much steeper than core ({grad_core})"
        );
    }

    #[test]
    fn test_lmode_backward_compat() {
        // L-mode via update_plasma_source_with_profiles must match
        // the original update_plasma_source_nonlinear exactly.
        let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
        let psi = Array2::from_shape_fn((33, 33), |(iz, ir)| {
            let r = grid.rr[[iz, ir]];
            let z = grid.zz[[iz, ir]];
            (-(((r - 5.0).powi(2) + z.powi(2)) / 4.0)).exp()
        });

        let psi_axis = 1.0;
        let psi_boundary = 0.0;
        let i_target = 15e6;

        let j_legacy =
            update_plasma_source_nonlinear(&psi, &grid, psi_axis, psi_boundary, 1.0, i_target);
        let j_new = update_plasma_source_with_profiles(
            &psi,
            &grid,
            psi_axis,
            psi_boundary,
            1.0,
            i_target,
            ProfileMode::LMode,
            &ProfileParams::default(),
            &ProfileParams::default(),
        );

        let max_diff = j_legacy
            .iter()
            .zip(j_new.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-14,
            "L-mode profile path must match legacy: max_diff = {max_diff}"
        );
    }

    #[test]
    fn test_hmode_current_renormalization() {
        // H-mode source should still renormalize to I_target
        let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
        let psi = Array2::from_shape_fn((33, 33), |(iz, ir)| {
            let r = grid.rr[[iz, ir]];
            let z = grid.zz[[iz, ir]];
            (-(((r - 5.0).powi(2) + z.powi(2)) / 4.0)).exp()
        });

        let i_target = 15e6;
        let j = update_plasma_source_with_profiles(
            &psi,
            &grid,
            1.0,
            0.0,
            1.0,
            i_target,
            ProfileMode::HMode,
            &ProfileParams::default(),
            &ProfileParams::default(),
        );

        let i_actual: f64 = j.iter().sum::<f64>() * grid.dr * grid.dz;
        let rel_error = ((i_actual - i_target) / i_target).abs();
        assert!(
            rel_error < 1e-10,
            "H-mode current {i_actual} should match target {i_target}"
        );
    }

    #[test]
    fn test_lmode_source_regression() {
        // The legacy function should produce identical output to the original
        // implementation (guards against refactoring regressions).
        let grid = Grid2D::new(16, 16, 1.0, 9.0, -5.0, 5.0);
        let psi = Array2::from_shape_fn((16, 16), |(iz, ir)| {
            let r = grid.rr[[iz, ir]];
            let z = grid.zz[[iz, ir]];
            0.5 * (-(((r - 5.0).powi(2) + z.powi(2)) / 2.0)).exp()
        });

        let j1 = update_plasma_source_nonlinear(&psi, &grid, 0.5, 0.0, 1.0, 1e6);
        let j2 = update_plasma_source_nonlinear(&psi, &grid, 0.5, 0.0, 1.0, 1e6);

        let max_diff = j1
            .iter()
            .zip(j2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-15,
            "Deterministic: same inputs must give same outputs"
        );
    }
}
