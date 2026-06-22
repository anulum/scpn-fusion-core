// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Analytic
//! Analytic Shafranov equilibrium solver.
//!
//! Port of `analytic_solver.py`.
//! Computes required vertical field and coil currents from plasma parameters.

use fusion_types::error::{FusionError, FusionResult};
use std::f64::consts::PI;

/// Permeability of free space [H/m].
const MU_0: f64 = 4.0e-7 * PI;

/// Default poloidal beta. Python: 0.5.
pub const BETA_P: f64 = 0.5;

/// Default internal inductance. Python: 0.8.
pub const LI: f64 = 0.8;

/// Result of Shafranov equilibrium calculation.
#[derive(Debug, Clone)]
pub struct ShafranovResult {
    /// Required vertical field [T].
    pub bv_required: f64,
    /// log term.
    pub term_log: f64,
    /// Physics term (beta + li/2 - 1.5).
    pub term_physics: f64,
}

/// Compute required vertical field from Shafranov formula.
///
/// `r_geo`: major radius [m], `a_min`: minor radius [m], `ip_ma`: plasma current [MA],
/// `beta_p`: poloidal beta, `li`: internal inductance.
///
/// Canonical contract shared with the NumPy tier
/// (`scpn_fusion.control.analytic_solver.shafranov_bv`): same formula, same
/// constants, same `> 0` domain on `r_geo`/`a_min`/`ip_ma`, so the returned
/// `bv_required` is bit-exact across both backends. Defaults `beta_p = 0.5`,
/// `li = 0.8` are exposed at the Python boundary via the pyfunction signature.
pub fn shafranov_bv(
    r_geo: f64,
    a_min: f64,
    ip_ma: f64,
    beta_p: f64,
    li: f64,
) -> FusionResult<ShafranovResult> {
    if !r_geo.is_finite() || r_geo <= 0.0 {
        return Err(FusionError::ConfigError(
            "analytic r_geo must be finite and > 0".to_string(),
        ));
    }
    if !a_min.is_finite() || a_min <= 0.0 {
        return Err(FusionError::ConfigError(
            "analytic a_min must be finite and > 0".to_string(),
        ));
    }
    if !ip_ma.is_finite() || ip_ma <= 0.0 {
        return Err(FusionError::ConfigError(
            "analytic ip_ma must be finite and > 0".to_string(),
        ));
    }
    if !beta_p.is_finite() || !li.is_finite() {
        return Err(FusionError::ConfigError(
            "analytic beta_p and li must be finite".to_string(),
        ));
    }

    let ip = ip_ma * 1e6; // Convert to Amps
    let term_log = (8.0 * r_geo / a_min).ln();
    let term_physics = beta_p + li / 2.0 - 1.5;
    let bv = -(MU_0 * ip) / (4.0 * PI * r_geo) * (term_log + term_physics);
    if !bv.is_finite() || !term_log.is_finite() {
        return Err(FusionError::ConfigError(
            "analytic Shafranov calculation produced non-finite result".to_string(),
        ));
    }

    Ok(ShafranovResult {
        bv_required: bv,
        term_log,
        term_physics,
    })
}

/// Solve for coil currents given Green's function efficiencies.
///
/// `green_func`: Bz contribution per coil per MA [T/MA].
/// `target_bv`: required vertical field [T].
/// `ridge_lambda`: Tikhonov regularisation added to `G·Gᵀ`; negative values are
/// clamped to zero (plain minimum norm).
///
/// Returns minimum-norm coil currents [MA]. Canonical contract shared with the
/// NumPy tier (`scpn_fusion.control.analytic_solver.solve_coil_currents`): the
/// direct closed form `Iᵢ = gᵢ·target_bv / (Σgⱼ² + λ)`, including the
/// `(Σg² + λ).max(1e-12)` ridge floor and the small-norm rejection in the
/// unregularised case. Agreement with the NumPy tier is tolerance-aware (not
/// bit-exact): the `Σg²` reduction is summed sequentially here but via `numpy.dot`
/// in NumPy, which can differ by a unit in the last place.
pub fn solve_coil_currents(
    green_func: &[f64],
    target_bv: f64,
    ridge_lambda: f64,
) -> FusionResult<Vec<f64>> {
    if green_func.is_empty() {
        return Err(FusionError::ConfigError(
            "analytic green_func must be non-empty".to_string(),
        ));
    }
    if green_func.iter().any(|g| !g.is_finite()) {
        return Err(FusionError::ConfigError(
            "analytic green_func must contain only finite values".to_string(),
        ));
    }
    if !target_bv.is_finite() {
        return Err(FusionError::ConfigError(
            "analytic target_bv must be finite".to_string(),
        ));
    }
    if !ridge_lambda.is_finite() {
        return Err(FusionError::ConfigError(
            "analytic ridge_lambda must be finite".to_string(),
        ));
    }

    // Minimum-norm solution for underdetermined G·I = target_bv
    // I = Gᵀ · (G·Gᵀ + λ)^{-1} · target_bv
    let lam = ridge_lambda.max(0.0);
    let ggt: f64 = green_func.iter().map(|g| g * g).sum();
    let denom = if lam > 0.0 {
        (ggt + lam).max(1e-12)
    } else {
        if ggt < 1e-20 {
            return Err(FusionError::ConfigError(
                "analytic green_func norm is too small for a stable solve".to_string(),
            ));
        }
        ggt
    };

    let scale = target_bv / denom;
    Ok(green_func.iter().map(|g| g * scale).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shafranov_bv_iter() {
        // ITER-like: R=6.2m, a=2.0m, Ip=15 MA
        let result = shafranov_bv(6.2, 2.0, 15.0, BETA_P, LI).expect("valid Shafranov inputs");
        assert!(
            result.bv_required.abs() > 0.0,
            "Bv should be nonzero: {}",
            result.bv_required
        );
        assert!(result.bv_required.is_finite());
        // Bv is negative (pointing downward for positive Ip)
        assert!(
            result.bv_required < 0.0,
            "Bv should be negative: {}",
            result.bv_required
        );
    }

    #[test]
    fn test_shafranov_scales_with_current() {
        let lo = shafranov_bv(6.2, 2.0, 10.0, BETA_P, LI).expect("valid Shafranov inputs");
        let hi = shafranov_bv(6.2, 2.0, 15.0, BETA_P, LI).expect("valid Shafranov inputs");
        assert!(
            hi.bv_required.abs() > lo.bv_required.abs(),
            "Higher Ip needs more Bv"
        );
    }

    #[test]
    fn test_coil_currents_solve() {
        let green = vec![0.01, 0.02, 0.015, 0.005, 0.01];
        let target_bv = -0.05; // Tesla
        let currents =
            solve_coil_currents(&green, target_bv, 0.0).expect("valid coil solve inputs");
        // Verify: sum(G_i * I_i) ≈ target_bv
        let bv_check: f64 = green.iter().zip(&currents).map(|(g, i)| g * i).sum();
        assert!(
            (bv_check - target_bv).abs() < 1e-10,
            "G·I should equal Bv: {bv_check} vs {target_bv}"
        );
    }

    #[test]
    fn test_coil_currents_ridge_shrinks_currents() {
        // Ridge regularisation increases the denominator, so |I| shrinks and the
        // projected field undershoots the target relative to the plain solve.
        let green = vec![0.01, 0.02, 0.015, 0.005, 0.01];
        let target_bv = -0.05;
        let plain = solve_coil_currents(&green, target_bv, 0.0).expect("valid plain solve");
        let ridged = solve_coil_currents(&green, target_bv, 1e-3).expect("valid ridge solve");
        let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            norm(&ridged) < norm(&plain),
            "ridge should shrink the current norm"
        );
        // Negative ridge clamps to zero, reproducing the plain solve bit-for-bit.
        let clamped = solve_coil_currents(&green, target_bv, -5.0).expect("clamped ridge solve");
        assert_eq!(clamped, plain);
    }

    #[test]
    fn test_coil_currents_minimum_norm() {
        let green = vec![1.0, 1.0];
        let target_bv = 1.0;
        let currents =
            solve_coil_currents(&green, target_bv, 0.0).expect("valid coil solve inputs");
        // Minimum norm: both currents should be 0.5
        assert!(
            (currents[0] - 0.5).abs() < 1e-10,
            "Min norm: I[0]=0.5: {}",
            currents[0]
        );
        assert!(
            (currents[1] - 0.5).abs() < 1e-10,
            "Min norm: I[1]=0.5: {}",
            currents[1]
        );
    }

    #[test]
    fn test_analytic_rejects_invalid_inputs() {
        assert!(shafranov_bv(0.0, 2.0, 15.0, BETA_P, LI).is_err());
        assert!(shafranov_bv(6.2, f64::NAN, 15.0, BETA_P, LI).is_err());
        // Ip <= 0 is rejected to match the NumPy canonical domain (Ip > 0).
        assert!(shafranov_bv(6.2, 2.0, 0.0, BETA_P, LI).is_err());
        assert!(shafranov_bv(6.2, 2.0, -15.0, BETA_P, LI).is_err());
        // Non-finite shaping parameters are rejected.
        assert!(shafranov_bv(6.2, 2.0, 15.0, f64::NAN, LI).is_err());
        assert!(shafranov_bv(6.2, 2.0, 15.0, BETA_P, f64::INFINITY).is_err());
        assert!(solve_coil_currents(&[], -0.05, 0.0).is_err());
        assert!(solve_coil_currents(&[0.0, 0.0], -0.05, 0.0).is_err());
        assert!(solve_coil_currents(&[0.01, f64::NAN], -0.05, 0.0).is_err());
        assert!(solve_coil_currents(&[0.01, 0.02], f64::INFINITY, 0.0).is_err());
        assert!(solve_coil_currents(&[0.01, 0.02], -0.05, f64::NAN).is_err());
        // A zero-norm Green's vector is solvable once a positive ridge is present.
        assert!(solve_coil_currents(&[0.0, 0.0], -0.05, 1e-6).is_ok());
    }

    #[test]
    fn test_shafranov_bv_responds_to_shaping_parameters() {
        // term_physics = beta_p + li/2 - 1.5, so larger beta_p or li raises
        // (term_log + term_physics) and the magnitude of the (negative) Bv.
        let base = shafranov_bv(6.2, 2.0, 15.0, 0.5, 0.8).expect("valid inputs");
        let high_beta = shafranov_bv(6.2, 2.0, 15.0, 0.9, 0.8).expect("valid inputs");
        let high_li = shafranov_bv(6.2, 2.0, 15.0, 0.5, 1.2).expect("valid inputs");
        assert!(high_beta.bv_required.abs() > base.bv_required.abs());
        assert!(high_li.bv_required.abs() > base.bv_required.abs());
        assert!((base.term_physics - (0.5 + 0.8 / 2.0 - 1.5)).abs() < 1e-12);
    }
}
