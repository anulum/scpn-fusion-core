// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Analytic
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Analytic Shafranov equilibrium solver.
//!
//! Port of `analytic_solver.py`.
//! Computes required vertical field and coil currents from plasma parameters.

use fusion_types::error::{FusionError, FusionResult};
use std::f64::consts::PI;

/// Permeability of free space [H/m].
const MU_0: f64 = 4.0e-7 * PI;

/// Default poloidal beta. Python: 0.5.
const BETA_P: f64 = 0.5;

/// Default internal inductance. Python: 0.8.
const LI: f64 = 0.8;

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
/// `r_geo`: major radius [m], `a_min`: minor radius [m], `ip_ma`: plasma current [MA].
pub fn shafranov_bv(r_geo: f64, a_min: f64, ip_ma: f64) -> FusionResult<ShafranovResult> {
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
    if !ip_ma.is_finite() {
        return Err(FusionError::ConfigError(
            "analytic ip_ma must be finite".to_string(),
        ));
    }

    let ip = ip_ma * 1e6; // Convert to Amps
    let term_log = (8.0 * r_geo / a_min).ln();
    let term_physics = BETA_P + LI / 2.0 - 1.5;
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
///
/// Returns minimum-norm coil currents [MA].
pub fn solve_coil_currents(green_func: &[f64], target_bv: f64) -> FusionResult<Vec<f64>> {
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

    // Minimum-norm solution for underdetermined G·I = target_bv
    // I = G^T · (G·G^T)^{-1} · target_bv
    let ggt: f64 = green_func.iter().map(|g| g * g).sum();

    if ggt.abs() < 1e-20 {
        return Err(FusionError::ConfigError(
            "analytic green_func norm is too small for a stable solve".to_string(),
        ));
    }

    let scale = target_bv / ggt;
    Ok(green_func.iter().map(|g| g * scale).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shafranov_bv_iter() {
        // ITER-like: R=6.2m, a=2.0m, Ip=15 MA
        let result = shafranov_bv(6.2, 2.0, 15.0).expect("valid Shafranov inputs");
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
        let lo = shafranov_bv(6.2, 2.0, 10.0).expect("valid Shafranov inputs");
        let hi = shafranov_bv(6.2, 2.0, 15.0).expect("valid Shafranov inputs");
        assert!(
            hi.bv_required.abs() > lo.bv_required.abs(),
            "Higher Ip needs more Bv"
        );
    }

    #[test]
    fn test_coil_currents_solve() {
        let green = vec![0.01, 0.02, 0.015, 0.005, 0.01];
        let target_bv = -0.05; // Tesla
        let currents = solve_coil_currents(&green, target_bv).expect("valid coil solve inputs");
        // Verify: sum(G_i * I_i) ≈ target_bv
        let bv_check: f64 = green.iter().zip(&currents).map(|(g, i)| g * i).sum();
        assert!(
            (bv_check - target_bv).abs() < 1e-10,
            "G·I should equal Bv: {bv_check} vs {target_bv}"
        );
    }

    #[test]
    fn test_coil_currents_minimum_norm() {
        let green = vec![1.0, 1.0];
        let target_bv = 1.0;
        let currents = solve_coil_currents(&green, target_bv).expect("valid coil solve inputs");
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
        assert!(shafranov_bv(0.0, 2.0, 15.0).is_err());
        assert!(shafranov_bv(6.2, f64::NAN, 15.0).is_err());
        assert!(solve_coil_currents(&[], -0.05).is_err());
        assert!(solve_coil_currents(&[0.0, 0.0], -0.05).is_err());
        assert!(solve_coil_currents(&[0.01, f64::NAN], -0.05).is_err());
        assert!(solve_coil_currents(&[0.01, 0.02], f64::INFINITY).is_err());
    }
}
