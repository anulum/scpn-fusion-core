// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Compact Optimizer
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Compact reactor parametric optimizer.
//!
//! Port of `compact_reactor_optimizer.py`.
//! Grid search over (R, B0, aspect_ratio) to find smallest viable reactor.

use std::f64::consts::PI;

/// Critical current density base [MA/m²]. Python: J_crit_base=1500.
const J_CRIT_BASE: f64 = 1500.0;

/// Maximum coil field [T]. Python: B_max_coil=30.
const B_MAX_COIL: f64 = 30.0;

/// Shield thickness [m]. Python: 0.10.
const D_SHIELD: f64 = 0.10;

/// Gap thickness [m]. Python: 0.02.
const D_GAP: f64 = 0.02;

/// Coil radial build [m]. Python: 0.2.
const D_COIL: f64 = 0.2;

/// Elongation. Python: kappa=2.0.
const KAPPA: f64 = 2.0;

/// Auxiliary power [MW]. Python: P_aux=50.
const P_AUX_MW: f64 = 50.0;

/// Max wall load [MW/m²].
const MAX_WALL_LOAD: f64 = 5.0;

/// Max divertor load [MW/m²] (TEMHD liquid metal target).
const MAX_DIV_LOAD: f64 = 100.0;

/// A viable reactor design point.
#[derive(Debug, Clone)]
pub struct DesignPoint {
    pub r_major: f64,
    pub a_minor: f64,
    pub b0: f64,
    pub b_coil: f64,
    pub p_fusion_mw: f64,
    pub volume: f64,
    pub i_plasma_ma: f64,
    pub q_div_mw_m2: f64,
    pub q_wall_mw_m2: f64,
    pub q_engineering: f64,
}

/// Simplified plasma physics scaling model.
///
/// Returns (P_fusion [MW], I_plasma [MA], Volume [m³]).
pub fn plasma_physics_model(r: f64, a: f64, b0: f64) -> (f64, f64, f64) {
    let volume = 2.0 * PI * r * PI * a * a;
    // Plasma current from aspect ratio and elongation
    let i_p = 5.0 * a * a * b0 * (1.0 + KAPPA * KAPPA) / (2.0 * 3.0 * r);
    // Troyon beta limit: β = β_N · Ip / (a·B) with β_N = 4
    let beta = 4.0 * (i_p / (a * b0)) / 100.0;
    let mu0 = 4e-7 * PI;
    let pressure = beta * (b0 * b0 / (2.0 * mu0));
    // Empirical scaling calibrated to ITER (~500 MW at R=6.2, B=5.3)
    let p_fus_density = 2.0 * (pressure / 1e6).powi(2);
    let p_fusion = p_fus_density * volume;
    (p_fusion, i_p, volume)
}

/// Check radial build engineering constraints.
///
/// Returns (feasible, B_coil [T]).
pub fn radial_build_constraints(r: f64, a: f64, b0: f64) -> (bool, f64) {
    let r_coil = r - a - D_SHIELD - D_GAP;
    if r_coil <= 0.0 {
        return (false, f64::INFINITY);
    }
    let b_coil = b0 * r / r_coil;
    if b_coil > B_MAX_COIL {
        return (false, b_coil);
    }
    // Current density check
    let coil_area = D_COIL * 2.0 * PI * r_coil;
    let mu0 = 4e-7 * PI;
    let i_total = b0 * 2.0 * PI * r / mu0;
    let j_real = i_total / (coil_area * 1e6); // MA/m²
    let j_limit = J_CRIT_BASE * (20.0 / b_coil);

    (j_real < j_limit, b_coil)
}

/// Find minimum-size reactor meeting target fusion power.
///
/// Scans R=0.3-5.0m, B0=5-20T, A in {2.0, 2.5, 3.0}.
pub fn find_minimum_reactor(target_power_mw: f64) -> Option<DesignPoint> {
    let aspect_ratios = [2.0, 2.5, 3.0];
    let n_r = 100;
    let n_b = 30;
    let mut best: Option<DesignPoint> = None;

    for &aspect in &aspect_ratios {
        for ir in 0..n_r {
            let r = 1.0 + (8.0 - 1.0) * (ir as f64) / (n_r as f64 - 1.0);
            let a = r / aspect;

            for ib in 0..n_b {
                let b0 = 5.0 + (20.0 - 5.0) * (ib as f64) / (n_b as f64 - 1.0);

                let (p_fus, i_p, vol) = plasma_physics_model(r, a, b0);
                if p_fus < target_power_mw {
                    continue;
                }

                let (feasible, b_coil) = radial_build_constraints(r, a, b0);
                if !feasible {
                    continue;
                }

                // Heat loads
                let surface = 4.0 * PI * PI * r * a;
                let q_wall = 0.8 * p_fus / surface;
                if q_wall > MAX_WALL_LOAD {
                    continue;
                }

                // Simplified divertor: SOL power spread over strike area
                let p_sol = 0.3 * p_fus;
                let strike_area = 2.0 * PI * r * 0.1 * 20.0; // wetted area
                let q_div = p_sol / strike_area;

                let q_eng = p_fus / P_AUX_MW;

                let design = DesignPoint {
                    r_major: r,
                    a_minor: a,
                    b0,
                    b_coil,
                    p_fusion_mw: p_fus,
                    volume: vol,
                    i_plasma_ma: i_p,
                    q_div_mw_m2: q_div,
                    q_wall_mw_m2: q_wall,
                    q_engineering: q_eng,
                };

                if best.as_ref().is_none_or(|b| r < b.r_major) {
                    best = Some(design);
                }
            }
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plasma_model_positive() {
        let (p_fus, i_p, vol) = plasma_physics_model(3.0, 1.0, 10.0);
        assert!(p_fus > 0.0, "Fusion power should be positive: {p_fus}");
        assert!(i_p > 0.0, "Plasma current should be positive: {i_p}");
        assert!(vol > 0.0, "Volume should be positive: {vol}");
    }

    #[test]
    fn test_constraints_small_infeasible() {
        let (ok, _) = radial_build_constraints(0.3, 0.1, 20.0);
        assert!(!ok, "Tiny reactor at 20T should be infeasible");
    }

    #[test]
    fn test_find_reactor_q_above_2() {
        let design = find_minimum_reactor(100.0);
        assert!(design.is_some(), "Should find a viable reactor");
        let d = design.unwrap();
        assert!(
            d.q_engineering > 2.0,
            "Q should be > 2: {}",
            d.q_engineering
        );
    }
}
