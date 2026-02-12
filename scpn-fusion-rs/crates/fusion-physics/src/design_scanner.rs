// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Design Scanner
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Monte Carlo global design space scanner.
//!
//! Port of `global_design_scanner.py`.
//! Evaluates reactor designs using scaling laws and finds Pareto frontier.

use rand::Rng;
use std::f64::consts::PI;

/// Auxiliary power [MW]. Python: P_aux=50.
const P_AUX_MW: f64 = 50.0;

/// Fusion power calibration constant. Python: C_fus=0.20.
const C_FUS: f64 = 0.20;

/// Minimum safety factor. Python: q95_min=3.0.
const Q95_MIN: f64 = 3.0;

/// Maximum viable wall load [MW/m²].
const MAX_WALL_LOAD: f64 = 5.0;

/// Minimum viable engineering Q.
const MIN_Q_VIABLE: f64 = 2.0;

/// A reactor design evaluation result.
#[derive(Debug, Clone)]
pub struct DesignResult {
    pub r_major: f64,
    pub b_field: f64,
    pub i_plasma: f64,
    pub p_fusion: f64,
    pub q_engineering: f64,
    pub wall_load: f64,
    pub div_load: f64,
    pub cost: f64,
}

/// Evaluate a single reactor design point using scaling laws.
pub fn evaluate_design(r: f64, b: f64, i_p: f64) -> DesignResult {
    let a = r / 3.0;
    let volume = 2.0 * PI * PI * r * a * a;
    let surface = 4.0 * PI * PI * r * a;

    // Troyon beta limit
    let beta = 2.5 * i_p / (a * b) / 100.0;

    // Fusion power from pressure scaling
    let mu0 = 4e-7 * PI;
    let p_max = beta * b * b / (2.0 * mu0);
    let p_fus = C_FUS * volume * (p_max / 1e6).powi(2);

    // Heat loads
    let wall_load = 0.8 * p_fus / surface;
    let lambda_q_m = 0.63e-3 * b.powf(-1.19);
    let p_sol = 0.3 * p_fus;
    let div_load = p_sol / (2.0 * PI * r * lambda_q_m * 20.0);

    let q_eng = p_fus / P_AUX_MW;
    let cost = r.powi(3) * b;

    DesignResult {
        r_major: r,
        b_field: b,
        i_plasma: i_p,
        p_fusion: p_fus,
        q_engineering: q_eng,
        wall_load,
        div_load,
        cost,
    }
}

/// Run Monte Carlo design space scan with rejection sampling.
pub fn run_scan(n_samples: usize) -> Vec<DesignResult> {
    let mut rng = rand::thread_rng();
    let mut results = Vec::with_capacity(n_samples);

    let mut attempts = 0;
    while results.len() < n_samples && attempts < n_samples * 10 {
        attempts += 1;

        let r = rng.gen_range(2.0..9.0);
        let b = rng.gen_range(4.0..12.0);
        let i_p = rng.gen_range(5.0..25.0);

        // Safety factor constraint: q95 > 3.0
        let a = r / 3.0;
        let q95 = 5.0 * a * a * b / (r * i_p) * 2.0;
        if q95 < Q95_MIN {
            continue;
        }

        results.push(evaluate_design(r, b, i_p));
    }

    results
}

/// Extract Pareto frontier from design results.
///
/// Viable: Q > 2.0 AND wall_load < 5.0.
/// Pareto-optimal: no other viable design has both lower cost and higher Q.
pub fn find_pareto_frontier(designs: &[DesignResult]) -> Vec<DesignResult> {
    let viable: Vec<&DesignResult> = designs
        .iter()
        .filter(|d| d.q_engineering > MIN_Q_VIABLE && d.wall_load < MAX_WALL_LOAD)
        .collect();

    let mut pareto = Vec::new();
    for d in &viable {
        let dominated = viable
            .iter()
            .any(|other| other.cost < d.cost && other.q_engineering > d.q_engineering);
        if !dominated {
            pareto.push((*d).clone());
        }
    }

    pareto
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_design_finite() {
        let d = evaluate_design(6.0, 5.3, 15.0);
        assert!(d.p_fusion.is_finite());
        assert!(d.q_engineering.is_finite());
        assert!(d.cost > 0.0);
    }

    #[test]
    fn test_scan_produces_results() {
        let results = run_scan(500);
        assert!(!results.is_empty(), "Scan should produce valid designs");
    }

    #[test]
    fn test_pareto_nonempty() {
        let results = run_scan(2000);
        let pareto = find_pareto_frontier(&results);
        assert!(!pareto.is_empty(), "Pareto frontier should be non-empty");
    }

    #[test]
    fn test_pareto_all_viable() {
        let results = run_scan(2000);
        let pareto = find_pareto_frontier(&results);
        for d in &pareto {
            assert!(d.q_engineering > MIN_Q_VIABLE);
            assert!(d.wall_load < MAX_WALL_LOAD);
        }
    }
}
