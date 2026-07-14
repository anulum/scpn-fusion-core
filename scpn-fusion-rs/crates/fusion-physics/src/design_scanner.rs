// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Design Scanner
//! Monte Carlo global design space scanner.
//!
//! Numerical twin of the Python `GlobalDesignExplorer` (`global_design_scanner.py`).
//! `evaluate_design` reproduces the Python physics-scaling surrogate to floating
//! point tolerance, including the HEAT-ML magnetic-shadow divertor attenuation
//! (`heat_ml_shadow_surrogate.py`). The shadow surrogate is a compact ridge model;
//! its trained weights are a deterministic function of the production fit
//! (`fit_synthetic(seed=42, samples=1536)` invoked in `GlobalDesignExplorer.__init__`)
//! and are frozen into `SHADOW_WEIGHTS` below. Because the fit only depends on the
//! seed/sample count (both fixed in the production explorer), embedding the solved
//! weights is exact — no RNG reproduction is needed. Regenerate `SHADOW_WEIGHTS`
//! only if that seed/sample count changes. `run_scan` samples with an independent
//! RNG, so scan-level output is not point-for-point identical to the Python scan
//! (both are valid Monte Carlo draws); `evaluate_design` is the deterministic,
//! parity-verified kernel. See `docs/ARCHITECTURE.md` section 3.2.

use rand::Rng;
use std::f64::consts::PI;

/// Auxiliary power [MW]. Python: `P_aux = 50.0`.
const P_AUX_MW: f64 = 50.0;

/// Fusion power calibration constant. Python: `C_fus = 2.5e-11`.
const C_FUS: f64 = 2.5e-11;

/// Minimum safety factor for the scan rejection sampler. Python: `q95_min = 3.0`.
const Q95_MIN: f64 = 3.0;

/// Maximum viable neutron wall load [MW/m²]. Python `analyze_pareto`: `Wall_Load < 5.0`.
const MAX_WALL_LOAD: f64 = 5.0;

/// Minimum viable engineering Q. Python `analyze_pareto`: `Q > 2.0`.
const MIN_Q_VIABLE: f64 = 2.0;

/// Divertor magnetic-shadow cap [MW/m²]. Python: `divertor_flux_cap_mw_m2 = 45.0`.
const DIVERTOR_FLUX_CAP_MW_M2: f64 = 45.0;

/// Effective charge cap. Python: `zeff_cap = 0.4`.
const ZEFF_CAP: f64 = 0.4;

/// HTS peak-field cap [T]. Python: `hts_peak_cap_t = 21.0`.
const HTS_PEAK_CAP_T: f64 = 21.0;

/// Frozen HEAT-ML shadow-surrogate ridge weights (15-dim feature map).
///
/// Solved once from `HeatMLShadowSurrogate.fit_synthetic(seed=42, samples=1536)`,
/// the production configuration built in `GlobalDesignExplorer.__init__`. The fit
/// is `w = (Φᵀ Φ + ridge·I)⁻¹ Φᵀ y` with `ridge = 1e-4` over the deterministic
/// synthetic reference dataset, so these values are exact for that seed/sample
/// count. Regenerate if the production fit configuration changes.
const SHADOW_WEIGHTS: [f64; 15] = [
    0.05697139161198549,
    -0.00024135033178771873,
    -0.0023352533875437044,
    -6.16945482916233e-5,
    -0.0004912129963549095,
    0.09791074988882432,
    0.05876465034841261,
    0.0001259649214514708,
    0.00036217007855476576,
    0.0011767645885520982,
    -0.0005420700082272534,
    0.07942651795053192,
    -0.18099692837559578,
    0.16090321661006354,
    0.2296279168453285,
];

/// A reactor design evaluation result. Mirrors the Python `evaluate_design` dict.
#[derive(Debug, Clone)]
pub struct DesignResult {
    pub r_major: f64,
    pub b_field: f64,
    pub i_plasma: f64,
    pub p_fusion: f64,
    pub q_engineering: f64,
    pub wall_load: f64,
    pub div_load_baseline: f64,
    pub shadow_fraction: f64,
    pub div_load_optimized: f64,
    pub b_peak_hts_t: f64,
    pub zeff_est: f64,
    pub constraint_ok: bool,
    pub beta_n_eff: f64,
    pub cost: f64,
}

/// Effective normalised beta after the fixed H-mode shaping correction.
///
/// Python: `kappa, delta = 1.7, 0.33`; `beta_N_nominal = 2.8`;
/// `beta_shape_gain = 1 + 0.18·(kappa-1.5) + 0.08·delta`;
/// `beta_N_eff = clip(beta_N_nominal·beta_shape_gain, 2.0, 4.2)`.
/// The shaping inputs are constant, so this evaluates to a fixed 2.974_72.
fn beta_n_eff() -> f64 {
    let kappa = 1.7_f64;
    let delta = 0.33_f64;
    let beta_n_nominal = 2.8_f64;
    let beta_shape_gain = 1.0 + 0.18 * (kappa - 1.5) + 0.08 * delta;
    (beta_n_nominal * beta_shape_gain).clamp(2.0, 4.2)
}

/// Build the 15-dim HEAT-ML shadow feature map for one divertor feature row.
///
/// Mirrors `HeatMLShadowSurrogate._feature_map`. Feature order:
/// `[1, r, b_pol, p_sol, fx, kappa, delta, xpt_z, b_pol·fx, p_sol/max(fx,1e-6),`
/// `kappa·delta, exp(-(xpt_z+1.7)²/0.30), tanh(0.02(p_sol-70)),`
/// `tanh(0.20(fx-10)), tanh(0.65(b_pol-1.3))]`.
fn shadow_feature_map(
    r: f64,
    b_pol: f64,
    p_sol: f64,
    fx: f64,
    kappa: f64,
    delta: f64,
    xpt_z: f64,
) -> [f64; 15] {
    [
        1.0,
        r,
        b_pol,
        p_sol,
        fx,
        kappa,
        delta,
        xpt_z,
        b_pol * fx,
        p_sol / fx.max(1e-6),
        kappa * delta,
        (-((xpt_z + 1.7).powi(2)) / 0.30).exp(),
        (0.02 * (p_sol - 70.0)).tanh(),
        (0.20 * (fx - 10.0)).tanh(),
        (0.65 * (b_pol - 1.3)).tanh(),
    ]
}

/// Predict the clipped magnetic-shadow fraction for one feature row.
///
/// Mirrors `HeatMLShadowSurrogate.predict_shadow_fraction`:
/// `clip(Φ·w, 0.0, 0.85)`.
fn predict_shadow_fraction(features: &[f64; 15]) -> f64 {
    let dot: f64 = features
        .iter()
        .zip(SHADOW_WEIGHTS.iter())
        .map(|(phi, w)| phi * w)
        .sum();
    dot.clamp(0.0, 0.85)
}

/// Evaluate a single reactor design point using the physics-scaling surrogate,
/// with explicit engineering-constraint caps.
///
/// Numerical twin of `GlobalDesignExplorer.evaluate_design(R_maj, B_field, I_plasma)`
/// under the instance caps `divertor_flux_cap_mw_m2`, `zeff_cap`, `hts_peak_cap_t`.
pub fn evaluate_design_with_caps(
    r: f64,
    b: f64,
    i_p: f64,
    divertor_flux_cap_mw_m2: f64,
    zeff_cap: f64,
    hts_peak_cap_t: f64,
) -> DesignResult {
    let a_min = r / 3.0;
    let vol = 2.0 * PI * PI * r * a_min * a_min;

    let beta_n = beta_n_eff();
    let i_n = i_p / (a_min * b);
    let beta_limit_pct = beta_n * i_n;

    let mu0 = 4.0 * PI * 1e-7;
    let max_pressure = (beta_limit_pct / 100.0) * (b * b) / (2.0 * mu0);
    let p_fus = C_FUS * vol * max_pressure * max_pressure;

    let surface = 4.0 * PI * PI * r * a_min;
    let wall_load = 0.8 * p_fus / surface;

    // Divertor baseline (Eich scaling with compact-device calibration).
    let lambda_q = 0.63 * b.powf(-1.19); // mm
    let p_sol = 0.2 * p_fus + 50.0; // alpha + aux
    let expansion_factor = 12.0 + 0.6 * b;
    let div_load_baseline = (p_sol / (2.0 * PI * r * lambda_q * 1e-3) / expansion_factor) * 1e-4;

    // HEAT-ML magnetic-shadow attenuation (GAI-03).
    let b_pol_equiv = (0.22 * b).max(0.4);
    let features = shadow_feature_map(r, b_pol_equiv, p_sol, 10.0, 1.65, 0.35, -1.8);
    let shadow_fraction = predict_shadow_fraction(&features);
    let atten = 1.0 - 0.58 * shadow_fraction;
    let div_load_optimized = (div_load_baseline * atten).max(1e-6);

    let b_peak_hts_t = 1.72 * b + 0.6;
    let zeff_est =
        (0.18 + 0.0035 * div_load_optimized + 0.015 * (1.6 - r).max(0.0)).clamp(0.15, 0.8);
    let constraint_ok = div_load_optimized <= divertor_flux_cap_mw_m2
        && zeff_est <= zeff_cap
        && b_peak_hts_t <= hts_peak_cap_t;

    let q_eng = p_fus / P_AUX_MW;
    let cost = r.powi(3) * b;

    DesignResult {
        r_major: r,
        b_field: b,
        i_plasma: i_p,
        p_fusion: p_fus,
        q_engineering: q_eng,
        wall_load,
        div_load_baseline,
        shadow_fraction,
        div_load_optimized,
        b_peak_hts_t,
        zeff_est,
        constraint_ok,
        beta_n_eff: beta_n,
        cost,
    }
}

/// Evaluate a single reactor design point using the default engineering caps.
///
/// Convenience twin of `GlobalDesignExplorer.evaluate_design` at the default
/// `divertor_flux_cap_mw_m2 = 45.0`, `zeff_cap = 0.4`, `hts_peak_cap_t = 21.0`.
pub fn evaluate_design(r: f64, b: f64, i_p: f64) -> DesignResult {
    evaluate_design_with_caps(r, b, i_p, DIVERTOR_FLUX_CAP_MW_M2, ZEFF_CAP, HTS_PEAK_CAP_T)
}

/// Run a Monte Carlo design space scan with q95 rejection sampling.
///
/// The RNG is independent of the Python scan, so the sampled points differ; each
/// evaluated point is nonetheless computed by the parity-verified `evaluate_design`.
pub fn run_scan(n_samples: usize) -> Vec<DesignResult> {
    let mut rng = rand::thread_rng();
    let mut results = Vec::with_capacity(n_samples);

    let mut attempts = 0;
    while results.len() < n_samples && attempts < n_samples * 10 {
        attempts += 1;

        let r = rng.gen_range(2.0..9.0);
        let b = rng.gen_range(4.0..12.0);
        let i_p = rng.gen_range(5.0..25.0);

        // Safety factor constraint: q95 > 3.0.
        let a = r / 3.0;
        let q95 = 5.0 * a * a * b / (r * i_p) * 2.0;
        if q95 < Q95_MIN {
            continue;
        }

        results.push(evaluate_design(r, b, i_p));
    }

    results
}

/// Extract the Pareto frontier from design results.
///
/// Viable (Python `analyze_pareto`): `Q > 2.0` AND `wall_load < 5.0` AND `constraint_ok`.
/// Pareto-optimal: no other viable design has both lower cost and higher Q.
pub fn find_pareto_frontier(designs: &[DesignResult]) -> Vec<DesignResult> {
    let viable: Vec<&DesignResult> = designs
        .iter()
        .filter(|d| {
            d.q_engineering > MIN_Q_VIABLE && d.wall_load < MAX_WALL_LOAD && d.constraint_ok
        })
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

    /// Golden parity anchor against the Python `GlobalDesignExplorer.evaluate_design`.
    #[test]
    fn test_evaluate_design_python_parity() {
        let d = evaluate_design(6.0, 5.3, 15.0);
        // Golden values from PYTHONPATH=src python -c
        // "GlobalDesignExplorer('dummy').evaluate_design(6.0, 5.3, 15.0)".
        assert!((d.p_fusion - 2621.6078257152008).abs() < 1e-6);
        assert!((d.q_engineering - 52.43215651430401).abs() < 1e-9);
        assert!((d.wall_load - 4.427073465791328).abs() < 1e-9);
        assert!((d.div_load_baseline - 1.1590441223233476).abs() < 1e-9);
        assert!((d.shadow_fraction - 0.1417188706819292).abs() < 1e-12);
        assert!((d.div_load_optimized - 1.063774236353356).abs() < 1e-9);
        assert!((d.b_peak_hts_t - 9.716).abs() < 1e-9);
        assert!((d.zeff_est - 0.18372320982723675).abs() < 1e-12);
        assert!((d.beta_n_eff - 2.97472).abs() < 1e-12);
        assert!((d.cost - 1144.8).abs() < 1e-9);
        assert!(d.constraint_ok);
    }

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
            assert!(d.constraint_ok);
        }
    }
}
