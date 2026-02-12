// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Plant Layout and Cost Model
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Plant layout scan and compact economic scaling laws.

use serde::{Deserialize, Serialize};

/// Compact plant design point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlantDesign {
    /// Major radius [m].
    pub r_major: f64,
    /// On-axis toroidal field [T].
    pub b_field: f64,
    /// Net electric power [MW].
    pub p_net: f64,
    /// Capacity factor [-].
    pub capacity_factor: f64,
    /// Capital cost estimate [USD].
    pub capital_cost: f64,
    /// Cost of electricity [USD/MWh].
    pub coe: f64,
}

/// ARIES-style scaling law.
///
/// `C = C0 * R^2.5 * B^0.8`.
pub fn aries_cost_scaling(c0: f64, r: f64, b: f64) -> f64 {
    c0.max(0.0) * r.max(0.0).powf(2.5) * b.max(0.0).powf(0.8)
}

/// Cost of electricity [USD/MWh].
pub fn cost_of_electricity(capital_annuity: f64, o_and_m: f64, p_net: f64, cf: f64) -> f64 {
    if p_net <= 0.0 || cf <= 0.0 {
        return f64::INFINITY;
    }
    let annual_energy_mwh = p_net * cf * 8_760.0;
    (capital_annuity + o_and_m).max(0.0) / annual_energy_mwh
}

/// Scan major radius and generate a family of design points.
pub fn scan_major_radius(r_min: f64, r_max: f64, steps: usize) -> Vec<PlantDesign> {
    if steps == 0 {
        return Vec::new();
    }

    let mut designs = Vec::with_capacity(steps);
    let dr = if steps > 1 {
        (r_max - r_min) / (steps as f64 - 1.0)
    } else {
        0.0
    };

    for i in 0..steps {
        let r_major = r_min + dr * i as f64;
        let b_field = (12.0 - 0.55 * (r_major - 2.0)).clamp(4.0, 12.0);
        let p_net = (250.0 * (r_major / 2.5).powi(2)).clamp(200.0, 2_000.0);
        let capacity_factor = 0.85;

        // Cost coefficient chosen to yield realistic single-digit billions for ITER/DEMO scales.
        let capital_cost = aries_cost_scaling(0.02, r_major, b_field) * 1.0e9;
        let capital_annuity = 0.10 * capital_cost;
        let o_and_m = 0.04 * capital_cost;
        let coe = cost_of_electricity(capital_annuity, o_and_m, p_net, capacity_factor);

        designs.push(PlantDesign {
            r_major,
            b_field,
            p_net,
            capacity_factor,
            capital_cost,
            coe,
        });
    }

    designs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_model_demo_plant() {
        let capital_cost = aries_cost_scaling(0.02, 6.2, 5.3) * 1.0e9;
        let coe = cost_of_electricity(0.10 * capital_cost, 0.04 * capital_cost, 1_000.0, 0.85);
        assert!(coe.is_finite());
        assert!(coe > 0.0);
        assert!(coe < 300.0, "Expected sane DEMO-range COE, got {coe}");
    }

    #[test]
    fn test_scan_major_radius_count_and_bounds() {
        let designs = scan_major_radius(3.0, 7.0, 9);
        assert_eq!(designs.len(), 9);
        assert!((designs.first().unwrap().r_major - 3.0).abs() < 1e-12);
        assert!((designs.last().unwrap().r_major - 7.0).abs() < 1e-12);
    }
}
