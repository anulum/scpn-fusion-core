// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Breeding Blanket and First Wall
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Blanket loading and lifetime engineering utilities.

const SECONDS_PER_YEAR: f64 = 31_556_952.0;
const DT_NEUTRON_ENERGY_J: f64 = 14.1e6 * 1.602_176_634e-19;
const FE_REFERENCE_ATOM_MASS_KG: f64 = 9.27e-26; // ~56 amu

/// Neutron wall loading [MW/m^2].
///
/// Geometric estimate:
/// `q_n = P_neutron / (4*pi^2*R*a*sqrt((1+kappa^2)/2))`.
pub fn neutron_wall_loading(p_neutron: f64, r: f64, a: f64, kappa: f64) -> f64 {
    let p_neutron = p_neutron.max(0.0);
    let r = r.max(1e-9);
    let a = a.max(1e-9);
    let shaping = ((1.0 + kappa * kappa) / 2.0).sqrt().max(1e-9);
    let area = 4.0 * std::f64::consts::PI * std::f64::consts::PI * r * a * shaping;
    p_neutron / area
}

/// Displacements-per-atom rate [dpa/year].
///
/// Converts wall loading to neutron flux with 14.1 MeV neutrons and applies a
/// one-group displacement cross-section model.
pub fn dpa_rate(q_n: f64, sigma_d: f64, m_atom: f64) -> f64 {
    let q_n = q_n.max(0.0);
    let sigma_d = sigma_d.max(0.0);
    let m_atom = m_atom.max(1e-30);

    let neutron_flux = q_n * 1.0e6 / DT_NEUTRON_ENERGY_J;
    let material_factor = (FE_REFERENCE_ATOM_MASS_KG / m_atom).clamp(0.1, 10.0);
    neutron_flux * sigma_d * SECONDS_PER_YEAR * material_factor
}

/// Blanket lifetime [years] until dpa limit is reached.
pub fn blanket_lifetime(dpa_limit: f64, dpa_rate: f64) -> f64 {
    if dpa_rate <= 0.0 {
        return f64::INFINITY;
    }
    dpa_limit.max(0.0) / dpa_rate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wall_loading_iter() {
        let q_n = neutron_wall_loading(400.0, 6.2, 2.0, 1.7);
        assert!(
            (q_n - 0.6).abs() < 0.1,
            "Expected ITER-like wall loading around 0.6 MW/m^2, got {q_n}"
        );
    }

    #[test]
    fn test_blanket_lifetime_iter() {
        let q_n = neutron_wall_loading(400.0, 6.2, 2.0, 1.7);
        let rate = dpa_rate(q_n, 4.5e-24, 3.0e-25);
        let lifetime = blanket_lifetime(100.0, rate);
        assert!(
            (5.0..=10.0).contains(&lifetime),
            "Expected ITER-like first-wall lifetime in [5, 10] years, got {lifetime}"
        );
    }
}
