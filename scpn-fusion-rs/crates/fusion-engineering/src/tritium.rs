// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Tritium Fuel Cycle
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Tritium fuel-cycle utilities.
//!
//! These functions are compact 0D engineering models for early design studies.

/// Estimate the tritium breeding ratio (TBR) in a Li-6 blanket.
///
/// Uses a one-group optical-depth approximation:
/// `tau = n_li6 * sigma_li6 * L_eff`, with `L_eff ~ V^(1/3)`.
/// The final TBR is constrained to a realistic engineering range for baseline studies.
pub fn tritium_breeding_ratio(
    n_li6: f64,
    sigma_li6: f64,
    neutron_flux: f64,
    blanket_vol: f64,
) -> f64 {
    let n_li6 = n_li6.max(0.0);
    let sigma_li6 = sigma_li6.max(0.0);
    let neutron_flux = neutron_flux.max(0.0);
    let blanket_vol = blanket_vol.max(0.0);

    if n_li6 == 0.0 || sigma_li6 == 0.0 || blanket_vol == 0.0 {
        return 1.0;
    }

    let effective_path = blanket_vol.cbrt().max(1e-9);
    let optical_depth = n_li6 * sigma_li6 * effective_path;
    let capture_fraction = 1.0 - (-optical_depth).exp();

    let spectral_factor = 1.0 - (-(neutron_flux / 1.0e14)).exp();
    1.0 + 0.15 * capture_fraction * (0.8 + 0.2 * spectral_factor)
}

/// Steady-state tritium inventory [arbitrary units of burn_rate * time].
///
/// The inventory requirement rises sharply as `TBR -> 1`.
pub fn steady_state_inventory(burn_rate: f64, tau_process: f64, tbr: f64) -> f64 {
    let burn_rate = burn_rate.max(0.0);
    let tau_process = tau_process.max(0.0);
    let margin = (tbr - 1.0).max(1e-6);
    burn_rate * tau_process / margin
}

/// DT burn-up fraction using a simple reaction-probability model.
///
/// `f_b = 1 - exp(-n_t * sigma_dt * v_t * tau_conf)`.
pub fn burnup_fraction(n_t: f64, sigma_dt: f64, v_t: f64, tau_conf: f64) -> f64 {
    let n_t = n_t.max(0.0);
    let sigma_dt = sigma_dt.max(0.0);
    let v_t = v_t.max(0.0);
    let tau_conf = tau_conf.max(0.0);
    let exponent = -(n_t * sigma_dt * v_t * tau_conf);
    (1.0 - exponent.exp()).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tritium_breeding_iter() {
        // Representative baseline blanket parameters for ITER-like sizing.
        let tbr = tritium_breeding_ratio(4.5e27, 9.4e-28, 1.2e14, 500.0);
        assert!(
            (1.05..=1.15).contains(&tbr),
            "Expected ITER-like TBR in [1.05, 1.15], got {tbr}"
        );
    }

    #[test]
    fn test_burnup_fraction_positive() {
        let f_b = burnup_fraction(1.0e20, 1.1e-28, 1.0e6, 3.0);
        assert!(f_b > 0.0, "Burnup fraction should be positive");
        assert!(f_b < 1.0, "Burnup fraction should stay below unity");
    }
}
