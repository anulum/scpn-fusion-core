// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Superconducting Magnet Engineering
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Magnet engineering utilities for toroidal-field and central-solenoid studies.

const REBCO_TC_K: f64 = 92.0;
const REBCO_I_C0_A: f64 = 12_000.0;
const REBCO_B0_T: f64 = 18.0;

/// Hoop stress estimate [Pa] from current density, field, and major radius.
///
/// `sigma = J * B * R`.
pub fn hoop_stress(j: f64, b: f64, r: f64) -> f64 {
    j.abs() * b.abs() * r.abs()
}

/// Magnetic stored energy [J].
///
/// `W = L * I^2 / 2`.
pub fn stored_energy(l: f64, i: f64) -> f64 {
    0.5 * l.max(0.0) * i * i
}

/// Characteristic dump time during a quench [s].
///
/// `tau = L / R`.
pub fn quench_time(l: f64, r_dump: f64) -> f64 {
    if r_dump <= 0.0 {
        return f64::INFINITY;
    }
    l.max(0.0) / r_dump
}

/// REBCO critical current scaling law [A].
///
/// Compact phenomenological form:
/// `I_c(B,T) = I_c0 * exp(-B/B0) * (1 - T/Tc)^(3/2)`.
pub fn rebco_critical_current(b: f64, t: f64) -> f64 {
    if t >= REBCO_TC_K {
        return 0.0;
    }
    let field_factor = (-(b.max(0.0) / REBCO_B0_T)).exp();
    let temp_factor = (1.0 - t.max(0.0) / REBCO_TC_K).max(0.0).powf(1.5);
    REBCO_I_C0_A * field_factor * temp_factor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magnet_stored_energy() {
        // ITER CS reference: L ~ 16 H, I ~ 28.3 kA, W ~ 6.4 GJ.
        let energy = stored_energy(16.0, 28_300.0);
        let target = 6.4e9;
        let rel_err = (energy - target).abs() / target;
        assert!(rel_err < 0.05, "Expected ~6.4 GJ, got {energy:e} J");
    }
}
