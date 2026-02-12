// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Ignition
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Ignition physics: Bosch-Hale D-T reaction rate and thermodynamics.
//!
//! Port of `fusion_ignition_sim.py` lines 14-107.
//! Calculates fusion power, alpha heating, and Q-factor.

use crate::kernel::FusionKernel;
use fusion_types::constants::E_FUSION_DT;
use fusion_types::state::ThermodynamicsResult;

/// Minimum temperature in keV (below this σv is negligible).
/// Python line 22: `T = max(T, 0.1)`
const T_MIN_KEV: f64 = 0.1;

/// Peak density [m⁻³]. Python line 56.
const N_PEAK: f64 = 1.0e20;

/// Peak temperature [keV]. Python line 57.
const T_PEAK_KEV: f64 = 20.0;

/// Density profile exponent. Python line 63.
const DENSITY_EXPONENT: f64 = 0.5;

/// Temperature profile exponent. Python line 64.
const TEMPERATURE_EXPONENT: f64 = 1.0;

/// Alpha particle energy fraction (3.5/17.6 MeV ≈ 0.199).
/// Python line 82: `P_alpha = P_fusion * 0.2`
const ALPHA_FRACTION: f64 = 0.2;

/// Energy confinement time [s]. Python line 88.
const TAU_E: f64 = 3.0;

/// keV to Joules conversion: 1 keV = 1000 eV × 1.602e-19 J/eV
const KEV_TO_JOULES: f64 = 1.602e-16;

/// Bosch-Hale D-T fusion reaction rate ⟨σv⟩ in m³/s.
///
/// Uses NRL Plasma Formulary approximation:
///   σv = 3.68e-18 / T^(2/3) × exp(-19.94 / T^(1/3))
///
/// Valid for T < 100 keV. T is clamped to T_MIN_KEV.
pub fn bosch_hale_dt(t_kev: f64) -> f64 {
    let t = t_kev.max(T_MIN_KEV);
    3.68e-18 / t.powf(2.0 / 3.0) * (-19.94 / t.powf(1.0 / 3.0)).exp()
}

/// Calculate full thermodynamics from equilibrium state.
///
/// Algorithm:
/// 1. Normalize flux: ψ_norm = (Ψ - Ψ_axis) / (Ψ_boundary - Ψ_axis)
/// 2. Profiles: n(ψ) = n_peak·(1 - ψ²)^0.5, T(ψ) = T_peak·(1 - ψ²)^1.0
/// 3. Fusion power: P_fus = ∫ nD·nT·⟨σv⟩·E_fus dV, dV = dR·dZ·2πR
/// 4. Alpha heating: P_alpha = 0.2·P_fusion
/// 5. Thermal energy: W_th = ∫ 3·n·T dV
/// 6. Losses: P_loss = W_th / τ_E
/// 7. Q = P_fusion / P_aux
pub fn calculate_thermodynamics(kernel: &FusionKernel, p_aux_mw: f64) -> ThermodynamicsResult {
    let grid = kernel.grid();
    let psi = kernel.psi();
    let state = kernel.state();
    let nz = grid.nz;
    let nr = grid.nr;
    let dr = grid.dr;
    let dz = grid.dz;

    let psi_axis = state.psi_axis;
    let psi_boundary = state.psi_boundary;
    let mut denom = psi_boundary - psi_axis;
    if denom.abs() < 1e-9 {
        denom = 1e-9;
    }

    let mut p_fusion_w = 0.0_f64;
    let mut w_thermal_j = 0.0_f64;
    let mut t_peak_actual = 0.0_f64;

    for iz in 0..nz {
        for ir in 0..nr {
            let psi_norm = (psi[[iz, ir]] - psi_axis) / denom;

            if (0.0..1.0).contains(&psi_norm) {
                let psi_norm_sq = psi_norm * psi_norm;
                let one_minus = (1.0 - psi_norm_sq).max(0.0);

                // Profiles
                let n_e = N_PEAK * one_minus.powf(DENSITY_EXPONENT); // m⁻³
                let t_kev = T_PEAK_KEV * one_minus.powf(TEMPERATURE_EXPONENT); // keV

                if t_kev > t_peak_actual {
                    t_peak_actual = t_kev;
                }

                // D-T: assume 50/50 mix → nD = nT = n_e / 2
                let n_d = n_e / 2.0;
                let n_t = n_e / 2.0;

                let sigma_v = bosch_hale_dt(t_kev);

                let r = grid.rr[[iz, ir]];
                let dv = dr * dz * 2.0 * std::f64::consts::PI * r; // toroidal volume element

                // Fusion power density × volume
                p_fusion_w += n_d * n_t * sigma_v * E_FUSION_DT * dv;

                // Thermal energy: W = 3 n T (using keV→J conversion)
                w_thermal_j += 3.0 * n_e * t_kev * KEV_TO_JOULES * dv;
            }
        }
    }

    let p_fusion_mw = p_fusion_w / 1e6;
    let p_alpha_mw = p_fusion_mw * ALPHA_FRACTION;
    let w_thermal_mj = w_thermal_j / 1e6;
    let p_loss_mw = w_thermal_mj / TAU_E; // MW = MJ/s

    let q_factor = if p_aux_mw > 1e-9 {
        p_fusion_mw / p_aux_mw
    } else {
        0.0
    };

    let net_mw = p_alpha_mw + p_aux_mw - p_loss_mw;

    ThermodynamicsResult {
        p_fusion_mw,
        p_alpha_mw,
        p_loss_mw,
        p_aux_mw,
        net_mw,
        q_factor,
        t_peak_kev: t_peak_actual,
        w_thermal_mj,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bosch_hale_zero_temp() {
        // At T=0 (clamped to 0.1 keV), rate should be extremely small
        assert!(bosch_hale_dt(0.0) < 1e-30, "Rate at T=0 should be tiny");
    }

    #[test]
    fn test_bosch_hale_20kev() {
        let rate = bosch_hale_dt(20.0);
        assert!(
            rate > 1e-22 && rate < 1e-21,
            "Rate at 20keV: {rate}, expected O(1e-22)"
        );
    }

    #[test]
    fn test_bosch_hale_peak_higher() {
        let rate_60 = bosch_hale_dt(60.0);
        let rate_1 = bosch_hale_dt(1.0);
        assert!(
            rate_60 > rate_1 * 1000.0,
            "Rate at 60keV ({rate_60}) should be >> rate at 1keV ({rate_1})"
        );
    }

    #[test]
    fn test_bosch_hale_monotonic_rise() {
        // Rate should increase monotonically from 1 to ~60 keV
        let mut prev = bosch_hale_dt(1.0);
        for t in [5.0, 10.0, 20.0, 40.0, 60.0] {
            let rate = bosch_hale_dt(t);
            assert!(
                rate > prev,
                "Rate should increase: T={t}, rate={rate}, prev={prev}"
            );
            prev = rate;
        }
    }

    #[test]
    fn test_thermodynamics_after_solve() {
        use std::path::PathBuf;

        fn project_root() -> PathBuf {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
                .join("..")
        }
        fn config_path(relative: &str) -> String {
            project_root().join(relative).to_string_lossy().to_string()
        }

        // Use smaller grid for faster test
        let mut kernel =
            FusionKernel::from_file(&config_path("validation/iter_validated_config.json")).unwrap();
        kernel.solve_equilibrium().unwrap();

        let result = calculate_thermodynamics(&kernel, 50.0);

        // Fusion power should be positive and finite
        assert!(
            result.p_fusion_mw > 0.0,
            "P_fusion = {}",
            result.p_fusion_mw
        );
        assert!(result.p_fusion_mw.is_finite(), "P_fusion must be finite");

        // Q-factor should be positive
        assert!(result.q_factor > 0.0, "Q = {}", result.q_factor);

        // Alpha power should be 20% of fusion
        let alpha_ratio = result.p_alpha_mw / result.p_fusion_mw;
        assert!(
            (alpha_ratio - 0.2).abs() < 1e-10,
            "Alpha ratio = {alpha_ratio}, expected 0.2"
        );
    }
}
