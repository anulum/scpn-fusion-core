// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — FRC Tilt-Mode Diagnostics
//! Conservative MIF/FRC n=1 tilt-mode diagnostics.

pub const MU_0: f64 = 4.0e-7 * std::f64::consts::PI;
pub const ATOMIC_MASS_KG: f64 = 1.660_539_066_60e-27;
pub const DEUTERIUM_MASS_AMU: f64 = 2.014;
pub const BELOVA_MHD_GROWTH_COEFFICIENT: f64 = 1.2;
pub const DIAMAGNETIC_S_OVER_E_THRESHOLD: f64 = 1.7;
pub const GYROVISCOUS_S_OVER_E_THRESHOLD: f64 = 2.2;
pub const COMBINED_FLR_S_OVER_E_THRESHOLD: f64 = 2.8;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrcTiltModeInputs {
    pub s_parameter: f64,
    pub b_reference_t: f64,
    pub density_peak_m3: f64,
    pub r_s_m: f64,
    pub elongation: f64,
    pub ion_mass_amu: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrcTiltModeThresholds {
    pub diamagnetic_s_over_e: f64,
    pub gyroviscous_s_over_e: f64,
    pub combined_flr_s_over_e: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrcTiltModeReport {
    pub growth_rate_s_inv: f64,
    pub alfven_speed_m_s: f64,
    pub alfven_transit_time_s: f64,
    pub s_parameter: f64,
    pub elongation: f64,
    pub s_over_elongation: f64,
    pub rigid_body_regime: &'static str,
    pub rigid_body_threshold_passed: bool,
    pub conservative_stable: bool,
    pub claim_status: &'static str,
    pub external_parity_status: &'static str,
}

impl Default for FrcTiltModeThresholds {
    fn default() -> Self {
        Self {
            diamagnetic_s_over_e: DIAMAGNETIC_S_OVER_E_THRESHOLD,
            gyroviscous_s_over_e: GYROVISCOUS_S_OVER_E_THRESHOLD,
            combined_flr_s_over_e: COMBINED_FLR_S_OVER_E_THRESHOLD,
        }
    }
}

fn require_positive(name: &str, value: f64) -> Result<f64, String> {
    if !value.is_finite() {
        return Err(format!("{name} must be finite"));
    }
    if value <= 0.0 {
        return Err(format!("{name} must be positive"));
    }
    Ok(value)
}

fn validate_thresholds(thresholds: FrcTiltModeThresholds) -> Result<(), String> {
    let dia = require_positive("diamagnetic_s_over_e", thresholds.diamagnetic_s_over_e)?;
    let gyro = require_positive("gyroviscous_s_over_e", thresholds.gyroviscous_s_over_e)?;
    let combined = require_positive("combined_flr_s_over_e", thresholds.combined_flr_s_over_e)?;
    if dia < gyro && gyro < combined {
        Ok(())
    } else {
        Err("rigid-body thresholds must be strictly increasing".to_string())
    }
}

pub fn alfven_speed_m_s(inputs: FrcTiltModeInputs) -> Result<f64, String> {
    let field = require_positive("b_reference_t", inputs.b_reference_t.abs())?;
    let density_m3 = require_positive("density_peak_m3", inputs.density_peak_m3)?;
    let mass_amu = require_positive("ion_mass_amu", inputs.ion_mass_amu)?;
    let mass_density = density_m3 * mass_amu * ATOMIC_MASS_KG;
    Ok(field / (MU_0 * mass_density).sqrt())
}

pub fn axial_half_length_m(inputs: FrcTiltModeInputs) -> Result<f64, String> {
    Ok(require_positive("r_s_m", inputs.r_s_m)?
        * require_positive("elongation", inputs.elongation)?)
}

pub fn frc_tilt_growth_rate(
    inputs: FrcTiltModeInputs,
    mhd_coefficient: f64,
) -> Result<f64, String> {
    let coefficient = require_positive("mhd_coefficient", mhd_coefficient)?;
    Ok(coefficient * alfven_speed_m_s(inputs)? / axial_half_length_m(inputs)?)
}

pub fn s_over_elongation(inputs: FrcTiltModeInputs) -> Result<f64, String> {
    Ok(require_positive("s_parameter", inputs.s_parameter)?
        / require_positive("elongation", inputs.elongation)?)
}

pub fn rigid_body_flr_regime(
    inputs: FrcTiltModeInputs,
    thresholds: FrcTiltModeThresholds,
) -> Result<(&'static str, bool), String> {
    validate_thresholds(thresholds)?;
    let ratio = s_over_elongation(inputs)?;
    if ratio <= thresholds.diamagnetic_s_over_e {
        Ok(("diamagnetic_flr_threshold_passed", true))
    } else if ratio <= thresholds.gyroviscous_s_over_e {
        Ok(("gyroviscous_flr_threshold_passed", true))
    } else if ratio <= thresholds.combined_flr_s_over_e {
        Ok(("combined_flr_threshold_passed", true))
    } else {
        Ok(("mhd_tilt_susceptible", false))
    }
}

pub fn tilt_mode_report(inputs: FrcTiltModeInputs) -> Result<FrcTiltModeReport, String> {
    let growth = frc_tilt_growth_rate(inputs, BELOVA_MHD_GROWTH_COEFFICIENT)?;
    let speed = alfven_speed_m_s(inputs)?;
    let length = axial_half_length_m(inputs)?;
    let (regime, threshold_passed) =
        rigid_body_flr_regime(inputs, FrcTiltModeThresholds::default())?;
    Ok(FrcTiltModeReport {
        growth_rate_s_inv: growth,
        alfven_speed_m_s: speed,
        alfven_transit_time_s: length / speed,
        s_parameter: inputs.s_parameter,
        elongation: inputs.elongation,
        s_over_elongation: s_over_elongation(inputs)?,
        rigid_body_regime: regime,
        rigid_body_threshold_passed: threshold_passed,
        conservative_stable: false,
        claim_status: "diagnostic_only_not_hybrid_eigenvalue_accepted",
        external_parity_status: "blocked_missing_public_digitised_reference",
    })
}

pub fn belova_table1_acceptance_status() -> (&'static str, &'static str) {
    (
        "belova_2001_table1_tilt_stability",
        "blocked_missing_public_digitised_reference",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(elongation: f64) -> FrcTiltModeInputs {
        FrcTiltModeInputs {
            s_parameter: 8.0,
            b_reference_t: 5.0,
            density_peak_m3: 3.0e21,
            r_s_m: 0.2,
            elongation,
            ion_mass_amu: DEUTERIUM_MASS_AMU,
        }
    }

    #[test]
    fn growth_matches_alfven_time_scaling() {
        let case = inputs(4.0);
        let growth = frc_tilt_growth_rate(case, BELOVA_MHD_GROWTH_COEFFICIENT).unwrap();
        let expected = BELOVA_MHD_GROWTH_COEFFICIENT * alfven_speed_m_s(case).unwrap()
            / axial_half_length_m(case).unwrap();
        assert!((growth - expected).abs() / expected < 1.0e-14);
    }

    #[test]
    fn growth_decreases_with_elongation() {
        let short = frc_tilt_growth_rate(inputs(2.0), BELOVA_MHD_GROWTH_COEFFICIENT).unwrap();
        let long = frc_tilt_growth_rate(inputs(6.0), BELOVA_MHD_GROWTH_COEFFICIENT).unwrap();
        assert!((long - short / 3.0).abs() / long < 1.0e-14);
    }

    #[test]
    fn rigid_body_regime_uses_thresholds() {
        let thresholds = FrcTiltModeThresholds::default();
        let stable_elongation = 8.0 / (0.5 * thresholds.diamagnetic_s_over_e);
        let unstable_elongation = 8.0 / (1.1 * thresholds.combined_flr_s_over_e);
        let stable = rigid_body_flr_regime(inputs(stable_elongation), thresholds).unwrap();
        let unstable = rigid_body_flr_regime(inputs(unstable_elongation), thresholds).unwrap();
        assert_eq!(stable, ("diamagnetic_flr_threshold_passed", true));
        assert_eq!(unstable, ("mhd_tilt_susceptible", false));
    }

    #[test]
    fn report_is_fail_closed() {
        let report = tilt_mode_report(inputs(4.0)).unwrap();
        assert!(report.growth_rate_s_inv > 0.0);
        assert!(!report.conservative_stable);
        assert_eq!(
            report.claim_status,
            "diagnostic_only_not_hybrid_eigenvalue_accepted"
        );
        assert_eq!(
            report.external_parity_status,
            "blocked_missing_public_digitised_reference"
        );
    }

    #[test]
    fn inputs_fail_closed() {
        let mut bad = inputs(4.0);
        bad.elongation = 0.0;
        assert!(frc_tilt_growth_rate(bad, BELOVA_MHD_GROWTH_COEFFICIENT).is_err());
        assert!(frc_tilt_growth_rate(inputs(4.0), 0.0).is_err());
        assert!(rigid_body_flr_regime(
            inputs(4.0),
            FrcTiltModeThresholds {
                diamagnetic_s_over_e: 2.0,
                gyroviscous_s_over_e: 1.0,
                combined_flr_s_over_e: 3.0,
            },
        )
        .is_err());
    }
}
