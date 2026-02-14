// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — EPED-Like Pedestal Model
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Simplified EPED-like pedestal model with ELM trigger logic.

use fusion_types::error::{FusionError, FusionResult};
use fusion_types::state::RadialProfiles;

const MIN_T_KEV: f64 = 0.05;
const MIN_NE: f64 = 1e-4;

#[derive(Debug, Clone, Copy)]
pub struct PedestalConfig {
    pub beta_p_ped: f64,
    pub rho_s: f64,
    pub r_major: f64,
    pub alpha_crit: f64,
    pub tau_elm: f64,
}

impl Default for PedestalConfig {
    fn default() -> Self {
        Self {
            beta_p_ped: 0.35,
            rho_s: 2.0e-3,
            r_major: 6.2,
            alpha_crit: 2.5,
            tau_elm: 1.0e-3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PedestalModel {
    pub config: PedestalConfig,
    last_gradient: f64,
    elm_cooldown_s: f64,
}

impl PedestalModel {
    pub fn new(config: PedestalConfig) -> FusionResult<Self> {
        if !config.beta_p_ped.is_finite() || config.beta_p_ped < 0.0 {
            return Err(FusionError::ConfigError(
                "pedestal beta_p_ped must be finite and >= 0".to_string(),
            ));
        }
        if !config.rho_s.is_finite() || config.rho_s <= 0.0 {
            return Err(FusionError::ConfigError(
                "pedestal rho_s must be finite and > 0".to_string(),
            ));
        }
        if !config.r_major.is_finite() || config.r_major <= 0.0 {
            return Err(FusionError::ConfigError(
                "pedestal r_major must be finite and > 0".to_string(),
            ));
        }
        if !config.alpha_crit.is_finite() || config.alpha_crit <= 0.0 {
            return Err(FusionError::ConfigError(
                "pedestal alpha_crit must be finite and > 0".to_string(),
            ));
        }
        if !config.tau_elm.is_finite() || config.tau_elm <= 0.0 {
            return Err(FusionError::ConfigError(
                "pedestal tau_elm must be finite and > 0".to_string(),
            ));
        }
        Ok(Self {
            config,
            last_gradient: 0.0,
            elm_cooldown_s: 0.0,
        })
    }

    /// EPED-inspired width scaling: Δ_ped ~ sqrt(beta_p,ped) * (rho_s / R)
    pub fn pedestal_width(&self) -> f64 {
        let beta = self.config.beta_p_ped;
        let rho_s = self.config.rho_s;
        let r_major = self.config.r_major;
        (beta.sqrt() * (rho_s / r_major)).clamp(0.03, 0.12)
    }

    /// Effective ELM trigger threshold: alpha_crit ~ 2.5 * s_ped.
    fn effective_alpha_crit(&self) -> f64 {
        let shear_proxy = 1.0 / self.pedestal_width().max(1e-4);
        self.config.alpha_crit.max(2.5 * shear_proxy)
    }

    pub fn is_elm_triggered(&self, pressure_gradient: f64) -> bool {
        self.elm_cooldown_s <= 0.0 && pressure_gradient.abs() > self.effective_alpha_crit()
    }

    /// Advance ELM cooldown timer.
    pub fn advance(&mut self, dt_s: f64) {
        self.elm_cooldown_s = (self.elm_cooldown_s - dt_s.max(0.0)).max(0.0);
    }

    pub fn last_gradient(&self) -> f64 {
        self.last_gradient
    }

    /// Apply a fast ELM crash to pedestal region profiles.
    pub fn apply_elm_crash(&mut self, profiles: &mut RadialProfiles) {
        let width = self.pedestal_width();
        let rho_start = (1.0 - width).clamp(0.75, 0.995);
        let tau = self.config.tau_elm;
        let burst_fraction = (1.0 - (-1.0e-3 / tau).exp()).clamp(0.05, 0.95);

        for i in 0..profiles.rho.len() {
            let rho = profiles.rho[i];
            if rho >= rho_start {
                let edge_w = ((rho - rho_start) / width).clamp(0.0, 1.0);
                let temperature_drop = (0.2 + 0.6 * edge_w * burst_fraction).clamp(0.0, 0.95);
                let density_drop = 0.5 * temperature_drop;

                profiles.te[i] = (profiles.te[i] * (1.0 - temperature_drop)).max(MIN_T_KEV);
                profiles.ti[i] = (profiles.ti[i] * (1.0 - temperature_drop)).max(MIN_T_KEV);
                profiles.ne[i] = (profiles.ne[i] * (1.0 - density_drop)).max(MIN_NE);
            }
        }

        self.elm_cooldown_s = 3.0 * tau;
    }

    pub fn record_gradient(&mut self, pressure_gradient: f64) {
        self.last_gradient = pressure_gradient;
    }
}

impl Default for PedestalModel {
    fn default() -> Self {
        Self::new(PedestalConfig::default()).expect("default pedestal config must be valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn sample_profiles() -> RadialProfiles {
        let n = 50;
        let rho = Array1::linspace(0.0, 1.0, n);
        let te = Array1::from_shape_fn(n, |i| 8.0 - 6.5 * rho[i]);
        let ti = te.clone();
        let ne = Array1::from_shape_fn(n, |i| 9.0 - 5.0 * rho[i]);
        let n_impurity = Array1::zeros(n);
        RadialProfiles {
            rho,
            te,
            ti,
            ne,
            n_impurity,
        }
    }

    #[test]
    fn test_pedestal_width_eped_scaling() {
        let model = PedestalModel::new(PedestalConfig {
            beta_p_ped: 0.5,
            rho_s: 3.0e-3,
            r_major: 6.0,
            alpha_crit: 2.5,
            tau_elm: 1.0e-3,
        })
        .expect("valid pedestal config");
        let width = model.pedestal_width();
        assert!(width > 0.0);
        assert!(width <= 0.12);
    }

    #[test]
    fn test_elm_trigger_threshold() {
        let model = PedestalModel::default();
        assert!(!model.is_elm_triggered(1.0));
        assert!(model.is_elm_triggered(5_000.0));
    }

    #[test]
    fn test_apply_elm_crash_reduces_edge_profiles() {
        let mut model = PedestalModel::default();
        let mut profiles = sample_profiles();

        let i_edge = profiles.rho.len() - 2;
        let te_before = profiles.te[i_edge];
        let ne_before = profiles.ne[i_edge];

        model.apply_elm_crash(&mut profiles);

        assert!(profiles.te[i_edge] < te_before);
        assert!(profiles.ne[i_edge] < ne_before);
    }

    #[test]
    fn test_elm_cooldown_blocks_retrigger_until_advanced() {
        let mut model = PedestalModel::default();
        let mut profiles = sample_profiles();

        assert!(model.is_elm_triggered(5_000.0));
        model.apply_elm_crash(&mut profiles);
        assert!(
            !model.is_elm_triggered(5_000.0),
            "ELM cooldown should block immediate retriggering"
        );

        model.advance(10.0);
        assert!(
            model.is_elm_triggered(5_000.0),
            "ELM trigger should recover after cooldown elapses"
        );
    }

    #[test]
    fn test_elm_crash_respects_minimum_profile_floors() {
        let mut model = PedestalModel::default();
        let n = 16;
        let mut profiles = RadialProfiles {
            rho: Array1::linspace(0.0, 1.0, n),
            te: Array1::from_elem(n, 0.06),
            ti: Array1::from_elem(n, 0.06),
            ne: Array1::from_elem(n, 2.0e-4),
            n_impurity: Array1::zeros(n),
        };

        model.apply_elm_crash(&mut profiles);
        assert!(profiles.te.iter().all(|v| *v >= MIN_T_KEV));
        assert!(profiles.ti.iter().all(|v| *v >= MIN_T_KEV));
        assert!(profiles.ne.iter().all(|v| *v >= MIN_NE));
    }

    #[test]
    fn test_pedestal_constructor_rejects_invalid_config() {
        let bad = [
            PedestalConfig {
                beta_p_ped: -1.0,
                ..Default::default()
            },
            PedestalConfig {
                rho_s: 0.0,
                ..Default::default()
            },
            PedestalConfig {
                r_major: 0.0,
                ..Default::default()
            },
            PedestalConfig {
                alpha_crit: 0.0,
                ..Default::default()
            },
            PedestalConfig {
                tau_elm: 0.0,
                ..Default::default()
            },
        ];
        for cfg in bad {
            let err = PedestalModel::new(cfg).expect_err("invalid config must error");
            match err {
                FusionError::ConfigError(msg) => {
                    assert!(msg.contains("pedestal"));
                }
                other => panic!("Unexpected error: {other:?}"),
            }
        }
    }
}
