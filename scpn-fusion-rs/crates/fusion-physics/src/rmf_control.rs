// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — RMF Phase-Lock Control (Hardened Rust Lane)

use crate::precision_pacer::{PacingMode, PrecisionPacer};
use std::f64::consts::PI;

fn wrapped_phase_delta(delta: f64) -> f64 {
    (delta + PI).rem_euclid(2.0 * PI) - PI
}

#[derive(Debug, Clone, Copy)]
pub struct RmfAotCertificate {
    pub max_freq_hz: f64,
    pub min_freq_hz: f64,
    pub max_phase_error: f64,
}

impl Default for RmfAotCertificate {
    fn default() -> Self {
        Self {
            max_freq_hz: 5.0e6,
            min_freq_hz: 1.0e5,
            max_phase_error: PI / 2.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RmfConfig {
    pub f_rmf_nom_hz: f64,
    pub f_sampling_hz: f64,
    pub k_p: f64,
    pub k_d: f64,
    pub n_neurons: usize,
    pub aot_safety: RmfAotCertificate,
}

impl Default for RmfConfig {
    fn default() -> Self {
        Self {
            f_rmf_nom_hz: 1.0e6,
            f_sampling_hz: 10.0e6,
            k_p: 1.0e6,
            k_d: 0.1,
            n_neurons: 64,
            aot_safety: RmfAotCertificate::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpikingPhaseDetector {
    v_pos: Vec<f64>,
    v_neg: Vec<f64>,
    v_threshold: f64,
    alpha: f64,
}

impl SpikingPhaseDetector {
    pub fn new(n: usize, dt: f64) -> Self {
        let tau_mem = 0.05e-3;
        Self {
            v_pos: vec![0.0; n],
            v_neg: vec![0.0; n],
            v_threshold: 0.2,
            alpha: dt / tau_mem,
        }
    }

    pub fn step(&mut self, error_signal: f64) -> f64 {
        let i_scale = 8.0;
        let i_bias = 0.05;
        let input_pos = (error_signal.max(0.0) * i_scale) + i_bias;
        let input_neg = ((-error_signal).max(0.0) * i_scale) + i_bias;
        let mut spikes_pos = 0;
        let mut spikes_neg = 0;
        for val in self.v_pos.iter_mut() {
            *val += self.alpha * (-*val + input_pos);
            if *val >= self.v_threshold {
                *val = 0.0;
                spikes_pos += 1;
            }
        }
        for val in self.v_neg.iter_mut() {
            *val += self.alpha * (-*val + input_neg);
            if *val >= self.v_threshold {
                *val = 0.0;
                spikes_neg += 1;
            }
        }
        (spikes_pos as f64 - spikes_neg as f64) / self.v_pos.len().max(1) as f64
    }
}

#[derive(Debug)]
pub struct RmfPhaseLockController {
    cfg: RmfConfig,
    dt: f64,
    omega_nom: f64,
    omega_bias: f64,
    last_phi_plasma: Option<f64>,
    pub phi_ant: f64,
    pub omega_rmf: f64,
    pub t: f64,
    detector: SpikingPhaseDetector,
    pacer: Option<PrecisionPacer>,
    pub safety_violations: u64,
}

impl RmfPhaseLockController {
    pub fn new(cfg: RmfConfig) -> Self {
        let dt = 1.0 / cfg.f_sampling_hz;
        let omega_nom = 2.0 * PI * cfg.f_rmf_nom_hz;
        Self {
            cfg,
            dt,
            omega_nom,
            omega_bias: 0.0,
            last_phi_plasma: None,
            phi_ant: 0.0,
            omega_rmf: omega_nom,
            t: 0.0,
            detector: SpikingPhaseDetector::new(cfg.n_neurons, dt),
            pacer: None,
            safety_violations: 0,
        }
    }

    pub fn enable_pacing(&mut self, mode: PacingMode) {
        self.pacer = Some(PrecisionPacer::new(self.cfg.f_sampling_hz, mode));
    }

    pub fn step(&mut self, phi_plasma: f64) -> f64 {
        if let Some(ref mut pacer) = self.pacer {
            pacer.wait_next();
        }

        let error = (self.phi_ant - phi_plasma).sin();
        if error.abs() > self.cfg.aot_safety.max_phase_error {
            self.safety_violations += 1;
            return self.phi_ant;
        }

        let phase_error = if self.cfg.n_neurons > 0 {
            self.detector.step(error)
        } else {
            error
        };

        let observed_bias = self
            .last_phi_plasma
            .map(|last| wrapped_phase_delta(phi_plasma - last) / self.dt - self.omega_nom)
            .unwrap_or(self.omega_bias);
        self.omega_bias = observed_bias - (self.cfg.k_p * phase_error * self.dt);
        let min_omega = 2.0 * PI * self.cfg.aot_safety.min_freq_hz;
        let max_omega = 2.0 * PI * self.cfg.aot_safety.max_freq_hz;
        let new_omega = self.omega_nom + self.omega_bias;

        if new_omega < min_omega || new_omega > max_omega {
            self.safety_violations += 1;
            self.omega_rmf = new_omega.clamp(min_omega, max_omega);
            self.omega_bias = self.omega_rmf - self.omega_nom;
        } else {
            self.omega_rmf = new_omega;
        }

        self.phi_ant = (self.phi_ant + self.omega_rmf * self.dt) % (2.0 * PI);
        self.t += self.dt;
        self.last_phi_plasma = Some(phi_plasma);
        self.phi_ant
    }

    pub fn step_horizon(&mut self, phi_plasma_traj: &[f64]) -> Vec<f64> {
        phi_plasma_traj.iter().map(|&p| self.step(p)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wrapped_abs_error(a: f64, b: f64) -> f64 {
        (a - b).sin().abs()
    }

    #[test]
    fn test_rmf_controller_init() {
        let cfg = RmfConfig::default();
        let ctrl = RmfPhaseLockController::new(cfg);
        assert_eq!(ctrl.phi_ant, 0.0);
        assert!(ctrl.omega_rmf > 0.0);
    }

    #[test]
    fn test_rmf_controller_increases_frequency_when_antenna_lags() {
        let cfg = RmfConfig {
            f_rmf_nom_hz: 1.0e6,
            f_sampling_hz: 10.0e6,
            k_p: 2.0e7,
            k_d: 0.0,
            n_neurons: 0,
            aot_safety: RmfAotCertificate::default(),
        };
        let mut ctrl = RmfPhaseLockController::new(cfg);
        let initial = ctrl.omega_rmf;

        ctrl.step(0.25);

        assert!(ctrl.omega_rmf > initial);
        assert_eq!(ctrl.safety_violations, 0);
    }

    #[test]
    fn test_rmf_controller_bounds_phase_error_for_frequency_offset() {
        let cfg = RmfConfig {
            f_rmf_nom_hz: 1.0e6,
            f_sampling_hz: 20.0e6,
            k_p: 5.0e8,
            k_d: 2.0e3,
            n_neurons: 0,
            aot_safety: RmfAotCertificate {
                max_freq_hz: 1.2e6,
                min_freq_hz: 0.8e6,
                max_phase_error: PI,
            },
        };
        let mut ctrl = RmfPhaseLockController::new(cfg);
        let plasma_omega = 2.0 * PI * 1.01e6;
        let mut early_error = 0.0;
        let mut late_error = 0.0;

        for step in 0..20_000 {
            let phi_plasma = (plasma_omega * ctrl.dt * step as f64) % (2.0 * PI);
            let phi_ant = ctrl.step(phi_plasma);
            let error = wrapped_abs_error(phi_ant, phi_plasma);
            if (1_000..2_000).contains(&step) {
                early_error += error;
            }
            if step >= 19_000 {
                late_error += error;
            }
        }

        let early_mean = early_error / 1_000.0;
        let late_mean = late_error / 1_000.0;
        let target = plasma_omega;

        assert!(early_mean < 0.40, "early={early_mean}");
        assert!(late_mean < 0.40, "late={late_mean}");
        assert!(
            (late_mean - early_mean).abs() < 0.02,
            "early={early_mean}, late={late_mean}"
        );
        assert!((ctrl.omega_rmf - target).abs() / target < 0.02);
        assert_eq!(ctrl.safety_violations, 0);
    }

    #[test]
    fn test_rmf_controller_blocks_unsafe_phase_error_without_state_advance() {
        let cfg = RmfConfig {
            aot_safety: RmfAotCertificate {
                max_phase_error: 0.1,
                ..RmfAotCertificate::default()
            },
            ..RmfConfig::default()
        };
        let mut ctrl = RmfPhaseLockController::new(cfg);
        let phi_before = ctrl.phi_ant;
        let omega_before = ctrl.omega_rmf;
        let t_before = ctrl.t;

        let out = ctrl.step(PI / 2.0);

        assert_eq!(out, phi_before);
        assert_eq!(ctrl.phi_ant, phi_before);
        assert_eq!(ctrl.omega_rmf, omega_before);
        assert_eq!(ctrl.t, t_before);
        assert_eq!(ctrl.safety_violations, 1);
    }
}
