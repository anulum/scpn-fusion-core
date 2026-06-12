// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — RMF Phase-Lock Control (Hardened Rust Lane)

use std::f64::consts::PI;
use crate::precision_pacer::{PrecisionPacer, PacingMode};

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
            if *val >= self.v_threshold { *val = 0.0; spikes_pos += 1; }
        }
        for val in self.v_neg.iter_mut() {
            *val += self.alpha * (-*val + input_neg);
            if *val >= self.v_threshold { *val = 0.0; spikes_neg += 1; }
        }
        (spikes_pos as f64 - spikes_neg as f64) / self.v_pos.len().max(1) as f64
    }
}

#[derive(Debug)]
pub struct RmfPhaseLockController {
    cfg: RmfConfig,
    dt: f64,
    omega_nom: f64,
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

        let d_omega = -self.cfg.k_p * phase_error - self.cfg.k_d * (self.omega_rmf - self.omega_nom);
        let new_omega = self.omega_rmf + d_omega * self.dt;

        if new_omega < 2.0 * PI * self.cfg.aot_safety.min_freq_hz || new_omega > 2.0 * PI * self.cfg.aot_safety.max_freq_hz {
            self.safety_violations += 1;
            self.omega_rmf = self.omega_rmf.clamp(
                2.0 * PI * self.cfg.aot_safety.min_freq_hz,
                2.0 * PI * self.cfg.aot_safety.max_freq_hz
            );
        } else {
            self.omega_rmf = new_omega;
        }

        self.phi_ant = (self.phi_ant + self.omega_rmf * self.dt) % (2.0 * PI);
        self.t += self.dt;
        self.phi_ant
    }

    pub fn step_horizon(&mut self, phi_plasma_traj: &[f64]) -> Vec<f64> {
        phi_plasma_traj.iter().map(|&p| self.step(p)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmf_controller_init() {
        let cfg = RmfConfig::default();
        let ctrl = RmfPhaseLockController::new(cfg);
        assert_eq!(ctrl.phi_ant, 0.0);
        assert!(ctrl.omega_rmf > 0.0);
    }
}
