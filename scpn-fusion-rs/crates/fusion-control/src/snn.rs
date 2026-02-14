// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — SNN
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Spiking Neural Network (SNN) controller with LIF neurons.
//!
//! Port of `neuro_cybernetic_controller.py`.
//! Biologically plausible rate-coded control using populations of LIF neurons.

use fusion_types::error::{FusionError, FusionResult};

/// Default number of neurons per population. Python: 20 (50 for control).
#[cfg(test)]
const N_NEURONS: usize = 20;

/// Rate coding window size. Python: 10.
#[cfg(test)]
const WINDOW_SIZE: usize = 10;

/// Input current scaling [nA/m]. Python: 5.0.
const I_SCALE: f64 = 5.0;

/// Spontaneous activity bias [nA]. Python: 0.1.
const I_BIAS: f64 = 0.1;

/// Stochastic Leaky Integrate-and-Fire neuron.
#[derive(Debug, Clone)]
pub struct LIFNeuron {
    v: f64,
    v_rest: f64,
    v_threshold: f64,
    v_reset: f64,
    tau_m: f64,
    refractory: usize,
    refractory_period: usize,
}

impl LIFNeuron {
    pub fn new() -> Self {
        LIFNeuron {
            v: -65e-3,
            v_rest: -65e-3,
            v_threshold: -55e-3,
            v_reset: -70e-3,
            tau_m: 20e-3,
            refractory: 0,
            refractory_period: 2,
        }
    }

    /// One timestep. Returns true if spike.
    pub fn step(&mut self, current: f64, dt: f64) -> FusionResult<bool> {
        if !current.is_finite() {
            return Err(FusionError::ConfigError(
                "snn neuron current must be finite".to_string(),
            ));
        }
        if !dt.is_finite() || dt <= 0.0 {
            return Err(FusionError::ConfigError(
                "snn neuron dt must be finite and > 0".to_string(),
            ));
        }
        if self.refractory > 0 {
            self.refractory -= 1;
            return Ok(false);
        }
        self.v += dt * (-(self.v - self.v_rest) / self.tau_m + current);
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.refractory = self.refractory_period;
            return Ok(true);
        }
        Ok(false)
    }
}

impl Default for LIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Population of LIF neurons with rate coding.
pub struct SpikingControllerPool {
    pub n_neurons: usize,
    pub gain: f64,
    pop_pos: Vec<LIFNeuron>,
    pop_neg: Vec<LIFNeuron>,
    history_pos: Vec<usize>,
    history_neg: Vec<usize>,
    window_size: usize,
}

impl SpikingControllerPool {
    pub fn new(n_neurons: usize, gain: f64, window_size: usize) -> FusionResult<Self> {
        if n_neurons == 0 {
            return Err(FusionError::ConfigError(
                "snn n_neurons must be > 0".to_string(),
            ));
        }
        if !gain.is_finite() {
            return Err(FusionError::ConfigError(
                "snn gain must be finite".to_string(),
            ));
        }
        if window_size == 0 {
            return Err(FusionError::ConfigError(
                "snn window_size must be > 0".to_string(),
            ));
        }
        Ok(SpikingControllerPool {
            n_neurons,
            gain,
            pop_pos: (0..n_neurons).map(|_| LIFNeuron::new()).collect(),
            pop_neg: (0..n_neurons).map(|_| LIFNeuron::new()).collect(),
            history_pos: Vec::new(),
            history_neg: Vec::new(),
            window_size,
        })
    }

    /// Process error signal through SNN. Returns control output.
    pub fn step(&mut self, error: f64) -> FusionResult<f64> {
        if !error.is_finite() {
            return Err(FusionError::ConfigError(
                "snn error input must be finite".to_string(),
            ));
        }
        let dt = 1e-3; // 1 ms timestep
        let input_pos = error.max(0.0) * I_SCALE;
        let input_neg = (-error).max(0.0) * I_SCALE;

        let mut spikes_pos = 0;
        for neuron in &mut self.pop_pos {
            if neuron.step(I_BIAS + input_pos, dt)? {
                spikes_pos += 1;
            }
        }

        let mut spikes_neg = 0;
        for neuron in &mut self.pop_neg {
            if neuron.step(I_BIAS + input_neg, dt)? {
                spikes_neg += 1;
            }
        }

        self.history_pos.push(spikes_pos);
        self.history_neg.push(spikes_neg);

        // Windowed rate
        let window = self.window_size.min(self.history_pos.len());
        let recent_pos: usize = self.history_pos[self.history_pos.len() - window..]
            .iter()
            .sum();
        let recent_neg: usize = self.history_neg[self.history_neg.len() - window..]
            .iter()
            .sum();

        let rate_pos = recent_pos as f64 / (window as f64 * self.n_neurons as f64);
        let rate_neg = recent_neg as f64 / (window as f64 * self.n_neurons as f64);

        Ok((rate_pos - rate_neg) * self.gain)
    }
}

/// Neuro-cybernetic controller with R and Z SNN pools.
pub struct NeuroCyberneticController {
    pub brain_r: SpikingControllerPool,
    pub brain_z: SpikingControllerPool,
    pub target_r: f64,
    pub target_z: f64,
}

impl NeuroCyberneticController {
    pub fn new(target_r: f64, target_z: f64) -> FusionResult<Self> {
        if !target_r.is_finite() || !target_z.is_finite() {
            return Err(FusionError::ConfigError(
                "snn targets must be finite".to_string(),
            ));
        }
        Ok(NeuroCyberneticController {
            brain_r: SpikingControllerPool::new(50, 10.0, 20)?,
            brain_z: SpikingControllerPool::new(50, 20.0, 20)?,
            target_r,
            target_z,
        })
    }

    /// Process measured position, return (ctrl_R, ctrl_Z).
    pub fn step(&mut self, measured_r: f64, measured_z: f64) -> FusionResult<(f64, f64)> {
        if !measured_r.is_finite() || !measured_z.is_finite() {
            return Err(FusionError::ConfigError(
                "snn measured positions must be finite".to_string(),
            ));
        }
        let err_r = self.target_r - measured_r;
        let err_z = self.target_z - measured_z;
        let ctrl_r = self.brain_r.step(err_r)?;
        let ctrl_z = self.brain_z.step(err_z)?;
        Ok((ctrl_r, ctrl_z))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lif_neuron_spikes() {
        let mut neuron = LIFNeuron::new();
        let mut spiked = false;
        // High current should produce a spike
        for _ in 0..100 {
            if neuron
                .step(10.0, 1e-3)
                .expect("valid finite current and dt")
            {
                spiked = true;
                break;
            }
        }
        assert!(spiked, "High current should produce a spike");
    }

    #[test]
    fn test_lif_no_spike_low_current() {
        let mut neuron = LIFNeuron::new();
        let mut spiked = false;
        for _ in 0..100 {
            if neuron.step(0.0, 1e-3).expect("valid finite current and dt") {
                spiked = true;
            }
        }
        assert!(!spiked, "Zero current should not spike");
    }

    #[test]
    fn test_pool_positive_error() {
        let mut pool =
            SpikingControllerPool::new(N_NEURONS, 1.0, WINDOW_SIZE).expect("valid pool config");
        // Sustained positive error → positive output
        let mut last_output = 0.0;
        for _ in 0..50 {
            last_output = pool.step(5.0).expect("valid finite error");
        }
        assert!(
            last_output > 0.0,
            "Positive error → positive output: {last_output}"
        );
    }

    #[test]
    fn test_pool_negative_error() {
        let mut pool =
            SpikingControllerPool::new(N_NEURONS, 1.0, WINDOW_SIZE).expect("valid pool config");
        let mut last_output = 0.0;
        for _ in 0..50 {
            last_output = pool.step(-5.0).expect("valid finite error");
        }
        assert!(
            last_output < 0.0,
            "Negative error → negative output: {last_output}"
        );
    }

    #[test]
    fn test_snn_rejects_invalid_constructor_and_step_inputs() {
        assert!(SpikingControllerPool::new(0, 1.0, WINDOW_SIZE).is_err());
        assert!(SpikingControllerPool::new(N_NEURONS, f64::NAN, WINDOW_SIZE).is_err());
        assert!(SpikingControllerPool::new(N_NEURONS, 1.0, 0).is_err());

        let mut pool =
            SpikingControllerPool::new(N_NEURONS, 1.0, WINDOW_SIZE).expect("valid pool config");
        assert!(pool.step(f64::NAN).is_err());

        let mut neuron = LIFNeuron::new();
        assert!(neuron.step(0.1, 0.0).is_err());
        assert!(neuron.step(f64::NAN, 1e-3).is_err());
    }

    #[test]
    fn test_neuro_cybernetic_controller_rejects_non_finite_inputs() {
        assert!(NeuroCyberneticController::new(f64::NAN, 0.0).is_err());
        let mut ctrl =
            NeuroCyberneticController::new(6.2, 0.0).expect("valid finite target inputs");
        assert!(ctrl.step(6.1, f64::INFINITY).is_err());
    }
}
