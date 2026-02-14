//! Reduced runtime kernel specialization and hot-swap lane.
//!
//! This module provides a deterministic stand-in for regime-triggered runtime
//! compilation. It keeps compilation/cache semantics explicit without external
//! LLVM/JIT dependencies so CI remains bounded.

use fusion_types::error::{FusionError, FusionResult};
use ndarray::Array1;
use std::collections::HashMap;

const MIN_DT_S: f64 = 1e-9;

/// Plasma operation regime used for runtime kernel specialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlasmaRegime {
    LMode,
    HMode,
    RampUp,
    RampDown,
}

/// Minimal observation bundle used for regime routing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegimeObservation {
    pub beta_n: f64,
    pub q95: f64,
    pub density_line_avg_1e20_m3: f64,
    pub current_ramp_ma_s: f64,
}

impl Default for RegimeObservation {
    fn default() -> Self {
        Self {
            beta_n: 1.5,
            q95: 4.5,
            density_line_avg_1e20_m3: 0.8,
            current_ramp_ma_s: 0.0,
        }
    }
}

/// Routing heuristic for reduced runtime specialization.
pub fn detect_regime(observation: &RegimeObservation) -> PlasmaRegime {
    if observation.current_ramp_ma_s > 0.2 {
        PlasmaRegime::RampUp
    } else if observation.current_ramp_ma_s < -0.2 {
        PlasmaRegime::RampDown
    } else if observation.beta_n >= 2.2
        || (observation.beta_n >= 2.0 && observation.q95 <= 3.7)
        || observation.density_line_avg_1e20_m3 >= 1.0
    {
        PlasmaRegime::HMode
    } else {
        PlasmaRegime::LMode
    }
}

fn validate_observation(observation: &RegimeObservation) -> FusionResult<()> {
    if !observation.beta_n.is_finite() {
        return Err(FusionError::ConfigError(
            "jit observation beta_n must be finite".to_string(),
        ));
    }
    if !observation.q95.is_finite() {
        return Err(FusionError::ConfigError(
            "jit observation q95 must be finite".to_string(),
        ));
    }
    if !observation.density_line_avg_1e20_m3.is_finite() {
        return Err(FusionError::ConfigError(
            "jit observation density_line_avg_1e20_m3 must be finite".to_string(),
        ));
    }
    if !observation.current_ramp_ma_s.is_finite() {
        return Err(FusionError::ConfigError(
            "jit observation current_ramp_ma_s must be finite".to_string(),
        ));
    }
    Ok(())
}

/// Compile-time shape metadata for generated kernels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KernelCompileSpec {
    pub n_state: usize,
    pub n_control: usize,
    pub dt_s: f64,
    pub unroll_factor: usize,
}

impl KernelCompileSpec {
    pub fn validated(self) -> FusionResult<Self> {
        if self.n_state == 0 {
            return Err(FusionError::ConfigError(
                "jit kernel n_state must be > 0".to_string(),
            ));
        }
        if self.n_control == 0 {
            return Err(FusionError::ConfigError(
                "jit kernel n_control must be > 0".to_string(),
            ));
        }
        if !self.dt_s.is_finite() || self.dt_s < MIN_DT_S {
            return Err(FusionError::ConfigError(format!(
                "jit kernel dt_s must be finite and >= {MIN_DT_S}"
            )));
        }
        if self.unroll_factor == 0 {
            return Err(FusionError::ConfigError(
                "jit kernel unroll_factor must be > 0".to_string(),
            ));
        }
        Ok(self)
    }
}

impl Default for KernelCompileSpec {
    fn default() -> Self {
        Self {
            n_state: 8,
            n_control: 4,
            dt_s: 1e-3,
            unroll_factor: 4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompiledKernel {
    pub regime: PlasmaRegime,
    pub generation: u64,
    pub spec: KernelCompileSpec,
    nonlinear_gain: f64,
    control_gain: f64,
    bias: f64,
}

impl CompiledKernel {
    fn from_regime(regime: PlasmaRegime, spec: KernelCompileSpec, generation: u64) -> Self {
        let (nonlinear_gain, control_gain, bias) = match regime {
            PlasmaRegime::LMode => (0.22, 0.08, -0.01),
            PlasmaRegime::HMode => (0.30, 0.12, 0.02),
            PlasmaRegime::RampUp => (0.26, 0.14, 0.04),
            PlasmaRegime::RampDown => (0.24, 0.13, -0.04),
        };
        Self {
            regime,
            generation,
            spec,
            nonlinear_gain,
            control_gain,
            bias,
        }
    }

    /// Execute one reduced control step for this specialized kernel.
    pub fn step(&self, state: &Array1<f64>, control: &Array1<f64>) -> FusionResult<Array1<f64>> {
        if state.len() != self.spec.n_state {
            return Err(FusionError::ConfigError(format!(
                "jit step state length mismatch: expected {}, got {}",
                self.spec.n_state,
                state.len()
            )));
        }
        if control.len() != self.spec.n_control {
            return Err(FusionError::ConfigError(format!(
                "jit step control length mismatch: expected {}, got {}",
                self.spec.n_control,
                control.len()
            )));
        }
        if state.iter().any(|v| !v.is_finite()) {
            return Err(FusionError::ConfigError(
                "jit step state vector must contain only finite values".to_string(),
            ));
        }
        if control.iter().any(|v| !v.is_finite()) {
            return Err(FusionError::ConfigError(
                "jit step control vector must contain only finite values".to_string(),
            ));
        }

        let control_mean = control.iter().copied().sum::<f64>() / self.spec.n_control as f64;
        if !control_mean.is_finite() {
            return Err(FusionError::ConfigError(
                "jit step control mean became non-finite".to_string(),
            ));
        }

        let forcing = self.control_gain * control_mean + self.bias;
        let dt = self.spec.dt_s;
        let mut next = Array1::zeros(self.spec.n_state);

        if let (Some(state_slice), Some(next_slice)) = (state.as_slice(), next.as_slice_mut()) {
            for start in (0..self.spec.n_state).step_by(self.spec.unroll_factor) {
                let end = (start + self.spec.unroll_factor).min(self.spec.n_state);
                for idx in start..end {
                    let x = state_slice[idx];
                    let drift = (self.nonlinear_gain * x).tanh() - 0.10 * x;
                    next_slice[idx] = x + dt * (drift + forcing);
                }
            }
            if next_slice.iter().any(|v| !v.is_finite()) {
                return Err(FusionError::ConfigError(
                    "jit step produced non-finite state output".to_string(),
                ));
            }
            return Ok(next);
        }

        for (idx, out) in next.iter_mut().enumerate() {
            let x = state[idx];
            let drift = (self.nonlinear_gain * x).tanh() - 0.10 * x;
            *out = x + dt * (drift + forcing);
        }
        if next.iter().any(|v| !v.is_finite()) {
            return Err(FusionError::ConfigError(
                "jit step produced non-finite state output".to_string(),
            ));
        }
        Ok(next)
    }
}

/// Runtime kernel specialization manager with cache + hot-swap semantics.
#[derive(Debug, Default, Clone)]
pub struct RuntimeKernelJit {
    kernels: HashMap<PlasmaRegime, CompiledKernel>,
    active: Option<PlasmaRegime>,
    compile_events: u64,
}

impl RuntimeKernelJit {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compile or reuse a kernel for the requested regime and activate it.
    pub fn compile_for_regime(
        &mut self,
        regime: PlasmaRegime,
        spec: KernelCompileSpec,
    ) -> FusionResult<u64> {
        let spec = spec.validated()?;
        if let Some(existing) = self.kernels.get(&regime) {
            if existing.spec == spec {
                self.active = Some(regime);
                return Ok(existing.generation);
            }
        }

        self.compile_events += 1;
        let generation = self.compile_events;
        let compiled = CompiledKernel::from_regime(regime, spec, generation);
        self.kernels.insert(regime, compiled);
        self.active = Some(regime);
        Ok(generation)
    }

    /// Detect regime, compile if needed, and activate specialized kernel.
    pub fn refresh_for_observation(
        &mut self,
        observation: &RegimeObservation,
        spec: KernelCompileSpec,
    ) -> FusionResult<(PlasmaRegime, u64)> {
        validate_observation(observation)?;
        let regime = detect_regime(observation);
        let generation = self.compile_for_regime(regime, spec)?;
        Ok((regime, generation))
    }

    pub fn active_regime(&self) -> Option<PlasmaRegime> {
        self.active
    }

    pub fn cache_size(&self) -> usize {
        self.kernels.len()
    }

    pub fn compile_events(&self) -> u64 {
        self.compile_events
    }

    /// Execute one step with the active specialized kernel.
    pub fn step_active(&self, state: &Array1<f64>, control: &Array1<f64>) -> FusionResult<Array1<f64>> {
        let Some(regime) = self.active else {
            return Err(FusionError::ConfigError(
                "jit step_active requires an active regime; compile or refresh first".to_string(),
            ));
        };
        let Some(kernel) = self.kernels.get(&regime) else {
            return Err(FusionError::ConfigError(
                "jit step_active active regime has no compiled kernel".to_string(),
            ));
        };
        kernel.step(state, control)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_regime_routing() {
        let ramp_up = RegimeObservation {
            current_ramp_ma_s: 0.4,
            ..RegimeObservation::default()
        };
        let ramp_down = RegimeObservation {
            current_ramp_ma_s: -0.5,
            ..RegimeObservation::default()
        };
        let h_mode = RegimeObservation {
            beta_n: 2.4,
            ..RegimeObservation::default()
        };
        let l_mode = RegimeObservation::default();

        assert_eq!(detect_regime(&ramp_up), PlasmaRegime::RampUp);
        assert_eq!(detect_regime(&ramp_down), PlasmaRegime::RampDown);
        assert_eq!(detect_regime(&h_mode), PlasmaRegime::HMode);
        assert_eq!(detect_regime(&l_mode), PlasmaRegime::LMode);
    }

    #[test]
    fn test_compile_cache_reuses_generation() {
        let mut jit = RuntimeKernelJit::new();
        let spec = KernelCompileSpec::default();
        let gen1 = jit
            .compile_for_regime(PlasmaRegime::LMode, spec)
            .expect("valid compile spec");
        let gen2 = jit
            .compile_for_regime(PlasmaRegime::LMode, spec)
            .expect("valid compile spec");
        assert_eq!(gen1, gen2);
        assert_eq!(jit.compile_events(), 1);
        assert_eq!(jit.cache_size(), 1);
    }

    #[test]
    fn test_hot_swap_changes_active_regime_and_response() {
        let mut jit = RuntimeKernelJit::new();
        let spec = KernelCompileSpec::default();
        let state = Array1::from_vec(vec![0.7; spec.n_state]);
        let control = Array1::from_vec(vec![0.2; spec.n_control]);

        jit.compile_for_regime(PlasmaRegime::LMode, spec)
            .expect("valid compile spec");
        let l_step = jit
            .step_active(&state, &control)
            .expect("valid active-kernel step inputs");

        jit.compile_for_regime(PlasmaRegime::HMode, spec)
            .expect("valid compile spec");
        let h_step = jit
            .step_active(&state, &control)
            .expect("valid active-kernel step inputs");

        assert_eq!(jit.active_regime(), Some(PlasmaRegime::HMode));
        let delta_sum = (&h_step - &l_step).iter().map(|v| v.abs()).sum::<f64>();
        assert!(
            delta_sum > 1e-9,
            "Expected regime hot-swap to alter response"
        );
    }

    #[test]
    fn test_refresh_for_observation_compiles_once_per_regime() {
        let mut jit = RuntimeKernelJit::new();
        let spec = KernelCompileSpec::default();

        let obs_l = RegimeObservation::default();
        let obs_h = RegimeObservation {
            beta_n: 2.6,
            ..RegimeObservation::default()
        };
        jit.refresh_for_observation(&obs_l, spec)
            .expect("valid compile spec");
        jit.refresh_for_observation(&obs_l, spec)
            .expect("valid compile spec");
        jit.refresh_for_observation(&obs_h, spec)
            .expect("valid compile spec");
        jit.refresh_for_observation(&obs_h, spec)
            .expect("valid compile spec");

        assert_eq!(jit.compile_events(), 2);
        assert_eq!(jit.cache_size(), 2);
        assert_eq!(jit.active_regime(), Some(PlasmaRegime::HMode));
    }

    #[test]
    fn test_step_active_without_kernel_rejects_missing_active_kernel() {
        let jit = RuntimeKernelJit::new();
        let state = Array1::from_vec(vec![1.0, -2.0, 0.5]);
        let control = Array1::from_vec(vec![0.1, 0.1]);
        assert!(jit.step_active(&state, &control).is_err());
    }

    #[test]
    fn test_compile_for_regime_rejects_invalid_compile_specs() {
        let mut jit = RuntimeKernelJit::new();
        let bad_n_state = KernelCompileSpec {
            n_state: 0,
            ..KernelCompileSpec::default()
        };
        let bad_n_control = KernelCompileSpec {
            n_control: 0,
            ..KernelCompileSpec::default()
        };
        let bad_dt = KernelCompileSpec {
            dt_s: f64::NAN,
            ..KernelCompileSpec::default()
        };
        let bad_unroll = KernelCompileSpec {
            unroll_factor: 0,
            ..KernelCompileSpec::default()
        };

        assert!(jit
            .compile_for_regime(PlasmaRegime::LMode, bad_n_state)
            .is_err());
        assert!(jit
            .compile_for_regime(PlasmaRegime::LMode, bad_n_control)
            .is_err());
        assert!(jit.compile_for_regime(PlasmaRegime::LMode, bad_dt).is_err());
        assert!(jit
            .compile_for_regime(PlasmaRegime::LMode, bad_unroll)
            .is_err());
        assert_eq!(jit.compile_events(), 0);
        assert_eq!(jit.cache_size(), 0);
    }

    #[test]
    fn test_refresh_for_observation_rejects_non_finite_inputs() {
        let mut jit = RuntimeKernelJit::new();
        let spec = KernelCompileSpec::default();
        let bad = RegimeObservation {
            beta_n: f64::NAN,
            ..RegimeObservation::default()
        };
        assert!(jit.refresh_for_observation(&bad, spec).is_err());
        assert_eq!(jit.compile_events(), 0);
        assert_eq!(jit.cache_size(), 0);
    }

    #[test]
    fn test_step_active_rejects_invalid_runtime_vectors() {
        let mut jit = RuntimeKernelJit::new();
        let spec = KernelCompileSpec::default();
        jit.compile_for_regime(PlasmaRegime::LMode, spec)
            .expect("valid compile spec");

        let bad_state = Array1::from_vec(vec![0.0; spec.n_state - 1]);
        let good_control = Array1::from_vec(vec![0.1; spec.n_control]);
        assert!(jit.step_active(&bad_state, &good_control).is_err());

        let good_state = Array1::from_vec(vec![0.0; spec.n_state]);
        let bad_control = Array1::from_vec(vec![0.1; spec.n_control - 1]);
        assert!(jit.step_active(&good_state, &bad_control).is_err());

        let nan_state = Array1::from_vec(vec![f64::NAN; spec.n_state]);
        assert!(jit.step_active(&nan_state, &good_control).is_err());
    }
}
