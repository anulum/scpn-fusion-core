// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Real-Time Control (RTC) Driver
// © 1998–2026 Miroslav Šotek. All rights reserved.
// ─────────────────────────────────────────────────────────────────────
//! Hardened Real-Time Control (RTC) driver for SCPN.
//!
//! Provides deterministic timing for high-frequency control loops.
//! Designed for Preempt-RT Linux and bare-metal targets.

use crate::flight_sim::{RustFlightSim, SimulationReport};
use fusion_types::error::{FusionError, FusionResult};

/// Configuration for the real-time driver.
pub struct RtcConfig {
    pub target_hz: f64,
    pub max_jitter_us: f64,
    pub use_busy_wait: bool,
}

/// Hardened RTC Driver.
pub struct RtcDriver {
    pub sim: RustFlightSim,
    pub config: RtcConfig,
}

impl RtcDriver {
    pub fn new(sim: RustFlightSim, config: RtcConfig) -> Self {
        Self { sim, config }
    }

    /// Execute a shot with deterministic timing.
    pub fn run_shot_hardened(&mut self, shot_duration_s: f64) -> FusionResult<SimulationReport> {
        if !shot_duration_s.is_finite() || shot_duration_s <= 0.0 {
            return Err(FusionError::ConfigError(
                "shot_duration_s must be finite and > 0".to_string(),
            ));
        }
        if !self.config.max_jitter_us.is_finite() || self.config.max_jitter_us < 0.0 {
            return Err(FusionError::ConfigError(
                "max_jitter_us must be finite and >= 0".to_string(),
            ));
        }

        let report = self
            .sim
            .run_shot(shot_duration_s, self.config.use_busy_wait)?;

        if self.config.max_jitter_us > 0.0 && report.max_step_time_us > self.config.max_jitter_us {
            return Err(FusionError::PhysicsViolation(format!(
                "RTC jitter exceeded threshold: max_step_time_us={:.3} > allowed={:.3}",
                report.max_step_time_us, self.config.max_jitter_us
            )));
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::{RtcConfig, RtcDriver};
    use crate::flight_sim::RustFlightSim;

    #[test]
    fn test_run_shot_hardened_rejects_invalid_duration() {
        let sim = RustFlightSim::new(6.2, 0.0, 5000.0).expect("valid sim");
        let mut driver = RtcDriver::new(
            sim,
            RtcConfig {
                target_hz: 5000.0,
                max_jitter_us: 10_000.0,
                use_busy_wait: false,
            },
        );
        assert!(driver.run_shot_hardened(0.0).is_err());
    }

    #[test]
    fn test_run_shot_hardened_enforces_jitter_threshold() {
        let sim = RustFlightSim::new(6.2, 0.0, 2000.0).expect("valid sim");
        let mut driver = RtcDriver::new(
            sim,
            RtcConfig {
                target_hz: 2000.0,
                max_jitter_us: 0.0001,
                use_busy_wait: false,
            },
        );
        assert!(driver.run_shot_hardened(0.01).is_err());
    }

    #[test]
    fn test_run_shot_hardened_passes_with_relaxed_jitter() {
        let sim = RustFlightSim::new(6.2, 0.0, 2000.0).expect("valid sim");
        let mut driver = RtcDriver::new(
            sim,
            RtcConfig {
                target_hz: 2000.0,
                max_jitter_us: 100_000.0,
                use_busy_wait: false,
            },
        );
        let report = driver
            .run_shot_hardened(0.01)
            .expect("jitter budget should pass");
        assert!(report.steps > 0);
    }
}
