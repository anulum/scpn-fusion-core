// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Real-Time Control (RTC) Driver
// © 1998–2026 Miroslav Šotek. All rights reserved.
// ─────────────────────────────────────────────────────────────────────
//! Hardened Real-Time Control (RTC) driver for SCPN.
//!
//! Provides deterministic timing for high-frequency control loops.
//! Designed for Preempt-RT Linux and bare-metal targets.

use crate::flight_sim::{RustFlightSim, ShotAggregate, SimulationReport};
use fusion_types::error::{FusionError, FusionResult};
use std::thread;
use std::time::{Duration, Instant};

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
        if !self.config.target_hz.is_finite() || self.config.target_hz <= 0.0 {
            return Err(FusionError::ConfigError(
                "target_hz must be finite and > 0".to_string(),
            ));
        }
        if !self.config.max_jitter_us.is_finite() || self.config.max_jitter_us < 0.0 {
            return Err(FusionError::ConfigError(
                "max_jitter_us must be finite and >= 0".to_string(),
            ));
        }

        let sim_hz = 1.0 / self.sim.control_dt;
        let rel_hz_mismatch = ((sim_hz - self.config.target_hz) / self.config.target_hz).abs();
        if rel_hz_mismatch > 1e-6 {
            return Err(FusionError::ConfigError(format!(
                "RtcConfig target_hz ({:.6}) does not match simulator frequency ({:.6})",
                self.config.target_hz, sim_hz
            )));
        }

        let steps = self.sim.prepare_shot(shot_duration_s)?;
        let step_duration = Duration::from_secs_f64(self.sim.control_dt);
        let mut next_tick = Instant::now();
        let t_start = Instant::now();

        let mut aggregate = ShotAggregate {
            max_step_time_us: 0.0,
            r_err_sum: 0.0,
            z_err_sum: 0.0,
            disrupted: false,
            max_beta: self.sim.curr_beta,
            max_heating_mw: self.sim.curr_heating_mw,
            vessel_contact_events: 0,
            pf_constraint_events: 0,
            heating_constraint_events: 0,
        };

        for step_idx in 0..steps {
            if self.config.use_busy_wait {
                while Instant::now() < next_tick {
                    std::hint::spin_loop();
                }
            } else if let Some(wait) = next_tick.checked_duration_since(Instant::now()) {
                thread::sleep(wait);
            }

            let tick_started_at = Instant::now();
            let jitter_us = if tick_started_at >= next_tick {
                (tick_started_at - next_tick).as_secs_f64() * 1_000_000.0
            } else {
                (next_tick - tick_started_at).as_secs_f64() * 1_000_000.0
            };
            if self.config.max_jitter_us > 0.0 && jitter_us > self.config.max_jitter_us {
                return Err(FusionError::PhysicsViolation(format!(
                    "RTC jitter exceeded threshold: jitter_us={:.3} > allowed={:.3}",
                    jitter_us, self.config.max_jitter_us
                )));
            }

            let step = self.sim.step_once(step_idx, shot_duration_s)?;
            aggregate.r_err_sum += step.r_error;
            aggregate.z_err_sum += step.z_error;
            aggregate.disrupted |= step.disrupted;
            aggregate.max_step_time_us = aggregate.max_step_time_us.max(step.step_time_us);
            aggregate.max_beta = aggregate.max_beta.max(step.beta);
            aggregate.max_heating_mw = aggregate.max_heating_mw.max(step.heating_mw);
            aggregate.vessel_contact_events += usize::from(step.vessel_contact);
            aggregate.pf_constraint_events += usize::from(step.pf_constraint_active);
            aggregate.heating_constraint_events += usize::from(step.heating_constraint_active);
            next_tick += step_duration;
        }

        let wall_time_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        Ok(self
            .sim
            .finalize_report(steps, shot_duration_s, wall_time_ms, aggregate))
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
    fn test_run_shot_hardened_rejects_target_hz_mismatch() {
        let sim = RustFlightSim::new(6.2, 0.0, 5000.0).expect("valid sim");
        let mut driver = RtcDriver::new(
            sim,
            RtcConfig {
                target_hz: 4000.0,
                max_jitter_us: 10_000.0,
                use_busy_wait: false,
            },
        );
        assert!(driver.run_shot_hardened(0.01).is_err());
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
