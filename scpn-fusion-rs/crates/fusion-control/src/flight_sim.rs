// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Rust Flight Simulator
// © 1998–2026 Miroslav Šotek. All rights reserved.
// ─────────────────────────────────────────────────────────────────────
//! High-speed (10kHz+) tokamak flight simulator.
//!
//! Migrated from `tokamak_flight_sim.py` to enable industry-standard
//! control loop frequencies without Python/GIL overhead.

use crate::constraints::SafetyEnvelope;
use crate::digital_twin::ActuatorDelayLine;
use crate::pid::IsoFluxController;
use crate::telemetry::TelemetrySuite;
use fusion_types::error::{FusionError, FusionResult};
use ndarray::Array1;
use std::time::{Duration, Instant};

/// Shot result metrics for analysis.
#[derive(Debug, Clone)]
pub struct SimulationReport {
    pub steps: usize,
    pub duration_s: f64,
    pub wall_time_ms: f64,
    pub max_step_time_us: f64,
    pub mean_abs_r_error: f64,
    pub mean_abs_z_error: f64,
    pub final_beta: f64,
    pub final_heating_mw: f64,
    pub max_beta: f64,
    pub max_heating_mw: f64,
    pub vessel_contact_events: usize,
    pub pf_constraint_events: usize,
    pub heating_constraint_events: usize,
    pub retained_steps: usize,
    pub history_truncated: bool,
    pub disrupted: bool,
    pub r_history: Vec<f64>,
    pub z_history: Vec<f64>,
    pub ip_history: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct StepMetrics {
    pub r_error: f64,
    pub z_error: f64,
    pub disrupted: bool,
    pub step_time_us: f64,
    pub beta: f64,
    pub heating_mw: f64,
    pub vessel_contact: bool,
    pub pf_constraint_active: bool,
    pub heating_constraint_active: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct ShotAggregate {
    pub max_step_time_us: f64,
    pub r_err_sum: f64,
    pub z_err_sum: f64,
    pub disrupted: bool,
    pub max_beta: f64,
    pub max_heating_mw: f64,
    pub vessel_contact_events: usize,
    pub pf_constraint_events: usize,
    pub heating_constraint_events: usize,
}

/// High-speed simulation engine.
pub struct RustFlightSim {
    pub controller: IsoFluxController,
    pub delay_line: ActuatorDelayLine,
    pub telemetry: TelemetrySuite,
    pub constraints: SafetyEnvelope,
    pub control_dt: f64,
    // Simulated state (Simplified Physics Model for high-speed loop)
    pub curr_r: f64,
    pub curr_z: f64,
    pub curr_ip_ma: f64,
    pub curr_beta: f64,
    pub curr_heating_mw: f64,
    // Actuator States (for slew rate tracking)
    pub pf_states: Vec<f64>,
}

impl RustFlightSim {
    fn validate_runtime_state(&self) -> FusionResult<()> {
        let state_fields = [
            ("curr_r", self.curr_r),
            ("curr_z", self.curr_z),
            ("curr_ip_ma", self.curr_ip_ma),
            ("curr_beta", self.curr_beta),
            ("curr_heating_mw", self.curr_heating_mw),
        ];
        for (name, value) in state_fields {
            if !value.is_finite() {
                return Err(FusionError::PhysicsViolation(format!(
                    "non-finite simulator state {name}={value}"
                )));
            }
        }
        if !(0.0..=100.0).contains(&self.curr_heating_mw) {
            return Err(FusionError::PhysicsViolation(format!(
                "heating command out of physical envelope: {} MW",
                self.curr_heating_mw
            )));
        }
        if !(0.2..=10.0).contains(&self.curr_beta) {
            return Err(FusionError::PhysicsViolation(format!(
                "beta out of physical envelope: {}",
                self.curr_beta
            )));
        }
        if self.pf_states.len() != 2 || self.pf_states.iter().any(|v| !v.is_finite()) {
            return Err(FusionError::PhysicsViolation(
                "PF actuator state vector invalid (must be 2 finite elements)".to_string(),
            ));
        }
        Ok(())
    }

    pub fn new(target_r: f64, target_z: f64, control_hz: f64) -> FusionResult<Self> {
        if !target_r.is_finite() || !target_z.is_finite() {
            return Err(FusionError::ConfigError(
                "target_r/target_z must be finite".to_string(),
            ));
        }
        if !control_hz.is_finite() || control_hz <= 0.0 {
            return Err(FusionError::ConfigError(
                "control_hz must be finite and > 0".to_string(),
            ));
        }
        let control_dt = 1.0 / control_hz;
        let mut controller = IsoFluxController::new(target_r, target_z)?;

        // Scale gains from 100Hz baseline to current frequency to maintain stability
        let dt_ref = 0.01;
        let scale_i = control_dt / dt_ref;
        let scale_d = dt_ref / control_dt;

        controller.pid_r.ki *= scale_i;
        controller.pid_r.kd *= scale_d;
        controller.pid_z.ki *= scale_i;
        controller.pid_z.kd *= scale_d;

        Ok(Self {
            controller,
            // 50ms delay @ control_hz
            delay_line: ActuatorDelayLine::new(2, (0.05 * control_hz) as usize, 0.5)
                .map_err(|e| FusionError::ConfigError(e.to_string()))?,
            telemetry: TelemetrySuite::new(1_000_000), // 1M points capacity
            constraints: SafetyEnvelope::default(),
            control_dt,
            curr_r: target_r,
            curr_z: target_z,
            curr_ip_ma: 5.0,
            curr_beta: 1.0,
            curr_heating_mw: 20.0,
            pf_states: vec![0.0, 0.0], // R, Z actuators
        })
    }

    fn validate_shot_duration(&self, shot_duration_s: f64) -> FusionResult<usize> {
        if !shot_duration_s.is_finite() || shot_duration_s <= 0.0 {
            return Err(FusionError::ConfigError(
                "shot_duration_s must be finite and > 0".to_string(),
            ));
        }
        let steps = (shot_duration_s / self.control_dt) as usize;
        if steps == 0 {
            return Err(FusionError::ConfigError(
                "shot_duration_s too short for selected control_dt".to_string(),
            ));
        }
        Ok(steps)
    }

    pub fn reset_for_shot(&mut self) {
        self.telemetry.clear();
        self.delay_line.reset();
        self.pf_states.fill(0.0);
    }

    pub fn reset_plasma_state(&mut self) {
        self.reset_for_shot();
        self.curr_r = self.controller.target_r;
        self.curr_z = self.controller.target_z;
        self.curr_ip_ma = 5.0;
        self.curr_beta = 1.0;
        self.curr_heating_mw = 20.0;
    }

    pub fn plasma_state(&self) -> (f64, f64, f64, f64, f64) {
        (
            self.curr_r,
            self.curr_z,
            self.curr_ip_ma,
            self.curr_beta,
            self.curr_heating_mw,
        )
    }

    pub fn prepare_shot(&mut self, shot_duration_s: f64) -> FusionResult<usize> {
        let steps = self.validate_shot_duration(shot_duration_s)?;
        self.validate_runtime_state()?;
        self.reset_for_shot();
        Ok(steps)
    }

    pub fn step_once(
        &mut self,
        step_index: usize,
        shot_duration_s: f64,
    ) -> FusionResult<StepMetrics> {
        if !shot_duration_s.is_finite() || shot_duration_s <= 0.0 {
            return Err(FusionError::ConfigError(
                "shot_duration_s must be finite and > 0".to_string(),
            ));
        }
        let steps = (shot_duration_s / self.control_dt) as usize;
        if steps == 0 {
            return Err(FusionError::ConfigError(
                "shot_duration_s too short for selected control_dt".to_string(),
            ));
        }
        if step_index >= steps {
            return Err(FusionError::ConfigError(format!(
                "step_index {step_index} out of bounds for shot with {steps} steps"
            )));
        }
        self.validate_runtime_state()?;
        let t_step_start = Instant::now();
        let phase = (step_index as f64 * self.control_dt / shot_duration_s).clamp(0.0, 1.0);

        // 1. Physics Evolution (Plant)
        self.curr_ip_ma = 5.0 + 10.0 * phase;
        // Heating command acts as a bounded actuator that drives beta.
        // This fills the heating-control safety gap with a stable surrogate
        // coupling: higher constrained heating -> higher target beta.
        let heating_request_mw = 20.0 + 60.0 * phase;
        self.curr_heating_mw = self.constraints.heating.enforce(
            heating_request_mw,
            self.curr_heating_mw,
            self.control_dt,
        );
        let heating_constraint_active = (self.curr_heating_mw - heating_request_mw).abs() > 1e-12;
        let beta_target = 0.6 + 0.03 * self.curr_heating_mw;
        self.curr_beta += 0.5 * (beta_target - self.curr_beta) * self.control_dt;
        self.curr_beta = self.curr_beta.clamp(0.2, 10.0);

        // Natural Drifts (Shafranov shift + Vertical instability)
        self.curr_r += 0.01 * self.curr_beta * self.control_dt;
        self.curr_z += 0.02 * self.control_dt;

        // Clamp to vessel boundaries
        self.curr_r = self.curr_r.clamp(2.0, 10.0);
        self.curr_z = self.curr_z.clamp(-6.0, 6.0);

        // 2. Control Action
        let (requested_r, requested_z) = self.controller.step(self.curr_r, self.curr_z)?;

        // 2b. Safety Enforcement (Hardening Task 1)
        let ctrl_r =
            self.constraints
                .pf_coils
                .enforce(requested_r, self.pf_states[0], self.control_dt);
        let ctrl_z =
            self.constraints
                .pf_coils
                .enforce(requested_z, self.pf_states[1], self.control_dt);
        let pf_constraint_active =
            (ctrl_r - requested_r).abs() > 1e-12 || (ctrl_z - requested_z).abs() > 1e-12;
        self.pf_states[0] = ctrl_r;
        self.pf_states[1] = ctrl_z;

        // 3. Apply Actuators (with delay/lag)
        let actions = self
            .delay_line
            .push(Array1::from_vec(vec![ctrl_r, ctrl_z]))
            .map_err(|e| FusionError::ConfigError(e.to_string()))?;

        let applied_r = actions[0];
        let applied_z = actions[1];

        // 4. Update state based on applied control
        let next_r = self.curr_r + applied_r * self.control_dt;
        let next_z = self.curr_z + applied_z * self.control_dt;
        self.curr_r = next_r.clamp(2.0, 10.0);
        self.curr_z = next_z.clamp(-6.0, 6.0);
        let vessel_contact = (next_r - self.curr_r).abs() > f64::EPSILON
            || (next_z - self.curr_z).abs() > f64::EPSILON;
        self.validate_runtime_state()?;

        // Record Telemetry (Zero Allocation)
        self.telemetry
            .record(self.curr_r, self.curr_z, self.curr_ip_ma, self.curr_beta);

        // 5. Metrics
        let r_err = (self.curr_r - self.controller.target_r).abs();
        let z_err = (self.curr_z - self.controller.target_z).abs();
        let disrupted = r_err > 0.5 || z_err > 0.5 || vessel_contact;
        let step_time_us = t_step_start.elapsed().as_secs_f64() * 1_000_000.0;

        Ok(StepMetrics {
            r_error: r_err,
            z_error: z_err,
            disrupted,
            step_time_us,
            beta: self.curr_beta,
            heating_mw: self.curr_heating_mw,
            vessel_contact,
            pf_constraint_active,
            heating_constraint_active,
        })
    }

    pub fn finalize_report(
        &self,
        steps: usize,
        shot_duration_s: f64,
        wall_time_ms: f64,
        aggregate: ShotAggregate,
    ) -> SimulationReport {
        let r_history = self.telemetry.r_axis.get_view();
        let z_history = self.telemetry.z_axis.get_view();
        let ip_history = self.telemetry.ip_ma.get_view();
        let retained_steps = r_history.len();
        let history_truncated = retained_steps < steps;

        SimulationReport {
            steps,
            duration_s: shot_duration_s,
            wall_time_ms,
            max_step_time_us: aggregate.max_step_time_us,
            mean_abs_r_error: aggregate.r_err_sum / steps as f64,
            mean_abs_z_error: aggregate.z_err_sum / steps as f64,
            final_beta: self.curr_beta,
            final_heating_mw: self.curr_heating_mw,
            max_beta: aggregate.max_beta,
            max_heating_mw: aggregate.max_heating_mw,
            vessel_contact_events: aggregate.vessel_contact_events,
            pf_constraint_events: aggregate.pf_constraint_events,
            heating_constraint_events: aggregate.heating_constraint_events,
            retained_steps,
            history_truncated,
            disrupted: aggregate.disrupted,
            r_history,
            z_history,
            ip_history,
        }
    }

    /// Execute a shot at the target frequency.
    ///
    /// If `deterministic` is true, uses a high-precision busy-wait loop
    /// to ensure sub-microsecond timing accuracy (at the cost of CPU usage).
    pub fn run_shot(
        &mut self,
        shot_duration_s: f64,
        deterministic: bool,
    ) -> FusionResult<SimulationReport> {
        let steps = self.prepare_shot(shot_duration_s)?;
        let t_start = Instant::now();
        let step_duration = Duration::from_secs_f64(self.control_dt);

        let mut aggregate = ShotAggregate {
            max_step_time_us: 0.0,
            r_err_sum: 0.0,
            z_err_sum: 0.0,
            disrupted: false,
            max_beta: self.curr_beta,
            max_heating_mw: self.curr_heating_mw,
            vessel_contact_events: 0,
            pf_constraint_events: 0,
            heating_constraint_events: 0,
        };

        let mut next_tick = Instant::now();
        for step_idx in 0..steps {
            if deterministic {
                while Instant::now() < next_tick {
                    std::hint::spin_loop();
                }
            }
            let step = self.step_once(step_idx, shot_duration_s)?;
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

        let wall_time = t_start.elapsed().as_secs_f64() * 1000.0;
        Ok(self.finalize_report(steps, shot_duration_s, wall_time, aggregate))
    }
}

#[cfg(test)]
mod tests {
    use super::RustFlightSim;
    use crate::telemetry::TelemetrySuite;

    #[test]
    fn test_new_rejects_invalid_control_hz() {
        assert!(RustFlightSim::new(6.2, 0.0, 0.0).is_err());
        assert!(RustFlightSim::new(6.2, 0.0, f64::NAN).is_err());
    }

    #[test]
    fn test_run_shot_rejects_invalid_duration() {
        let mut sim = RustFlightSim::new(6.2, 0.0, 10_000.0).expect("valid sim");
        assert!(sim.run_shot(0.0, false).is_err());
        assert!(sim.run_shot(f64::NAN, false).is_err());
    }

    #[test]
    fn test_run_shot_keeps_beta_and_heating_finite() {
        let mut sim = RustFlightSim::new(6.2, 0.0, 10_000.0).expect("valid sim");
        let report = sim.run_shot(0.02, false).expect("shot should run");
        assert_eq!(report.steps, 200);
        assert!(sim.curr_beta.is_finite());
        assert!(sim.curr_heating_mw.is_finite());
        assert!(report.final_beta.is_finite());
        assert!(report.final_heating_mw.is_finite());
        assert!(report.max_beta.is_finite());
        assert!(report.max_heating_mw.is_finite());
        assert!((0.2..=10.0).contains(&report.final_beta));
        assert!((0.2..=10.0).contains(&report.max_beta));
        assert!(report.max_beta >= report.final_beta);
        assert!((0.0..=100.0).contains(&report.final_heating_mw));
        assert!((0.0..=100.0).contains(&report.max_heating_mw));
        assert!(report.max_heating_mw >= report.final_heating_mw);
        assert!((0.0..=100.0).contains(&sim.curr_heating_mw));
        assert!(report.vessel_contact_events <= report.steps);
        assert!(report.pf_constraint_events <= report.steps);
        assert!(report.heating_constraint_events <= report.steps);
        assert_eq!(report.retained_steps, report.steps);
        assert!(!report.history_truncated);
    }

    #[test]
    fn test_run_shot_resets_telemetry_between_runs() {
        let mut sim = RustFlightSim::new(6.2, 0.0, 10_000.0).expect("valid sim");
        let first = sim.run_shot(0.01, false).expect("first shot");
        assert_eq!(first.steps, 100);
        assert_eq!(first.r_history.len(), 100);

        let second = sim.run_shot(0.005, false).expect("second shot");
        assert_eq!(second.steps, 50);
        assert_eq!(second.r_history.len(), 50);
        assert_eq!(second.z_history.len(), 50);
        assert_eq!(second.ip_history.len(), 50);
    }

    #[test]
    fn test_step_once_rejects_non_finite_runtime_state() {
        let mut sim = RustFlightSim::new(6.2, 0.0, 10_000.0).expect("valid sim");
        sim.curr_beta = f64::NAN;
        assert!(sim.step_once(0, 0.01).is_err());
    }

    #[test]
    fn test_step_once_rejects_out_of_bounds_index() {
        let mut sim = RustFlightSim::new(6.2, 0.0, 10_000.0).expect("valid sim");
        assert!(sim.step_once(100, 0.01).is_err());
    }

    #[test]
    fn test_report_flags_history_truncation() {
        let mut sim = RustFlightSim::new(6.2, 0.0, 10_000.0).expect("valid sim");
        sim.telemetry = TelemetrySuite::new(16);
        let report = sim.run_shot(0.01, false).expect("shot should run");
        assert_eq!(report.steps, 100);
        assert_eq!(report.retained_steps, 16);
        assert!(report.history_truncated);
        assert_eq!(report.r_history.len(), 16);
    }

    #[test]
    fn test_reset_plasma_state_restores_nominal_values() {
        let mut sim = RustFlightSim::new(6.2, 0.0, 10_000.0).expect("valid sim");
        let _ = sim.run_shot(0.01, false).expect("shot should run");
        sim.reset_plasma_state();
        let (r, z, ip, beta, heating) = sim.plasma_state();
        assert!((r - 6.2).abs() < 1e-12);
        assert!(z.abs() < 1e-12);
        assert!((ip - 5.0).abs() < 1e-12);
        assert!((beta - 1.0).abs() < 1e-12);
        assert!((heating - 20.0).abs() < 1e-12);
    }
}
