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
use std::time::Instant;

/// Shot result metrics for analysis.
#[derive(Debug, Clone)]
pub struct SimulationReport {
    pub steps: usize,
    pub duration_s: f64,
    pub wall_time_ms: f64,
    pub max_step_time_us: f64,
    pub mean_abs_r_error: f64,
    pub mean_abs_z_error: f64,
    pub disrupted: bool,
    pub r_history: Vec<f64>,
    pub z_history: Vec<f64>,
    pub ip_history: Vec<f64>,
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
    // Actuator States (for slew rate tracking)
    pub pf_states: Vec<f64>,
}

impl RustFlightSim {
    pub fn new(target_r: f64, target_z: f64, control_hz: f64) -> FusionResult<Self> {
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
            pf_states: vec![0.0, 0.0], // R, Z actuators
        })
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
        let t_start = Instant::now();
        let steps = (shot_duration_s / self.control_dt) as usize;
        let step_duration = std::time::Duration::from_secs_f64(self.control_dt);

        let mut r_err_sum = 0.0;
        let mut z_err_sum = 0.0;
        let mut max_step_us = 0.0;
        let mut disrupted = false;

        let mut next_tick = Instant::now();

        for t in 0..steps {
            if deterministic {
                while Instant::now() < next_tick {
                    std::hint::spin_loop();
                }
            }

            let t_step_start = Instant::now();
            let time_s = t as f64 * self.control_dt;

            // 1. Physics Evolution (Plant)
            self.curr_ip_ma = 5.0 + (10.0 * time_s / shot_duration_s);
            self.curr_beta = 1.0 + (0.01 * time_s);

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
            self.pf_states[0] = ctrl_r;
            self.pf_states[1] = ctrl_z;

            // TODO: Enforce heating constraints when heating control is active

            // 3. Apply Actuators (with delay/lag)
            use ndarray::Array1;
            let actions = self
                .delay_line
                .push(Array1::from_vec(vec![ctrl_r, ctrl_z]))
                .map_err(|e| FusionError::ConfigError(e.to_string()))?;

            let applied_r = actions[0];
            let applied_z = actions[1];

            // 4. Update state based on applied control
            self.curr_r += applied_r * self.control_dt;
            self.curr_z += applied_z * self.control_dt;

            // Record Telemetry (Zero Allocation)
            self.telemetry
                .record(self.curr_r, self.curr_z, self.curr_ip_ma, self.curr_beta);

            // 5. Metrics
            let r_err = (self.curr_r - self.controller.target_r).abs();
            let z_err = (self.curr_z - self.controller.target_z).abs();
            r_err_sum += r_err;
            z_err_sum += z_err;

            if r_err > 0.5 || z_err > 0.5 {
                disrupted = true;
            }

            let step_us = t_step_start.elapsed().as_secs_f64() * 1_000_000.0;
            if step_us > max_step_us {
                max_step_us = step_us;
            }

            next_tick += step_duration;
        }

        let wall_time = t_start.elapsed().as_secs_f64() * 1000.0;

        Ok(SimulationReport {
            steps,
            duration_s: shot_duration_s,
            wall_time_ms: wall_time,
            max_step_time_us: max_step_us,
            mean_abs_r_error: r_err_sum / steps as f64,
            mean_abs_z_error: z_err_sum / steps as f64,
            disrupted,
            r_history: self.telemetry.r_axis.get_view(),
            z_history: self.telemetry.z_axis.get_view(),
            ip_history: self.telemetry.ip_ma.get_view(),
        })
    }
}
