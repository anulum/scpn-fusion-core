// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — PID
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! PID controller for tokamak position control.
//!
//! Port of `tokamak_flight_sim.py`.
//! Implements decoupled radial and vertical position PID.

use fusion_types::error::{FusionError, FusionResult};

/// Radial PID gains. Python: Kp=2.0, Ki=0.1, Kd=0.5.
const PID_R_KP: f64 = 2.0;
const PID_R_KI: f64 = 0.1;
const PID_R_KD: f64 = 0.5;

/// Vertical PID gains. Python: Kp=5.0, Ki=0.2, Kd=2.0.
const PID_Z_KP: f64 = 5.0;
const PID_Z_KI: f64 = 0.2;
const PID_Z_KD: f64 = 2.0;

/// Generic PID controller.
#[derive(Debug, Clone)]
pub struct PIDController {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    err_sum: f64,
    last_err: f64,
}

impl PIDController {
    pub fn new(kp: f64, ki: f64, kd: f64) -> FusionResult<Self> {
        if !kp.is_finite() || !ki.is_finite() || !kd.is_finite() {
            return Err(FusionError::ConfigError(
                "pid gains must be finite".to_string(),
            ));
        }
        Ok(PIDController {
            kp,
            ki,
            kd,
            err_sum: 0.0,
            last_err: 0.0,
        })
    }

    /// Default radial position controller.
    pub fn radial() -> FusionResult<Self> {
        Self::new(PID_R_KP, PID_R_KI, PID_R_KD)
    }

    /// Default vertical position controller.
    pub fn vertical() -> FusionResult<Self> {
        Self::new(PID_Z_KP, PID_Z_KI, PID_Z_KD)
    }

    /// One PID step. Returns control output.
    pub fn step(&mut self, error: f64) -> FusionResult<f64> {
        if !error.is_finite() {
            return Err(FusionError::ConfigError(
                "pid error input must be finite".to_string(),
            ));
        }
        self.err_sum += error;
        let d_err = error - self.last_err;
        self.last_err = error;
        Ok(self.kp * error + self.ki * self.err_sum + self.kd * d_err)
    }

    /// Reset accumulated state.
    pub fn reset(&mut self) {
        self.err_sum = 0.0;
        self.last_err = 0.0;
    }
}

/// Isoflux position controller with R and Z PIDs.
pub struct IsoFluxController {
    pub pid_r: PIDController,
    pub pid_z: PIDController,
    pub target_r: f64,
    pub target_z: f64,
    pub r_history: Vec<f64>,
    pub z_history: Vec<f64>,
}

impl IsoFluxController {
    pub fn new(target_r: f64, target_z: f64) -> FusionResult<Self> {
        if !target_r.is_finite() || !target_z.is_finite() {
            return Err(FusionError::ConfigError(
                "isoflux targets must be finite".to_string(),
            ));
        }
        Ok(IsoFluxController {
            pid_r: PIDController::radial()?,
            pid_z: PIDController::vertical()?,
            target_r,
            target_z,
            r_history: Vec::new(),
            z_history: Vec::new(),
        })
    }

    /// Compute coil corrections given measured position.
    /// Returns (ctrl_radial, ctrl_vertical).
    pub fn step(&mut self, measured_r: f64, measured_z: f64) -> FusionResult<(f64, f64)> {
        if !measured_r.is_finite() || !measured_z.is_finite() {
            return Err(FusionError::ConfigError(
                "isoflux measured position must be finite".to_string(),
            ));
        }
        let err_r = self.target_r - measured_r;
        let err_z = self.target_z - measured_z;
        let ctrl_r = self.pid_r.step(err_r)?;
        let ctrl_z = self.pid_z.step(err_z)?;
        
        // Note: history is now handled by the circular buffer in flight_sim.rs
        // to prevent large Vec allocations in the 10kHz loop.
        
        Ok((ctrl_r, ctrl_z))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pid_zero_error() {
        let mut pid = PIDController::new(1.0, 0.1, 0.5).expect("valid gains");
        let out = pid.step(0.0).expect("valid finite input");
        assert!((out).abs() < 1e-10, "Zero error → zero output: {out}");
    }

    #[test]
    fn test_pid_proportional() {
        let mut pid = PIDController::new(2.0, 0.0, 0.0).expect("valid gains");
        let out = pid.step(5.0).expect("valid finite input");
        assert!(
            (out - 10.0).abs() < 1e-10,
            "Pure P: 2.0 * 5.0 = 10.0: {out}"
        );
    }

    #[test]
    fn test_pid_integral_accumulates() {
        let mut pid = PIDController::new(0.0, 1.0, 0.0).expect("valid gains");
        pid.step(1.0).expect("valid finite input");
        pid.step(1.0).expect("valid finite input");
        let out = pid.step(1.0).expect("valid finite input");
        assert!((out - 3.0).abs() < 1e-10, "Integral of three 1s = 3: {out}");
    }

    #[test]
    fn test_isoflux_converges() {
        let mut ctrl = IsoFluxController::new(6.2, 0.0).expect("valid targets");
        // Simple plant model: position moves toward target under control
        let mut r = 5.0;
        let mut z = 1.0;
        for _ in 0..100 {
            let (cr, cz) = ctrl.step(r, z).expect("valid finite position inputs");
            // Simple plant: position += gain * control
            r += 0.01 * cr;
            z += 0.01 * cz;
        }
        assert!((r - 6.2).abs() < 0.5, "R should approach target: {r}");
        assert!((z).abs() < 0.5, "Z should approach 0: {z}");
    }

    #[test]
    fn test_pid_rejects_non_finite_gains_and_error() {
        assert!(PIDController::new(f64::NAN, 0.1, 0.2).is_err());
        let mut pid = PIDController::new(1.0, 0.1, 0.2).expect("valid gains");
        assert!(pid.step(f64::INFINITY).is_err());
    }

    #[test]
    fn test_isoflux_rejects_non_finite_targets_and_measurements() {
        assert!(IsoFluxController::new(f64::NAN, 0.0).is_err());
        let mut ctrl = IsoFluxController::new(6.2, 0.0).expect("valid targets");
        assert!(ctrl.step(6.0, f64::NAN).is_err());
    }
}
