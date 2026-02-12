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
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self {
        PIDController {
            kp,
            ki,
            kd,
            err_sum: 0.0,
            last_err: 0.0,
        }
    }

    /// Default radial position controller.
    pub fn radial() -> Self {
        Self::new(PID_R_KP, PID_R_KI, PID_R_KD)
    }

    /// Default vertical position controller.
    pub fn vertical() -> Self {
        Self::new(PID_Z_KP, PID_Z_KI, PID_Z_KD)
    }

    /// One PID step. Returns control output.
    pub fn step(&mut self, error: f64) -> f64 {
        self.err_sum += error;
        let d_err = error - self.last_err;
        self.last_err = error;
        self.kp * error + self.ki * self.err_sum + self.kd * d_err
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
    pub fn new(target_r: f64, target_z: f64) -> Self {
        IsoFluxController {
            pid_r: PIDController::radial(),
            pid_z: PIDController::vertical(),
            target_r,
            target_z,
            r_history: Vec::new(),
            z_history: Vec::new(),
        }
    }

    /// Compute coil corrections given measured position.
    /// Returns (ctrl_radial, ctrl_vertical).
    pub fn step(&mut self, measured_r: f64, measured_z: f64) -> (f64, f64) {
        let err_r = self.target_r - measured_r;
        let err_z = self.target_z - measured_z;
        let ctrl_r = self.pid_r.step(err_r);
        let ctrl_z = self.pid_z.step(err_z);
        self.r_history.push(measured_r);
        self.z_history.push(measured_z);
        (ctrl_r, ctrl_z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pid_zero_error() {
        let mut pid = PIDController::new(1.0, 0.1, 0.5);
        let out = pid.step(0.0);
        assert!((out).abs() < 1e-10, "Zero error → zero output: {out}");
    }

    #[test]
    fn test_pid_proportional() {
        let mut pid = PIDController::new(2.0, 0.0, 0.0);
        let out = pid.step(5.0);
        assert!(
            (out - 10.0).abs() < 1e-10,
            "Pure P: 2.0 * 5.0 = 10.0: {out}"
        );
    }

    #[test]
    fn test_pid_integral_accumulates() {
        let mut pid = PIDController::new(0.0, 1.0, 0.0);
        pid.step(1.0);
        pid.step(1.0);
        let out = pid.step(1.0);
        assert!((out - 3.0).abs() < 1e-10, "Integral of three 1s = 3: {out}");
    }

    #[test]
    fn test_isoflux_converges() {
        let mut ctrl = IsoFluxController::new(6.2, 0.0);
        // Simple plant model: position moves toward target under control
        let mut r = 5.0;
        let mut z = 1.0;
        for _ in 0..100 {
            let (cr, cz) = ctrl.step(r, z);
            // Simple plant: position += gain * control
            r += 0.01 * cr;
            z += 0.01 * cz;
        }
        assert!((r - 6.2).abs() < 0.5, "R should approach target: {r}");
        assert!((z).abs() < 0.5, "Z should approach 0: {z}");
    }
}
