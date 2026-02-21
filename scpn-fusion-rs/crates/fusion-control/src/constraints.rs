// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Actuator Constraints
// © 1998–2026 Miroslav Šotek. All rights reserved.
// ─────────────────────────────────────────────────────────────────────
//! Physical safety constraints for tokamak actuators.
//! Enforces current limits and slew rates (voltage limits).

/// Physical constraints for a single coil or heating system.
#[derive(Debug, Clone, Copy)]
pub struct PhysicalConstraint {
    pub max_value: f64,
    pub min_value: f64,
    pub max_slew_rate: f64, // max change per second
}

impl PhysicalConstraint {
    pub fn new(min: f64, max: f64, slew: f64) -> Self {
        Self {
            min_value: min,
            max_value: max,
            max_slew_rate: slew,
        }
    }

    /// Clamps the requested command based on limits and current state.
    pub fn enforce(&self, requested: f64, current: f64, dt: f64) -> f64 {
        // 1. Slew Rate (Voltage limit)
        let max_delta = self.max_slew_rate * dt;
        let delta = (requested - current).clamp(-max_delta, max_delta);
        let slewed = current + delta;

        // 2. Absolute Limits
        slewed.clamp(self.min_value, self.max_value)
    }
}

/// Global safety envelope for all actuators.
pub struct SafetyEnvelope {
    pub pf_coils: PhysicalConstraint,
    pub heating: PhysicalConstraint,
}

impl Default for SafetyEnvelope {
    fn default() -> Self {
        Self {
            // Default ITER-like limits: 50kA max, 10kA/s slew
            pf_coils: PhysicalConstraint::new(-1e6, 1e6, 1e5),
            // Default Heating: 0-100MW, 10MW/s slew
            heating: PhysicalConstraint::new(0.0, 100.0, 10.0),
        }
    }
}
