// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Actuator Constraints
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
            // Default ITER-like PF/CS limits: +/-50 kA absolute, 10 kA/s slew.
            // (Previously shipped +/-1 MA at 100 kA/s while this comment claimed
            // 50 kA -- an ineffective envelope that bounded nothing physical.)
            pf_coils: PhysicalConstraint::new(-5.0e4, 5.0e4, 1.0e4),
            // Default Heating: 0-100MW, 10MW/s slew
            heating: PhysicalConstraint::new(0.0, 100.0, 10.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_pf_coil_limits_are_physical() {
        let env = SafetyEnvelope::default();
        assert_eq!(env.pf_coils.min_value, -5.0e4);
        assert_eq!(env.pf_coils.max_value, 5.0e4);
        assert_eq!(env.pf_coils.max_slew_rate, 1.0e4);
        // Not the old non-physical +/-1 MA default.
        assert!(env.pf_coils.max_value < 1.0e6);
    }

    #[test]
    fn enforce_clamps_to_absolute_limit() {
        // Huge slew allowance isolates the absolute-limit clamp.
        let c = PhysicalConstraint::new(-5.0e4, 5.0e4, 1.0e9);
        assert_eq!(c.enforce(1.0e8, 0.0, 1.0), 5.0e4);
        assert_eq!(c.enforce(-1.0e8, 0.0, 1.0), -5.0e4);
    }

    #[test]
    fn enforce_respects_slew_rate() {
        // dt=1 s, slew=100 -> at most 100 change per step from the current value.
        let c = PhysicalConstraint::new(-5.0e4, 5.0e4, 100.0);
        assert_eq!(c.enforce(1000.0, 0.0, 1.0), 100.0);
        assert_eq!(c.enforce(-1000.0, 0.0, 1.0), -100.0);
    }
}
