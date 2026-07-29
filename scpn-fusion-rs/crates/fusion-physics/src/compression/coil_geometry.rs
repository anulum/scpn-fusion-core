// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Compression Coil Geometry
//! Uniform-solenoid compression coil contract.

#[derive(Debug, Clone, Copy, PartialEq)]
/// Geometry and electrical limits of a pulsed-compression coil.
pub struct CoilGeometry {
    /// Number of turns.
    pub n_turns: u32,
    /// Axial coil length in metres.
    pub l_coil_m: f64,
    /// Coil radius in metres.
    pub r_coil_m: f64,
    /// Coil inductance in henries.
    pub l_inductance_h: f64,
    /// Coil resistance in ohms.
    pub r_resistance_ohm: f64,
    /// Bank voltage maximum in volts.
    pub bank_voltage_max_v: f64,
}

impl CoilGeometry {
    /// Validate the public parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.n_turns == 0 {
            return Err("coil.n_turns must be positive".to_string());
        }
        for (name, value) in [
            ("coil.l_coil_m", self.l_coil_m),
            ("coil.r_coil_m", self.r_coil_m),
            ("coil.l_inductance_h", self.l_inductance_h),
            ("coil.r_resistance_ohm", self.r_resistance_ohm),
            ("coil.bank_voltage_max_v", self.bank_voltage_max_v),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{name} must be positive"));
            }
        }
        Ok(())
    }
}
