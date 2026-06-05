// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — MIF/FRC Compression
//! Pulsed MIF/FRC compression contracts.

pub mod coil_geometry;
pub mod pulsed;

pub use coil_geometry::CoilGeometry;
pub use pulsed::{
    adiabatic_temperature_update_ev, coil_field_t, initial_coil_circuit_state, plasma_volume_m3,
    run_coil_circuit, run_pulsed_compression, run_voltage_driven_pulsed_compression,
    spitzer_resistivity_ohm_m, step_coil_circuit, step_pulsed_compression, CoilCircuitState,
    PulsedCompressionConfig, PulsedCompressionState, VoltageDrivenPulsedCompressionResult,
};
