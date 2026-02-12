// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Constants
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
/// Vacuum permeability (H/m) - real SI value.
/// NOTE: The Python code uses vacuum_permeability=1.0 in configs (normalized units).
/// We support both via config override.
pub const MU0_SI: f64 = 1.2566370614e-6;

/// Elementary charge (C)
pub const Q_ELECTRON: f64 = 1.602176634e-19;

/// Deuterium mass (kg)
pub const M_DEUTERIUM: f64 = 3.3435837724e-27;

/// Tritium mass (kg)
pub const M_TRITIUM: f64 = 5.0073567446e-27;

/// D-T fusion energy release (J) - 17.6 MeV
pub const E_FUSION_DT: f64 = 17.6 * 1.602e-13;

/// Alpha particle energy fraction (3.5/17.6)
pub const ALPHA_FRACTION: f64 = 0.2;

/// Boltzmann constant (J/K)
pub const K_BOLTZMANN: f64 = 1.380649e-23;

/// Golden ratio (used in Lazarus bridge)
pub const PHI_GOLDEN: f64 = 1.618033988749895;
