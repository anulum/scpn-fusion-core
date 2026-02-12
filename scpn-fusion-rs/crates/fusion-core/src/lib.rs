// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Fusion Core
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Grad-Shafranov kernel and equilibrium solver.
//!
//! Stage 3: core kernel modules
//! Stage 4: ignition, transport, stability, RF heating

pub mod bfield;
pub mod ignition;
pub mod kernel;
pub mod rf_heating;
pub mod source;
pub mod stability;
pub mod transport;
pub mod vacuum;
pub mod xpoint;
