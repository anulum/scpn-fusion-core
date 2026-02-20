// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Fusion Control
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Control systems modules.
//!
//! Stage 7: PID, optimal, MPC, SNN, digital twin, SPI, SOC-learning, analytic.

pub mod analytic;
pub mod digital_twin;
pub mod flight_sim;
pub mod mpc;
pub mod optimal;
pub mod pid;
pub mod realtime;
pub mod snn;
pub mod soc_learning;
pub mod spi;
