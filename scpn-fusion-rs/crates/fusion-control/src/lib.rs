// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Fusion Control
//! Control systems modules.
//!
//! Stage 7: PID, optimal, MPC, SNN, digital twin, SPI, SOC-learning, analytic.

pub mod analytic;
pub mod constraints;
pub mod digital_twin;
pub mod flight_sim;
pub mod mpc;
pub mod optimal;
pub mod pid;
pub mod realtime;
pub mod snn;
pub mod soc_learning;
pub mod spi;
pub mod spi_ablation;
pub mod telemetry;
