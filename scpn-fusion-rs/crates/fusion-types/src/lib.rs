// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Fusion Types
//! Shared configuration, constants, errors, and state containers.
//!
//! These types define the native Rust contracts exchanged by the solver,
//! diagnostics, transport, engineering, and Python-binding crates.
#![deny(missing_docs)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

/// Governed reactor and solver configuration schema.
pub mod config;
/// Physical and mathematical constants used across native crates.
pub mod constants;
/// Shared typed error and result contracts.
pub mod error;
/// Grid, plasma, transport, and solver-result state containers.
pub mod state;
