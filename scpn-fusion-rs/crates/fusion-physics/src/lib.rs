// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Fusion Physics
//! Standalone physics modules for SCPN Fusion Core.
//!
//! Stage 5: sandpile, sawtooth, FNO, turbulence, Hall-MHD,
//! compact optimizer, design scanner, FRC analytical contracts,
//! pulsed Hall-MHD flux-carrier contracts, MRTI growth-spectrum contracts,
//! Faraday recovery contracts,
//! supplied-current pulsed-compression contracts, and conservative FRC
//! tilt-mode diagnostics.

pub mod compact_optimizer;
pub mod compression;
pub mod design_scanner;
pub mod faraday_recovery;
pub mod fno;
pub mod fokker_planck;
pub mod frc;
pub mod gk_nonlinear;
pub mod hall_mhd;
pub mod hall_mhd_pulsed;
pub mod mrti;
pub mod sandpile;
pub mod sawtooth;
pub mod tilt_mode_frc;
pub mod turbulence;
pub mod rmf_control;
pub mod precision_pacer;
