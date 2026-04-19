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
//! compact optimizer, design scanner.

pub mod compact_optimizer;
pub mod design_scanner;
pub mod fno;
pub mod fokker_planck;
pub mod gk_nonlinear;
pub mod hall_mhd;
pub mod sandpile;
pub mod sawtooth;
pub mod turbulence;
