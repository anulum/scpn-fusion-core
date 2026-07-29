// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Fusion Nuclear
//! Nuclear engineering modules.
//!
//! Stage 6: TEMHD, neutronics, sputtering, wall interaction, divertor, BOP.
#![deny(missing_docs)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

pub mod bop;
pub mod divertor;
pub mod neutronics;
pub mod pwi;
pub mod temhd;
pub mod wall_interaction;
