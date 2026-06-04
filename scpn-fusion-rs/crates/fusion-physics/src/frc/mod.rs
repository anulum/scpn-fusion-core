// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — FRC Physics
//! Field-reversed-configuration physics surfaces.

pub mod data;
pub mod rigid_rotor;

pub use data::{FrcEquilibriumState, FrcSolverError, RigidRotorFrcInputs};
pub use rigid_rotor::{ion_gyroradius_m, solve_frc_equilibrium};
