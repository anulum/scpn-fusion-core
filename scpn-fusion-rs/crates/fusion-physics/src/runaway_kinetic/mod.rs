// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Full Runaway Kinetics
//! Full-fidelity radius-pitch-momentum runaway-electron kinetics.

mod coefficients;
mod grid;
mod operator;
mod solver;

pub use coefficients::RunawayKineticCoefficients;
pub use grid::RunawayKineticGrid;
pub use operator::{RunawayKineticGeometry, RunawayKineticOperator, RunawayKineticTendencies};
pub use solver::{RunawayKineticMoments, RunawayKineticSolver, RunawayKineticTrajectory};
