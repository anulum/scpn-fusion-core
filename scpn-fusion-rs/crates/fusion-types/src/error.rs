// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Error
//! Shared typed error and result contracts for native fusion crates.

use thiserror::Error;

/// Errors returned by native SCPN Fusion computations and data loading.
#[derive(Error, Debug)]
pub enum FusionError {
    /// An iterative solver diverged instead of satisfying its convergence gate.
    #[error("Solver diverged at iteration {iteration}: {message}")]
    SolverDiverged {
        /// Zero-based iteration at which divergence was detected.
        iteration: usize,
        /// Human-readable divergence diagnostic.
        message: String,
    },

    /// Configuration input is absent, inconsistent, or outside its contract.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// A requested two-dimensional grid index is outside the allocated shape.
    #[error("Grid index out of bounds: row={row}, col={col}")]
    GridOutOfBounds {
        /// Requested row index.
        row: usize,
        /// Requested column index.
        col: usize,
    },

    /// A state or parameter violates a required physical invariant.
    #[error("Physics constraint violated: {0}")]
    PhysicsViolation(String),

    /// Filesystem or stream input/output failed.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization or deserialization failed.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// A linear-algebra operation failed or returned an invalid shape.
    #[error("Linear algebra error: {0}")]
    LinAlg(String),
}

/// Native result alias using [`FusionError`].
pub type FusionResult<T> = Result<T, FusionError>;
