// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Error
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FusionError {
    #[error("Solver diverged at iteration {iteration}: {message}")]
    SolverDiverged { iteration: usize, message: String },

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Grid index out of bounds: row={row}, col={col}")]
    GridOutOfBounds { row: usize, col: usize },

    #[error("Physics constraint violated: {0}")]
    PhysicsViolation(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Linear algebra error: {0}")]
    LinAlg(String),
}

pub type FusionResult<T> = Result<T, FusionError>;
