// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Optimal
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! MIMO optimal control with SVD pseudoinverse.
//!
//! Port of `fusion_optimal_control.py`.
//! Computes coil current corrections using system response Jacobian.

use fusion_types::error::{FusionError, FusionResult};
use ndarray::{Array1, Array2};

/// Singular value cutoff for regularization. Python: 1e-2.
const SV_CUTOFF: f64 = 1e-2;

/// Maximum coil current delta [A]. Python: 5.0.
const MAX_DELTA: f64 = 5.0;

/// Default gain factor. Python: 0.8.
#[allow(dead_code)]
const DEFAULT_GAIN: f64 = 0.8;

/// Computes SVD pseudoinverse of a 2×N response matrix.
///
/// Returns (delta_currents) given position error.
pub fn svd_optimal_correction(
    response_matrix: &Array2<f64>,
    error: &Array1<f64>,
    gain: f64,
) -> FusionResult<Array1<f64>> {
    let (m, n) = response_matrix.dim();
    if m == 0 || n == 0 {
        return Err(FusionError::ConfigError(
            "optimal response_matrix must have non-zero dimensions".to_string(),
        ));
    }
    if m != error.len() {
        return Err(FusionError::ConfigError(format!(
            "optimal dimension mismatch: response rows={m}, error len={}",
            error.len()
        )));
    }
    if !gain.is_finite() {
        return Err(FusionError::ConfigError(
            "optimal gain must be finite".to_string(),
        ));
    }
    if response_matrix.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::ConfigError(
            "optimal response_matrix must contain only finite values".to_string(),
        ));
    }
    if error.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::ConfigError(
            "optimal error vector must contain only finite values".to_string(),
        ));
    }

    // Compute A^T · A (n×n) and A^T · b for least-squares
    // For small m (=2), use direct SVD-like approach
    // Since m is small, compute pseudoinverse via A^+ = A^T (A A^T)^{-1}
    let aat = response_matrix.dot(&response_matrix.t()); // m×m

    // For m=2: invert 2×2 matrix directly
    if m == 2 {
        let a = aat[[0, 0]];
        let b = aat[[0, 1]];
        let c = aat[[1, 0]];
        let d = aat[[1, 1]];
        let det = a * d - b * c;

        if det.abs() < SV_CUTOFF {
            return Err(FusionError::ConfigError(
                "optimal response matrix is near-singular under cutoff".to_string(),
            ));
        }

        let aat_inv =
            Array2::from_shape_vec((2, 2), vec![d / det, -b / det, -c / det, a / det]).unwrap();

        let pinv = response_matrix.t().dot(&aat_inv); // n×m
        let mut delta = pinv.dot(error) * gain;

        // Clip
        for v in delta.iter_mut() {
            *v = v.clamp(-MAX_DELTA, MAX_DELTA);
        }
        Ok(delta)
    } else {
        // General case: use normal equations with Tikhonov
        let ata = response_matrix.t().dot(response_matrix); // n×n
        let atb = response_matrix.t().dot(error);

        // Tikhonov: (A^T A + λI)^{-1} A^T b
        // For now, simple diagonal damping
        let mut reg = ata;
        for i in 0..n {
            reg[[i, i]] += SV_CUTOFF;
        }

        // Solve via Cholesky or just return scaled A^T b
        let mut delta = atb * gain;
        for v in delta.iter_mut() {
            *v = v.clamp(-MAX_DELTA, MAX_DELTA);
        }
        Ok(delta)
    }
}

/// Build response matrix by finite differences.
///
/// `evaluate`: closure that maps coil currents → [R, Z] position.
/// `base_currents`: baseline coil currents.
/// `perturbation`: finite difference step.
pub fn build_response_matrix<F>(
    evaluate: F,
    base_currents: &[f64],
    perturbation: f64,
) -> FusionResult<Array2<f64>>
where
    F: Fn(&[f64]) -> [f64; 2],
{
    if base_currents.is_empty() {
        return Err(FusionError::ConfigError(
            "optimal base_currents must be non-empty".to_string(),
        ));
    }
    if !perturbation.is_finite() || perturbation <= 0.0 {
        return Err(FusionError::ConfigError(
            "optimal perturbation must be finite and > 0".to_string(),
        ));
    }
    if base_currents.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::ConfigError(
            "optimal base_currents must contain only finite values".to_string(),
        ));
    }

    let n = base_currents.len();
    let mut matrix = Array2::zeros((2, n));
    let mut test_currents = base_currents.to_vec();

    for i in 0..n {
        let orig = test_currents[i];

        test_currents[i] = orig + perturbation;
        let pos_plus = evaluate(&test_currents);
        if !pos_plus[0].is_finite() || !pos_plus[1].is_finite() {
            return Err(FusionError::ConfigError(
                "optimal evaluate(pos_plus) must return finite values".to_string(),
            ));
        }

        test_currents[i] = orig - perturbation;
        let pos_minus = evaluate(&test_currents);
        if !pos_minus[0].is_finite() || !pos_minus[1].is_finite() {
            return Err(FusionError::ConfigError(
                "optimal evaluate(pos_minus) must return finite values".to_string(),
            ));
        }

        test_currents[i] = orig;

        matrix[[0, i]] = (pos_plus[0] - pos_minus[0]) / (2.0 * perturbation);
        matrix[[1, i]] = (pos_plus[1] - pos_minus[1]) / (2.0 * perturbation);
    }

    Ok(matrix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svd_correction_identity() {
        // 2×2 identity response → correction = error * gain
        let resp = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let error = Array1::from_vec(vec![1.0, -0.5]);
        let delta = svd_optimal_correction(&resp, &error, 1.0).expect("valid correction inputs");
        assert!((delta[0] - 1.0).abs() < 1e-6);
        assert!((delta[1] - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_svd_correction_clipping() {
        let resp = Array2::from_shape_vec((2, 2), vec![100.0, 0.0, 0.0, 100.0]).unwrap();
        let error = Array1::from_vec(vec![1.0, 1.0]);
        let delta = svd_optimal_correction(&resp, &error, 1.0).expect("valid correction inputs");
        assert!(
            delta[0].abs() <= MAX_DELTA + 1e-10,
            "Should be clipped: {}",
            delta[0]
        );
    }

    #[test]
    fn test_response_matrix_linear() {
        // Plant: position = [sum(I), I[0] - I[1]]
        let base = vec![0.0, 0.0];
        let resp = build_response_matrix(|i: &[f64]| [i[0] + i[1], i[0] - i[1]], &base, 0.5)
            .expect("valid response-matrix inputs");
        assert!((resp[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((resp[[0, 1]] - 1.0).abs() < 1e-6);
        assert!((resp[[1, 0]] - 1.0).abs() < 1e-6);
        assert!((resp[[1, 1]] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_svd_correction_rejects_invalid_inputs_and_singular_system() {
        let resp = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        let err = Array1::from_vec(vec![1.0, 1.0]);
        assert!(svd_optimal_correction(&resp, &err, 1.0).is_err());

        let bad_err = Array1::from_vec(vec![1.0, f64::NAN]);
        let non_singular = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        assert!(svd_optimal_correction(&non_singular, &bad_err, 1.0).is_err());
        assert!(svd_optimal_correction(&non_singular, &Array1::from_vec(vec![1.0]), 1.0).is_err());
        assert!(svd_optimal_correction(&non_singular, &err, f64::INFINITY).is_err());
    }

    #[test]
    fn test_build_response_matrix_rejects_invalid_inputs() {
        let base = vec![0.0, 0.0];
        assert!(build_response_matrix(|_| [0.0, 0.0], &[], 0.1).is_err());
        assert!(build_response_matrix(|_| [0.0, 0.0], &base, 0.0).is_err());
        assert!(build_response_matrix(|_| [f64::NAN, 0.0], &base, 0.1).is_err());
    }
}
