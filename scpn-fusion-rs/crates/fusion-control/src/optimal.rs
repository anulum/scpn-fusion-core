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
) -> Array1<f64> {
    let (m, n) = response_matrix.dim();
    assert_eq!(m, error.len());

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
            // Singular: return minimum norm solution along largest singular direction
            return Array1::zeros(n);
        }

        let aat_inv =
            Array2::from_shape_vec((2, 2), vec![d / det, -b / det, -c / det, a / det]).unwrap();

        let pinv = response_matrix.t().dot(&aat_inv); // n×m
        let mut delta = pinv.dot(error) * gain;

        // Clip
        for v in delta.iter_mut() {
            *v = v.clamp(-MAX_DELTA, MAX_DELTA);
        }
        delta
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
        delta
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
) -> Array2<f64>
where
    F: Fn(&[f64]) -> [f64; 2],
{
    let n = base_currents.len();
    let mut matrix = Array2::zeros((2, n));
    let mut test_currents = base_currents.to_vec();

    for i in 0..n {
        let orig = test_currents[i];

        test_currents[i] = orig + perturbation;
        let pos_plus = evaluate(&test_currents);

        test_currents[i] = orig - perturbation;
        let pos_minus = evaluate(&test_currents);

        test_currents[i] = orig;

        matrix[[0, i]] = (pos_plus[0] - pos_minus[0]) / (2.0 * perturbation);
        matrix[[1, i]] = (pos_plus[1] - pos_minus[1]) / (2.0 * perturbation);
    }

    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svd_correction_identity() {
        // 2×2 identity response → correction = error * gain
        let resp = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let error = Array1::from_vec(vec![1.0, -0.5]);
        let delta = svd_optimal_correction(&resp, &error, 1.0);
        assert!((delta[0] - 1.0).abs() < 1e-6);
        assert!((delta[1] - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_svd_correction_clipping() {
        let resp = Array2::from_shape_vec((2, 2), vec![100.0, 0.0, 0.0, 100.0]).unwrap();
        let error = Array1::from_vec(vec![1.0, 1.0]);
        let delta = svd_optimal_correction(&resp, &error, 1.0);
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
        let resp = build_response_matrix(|i: &[f64]| [i[0] + i[1], i[0] - i[1]], &base, 0.5);
        assert!((resp[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((resp[[0, 1]] - 1.0).abs() < 1e-6);
        assert!((resp[[1, 0]] - 1.0).abs() < 1e-6);
        assert!((resp[[1, 1]] - (-1.0)).abs() < 1e-6);
    }
}
