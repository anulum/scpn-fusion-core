//! Linear algebra utilities.
//!
//! SVD, 2x2 eigendecomposition, pseudoinverse with Tikhonov regularization.

use ndarray::{Array1, Array2};

/// 2x2 eigenvalue decomposition.
///
/// Used by stability_analyzer.py for the stiffness matrix.
/// Returns (eigenvalues, eigenvectors) sorted by ascending eigenvalue.
pub fn eig_2x2(a: &[[f64; 2]; 2]) -> ([f64; 2], [[f64; 2]; 2]) {
    let trace = a[0][0] + a[1][1];
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let disc = trace * trace - 4.0 * det;

    if disc < 0.0 {
        // Complex eigenvalues — return real parts
        let re = trace / 2.0;
        return ([re, re], [[1.0, 0.0], [0.0, 1.0]]);
    }

    let sqrt_disc = disc.sqrt();
    let l1 = (trace - sqrt_disc) / 2.0;
    let l2 = (trace + sqrt_disc) / 2.0;

    // Eigenvectors
    let v1 = if a[0][1].abs() > 1e-15 {
        let x = l1 - a[1][1];
        let y = a[1][0];
        let norm = (x * x + y * y).sqrt();
        [x / norm, y / norm]
    } else if a[1][0].abs() > 1e-15 {
        let x = a[0][1];
        let y = l1 - a[0][0];
        let norm = (x * x + y * y).sqrt();
        [x / norm, y / norm]
    } else {
        [1.0, 0.0]
    };

    let v2 = if a[0][1].abs() > 1e-15 {
        let x = l2 - a[1][1];
        let y = a[1][0];
        let norm = (x * x + y * y).sqrt();
        [x / norm, y / norm]
    } else if a[1][0].abs() > 1e-15 {
        let x = a[0][1];
        let y = l2 - a[0][0];
        let norm = (x * x + y * y).sqrt();
        [x / norm, y / norm]
    } else {
        [0.0, 1.0]
    };

    ([l1, l2], [v1, v2])
}

/// Simple SVD for small matrices using one-sided Jacobi rotations.
///
/// Returns (U, sigma, Vt) where A ≈ U * diag(sigma) * Vt.
/// For the small matrices in this project (2x7 to 20x20), this is sufficient.
///
/// Matches `numpy.linalg.svd(A, full_matrices=False)`.
pub fn svd_small(a: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let (m, n) = a.dim();
    let k = m.min(n);

    // Form A^T * A
    let mut ata = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for r in 0..m {
                sum += a[[r, i]] * a[[r, j]];
            }
            ata[[i, j]] = sum;
        }
    }

    // Jacobi eigenvalue iteration on A^T*A to get V and sigma^2
    let mut v = Array2::eye(n);
    let max_iter = 100;

    for _ in 0..max_iter {
        let mut off_diag = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off_diag += ata[[i, j]].abs();
            }
        }
        if off_diag < 1e-14 {
            break;
        }

        for i in 0..n {
            for j in (i + 1)..n {
                if ata[[i, j]].abs() < 1e-15 {
                    continue;
                }
                let tau = (ata[[j, j]] - ata[[i, i]]) / (2.0 * ata[[i, j]]);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let cos = 1.0 / (1.0 + t * t).sqrt();
                let sin = t * cos;

                // Apply Jacobi rotation to ATA
                let aii = ata[[i, i]];
                let ajj = ata[[j, j]];
                let aij = ata[[i, j]];
                ata[[i, i]] = cos * cos * aii - 2.0 * sin * cos * aij + sin * sin * ajj;
                ata[[j, j]] = sin * sin * aii + 2.0 * sin * cos * aij + cos * cos * ajj;
                ata[[i, j]] = 0.0;
                ata[[j, i]] = 0.0;

                for r in 0..n {
                    if r == i || r == j {
                        continue;
                    }
                    let ri = ata[[r, i]];
                    let rj = ata[[r, j]];
                    ata[[r, i]] = cos * ri - sin * rj;
                    ata[[i, r]] = ata[[r, i]];
                    ata[[r, j]] = sin * ri + cos * rj;
                    ata[[j, r]] = ata[[r, j]];
                }

                // Update V
                for r in 0..n {
                    let vi = v[[r, i]];
                    let vj = v[[r, j]];
                    v[[r, i]] = cos * vi - sin * vj;
                    v[[r, j]] = sin * vi + cos * vj;
                }
            }
        }
    }

    // Extract singular values (sqrt of eigenvalues of A^T*A)
    let mut sigma = Array1::zeros(k);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        ata[[j, j]]
            .partial_cmp(&ata[[i, i]])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (idx, &col) in order.iter().take(k).enumerate() {
        sigma[idx] = ata[[col, col]].max(0.0).sqrt();
    }

    // Reorder V columns
    let mut vt = Array2::zeros((k, n));
    for (idx, &col) in order.iter().take(k).enumerate() {
        for j in 0..n {
            vt[[idx, j]] = v[[j, col]];
        }
    }

    // Compute U = A * V * diag(1/sigma)
    let mut u = Array2::zeros((m, k));
    for idx in 0..k {
        if sigma[idx] > 1e-14 {
            let inv_s = 1.0 / sigma[idx];
            for i in 0..m {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += a[[i, j]] * vt[[idx, j]];
                }
                u[[i, idx]] = sum * inv_s;
            }
        }
    }

    (u, sigma, vt)
}

/// Pseudoinverse with SVD and singular value cutoff.
///
/// Used by fusion_optimal_control.py compute_optimal_correction().
pub fn pinv_svd(a: &Array2<f64>, sv_cutoff: f64) -> Array2<f64> {
    let (u, sigma, vt) = svd_small(a);
    let (m, n) = a.dim();
    let k = sigma.len();

    let mut result = Array2::zeros((n, m));

    for idx in 0..k {
        if sigma[idx] > sv_cutoff {
            let inv_s = 1.0 / sigma[idx];
            for i in 0..n {
                for j in 0..m {
                    result[[i, j]] += vt[[idx, i]] * inv_s * u[[j, idx]];
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eig_2x2_diagonal() {
        let a = [[3.0, 0.0], [0.0, 5.0]];
        let (vals, _vecs) = eig_2x2(&a);
        assert!((vals[0] - 3.0).abs() < 1e-10);
        assert!((vals[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_eig_2x2_symmetric() {
        let a = [[2.0, 1.0], [1.0, 2.0]];
        let (vals, _vecs) = eig_2x2(&a);
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_svd_identity() {
        let a = Array2::eye(3);
        let (u, sigma, vt) = svd_small(&a);
        for i in 0..3 {
            assert!((sigma[i] - 1.0).abs() < 1e-10, "sigma[{i}] = {}", sigma[i]);
        }
        // U * diag(sigma) * Vt should reconstruct A
        let mut reconstructed: Array2<f64> = Array2::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    reconstructed[[i, j]] += u[[i, k]] * sigma[k] * vt[[k, j]];
                }
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    {
                        let diff: f64 = reconstructed[[i, j]] - a[[i, j]];
                        diff.abs() < 1e-10
                    },
                    "Reconstruction failed at ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn test_pinv_svd_identity() {
        let a = Array2::eye(3);
        let pinv = pinv_svd(&a, 1e-10);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (pinv[[i, j]] - expected).abs() < 1e-10,
                    "pinv identity failed at ({i}, {j})"
                );
            }
        }
    }
}
