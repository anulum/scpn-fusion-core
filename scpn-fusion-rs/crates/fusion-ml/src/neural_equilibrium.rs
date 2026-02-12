// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Neural Equilibrium
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Neural equilibrium accelerator: PCA + MLP surrogate.
//!
//! Port of `neural_equilibrium.py`.
//! PCA compresses 2D flux maps to low-rank coefficients;
//! MLP maps coil currents → PCA coefficients for ~1000× speedup.

use ndarray::{Array1, Array2, Axis};
use rand::Rng;

/// Default number of PCA components. Python: 15.
#[cfg(test)]
const N_COMPONENTS: usize = 15;

/// MLP architecture: input → 64 (tanh) → 32 (tanh) → n_components.
const HIDDEN1: usize = 64;
const HIDDEN2: usize = 32;

/// PCA model fitted via truncated SVD.
pub struct PCA {
    /// Number of retained components.
    pub n_components: usize,
    /// Mean of training data: (n_features,).
    pub mean: Array1<f64>,
    /// Principal components: (n_components, n_features).
    pub components: Array2<f64>,
    /// Explained variance per component.
    pub explained_variance: Array1<f64>,
}

impl PCA {
    /// Fit PCA from data matrix X: (n_samples, n_features).
    /// Uses thin SVD on centered data.
    pub fn fit(x: &Array2<f64>, n_components: usize) -> Self {
        let (n, _p) = x.dim();
        let k = n_components.min(n);

        // Center data
        let mean = x.mean_axis(Axis(0)).unwrap();
        let x_centered = x - &mean.broadcast(x.raw_dim()).unwrap();

        // Compute covariance via X^T X (p×p might be large; use X X^T if n < p)
        // For our use case, n_samples < n_features, so work in (n×n) space.
        let gram = x_centered.dot(&x_centered.t()); // (n, n)

        // Eigendecompose gram matrix using power iteration for top-k
        let (eigenvalues, eigenvectors) = symmetric_eigen_topk(&gram, k);

        // Recover principal components: V_j = X^T u_j / sqrt(λ_j)
        let mut components = Array2::zeros((k, x.ncols()));
        for (j, &ev) in eigenvalues.iter().enumerate().take(k) {
            let u_j = eigenvectors.column(j).to_owned();
            let v_j = x_centered.t().dot(&u_j);
            let norm = ev.sqrt().max(1e-15);
            components.row_mut(j).assign(&(&v_j / norm));
        }

        let explained_variance = Array1::from_vec(
            eigenvalues.iter().map(|&ev| ev / (n - 1) as f64).collect(),
        );

        PCA {
            n_components: k,
            mean,
            components,
            explained_variance,
        }
    }

    /// Transform data to PCA space: X → coefficients.
    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let centered = x - &self.mean.broadcast(x.raw_dim()).unwrap();
        centered.dot(&self.components.t())
    }

    /// Inverse transform: coefficients → reconstructed data.
    pub fn inverse_transform(&self, coeffs: &Array2<f64>) -> Array2<f64> {
        coeffs.dot(&self.components) + self.mean.broadcast((coeffs.nrows(), self.mean.len())).unwrap()
    }
}

/// Simple eigendecomposition of symmetric matrix, returning top-k eigenvalues/vectors.
/// Uses repeated power iteration with deflation.
fn symmetric_eigen_topk(a: &Array2<f64>, k: usize) -> (Vec<f64>, Array2<f64>) {
    let n = a.nrows();
    let k = k.min(n);
    let mut eigenvalues = Vec::with_capacity(k);
    let mut eigenvectors = Array2::zeros((n, k));
    let mut deflated = a.clone();

    let mut rng = rand::thread_rng();

    for j in 0..k {
        // Power iteration for largest eigenvalue of deflated matrix
        let mut v = Array1::from_shape_fn(n, |_| rng.gen::<f64>() - 0.5);
        let norm = v.dot(&v).sqrt();
        v /= norm;

        for _ in 0..200 {
            let av = deflated.dot(&v);
            let norm = av.dot(&av).sqrt();
            if norm < 1e-15 {
                break;
            }
            v = av / norm;
        }

        let eigenvalue = v.dot(&deflated.dot(&v));
        eigenvalues.push(eigenvalue.max(0.0));
        eigenvectors.column_mut(j).assign(&v);

        // Deflate: A ← A - λ v v^T
        let vvt = {
            let vc = v.clone().insert_axis(Axis(1)); // (n, 1)
            let vr = v.clone().insert_axis(Axis(0)); // (1, n)
            vc.dot(&vr)
        };
        deflated = &deflated - &(&vvt * eigenvalue);
    }

    (eigenvalues, eigenvectors)
}

/// Simple feedforward MLP: input → 64 (tanh) → 32 (tanh) → output.
pub struct EquilibriumMLP {
    pub w1: Array2<f64>,
    pub b1: Array1<f64>,
    pub w2: Array2<f64>,
    pub b2: Array1<f64>,
    pub w3: Array2<f64>,
    pub b3: Array1<f64>,
}

impl EquilibriumMLP {
    /// Create with random Xavier initialization.
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let s1 = (2.0 / (input_dim + HIDDEN1) as f64).sqrt();
        let s2 = (2.0 / (HIDDEN1 + HIDDEN2) as f64).sqrt();
        let s3 = (2.0 / (HIDDEN2 + output_dim) as f64).sqrt();

        EquilibriumMLP {
            w1: Array2::from_shape_fn((input_dim, HIDDEN1), |_| (rng.gen::<f64>() - 0.5) * 2.0 * s1),
            b1: Array1::zeros(HIDDEN1),
            w2: Array2::from_shape_fn((HIDDEN1, HIDDEN2), |_| (rng.gen::<f64>() - 0.5) * 2.0 * s2),
            b2: Array1::zeros(HIDDEN2),
            w3: Array2::from_shape_fn((HIDDEN2, output_dim), |_| (rng.gen::<f64>() - 0.5) * 2.0 * s3),
            b3: Array1::zeros(output_dim),
        }
    }

    /// Forward pass: returns output of shape (n_components,).
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let z1 = x.dot(&self.w1) + &self.b1;
        let a1 = z1.mapv(|v| v.tanh());
        let z2 = a1.dot(&self.w2) + &self.b2;
        let a2 = z2.mapv(|v| v.tanh());
        a2.dot(&self.w3) + &self.b3
    }

    /// Batch forward: (n_samples, input_dim) → (n_samples, output_dim).
    pub fn forward_batch(&self, x: &Array2<f64>) -> Array2<f64> {
        let z1 = x.dot(&self.w1) + &self.b1;
        let a1 = z1.mapv(|v| v.tanh());
        let z2 = a1.dot(&self.w2) + &self.b2;
        let a2 = z2.mapv(|v| v.tanh());
        a2.dot(&self.w3) + &self.b3
    }
}

/// Combined PCA + MLP accelerator.
pub struct NeuralEquilibrium {
    pub pca: PCA,
    pub mlp: EquilibriumMLP,
}

impl NeuralEquilibrium {
    /// Predict equilibrium from coil currents.
    pub fn predict(&self, currents: &Array1<f64>) -> Array1<f64> {
        let coeffs = self.mlp.forward(currents);
        let coeffs_2d = coeffs.insert_axis(Axis(0));
        let psi_flat = self.pca.inverse_transform(&coeffs_2d);
        psi_flat.row(0).to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pca_fit_components() {
        // 10 samples of 50 features
        let mut rng = rand::thread_rng();
        let x = Array2::from_shape_fn((10, 50), |_| rng.gen::<f64>());
        let pca = PCA::fit(&x, 5);
        assert_eq!(pca.n_components, 5);
        assert_eq!(pca.components.nrows(), 5);
        assert_eq!(pca.components.ncols(), 50);
    }

    #[test]
    fn test_pca_roundtrip() {
        // Low-rank data should reconstruct well
        let mut rng = rand::thread_rng();
        let n = 20;
        let p = 100;
        let k = 3;

        // Create rank-3 data
        let a = Array2::from_shape_fn((n, k), |_| rng.gen::<f64>());
        let b = Array2::from_shape_fn((k, p), |_| rng.gen::<f64>());
        let x = a.dot(&b);

        let pca = PCA::fit(&x, k);
        let coeffs = pca.transform(&x);
        let x_recon = pca.inverse_transform(&coeffs);

        let error: f64 = (&x - &x_recon).mapv(|v| v * v).sum();
        let energy: f64 = x.mapv(|v| v * v).sum();
        let rel_error = error / energy.max(1e-15);
        assert!(
            rel_error < 0.01,
            "PCA roundtrip error too high: {rel_error:.6}"
        );
    }

    #[test]
    fn test_mlp_output_shape() {
        let mlp = EquilibriumMLP::new(6, N_COMPONENTS);
        let x = Array1::from_elem(6, 1.0);
        let out = mlp.forward(&x);
        assert_eq!(out.len(), N_COMPONENTS);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_mlp_batch_forward() {
        let mlp = EquilibriumMLP::new(6, N_COMPONENTS);
        let x = Array2::from_elem((5, 6), 1.0);
        let out = mlp.forward_batch(&x);
        assert_eq!(out.dim(), (5, N_COMPONENTS));
    }

    #[test]
    fn test_neural_equilibrium_predict() {
        let mut rng = rand::thread_rng();
        let n = 20;
        let p = 100;
        let k = 5;
        let a = Array2::from_shape_fn((n, k), |_| rng.gen::<f64>());
        let b = Array2::from_shape_fn((k, p), |_| rng.gen::<f64>());
        let x = a.dot(&b);
        let pca = PCA::fit(&x, k);
        let mlp = EquilibriumMLP::new(6, k);
        let ne = NeuralEquilibrium { pca, mlp };
        let result = ne.predict(&Array1::from_elem(6, 0.5));
        assert_eq!(result.len(), p);
        assert!(result.iter().all(|v| v.is_finite()));
    }
}
