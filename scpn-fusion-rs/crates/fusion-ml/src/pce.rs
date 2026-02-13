// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Polynomial Chaos Expansion
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Polynomial Chaos Expansion (PCE) for uncertainty quantification.
//!
//! Implements multivariate total-order Hermite chaos with least-squares fitting.

use fusion_math::linalg::pinv_svd;
use fusion_types::error::{FusionError, FusionResult};
use ndarray::{Array1, Array2, ArrayView1};
use rand::seq::SliceRandom;
use rand::Rng;

/// Polynomial chaos model with total-order multi-index basis.
#[derive(Debug, Clone)]
pub struct PCEModel {
    /// Basis coefficients with shape (n_terms, n_outputs).
    pub coefficients: Array2<f64>,
    /// Total-order multi-index for each basis term.
    pub multi_index: Vec<Vec<usize>>,
}

impl PCEModel {
    /// Checked fit API returning structured errors on invalid data.
    pub fn try_fit(
        samples: &Array2<f64>,
        outputs: &Array2<f64>,
        order: usize,
    ) -> FusionResult<Self> {
        if samples.nrows() != outputs.nrows() {
            return Err(FusionError::ConfigError(
                "PCE fit requires matching sample and output rows".to_string(),
            ));
        }
        if samples.nrows() == 0 {
            return Err(FusionError::ConfigError(
                "PCE fit requires at least one sample".to_string(),
            ));
        }
        if samples.ncols() == 0 {
            return Err(FusionError::ConfigError(
                "PCE fit requires at least one dimension".to_string(),
            ));
        }
        if outputs.ncols() == 0 {
            return Err(FusionError::ConfigError(
                "PCE fit requires at least one output column".to_string(),
            ));
        }
        if !samples.iter().all(|v| v.is_finite()) || !outputs.iter().all(|v| v.is_finite()) {
            return Err(FusionError::ConfigError(
                "PCE fit received non-finite sample or output values".to_string(),
            ));
        }

        let multi_index = total_order_multi_index(samples.ncols(), order);
        let design = design_matrix(samples, &multi_index);
        let pinv = pinv_svd(&design, 1e-10);
        let coefficients = pinv.dot(outputs);
        if !coefficients.iter().all(|v| v.is_finite()) {
            return Err(FusionError::LinAlg(
                "PCE coefficients contain non-finite values".to_string(),
            ));
        }

        Ok(Self {
            coefficients,
            multi_index,
        })
    }

    /// Fit a PCE model with multivariate Hermite basis up to total `order`.
    ///
    /// `samples`: (n_samples, n_dims), `outputs`: (n_samples, n_outputs)
    pub fn fit(samples: &Array2<f64>, outputs: &Array2<f64>, order: usize) -> Self {
        Self::try_fit(samples, outputs, order).expect("PCE fit failed")
    }

    /// Checked prediction API returning error on shape/data mismatch.
    pub fn try_predict(&self, x: &Array1<f64>) -> FusionResult<Array1<f64>> {
        let Some(first_term) = self.multi_index.first() else {
            return Err(FusionError::ConfigError(
                "PCE model has an empty basis".to_string(),
            ));
        };
        if x.len() != first_term.len() {
            return Err(FusionError::ConfigError(format!(
                "PCE input dimension mismatch: expected {}, got {}",
                first_term.len(),
                x.len()
            )));
        }
        if !x.iter().all(|v| v.is_finite()) {
            return Err(FusionError::ConfigError(
                "PCE prediction received non-finite input values".to_string(),
            ));
        }

        let basis = basis_row(x.view(), &self.multi_index);
        Ok(basis.dot(&self.coefficients))
    }

    /// Predict model output at a single input point.
    pub fn predict(&self, x: &Array1<f64>) -> Array1<f64> {
        self.try_predict(x).expect("PCE prediction failed")
    }

    /// First-order Sobol sensitivity indices estimated from PCE coefficients.
    pub fn sobol_indices(&self) -> Vec<f64> {
        let Some(first_term) = self.multi_index.first() else {
            return Vec::new();
        };
        let n_dims = first_term.len();
        if n_dims == 0 {
            return Vec::new();
        }

        let mut first_order_var = vec![0.0; n_dims];
        let mut total_var = 0.0;

        // Skip constant term (index 0)
        for (term_idx, alpha) in self.multi_index.iter().enumerate().skip(1) {
            let norm = hermite_multi_norm(alpha);
            let coeff_energy = self
                .coefficients
                .row(term_idx)
                .iter()
                .map(|c| c * c)
                .sum::<f64>()
                * norm;
            total_var += coeff_energy;

            let mut nonzero_count = 0usize;
            let mut only_dim = 0usize;
            for (d, &exp) in alpha.iter().enumerate() {
                if exp > 0 {
                    nonzero_count += 1;
                    only_dim = d;
                }
            }
            if nonzero_count == 1 {
                first_order_var[only_dim] += coeff_energy;
            }
        }

        if total_var <= 1e-16 {
            return vec![0.0; n_dims];
        }

        first_order_var
            .into_iter()
            .map(|v| (v / total_var).clamp(0.0, 1.0))
            .collect()
    }
}

/// Latin Hypercube Sampling in [0, 1] for each dimension.
pub fn latin_hypercube(n_samples: usize, n_dims: usize) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    latin_hypercube_with_rng(n_samples, n_dims, &mut rng)
}

fn latin_hypercube_with_rng<R: Rng + ?Sized>(
    n_samples: usize,
    n_dims: usize,
    rng: &mut R,
) -> Array2<f64> {
    assert!(n_samples > 0, "n_samples must be > 0");
    assert!(n_dims > 0, "n_dims must be > 0");

    let mut samples = Array2::zeros((n_samples, n_dims));
    for dim in 0..n_dims {
        let mut values: Vec<f64> = (0..n_samples)
            .map(|i| (i as f64 + rng.gen::<f64>()) / n_samples as f64)
            .collect();
        values.shuffle(rng);
        for i in 0..n_samples {
            samples[[i, dim]] = values[i];
        }
    }
    samples
}

fn total_order_multi_index(n_dims: usize, order: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut current = vec![0usize; n_dims];
    for total in 0..=order {
        enumerate_multi_index(0, total, &mut current, &mut out);
    }
    out
}

fn enumerate_multi_index(
    dim: usize,
    remaining: usize,
    current: &mut [usize],
    out: &mut Vec<Vec<usize>>,
) {
    if dim + 1 == current.len() {
        current[dim] = remaining;
        out.push(current.to_vec());
        return;
    }
    for v in 0..=remaining {
        current[dim] = v;
        enumerate_multi_index(dim + 1, remaining - v, current, out);
    }
}

fn design_matrix(samples: &Array2<f64>, multi_index: &[Vec<usize>]) -> Array2<f64> {
    let mut a = Array2::zeros((samples.nrows(), multi_index.len()));
    for i in 0..samples.nrows() {
        let row = basis_row(samples.row(i), multi_index);
        for j in 0..multi_index.len() {
            a[[i, j]] = row[j];
        }
    }
    a
}

fn basis_row(x: ArrayView1<'_, f64>, multi_index: &[Vec<usize>]) -> Array1<f64> {
    let mut row = Array1::zeros(multi_index.len());
    for (term_idx, alpha) in multi_index.iter().enumerate() {
        let mut val = 1.0;
        for d in 0..alpha.len() {
            val *= hermite_prob(alpha[d], x[d]);
        }
        row[term_idx] = val;
    }
    row
}

fn hermite_prob(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }

    let mut hm2 = 1.0;
    let mut hm1 = x;
    for k in 2..=n {
        let h = x * hm1 - (k as f64 - 1.0) * hm2;
        hm2 = hm1;
        hm1 = h;
    }
    hm1
}

fn factorial(n: usize) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    (2..=n).fold(1.0, |acc, v| acc * v as f64)
}

fn hermite_multi_norm(alpha: &[usize]) -> f64 {
    alpha.iter().fold(1.0, |acc, &a| acc * factorial(a))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn to_centered_domain(u: &Array2<f64>) -> Array2<f64> {
        u.mapv(|v| 2.0 * v - 1.0)
    }

    fn ipb98_like(x: ArrayView1<'_, f64>) -> f64 {
        // Synthetic quadratic surrogate for confinement-like scaling trends.
        let linear = 0.22 * x[0] - 0.15 * x[1] + 0.11 * x[2] + 0.08 * x[3] - 0.07 * x[4]
            + 0.05 * x[5]
            + 0.04 * x[6]
            - 0.03 * x[7]
            + 0.02 * x[8];
        let quadratic = 0.09 * (x[0] * x[0] - 1.0) + 0.05 * (x[2] * x[2] - 1.0)
            - 0.04 * (x[5] * x[5] - 1.0)
            + 0.06 * x[0] * x[1]
            - 0.03 * x[3] * x[4]
            + 0.02 * x[7] * x[8];
        1.0 + linear + quadratic
    }

    fn mean_std(values: &[f64]) -> (f64, f64) {
        let n = values.len().max(1) as f64;
        let mean = values.iter().sum::<f64>() / n;
        let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
        (mean, var.sqrt())
    }

    #[test]
    fn test_multi_index_count_order2_dim9() {
        let terms = total_order_multi_index(9, 2);
        assert_eq!(terms.len(), 55, "Expected 55 terms for n=9, p=2");
    }

    #[test]
    fn test_latin_hypercube_bounds_and_marginals() {
        let mut rng = StdRng::seed_from_u64(7);
        let lhs = latin_hypercube_with_rng(256, 4, &mut rng);
        assert_eq!(lhs.dim(), (256, 4));
        assert!(lhs.iter().all(|v| *v >= 0.0 && *v <= 1.0));

        for dim in 0..4 {
            let mean = lhs.column(dim).iter().copied().sum::<f64>() / 256.0;
            assert!(
                (mean - 0.5).abs() < 0.03,
                "LHS marginal mean drifted on dim {dim}: {mean}"
            );
        }
    }

    #[test]
    fn test_pce_matches_mc_on_ipb98() {
        let n_dims = 9;
        let order = 2;

        let mut rng_train = StdRng::seed_from_u64(42);
        let x_train = to_centered_domain(&latin_hypercube_with_rng(110, n_dims, &mut rng_train));
        let y_train =
            Array2::from_shape_fn((x_train.nrows(), 1), |(i, _)| ipb98_like(x_train.row(i)));

        let model = PCEModel::fit(&x_train, &y_train, order);
        assert_eq!(model.multi_index.len(), 55);

        let mut rng_mc = StdRng::seed_from_u64(2026);
        let x_mc = to_centered_domain(&latin_hypercube_with_rng(100_000, n_dims, &mut rng_mc));

        let mut truth = Vec::with_capacity(x_mc.nrows());
        let mut pce = Vec::with_capacity(x_mc.nrows());
        for i in 0..x_mc.nrows() {
            let x = x_mc.row(i);
            truth.push(ipb98_like(x));
            pce.push(model.predict(&x.to_owned())[0]);
        }

        let (mean_truth, std_truth) = mean_std(&truth);
        let (mean_pce, std_pce) = mean_std(&pce);

        let mean_rel = (mean_pce - mean_truth).abs() / mean_truth.abs().max(1e-12);
        let std_rel = (std_pce - std_truth).abs() / std_truth.abs().max(1e-12);
        assert!(mean_rel < 0.05, "Mean mismatch too large: {mean_rel:.4}");
        assert!(std_rel < 0.05, "Std mismatch too large: {std_rel:.4}");
    }

    #[test]
    fn test_sobol_indices_are_bounded() {
        let mut rng = StdRng::seed_from_u64(11);
        let x = to_centered_domain(&latin_hypercube_with_rng(150, 3, &mut rng));
        let y = Array2::from_shape_fn((150, 2), |(i, j)| {
            let row = x.row(i);
            if j == 0 {
                0.7 * row[0] + 0.2 * row[1] * row[1]
            } else {
                -0.5 * row[1] + 0.3 * row[2]
            }
        });

        let pce = PCEModel::fit(&x, &y, 2);
        let s = pce.sobol_indices();
        assert_eq!(s.len(), 3);
        assert!(s.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    #[test]
    fn test_try_fit_rejects_non_finite_inputs() {
        let mut x = Array2::zeros((4, 2));
        let y = Array2::ones((4, 1));
        x[[1, 1]] = f64::NAN;

        let err = PCEModel::try_fit(&x, &y, 2).unwrap_err();
        match err {
            FusionError::ConfigError(msg) => assert!(msg.contains("non-finite")),
            _ => panic!("Expected ConfigError for non-finite PCE input"),
        }
    }

    #[test]
    fn test_try_predict_rejects_dimension_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 0.2, -0.1, -0.3, 0.4]).unwrap();
        let y = Array2::from_shape_vec((3, 1), vec![1.0, 1.2, 0.8]).unwrap();
        let model = PCEModel::fit(&x, &y, 2);
        let bad = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        let err = model.try_predict(&bad).unwrap_err();
        match err {
            FusionError::ConfigError(msg) => assert!(msg.contains("dimension mismatch")),
            _ => panic!("Expected ConfigError for PCE dimension mismatch"),
        }
    }
}
