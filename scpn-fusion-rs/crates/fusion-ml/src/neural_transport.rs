// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Neural Transport Surrogate
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Neural transport surrogate (10 -> 64 -> 32 -> 3) with analytic fallback.

use fusion_types::error::{FusionError, FusionResult};
use ndarray::{Array1, Array2};
use ndarray_npy::NpzReader;
use std::fs::File;

const INPUT_DIM: usize = 10;
const HIDDEN1: usize = 64;
const HIDDEN2: usize = 32;
const OUTPUT_DIM: usize = 3;
const EPS_STD: f64 = 1e-8;

#[derive(Debug, Clone)]
pub struct NeuralTransportWeights {
    pub w1: Array2<f64>,           // (10, 64)
    pub b1: Array1<f64>,           // (64,)
    pub w2: Array2<f64>,           // (64, 32)
    pub b2: Array1<f64>,           // (32,)
    pub w3: Array2<f64>,           // (32, 3)
    pub b3: Array1<f64>,           // (3,)
    pub input_mean: Array1<f64>,   // (10,)
    pub input_std: Array1<f64>,    // (10,)
    pub output_scale: Array1<f64>, // (3,)
}

#[derive(Debug, Clone)]
pub struct NeuralTransportModel {
    weights: Option<NeuralTransportWeights>,
}

impl NeuralTransportModel {
    /// Construct an analytic fallback model (no neural weights loaded).
    pub fn new() -> Self {
        Self { weights: None }
    }

    /// Load neural transport weights from a NumPy `.npz` archive.
    pub fn from_npz(path: &str) -> FusionResult<Self> {
        let file = File::open(path)?;
        let mut npz = NpzReader::new(file)
            .map_err(|e| FusionError::ConfigError(format!("Failed to open npz '{path}': {e}")))?;

        let weights = NeuralTransportWeights {
            w1: read_array2(&mut npz, "w1")?,
            b1: read_array1(&mut npz, "b1")?,
            w2: read_array2(&mut npz, "w2")?,
            b2: read_array1(&mut npz, "b2")?,
            w3: read_array2(&mut npz, "w3")?,
            b3: read_array1(&mut npz, "b3")?,
            input_mean: read_array1(&mut npz, "input_mean")?,
            input_std: read_array1(&mut npz, "input_std")?,
            output_scale: read_array1(&mut npz, "output_scale")?,
        };

        validate_shapes(&weights)?;

        Ok(Self {
            weights: Some(weights),
        })
    }

    /// Predict transport coefficients [chi_i, chi_e, d_eff].
    pub fn predict(&self, input: &Array1<f64>) -> Array1<f64> {
        if let Some(weights) = &self.weights {
            if input.len() == INPUT_DIM {
                return neural_forward(input, weights);
            }
        }
        critical_gradient_model(input)
    }

    /// Batch prediction over radial points.
    pub fn predict_profile(&self, inputs: &Array2<f64>) -> Array2<f64> {
        let mut outputs = Array2::zeros((inputs.nrows(), OUTPUT_DIM));
        for (i, row) in inputs.outer_iter().enumerate() {
            let pred = self.predict(&row.to_owned());
            outputs.row_mut(i).assign(&pred);
        }
        outputs
    }

    /// True when a neural model is loaded; false when running analytic fallback.
    pub fn is_neural(&self) -> bool {
        self.weights.is_some()
    }
}

fn validate_shapes(weights: &NeuralTransportWeights) -> FusionResult<()> {
    if weights.w1.dim() != (INPUT_DIM, HIDDEN1) {
        return Err(FusionError::ConfigError(format!(
            "Invalid w1 shape {:?}, expected ({INPUT_DIM}, {HIDDEN1})",
            weights.w1.dim()
        )));
    }
    if weights.b1.len() != HIDDEN1 {
        return Err(FusionError::ConfigError(format!(
            "Invalid b1 length {}, expected {HIDDEN1}",
            weights.b1.len()
        )));
    }
    if weights.w2.dim() != (HIDDEN1, HIDDEN2) {
        return Err(FusionError::ConfigError(format!(
            "Invalid w2 shape {:?}, expected ({HIDDEN1}, {HIDDEN2})",
            weights.w2.dim()
        )));
    }
    if weights.b2.len() != HIDDEN2 {
        return Err(FusionError::ConfigError(format!(
            "Invalid b2 length {}, expected {HIDDEN2}",
            weights.b2.len()
        )));
    }
    if weights.w3.dim() != (HIDDEN2, OUTPUT_DIM) {
        return Err(FusionError::ConfigError(format!(
            "Invalid w3 shape {:?}, expected ({HIDDEN2}, {OUTPUT_DIM})",
            weights.w3.dim()
        )));
    }
    if weights.b3.len() != OUTPUT_DIM {
        return Err(FusionError::ConfigError(format!(
            "Invalid b3 length {}, expected {OUTPUT_DIM}",
            weights.b3.len()
        )));
    }
    if weights.input_mean.len() != INPUT_DIM || weights.input_std.len() != INPUT_DIM {
        return Err(FusionError::ConfigError(format!(
            "Invalid input normalization lengths mean={}, std={}, expected {INPUT_DIM}",
            weights.input_mean.len(),
            weights.input_std.len()
        )));
    }
    if weights.output_scale.len() != OUTPUT_DIM {
        return Err(FusionError::ConfigError(format!(
            "Invalid output_scale length {}, expected {OUTPUT_DIM}",
            weights.output_scale.len()
        )));
    }
    Ok(())
}

fn read_array1(npz: &mut NpzReader<File>, key: &str) -> FusionResult<Array1<f64>> {
    npz.by_name::<ndarray::OwnedRepr<f64>, ndarray::Ix1>(&format!("{key}.npy"))
        .or_else(|_| npz.by_name::<ndarray::OwnedRepr<f64>, ndarray::Ix1>(key))
        .map_err(|e| FusionError::ConfigError(format!("Failed to read {key} from npz: {e}")))
}

fn read_array2(npz: &mut NpzReader<File>, key: &str) -> FusionResult<Array2<f64>> {
    npz.by_name::<ndarray::OwnedRepr<f64>, ndarray::Ix2>(&format!("{key}.npy"))
        .or_else(|_| npz.by_name::<ndarray::OwnedRepr<f64>, ndarray::Ix2>(key))
        .map_err(|e| FusionError::ConfigError(format!("Failed to read {key} from npz: {e}")))
}

fn neural_forward(input: &Array1<f64>, w: &NeuralTransportWeights) -> Array1<f64> {
    let std_safe = w.input_std.mapv(|v| v.abs().max(EPS_STD));
    let x_norm = (input - &w.input_mean) / &std_safe;

    let h1 = (x_norm.dot(&w.w1) + &w.b1).mapv(relu);
    let h2 = (h1.dot(&w.w2) + &w.b2).mapv(relu);
    (h2.dot(&w.w3) + &w.b3).mapv(softplus) * &w.output_scale
}

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn feature(input: &Array1<f64>, idx: usize, default: f64) -> f64 {
    input.get(idx).copied().unwrap_or(default)
}

/// Analytic fallback transport model with critical-gradient triggering.
fn critical_gradient_model(input: &Array1<f64>) -> Array1<f64> {
    // Input convention for fallback:
    // [grad_ti, grad_te, grad_ne, shear, collisionality, zeff, q95, beta_n, rho, aspect]
    let grad_ti = feature(input, 0, 4.0);
    let grad_te = feature(input, 1, 3.5);
    let grad_ne = feature(input, 2, 2.5);
    let shear = feature(input, 3, 0.0).max(0.0);
    let collisionality = feature(input, 4, 0.0).max(0.0);
    let zeff = feature(input, 5, 1.0).max(1.0);
    let q95 = feature(input, 6, 3.0).max(0.0);
    let beta_n = feature(input, 7, 0.0).max(0.0);
    let rho = feature(input, 8, 0.5).clamp(0.0, 1.0);
    let aspect = feature(input, 9, 3.0).max(0.1);

    let mut chi_i = (grad_ti - 4.0).max(0.0).powi(2);
    let mut chi_e = 0.6 * (grad_te - 3.5).max(0.0).powi(2);
    let mut d_eff = 0.1 + 0.5 * (grad_ne - 2.5).max(0.0).powi(2) + 0.05 * collisionality;

    let stabilization = 1.0 / (1.0 + 0.08 * shear);
    let beta_suppression = 1.0 / (1.0 + 0.2 * beta_n);
    let geometry_factor = (1.0 + 0.02 * (q95 - 3.0).abs() + 0.01 * (aspect - 3.0).abs()).min(2.0);
    let impurity_factor = 1.0 + 0.03 * (zeff - 1.0);
    let edge_boost = 1.0 + 0.1 * rho;

    chi_i *= stabilization * beta_suppression * geometry_factor;
    chi_e *= stabilization * beta_suppression * impurity_factor;
    d_eff *= edge_boost;

    Array1::from_vec(vec![chi_i.max(0.0), chi_e.max(0.0), d_eff.max(0.0)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};
    use ndarray_npy::NpzWriter;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn synthetic_weights() -> NeuralTransportWeights {
        let mut w1 = Array2::zeros((INPUT_DIM, HIDDEN1));
        let b1 = Array1::zeros(HIDDEN1);
        let mut w2 = Array2::zeros((HIDDEN1, HIDDEN2));
        let b2 = Array1::zeros(HIDDEN2);
        let mut w3 = Array2::zeros((HIDDEN2, OUTPUT_DIM));
        let b3 = Array1::zeros(OUTPUT_DIM);

        // Sparse deterministic pathway.
        w1[[0, 0]] = 1.0;
        w1[[1, 1]] = 1.0;
        w2[[0, 0]] = 1.0;
        w2[[1, 1]] = 1.0;
        w3[[0, 0]] = 1.0;
        w3[[1, 1]] = 1.0;
        w3[[0, 2]] = -1.0;

        NeuralTransportWeights {
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            input_mean: Array1::zeros(INPUT_DIM),
            input_std: Array1::ones(INPUT_DIM),
            output_scale: array![1.0, 2.0, 1.0],
        }
    }

    #[test]
    fn test_neural_transport_forward_pass() {
        let weights = synthetic_weights();
        let model = NeuralTransportModel {
            weights: Some(weights),
        };

        let input = array![2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let out = model.predict(&input);

        assert!((out[0] - 2.126_928_011).abs() < 1e-9);
        assert!((out[1] - 6.097_174_703).abs() < 1e-9);
        assert!((out[2] - 0.126_928_011).abs() < 1e-9);
    }

    #[test]
    fn test_critical_gradient_fallback() {
        let model = NeuralTransportModel::new();
        let input = array![5.0, 4.0, 3.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.5, 3.0];
        let out = model.predict(&input);

        assert!((out[0] - 1.0).abs() < 1e-12);
        assert!((out[1] - 0.15).abs() < 1e-12);
        assert!((out[2] - 0.23625).abs() < 1e-12);
        assert!(!model.is_neural());
    }

    #[test]
    fn test_batch_profile_prediction() {
        let model = NeuralTransportModel::new();
        let inputs = Array2::from_shape_fn((50, INPUT_DIM), |(i, j)| {
            (i as f64) * 0.02 + (j as f64) * 0.1
        });
        let out = model.predict_profile(&inputs);

        assert_eq!(out.dim(), (50, OUTPUT_DIM));
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_weight_loading_roundtrip() {
        let weights = synthetic_weights();

        let epoch_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "fusion_neural_transport_{}_{}.npz",
            std::process::id(),
            epoch_ns
        ));

        let file = File::create(&path).unwrap();
        let mut writer = NpzWriter::new(file);
        writer.add_array("w1", &weights.w1).unwrap();
        writer.add_array("b1", &weights.b1).unwrap();
        writer.add_array("w2", &weights.w2).unwrap();
        writer.add_array("b2", &weights.b2).unwrap();
        writer.add_array("w3", &weights.w3).unwrap();
        writer.add_array("b3", &weights.b3).unwrap();
        writer.add_array("input_mean", &weights.input_mean).unwrap();
        writer.add_array("input_std", &weights.input_std).unwrap();
        writer
            .add_array("output_scale", &weights.output_scale)
            .unwrap();
        writer.finish().unwrap();

        let loaded = NeuralTransportModel::from_npz(path.to_str().unwrap()).unwrap();
        assert!(loaded.is_neural());

        let input = array![2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let out_loaded = loaded.predict(&input);
        let out_direct = NeuralTransportModel {
            weights: Some(weights),
        }
        .predict(&input);

        for i in 0..OUTPUT_DIM {
            assert!((out_loaded[i] - out_direct[i]).abs() < 1e-12);
        }

        std::fs::remove_file(path).ok();
    }
}
