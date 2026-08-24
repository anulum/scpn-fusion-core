// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Native DeepONet Equilibrium Runtime
//! Native branch-trunk inference for manifest-validated equilibrium artifacts.
//!
//! The operator follows Lu et al. (2021), DOI
//! `10.1038/s42256-021-00302-5`: a control-vector branch and coordinate trunk
//! form a scaled inner product. Python owns NPZ authentication and passes the
//! validated numerical contract through PyO3; this module owns only native
//! inference and shape/finiteness invariants.

use fusion_types::error::{FusionError, FusionResult};
use ndarray::{Array1, Array2, Axis};

/// One fully connected layer stored in input-by-output orientation.
#[derive(Clone, Debug)]
pub struct DenseLayer {
    /// Weight matrix with shape `(input_width, output_width)`.
    pub weights: Array2<f64>,
    /// Output bias with `output_width` elements.
    pub bias: Array1<f64>,
}

/// Complete construction contract for one fixed-grid DeepONet runtime.
#[derive(Clone, Debug)]
pub struct DeepOnetEquilibriumConfig {
    /// Dense layers of the causal-control branch network.
    pub branch: Vec<DenseLayer>,
    /// Dense layers of the spatial-coordinate trunk network.
    pub trunk: Vec<DenseLayer>,
    /// Training-only mean for each branch input.
    pub input_mean: Array1<f64>,
    /// Positive training-only scale for each branch input.
    pub input_std: Array1<f64>,
    /// Fixed `(R, Z)` coordinate rows in metres.
    pub coordinates_rz_m: Array2<f64>,
    /// Training-only coordinate mean in metres.
    pub coordinate_mean: Array1<f64>,
    /// Positive training-only coordinate scale in metres.
    pub coordinate_std: Array1<f64>,
    /// Training-only spatial field mean in Wb/rad.
    pub field_mean: Array1<f64>,
    /// Positive global field scale in Wb/rad.
    pub field_scale: f64,
    /// Shared branch/trunk latent width.
    pub basis_width: usize,
    /// Fixed output grid as `(n_z, n_r)`.
    pub grid_shape: (usize, usize),
}

/// Manifest-bound native DeepONet equilibrium inference state.
#[derive(Clone, Debug)]
pub struct DeepOnetEquilibrium {
    branch: Vec<DenseLayer>,
    trunk_basis: Array2<f64>,
    input_mean: Array1<f64>,
    input_std: Array1<f64>,
    field_mean: Array1<f64>,
    field_scale: f64,
    basis_width: usize,
    grid_shape: (usize, usize),
}

impl DeepOnetEquilibrium {
    /// Validate and construct one fixed-grid branch-trunk runtime.
    ///
    /// Coordinates are in metres and the field mean/scale are in Wb/rad.
    /// The constructor precomputes the trunk basis because the manifest-bound
    /// coordinate grid cannot change between predictions.
    ///
    /// # Errors
    ///
    /// Returns a configuration error for empty, non-finite, dimensionally
    /// inconsistent, or non-positive-scale inputs.
    pub fn new(config: DeepOnetEquilibriumConfig) -> FusionResult<Self> {
        let DeepOnetEquilibriumConfig {
            branch,
            trunk,
            input_mean,
            input_std,
            coordinates_rz_m,
            coordinate_mean,
            coordinate_std,
            field_mean,
            field_scale,
            basis_width,
            grid_shape,
        } = config;
        validate_layers("branch", &branch)?;
        validate_layers("trunk", &trunk)?;
        if input_mean.is_empty() || input_mean.len() != input_std.len() {
            return config_error("DeepONet input normalisation dimensions are inconsistent");
        }
        if coordinates_rz_m.ncols() != 2 || coordinate_mean.len() != 2 || coordinate_std.len() != 2
        {
            return config_error("DeepONet coordinate contract must contain R and Z");
        }
        let grid_points = grid_shape
            .0
            .checked_mul(grid_shape.1)
            .ok_or_else(|| FusionError::ConfigError("DeepONet grid size overflow".to_string()))?;
        if grid_points == 0
            || coordinates_rz_m.nrows() != grid_points
            || field_mean.len() != grid_points
        {
            return config_error("DeepONet grid and field-mean dimensions are inconsistent");
        }
        if branch[0].weights.nrows() != input_mean.len()
            || trunk[0].weights.nrows() != 2
            || branch.last().map(|layer| layer.weights.ncols()) != Some(basis_width)
            || trunk.last().map(|layer| layer.weights.ncols()) != Some(basis_width)
            || basis_width == 0
        {
            return config_error("DeepONet branch, trunk, and basis dimensions disagree");
        }
        if !all_finite_1d(&input_mean)
            || !all_finite_1d(&input_std)
            || !all_finite_2d(&coordinates_rz_m)
            || !all_finite_1d(&coordinate_mean)
            || !all_finite_1d(&coordinate_std)
            || !all_finite_1d(&field_mean)
            || !field_scale.is_finite()
        {
            return config_error("DeepONet normalisation state contains non-finite values");
        }
        if input_std.iter().any(|value| *value <= 0.0)
            || coordinate_std.iter().any(|value| *value <= 0.0)
            || field_scale <= 0.0
        {
            return config_error("DeepONet normalisation scales must be positive");
        }

        let mut normalised_coordinates = coordinates_rz_m;
        for mut row in normalised_coordinates.rows_mut() {
            for column in 0..2 {
                row[column] = (row[column] - coordinate_mean[column]) / coordinate_std[column];
            }
        }
        let trunk_basis = forward(&trunk, &normalised_coordinates);
        if !all_finite_2d(&trunk_basis) {
            return config_error("DeepONet trunk produced non-finite basis values");
        }
        Ok(Self {
            branch,
            trunk_basis,
            input_mean,
            input_std,
            field_mean,
            field_scale,
            basis_width,
            grid_shape,
        })
    }

    /// Predict flattened poloidal-flux fields for a feature-row batch.
    ///
    /// Inputs use the artifact-declared feature order and units. The returned
    /// matrix has shape `(batch, n_z * n_r)` and values in Wb/rad.
    ///
    /// # Errors
    ///
    /// Returns a configuration error for a wrong feature width, non-finite
    /// controls, or non-finite native output.
    pub fn predict_batch(&self, features: &Array2<f64>) -> FusionResult<Array2<f64>> {
        if features.ncols() != self.input_mean.len() {
            return config_error(&format!(
                "DeepONet feature width {} does not match {}",
                features.ncols(),
                self.input_mean.len()
            ));
        }
        if !all_finite_2d(features) {
            return config_error("DeepONet controls must be finite");
        }
        let mut normalised = features.to_owned();
        for mut row in normalised.rows_mut() {
            for column in 0..self.input_mean.len() {
                row[column] = (row[column] - self.input_mean[column]) / self.input_std[column];
            }
        }
        let branch = forward(&self.branch, &normalised);
        let mut fields = branch.dot(&self.trunk_basis.t()) / (self.basis_width as f64).sqrt();
        for mut row in fields.rows_mut() {
            for column in 0..self.field_mean.len() {
                row[column] = self.field_mean[column] + self.field_scale * row[column];
            }
        }
        if !all_finite_2d(&fields) {
            return config_error("DeepONet native inference produced non-finite output");
        }
        Ok(fields)
    }

    /// Predict one flattened poloidal-flux field in Wb/rad.
    ///
    /// # Errors
    ///
    /// Returns the same contract errors as [`Self::predict_batch`].
    pub fn predict(&self, features: &Array1<f64>) -> FusionResult<Array1<f64>> {
        let batch = features.to_owned().insert_axis(Axis(0));
        Ok(self.predict_batch(&batch)?.row(0).to_owned())
    }

    /// Return the fixed output grid as `(n_z, n_r)`.
    pub fn grid_shape(&self) -> (usize, usize) {
        self.grid_shape
    }
}

fn config_error<T>(message: &str) -> FusionResult<T> {
    Err(FusionError::ConfigError(message.to_string()))
}

fn validate_layers(name: &str, layers: &[DenseLayer]) -> FusionResult<()> {
    if layers.is_empty() || layers.len() > 16 {
        return config_error(&format!("DeepONet {name} layer count is invalid"));
    }
    for (index, layer) in layers.iter().enumerate() {
        if layer.weights.ncols() != layer.bias.len()
            || layer.weights.nrows() == 0
            || layer.weights.ncols() == 0
        {
            return config_error(&format!("DeepONet {name} layer {index} shape is invalid"));
        }
        if index > 0 && layers[index - 1].weights.ncols() != layer.weights.nrows() {
            return config_error(&format!(
                "DeepONet {name} layer {index} input width is inconsistent"
            ));
        }
        if !all_finite_2d(&layer.weights) || !all_finite_1d(&layer.bias) {
            return config_error(&format!(
                "DeepONet {name} layer {index} contains non-finite values"
            ));
        }
    }
    Ok(())
}

fn forward(layers: &[DenseLayer], values: &Array2<f64>) -> Array2<f64> {
    let mut activation = values.to_owned();
    for (index, layer) in layers.iter().enumerate() {
        activation = activation.dot(&layer.weights) + &layer.bias;
        if index + 1 < layers.len() {
            activation.mapv_inplace(silu);
        }
    }
    activation
}

fn silu(value: f64) -> f64 {
    if value >= 0.0 {
        value / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        value * exponential / (1.0 + exponential)
    }
}

fn all_finite_1d(values: &Array1<f64>) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn all_finite_2d(values: &Array2<f64>) -> bool {
    values.iter().all(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn model() -> DeepOnetEquilibrium {
        DeepOnetEquilibrium::new(DeepOnetEquilibriumConfig {
            branch: vec![DenseLayer {
                weights: array![[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]],
                bias: array![0.0, 0.0],
            }],
            trunk: vec![DenseLayer {
                weights: array![[1.0, 0.5], [-0.25, 1.0]],
                bias: array![0.0, 0.0],
            }],
            input_mean: array![0.0, 0.0, 0.0],
            input_std: array![1.0, 1.0, 1.0],
            coordinates_rz_m: array![[3.0, -1.0], [4.0, -1.0], [3.0, 1.0], [4.0, 1.0]],
            coordinate_mean: array![3.5, 0.0],
            coordinate_std: array![0.5, 1.0],
            field_mean: array![0.0, 1.0, 2.0, 3.0],
            field_scale: 2.0,
            basis_width: 2,
            grid_shape: (2, 2),
        })
        .expect("valid fixed-grid DeepONet model should construct")
    }

    #[test]
    fn fixed_operator_matches_analytic_inner_product() {
        let predicted = model()
            .predict(&array![1.0, 2.0, 3.0])
            .expect("finite canonical controls should predict");
        let expected = array![
            -3.7123106012293743,
            5.065863991822648,
            -2.0658639918226482,
            6.712310601229374
        ];
        assert_eq!(predicted, expected);
        assert_eq!(model().grid_shape(), (2, 2));
    }

    #[test]
    fn batch_and_single_prediction_agree() {
        let runtime = model();
        let inputs = array![[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]];
        let batch = runtime
            .predict_batch(&inputs)
            .expect("finite canonical batch should predict");
        assert_eq!(
            batch.row(0),
            runtime.predict(&inputs.row(0).to_owned()).unwrap()
        );
        assert!(batch.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn construction_and_prediction_fail_closed() {
        assert!(matches!(
            DeepOnetEquilibrium::new(DeepOnetEquilibriumConfig {
                branch: vec![DenseLayer {
                    weights: array![[1.0], [1.0], [1.0]],
                    bias: array![0.0],
                }],
                trunk: vec![DenseLayer {
                    weights: array![[1.0], [1.0]],
                    bias: array![0.0],
                }],
                input_mean: array![0.0, 0.0, 0.0],
                input_std: array![1.0, 0.0, 1.0],
                coordinates_rz_m: array![[3.0, 0.0]],
                coordinate_mean: array![3.0, 0.0],
                coordinate_std: array![1.0, 1.0],
                field_mean: array![0.0],
                field_scale: 1.0,
                basis_width: 1,
                grid_shape: (1, 1),
            }),
            Err(FusionError::ConfigError(message)) if message.contains("scales must be positive")
        ));
        assert!(matches!(
            model().predict(&array![1.0, 2.0]),
            Err(FusionError::ConfigError(message)) if message.contains("feature width")
        ));
        assert!(matches!(
            model().predict(&array![1.0, f64::NAN, 3.0]),
            Err(FusionError::ConfigError(message)) if message.contains("must be finite")
        ));
    }
}
