// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — MPC
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Model Predictive Control for tokamak shape control.
//!
//! Port of `fusion_sota_mpc.py`.
//! Linear surrogate + gradient descent over prediction horizon.

use ndarray::{Array1, Array2};

/// Prediction horizon. Python: 10.
const HORIZON: usize = 10;

/// Gradient descent iterations. Python: 20.
const GD_ITERATIONS: usize = 20;

/// Learning rate. Python: 0.5.
const LR: f64 = 0.5;

/// Regularization weight. Python: 0.1.
const LAMBDA: f64 = 0.1;

/// Action clipping. Python: 2.0.
const ACTION_CLIP: f64 = 2.0;

/// Linear surrogate model: x_{t+1} = x_t + B·u_t.
pub struct NeuralSurrogate {
    /// Control impact matrix (n_state × n_coils).
    pub b_matrix: Array2<f64>,
}

impl NeuralSurrogate {
    pub fn new(b_matrix: Array2<f64>) -> Self {
        NeuralSurrogate { b_matrix }
    }

    /// Predict next state.
    pub fn predict(&self, state: &Array1<f64>, action: &Array1<f64>) -> Array1<f64> {
        state + &self.b_matrix.dot(action)
    }
}

/// Model Predictive Controller.
pub struct MPController {
    pub model: NeuralSurrogate,
    pub target: Array1<f64>,
    pub horizon: usize,
}

impl MPController {
    pub fn new(model: NeuralSurrogate, target: Array1<f64>) -> Self {
        MPController {
            model,
            target,
            horizon: HORIZON,
        }
    }

    /// Plan optimal action via gradient descent over horizon.
    /// Returns first action to apply.
    pub fn plan(&self, current_state: &Array1<f64>) -> Array1<f64> {
        let n_coils = self.model.b_matrix.ncols();
        let mut actions: Vec<Array1<f64>> =
            (0..self.horizon).map(|_| Array1::zeros(n_coils)).collect();

        for _ in 0..GD_ITERATIONS {
            // Forward rollout to compute gradients
            let mut grads: Vec<Array1<f64>> =
                (0..self.horizon).map(|_| Array1::zeros(n_coils)).collect();

            let mut state = current_state.clone();
            for t in 0..self.horizon {
                let next = self.model.predict(&state, &actions[t]);
                let error = &next - &self.target;
                // Gradient of ||error||² w.r.t. u_t = B^T · error
                let grad = self.model.b_matrix.t().dot(&error) + &actions[t] * LAMBDA;
                grads[t] = grad;
                state = next;
            }

            // Update actions
            for t in 0..self.horizon {
                actions[t] = &actions[t] - &(&grads[t] * LR);
                // Clip
                for v in actions[t].iter_mut() {
                    *v = v.clamp(-ACTION_CLIP, ACTION_CLIP);
                }
            }
        }

        actions[0].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surrogate_prediction() {
        let b = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let model = NeuralSurrogate::new(b);
        let state = Array1::from_vec(vec![1.0, 2.0]);
        let action = Array1::from_vec(vec![0.5, -0.3]);
        let next = model.predict(&state, &action);
        assert!((next[0] - 1.5).abs() < 1e-10);
        assert!((next[1] - 1.7).abs() < 1e-10);
    }

    #[test]
    fn test_mpc_moves_toward_target() {
        let b = Array2::from_shape_vec((2, 2), vec![0.1, 0.0, 0.0, 0.1]).unwrap();
        let model = NeuralSurrogate::new(b);
        let target = Array1::from_vec(vec![6.0, 0.0]);
        let mpc = MPController::new(model, target.clone());

        let state = Array1::from_vec(vec![5.0, 1.0]);
        let action = mpc.plan(&state);

        // Action should push state toward target
        assert!(action[0] > 0.0, "Should push R positive: {}", action[0]);
        assert!(action[1] < 0.0, "Should push Z negative: {}", action[1]);
    }

    #[test]
    fn test_mpc_action_clipped() {
        let b = Array2::from_shape_vec((2, 2), vec![10.0, 0.0, 0.0, 10.0]).unwrap();
        let model = NeuralSurrogate::new(b);
        let target = Array1::from_vec(vec![100.0, 0.0]);
        let mpc = MPController::new(model, target);

        let state = Array1::from_vec(vec![0.0, 0.0]);
        let action = mpc.plan(&state);
        for &v in action.iter() {
            assert!(
                v.abs() <= ACTION_CLIP + 1e-10,
                "Action should be clipped: {v}"
            );
        }
    }
}
