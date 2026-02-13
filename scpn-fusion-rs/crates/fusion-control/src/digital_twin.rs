// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Digital Twin
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Tokamak digital twin with simple MLP controller.
//!
//! Port of `tokamak_digital_twin.py`.
//! 2D thermal dynamics with q-profile and RL-trained neural controller.

use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::StandardNormal;

/// Grid size. Python: 40.
const GRID: usize = 40;

/// MLP hidden layer size. Python: 64.
const HIDDEN: usize = 64;

/// Learning rate. Python: 0.0001.
const LR: f64 = 0.0001;

/// Base diffusivity. Python: 0.01.
const D_BASE: f64 = 0.01;

/// Turbulent diffusivity (at magnetic islands). Python: 0.5.
const D_TURB: f64 = 0.5;

/// Default Ornstein-Uhlenbeck theta.
const DEFAULT_OU_THETA: f64 = 0.25;

/// Default Ornstein-Uhlenbeck sigma.
const DEFAULT_OU_SIGMA: f64 = 0.05;

/// Default noise time step.
const DEFAULT_OU_DT: f64 = 1.0;

/// Single-bit floating-point fault injection.
pub fn apply_bit_flip_fault(value: f64, bit_index: u8) -> f64 {
    let bit = u32::from(bit_index % 64);
    let flipped = f64::from_bits(value.to_bits() ^ (1_u64 << bit));
    if flipped.is_finite() {
        flipped
    } else {
        value
    }
}

/// Ornstein-Uhlenbeck noise process for sensor/actuator perturbations.
#[derive(Debug, Clone, Copy)]
pub struct NoiseInjectionLayer {
    pub theta: f64,
    pub sigma: f64,
    pub dt: f64,
    pub state: f64,
}

impl NoiseInjectionLayer {
    pub fn new(theta: f64, sigma: f64, dt: f64) -> Self {
        Self {
            theta: theta.max(0.0),
            sigma: sigma.max(0.0),
            dt: dt.max(1e-9),
            state: 0.0,
        }
    }

    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> f64 {
        let xi: f64 = rng.sample(StandardNormal);
        self.state += self.theta * (-self.state) * self.dt + self.sigma * self.dt.sqrt() * xi;
        self.state
    }
}

impl Default for NoiseInjectionLayer {
    fn default() -> Self {
        Self::new(DEFAULT_OU_THETA, DEFAULT_OU_SIGMA, DEFAULT_OU_DT)
    }
}

/// Simple feedforward neural network: input → 64 (tanh) → 1 (tanh).
pub struct SimpleMLP {
    pub w1: Array2<f64>,
    pub b1: Array1<f64>,
    pub w2: Array2<f64>,
    pub b2: Array1<f64>,
    // Cached activations
    z1: Array1<f64>,
    a1: Array1<f64>,
}

impl SimpleMLP {
    pub fn new(input_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale1 = (1.0 / input_dim as f64).sqrt();
        let scale2 = (1.0 / HIDDEN as f64).sqrt();

        let w1 = Array2::from_shape_fn((input_dim, HIDDEN), |_| {
            rng.sample::<f64, _>(StandardNormal) * scale1
        });
        let b1 = Array1::zeros(HIDDEN);
        let w2 = Array2::from_shape_fn((HIDDEN, 1), |_| {
            rng.sample::<f64, _>(StandardNormal) * scale2
        });
        let b2 = Array1::zeros(1);

        SimpleMLP {
            w1,
            b1,
            w2,
            b2,
            z1: Array1::zeros(HIDDEN),
            a1: Array1::zeros(HIDDEN),
        }
    }

    /// Forward pass. Returns scalar action in [-1, 1].
    pub fn forward(&mut self, x: &Array1<f64>) -> f64 {
        self.z1 = x.dot(&self.w1) + &self.b1;
        self.a1 = self.z1.mapv(|v| v.tanh());
        let z2 = self.a1.dot(&self.w2) + &self.b2;
        z2[0].tanh()
    }

    /// Backprop with advantage-weighted gradient. Returns loss.
    pub fn train_step(&mut self, x: &Array1<f64>, advantage: f64) -> f64 {
        let out = self.forward(x);
        let grad_out = -advantage;

        // Through output tanh
        let d_z2 = grad_out * (1.0 - out * out);

        // dW2 = a1^T · d_z2
        let d_w2 = self.a1.clone().insert_axis(ndarray::Axis(1)) * d_z2;
        let d_b2 = Array1::from_vec(vec![d_z2]);

        // dA1 = d_z2 · W2^T
        let d_a1 = &self.w2 * d_z2;
        let d_a1 = d_a1.column(0).to_owned();

        // Through hidden tanh
        let d_z1 = &d_a1 * &self.a1.mapv(|v| 1.0 - v * v);

        // dW1 = x^T · d_z1
        let d_w1_col = &d_z1;
        let x_col = x.clone().insert_axis(ndarray::Axis(1));
        let d_z1_row = d_w1_col.clone().insert_axis(ndarray::Axis(0));
        let d_w1 = x_col.dot(&d_z1_row);
        let d_b1 = d_z1.clone();

        // Update
        self.w1 = &self.w1 - &(&d_w1 * LR);
        self.b1 = &self.b1 - &(&d_b1 * LR);
        self.w2 = &self.w2 - &(&d_w2 * LR);
        self.b2 = &self.b2 - &(&d_b2 * LR);

        grad_out.abs()
    }
}

/// 2D plasma with diffusion and q-profile.
pub struct Plasma2D {
    pub temp: Array2<f64>,
    pub mask: Array2<f64>,
    pub q0: f64,
    pub qa: f64,
    pub core_temp_history: Vec<f64>,
}

impl Plasma2D {
    pub fn new() -> Self {
        let center = GRID / 2;
        let radius = GRID as f64 / 2.5;

        let mask = Array2::from_shape_fn((GRID, GRID), |(i, j)| {
            let di = i as f64 - center as f64;
            let dj = j as f64 - center as f64;
            if (di * di + dj * dj).sqrt() < radius {
                1.0
            } else {
                0.0
            }
        });

        Plasma2D {
            temp: Array2::zeros((GRID, GRID)),
            mask,
            q0: 1.0,
            qa: 3.0,
            core_temp_history: Vec::new(),
        }
    }

    /// One physics step. Returns (core_temp, avg_temp).
    pub fn step(&mut self, action: f64) -> (f64, f64) {
        let n = GRID;
        let center = n / 2;

        // Update q-profile with current drive
        let mod_q0 = self.q0 - 0.2 * action;
        let mod_qa = self.qa + 0.5 * action;

        // Compute rational surface danger map
        let mut diffusivity = Array2::from_elem((n, n), D_BASE);
        let resonances = [1.5, 2.0, 2.5, 3.0];
        let island_width = 0.05;

        for i in 0..n {
            for j in 0..n {
                let di = i as f64 - center as f64;
                let dj = j as f64 - center as f64;
                let r_norm = (di * di + dj * dj).sqrt() / (n as f64 / 2.5);
                let q = mod_q0 + (mod_qa - mod_q0) * r_norm * r_norm;
                for &res in &resonances {
                    if (q - res).abs() < island_width {
                        diffusivity[[i, j]] = D_TURB;
                    }
                }
            }
        }

        // Source: core heating
        self.temp[[center, center]] += 5.0;

        // Laplacian (periodic boundary via wrapping)
        let mut laplacian = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let up = self.temp[[(i + n - 1) % n, j]];
                let down = self.temp[[(i + 1) % n, j]];
                let left = self.temp[[i, (j + n - 1) % n]];
                let right = self.temp[[i, (j + 1) % n]];
                laplacian[[i, j]] = up + down + left + right - 4.0 * self.temp[[i, j]];
            }
        }

        // Radiation cooling
        let radiation = self.temp.mapv(|t| 0.0001 * t * t);

        // Update
        self.temp = &self.temp + &(&diffusivity * &laplacian) - &radiation;

        // Boundary
        for i in 0..n {
            for j in 0..n {
                if self.mask[[i, j]] < 0.5 {
                    self.temp[[i, j]] = 0.0;
                }
                self.temp[[i, j]] = self.temp[[i, j]].clamp(0.0, 100.0);
            }
        }

        let core_temp = self.temp[[center, center]];
        self.core_temp_history.push(core_temp);

        let total: f64 = self.temp.iter().sum();
        let count = self.mask.iter().filter(|&&v| v > 0.5).count();
        let avg = if count > 0 { total / count as f64 } else { 0.0 };

        (core_temp, avg)
    }

    /// One step with additive process noise on central heating source.
    pub fn step_with_process_noise(&mut self, action: f64, process_noise: f64) -> (f64, f64) {
        let center = GRID / 2;
        self.temp[[center, center]] =
            (self.temp[[center, center]] + process_noise).clamp(0.0, 100.0);
        self.step(action)
    }

    /// Sensor readout helper with additive measurement noise.
    pub fn measure_core_temp(&self, measurement_noise: f64) -> f64 {
        let center = GRID / 2;
        (self.temp[[center, center]] + measurement_noise).clamp(0.0, 100.0)
    }
}

impl Default for Plasma2D {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_mlp_forward_bounded() {
        let mut mlp = SimpleMLP::new(GRID);
        let x = Array1::zeros(GRID);
        let out = mlp.forward(&x);
        assert!(
            (-1.0..=1.0).contains(&out),
            "Output should be in [-1, 1]: {out}"
        );
    }

    #[test]
    fn test_mlp_train_step() {
        let mut mlp = SimpleMLP::new(GRID);
        let x = Array1::from_elem(GRID, 0.5);
        let loss = mlp.train_step(&x, 1.0);
        assert!(loss.is_finite(), "Loss should be finite: {loss}");
    }

    #[test]
    fn test_plasma_heats_up() {
        let mut plasma = Plasma2D::new();
        for _ in 0..100 {
            plasma.step(0.0);
        }
        let core = plasma.temp[[GRID / 2, GRID / 2]];
        assert!(core > 0.0, "Core should heat up: {core}");
    }

    #[test]
    fn test_plasma_bounded() {
        let mut plasma = Plasma2D::new();
        for _ in 0..500 {
            plasma.step(0.0);
        }
        for &t in plasma.temp.iter() {
            assert!((0.0..=100.0).contains(&t), "Temp out of bounds: {t}");
        }
    }

    #[test]
    fn test_bit_flip_fault_changes_value_or_preserves_finite() {
        let value = 0.75_f64;
        let flipped = apply_bit_flip_fault(value, 7);
        assert!(flipped.is_finite(), "Flipped value must remain finite");
        assert_ne!(flipped, value, "Bit flip should modify value for this case");
    }

    #[test]
    fn test_ou_noise_deterministic_with_seed() {
        let mut rng1 = StdRng::seed_from_u64(1234);
        let mut rng2 = StdRng::seed_from_u64(1234);
        let mut n1 = NoiseInjectionLayer::default();
        let mut n2 = NoiseInjectionLayer::default();
        for _ in 0..32 {
            let a = n1.step(&mut rng1);
            let b = n2.step(&mut rng2);
            assert!(
                (a - b).abs() < 1e-12,
                "Noise sequence should be deterministic"
            );
        }
    }

    #[test]
    fn test_plasma_noise_helpers_bounded() {
        let mut plasma = Plasma2D::new();
        for _ in 0..32 {
            let _ = plasma.step_with_process_noise(0.0, 0.5);
        }
        let meas = plasma.measure_core_temp(0.2);
        assert!(
            (0.0..=100.0).contains(&meas),
            "Measurement must remain bounded"
        );
    }
}
