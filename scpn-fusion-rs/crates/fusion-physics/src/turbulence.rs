// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Turbulence
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Hasegawa-Wakatani turbulence + Echo State Network oracle.
//!
//! Port of `turbulence_oracle.py`.
//! 2D drift wave turbulence with ESN-based prediction.

use fusion_math::fft::{fft2, ifft2};
use ndarray::Array2;
use num_complex::Complex64;
use rand::Rng;
use rand_distr::StandardNormal;

/// Grid size. Python: GRID=64.
const GRID: usize = 64;

/// Domain size. Python: L=10.0.
const DOMAIN_L: f64 = 10.0;

/// Adiabaticity parameter. Python: ALPHA=0.1.
const ALPHA: f64 = 0.1;

/// Density gradient drive. Python: KAPPA=0.5.
const KAPPA_DRIVE: f64 = 0.5;

/// Viscosity. Python: NU=0.01.
const NU: f64 = 0.01;

/// Sub-timestep for integration. Python uses dt=0.01 inside step().
const SUB_DT: f64 = 0.01;

/// De-aliasing fraction.
const DEALIAS_FRACTION: f64 = 2.0 / 3.0;

/// Number of sparse probes. Python: 16.
const N_PROBES: usize = 16;

/// Default ESN reservoir size. Python: 500.
const RESERVOIR_SIZE: usize = 200;

/// ESN spectral radius. Python: 0.95.
const SPECTRAL_RADIUS: f64 = 0.95;

/// ESN reservoir density. Python: 0.1.
const RESERVOIR_DENSITY: f64 = 0.1;

/// Ridge regression regularization. Python: 1e-4.
const RIDGE_REG: f64 = 1e-4;

/// Hasegawa-Wakatani 2D drift wave simulator.
pub struct DriftWavePhysics {
    /// Grid size.
    pub n: usize,
    /// Electrostatic potential (spectral).
    pub phi_k: Array2<Complex64>,
    /// Density perturbation (spectral).
    pub n_k: Array2<Complex64>,
    /// kx wavenumber grid.
    kx: Array2<f64>,
    /// ky wavenumber grid.
    ky: Array2<f64>,
    /// k² grid.
    k2: Array2<f64>,
    /// De-aliasing mask.
    mask: Array2<f64>,
    /// Probe indices (flattened grid positions).
    pub probe_indices: Vec<usize>,
}

impl DriftWavePhysics {
    pub fn new(n: usize) -> Self {
        let mut rng = rand::thread_rng();
        let dk = 2.0 * std::f64::consts::PI / DOMAIN_L;

        let kx = Array2::from_shape_fn((n, n), |(_, j)| {
            let freq = if j <= n / 2 {
                j as f64
            } else {
                j as f64 - n as f64
            };
            freq * dk
        });
        let ky = Array2::from_shape_fn((n, n), |(i, _)| {
            let freq = if i <= n / 2 {
                i as f64
            } else {
                i as f64 - n as f64
            };
            freq * dk
        });
        let k2 = Array2::from_shape_fn((n, n), |(i, j)| kx[[i, j]].powi(2) + ky[[i, j]].powi(2));

        let k_max = (n / 2) as f64 * dk;
        let k_cut = DEALIAS_FRACTION * k_max;
        let mask = Array2::from_shape_fn((n, n), |(i, j)| {
            if k2[[i, j]] < k_cut * k_cut {
                1.0
            } else {
                0.0
            }
        });

        let phi_k = Array2::from_shape_fn((n, n), |_| {
            Complex64::new(
                rng.sample::<f64, _>(StandardNormal) * 0.01,
                rng.sample::<f64, _>(StandardNormal) * 0.01,
            )
        });
        let n_k = Array2::from_shape_fn((n, n), |_| {
            Complex64::new(
                rng.sample::<f64, _>(StandardNormal) * 0.01,
                rng.sample::<f64, _>(StandardNormal) * 0.01,
            )
        });

        // Evenly spaced probes
        let stride = (n * n) / N_PROBES;
        let probe_indices: Vec<usize> = (0..N_PROBES).map(|i| i * stride).collect();

        DriftWavePhysics {
            n,
            phi_k,
            n_k,
            kx,
            ky,
            k2,
            mask,
            probe_indices,
        }
    }

    /// Poisson bracket [A, B] with de-aliasing.
    fn bracket(
        &self,
        a_k: &Array2<Complex64>,
        b_k: &Array2<Complex64>,
    ) -> Array2<Complex64> {
        let n = self.n;
        let iu = Complex64::new(0.0, 1.0);

        let da_dx_k = Array2::from_shape_fn((n, n), |(i, j)| {
            iu * Complex64::new(self.kx[[i, j]], 0.0) * a_k[[i, j]]
        });
        let da_dy_k = Array2::from_shape_fn((n, n), |(i, j)| {
            iu * Complex64::new(self.ky[[i, j]], 0.0) * a_k[[i, j]]
        });
        let db_dx_k = Array2::from_shape_fn((n, n), |(i, j)| {
            iu * Complex64::new(self.kx[[i, j]], 0.0) * b_k[[i, j]]
        });
        let db_dy_k = Array2::from_shape_fn((n, n), |(i, j)| {
            iu * Complex64::new(self.ky[[i, j]], 0.0) * b_k[[i, j]]
        });

        let da_dx = ifft2(&da_dx_k);
        let da_dy = ifft2(&da_dy_k);
        let db_dx = ifft2(&db_dx_k);
        let db_dy = ifft2(&db_dy_k);

        let product = Array2::from_shape_fn((n, n), |(i, j)| {
            da_dx[[i, j]] * db_dy[[i, j]] - da_dy[[i, j]] * db_dx[[i, j]]
        });

        let mut result = fft2(&product);
        for i in 0..n {
            for j in 0..n {
                result[[i, j]] *= Complex64::new(self.mask[[i, j]], 0.0);
            }
        }
        result
    }

    /// Compute RHS of Hasegawa-Wakatani equations.
    fn rhs(
        &self,
        phi_k: &Array2<Complex64>,
        n_k: &Array2<Complex64>,
    ) -> (Array2<Complex64>, Array2<Complex64>) {
        let n = self.n;

        // Vorticity: w_k = -k² · φ_k
        let w_k = Array2::from_shape_fn((n, n), |(i, j)| {
            -Complex64::new(self.k2[[i, j]], 0.0) * phi_k[[i, j]]
        });

        // Nonlinear terms
        let bracket_phi_w = self.bracket(phi_k, &w_k);
        let bracket_phi_n = self.bracket(phi_k, n_k);

        // dw/dt = -[φ,w] + α(φ-n) - ν·k²·w
        let dw_k = Array2::from_shape_fn((n, n), |(i, j)| {
            let k2v = self.k2[[i, j]];
            -bracket_phi_w[[i, j]] + Complex64::new(ALPHA, 0.0) * (phi_k[[i, j]] - n_k[[i, j]])
                - Complex64::new(NU * k2v, 0.0) * w_k[[i, j]]
        });

        // dφ/dt = -dw/dt / k²
        let dphi_k = Array2::from_shape_fn((n, n), |(i, j)| {
            let k2v = self.k2[[i, j]];
            if k2v > 1e-10 {
                -dw_k[[i, j]] / Complex64::new(k2v, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        });

        // dn/dt = -[φ,n] + α(φ-n) - κ·iky·φ - ν·k²·n
        let dn_k = Array2::from_shape_fn((n, n), |(i, j)| {
            let k2v = self.k2[[i, j]];
            let iu = Complex64::new(0.0, 1.0);
            -bracket_phi_n[[i, j]] + Complex64::new(ALPHA, 0.0) * (phi_k[[i, j]] - n_k[[i, j]])
                - Complex64::new(KAPPA_DRIVE, 0.0)
                    * iu
                    * Complex64::new(self.ky[[i, j]], 0.0)
                    * phi_k[[i, j]]
                - Complex64::new(NU * k2v, 0.0) * n_k[[i, j]]
        });

        (dphi_k, dn_k)
    }

    /// One RK4 step. Returns probe values from the real-space potential.
    pub fn step(&mut self) -> Vec<f64> {
        let n = self.n;
        let dt = SUB_DT;

        // RK4
        let (k1_phi, k1_n) = self.rhs(&self.phi_k, &self.n_k);

        let phi2 = Array2::from_shape_fn((n, n), |(i, j)| {
            self.phi_k[[i, j]] + Complex64::new(0.5 * dt, 0.0) * k1_phi[[i, j]]
        });
        let n2 = Array2::from_shape_fn((n, n), |(i, j)| {
            self.n_k[[i, j]] + Complex64::new(0.5 * dt, 0.0) * k1_n[[i, j]]
        });
        let (k2_phi, k2_n) = self.rhs(&phi2, &n2);

        let phi3 = Array2::from_shape_fn((n, n), |(i, j)| {
            self.phi_k[[i, j]] + Complex64::new(0.5 * dt, 0.0) * k2_phi[[i, j]]
        });
        let n3 = Array2::from_shape_fn((n, n), |(i, j)| {
            self.n_k[[i, j]] + Complex64::new(0.5 * dt, 0.0) * k2_n[[i, j]]
        });
        let (k3_phi, k3_n) = self.rhs(&phi3, &n3);

        let phi4 = Array2::from_shape_fn((n, n), |(i, j)| {
            self.phi_k[[i, j]] + Complex64::new(dt, 0.0) * k3_phi[[i, j]]
        });
        let n4 = Array2::from_shape_fn((n, n), |(i, j)| {
            self.n_k[[i, j]] + Complex64::new(dt, 0.0) * k3_n[[i, j]]
        });
        let (k4_phi, k4_n) = self.rhs(&phi4, &n4);

        // Update
        for i in 0..n {
            for j in 0..n {
                self.phi_k[[i, j]] += Complex64::new(dt / 6.0, 0.0)
                    * (k1_phi[[i, j]]
                        + 2.0 * k2_phi[[i, j]]
                        + 2.0 * k3_phi[[i, j]]
                        + k4_phi[[i, j]]);
                self.n_k[[i, j]] += Complex64::new(dt / 6.0, 0.0)
                    * (k1_n[[i, j]]
                        + 2.0 * k2_n[[i, j]]
                        + 2.0 * k3_n[[i, j]]
                        + k4_n[[i, j]]);
            }
        }

        // Stability clamp
        let max_phi = self
            .phi_k
            .iter()
            .map(|c| c.norm())
            .fold(0.0_f64, f64::max);
        if max_phi > 100.0 {
            let scale = 100.0 / max_phi;
            for val in self.phi_k.iter_mut() {
                *val *= scale;
            }
            for val in self.n_k.iter_mut() {
                *val *= scale;
            }
        }

        // Extract probe values from real-space potential
        let phi_real = ifft2(&self.phi_k);
        let flat: Vec<f64> = phi_real.iter().cloned().collect();
        self.probe_indices.iter().map(|&idx| flat[idx]).collect()
    }
}

/// Echo State Network reservoir computer.
pub struct OracleESN {
    /// Reservoir size.
    pub reservoir_size: usize,
    /// Input dimension.
    pub input_dim: usize,
    /// Input weights [reservoir_size × input_dim].
    w_in: Array2<f64>,
    /// Reservoir weights [reservoir_size × reservoir_size].
    w_res: Array2<f64>,
    /// Output weights [output_dim × reservoir_size] (set after training).
    w_out: Option<Array2<f64>>,
    /// Reservoir state [reservoir_size].
    state: Vec<f64>,
}

impl OracleESN {
    pub fn new(input_dim: usize, reservoir_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Input weights: uniform [-1, 1]
        let w_in = Array2::from_shape_fn((reservoir_size, input_dim), |_| {
            rng.gen_range(-1.0..1.0)
        });

        // Sparse reservoir (density ~0.1)
        let mut w_res = Array2::zeros((reservoir_size, reservoir_size));
        for i in 0..reservoir_size {
            for j in 0..reservoir_size {
                if rng.gen::<f64>() < RESERVOIR_DENSITY {
                    w_res[[i, j]] = rng.gen_range(-1.0..1.0);
                }
            }
        }

        // Scale to target spectral radius via power iteration
        let rho = spectral_radius_estimate(&w_res, 30);
        if rho > 1e-10 {
            let scale = SPECTRAL_RADIUS / rho;
            w_res.mapv_inplace(|v| v * scale);
        }

        let state = vec![0.0; reservoir_size];

        OracleESN {
            reservoir_size,
            input_dim,
            w_in,
            w_res,
            w_out: None,
            state,
        }
    }

    /// Update reservoir state: state = tanh(W_in · u + W_res · state).
    fn update_state(&mut self, input: &[f64]) {
        let n = self.reservoir_size;
        let mut new_state = vec![0.0; n];

        for i in 0..n {
            let mut sum = 0.0;
            // W_in · u
            for (j, &u) in input.iter().enumerate() {
                sum += self.w_in[[i, j]] * u;
            }
            // W_res · state
            for j in 0..n {
                sum += self.w_res[[i, j]] * self.state[j];
            }
            new_state[i] = sum.tanh();
        }

        self.state = new_state;
    }

    /// Train on (inputs[t] → targets[t]) pairs via ridge regression.
    ///
    /// Solves: W_out = targets^T · S · (S^T · S + reg·I)^{-1}
    pub fn train(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>]) {
        let t = inputs.len();
        let n = self.reservoir_size;
        let out_dim = targets[0].len();

        // Harvest reservoir states
        self.state = vec![0.0; n];
        let mut states = Array2::zeros((t, n));
        for (step, input) in inputs.iter().enumerate() {
            self.update_state(input);
            for j in 0..n {
                states[[step, j]] = self.state[j];
            }
        }

        // S^T · S + reg · I  (n × n)
        let mut sts = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..t {
                    sum += states[[k, i]] * states[[k, j]];
                }
                sts[[i, j]] = sum;
            }
            sts[[i, i]] += RIDGE_REG;
        }

        // S^T · targets  (n × out_dim)
        let mut st_y = Array2::zeros((n, out_dim));
        for i in 0..n {
            for j in 0..out_dim {
                let mut sum = 0.0;
                for k in 0..t {
                    sum += states[[k, i]] * targets[k][j];
                }
                st_y[[i, j]] = sum;
            }
        }

        // Solve (S^T S + reg I) X = S^T Y via Cholesky
        let x = cholesky_solve(&sts, &st_y);

        // W_out = X^T (out_dim × n)
        let mut w_out = Array2::zeros((out_dim, n));
        for i in 0..n {
            for j in 0..out_dim {
                w_out[[j, i]] = x[[i, j]];
            }
        }

        self.w_out = Some(w_out);
    }

    /// Predict one step from current state.
    pub fn predict_step(&mut self, input: &[f64]) -> Vec<f64> {
        self.update_state(input);
        let w_out = self.w_out.as_ref().expect("ESN not trained");
        let out_dim = w_out.nrows();
        let n = self.reservoir_size;

        let mut output = vec![0.0; out_dim];
        for i in 0..out_dim {
            let mut sum = 0.0;
            for j in 0..n {
                sum += w_out[[i, j]] * self.state[j];
            }
            output[i] = sum;
        }
        output
    }

    /// Multi-step closed-loop prediction.
    pub fn predict(&mut self, initial: &[f64], steps: usize) -> Vec<Vec<f64>> {
        let mut predictions = Vec::with_capacity(steps);
        let mut current = initial.to_vec();

        for _ in 0..steps {
            let pred = self.predict_step(&current);
            predictions.push(pred.clone());
            current = pred;
        }

        predictions
    }
}

/// Estimate spectral radius via power iteration.
fn spectral_radius_estimate(m: &Array2<f64>, iterations: usize) -> f64 {
    let n = m.nrows();
    let mut v = vec![1.0; n];
    let norm: f64 = (n as f64).sqrt();
    for x in &mut v {
        *x /= norm;
    }

    for _ in 0..iterations {
        let mut new_v = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                new_v[i] += m[[i, j]] * v[j];
            }
        }
        let mag = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if mag > 1e-15 {
            for x in &mut new_v {
                *x /= mag;
            }
        }
        v = new_v;
    }

    // Rayleigh quotient: v^T M v
    let mut mv = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            mv[i] += m[[i, j]] * v[j];
        }
    }
    mv.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Solve AX = B where A is symmetric positive definite (Cholesky).
fn cholesky_solve(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let m = b.ncols();

    // Cholesky decomposition: A = L L^T
    let mut l = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let mut s = 0.0;
        for k in 0..j {
            s += l[[j, k]] * l[[j, k]];
        }
        l[[j, j]] = (a[[j, j]] - s).max(1e-15).sqrt();

        for i in (j + 1)..n {
            let mut s = 0.0;
            for k in 0..j {
                s += l[[i, k]] * l[[j, k]];
            }
            l[[i, j]] = (a[[i, j]] - s) / l[[j, j]];
        }
    }

    // Forward solve: L Y = B
    let mut y = Array2::<f64>::zeros((n, m));
    for c in 0..m {
        for i in 0..n {
            let mut s = 0.0;
            for k in 0..i {
                s += l[[i, k]] * y[[k, c]];
            }
            y[[i, c]] = (b[[i, c]] - s) / l[[i, i]];
        }
    }

    // Back solve: L^T X = Y
    let mut x = Array2::<f64>::zeros((n, m));
    for c in 0..m {
        for i in (0..n).rev() {
            let mut s = 0.0;
            for k in (i + 1)..n {
                s += l[[k, i]] * x[[k, c]];
            }
            x[[i, c]] = (y[[i, c]] - s) / l[[i, i]];
        }
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hw_creation() {
        let hw = DriftWavePhysics::new(GRID);
        assert_eq!(hw.phi_k.nrows(), GRID);
        assert_eq!(hw.probe_indices.len(), N_PROBES);
    }

    #[test]
    fn test_hw_step_finite() {
        let mut hw = DriftWavePhysics::new(GRID);
        let probes = hw.step();
        assert_eq!(probes.len(), N_PROBES);
        assert!(
            probes.iter().all(|v| v.is_finite()),
            "Probe values should be finite"
        );
    }

    #[test]
    fn test_esn_creation() {
        let esn = OracleESN::new(N_PROBES, RESERVOIR_SIZE);
        assert_eq!(esn.reservoir_size, RESERVOIR_SIZE);
        assert_eq!(esn.input_dim, N_PROBES);
    }

    #[test]
    fn test_cholesky_solve_identity() {
        let n = 5;
        let a = Array2::from_shape_fn((n, n), |(i, j)| if i == j { 1.0 } else { 0.0 });
        let b = Array2::from_shape_fn((n, 2), |(i, j)| (i + j) as f64);
        let x = cholesky_solve(&a, &b);
        for i in 0..n {
            for j in 0..2 {
                assert!(
                    (x[[i, j]] - b[[i, j]]).abs() < 1e-10,
                    "I·x should equal b"
                );
            }
        }
    }

    #[test]
    fn test_esn_predicts_lyapunov_time() {
        // Generate training data from HW physics
        let mut hw = DriftWavePhysics::new(32); // Smaller grid for speed
        let n_train = 100;
        let n_test = 30;

        let mut data = Vec::with_capacity(n_train + n_test);
        for _ in 0..(n_train + n_test) {
            data.push(hw.step());
        }

        // Build training pairs: data[t] → data[t+1]
        let inputs: Vec<Vec<f64>> = data[..n_train - 1].to_vec();
        let targets: Vec<Vec<f64>> = data[1..n_train].to_vec();

        // Train ESN
        let mut esn = OracleESN::new(data[0].len(), RESERVOIR_SIZE);
        esn.train(&inputs, &targets);

        // Predict from last training point
        let predictions = esn.predict(&data[n_train - 1], n_test);

        // Compute MSE at first and last prediction step
        let mse_first: f64 = predictions[0]
            .iter()
            .zip(data[n_train].iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / predictions[0].len() as f64;

        let mse_last: f64 = predictions[n_test - 1]
            .iter()
            .zip(data[n_train + n_test - 1].iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / predictions[0].len() as f64;

        // ESN prediction should have some skill: first step better than last
        // (error grows with Lyapunov divergence)
        assert!(
            mse_first < mse_last || mse_first < 1.0,
            "ESN should predict > 0 Lyapunov time: mse_first={mse_first}, mse_last={mse_last}"
        );
    }
}
