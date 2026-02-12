//! FNO Turbulence Suppressor.
//!
//! Port of `fno_turbulence_suppressor.py`.
//! Spectral turbulence generation with Fourier Neural Operator control.

use fusion_math::fft::{fft2, ifft2};
use ndarray::Array2;
use num_complex::Complex64;
use rand::Rng;
use rand_distr::StandardNormal;

/// Fourier modes retained in FNO. Python: MODES=12.
const MODES: usize = 12;

/// Default grid size. Python: GRID_SIZE=64.
const GRID_SIZE: usize = 64;

/// Spectral turbulence generator (drift wave physics).
pub struct SpectralTurbulenceGenerator {
    /// Density fluctuation field.
    pub field: Array2<f64>,
    /// Grid size.
    pub size: usize,
    /// ky wavenumber grid (row index).
    ky: Array2<f64>,
    /// k² wavenumber squared.
    k2: Array2<f64>,
}

impl SpectralTurbulenceGenerator {
    pub fn new(size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let field = Array2::from_shape_fn((size, size), |_| {
            rng.sample::<f64, _>(StandardNormal) * 0.1
        });

        let ky = Array2::from_shape_fn((size, size), |(i, _)| {
            if i <= size / 2 {
                i as f64
            } else {
                i as f64 - size as f64
            }
        });
        let k2 = Array2::from_shape_fn((size, size), |(i, j)| {
            let kxi = if j <= size / 2 {
                j as f64
            } else {
                j as f64 - size as f64
            };
            let kyi = if i <= size / 2 {
                i as f64
            } else {
                i as f64 - size as f64
            };
            kxi * kxi + kyi * kyi
        });

        SpectralTurbulenceGenerator {
            field,
            size,
            ky,
            k2,
        }
    }

    /// Evolve turbulence one step in Fourier space.
    ///
    /// - Drift wave dispersion: ω = ky / (1 + k²)
    /// - Kolmogorov forcing at low-k (k² < 25)
    /// - Viscous + active damping
    pub fn step(&mut self, dt: f64, damping: f64) {
        let mut field_k = fft2(&self.field);
        let n = self.size;

        for i in 0..n {
            for j in 0..n {
                let k2v = self.k2[[i, j]];
                let kyv = self.ky[[i, j]];

                // Drift wave dispersion
                let omega = kyv / (1.0 + k2v);
                let phase = Complex64::new(0.0, -omega * dt);
                field_k[[i, j]] *= phase.exp();

                // Kolmogorov forcing at low-k
                if k2v > 0.0 && k2v < 25.0 {
                    field_k[[i, j]] *= Complex64::new(1.001, 0.0);
                }

                // Viscous + active damping
                let visc = (-0.001 * k2v * dt).exp();
                field_k[[i, j]] *= Complex64::new(visc * (1.0 - damping), 0.0);
            }
        }

        self.field = ifft2(&field_k);
    }

    /// Compute mean-square turbulence energy.
    pub fn energy(&self) -> f64 {
        self.field.iter().map(|v| v * v).sum::<f64>() / (self.size * self.size) as f64
    }
}

/// FNO spectral convolution block.
pub struct FnoBlock {
    pub modes: usize,
    weights_re: Array2<f64>,
    weights_im: Array2<f64>,
}

impl FnoBlock {
    pub fn new(modes: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (modes * modes) as f64;
        let weights_re = Array2::from_shape_fn((modes, modes), |_| {
            rng.sample::<f64, _>(StandardNormal) * scale
        });
        let weights_im = Array2::from_shape_fn((modes, modes), |_| {
            rng.sample::<f64, _>(StandardNormal) * scale
        });

        FnoBlock {
            modes,
            weights_re,
            weights_im,
        }
    }

    /// Forward pass: spectral convolution.
    ///
    /// FFT → truncate to modes × modes → complex multiply → pad → IFFT.
    pub fn forward(&self, x_field: &Array2<f64>) -> Array2<f64> {
        let n = x_field.nrows();
        let x_k = fft2(x_field);
        let modes = self.modes.min(n);

        let mut out_k = Array2::from_elem((n, n), Complex64::new(0.0, 0.0));

        for i in 0..modes {
            for j in 0..modes {
                let xk = x_k[[i, j]];
                let w = Complex64::new(self.weights_re[[i, j]], self.weights_im[[i, j]]);
                out_k[[i, j]] = xk * w;
            }
        }

        ifft2(&out_k)
    }
}

/// FNO controller: predict turbulence and compute suppression factor.
pub struct FnoController {
    fno: FnoBlock,
}

impl FnoController {
    pub fn new() -> Self {
        FnoController {
            fno: FnoBlock::new(MODES),
        }
    }

    /// Predict turbulence and compute suppression factor [0, 1].
    pub fn predict_and_suppress(&self, field: &Array2<f64>) -> (f64, Array2<f64>) {
        let prediction = self.fno.forward(field);
        let energy: f64 =
            prediction.iter().map(|v| v * v).sum::<f64>() / prediction.len() as f64;
        let suppression = (energy * 10.0).tanh();
        (suppression, prediction)
    }
}

impl Default for FnoController {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turbulence_creation() {
        let gen = SpectralTurbulenceGenerator::new(GRID_SIZE);
        assert_eq!(gen.field.nrows(), GRID_SIZE);
        assert!(gen.energy() > 0.0, "Initial energy should be positive");
    }

    #[test]
    fn test_turbulence_step_finite() {
        let mut gen = SpectralTurbulenceGenerator::new(GRID_SIZE);
        gen.step(0.01, 0.0);
        assert!(
            gen.field.iter().all(|v| v.is_finite()),
            "Field should be finite after step"
        );
    }

    #[test]
    fn test_fno_forward_shape() {
        let fno = FnoBlock::new(MODES);
        let input = Array2::from_shape_fn((GRID_SIZE, GRID_SIZE), |_| 0.1);
        let output = fno.forward(&input);
        assert_eq!(output.shape(), &[GRID_SIZE, GRID_SIZE]);
    }

    #[test]
    fn test_energy_decreases_under_suppression() {
        let mut gen = SpectralTurbulenceGenerator::new(GRID_SIZE);
        let controller = FnoController::new();

        // Run without control to establish baseline energy
        for _ in 0..50 {
            gen.step(0.01, 0.0);
        }
        let e_before = gen.energy();

        // Run with control
        for _ in 0..50 {
            let (suppression, _) = controller.predict_and_suppress(&gen.field);
            gen.step(0.01, suppression);
        }
        let e_after = gen.energy();

        assert!(
            e_after < e_before,
            "Energy should decrease under suppression: {e_before} -> {e_after}"
        );
    }
}
