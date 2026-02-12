//! Soft X-ray tomographic reconstruction.
//!
//! Port of `tomography.py`.
//! Builds Radon geometry matrix from bolometer chords and reconstructs
//! emissivity profiles using Tikhonov-regularised non-negative least squares.

use ndarray::{Array1, Array2};

/// A bolometer chord: ((start_R, start_Z), (end_R, end_Z)).
pub type Chord = ((f64, f64), (f64, f64));

/// Default reconstruction grid resolution. Python: 20.
const DEFAULT_RES: usize = 20;

/// Ray march samples per chord. Python: 100.
const RAY_SAMPLES: usize = 100;

/// Tikhonov regularization weight. Python: 0.1.
const LAMBDA_REG: f64 = 0.1;

/// Projected gradient iterations for NNLS.
const NNLS_ITERS: usize = 1000;

/// Tomographic reconstruction engine.
pub struct PlasmaTomography {
    /// Reconstruction grid resolution (res × res).
    pub res: usize,
    /// Geometry matrix A: (n_chords, res²).
    pub geometry: Array2<f64>,
    /// R coordinates of reconstruction grid.
    pub r_grid: Array1<f64>,
    /// Z coordinates of reconstruction grid.
    pub z_grid: Array1<f64>,
}

impl PlasmaTomography {
    /// Build tomography from bolometer chord geometry.
    ///
    /// `chords`: list of (start, end) where each is (R, Z).
    /// `r_range`: (R_min, R_max) of reconstruction domain.
    /// `z_range`: (Z_min, Z_max) of reconstruction domain.
    pub fn new(
        chords: &[Chord],
        r_range: (f64, f64),
        z_range: (f64, f64),
        res: usize,
    ) -> Self {
        let r_grid = Array1::linspace(r_range.0, r_range.1, res);
        let z_grid = Array1::linspace(z_range.0, z_range.1, res);
        let dr = r_grid[1] - r_grid[0];
        let dz = z_grid[1] - z_grid[0];

        let n_chords = chords.len();
        let n_pixels = res * res;
        let mut geometry = Array2::zeros((n_chords, n_pixels));

        for (i, &(start, end)) in chords.iter().enumerate() {
            let (sr, sz) = start;
            let (er, ez) = end;
            let length = ((er - sr).powi(2) + (ez - sz).powi(2)).sqrt();
            let dl = length / RAY_SAMPLES as f64;

            for k in 0..RAY_SAMPLES {
                let t = k as f64 / RAY_SAMPLES as f64;
                let r = sr + t * (er - sr);
                let z = sz + t * (ez - sz);

                let ir = ((r - r_range.0) / dr).floor() as isize;
                let iz = ((z - z_range.0) / dz).floor() as isize;

                if ir >= 0 && ir < res as isize && iz >= 0 && iz < res as isize {
                    let pixel_idx = iz as usize * res + ir as usize;
                    geometry[[i, pixel_idx]] += dl;
                }
            }
        }

        PlasmaTomography {
            res,
            geometry,
            r_grid,
            z_grid,
        }
    }

    /// Build with default resolution.
    pub fn with_default_res(
        chords: &[Chord],
        r_range: (f64, f64),
        z_range: (f64, f64),
    ) -> Self {
        Self::new(chords, r_range, z_range, DEFAULT_RES)
    }

    /// Reconstruct emissivity from bolometer signals.
    ///
    /// Solves: min ||Ax - b||² + λ||x||² subject to x ≥ 0
    /// using projected gradient descent.
    ///
    /// Returns flattened reconstruction of length res².
    pub fn reconstruct(&self, signals: &[f64]) -> Vec<f64> {
        let n_chords = self.geometry.nrows();
        let n_pixels = self.geometry.ncols();

        assert_eq!(
            signals.len(),
            n_chords,
            "Signal length {} != chord count {}",
            signals.len(),
            n_chords
        );

        let b = Array1::from_vec(signals.to_vec());

        // Precompute A^T A + λI and A^T b
        let at = self.geometry.t();
        let ata = at.dot(&self.geometry);
        let atb = at.dot(&b);

        // Lipschitz constant: ||A^T A||_2 + λ ≤ ||A||_F^2 + λ
        let a_frob_sq: f64 = self.geometry.iter().map(|v| v * v).sum();
        let lipschitz = a_frob_sq + LAMBDA_REG;
        let step = 1.0 / (lipschitz + 1e-10);

        // Projected gradient descent (accelerated with Nesterov momentum)
        let mut x = Array1::zeros(n_pixels);
        let mut x_prev = x.clone();

        for k in 0..NNLS_ITERS {
            let momentum = k as f64 / (k as f64 + 3.0);
            let y = &x + &(&(&x - &x_prev) * momentum);
            // gradient at y: (A^T A + λI) y - A^T b
            let grad = ata.dot(&y) + &y * LAMBDA_REG - &atb;
            x_prev = x.clone();
            x = (&y - &(&grad * step)).mapv(|v| v.max(0.0));
        }

        x.to_vec()
    }

    /// Reconstruct and reshape to 2D grid (res × res).
    pub fn reconstruct_2d(&self, signals: &[f64]) -> Array2<f64> {
        let flat = self.reconstruct(signals);
        Array2::from_shape_vec((self.res, self.res), flat)
            .expect("reshape failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build simple test chords: fan from (5.0, 4.0) to targets at Z=-3.0.
    fn test_chords() -> Vec<((f64, f64), (f64, f64))> {
        let n = 8;
        (0..n)
            .map(|i| {
                let r_target = 2.0 + 6.0 * i as f64 / (n - 1) as f64;
                ((5.0, 4.0), (r_target, -3.0))
            })
            .collect()
    }

    #[test]
    fn test_geometry_matrix_shape() {
        let chords = test_chords();
        let tomo = PlasmaTomography::new(&chords, (1.0, 9.0), (-5.0, 5.0), 10);
        assert_eq!(tomo.geometry.nrows(), 8);
        assert_eq!(tomo.geometry.ncols(), 100); // 10×10
    }

    #[test]
    fn test_geometry_non_negative() {
        let chords = test_chords();
        let tomo = PlasmaTomography::new(&chords, (1.0, 9.0), (-5.0, 5.0), 10);
        for &v in tomo.geometry.iter() {
            assert!(v >= 0.0, "Geometry matrix has negative entry: {v}");
        }
    }

    #[test]
    fn test_geometry_has_nonzero_rows() {
        let chords = test_chords();
        let tomo = PlasmaTomography::new(&chords, (1.0, 9.0), (-5.0, 5.0), 10);
        for i in 0..tomo.geometry.nrows() {
            let row_sum: f64 = tomo.geometry.row(i).iter().sum();
            assert!(
                row_sum > 0.0,
                "Chord {i} has zero geometry (no pixels hit)"
            );
        }
    }

    #[test]
    fn test_reconstruct_non_negative() {
        let chords = test_chords();
        let tomo = PlasmaTomography::new(&chords, (1.0, 9.0), (-5.0, 5.0), 10);
        let signals = vec![1.0; 8];
        let recon = tomo.reconstruct(&signals);
        assert_eq!(recon.len(), 100);
        for &v in &recon {
            assert!(v >= 0.0, "Reconstruction has negative value: {v}");
        }
    }

    #[test]
    fn test_reconstruct_2d_shape() {
        let chords = test_chords();
        let tomo = PlasmaTomography::new(&chords, (1.0, 9.0), (-5.0, 5.0), 10);
        let signals = vec![1.0; 8];
        let recon = tomo.reconstruct_2d(&signals);
        assert_eq!(recon.shape(), &[10, 10]);
    }

    #[test]
    fn test_zero_signal_zero_reconstruction() {
        let chords = test_chords();
        let tomo = PlasmaTomography::new(&chords, (1.0, 9.0), (-5.0, 5.0), 10);
        let signals = vec![0.0; 8];
        let recon = tomo.reconstruct(&signals);
        let total: f64 = recon.iter().sum();
        assert!(
            total.abs() < 1e-10,
            "Zero signals should give zero reconstruction: {total}"
        );
    }

    #[test]
    fn test_reconstruct_forward_consistency() {
        // Create a known emissivity, forward-project, then reconstruct
        let chords = test_chords();
        let tomo = PlasmaTomography::new(&chords, (1.0, 9.0), (-5.0, 5.0), 10);

        // Create a simple emissivity: uniform = 1.0
        let emissivity = Array1::from_elem(100, 1.0);
        // Forward project: signals = A · emissivity
        let signals: Vec<f64> = (0..tomo.geometry.nrows())
            .map(|i| tomo.geometry.row(i).dot(&emissivity))
            .collect();

        // Reconstruct
        let recon = tomo.reconstruct(&signals);

        // Forward project reconstruction: should be close to original signals
        let recon_arr = Array1::from_vec(recon);
        for i in 0..signals.len() {
            let recon_signal: f64 = tomo.geometry.row(i).dot(&recon_arr);
            let rel_err = if signals[i].abs() > 1e-10 {
                (recon_signal - signals[i]).abs() / signals[i]
            } else {
                recon_signal.abs()
            };
            assert!(
                rel_err < 0.5,
                "Forward consistency: chord {i} error {rel_err:.3}"
            );
        }
    }
}
