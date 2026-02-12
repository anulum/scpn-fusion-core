//! 2D FFT wrappers around rustfft.
//!
//! Convention matches numpy:
//! - Forward FFT (fft2): unnormalized
//! - Inverse FFT (ifft2): normalized by 1/(nr*nz)

use ndarray::Array2;
use num_complex::Complex64;
use rustfft::FftPlanner;

/// Forward 2D FFT. Matches `numpy.fft.fft2()`.
///
/// numpy does NOT normalize on forward FFT.
pub fn fft2(input: &Array2<f64>) -> Array2<Complex64> {
    let (nrows, ncols) = input.dim();
    let mut planner = FftPlanner::new();

    // Convert to complex
    let mut data = input.mapv(|v| Complex64::new(v, 0.0));

    // FFT along each row (axis 1)
    let fft_row = planner.plan_fft_forward(ncols);
    for mut row in data.rows_mut() {
        let slice = row.as_slice_mut().expect("row must be contiguous");
        fft_row.process(slice);
    }

    // FFT along each column (axis 0)
    // Transpose, FFT rows, transpose back
    let fft_col = planner.plan_fft_forward(nrows);
    let mut transposed = Array2::zeros((ncols, nrows));
    for i in 0..nrows {
        for j in 0..ncols {
            transposed[[j, i]] = data[[i, j]];
        }
    }
    for mut row in transposed.rows_mut() {
        let slice = row.as_slice_mut().expect("row must be contiguous");
        fft_col.process(slice);
    }
    for i in 0..nrows {
        for j in 0..ncols {
            data[[i, j]] = transposed[[j, i]];
        }
    }

    data
}

/// Inverse 2D FFT. Matches `numpy.fft.ifft2()`.
///
/// Applies 1/(nr*nz) normalization.
pub fn ifft2(input: &Array2<Complex64>) -> Array2<f64> {
    let (nrows, ncols) = input.dim();
    let mut planner = FftPlanner::new();
    let norm = 1.0 / (nrows * ncols) as f64;

    let mut data = input.clone();

    // IFFT along each row
    let ifft_row = planner.plan_fft_inverse(ncols);
    for mut row in data.rows_mut() {
        let slice = row.as_slice_mut().expect("row must be contiguous");
        ifft_row.process(slice);
    }

    // IFFT along each column (via transpose)
    let ifft_col = planner.plan_fft_inverse(nrows);
    let mut transposed = Array2::zeros((ncols, nrows));
    for i in 0..nrows {
        for j in 0..ncols {
            transposed[[j, i]] = data[[i, j]];
        }
    }
    for mut row in transposed.rows_mut() {
        let slice = row.as_slice_mut().expect("row must be contiguous");
        ifft_col.process(slice);
    }
    for i in 0..nrows {
        for j in 0..ncols {
            data[[i, j]] = transposed[[j, i]];
        }
    }

    // Normalize and take real part
    data.mapv(|c| c.re * norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft2_roundtrip() {
        let original = Array2::from_shape_fn((16, 16), |(i, j)| (i * 16 + j) as f64);
        let spectrum = fft2(&original);
        let recovered = ifft2(&spectrum);

        for ((i, j), &val) in original.indexed_iter() {
            assert!(
                (recovered[[i, j]] - val).abs() < 1e-10,
                "FFT roundtrip failed at ({i}, {j}): {} vs {val}",
                recovered[[i, j]]
            );
        }
    }

    #[test]
    fn test_fft2_dc_component() {
        // For a constant field, the DC component (0,0) should be N*M*value
        let n = 8;
        let val = 3.0;
        let input = Array2::from_elem((n, n), val);
        let spectrum = fft2(&input);

        let expected_dc = (n * n) as f64 * val;
        assert!(
            (spectrum[[0, 0]].re - expected_dc).abs() < 1e-10,
            "DC component: {} vs {expected_dc}",
            spectrum[[0, 0]].re
        );
        assert!(
            spectrum[[0, 0]].im.abs() < 1e-10,
            "DC imaginary should be zero"
        );
    }

    #[test]
    fn test_fft2_zeros() {
        let input = Array2::zeros((8, 8));
        let spectrum = fft2(&input);
        for &v in spectrum.iter() {
            assert!(v.norm() < 1e-15, "FFT of zeros should be zero");
        }
    }
}
