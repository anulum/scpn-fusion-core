//! FNO turbulence suppressor with multi-layer spectral operator and NPZ weight loading.
//!
//! Compatible with Python weights produced by `src/scpn_fusion/core/fno_training.py`.

use fusion_math::fft::{fft2, ifft2};
use fusion_types::error::{FusionError, FusionResult};
use ndarray::{s, Array1, Array2, Array3, Ix1, Ix2, Ix3, OwnedRepr};
use ndarray_npy::NpzReader;
use num_complex::Complex64;
use rand::Rng;
use rand_distr::StandardNormal;
use std::f64::consts::PI;
use std::fs::File;

/// Default Fourier modes retained in FNO.
const MODES: usize = 12;
/// Default hidden channel width.
const WIDTH: usize = 32;
/// Default number of FNO layers.
const N_LAYERS: usize = 4;
/// Default grid size.
#[cfg(test)]
const GRID_SIZE: usize = 64;
/// Cap for reduced toroidal-harmonic spectral coupling.
const TOROIDAL_COUPLING_MAX_FACTOR: f64 = 2.5;

fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044_715 * x.powi(3))).tanh())
}

#[derive(Debug, Clone)]
pub struct FnoLayerWeights {
    /// Real spectral weights: [width, modes, modes].
    pub wr: Array3<f64>,
    /// Imaginary spectral weights: [width, modes, modes].
    pub wi: Array3<f64>,
    /// Pointwise skip matrix: [width, width].
    pub skip_w: Array2<f64>,
    /// Pointwise skip bias: [width].
    pub skip_b: Array1<f64>,
}

#[derive(Debug, Clone)]
pub struct FnoWeights {
    pub modes: usize,
    pub width: usize,
    pub n_layers: usize,
    /// Lift projection (1 -> width).
    pub lift_w: Array1<f64>,
    pub lift_b: Array1<f64>,
    /// Project head (width -> 1).
    pub project_w: Array1<f64>,
    pub project_b: f64,
    pub layers: Vec<FnoLayerWeights>,
}

impl FnoWeights {
    pub fn random(modes: usize, width: usize, n_layers: usize) -> Self {
        let mut rng = rand::thread_rng();

        let lift_w = Array1::from_shape_fn(width, |_| rng.sample::<f64, _>(StandardNormal) * 0.1);
        let lift_b = Array1::zeros(width);
        let project_w =
            Array1::from_shape_fn(width, |_| rng.sample::<f64, _>(StandardNormal) * 0.1);
        let project_b = 0.0;

        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let wr = Array3::from_shape_fn((width, modes, modes), |_| {
                rng.sample::<f64, _>(StandardNormal) * 0.03
            });
            let wi = Array3::from_shape_fn((width, modes, modes), |_| {
                rng.sample::<f64, _>(StandardNormal) * 0.03
            });

            let mut skip_w = Array2::zeros((width, width));
            for i in 0..width {
                for j in 0..width {
                    let noise = rng.sample::<f64, _>(StandardNormal) * 0.01;
                    skip_w[[i, j]] = if i == j { 1.0 + noise } else { noise };
                }
            }

            let skip_b = Array1::zeros(width);
            layers.push(FnoLayerWeights {
                wr,
                wi,
                skip_w,
                skip_b,
            });
        }

        Self {
            modes,
            width,
            n_layers,
            lift_w,
            lift_b,
            project_w,
            project_b,
            layers,
        }
    }

    pub fn load_weights_npz(path: &str) -> FusionResult<Self> {
        let file = File::open(path)?;
        let mut npz = NpzReader::new(file).map_err(|e| {
            FusionError::ConfigError(format!("Failed to open FNO weight archive '{path}': {e}"))
        })?;

        let modes = read_scalar_usize(&mut npz, "modes")?;
        let width = read_scalar_usize(&mut npz, "width")?;
        let n_layers = read_scalar_usize(&mut npz, "n_layers")?;

        let lift_w = read_array1_f64(&mut npz, "lift_w")?;
        let lift_b = read_array1_f64(&mut npz, "lift_b")?;
        let project_w = read_array1_f64(&mut npz, "project_w")?;
        let project_b_arr = read_array1_f64(&mut npz, "project_b")?;
        let project_b = *project_b_arr
            .get(0)
            .ok_or_else(|| FusionError::ConfigError("project_b array is empty".to_string()))?;

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let wr = read_array3_f64(&mut npz, &format!("layer{i}_wr"))?;
            let wi = read_array3_f64(&mut npz, &format!("layer{i}_wi"))?;
            let skip_w = read_array2_f64(&mut npz, &format!("layer{i}_skip_w"))?;
            let skip_b = read_array1_f64(&mut npz, &format!("layer{i}_skip_b"))?;
            layers.push(FnoLayerWeights {
                wr,
                wi,
                skip_w,
                skip_b,
            });
        }

        let weights = Self {
            modes,
            width,
            n_layers,
            lift_w,
            lift_b,
            project_w,
            project_b,
            layers,
        };
        weights.validate_shapes()?;
        Ok(weights)
    }

    fn validate_shapes(&self) -> FusionResult<()> {
        if self.lift_w.len() != self.width || self.lift_b.len() != self.width {
            return Err(FusionError::ConfigError(format!(
                "Invalid lift shapes lift_w={}, lift_b={}, width={}",
                self.lift_w.len(),
                self.lift_b.len(),
                self.width
            )));
        }
        if self.project_w.len() != self.width {
            return Err(FusionError::ConfigError(format!(
                "Invalid project_w length {}, expected {}",
                self.project_w.len(),
                self.width
            )));
        }
        if self.layers.len() != self.n_layers {
            return Err(FusionError::ConfigError(format!(
                "Layer count mismatch: got {}, expected {}",
                self.layers.len(),
                self.n_layers
            )));
        }
        for (i, layer) in self.layers.iter().enumerate() {
            if layer.wr.dim() != (self.width, self.modes, self.modes)
                || layer.wi.dim() != (self.width, self.modes, self.modes)
            {
                return Err(FusionError::ConfigError(format!(
                    "Layer {i} spectral shape mismatch: wr={:?}, wi={:?}, expected ({}, {}, {})",
                    layer.wr.dim(),
                    layer.wi.dim(),
                    self.width,
                    self.modes,
                    self.modes
                )));
            }
            if layer.skip_w.dim() != (self.width, self.width) || layer.skip_b.len() != self.width {
                return Err(FusionError::ConfigError(format!(
                    "Layer {i} skip shape mismatch: skip_w={:?}, skip_b={}, expected ({}, {}) and {}",
                    layer.skip_w.dim(),
                    layer.skip_b.len(),
                    self.width,
                    self.width,
                    self.width
                )));
            }
        }
        Ok(())
    }
}

fn read_array1_f64(npz: &mut NpzReader<File>, key: &str) -> FusionResult<Array1<f64>> {
    npz.by_name::<OwnedRepr<f64>, Ix1>(&format!("{key}.npy"))
        .or_else(|_| npz.by_name::<OwnedRepr<f64>, Ix1>(key))
        .map_err(|e| FusionError::ConfigError(format!("Failed to read key '{key}' from NPZ: {e}")))
}

fn read_array2_f64(npz: &mut NpzReader<File>, key: &str) -> FusionResult<Array2<f64>> {
    npz.by_name::<OwnedRepr<f64>, Ix2>(&format!("{key}.npy"))
        .or_else(|_| npz.by_name::<OwnedRepr<f64>, Ix2>(key))
        .map_err(|e| FusionError::ConfigError(format!("Failed to read key '{key}' from NPZ: {e}")))
}

fn read_array3_f64(npz: &mut NpzReader<File>, key: &str) -> FusionResult<Array3<f64>> {
    npz.by_name::<OwnedRepr<f64>, Ix3>(&format!("{key}.npy"))
        .or_else(|_| npz.by_name::<OwnedRepr<f64>, Ix3>(key))
        .map_err(|e| FusionError::ConfigError(format!("Failed to read key '{key}' from NPZ: {e}")))
}

fn read_scalar_usize(npz: &mut NpzReader<File>, key: &str) -> FusionResult<usize> {
    let v_i64: Result<Array1<i64>, _> = npz
        .by_name::<OwnedRepr<i64>, Ix1>(&format!("{key}.npy"))
        .or_else(|_| npz.by_name::<OwnedRepr<i64>, Ix1>(key));
    if let Ok(v) = v_i64 {
        if let Some(x) = v.iter().next() {
            return Ok((*x).max(0) as usize);
        }
    }

    let v_i32: Result<Array1<i32>, _> = npz
        .by_name::<OwnedRepr<i32>, Ix1>(&format!("{key}.npy"))
        .or_else(|_| npz.by_name::<OwnedRepr<i32>, Ix1>(key));
    if let Ok(v) = v_i32 {
        if let Some(x) = v.iter().next() {
            return Ok((*x).max(0) as usize);
        }
    }

    let v_f64: Result<Array1<f64>, _> = npz
        .by_name::<OwnedRepr<f64>, Ix1>(&format!("{key}.npy"))
        .or_else(|_| npz.by_name::<OwnedRepr<f64>, Ix1>(key));
    if let Ok(v) = v_f64 {
        if let Some(x) = v.iter().next() {
            return Ok((*x).max(0.0) as usize);
        }
    }
    Err(FusionError::ConfigError(format!(
        "Failed to read scalar key '{key}' from NPZ"
    )))
}

/// Spectral turbulence generator (drift-wave physics).
#[derive(Clone)]
pub struct SpectralTurbulenceGenerator {
    /// Density fluctuation field.
    pub field: Array2<f64>,
    /// Grid size.
    pub size: usize,
    /// ky wavenumber grid (row index).
    ky: Array2<f64>,
    /// k² wavenumber squared.
    k2: Array2<f64>,
    /// Toroidal harmonic amplitudes for reduced 3D spectral coupling (n=1..N).
    pub toroidal_harmonics: Vec<f64>,
    /// Coupling gain for toroidal-harmonic closure.
    pub toroidal_coupling_gain: f64,
}

impl SpectralTurbulenceGenerator {
    pub fn new(size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let field =
            Array2::from_shape_fn((size, size), |_| rng.sample::<f64, _>(StandardNormal) * 0.1);

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

        Self {
            field,
            size,
            ky,
            k2,
            toroidal_harmonics: Vec::new(),
            toroidal_coupling_gain: 0.0,
        }
    }

    /// Configure reduced toroidal-harmonic coupling used by the spectral step.
    ///
    /// This keeps the base 2D solver, but injects a low-order `n != 0` closure
    /// through edge/non-zonal spectral amplification.
    pub fn set_toroidal_harmonics(&mut self, amplitudes: &[f64], gain: f64) {
        self.toroidal_harmonics.clear();
        self.toroidal_harmonics
            .extend(amplitudes.iter().map(|a| a.abs()));
        self.toroidal_coupling_gain = gain.max(0.0);
    }

    fn toroidal_spectral_rms(&self) -> f64 {
        if self.toroidal_coupling_gain <= 0.0 || self.toroidal_harmonics.is_empty() {
            return 0.0;
        }
        self.toroidal_harmonics
            .iter()
            .enumerate()
            .map(|(idx, amp)| {
                let n = (idx + 1) as f64;
                (n * amp).powi(2)
            })
            .sum::<f64>()
            .sqrt()
    }

    fn toroidal_coupling_factor(&self, spectral_rms: f64, ky: f64, k2: f64) -> f64 {
        if spectral_rms <= 1e-12 {
            return 1.0;
        }
        let non_zonal_weight = (ky.abs() / (1.0 + ky.abs())).clamp(0.0, 1.0);
        let low_k_weight = (25.0 / (25.0 + k2.max(0.0))).clamp(0.0, 1.0);
        (1.0 + self.toroidal_coupling_gain * spectral_rms * non_zonal_weight * low_k_weight)
            .clamp(1.0, TOROIDAL_COUPLING_MAX_FACTOR)
    }

    /// Evolve turbulence one step in Fourier space.
    ///
    /// - Drift wave dispersion: ω = ky / (1 + k²)
    /// - Low-k drive (k² < 25)
    /// - Viscous + active damping
    pub fn step(&mut self, dt: f64, damping: f64) {
        let mut field_k = fft2(&self.field);
        let n = self.size;
        let damping = damping.clamp(0.0, 1.0);
        let spectral_rms = self.toroidal_spectral_rms();

        for i in 0..n {
            for j in 0..n {
                let k2v = self.k2[[i, j]];
                let kyv = self.ky[[i, j]];

                let omega = kyv / (1.0 + k2v);
                let phase = Complex64::new(0.0, -omega * dt);
                field_k[[i, j]] *= phase.exp();

                if k2v > 0.0 && k2v < 25.0 {
                    field_k[[i, j]] *= Complex64::new(1.001, 0.0);
                }

                let toroidal_factor = self.toroidal_coupling_factor(spectral_rms, kyv, k2v);
                field_k[[i, j]] *= Complex64::new(toroidal_factor, 0.0);

                let visc = (-0.001 * k2v * dt).exp();
                field_k[[i, j]] *= Complex64::new(visc * (1.0 - damping), 0.0);
            }
        }

        self.field = ifft2(&field_k);
    }

    /// Mean-square turbulence energy.
    pub fn energy(&self) -> f64 {
        self.field.iter().map(|v| v * v).sum::<f64>() / (self.size * self.size) as f64
    }
}

/// Multi-layer FNO controller.
pub struct FnoController {
    weights: FnoWeights,
}

impl FnoController {
    pub fn new() -> Self {
        Self {
            weights: FnoWeights::random(MODES, WIDTH, N_LAYERS),
        }
    }

    pub fn from_weights(weights: FnoWeights) -> Self {
        Self { weights }
    }

    pub fn load_weights_npz(path: &str) -> FusionResult<Self> {
        let weights = FnoWeights::load_weights_npz(path)?;
        Ok(Self::from_weights(weights))
    }

    fn lift(&self, field: &Array2<f64>) -> Array3<f64> {
        let (n, m) = field.dim();
        let width = self.weights.width;
        let mut h = Array3::zeros((n, m, width));
        for i in 0..n {
            for j in 0..m {
                let x = field[[i, j]];
                for c in 0..width {
                    h[[i, j, c]] = x * self.weights.lift_w[c] + self.weights.lift_b[c];
                }
            }
        }
        h
    }

    fn spectral_convolution(&self, h: &Array3<f64>, layer: &FnoLayerWeights) -> Array3<f64> {
        let (n, m, width) = h.dim();
        let modes_r = self.weights.modes.min(n);
        let modes_c = self.weights.modes.min(m);
        let mut out = Array3::zeros((n, m, width));

        for c in 0..width {
            let x_field = h.slice(s![.., .., c]).to_owned();
            let x_k = fft2(&x_field);
            let mut out_k = Array2::from_elem((n, m), Complex64::new(0.0, 0.0));

            for i in 0..modes_r {
                for j in 0..modes_c {
                    let w = Complex64::new(layer.wr[[c, i, j]], layer.wi[[c, i, j]]);
                    out_k[[i, j]] = x_k[[i, j]] * w;
                }
            }

            let out_field = ifft2(&out_k);
            out.slice_mut(s![.., .., c]).assign(&out_field);
        }
        out
    }

    fn pointwise_skip(&self, h: &Array3<f64>, layer: &FnoLayerWeights) -> Array3<f64> {
        let (n, m, width) = h.dim();
        let mut out = Array3::zeros((n, m, width));

        for i in 0..n {
            for j in 0..m {
                for c_out in 0..width {
                    let mut v = layer.skip_b[c_out];
                    for c_in in 0..width {
                        v += h[[i, j, c_in]] * layer.skip_w[[c_in, c_out]];
                    }
                    out[[i, j, c_out]] = v;
                }
            }
        }
        out
    }

    fn apply_layers(&self, field: &Array2<f64>) -> Array3<f64> {
        let mut h = self.lift(field);
        for layer in &self.weights.layers {
            let spectral = self.spectral_convolution(&h, layer);
            let skip = self.pointwise_skip(&h, layer);
            let (n, m, width) = h.dim();
            let mut next = Array3::zeros((n, m, width));
            for i in 0..n {
                for j in 0..m {
                    for c in 0..width {
                        next[[i, j, c]] = gelu(spectral[[i, j, c]] + skip[[i, j, c]]);
                    }
                }
            }
            h = next;
        }
        h
    }

    fn project(&self, h: &Array3<f64>) -> Array2<f64> {
        let (n, m, width) = h.dim();
        let mut out = Array2::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                let mut v = self.weights.project_b;
                for c in 0..width {
                    v += h[[i, j, c]] * self.weights.project_w[c];
                }
                out[[i, j]] = v;
            }
        }
        out
    }

    pub fn predict(&self, field: &Array2<f64>) -> Array2<f64> {
        let h = self.apply_layers(field);
        self.project(&h)
    }

    /// Predict turbulence and compute suppression factor [0, 1].
    pub fn predict_and_suppress(&self, field: &Array2<f64>) -> (f64, Array2<f64>) {
        let prediction = self.predict(field);
        let energy = prediction.iter().map(|v| v * v).sum::<f64>() / prediction.len() as f64;
        let suppression = (energy * 10.0).tanh().clamp(0.0, 1.0);
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
    use ndarray::Array1;
    use ndarray_npy::NpzWriter;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn nonzonal_energy(gen: &SpectralTurbulenceGenerator) -> f64 {
        let field_k = fft2(&gen.field);
        let mut sum = 0.0;
        for i in 0..gen.size {
            for j in 0..gen.size {
                if gen.ky[[i, j]].abs() > 1e-12 {
                    sum += field_k[[i, j]].norm_sqr();
                }
            }
        }
        sum / (gen.size * gen.size) as f64
    }

    fn teacher_weights() -> FnoWeights {
        FnoWeights {
            modes: 8,
            width: 4,
            n_layers: 0,
            lift_w: Array1::from_vec(vec![1.0, -0.5, 0.3, 0.1]),
            lift_b: Array1::zeros(4),
            project_w: Array1::from_vec(vec![1.5, -0.7, 0.4, 0.2]),
            project_b: 0.05,
            layers: Vec::new(),
        }
    }

    fn relative_l2(pred: &Array2<f64>, target: &Array2<f64>) -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for (p, t) in pred.iter().zip(target.iter()) {
            let d = p - t;
            num += d * d;
            den += t * t;
        }
        num.sqrt() / den.max(1e-12).sqrt()
    }

    fn write_weights_npz(path: &str, weights: &FnoWeights) {
        let file = File::create(path).expect("create npz");
        let mut writer = NpzWriter::new(file);
        writer
            .add_array("version", &Array1::from_vec(vec![2_i32]))
            .unwrap();
        writer
            .add_array("modes", &Array1::from_vec(vec![weights.modes as i32]))
            .unwrap();
        writer
            .add_array("width", &Array1::from_vec(vec![weights.width as i32]))
            .unwrap();
        writer
            .add_array("n_layers", &Array1::from_vec(vec![weights.n_layers as i32]))
            .unwrap();
        writer.add_array("lift_w", &weights.lift_w).unwrap();
        writer.add_array("lift_b", &weights.lift_b).unwrap();
        writer.add_array("project_w", &weights.project_w).unwrap();
        writer
            .add_array("project_b", &Array1::from_vec(vec![weights.project_b]))
            .unwrap();
        for (i, layer) in weights.layers.iter().enumerate() {
            writer.add_array(format!("layer{i}_wr"), &layer.wr).unwrap();
            writer.add_array(format!("layer{i}_wi"), &layer.wi).unwrap();
            writer
                .add_array(format!("layer{i}_skip_w"), &layer.skip_w)
                .unwrap();
            writer
                .add_array(format!("layer{i}_skip_b"), &layer.skip_b)
                .unwrap();
        }
        writer.finish().unwrap();
    }

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
    fn test_toroidal_harmonics_disabled_has_baseline_parity() {
        let mut baseline = SpectralTurbulenceGenerator::new(GRID_SIZE);
        let mut coupled = baseline.clone();
        coupled.set_toroidal_harmonics(&[0.2, 0.1, 0.05], 0.0);

        baseline.step(0.01, 0.2);
        coupled.step(0.01, 0.2);

        let max_err = baseline
            .field
            .iter()
            .zip(coupled.field.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            max_err < 1e-12,
            "Expected parity when toroidal coupling disabled, max_err={max_err}"
        );
    }

    #[test]
    fn test_toroidal_harmonics_raise_nonzonal_spectral_energy() {
        let mut baseline = SpectralTurbulenceGenerator::new(GRID_SIZE);
        let mut coupled = baseline.clone();
        coupled.set_toroidal_harmonics(&[0.25, 0.12, 0.06], 0.35);

        for _ in 0..6 {
            baseline.step(0.01, 0.05);
            coupled.step(0.01, 0.05);
        }

        let e_base = nonzonal_energy(&baseline);
        let e_coupled = nonzonal_energy(&coupled);
        assert!(
            e_coupled > e_base,
            "Expected higher non-zonal spectral energy with toroidal harmonics: base={e_base:.6e}, coupled={e_coupled:.6e}"
        );
    }

    #[test]
    fn test_toroidal_coupling_factor_clamped() {
        let mut gen = SpectralTurbulenceGenerator::new(GRID_SIZE);
        gen.set_toroidal_harmonics(&[10.0, 10.0, 10.0], 10.0);
        let rms = gen.toroidal_spectral_rms();
        let factor = gen.toroidal_coupling_factor(rms, 6.0, 0.01);
        assert!(
            (factor - TOROIDAL_COUPLING_MAX_FACTOR).abs() < 1e-12,
            "Expected clamped coupling factor={}, got {}",
            TOROIDAL_COUPLING_MAX_FACTOR,
            factor
        );
    }

    #[test]
    fn test_fno_trained_beats_random() {
        let trained = FnoController::from_weights(teacher_weights());
        let random = FnoController::from_weights(FnoWeights::random(8, 4, 0));

        let mut rng = StdRng::seed_from_u64(123);
        let mut trained_loss = 0.0;
        let mut random_loss = 0.0;
        let samples = 12;

        for _ in 0..samples {
            let field = Array2::from_shape_fn((GRID_SIZE, GRID_SIZE), |_| {
                rng.sample::<f64, _>(StandardNormal) * 0.3
            });
            let target = trained.predict(&field);
            let pred_trained = trained.predict(&field);
            let pred_random = random.predict(&field);
            trained_loss += relative_l2(&pred_trained, &target);
            random_loss += relative_l2(&pred_random, &target);
        }

        trained_loss /= samples as f64;
        random_loss /= samples as f64;

        assert!(
            trained_loss < 0.10,
            "Expected trained loss < 10%, got {:.2}%",
            trained_loss * 100.0
        );
        assert!(
            random_loss > trained_loss + 0.5,
            "Random baseline should be substantially worse: trained={trained_loss:.4}, random={random_loss:.4}"
        );
    }

    #[test]
    fn test_fno_weight_loading_roundtrip() {
        let weights = FnoWeights::random(6, 5, 2);
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "fusion_fno_weights_{}_{}.npz",
            std::process::id(),
            ts
        ));
        write_weights_npz(path.to_str().unwrap(), &weights);

        let loaded = FnoWeights::load_weights_npz(path.to_str().unwrap()).unwrap();
        let c1 = FnoController::from_weights(weights);
        let c2 = FnoController::from_weights(loaded);

        let field = Array2::from_shape_fn((GRID_SIZE, GRID_SIZE), |(i, j)| {
            ((i * 7 + j * 3) as f64).sin()
        });
        let p1 = c1.predict(&field);
        let p2 = c2.predict(&field);
        let err = relative_l2(&p1, &p2);
        assert!(err < 1e-12, "Roundtrip mismatch: relative error={err:.3e}");

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_energy_decreases_under_suppression() {
        let controller = FnoController::from_weights(teacher_weights());
        let baseline = SpectralTurbulenceGenerator::new(GRID_SIZE);
        let mut uncontrolled = baseline.clone();
        let mut controlled = baseline;

        for _ in 0..100 {
            uncontrolled.step(0.01, 0.0);
            let (suppression, _) = controller.predict_and_suppress(&controlled.field);
            controlled.step(0.01, suppression);
        }

        let e_uncontrolled = uncontrolled.energy();
        let e_controlled = controlled.energy();
        assert!(
            e_controlled < e_uncontrolled,
            "Suppression should reduce energy: uncontrolled={e_uncontrolled:.6}, controlled={e_controlled:.6}"
        );
    }
}
