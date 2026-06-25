// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Disruption
//! Transformer-based disruption predictor.
//!
//! Port of `disruption_predictor.py`.
//! Tearing mode simulator (modified Rutherford) + tiny Transformer classifier.

use ndarray::{Array1, Array2, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

/// Time step for tearing mode simulation. Python: 0.01.
const DT: f64 = 0.01;

/// Initial island width. Python: 0.01.
const W_INIT: f64 = 0.01;

/// Stable delta-prime. Python: -0.5.
const DELTA_PRIME_STABLE: f64 = -0.5;

/// Unstable delta-prime (post trigger). Python: -0.1.
const DELTA_PRIME_UNSTABLE: f64 = -0.1;

/// Saturation width. Python: 12.0.
const W_SAT: f64 = 12.0;

/// Disruption threshold. Python: 8.0.
const W_DISRUPTION: f64 = 8.0;

/// Island-width Gaussian process-noise σ. Python: 0.02.
const NOISE_STD: f64 = 0.02;

/// Default poloidal beta scaling the bootstrap drive. Python: 0.8.
pub const DEFAULT_BETA_P: f64 = 0.8;

/// Default critical island width regularising the bootstrap term. Python: 0.05.
pub const DEFAULT_W_CRIT: f64 = 0.05;

/// Island width seeded post-trigger when below 0.1. Python: 0.15.
const SEED_ISLAND_WIDTH: f64 = 0.15;

/// Trigger-time sentinel for a non-disruptive shot. Python: 9999.
const SAFE_SENTINEL: usize = 9999;

/// Fixed sequence length for transformer input. Python: 100.
const SEQ_LEN: usize = 100;

/// Embedding dimension. Python: 32.
const D_MODEL: usize = 32;

/// Number of attention heads. Python: 4.
const N_HEADS: usize = 4;

/// Feedforward hidden dimension. Python: 64.
const DIM_FF: usize = 64;

/// Number of transformer layers. Python: 2.
const N_LAYERS: usize = 2;

/// Result of tearing mode simulation.
pub struct TearingModeShot {
    /// Island width history.
    pub signal: Vec<f64>,
    /// 1 = disruptive, 0 = safe.
    pub label: u8,
    /// Steps until disruption, or -1 if safe.
    pub time_to_disruption: i64,
}

/// Island-width floor. Python: 0.001.
const W_FLOOR: f64 = 0.001;

/// Deterministic Modified Rutherford island-width increment for one step.
///
/// Canonical contract shared with the NumPy tier
/// (`scpn_fusion.control.disruption_risk_runtime.rutherford_island_growth`):
/// `dw = (delta_prime + beta_p·w/(w² + w_crit²))·(1 - w/w_sat)·dt`, bit-exact
/// across both backends. The bootstrap-current drive `beta_p·w/(w² + w_crit²)`
/// is what makes the island grow unstably; omitting it is not the Modified
/// Rutherford Equation.
pub fn rutherford_island_growth(
    w: f64,
    delta_prime: f64,
    beta_p: f64,
    w_crit: f64,
    dt: f64,
) -> f64 {
    let f_bs = beta_p * (w / (w * w + w_crit * w_crit));
    (delta_prime + f_bs) * (1.0 - w / W_SAT) * dt
}

/// Simulate a single plasma shot with tearing-mode physics.
///
/// `seed` makes the stochastic trajectory reproducible within this backend;
/// `beta_p`/`w_crit` parametrise the bootstrap drive. The deterministic per-step
/// physics matches the NumPy tier bit-for-bit (see [`rutherford_island_growth`]);
/// the trajectory is statistically equivalent (independent RNG streams).
pub fn simulate_tearing_mode(
    steps: usize,
    seed: Option<u64>,
    beta_p: f64,
    w_crit: f64,
) -> TearingModeShot {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let noise = Normal::new(0.0, NOISE_STD).expect("NOISE_STD is a valid Gaussian sigma");

    let mut w = W_INIT;
    let is_disruptive = rng.gen::<f64>() > 0.5;
    let trigger_time = if is_disruptive {
        rng.gen_range(200..800)
    } else {
        SAFE_SENTINEL
    };

    let mut delta_prime = DELTA_PRIME_STABLE;
    let mut signal = Vec::with_capacity(steps);

    for t in 0..steps {
        if t > trigger_time {
            delta_prime = DELTA_PRIME_UNSTABLE;
            if w < 0.1 {
                w = SEED_ISLAND_WIDTH; // seed island
            }
        }

        w += rutherford_island_growth(w, delta_prime, beta_p, w_crit, DT);
        w += noise.sample(&mut rng);
        w = w.max(W_FLOOR);

        signal.push(w);

        // Mode lock → disruption
        if w > W_DISRUPTION {
            return TearingModeShot {
                signal,
                label: 1,
                time_to_disruption: (t as i64 - trigger_time as i64),
            };
        }
    }

    // Matches the NumPy tier: the label is 1 only if the island actually crossed
    // the disruption threshold (handled in-loop above); a shot that merely had a
    // tearing trigger but never disrupted within `steps` is labelled safe.
    TearingModeShot {
        signal,
        label: 0,
        time_to_disruption: -1,
    }
}

/// Pad or truncate signal to fixed length.
pub fn normalize_sequence(signal: &[f64], target_len: usize) -> Vec<f64> {
    let mut out = vec![0.0; target_len];
    let copy_len = signal.len().min(target_len);
    out[..copy_len].copy_from_slice(&signal[..copy_len]);
    out
}

/// Reduced toroidal asymmetry observables for disruption-risk coupling.
#[derive(Debug, Clone, Copy)]
pub struct ToroidalAsymmetryObservables {
    pub n1_amp: f64,
    pub n2_amp: f64,
    pub n3_amp: f64,
    pub asymmetry_index: f64,
    pub radial_spread: f64,
}

impl Default for ToroidalAsymmetryObservables {
    fn default() -> Self {
        Self {
            n1_amp: 0.0,
            n2_amp: 0.0,
            n3_amp: 0.0,
            asymmetry_index: 0.0,
            radial_spread: 0.0,
        }
    }
}

/// Build compact disruption features and append toroidal asymmetry indicators.
pub fn build_disruption_feature_vector(
    signal: &[f64],
    toroidal: Option<ToroidalAsymmetryObservables>,
) -> Vec<f64> {
    if signal.is_empty() {
        return vec![0.0; 11];
    }

    let n = signal.len() as f64;
    let mean = signal.iter().sum::<f64>() / n;
    let var = signal.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    let max_val = signal
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(0.0);
    let slope = if signal.len() > 1 {
        (signal[signal.len() - 1] - signal[0]) / (signal.len() - 1) as f64
    } else {
        0.0
    };
    let energy = signal.iter().map(|v| v * v).sum::<f64>() / n;
    let last = *signal.last().unwrap_or(&0.0);

    let obs = toroidal.unwrap_or_default();
    vec![
        mean,
        std,
        max_val,
        slope,
        energy,
        last,
        obs.n1_amp,
        obs.n2_amp,
        obs.n3_amp,
        obs.asymmetry_index,
        obs.radial_spread,
    ]
}

fn logistic(v: f64) -> f64 {
    1.0 / (1.0 + (-v).exp())
}

/// Hybrid supervised/unsupervised anomaly detector for early alarms.
#[derive(Debug, Clone, Copy)]
pub struct HybridAnomalyDetector {
    /// Alarm threshold in [0, 1].
    pub threshold: f64,
    /// EMA factor for online moments.
    pub ema: f64,
    mean: f64,
    var: f64,
    initialized: bool,
}

impl HybridAnomalyDetector {
    pub fn new(threshold: f64, ema: f64) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
            ema: ema.clamp(1e-4, 1.0),
            mean: 0.0,
            var: 1.0,
            initialized: false,
        }
    }

    /// Returns `(supervised_score, unsupervised_score, anomaly_score, alarm)`.
    pub fn score(
        &mut self,
        signal: &[f64],
        toroidal: Option<ToroidalAsymmetryObservables>,
    ) -> (f64, f64, f64, bool) {
        let features = build_disruption_feature_vector(signal, toroidal);
        // Linear disruption-risk logit; weights match the NumPy tier's
        // DISRUPTION_RISK_LINEAR_WEIGHTS (disruption_risk_runtime.py), grouped as
        // thermal + toroidal-asymmetry + state terms. features = [mean, std,
        // max_val, slope, energy, last, n1, n2, n3, asym, spread].
        let thermal =
            0.03 * features[2] + 0.55 * features[1] + 0.005 * features[4] + 0.50 * features[3];
        let asymmetry = 1.10 * features[6]
            + 0.70 * features[7]
            + 0.45 * features[8]
            + 0.50 * features[9]
            + 0.15 * features[10];
        let state = 0.02 * features[0] + 0.02 * features[5];
        let supervised = logistic(-4.0 + thermal + asymmetry + state);

        let unsupervised = if self.initialized {
            let z = (supervised - self.mean).abs() / (self.var + 1e-9).sqrt();
            1.0 - (-0.5 * z).exp()
        } else {
            self.initialized = true;
            0.0
        };

        let alpha = self.ema;
        let delta = supervised - self.mean;
        self.mean += alpha * delta;
        self.var = ((1.0 - alpha) * self.var + alpha * delta * delta).max(1e-9);

        let anomaly_score = (0.7 * supervised + 0.3 * unsupervised).clamp(0.0, 1.0);
        let alarm = anomaly_score >= self.threshold;
        (supervised, unsupervised, anomaly_score, alarm)
    }
}

impl Default for HybridAnomalyDetector {
    fn default() -> Self {
        Self::new(0.65, 0.05)
    }
}

// --- Tiny Transformer Implementation ---

/// Layer normalization over last dimension.
fn layer_norm(x: &Array1<f64>) -> Array1<f64> {
    let mean = x.mean().unwrap_or(0.0);
    let var: f64 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64;
    let std = (var + 1e-5).sqrt();
    x.mapv(|v| (v - mean) / std)
}

/// Scaled dot-product attention: softmax(Q K^T / sqrt(d)) V.
/// Q, K, V: (seq_len, d_head).
fn scaled_attention(q: &Array2<f64>, k: &Array2<f64>, v: &Array2<f64>) -> Array2<f64> {
    let d = q.ncols() as f64;
    let scores = q.dot(&k.t()) / d.sqrt(); // (seq, seq)

    // Softmax per row
    let mut attn = Array2::zeros(scores.raw_dim());
    for i in 0..scores.nrows() {
        let row = scores.row(i);
        let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_row: Array1<f64> = row.mapv(|v| (v - max_val).exp());
        let sum: f64 = exp_row.sum();
        attn.row_mut(i).assign(&(&exp_row / sum));
    }

    attn.dot(v) // (seq, d_head)
}

/// Multi-head self-attention layer.
pub struct MultiHeadAttention {
    /// Projection weights: (d_model, d_model) for Q, K, V.
    wq: Array2<f64>,
    wk: Array2<f64>,
    wv: Array2<f64>,
    wo: Array2<f64>,
    n_heads: usize,
    d_head: usize,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        let d_head = d_model / n_heads;
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (d_model + d_model) as f64).sqrt();
        let rand_init = |rng: &mut rand::rngs::ThreadRng| {
            Array2::from_shape_fn((d_model, d_model), |_| {
                (rng.gen::<f64>() - 0.5) * 2.0 * scale
            })
        };

        MultiHeadAttention {
            wq: rand_init(&mut rng),
            wk: rand_init(&mut rng),
            wv: rand_init(&mut rng),
            wo: rand_init(&mut rng),
            n_heads,
            d_head,
        }
    }

    /// Forward: x (seq_len, d_model) → (seq_len, d_model).
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let seq = x.nrows();
        let q_all = x.dot(&self.wq); // (seq, d_model)
        let k_all = x.dot(&self.wk);
        let v_all = x.dot(&self.wv);

        let mut heads_out = Array2::zeros((seq, 0));

        for h in 0..self.n_heads {
            let start = h * self.d_head;
            let end = start + self.d_head;
            let q = q_all.slice(ndarray::s![.., start..end]).to_owned();
            let k = k_all.slice(ndarray::s![.., start..end]).to_owned();
            let v = v_all.slice(ndarray::s![.., start..end]).to_owned();
            let head = scaled_attention(&q, &k, &v);
            heads_out = ndarray::concatenate![Axis(1), heads_out, head];
        }

        heads_out.dot(&self.wo)
    }
}

/// Feedforward sublayer: linear → relu → linear.
pub struct FeedForward {
    w1: Array2<f64>,
    b1: Array1<f64>,
    w2: Array2<f64>,
    b2: Array1<f64>,
}

impl FeedForward {
    pub fn new(d_model: usize, dim_ff: usize) -> Self {
        let mut rng = rand::thread_rng();
        let s1 = (2.0 / (d_model + dim_ff) as f64).sqrt();
        let s2 = (2.0 / (dim_ff + d_model) as f64).sqrt();

        FeedForward {
            w1: Array2::from_shape_fn((d_model, dim_ff), |_| (rng.gen::<f64>() - 0.5) * 2.0 * s1),
            b1: Array1::zeros(dim_ff),
            w2: Array2::from_shape_fn((dim_ff, d_model), |_| (rng.gen::<f64>() - 0.5) * 2.0 * s2),
            b2: Array1::zeros(d_model),
        }
    }

    /// Forward: (seq, d_model) → (seq, d_model).
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let h = (x.dot(&self.w1) + &self.b1).mapv(|v| v.max(0.0)); // ReLU
        h.dot(&self.w2) + &self.b2
    }
}

/// Single Transformer encoder layer.
pub struct TransformerLayer {
    attn: MultiHeadAttention,
    ff: FeedForward,
}

impl TransformerLayer {
    pub fn new(d_model: usize, n_heads: usize, dim_ff: usize) -> Self {
        TransformerLayer {
            attn: MultiHeadAttention::new(d_model, n_heads),
            ff: FeedForward::new(d_model, dim_ff),
        }
    }

    /// Forward with residual connections and layer norm.
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        // Self-attention + residual + layer norm
        let attn_out = self.attn.forward(x);
        let mut post_attn = x + &attn_out;
        for mut row in post_attn.rows_mut() {
            let normed = layer_norm(&row.to_owned());
            row.assign(&normed);
        }

        // Feedforward + residual + layer norm
        let ff_out = self.ff.forward(&post_attn);
        let mut post_ff = &post_attn + &ff_out;
        for mut row in post_ff.rows_mut() {
            let normed = layer_norm(&row.to_owned());
            row.assign(&normed);
        }

        post_ff
    }
}

/// Disruption prediction transformer.
pub struct DisruptionTransformer {
    /// Input embedding: Linear(1, d_model).
    w_embed: Array2<f64>,
    b_embed: Array1<f64>,
    /// Learnable positional encoding: (seq_len, d_model).
    pos_encoding: Array2<f64>,
    /// Transformer encoder layers.
    layers: Vec<TransformerLayer>,
    /// Classifier: Linear(d_model, 1).
    w_class: Array1<f64>,
    b_class: f64,
}

impl DisruptionTransformer {
    /// Create with random initialization.
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let s_embed = (2.0 / (1 + D_MODEL) as f64).sqrt();

        let layers = (0..N_LAYERS)
            .map(|_| TransformerLayer::new(D_MODEL, N_HEADS, DIM_FF))
            .collect();

        DisruptionTransformer {
            w_embed: Array2::from_shape_fn((1, D_MODEL), |_| {
                (rng.gen::<f64>() - 0.5) * 2.0 * s_embed
            }),
            b_embed: Array1::zeros(D_MODEL),
            pos_encoding: Array2::from_shape_fn((SEQ_LEN, D_MODEL), |_| rng.gen::<f64>() * 0.1),
            layers,
            w_class: Array1::from_shape_fn(D_MODEL, |_| (rng.gen::<f64>() - 0.5) * 0.1),
            b_class: 0.0,
        }
    }

    /// Forward pass: signal (seq_len,) → disruption probability [0, 1].
    pub fn forward(&self, signal: &[f64]) -> f64 {
        let seq = normalize_sequence(signal, SEQ_LEN);
        let seq_arr = Array2::from_shape_fn((SEQ_LEN, 1), |(i, _)| seq[i]);

        // Embed: (seq, 1) × (1, d_model) → (seq, d_model)
        let embedded = seq_arr.dot(&self.w_embed) + &self.b_embed;
        let mut x = &embedded + &self.pos_encoding;

        // Transformer layers
        for layer in &self.layers {
            x = layer.forward(&x);
        }

        // Take last time step, classify
        let last = x.row(SEQ_LEN - 1);
        let logit = last.dot(&self.w_class) + self.b_class;
        1.0 / (1.0 + (-logit).exp()) // sigmoid
    }

    /// Forward pass augmented with toroidal asymmetry observables.
    pub fn forward_with_observables(
        &self,
        signal: &[f64],
        toroidal: Option<ToroidalAsymmetryObservables>,
    ) -> f64 {
        let base_prob = self.forward(signal);
        let features = build_disruption_feature_vector(signal, toroidal);

        let asym_logit = -4.0
            + 0.55 * features[2]
            + 0.35 * features[1]
            + 0.10 * features[4]
            + 0.25 * features[3]
            + 1.10 * features[6]
            + 0.70 * features[7]
            + 0.45 * features[8]
            + 0.50 * features[9]
            + 0.15 * features[10]
            + 0.15 * features[0]
            + 0.20 * features[5];

        let asym_prob = logistic(asym_logit);
        (0.75 * base_prob + 0.25 * asym_prob).clamp(0.0, 1.0)
    }
}

impl Default for DisruptionTransformer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_simulate_produces_signal() {
        let shot = simulate_tearing_mode(1000, None, DEFAULT_BETA_P, DEFAULT_W_CRIT);
        assert!(!shot.signal.is_empty());
        assert!(shot.label == 0 || shot.label == 1);
    }

    #[test]
    fn test_simulate_island_grows_and_rarely_disrupts() {
        // The Modified Rutherford drive balances delta_prime near w ~ 8, so the
        // island grows above its seed width but rarely crosses the disruption
        // threshold within 1000 steps — matching the NumPy tier (label is 1 only
        // on an actual w > 8 crossing, which is rare for these parameters).
        let mut n_disruptive = 0;
        let mut grew = 0;
        for _ in 0..100 {
            let shot = simulate_tearing_mode(1000, None, DEFAULT_BETA_P, DEFAULT_W_CRIT);
            assert!(shot.label == 0 || shot.label == 1);
            assert!(shot.signal.iter().all(|w| w.is_finite()));
            if shot.label == 1 {
                n_disruptive += 1;
            }
            let max_w = shot.signal.iter().copied().fold(0.0_f64, f64::max);
            if max_w > 5.0 * W_INIT {
                grew += 1;
            }
        }
        assert!(
            n_disruptive < 20,
            "saturating drive should rarely disrupt: {n_disruptive}/100"
        );
        assert!(
            grew > 50,
            "the bootstrap drive should grow the island: {grew}/100"
        );
    }

    #[test]
    fn test_simulate_is_reproducible_with_seed() {
        let a = simulate_tearing_mode(500, Some(2026), DEFAULT_BETA_P, DEFAULT_W_CRIT);
        let b = simulate_tearing_mode(500, Some(2026), DEFAULT_BETA_P, DEFAULT_W_CRIT);
        assert_eq!(a.signal, b.signal);
        assert_eq!(a.label, b.label);
        assert_eq!(a.time_to_disruption, b.time_to_disruption);
    }

    #[test]
    fn test_rutherford_island_growth_includes_bootstrap_drive() {
        // The bootstrap term beta_p·w/(w²+w_crit²) adds to delta_prime, so a
        // positive beta_p raises dw above the bare-delta_prime increment.
        let w = 0.2;
        let delta_prime = -0.1;
        let bare = delta_prime * (1.0 - w / W_SAT) * DT;
        let with_bootstrap = rutherford_island_growth(w, delta_prime, 0.8, 0.05, DT);
        assert!(with_bootstrap > bare);
        // Zero bootstrap recovers the bare delta_prime increment exactly.
        assert_eq!(
            rutherford_island_growth(w, delta_prime, 0.0, 0.05, DT),
            bare
        );
    }

    #[test]
    fn test_normalize_sequence_padding() {
        let short = vec![1.0, 2.0, 3.0];
        let padded = normalize_sequence(&short, 5);
        assert_eq!(padded.len(), 5);
        assert_eq!(padded[0], 1.0);
        assert_eq!(padded[3], 0.0);
    }

    #[test]
    fn test_normalize_sequence_truncation() {
        let long = vec![1.0; 200];
        let truncated = normalize_sequence(&long, SEQ_LEN);
        assert_eq!(truncated.len(), SEQ_LEN);
    }

    #[test]
    fn test_transformer_output_bounded() {
        let model = DisruptionTransformer::new();
        let signal = vec![0.5; 100];
        let prob = model.forward(&signal);
        assert!(
            (0.0..=1.0).contains(&prob),
            "Probability should be in [0,1]: {prob}"
        );
    }

    #[test]
    fn test_transformer_handles_short_signal() {
        let model = DisruptionTransformer::new();
        let signal = vec![1.0; 10]; // shorter than SEQ_LEN
        let prob = model.forward(&signal);
        assert!(prob.is_finite(), "Should handle short signal: {prob}");
    }

    #[test]
    fn test_attention_preserves_shape() {
        let q = Array2::from_elem((10, 8), 0.1);
        let k = Array2::from_elem((10, 8), 0.1);
        let v = Array2::from_elem((10, 8), 0.1);
        let out = scaled_attention(&q, &k, &v);
        assert_eq!(out.dim(), (10, 8));
    }

    #[test]
    fn test_feature_vector_carries_toroidal_observables() {
        let signal = vec![0.5, 0.7, 0.8, 1.0];
        let obs = ToroidalAsymmetryObservables {
            n1_amp: 0.2,
            n2_amp: 0.1,
            n3_amp: 0.05,
            asymmetry_index: 0.25,
            radial_spread: 0.04,
        };
        let features = build_disruption_feature_vector(&signal, Some(obs));
        assert_eq!(features.len(), 11);
        assert!((features[6] - 0.2).abs() < 1e-12);
        assert!((features[9] - 0.25).abs() < 1e-12);
        assert!((features[10] - 0.04).abs() < 1e-12);
    }

    #[test]
    fn test_forward_with_observables_increases_risk_when_asymmetry_high() {
        let model = DisruptionTransformer::new();
        let signal = vec![0.2; 100];
        let low = model.forward_with_observables(&signal, None);
        let high = model.forward_with_observables(
            &signal,
            Some(ToroidalAsymmetryObservables {
                n1_amp: 1.6,
                n2_amp: 1.1,
                n3_amp: 0.8,
                asymmetry_index: 2.1,
                radial_spread: 0.5,
            }),
        );
        assert!(
            high > low,
            "High toroidal asymmetry should increase disruption risk: {} <= {}",
            high,
            low
        );
    }

    #[test]
    fn test_hybrid_anomaly_detector_outputs_bounded_scores() {
        let mut det = HybridAnomalyDetector::default();
        let signal = vec![0.4; 64];
        let (_s, _u, score, _alarm) = det.score(
            &signal,
            Some(ToroidalAsymmetryObservables {
                n1_amp: 0.2,
                n2_amp: 0.1,
                n3_amp: 0.05,
                asymmetry_index: 0.25,
                radial_spread: 0.03,
            }),
        );
        assert!(score.is_finite(), "Anomaly score must be finite");
        assert!(
            (0.0..=1.0).contains(&score),
            "Anomaly score must be in [0,1]: {score}"
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_hybrid_detector_stays_finite_under_random_perturbations(
            signal in proptest::collection::vec(-5.0f64..5.0f64, 1..192),
            n1 in 0.0f64..2.5f64,
            n2 in 0.0f64..2.5f64,
            n3 in 0.0f64..2.5f64,
            spread in 0.0f64..1.0f64
        ) {
            let mut det = HybridAnomalyDetector::default();
            let asym = (n1*n1 + n2*n2 + n3*n3).sqrt();
            let obs = ToroidalAsymmetryObservables {
                n1_amp: n1,
                n2_amp: n2,
                n3_amp: n3,
                asymmetry_index: asym,
                radial_spread: spread,
            };
            let (_s, _u, score, _alarm) = det.score(&signal, Some(obs));
            prop_assert!(score.is_finite());
            prop_assert!((0.0..=1.0).contains(&score));
        }
    }
}
