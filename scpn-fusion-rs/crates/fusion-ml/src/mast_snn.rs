// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Rust MAST SNN Validation
//! Rust MAST SNN validation lane.

use fusion_types::error::{FusionError, FusionResult};
use ndarray::ArrayD;
use ndarray_npy::NpzReader;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs::File;
use std::path::Path;

const DEFAULT_N_NEURONS: usize = 64;
const DEFAULT_DT_S: f64 = 1e-4;
const DEFAULT_THRESHOLD: f64 = 0.3;
const MEMBRANE_TAU_S: f64 = 0.05e-3;
const SPIKE_SCORE_THRESHOLD: f64 = 0.85;
const MAX_NPZ_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct ShotTrace {
    pub time_s: Vec<f64>,
    pub plasma_current_a: Vec<f64>,
    pub magnetic_trace_t: Vec<f64>,
    pub magnetic_traces_t: Vec<Vec<f64>>,
    pub disruption_time_s: f64,
    pub source: String,
}

#[derive(Debug, Clone)]
pub struct MastSnnConfig {
    pub n_neurons: usize,
    pub epochs: usize,
    pub seed: u64,
    pub min_train_shots: usize,
    pub min_validation_shots: usize,
}

impl Default for MastSnnConfig {
    fn default() -> Self {
        Self {
            n_neurons: DEFAULT_N_NEURONS,
            epochs: 5,
            seed: 1729,
            min_train_shots: 3,
            min_validation_shots: 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShotStatus {
    Detected,
    NoDetection,
    Unavailable,
    NoFlattop,
}

impl ShotStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Detected => "detected",
            Self::NoDetection => "no_detection",
            Self::Unavailable => "unavailable",
            Self::NoFlattop => "no_flattop",
        }
    }
}

impl Serialize for ShotStatus {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ShotEvaluation {
    pub shot_id: i32,
    pub status: ShotStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disruption_time_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alarm_time_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lead_time_ms: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MastSnnReport {
    pub status: String,
    pub accepted_full_fidelity_ready: bool,
    pub claim_boundary: String,
    pub cache_dir: String,
    pub rust_runtime_backend: String,
    pub train_shots: Vec<i32>,
    pub val_shots: Vec<i32>,
    pub min_train_shots: usize,
    pub min_validation_shots: usize,
    pub train_available_count: usize,
    pub validation_available_count: usize,
    pub detected_validation_count: usize,
    pub local_count_gate_passed: bool,
    pub epochs: usize,
    pub seed: u64,
    pub average_lead_time_ms: Option<f64>,
    pub shots: Vec<ShotEvaluation>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct AlarmPolicy {
    pub spike_score_threshold: f64,
    pub membrane_threshold: f64,
    pub consecutive_samples: usize,
    pub rolling_window_samples: usize,
    pub scale_floor: f64,
    pub min_lead_time_ms: f64,
}

impl Default for AlarmPolicy {
    fn default() -> Self {
        Self {
            spike_score_threshold: SPIKE_SCORE_THRESHOLD,
            membrane_threshold: DEFAULT_THRESHOLD,
            consecutive_samples: 1,
            rolling_window_samples: 16,
            scale_floor: 1e-9,
            min_lead_time_ms: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PolicySweepRow {
    pub policy_index: usize,
    pub policy: AlarmPolicy,
    pub disruptive_available_count: usize,
    pub disruptive_detected_count: usize,
    pub negative_available_count: usize,
    pub false_positive_count: usize,
    pub recall: Option<f64>,
    pub false_positive_rate: Option<f64>,
    pub median_lead_time_ms: Option<f64>,
    pub min_lead_time_ms: Option<f64>,
    pub average_lead_time_ms: Option<f64>,
    pub score: f64,
    pub disruptive_shots: Vec<ShotEvaluation>,
    pub negative_shots: Vec<ShotEvaluation>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MastSnnSweepReport {
    pub status: String,
    pub accepted_full_fidelity_ready: bool,
    pub claim_boundary: String,
    pub cache_dir: String,
    pub rust_runtime_backend: String,
    pub train_shots: Vec<i32>,
    pub disruptive_validation_shots: Vec<i32>,
    pub negative_validation_shots: Vec<i32>,
    pub train_available_count: usize,
    pub disruptive_available_count: usize,
    pub negative_available_count: usize,
    pub best_policy_index: Option<usize>,
    pub recall: Option<f64>,
    pub false_positive_rate: Option<f64>,
    pub median_lead_time_ms: Option<f64>,
    pub min_lead_time_ms: Option<f64>,
    pub average_lead_time_ms: Option<f64>,
    pub policies: Vec<PolicySweepRow>,
}

#[derive(Debug, Clone, Deserialize)]
struct IndependentLabelManifest {
    manifest_version: String,
    dataset: String,
    label_authority: String,
    shots: Vec<IndependentLabelShot>,
}

#[derive(Debug, Clone, Deserialize)]
struct IndependentLabelShot {
    shot_id: i32,
    label: String,
    source_type: String,
    source_reference: String,
    labeled_by: String,
    labeled_at_utc: String,
    review_status: String,
    disruption_time_s: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IndependentDisruptiveShot {
    pub shot_id: i32,
    pub disruption_time_s: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IndependentMastLabels {
    pub disruptive: Vec<IndependentDisruptiveShot>,
    pub non_disruptive: Vec<i32>,
}

impl IndependentMastLabels {
    pub fn disruptive_ids(&self) -> Vec<i32> {
        self.disruptive.iter().map(|shot| shot.shot_id).collect()
    }
}

#[derive(Debug, Clone)]
pub struct HardwareSnn {
    alpha: f64,
    threshold: f64,
    weights: Vec<f64>,
    voltage: Vec<f64>,
}

impl HardwareSnn {
    pub fn new(weights: Vec<f64>) -> FusionResult<Self> {
        Self::with_threshold(weights, DEFAULT_THRESHOLD)
    }

    pub fn with_threshold(weights: Vec<f64>, threshold: f64) -> FusionResult<Self> {
        if weights.is_empty() {
            return Err(FusionError::ConfigError(
                "MAST SNN requires at least one neuron".to_string(),
            ));
        }
        if threshold <= 0.0 || !threshold.is_finite() {
            return Err(FusionError::ConfigError(
                "MAST SNN membrane threshold must be finite and positive".to_string(),
            ));
        }
        Ok(Self {
            alpha: DEFAULT_DT_S / MEMBRANE_TAU_S,
            threshold,
            voltage: vec![0.0; weights.len()],
            weights,
        })
    }

    pub fn reset(&mut self) {
        self.voltage.fill(0.0);
    }

    pub fn step(&mut self, signal: f64) -> f64 {
        let mut spikes = 0usize;
        for (voltage, weight) in self.voltage.iter_mut().zip(&self.weights) {
            let current = signal * weight * 10.0 + 0.01;
            *voltage += self.alpha * (-*voltage + current);
            if *voltage >= self.threshold {
                *voltage = 0.0;
                spikes += 1;
            }
        }
        spikes as f64 / self.weights.len() as f64
    }
}

pub fn initialise_base_weights(n_neurons: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n_neurons)
        .map(|_| rng.gen_range(4.5_f64..5.5_f64))
        .collect()
}

pub fn adapt_weights_for_available_count(
    n_neurons: usize,
    seed: u64,
    epochs: usize,
    available_count: usize,
) -> Vec<f64> {
    let scale = 1.02_f64.powi((epochs * available_count) as i32);
    initialise_base_weights(n_neurons, seed)
        .into_iter()
        .map(|weight| weight * scale)
        .collect()
}

pub fn load_local_npz_shot(cache_dir: &Path, shot_id: i32) -> FusionResult<Option<ShotTrace>> {
    let path = cache_dir.join(format!("mast_shot_{shot_id}.npz"));
    if !path.exists() {
        return Ok(None);
    }
    let metadata = std::fs::metadata(&path)?;
    if metadata.len() > MAX_NPZ_BYTES {
        return Err(FusionError::ConfigError(format!(
            "MAST NPZ archive exceeds {MAX_NPZ_BYTES} byte limit: {}",
            path.display()
        )));
    }

    let file = File::open(&path)?;
    let mut npz = NpzReader::new(file).map_err(|err| {
        FusionError::ConfigError(format!(
            "Failed to open MAST NPZ '{}': {err}",
            path.display()
        ))
    })?;
    let time_s = read_array1(&mut npz, "time")?;
    let plasma_current_a = read_array1(&mut npz, "ip")?;
    if time_s.len() != plasma_current_a.len() || time_s.len() < 2 {
        return Ok(None);
    }

    let max_ip = plasma_current_a
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max);
    if max_ip <= 0.0 || !max_ip.is_finite() {
        return Ok(None);
    }
    let Some(first_flattop) = plasma_current_a
        .iter()
        .position(|value| value.abs() > 0.8 * max_ip)
    else {
        return Ok(None);
    };
    let mut disruption_idx = plasma_current_a
        .iter()
        .rposition(|value| value.abs() > 0.8 * max_ip)
        .unwrap_or(first_flattop);
    for (idx, value) in plasma_current_a.iter().enumerate().skip(disruption_idx) {
        if value.abs() < 0.2 * max_ip {
            disruption_idx = idx;
            break;
        }
    }

    let magnetic_keys = magnetic_keys(&mut npz)?;
    if magnetic_keys.is_empty() {
        return Ok(None);
    }
    let mut magnetic_traces_t = Vec::with_capacity(magnetic_keys.len());
    for key in magnetic_keys {
        let magnetic = read_array_dynamic(&mut npz, &key)?;
        magnetic_traces_t.push(resample_to_summary_time(&magnetic, time_s.len()));
    }
    let magnetic_trace_t = magnetic_traces_t
        .first()
        .cloned()
        .unwrap_or_else(|| vec![0.0; time_s.len()]);

    Ok(Some(ShotTrace {
        disruption_time_s: time_s[disruption_idx],
        source: format!("local_npz:mast_shot_{shot_id}.npz"),
        time_s,
        plasma_current_a,
        magnetic_trace_t,
        magnetic_traces_t,
    }))
}

pub fn evaluate_trace(shot_id: i32, trace: &ShotTrace, weights: &[f64]) -> ShotEvaluation {
    let Some(dt_s) = mean_dt(&trace.time_s) else {
        return shot_status(shot_id, ShotStatus::Unavailable, Some(trace.source.clone()));
    };
    let max_ip = trace
        .plasma_current_a
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max);
    let Some(flattop_start) = trace
        .plasma_current_a
        .iter()
        .position(|value| value.abs() > 0.8 * max_ip)
    else {
        return shot_status(shot_id, ShotStatus::NoFlattop, Some(trace.source.clone()));
    };

    let magnetic_scale_t = trace
        .magnetic_trace_t
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max)
        .max(1.0);
    let mut detector = match HardwareSnn::new(weights.to_vec()) {
        Ok(detector) => detector,
        Err(_) => return shot_status(shot_id, ShotStatus::Unavailable, Some(trace.source.clone())),
    };
    detector.reset();

    for idx in 1..trace.time_s.len().min(trace.magnetic_trace_t.len()) {
        if trace.time_s[idx] >= trace.disruption_time_s {
            break;
        }
        let db_dt = (trace.magnetic_trace_t[idx] - trace.magnetic_trace_t[idx - 1]) / dt_s;
        let score = detector.step((db_dt / magnetic_scale_t).abs());
        if score > SPIKE_SCORE_THRESHOLD && idx > flattop_start {
            let alarm_time_s = trace.time_s[idx];
            return ShotEvaluation {
                shot_id,
                status: ShotStatus::Detected,
                source: Some(trace.source.clone()),
                disruption_time_s: Some(trace.disruption_time_s),
                alarm_time_s: Some(alarm_time_s),
                lead_time_ms: Some((trace.disruption_time_s - alarm_time_s) * 1000.0),
            };
        }
    }

    shot_status(shot_id, ShotStatus::NoDetection, Some(trace.source.clone()))
}

pub fn evaluate_trace_with_policy(
    shot_id: i32,
    trace: &ShotTrace,
    weights: &[f64],
    policy: &AlarmPolicy,
) -> ShotEvaluation {
    let Some(dt_s) = mean_dt(&trace.time_s) else {
        return shot_status(shot_id, ShotStatus::Unavailable, Some(trace.source.clone()));
    };
    let max_ip = trace
        .plasma_current_a
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max);
    let Some(flattop_start) = trace
        .plasma_current_a
        .iter()
        .position(|value| value.abs() > 0.8 * max_ip)
    else {
        return shot_status(shot_id, ShotStatus::NoFlattop, Some(trace.source.clone()));
    };
    let mut detector =
        match HardwareSnn::with_threshold(weights.to_vec(), policy.membrane_threshold) {
            Ok(detector) => detector,
            Err(_) => {
                return shot_status(shot_id, ShotStatus::Unavailable, Some(trace.source.clone()))
            }
        };
    detector.reset();

    let traces = if trace.magnetic_traces_t.is_empty() {
        vec![trace.magnetic_trace_t.as_slice()]
    } else {
        trace
            .magnetic_traces_t
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>()
    };
    let mut consecutive = 0usize;
    for idx in 1..trace.time_s.len() {
        if trace.time_s[idx] >= trace.disruption_time_s {
            break;
        }
        let signal = causal_multichannel_signal(&traces, idx, dt_s, policy);
        let score = detector.step(signal);
        let lead_time_ms = (trace.disruption_time_s - trace.time_s[idx]) * 1000.0;
        if score > policy.spike_score_threshold
            && idx > flattop_start
            && lead_time_ms >= policy.min_lead_time_ms
        {
            consecutive += 1;
            if consecutive >= policy.consecutive_samples.max(1) {
                let alarm_idx = idx + 1 - policy.consecutive_samples.max(1);
                let alarm_time_s = trace.time_s[alarm_idx];
                return ShotEvaluation {
                    shot_id,
                    status: ShotStatus::Detected,
                    source: Some(trace.source.clone()),
                    disruption_time_s: Some(trace.disruption_time_s),
                    alarm_time_s: Some(alarm_time_s),
                    lead_time_ms: Some((trace.disruption_time_s - alarm_time_s) * 1000.0),
                };
            }
        } else {
            consecutive = 0;
        }
    }

    shot_status(shot_id, ShotStatus::NoDetection, Some(trace.source.clone()))
}

pub fn evaluate_mast_snn_npz_panel(
    cache_dir: &Path,
    train_shots: &[i32],
    val_shots: &[i32],
    config: MastSnnConfig,
) -> FusionResult<MastSnnReport> {
    if config.n_neurons == 0 {
        return Err(FusionError::ConfigError(
            "MAST SNN n_neurons must be positive".to_string(),
        ));
    }

    let mut train_available_count = 0usize;
    for shot_id in train_shots {
        if load_local_npz_shot(cache_dir, *shot_id)?.is_some() {
            train_available_count += 1;
        }
    }
    let weights = adapt_weights_for_available_count(
        config.n_neurons,
        config.seed,
        config.epochs,
        train_available_count,
    );

    let mut shots = Vec::with_capacity(val_shots.len());
    for shot_id in val_shots {
        match load_local_npz_shot(cache_dir, *shot_id)? {
            Some(trace) => shots.push(evaluate_trace(*shot_id, &trace, &weights)),
            None => shots.push(shot_status(*shot_id, ShotStatus::Unavailable, None)),
        }
    }

    let detected: Vec<f64> = shots.iter().filter_map(|row| row.lead_time_ms).collect();
    let validation_available_count = shots
        .iter()
        .filter(|row| row.status != ShotStatus::Unavailable)
        .count();
    let detected_validation_count = detected.len();
    let status = classify_local_status(
        train_available_count,
        detected_validation_count,
        config.min_train_shots,
        config.min_validation_shots,
    );
    let local_count_gate_passed = train_available_count >= config.min_train_shots
        && detected_validation_count >= config.min_validation_shots;

    Ok(MastSnnReport {
        status: status.to_string(),
        accepted_full_fidelity_ready: false,
        claim_boundary: "Rust MAST SNN reports are local disruption-detection diagnostics only. Full-fidelity claims require same-case magnetic-geometry validation, shot provenance review, and independent acceptance gates beyond local train and detected-validation counts.".to_string(),
        cache_dir: cache_dir.display().to_string(),
        rust_runtime_backend: "fusion-ml::mast_snn".to_string(),
        train_shots: train_shots.to_vec(),
        val_shots: val_shots.to_vec(),
        min_train_shots: config.min_train_shots,
        min_validation_shots: config.min_validation_shots,
        train_available_count,
        validation_available_count,
        detected_validation_count,
        local_count_gate_passed,
        epochs: config.epochs,
        seed: config.seed,
        average_lead_time_ms: mean(&detected),
        shots,
    })
}

pub fn sweep_mast_snn_npz_panel(
    cache_dir: &Path,
    train_shots: &[i32],
    disruptive_val_shots: &[i32],
    negative_val_shots: &[i32],
    config: MastSnnConfig,
    policies: &[AlarmPolicy],
) -> FusionResult<MastSnnSweepReport> {
    if policies.is_empty() {
        return Err(FusionError::ConfigError(
            "MAST SNN policy sweep requires at least one policy".to_string(),
        ));
    }
    let mut train_available_count = 0usize;
    for shot_id in train_shots {
        if load_local_npz_shot(cache_dir, *shot_id)?.is_some() {
            train_available_count += 1;
        }
    }
    let weights = adapt_weights_for_available_count(
        config.n_neurons,
        config.seed,
        config.epochs,
        train_available_count,
    );

    let disruptive_traces = load_labeled_traces(cache_dir, disruptive_val_shots)?;
    let negative_traces = load_negative_traces(cache_dir, negative_val_shots)?;
    sweep_loaded_panel(
        cache_dir,
        train_shots,
        disruptive_val_shots,
        negative_val_shots,
        train_available_count,
        &weights,
        disruptive_traces,
        negative_traces,
        policies,
    )
}

pub fn sweep_mast_snn_npz_panel_with_independent_labels(
    cache_dir: &Path,
    train_shots: &[i32],
    labels: &IndependentMastLabels,
    config: MastSnnConfig,
    policies: &[AlarmPolicy],
) -> FusionResult<MastSnnSweepReport> {
    if policies.is_empty() {
        return Err(FusionError::ConfigError(
            "MAST SNN policy sweep requires at least one policy".to_string(),
        ));
    }
    let mut train_available_count = 0usize;
    for shot_id in train_shots {
        if load_local_npz_shot(cache_dir, *shot_id)?.is_some() {
            train_available_count += 1;
        }
    }
    let weights = adapt_weights_for_available_count(
        config.n_neurons,
        config.seed,
        config.epochs,
        train_available_count,
    );

    let disruptive_ids = labels.disruptive_ids();
    let disruptive_traces = load_disruptive_traces_with_label_times(cache_dir, &labels.disruptive)?;
    let negative_traces = load_negative_traces(cache_dir, &labels.non_disruptive)?;
    sweep_loaded_panel(
        cache_dir,
        train_shots,
        &disruptive_ids,
        &labels.non_disruptive,
        train_available_count,
        &weights,
        disruptive_traces,
        negative_traces,
        policies,
    )
}

#[allow(clippy::too_many_arguments)]
fn sweep_loaded_panel(
    cache_dir: &Path,
    train_shots: &[i32],
    disruptive_val_shots: &[i32],
    negative_val_shots: &[i32],
    train_available_count: usize,
    weights: &[f64],
    disruptive_traces: Vec<(i32, ShotTrace)>,
    negative_traces: Vec<(i32, ShotTrace)>,
    policies: &[AlarmPolicy],
) -> FusionResult<MastSnnSweepReport> {
    let mut rows = Vec::with_capacity(policies.len());
    for (policy_index, policy) in policies.iter().enumerate() {
        let disruptive_shots = disruptive_traces
            .iter()
            .map(|(shot_id, trace)| evaluate_trace_with_policy(*shot_id, trace, weights, policy))
            .collect::<Vec<_>>();
        let negative_shots = negative_traces
            .iter()
            .map(|(shot_id, trace)| evaluate_trace_with_policy(*shot_id, trace, weights, policy))
            .collect::<Vec<_>>();
        let disruptive_detected_count = disruptive_shots
            .iter()
            .filter(|row| row.status == ShotStatus::Detected)
            .count();
        let false_positive_count = negative_shots
            .iter()
            .filter(|row| row.status == ShotStatus::Detected)
            .count();
        let lead_times = disruptive_shots
            .iter()
            .filter_map(|row| row.lead_time_ms)
            .collect::<Vec<_>>();
        let recall = ratio(disruptive_detected_count, disruptive_traces.len());
        let false_positive_rate = ratio(false_positive_count, negative_traces.len());
        let median_lead_time_ms = median(&lead_times);
        let min_lead_time_ms = lead_times.iter().copied().reduce(f64::min);
        let average_lead_time_ms = mean(&lead_times);
        let score = policy_score(recall, false_positive_rate, median_lead_time_ms);
        rows.push(PolicySweepRow {
            policy_index,
            policy: *policy,
            disruptive_available_count: disruptive_traces.len(),
            disruptive_detected_count,
            negative_available_count: negative_traces.len(),
            false_positive_count,
            recall,
            false_positive_rate,
            median_lead_time_ms,
            min_lead_time_ms,
            average_lead_time_ms,
            score,
            disruptive_shots,
            negative_shots,
        });
    }

    let best_policy_index = rows
        .iter()
        .max_by(|left, right| left.score.total_cmp(&right.score))
        .map(|row| row.policy_index);
    let best = best_policy_index.and_then(|idx| rows.iter().find(|row| row.policy_index == idx));
    let negative_available_count = negative_traces.len();
    let status = if negative_available_count == 0 {
        "blocked_missing_negative_validation_shots"
    } else if best.and_then(|row| row.recall).unwrap_or(0.0) < 0.98 {
        "blocked_recall_below_acceptance"
    } else if best.and_then(|row| row.false_positive_rate).unwrap_or(1.0) > 0.02 {
        "blocked_false_positive_rate_above_acceptance"
    } else {
        "blocked_local_mast_snn_not_physics_validation"
    };

    Ok(MastSnnSweepReport {
        status: status.to_string(),
        accepted_full_fidelity_ready: false,
        claim_boundary: "Rust MAST SNN policy sweeps are local diagnostics. Full acceptance requires labeled disruptive and non-disruptive validation shots, causal feature review, and independent physics-validation gates.".to_string(),
        cache_dir: cache_dir.display().to_string(),
        rust_runtime_backend: "fusion-ml::mast_snn_policy_sweep".to_string(),
        train_shots: train_shots.to_vec(),
        disruptive_validation_shots: disruptive_val_shots.to_vec(),
        negative_validation_shots: negative_val_shots.to_vec(),
        train_available_count,
        disruptive_available_count: disruptive_traces.len(),
        negative_available_count,
        best_policy_index,
        recall: best.and_then(|row| row.recall),
        false_positive_rate: best.and_then(|row| row.false_positive_rate),
        median_lead_time_ms: best.and_then(|row| row.median_lead_time_ms),
        min_lead_time_ms: best.and_then(|row| row.min_lead_time_ms),
        average_lead_time_ms: best.and_then(|row| row.average_lead_time_ms),
        policies: rows,
    })
}

pub fn default_policy_grid() -> Vec<AlarmPolicy> {
    let mut policies = Vec::new();
    for spike_score_threshold in [0.45, 0.55, 0.65, 0.75, 0.85] {
        for consecutive_samples in [1, 2, 3] {
            for rolling_window_samples in [4, 8, 16, 32] {
                policies.push(AlarmPolicy {
                    spike_score_threshold,
                    consecutive_samples,
                    rolling_window_samples,
                    ..AlarmPolicy::default()
                });
            }
        }
    }
    policies
}

pub fn load_independent_label_manifest(path: &Path) -> FusionResult<IndependentMastLabels> {
    let text = std::fs::read_to_string(path)?;
    let manifest: IndependentLabelManifest = serde_json::from_str(&text)?;
    if manifest.manifest_version != "mast-independent-disruption-labels-v1" {
        return Err(FusionError::ConfigError(
            "MAST label manifest has unsupported manifest_version".to_string(),
        ));
    }
    if manifest.dataset.trim().is_empty() || manifest.label_authority.trim().is_empty() {
        return Err(FusionError::ConfigError(
            "MAST label manifest must identify dataset and label_authority".to_string(),
        ));
    }
    let mut disruptive = Vec::new();
    let mut non_disruptive = Vec::new();
    let mut seen = BTreeSet::new();
    for (index, shot) in manifest.shots.iter().enumerate() {
        validate_independent_label_shot(index, shot)?;
        if !seen.insert(shot.shot_id) {
            return Err(FusionError::ConfigError(format!(
                "MAST label manifest shot[{index}] duplicates shot {}",
                shot.shot_id
            )));
        }
        match shot.label.as_str() {
            "disruptive" => disruptive.push(IndependentDisruptiveShot {
                shot_id: shot.shot_id,
                disruption_time_s: shot.disruption_time_s.unwrap_or(0.0),
            }),
            "non_disruptive" => non_disruptive.push(shot.shot_id),
            _ => {
                return Err(FusionError::ConfigError(format!(
                    "MAST label manifest shot[{index}] has unsupported label '{}'",
                    shot.label
                )));
            }
        }
    }
    if disruptive.is_empty() || non_disruptive.is_empty() {
        return Err(FusionError::ConfigError(
            "MAST label manifest must include disruptive and non_disruptive shots".to_string(),
        ));
    }
    Ok(IndependentMastLabels {
        disruptive,
        non_disruptive,
    })
}

fn validate_independent_label_shot(index: usize, shot: &IndependentLabelShot) -> FusionResult<()> {
    if shot.shot_id <= 0 {
        return Err(FusionError::ConfigError(format!(
            "MAST label manifest shot[{index}].shot_id must be positive"
        )));
    }
    if !matches!(
        shot.source_type.as_str(),
        "curated_database" | "facility_log" | "operator_log" | "published_table"
    ) {
        return Err(FusionError::ConfigError(format!(
            "MAST label manifest shot[{index}].source_type is not accepted independent evidence"
        )));
    }
    if shot.source_reference.trim().is_empty()
        || shot.labeled_by.trim().is_empty()
        || shot.labeled_at_utc.trim().is_empty()
        || !is_plausible_iso8601_utc(&shot.labeled_at_utc)
        || shot.review_status != "accepted"
    {
        return Err(FusionError::ConfigError(format!(
            "MAST label manifest shot[{index}] is missing accepted provenance"
        )));
    }
    if shot.label == "disruptive" && shot.disruption_time_s.unwrap_or(0.0) <= 0.0 {
        return Err(FusionError::ConfigError(format!(
            "MAST label manifest shot[{index}].disruption_time_s must be positive"
        )));
    }
    if shot.label == "non_disruptive" && shot.disruption_time_s.is_some() {
        return Err(FusionError::ConfigError(format!(
            "MAST label manifest shot[{index}].disruption_time_s must be absent"
        )));
    }
    Ok(())
}

fn is_plausible_iso8601_utc(value: &str) -> bool {
    let value = value.trim();
    value.len() >= "2026-06-16T00:00:00Z".len() && value.contains('T') && value.ends_with('Z')
}

fn classify_local_status(
    train_available_count: usize,
    detected_validation_count: usize,
    min_train_shots: usize,
    min_validation_shots: usize,
) -> &'static str {
    if train_available_count < min_train_shots {
        "blocked_insufficient_training_shots"
    } else if detected_validation_count < min_validation_shots {
        "blocked_insufficient_detected_validation_shots"
    } else {
        "blocked_local_mast_snn_not_physics_validation"
    }
}

fn read_array1(npz: &mut NpzReader<File>, key: &str) -> FusionResult<Vec<f64>> {
    let arr = npz
        .by_name::<ndarray::OwnedRepr<f64>, ndarray::Ix1>(&format!("{key}.npy"))
        .or_else(|_| npz.by_name::<ndarray::OwnedRepr<f64>, ndarray::Ix1>(key))
        .map_err(|err| FusionError::ConfigError(format!("Failed to read {key}: {err}")))?;
    Ok(clean_values(arr.to_vec()))
}

fn read_array_dynamic(npz: &mut NpzReader<File>, key: &str) -> FusionResult<ArrayD<f64>> {
    npz.by_name::<ndarray::OwnedRepr<f64>, ndarray::IxDyn>(&format!("{key}.npy"))
        .or_else(|_| npz.by_name::<ndarray::OwnedRepr<f64>, ndarray::IxDyn>(key))
        .map_err(|err| FusionError::ConfigError(format!("Failed to read {key}: {err}")))
}

fn magnetic_keys(npz: &mut NpzReader<File>) -> FusionResult<Vec<String>> {
    let names = npz
        .names()
        .map_err(|err| FusionError::ConfigError(format!("Failed to inspect NPZ names: {err}")))?;
    Ok(names
        .into_iter()
        .filter_map(|name| {
            let key = name.strip_suffix(".npy").unwrap_or(&name);
            (key.starts_with("mag_") && key.ends_with("_field")).then(|| key.to_string())
        })
        .collect())
}

fn resample_to_summary_time(trace: &ArrayD<f64>, n_time: usize) -> Vec<f64> {
    let flat = clean_values(trace.iter().copied().collect());
    if flat.is_empty() {
        return vec![0.0; n_time];
    }
    if flat.len() == n_time {
        return flat;
    }
    if n_time == 1 {
        return vec![flat[0]];
    }
    (0..n_time)
        .map(|idx| {
            let src_idx = idx * (flat.len() - 1) / (n_time - 1);
            flat[src_idx]
        })
        .collect()
}

fn clean_values(values: Vec<f64>) -> Vec<f64> {
    values
        .into_iter()
        .map(|value| if value.is_finite() { value } else { 0.0 })
        .collect()
}

fn mean_dt(time_s: &[f64]) -> Option<f64> {
    if time_s.len() < 2 {
        return None;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for pair in time_s.windows(2) {
        let dt = pair[1] - pair[0];
        if dt.is_finite() && dt > 0.0 {
            sum += dt;
            count += 1;
        }
    }
    (count > 0).then_some(sum / count as f64)
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then_some(values.iter().sum::<f64>() / values.len() as f64)
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    Some(sorted[sorted.len() / 2])
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
}

fn policy_score(
    recall: Option<f64>,
    false_positive_rate: Option<f64>,
    median_lead_time_ms: Option<f64>,
) -> f64 {
    let recall = recall.unwrap_or(0.0);
    let fpr_penalty = false_positive_rate.unwrap_or(0.5);
    let lead_bonus = median_lead_time_ms.unwrap_or(0.0).min(150.0) / 150.0;
    recall * 10.0 + lead_bonus - fpr_penalty * 5.0
}

fn causal_multichannel_signal(
    traces: &[&[f64]],
    idx: usize,
    dt_s: f64,
    policy: &AlarmPolicy,
) -> f64 {
    traces
        .iter()
        .filter_map(|trace| causal_channel_signal(trace, idx, dt_s, policy))
        .fold(0.0, f64::max)
}

fn causal_channel_signal(
    trace: &[f64],
    idx: usize,
    dt_s: f64,
    policy: &AlarmPolicy,
) -> Option<f64> {
    if idx == 0 || idx >= trace.len() || dt_s <= 0.0 {
        return None;
    }
    let db_dt = (trace[idx] - trace[idx - 1]) / dt_s;
    let start = idx.saturating_sub(policy.rolling_window_samples.max(1));
    let mut scale = policy.scale_floor.max(1e-12);
    for cursor in start.max(1)..idx {
        let prev = ((trace[cursor] - trace[cursor - 1]) / dt_s).abs();
        if prev.is_finite() {
            scale = scale.max(prev);
        }
    }
    Some((db_dt / scale).abs())
}

fn load_labeled_traces(cache_dir: &Path, shot_ids: &[i32]) -> FusionResult<Vec<(i32, ShotTrace)>> {
    let mut traces = Vec::with_capacity(shot_ids.len());
    for shot_id in shot_ids {
        if let Some(trace) = load_local_npz_shot(cache_dir, *shot_id)? {
            traces.push((*shot_id, trace));
        }
    }
    Ok(traces)
}

fn load_disruptive_traces_with_label_times(
    cache_dir: &Path,
    labels: &[IndependentDisruptiveShot],
) -> FusionResult<Vec<(i32, ShotTrace)>> {
    let mut traces = Vec::with_capacity(labels.len());
    for label in labels {
        if let Some(mut trace) = load_local_npz_shot(cache_dir, label.shot_id)? {
            trace.disruption_time_s = label.disruption_time_s;
            traces.push((label.shot_id, trace));
        }
    }
    Ok(traces)
}

fn load_negative_traces(cache_dir: &Path, shot_ids: &[i32]) -> FusionResult<Vec<(i32, ShotTrace)>> {
    let mut traces = Vec::with_capacity(shot_ids.len());
    for shot_id in shot_ids {
        if let Some(mut trace) = load_local_npz_shot(cache_dir, *shot_id)? {
            if let Some(last_time_s) = trace.time_s.last().copied() {
                trace.disruption_time_s = last_time_s;
            }
            traces.push((*shot_id, trace));
        }
    }
    Ok(traces)
}

fn shot_status(shot_id: i32, status: ShotStatus, source: Option<String>) -> ShotEvaluation {
    ShotEvaluation {
        shot_id,
        status,
        source,
        disruption_time_s: None,
        alarm_time_s: None,
        lead_time_ms: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};
    use ndarray_npy::NpzWriter;
    use std::fs::File;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TEMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "fusion_ml_mast_snn_{}_{}",
                std::process::id(),
                id
            ));
            std::fs::create_dir_all(&path).expect("create temp test dir");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write_fixture(path: &Path, time: &[f64], ip: &[f64], magnetic: &[f64]) {
        let file = File::create(path).expect("create npz fixture");
        let mut npz = NpzWriter::new(file);
        npz.add_array("time", &Array1::from_vec(time.to_vec()))
            .expect("write time");
        npz.add_array("ip", &Array1::from_vec(ip.to_vec()))
            .expect("write ip");
        npz.add_array(
            "mag_b_field_pol_probe_cc_field",
            &Array2::from_shape_vec((1, magnetic.len()), magnetic.to_vec()).expect("mag shape"),
        )
        .expect("write magnetic");
        npz.finish().expect("finish npz");
    }

    #[test]
    fn local_npz_loader_derives_disruption_and_resamples_magnetics() {
        let dir = TestDir::new();
        let path = dir.path().join("mast_shot_12345.npz");
        write_fixture(
            &path,
            &[0.0, 0.1, 0.2, 0.3, 0.4],
            &[0.0, 0.9, 1.0, 0.1, 0.0],
            &[0.0, 1.0, 2.0],
        );

        let trace = load_local_npz_shot(dir.path(), 12345)
            .expect("loader result")
            .expect("loaded trace");

        assert_eq!(trace.source, "local_npz:mast_shot_12345.npz");
        assert_eq!(trace.time_s.len(), 5);
        assert_eq!(trace.magnetic_trace_t.len(), 5);
        assert!((trace.disruption_time_s - 0.3).abs() < 1e-12);
    }

    #[test]
    fn rust_mast_snn_detects_step_before_disruption() {
        let weights = initialise_base_weights(64, 1729);
        let trace = ShotTrace {
            time_s: (0..20).map(|idx| idx as f64 * 0.01).collect(),
            plasma_current_a: vec![
                0.0, 0.2, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.95, 0.8, 0.6, 0.1, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
            ],
            magnetic_trace_t: vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 110.0, 112.0, 112.0, 112.0,
                112.0, 112.0, 112.0, 112.0, 112.0, 112.0,
            ],
            magnetic_traces_t: vec![vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 110.0, 112.0, 112.0, 112.0,
                112.0, 112.0, 112.0, 112.0, 112.0, 112.0,
            ]],
            disruption_time_s: 0.12,
            source: "fixture".to_string(),
        };

        let row = evaluate_trace(30456, &trace, &weights);

        assert_eq!(row.status, ShotStatus::Detected);
        assert!(row.lead_time_ms.expect("lead time") > 0.0);
    }

    #[test]
    fn rust_mast_snn_report_preserves_local_claim_boundary() {
        let dir = TestDir::new();
        for shot_id in [1, 2, 3, 4] {
            let path = dir.path().join(format!("mast_shot_{shot_id}.npz"));
            write_fixture(
                &path,
                &[0.0, 0.01, 0.02, 0.03, 0.04],
                &[0.0, 0.9, 1.0, 0.1, 0.0],
                &[0.0, 0.0, 200.0, 200.0, 200.0],
            );
        }

        let report = evaluate_mast_snn_npz_panel(
            dir.path(),
            &[1, 2],
            &[3, 4],
            MastSnnConfig {
                min_train_shots: 2,
                min_validation_shots: 1,
                ..MastSnnConfig::default()
            },
        )
        .expect("report");

        assert_eq!(report.train_available_count, 2);
        assert_eq!(report.validation_available_count, 2);
        assert!(!report.accepted_full_fidelity_ready);
        assert_eq!(
            report.status,
            "blocked_local_mast_snn_not_physics_validation"
        );
    }

    #[test]
    fn causal_policy_detects_precursor_without_full_trace_normalization() {
        let weights = adapt_weights_for_available_count(64, 1729, 5, 37);
        let trace = ShotTrace {
            time_s: (0..24).map(|idx| idx as f64 * 0.01).collect(),
            plasma_current_a: vec![
                0.0, 0.2, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4,
                0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            magnetic_trace_t: vec![
                0.0, 0.0, 0.0, 0.02, 0.04, 0.06, 0.09, 0.13, 0.18, 0.24, 0.31, 0.39, 0.48, 0.58,
                0.69, 0.81, 0.94, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6,
            ],
            magnetic_traces_t: vec![vec![
                0.0, 0.0, 0.0, 0.02, 0.04, 0.06, 0.09, 0.13, 0.18, 0.24, 0.31, 0.39, 0.48, 0.58,
                0.69, 0.81, 0.94, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6,
            ]],
            disruption_time_s: 0.17,
            source: "fixture".to_string(),
        };

        let legacy = evaluate_trace(30456, &trace, &weights);
        let causal = evaluate_trace_with_policy(
            30456,
            &trace,
            &weights,
            &AlarmPolicy {
                spike_score_threshold: 0.55,
                consecutive_samples: 2,
                rolling_window_samples: 4,
                ..AlarmPolicy::default()
            },
        );

        assert_eq!(causal.status, ShotStatus::Detected);
        assert_eq!(legacy.status, ShotStatus::NoDetection);
        assert!(causal.lead_time_ms.expect("causal lead") >= 40.0);
    }

    #[test]
    fn confirmation_policy_rejects_single_sample_spike() {
        let weights = adapt_weights_for_available_count(64, 1729, 5, 37);
        let trace = ShotTrace {
            time_s: (0..14).map(|idx| idx as f64 * 0.01).collect(),
            plasma_current_a: vec![
                0.0, 0.2, 0.9, 1.0, 1.0, 1.0, 0.95, 0.9, 0.7, 0.5, 0.2, 0.0, 0.0, 0.0,
            ],
            magnetic_trace_t: vec![
                0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            magnetic_traces_t: vec![vec![
                0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]],
            disruption_time_s: 0.11,
            source: "fixture".to_string(),
        };

        let row = evaluate_trace_with_policy(
            30456,
            &trace,
            &weights,
            &AlarmPolicy {
                spike_score_threshold: 0.55,
                consecutive_samples: 3,
                rolling_window_samples: 3,
                ..AlarmPolicy::default()
            },
        );

        assert_eq!(row.status, ShotStatus::NoDetection);
    }

    #[test]
    fn policy_sweep_reports_best_tradeoff_without_negative_claims() {
        let dir = TestDir::new();
        for shot_id in [1, 2, 3] {
            let path = dir.path().join(format!("mast_shot_{shot_id}.npz"));
            write_fixture(
                &path,
                &[0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
                &[0.0, 0.9, 1.0, 0.9, 0.1, 0.0],
                &[0.0, 0.0, 0.1, 20.0, 20.0, 20.0],
            );
        }

        let sweep = sweep_mast_snn_npz_panel(
            dir.path(),
            &[1],
            &[2, 3],
            &[],
            MastSnnConfig {
                min_train_shots: 1,
                min_validation_shots: 1,
                ..MastSnnConfig::default()
            },
            &[AlarmPolicy {
                spike_score_threshold: 0.55,
                consecutive_samples: 1,
                rolling_window_samples: 2,
                ..AlarmPolicy::default()
            }],
        )
        .expect("sweep report");

        assert_eq!(sweep.best_policy_index, Some(0));
        assert_eq!(sweep.negative_available_count, 0);
        assert!(sweep.false_positive_rate.is_none());
        assert!(sweep.status.contains("negative"));
    }

    #[test]
    fn independent_label_manifest_separates_disruptive_and_negative_ids() {
        let dir = TestDir::new();
        let manifest = dir.path().join("labels.json");
        std::fs::write(
            &manifest,
            r#"{
              "manifest_version": "mast-independent-disruption-labels-v1",
              "dataset": "FAIR-MAST Level-2 bounded disruption panel",
              "label_authority": "facility log export",
              "shots": [
                {
                  "shot_id": 30456,
                  "label": "disruptive",
                  "source_type": "facility_log",
                  "source_reference": "facility-log://mast/30456",
                  "labeled_by": "facility",
                  "labeled_at_utc": "2026-06-16T00:00:00Z",
                  "review_status": "accepted",
                  "disruption_time_s": 0.22
                },
                {
                  "shot_id": 30420,
                  "label": "non_disruptive",
                  "source_type": "published_table",
                  "source_reference": "doi:10.example/mast",
                  "labeled_by": "paper",
                  "labeled_at_utc": "2026-06-16T00:00:00Z",
                  "review_status": "accepted"
                }
              ]
            }"#,
        )
        .expect("write manifest");

        let labels = load_independent_label_manifest(&manifest).expect("load labels");

        assert_eq!(
            labels.disruptive,
            vec![IndependentDisruptiveShot {
                shot_id: 30456,
                disruption_time_s: 0.22
            }]
        );
        assert_eq!(labels.non_disruptive, vec![30420]);
    }

    #[test]
    fn independent_label_manifest_rejects_proxy_source() {
        let dir = TestDir::new();
        let manifest = dir.path().join("proxy_labels.json");
        std::fs::write(
            &manifest,
            r#"{
              "manifest_version": "mast-independent-disruption-labels-v1",
              "dataset": "FAIR-MAST Level-2 bounded disruption panel",
              "label_authority": "facility log export",
              "shots": [
                {
                  "shot_id": 30456,
                  "label": "disruptive",
                  "source_type": "current_collapse_proxy",
                  "source_reference": "local-ip-drop",
                  "labeled_by": "local detector",
                  "labeled_at_utc": "2026-06-16T00:00:00Z",
                  "review_status": "accepted",
                  "disruption_time_s": 0.22
                },
                {
                  "shot_id": 30420,
                  "label": "non_disruptive",
                  "source_type": "published_table",
                  "source_reference": "doi:10.example/mast",
                  "labeled_by": "paper",
                  "labeled_at_utc": "2026-06-16T00:00:00Z",
                  "review_status": "accepted"
                }
              ]
            }"#,
        )
        .expect("write manifest");

        let err = load_independent_label_manifest(&manifest).expect_err("reject proxy label");

        assert!(err
            .to_string()
            .contains("not accepted independent evidence"));
    }

    #[test]
    fn independent_label_sweep_uses_label_time_and_full_negative_trace() {
        let dir = TestDir::new();
        for shot_id in [1, 2, 3] {
            let path = dir.path().join(format!("mast_shot_{shot_id}.npz"));
            write_fixture(
                &path,
                &[0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
                &[0.0, 0.9, 1.0, 0.9, 0.1, 0.0],
                &[0.0, 0.0, 0.1, 20.0, 20.0, 20.0],
            );
        }
        let labels = IndependentMastLabels {
            disruptive: vec![IndependentDisruptiveShot {
                shot_id: 2,
                disruption_time_s: 0.05,
            }],
            non_disruptive: vec![3],
        };

        let sweep = sweep_mast_snn_npz_panel_with_independent_labels(
            dir.path(),
            &[1],
            &labels,
            MastSnnConfig {
                min_train_shots: 1,
                min_validation_shots: 1,
                ..MastSnnConfig::default()
            },
            &[AlarmPolicy {
                spike_score_threshold: 0.55,
                consecutive_samples: 1,
                rolling_window_samples: 2,
                ..AlarmPolicy::default()
            }],
        )
        .expect("sweep report");

        assert_eq!(sweep.disruptive_available_count, 1);
        assert_eq!(sweep.negative_available_count, 1);
        assert_eq!(sweep.false_positive_rate, Some(1.0));
        assert!(sweep.median_lead_time_ms.expect("lead time") > 0.0);
    }
}
