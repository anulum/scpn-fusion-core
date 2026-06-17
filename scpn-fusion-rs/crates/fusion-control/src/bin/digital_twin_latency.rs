// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Rust Digital-Twin Latency Benchmark
//! Native Rust digital-twin sensor-to-control latency benchmark.

use fusion_control::digital_twin::{ActuatorDelayLine, Plasma2D, SimpleMLP};
use ndarray::Array1;
use std::env;
use std::time::Instant;

const GRID: usize = 40;

#[derive(Debug)]
struct StageSamples {
    validate_ms: Vec<f64>,
    feature_ms: Vec<f64>,
    twin_ms: Vec<f64>,
    controller_ms: Vec<f64>,
    fallback_ms: Vec<f64>,
    serialization_ms: Vec<f64>,
    loop_ms: Vec<f64>,
}

impl StageSamples {
    fn new(steps: usize) -> Self {
        Self {
            validate_ms: Vec::with_capacity(steps),
            feature_ms: Vec::with_capacity(steps),
            twin_ms: Vec::with_capacity(steps),
            controller_ms: Vec::with_capacity(steps),
            fallback_ms: Vec::with_capacity(steps),
            serialization_ms: Vec::with_capacity(steps),
            loop_ms: Vec::with_capacity(steps),
        }
    }
}

fn parse_steps() -> Result<usize, String> {
    let mut steps = 320_usize;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--steps" => {
                let raw = args
                    .next()
                    .ok_or_else(|| "--steps requires a positive integer".to_string())?;
                steps = raw
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --steps value {raw:?}: {e}"))?;
            }
            "--json" => {}
            "--help" | "-h" => {
                println!("usage: digital_twin_latency [--steps N] [--json]");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    if steps < 32 {
        return Err("steps must be >= 32".to_string());
    }
    Ok(steps)
}

fn pct(values: &[f64], pct: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = (pct / 100.0) * (sorted.len().saturating_sub(1) as f64);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let w = rank - lo as f64;
        sorted[lo] * (1.0 - w) + sorted[hi] * w
    }
}

fn json_stage(name: &str, values: &[f64]) -> String {
    format!(
        "\"{name}\":{{\"p50_ms\":{:.9},\"p95_ms\":{:.9},\"p99_ms\":{:.9}}}",
        pct(values, 50.0),
        pct(values, 95.0),
        pct(values, 99.0)
    )
}

fn main() -> Result<(), String> {
    let steps = parse_steps()?;
    let mut plasma = Plasma2D::new();
    let mut controller = SimpleMLP::new(GRID).map_err(|e| e.to_string())?;
    let mut actuator = ActuatorDelayLine::new(1, 1, 0.5).map_err(|e| e.to_string())?;
    let mut samples = StageSamples::new(steps);
    let mut fallback_count = 0_usize;
    let mut checksum = 0.0_f64;

    for k in 0..steps {
        let loop_start = Instant::now();

        let t0 = Instant::now();
        let measured = plasma.measure_core_temp(0.0).map_err(|e| e.to_string())?;
        if !measured.is_finite() {
            return Err("sensor snapshot produced a non-finite value".to_string());
        }
        samples.validate_ms.push(t0.elapsed().as_secs_f64() * 1.0e3);

        let t0 = Instant::now();
        let mut features = Array1::zeros(GRID);
        let centre = GRID / 2;
        for j in 0..GRID {
            features[j] = (plasma.temp[[centre, j]] / 100.0).clamp(0.0, 1.0);
        }
        samples.feature_ms.push(t0.elapsed().as_secs_f64() * 1.0e3);

        let t0 = Instant::now();
        let process_noise = 0.02 * ((k as f64) * 0.031).sin();
        let (_core, avg) = plasma
            .step_with_process_noise(0.0, process_noise)
            .map_err(|e| e.to_string())?;
        samples.twin_ms.push(t0.elapsed().as_secs_f64() * 1.0e3);

        let t0 = Instant::now();
        let action = controller.forward(&features).map_err(|e| e.to_string())?;
        samples
            .controller_ms
            .push(t0.elapsed().as_secs_f64() * 1.0e3);

        let t0 = Instant::now();
        let safe_action = if action.is_finite() {
            action.clamp(-1.0, 1.0)
        } else {
            fallback_count += 1;
            0.0
        };
        let applied = actuator
            .push(Array1::from_vec(vec![safe_action]))
            .map_err(|e| e.to_string())?;
        samples.fallback_ms.push(t0.elapsed().as_secs_f64() * 1.0e3);

        let t0 = Instant::now();
        let serialised = format!(
            "{{\"step\":{k},\"action\":{:.9},\"applied\":{:.9},\"avg_temp\":{:.9}}}",
            safe_action, applied[0], avg
        );
        checksum += serialised.len() as f64 * 1.0e-6 + safe_action + applied[0];
        samples
            .serialization_ms
            .push(t0.elapsed().as_secs_f64() * 1.0e3);

        samples
            .loop_ms
            .push(loop_start.elapsed().as_secs_f64() * 1.0e3);
    }

    println!(
        "{{\"status\":\"measured\",\"backend\":\"rust_release\",\"steps\":{},\
         \"p50_loop_ms\":{:.9},\"p95_loop_ms\":{:.9},\"p99_loop_ms\":{:.9},\
         \"fallback_count\":{},\"checksum\":{:.9},\"stages\":{{{},{},{},{},{},{}}}}}",
        steps,
        pct(&samples.loop_ms, 50.0),
        pct(&samples.loop_ms, 95.0),
        pct(&samples.loop_ms, 99.0),
        fallback_count,
        checksum,
        json_stage("input_validation", &samples.validate_ms),
        json_stage("feature_assembly", &samples.feature_ms),
        json_stage("digital_twin_update", &samples.twin_ms),
        json_stage("controller_decision", &samples.controller_ms),
        json_stage("fallback_policy", &samples.fallback_ms),
        json_stage("output_serialization", &samples.serialization_ms)
    );
    Ok(())
}
