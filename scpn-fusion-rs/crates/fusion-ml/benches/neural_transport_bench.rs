// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Neural Transport Benchmark
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────

use criterion::{criterion_group, criterion_main, Criterion};
use fusion_ml::neural_transport::NeuralTransportModel;
use ndarray::{Array1, Array2};

/// Benchmark: single-input predict via analytic fallback.
///
/// Input features (10 elements):
///   [grad_ti, grad_te, grad_ne, shear, collisionality, zeff, q95, beta_n, rho, aspect_ratio]
fn bench_neural_transport_predict_analytic(c: &mut Criterion) {
    let model = NeuralTransportModel::new();

    // Physically reasonable input values for the analytic (critical-gradient) fallback.
    let input: Array1<f64> = Array1::from_vec(vec![
        5.0,  // grad_ti      — ion temperature gradient (above critical ~4.0)
        4.2,  // grad_te      — electron temperature gradient (above critical ~3.5)
        3.1,  // grad_ne      — electron density gradient (above critical ~2.5)
        0.3,  // shear        — magnetic shear (stabilises turbulence)
        0.05, // collisionality — normalised collision frequency
        1.5,  // zeff         — effective ion charge (>= 1.0)
        3.2,  // q95          — safety factor at 95% flux surface
        1.0,  // beta_n       — normalised plasma pressure
        0.6,  // rho          — normalised minor radius [0, 1]
        3.0,  // aspect_ratio — R/a torus aspect ratio
    ]);

    c.bench_function("bench_neural_transport_predict_analytic", |b| {
        b.iter(|| std::hint::black_box(model.predict(&input)))
    });
}

/// Benchmark: batch predict_profile over 50 radial points via analytic fallback.
///
/// The profile matrix has shape (50, 10); each row encodes one radial station
/// with the same feature ordering as above, swept across the minor radius.
fn bench_neural_transport_predict_profile_analytic(c: &mut Criterion) {
    let model = NeuralTransportModel::new();

    // Build a 50-point radial profile sweep.
    // rho runs from 0.02 (core) to 1.00 (edge); other features vary gently.
    let inputs: Array2<f64> = Array2::from_shape_fn((50, 10), |(i, j)| {
        let rho = (i as f64 + 1.0) / 50.0; // 0.02 … 1.00
        match j {
            0 => 4.0 + 2.0 * (1.0 - rho),  // grad_ti: peaks at core
            1 => 3.5 + 1.5 * (1.0 - rho),  // grad_te: peaks at core
            2 => 2.5 + 1.0 * (1.0 - rho),  // grad_ne: peaks at core
            3 => 0.1 + 0.4 * rho,           // shear: increases toward edge
            4 => 0.02 + 0.1 * rho,          // collisionality: increases toward edge
            5 => 1.2 + 0.3 * rho,           // zeff: slight impurity gradient
            6 => 3.0 + 0.5 * rho,           // q95: increases toward edge
            7 => 1.5 * (1.0 - rho),         // beta_n: peaks at core
            8 => rho,                        // rho itself
            _ => 3.0,                        // aspect_ratio: constant
        }
    });

    c.bench_function("bench_neural_transport_predict_profile_analytic", |b| {
        b.iter(|| std::hint::black_box(model.predict_profile(&inputs)))
    });
}

criterion_group!(
    benches,
    bench_neural_transport_predict_analytic,
    bench_neural_transport_predict_profile_analytic,
);
criterion_main!(benches);
