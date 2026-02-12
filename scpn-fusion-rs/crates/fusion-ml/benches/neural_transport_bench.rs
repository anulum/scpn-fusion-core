// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Neural Transport Benchmarks
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────

use criterion::{criterion_group, criterion_main, Criterion};
use fusion_ml::neural_transport::NeuralTransportModel;
use ndarray::{array, Array1, Array2};
use ndarray_npy::NpzWriter;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const INPUT_DIM: usize = 10;
const HIDDEN1: usize = 64;
const HIDDEN2: usize = 32;
const OUTPUT_DIM: usize = 3;

fn make_npz_path() -> PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "fusion_neural_transport_bench_{}_{}.npz",
        std::process::id(),
        ts
    ))
}

fn build_neural_model() -> NeuralTransportModel {
    let path = make_npz_path();
    let file = std::fs::File::create(&path).expect("create npz");
    let mut writer = NpzWriter::new(file);

    // Deterministic sparse-ish weights for benchmark consistency.
    let mut w1 = Array2::<f64>::zeros((INPUT_DIM, HIDDEN1));
    let b1 = Array1::<f64>::zeros(HIDDEN1);
    let mut w2 = Array2::<f64>::zeros((HIDDEN1, HIDDEN2));
    let b2 = Array1::<f64>::zeros(HIDDEN2);
    let mut w3 = Array2::<f64>::zeros((HIDDEN2, OUTPUT_DIM));
    let b3 = Array1::<f64>::zeros(OUTPUT_DIM);

    for i in 0..INPUT_DIM.min(HIDDEN1) {
        w1[[i, i]] = 0.2 + i as f64 * 0.01;
    }
    for i in 0..HIDDEN2 {
        w2[[i % HIDDEN1, i]] = 0.15;
    }
    for i in 0..OUTPUT_DIM {
        w3[[i, i]] = 0.3;
    }

    writer.add_array("w1", &w1).unwrap();
    writer.add_array("b1", &b1).unwrap();
    writer.add_array("w2", &w2).unwrap();
    writer.add_array("b2", &b2).unwrap();
    writer.add_array("w3", &w3).unwrap();
    writer.add_array("b3", &b3).unwrap();
    writer
        .add_array("input_mean", &Array1::<f64>::zeros(INPUT_DIM))
        .unwrap();
    writer
        .add_array("input_std", &Array1::<f64>::ones(INPUT_DIM))
        .unwrap();
    writer
        .add_array("output_scale", &array![1.0, 1.0, 1.0])
        .unwrap();
    writer.finish().unwrap();

    let model = NeuralTransportModel::from_npz(path.to_str().unwrap()).expect("load model");
    std::fs::remove_file(path).ok();
    model
}

fn bench_neural_vs_analytical_transport(c: &mut Criterion) {
    let neural_model = build_neural_model();
    let analytical_model = NeuralTransportModel::new();

    let input = array![5.0, 4.0, 3.0, 0.2, 0.1, 1.5, 3.2, 1.0, 0.6, 3.0];
    let batch_inputs = Array2::from_shape_fn((50, INPUT_DIM), |(i, j)| {
        0.2 + (i as f64) * 0.01 + (j as f64) * 0.05
    });

    let mut group = c.benchmark_group("neural_vs_analytical_transport");
    group.sample_size(20);

    group.bench_function("neural_single_predict", |b| {
        b.iter(|| {
            let out = neural_model.predict(black_box(&input));
            black_box(out);
        })
    });

    group.bench_function("analytical_single_predict", |b| {
        b.iter(|| {
            let out = analytical_model.predict(black_box(&input));
            black_box(out);
        })
    });

    group.bench_function("neural_batch_predict_50", |b| {
        b.iter(|| {
            let out = neural_model.predict_profile(black_box(&batch_inputs));
            black_box(out);
        })
    });

    group.bench_function("analytical_batch_predict_50", |b| {
        b.iter(|| {
            let out = analytical_model.predict_profile(black_box(&batch_inputs));
            black_box(out);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_neural_vs_analytical_transport);
criterion_main!(benches);
