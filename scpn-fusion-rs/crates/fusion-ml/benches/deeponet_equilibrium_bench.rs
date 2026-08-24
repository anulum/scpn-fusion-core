// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Native DeepONet Equilibrium Benchmark

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use fusion_ml::deeponet_equilibrium::{DeepOnetEquilibrium, DeepOnetEquilibriumConfig, DenseLayer};
use ndarray::{Array1, Array2};

fn layer(input: usize, output: usize, value: f64) -> DenseLayer {
    DenseLayer {
        weights: Array2::from_elem((input, output), value),
        bias: Array1::zeros(output),
    }
}

fn runtime() -> DeepOnetEquilibrium {
    let grid_nh = 129;
    let grid_nw = 129;
    let coordinates = Array2::from_shape_fn((grid_nh * grid_nw, 2), |(index, axis)| {
        if axis == 0 {
            3.0 + 6.0 * (index % grid_nw) as f64 / (grid_nw - 1) as f64
        } else {
            -5.0 + 10.0 * (index / grid_nw) as f64 / (grid_nh - 1) as f64
        }
    });
    DeepOnetEquilibrium::new(DeepOnetEquilibriumConfig {
        branch: vec![
            layer(17, 256, 0.001),
            layer(256, 256, 0.001),
            layer(256, 64, 0.001),
        ],
        trunk: vec![
            layer(2, 128, 0.001),
            layer(128, 128, 0.001),
            layer(128, 64, 0.001),
        ],
        input_mean: Array1::zeros(17),
        input_std: Array1::ones(17),
        coordinates_rz_m: coordinates,
        coordinate_mean: Array1::from_vec(vec![6.0, 0.0]),
        coordinate_std: Array1::from_vec(vec![3.0, 5.0]),
        field_mean: Array1::zeros(grid_nh * grid_nw),
        field_scale: 1.0,
        basis_width: 64,
        grid_shape: (grid_nh, grid_nw),
    })
    .unwrap_or_else(|error| panic!("benchmark DeepONet contract is invalid: {error}"))
}

fn bench_deeponet_predict(c: &mut Criterion) {
    let model = runtime();
    let features = Array1::from_elem(17, 0.25);
    c.bench_function("deeponet_equilibrium_predict_129x129", |bencher| {
        bencher.iter(|| {
            model
                .predict(black_box(&features))
                .unwrap_or_else(|error| panic!("benchmark inference failed: {error}"))
        });
    });
}

criterion_group!(benches, bench_deeponet_predict);
criterion_main!(benches);
