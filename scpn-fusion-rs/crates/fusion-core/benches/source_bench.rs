// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — GEQDSK Profile Source Benchmarks

use criterion::{criterion_group, criterion_main, Criterion};
use fusion_core::source::compute_geqdsk_profile_source_components;
use ndarray::Array2;
use std::hint::black_box;

/// Physical vacuum permeability mu0 [H/m].
const MU0: f64 = 1.256_637_062e-6;

fn synthetic_psi_norm(nz: usize, nr: usize) -> Array2<f64> {
    Array2::from_shape_fn((nz, nr), |(iz, ir)| {
        let z = iz as f64 / (nz - 1) as f64;
        let r = ir as f64 / (nr - 1) as f64;
        let radial = (r - 0.5).powi(2);
        let vertical = (z - 0.5).powi(2);
        (0.08 + 1.84 * (radial + vertical)).clamp(0.0, 1.0)
    })
}

fn synthetic_rr(nz: usize, nr: usize) -> Array2<f64> {
    Array2::from_shape_fn((nz, nr), |(_, ir)| 1.0 + 2.0 * ir as f64 / (nr - 1) as f64)
}

fn pprime_profile(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = i as f64 / (n - 1) as f64;
            1.0e4 * (1.0 - x).powi(2)
        })
        .collect()
}

fn ffprime_profile(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = i as f64 / (n - 1) as f64;
            -0.3 * (1.0 - 0.5 * x + 0.25 * x * x)
        })
        .collect()
}

/// Benchmark native GEQDSK pressure/FFprime source assembly on representative grids.
fn bench_geqdsk_profile_source_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("geqdsk_profile_source_components");
    group.sample_size(10);

    for &(nz, nr) in &[(33usize, 33usize), (65usize, 65usize)] {
        let psi_norm = synthetic_psi_norm(nz, nr);
        let rr = synthetic_rr(nz, nr);
        let pprime = pprime_profile(nr);
        let ffprime = ffprime_profile(nr);

        group.bench_function(format!("source_components_{nz}x{nr}"), |b| {
            b.iter(|| {
                let components = compute_geqdsk_profile_source_components(
                    black_box(&psi_norm),
                    black_box(&rr),
                    black_box(&pprime),
                    black_box(&ffprime),
                    black_box(MU0),
                )
                .expect("GEQDSK source components should assemble");
                black_box((
                    components.pressure_source_norm,
                    components.ffprime_source_norm,
                    components.total_source_norm,
                ));
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_geqdsk_profile_source_components);
criterion_main!(benches);
