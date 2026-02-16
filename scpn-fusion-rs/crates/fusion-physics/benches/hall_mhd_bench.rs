// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Hall MHD Benchmark
// © 1998–2026 Miroslav Šotek. All rights reserved.
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────

use criterion::{criterion_group, criterion_main, Criterion};
use fusion_physics::hall_mhd::HallMHD;
use std::hint::black_box;

fn bench_hall_mhd_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("hall_mhd_step");
    group.sample_size(20);

    group.bench_function("64x64_single_step", |b| {
        b.iter_batched(
            || HallMHD::new(64),
            |mut sim| {
                let (e_total, e_zonal) = sim.step();
                black_box((e_total, e_zonal));
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("64x64_100_steps", |b| {
        b.iter_batched(
            || HallMHD::new(64),
            |mut sim| {
                let history = sim.run(100);
                black_box(history.len());
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn bench_hall_mhd_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("hall_mhd_grid_scaling");
    group.sample_size(10);

    for &n in &[32, 64, 128] {
        let label = format!("{}x{}_10steps", n, n);
        group.bench_function(&label, |b| {
            b.iter_batched(
                || HallMHD::new(n),
                |mut sim| {
                    let history = sim.run(10);
                    black_box(history.len());
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

criterion_group!(benches, bench_hall_mhd_step, bench_hall_mhd_scaling);
criterion_main!(benches);
