// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Transport Solver Benchmark
// © 1998–2026 Miroslav Šotek. All rights reserved.
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────

use criterion::{criterion_group, criterion_main, Criterion};
use fusion_core::transport::{NeoclassicalParams, TransportSolver};
use ndarray::Array1;
use std::hint::black_box;

fn bench_transport_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("transport_step");

    group.bench_function("lmode_single_step", |b| {
        b.iter_batched(
            || TransportSolver::new(),
            |mut solver| {
                solver.step(30.0, 0.0).expect("step should succeed");
                black_box(solver.profiles.te[0]);
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("lmode_100_steps", |b| {
        b.iter_batched(
            || TransportSolver::new(),
            |mut solver| {
                for _ in 0..100 {
                    solver.step(30.0, 0.0).expect("step should succeed");
                }
                black_box(solver.profiles.te[0]);
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("hmode_nc_single_step", |b| {
        b.iter_batched(
            || {
                let mut solver = TransportSolver::new();
                solver.neoclassical = Some(NeoclassicalParams {
                    r_major: 6.2,
                    a_minor: 2.0,
                    b_toroidal: 5.3,
                    a_ion: 2.0,
                    z_eff: 1.7,
                    q_profile: Array1::linspace(1.0, 4.0, 50),
                });
                solver
            },
            |mut solver| {
                solver.step(50.0, 0.0).expect("step should succeed");
                black_box(solver.profiles.te[0]);
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(benches, bench_transport_step);
criterion_main!(benches);
