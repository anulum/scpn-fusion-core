// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — RMF Phase-Lock Benchmark

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fusion_physics::rmf_control::{RmfConfig, RmfPhaseLockController};
use std::hint::black_box;

pub fn bench_rmf_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmf_phase_lock");

    let cfg = RmfConfig::default();
    let mut ctrl = RmfPhaseLockController::new(cfg);

    group.bench_function("step", |b| b.iter(|| ctrl.step(black_box(0.1))));

    for horizon in [100, 1000, 10000].iter() {
        let traj = vec![0.1; *horizon];
        group.bench_with_input(
            BenchmarkId::new("step_horizon", horizon),
            horizon,
            |b, _| b.iter(|| ctrl.step_horizon(black_box(&traj))),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_rmf_step);
criterion_main!(benches);
