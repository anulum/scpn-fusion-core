// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — MRTI Criterion Benchmark
use criterion::{criterion_group, criterion_main, Criterion};
use fusion_physics::mrti::MrtiSpectrumTracker;
use std::hint::black_box;

fn bench_mrti_tracker(c: &mut Criterion) {
    let mut group = c.benchmark_group("mrti");
    for n_modes in [64_usize, 256, 1024] {
        group.bench_function(format!("rust_{n_modes}_modes_512_steps"), |b| {
            b.iter(|| {
                let mut tracker = MrtiSpectrumTracker::new(1.0e4, n_modes, 1.0e-9, 1.0e-3, 1.0e-3)
                    .expect("valid tracker");
                for _ in 0..512 {
                    black_box(tracker.step(2.0e-7, 6.5e6, 8.0e-4).expect("valid step"));
                }
                black_box(tracker.state().max_amplitude_m)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_mrti_tracker);
criterion_main!(benches);
