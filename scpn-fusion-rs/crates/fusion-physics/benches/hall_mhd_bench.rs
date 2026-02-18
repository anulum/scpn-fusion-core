// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Hall MHD Benchmark
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────

use criterion::{criterion_group, criterion_main, Criterion};
use fusion_physics::hall_mhd::HallMHD;

/// Benchmark a single RK2 step on a 64×64 grid.
///
/// A fresh `HallMHD` is constructed inside the closure so that each
/// iteration starts from the same initial random state and the
/// construction cost is folded into the measurement.  For a pure
/// per-step cost use the 128 variant (larger grid, same pattern).
fn bench_hall_mhd_step_64(c: &mut Criterion) {
    c.bench_function("bench_hall_mhd_step_64", |b| {
        b.iter(|| {
            let mut mhd = HallMHD::new(64);
            std::hint::black_box(mhd.step())
        });
    });
}

/// Benchmark a single RK2 step on a 128×128 grid.
fn bench_hall_mhd_step_128(c: &mut Criterion) {
    c.bench_function("bench_hall_mhd_step_128", |b| {
        b.iter(|| {
            let mut mhd = HallMHD::new(128);
            std::hint::black_box(mhd.step())
        });
    });
}

/// Benchmark 100 consecutive RK2 steps on a 64×64 grid.
///
/// This exercises the full turbulence evolution loop and gives a
/// realistic throughput figure for production simulations.
fn bench_hall_mhd_run_100_64(c: &mut Criterion) {
    c.bench_function("bench_hall_mhd_run_100_64", |b| {
        b.iter(|| {
            let mut mhd = HallMHD::new(64);
            std::hint::black_box(mhd.run(100))
        });
    });
}

criterion_group!(
    benches,
    bench_hall_mhd_step_64,
    bench_hall_mhd_step_128,
    bench_hall_mhd_run_100_64
);
criterion_main!(benches);
