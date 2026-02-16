// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Vacuum Field Benchmark
// © 1998–2026 Miroslav Šotek. All rights reserved.
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────

use criterion::{criterion_group, criterion_main, Criterion};
use fusion_core::vacuum::calculate_vacuum_field;
use fusion_types::config::CoilConfig;
use fusion_types::state::Grid2D;
use std::hint::black_box;

fn make_grid(nz: usize, nr: usize) -> Grid2D {
    Grid2D::new(nr, nz, 4.0, 8.4, -4.6, 4.6)
}

fn iter_coils() -> Vec<CoilConfig> {
    vec![
        CoilConfig { name: "CS1".into(), r: 2.0, z: 2.0, current: 40e6 },
        CoilConfig { name: "CS2".into(), r: 2.0, z: -2.0, current: 40e6 },
        CoilConfig { name: "PF1".into(), r: 8.5, z: 5.0, current: -10e6 },
        CoilConfig { name: "PF2".into(), r: 8.5, z: -5.0, current: -10e6 },
        CoilConfig { name: "PF3".into(), r: 4.0, z: 7.5, current: -5e6 },
        CoilConfig { name: "PF4".into(), r: 4.0, z: -7.5, current: -5e6 },
    ]
}

fn bench_vacuum_field(c: &mut Criterion) {
    let mu0 = 1.256_637_062e-6;
    let coils = iter_coils();
    let mut group = c.benchmark_group("vacuum_field");

    for &(nz, nr) in &[(33, 33), (65, 65), (129, 129)] {
        let grid = make_grid(nz, nr);
        let label = format!("{}x{}_{}coils", nz, nr, coils.len());
        group.bench_function(&label, |b| {
            b.iter(|| {
                let psi = calculate_vacuum_field(&grid, &coils, mu0)
                    .expect("vacuum field should succeed");
                black_box(psi[[nz / 2, nr / 2]]);
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_vacuum_field);
criterion_main!(benches);
