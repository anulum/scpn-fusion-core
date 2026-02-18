// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Vacuum Field Benchmarks
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────

use criterion::{criterion_group, criterion_main, Criterion};
use fusion_core::vacuum::calculate_vacuum_field;
use fusion_types::config::CoilConfig;
use fusion_types::state::Grid2D;
use std::hint::black_box;

/// ITER-like 6-coil set covering the poloidal cross-section.
fn iter_coils_6() -> Vec<CoilConfig> {
    vec![
        CoilConfig {
            name: "CS1".into(),
            r: 2.0,
            z: 2.0,
            current: 40e6,
        },
        CoilConfig {
            name: "CS2".into(),
            r: 2.0,
            z: -2.0,
            current: 40e6,
        },
        CoilConfig {
            name: "PF1".into(),
            r: 8.5,
            z: 5.0,
            current: -10e6,
        },
        CoilConfig {
            name: "PF2".into(),
            r: 8.5,
            z: -5.0,
            current: -10e6,
        },
        CoilConfig {
            name: "PF3".into(),
            r: 4.0,
            z: 7.5,
            current: -5e6,
        },
        CoilConfig {
            name: "PF4".into(),
            r: 4.0,
            z: -7.5,
            current: -5e6,
        },
    ]
}

/// Physical vacuum permeability μ₀ [H/m].
const MU0: f64 = 1.256_637_062e-6;

/// Benchmark vacuum field computation from 6 ITER-like coils on a 33x33 grid.
///
/// `Grid2D::new(nr, nz, r_min, r_max, z_min, z_max)` — the ITER cross-section spans
/// R ∈ [4.0, 8.4] m, Z ∈ [-4.6, 4.6] m.
fn bench_vacuum_field_33x33_6coils(c: &mut Criterion) {
    let grid = Grid2D::new(33, 33, 4.0, 8.4, -4.6, 4.6);
    let coils = iter_coils_6();

    c.bench_function("vacuum_field_33x33_6coils", |b| {
        b.iter(|| {
            let psi = calculate_vacuum_field(&grid, &coils, MU0)
                .expect("vacuum field should succeed on 33x33");
            // Prevent the result from being optimised away.
            black_box(psi[[16, 16]]);
        })
    });
}

/// Benchmark vacuum field computation from 6 ITER-like coils on a 65x65 grid.
fn bench_vacuum_field_65x65_6coils(c: &mut Criterion) {
    let grid = Grid2D::new(65, 65, 4.0, 8.4, -4.6, 4.6);
    let coils = iter_coils_6();

    c.bench_function("vacuum_field_65x65_6coils", |b| {
        b.iter(|| {
            let psi = calculate_vacuum_field(&grid, &coils, MU0)
                .expect("vacuum field should succeed on 65x65");
            black_box(psi[[32, 32]]);
        })
    });
}

criterion_group!(
    benches,
    bench_vacuum_field_33x33_6coils,
    bench_vacuum_field_65x65_6coils
);
criterion_main!(benches);
