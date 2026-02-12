// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — SOR Bench
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
use criterion::{criterion_group, criterion_main, Criterion};
use fusion_math::sor::sor_step;
use fusion_types::state::Grid2D;
use ndarray::Array2;

fn bench_sor_128(c: &mut Criterion) {
    let grid = Grid2D::new(128, 128, 1.0, 9.0, -5.0, 5.0);
    let mut psi = Array2::zeros((128, 128));
    let source = Array2::from_elem((128, 128), -1.0);

    c.bench_function("sor_step_128x128", |b| {
        b.iter(|| sor_step(&mut psi, &source, &grid, 1.8))
    });
}

fn bench_sor_65(c: &mut Criterion) {
    let grid = Grid2D::new(65, 65, 2.0, 10.0, -6.0, 6.0);
    let mut psi = Array2::zeros((65, 65));
    let source = Array2::from_elem((65, 65), -1.0);

    c.bench_function("sor_step_65x65", |b| {
        b.iter(|| sor_step(&mut psi, &source, &grid, 1.8))
    });
}

criterion_group!(benches, bench_sor_128, bench_sor_65);
criterion_main!(benches);
