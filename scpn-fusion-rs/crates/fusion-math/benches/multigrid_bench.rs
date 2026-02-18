// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Multigrid Benchmarks
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────

use criterion::{criterion_group, criterion_main, Criterion};
use fusion_math::multigrid::{multigrid_solve, MultigridConfig};
use fusion_types::state::Grid2D;
use ndarray::Array2;
use std::hint::black_box;

/// Full multigrid V-cycle solve on a 33×33 grid.
///
/// psi is cloned inside the closure so each Criterion iteration begins
/// from the same zero initial guess and the in-place update does not
/// carry over between timing samples.
fn bench_multigrid_33x33(c: &mut Criterion) {
    let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
    let psi_init = Array2::zeros((33, 33));
    let source = Array2::from_elem((33, 33), -1.0);
    let config = MultigridConfig::default();

    c.bench_function("multigrid_33x33", |b| {
        b.iter(|| {
            let mut psi = psi_init.clone();
            let result = multigrid_solve(
                &mut psi,
                black_box(&source),
                black_box(&grid),
                black_box(&config),
                black_box(20),
                black_box(1e-8),
            );
            black_box(result.cycles);
            black_box(psi);
        })
    });
}

/// Full multigrid V-cycle solve on a 65×65 grid.
fn bench_multigrid_65x65(c: &mut Criterion) {
    let grid = Grid2D::new(65, 65, 1.0, 9.0, -5.0, 5.0);
    let psi_init = Array2::zeros((65, 65));
    let source = Array2::from_elem((65, 65), -1.0);
    let config = MultigridConfig::default();

    c.bench_function("multigrid_65x65", |b| {
        b.iter(|| {
            let mut psi = psi_init.clone();
            let result = multigrid_solve(
                &mut psi,
                black_box(&source),
                black_box(&grid),
                black_box(&config),
                black_box(20),
                black_box(1e-8),
            );
            black_box(result.cycles);
            black_box(psi);
        })
    });
}

criterion_group!(
    multigrid_benches,
    bench_multigrid_33x33,
    bench_multigrid_65x65,
);
criterion_main!(multigrid_benches);
