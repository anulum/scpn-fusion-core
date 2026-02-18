// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — SOR Benchmarks
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────

use criterion::{criterion_group, criterion_main, Criterion};
use fusion_math::sor::{sor_residual, sor_solve, sor_step};
use fusion_types::state::Grid2D;
use ndarray::Array2;
use std::hint::black_box;

/// Single Red-Black SOR step on a 33×33 grid.
///
/// Clones psi inside the closure so each iteration starts from the same
/// initial state and the in-place mutation does not accumulate across
/// Criterion's timing loop.
fn bench_sor_step_33x33(c: &mut Criterion) {
    let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
    let psi_init = Array2::zeros((33, 33));
    let source = Array2::from_elem((33, 33), -1.0);
    let omega = 1.8_f64;

    c.bench_function("sor_step_33x33", |b| {
        b.iter(|| {
            let mut psi = psi_init.clone();
            sor_step(
                &mut psi,
                black_box(&source),
                black_box(&grid),
                black_box(omega),
            );
            black_box(psi);
        })
    });
}

/// 500 SOR iterations on a 33×33 grid.
fn bench_sor_solve_33x33_500iter(c: &mut Criterion) {
    let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
    let psi_init = Array2::zeros((33, 33));
    let source = Array2::from_elem((33, 33), -1.0);
    let omega = 1.8_f64;

    c.bench_function("sor_solve_33x33_500iter", |b| {
        b.iter(|| {
            let mut psi = psi_init.clone();
            sor_solve(
                &mut psi,
                black_box(&source),
                black_box(&grid),
                black_box(omega),
                black_box(500),
            );
            black_box(psi);
        })
    });
}

/// 500 SOR iterations on a 65×65 grid.
fn bench_sor_solve_65x65_500iter(c: &mut Criterion) {
    let grid = Grid2D::new(65, 65, 1.0, 9.0, -5.0, 5.0);
    let psi_init = Array2::zeros((65, 65));
    let source = Array2::from_elem((65, 65), -1.0);
    let omega = 1.8_f64;

    c.bench_function("sor_solve_65x65_500iter", |b| {
        b.iter(|| {
            let mut psi = psi_init.clone();
            sor_solve(
                &mut psi,
                black_box(&source),
                black_box(&grid),
                black_box(omega),
                black_box(500),
            );
            black_box(psi);
        })
    });
}

/// L-infinity residual computation on a 65×65 grid.
///
/// The residual function is read-only, so no clone is needed inside the
/// timing loop.  A partially-converged psi is pre-computed outside the
/// loop so the call exercises real non-trivial data.
fn bench_sor_residual_65x65(c: &mut Criterion) {
    let grid = Grid2D::new(65, 65, 1.0, 9.0, -5.0, 5.0);
    let mut psi = Array2::zeros((65, 65));
    let source = Array2::from_elem((65, 65), -1.0);

    // Warm up psi with 200 iterations so the residual call is non-trivial.
    sor_solve(&mut psi, &source, &grid, 1.8, 200);

    c.bench_function("sor_residual_65x65", |b| {
        b.iter(|| {
            let res = sor_residual(black_box(&psi), black_box(&source), black_box(&grid));
            black_box(res);
        })
    });
}

criterion_group!(
    sor_benches,
    bench_sor_step_33x33,
    bench_sor_solve_33x33_500iter,
    bench_sor_solve_65x65_500iter,
    bench_sor_residual_65x65,
);
criterion_main!(sor_benches);
