// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — GMRES Benchmarks
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────

use criterion::{criterion_group, criterion_main, Criterion};
use fusion_math::gmres::{gmres_solve, GmresConfig};
use fusion_types::state::Grid2D;
use ndarray::Array2;
use std::hint::black_box;

/// Full GMRES(30) solve on a 33×33 grid.
///
/// psi is cloned inside the closure so each Criterion iteration begins
/// from the same zero initial guess and the in-place update does not
/// carry over between timing samples.
fn bench_gmres_33x33(c: &mut Criterion) {
    let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
    let psi_init = Array2::zeros((33, 33));
    let source = Array2::from_elem((33, 33), -1.0);
    let config = GmresConfig::default();

    c.bench_function("gmres_33x33", |b| {
        b.iter(|| {
            let mut psi = psi_init.clone();
            let result = gmres_solve(
                &mut psi,
                black_box(&source),
                black_box(&grid),
                black_box(&config),
            );
            black_box(result.iterations);
            black_box(psi);
        })
    });
}

/// Full GMRES(30) solve on a 65×65 grid.
///
/// A custom config with a slightly larger Krylov space and more
/// preconditioner sweeps is used to ensure convergence on the coarser
/// 65×65 problem within a reasonable number of restarts.
fn bench_gmres_65x65(c: &mut Criterion) {
    let grid = Grid2D::new(65, 65, 1.0, 9.0, -5.0, 5.0);
    let psi_init = Array2::zeros((65, 65));
    let source = Array2::from_elem((65, 65), -1.0);
    let config = GmresConfig {
        restart: 40,
        max_iter: 200,
        tol: 1e-6,
        precond_sweeps: 5,
        precond_omega: 1.5,
    };

    c.bench_function("gmres_65x65", |b| {
        b.iter(|| {
            let mut psi = psi_init.clone();
            let result = gmres_solve(
                &mut psi,
                black_box(&source),
                black_box(&grid),
                black_box(&config),
            );
            black_box(result.iterations);
            black_box(psi);
        })
    });
}

criterion_group!(gmres_benches, bench_gmres_33x33, bench_gmres_65x65,);
criterion_main!(gmres_benches);
