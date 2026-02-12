// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — SOR Solver Benchmarks
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Criterion benchmarks for the Red-Black SOR Grad-Shafranov solver.
//!
//! Grid sizes: 33×33, 65×65, 128×128, 256×256
//! Measures: single SOR step, 50-step solve, and residual computation.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fusion_math::multigrid::{multigrid_solve, MultigridConfig};
use fusion_math::sor::{sor_residual, sor_solve, sor_step};
use fusion_types::state::Grid2D;
use ndarray::Array2;

/// ITER-like tokamak grid: R in [1, 9] m, Z in [-5, 5] m.
fn make_grid(n: usize) -> (Grid2D, Array2<f64>, Array2<f64>) {
    let grid = Grid2D::new(n, n, 1.0, 9.0, -5.0, 5.0);
    let psi = Array2::zeros((n, n));
    let source = Array2::from_elem((n, n), -1.0);
    (grid, psi, source)
}

fn bench_sor_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("sor_step");
    for n in [33, 65, 128, 256] {
        let (grid, mut psi, source) = make_grid(n);
        group.bench_with_input(BenchmarkId::new("grid", n), &n, |b, _| {
            b.iter(|| sor_step(&mut psi, &source, &grid, 1.8))
        });
    }
    group.finish();
}

fn bench_sor_solve_50(c: &mut Criterion) {
    let mut group = c.benchmark_group("sor_solve_50_iters");
    for n in [33, 65, 128] {
        let (grid, mut psi, source) = make_grid(n);
        group.bench_with_input(BenchmarkId::new("grid", n), &n, |b, _| {
            psi.fill(0.0);
            b.iter(|| sor_solve(&mut psi, &source, &grid, 1.8, 50))
        });
    }
    group.finish();
}

fn bench_sor_residual(c: &mut Criterion) {
    let mut group = c.benchmark_group("sor_residual");
    for n in [65, 128, 256] {
        let (grid, mut psi, source) = make_grid(n);
        // Pre-solve a few steps so residual computation is realistic
        sor_solve(&mut psi, &source, &grid, 1.8, 20);
        group.bench_with_input(BenchmarkId::new("grid", n), &n, |b, _| {
            b.iter(|| sor_residual(&psi, &source, &grid))
        });
    }
    group.finish();
}

fn bench_omega_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("sor_omega_sweep_128");
    let (grid, _, source) = make_grid(128);
    for omega in [1.0, 1.4, 1.6, 1.8, 1.9] {
        let mut psi = Array2::zeros((128, 128));
        let label = format!("omega_{:.1}", omega);
        group.bench_function(&label, |b| {
            psi.fill(0.0);
            b.iter(|| sor_solve(&mut psi, &source, &grid, omega, 50))
        });
    }
    group.finish();
}

fn bench_multigrid_vs_sor(c: &mut Criterion) {
    let mut group = c.benchmark_group("multigrid_vs_sor");
    let config = MultigridConfig::default();

    for n in [33, 65, 129] {
        let (grid, _, source) = make_grid(n);

        // SOR: solve to tol=1e-6 with 500 iters
        group.bench_with_input(BenchmarkId::new("sor_500", n), &n, |b, _| {
            let mut psi = Array2::zeros((n, n));
            b.iter(|| sor_solve(&mut psi, &source, &grid, 1.8, 500))
        });

        // Multigrid: 10 V-cycles
        group.bench_with_input(BenchmarkId::new("multigrid_10", n), &n, |b, _| {
            let mut psi = Array2::zeros((n, n));
            b.iter(|| multigrid_solve(&mut psi, &source, &grid, &config, 10, 1e-10))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_sor_step,
    bench_sor_solve_50,
    bench_sor_residual,
    bench_omega_sweep,
    bench_multigrid_vs_sor
);
criterion_main!(benches);
