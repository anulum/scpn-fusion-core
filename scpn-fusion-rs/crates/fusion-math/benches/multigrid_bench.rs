use criterion::{criterion_group, criterion_main, Criterion};
use fusion_math::gmres::{gmres_solve, GmresConfig};
use fusion_math::multigrid::{multigrid_solve, MultigridConfig};
use fusion_math::sor::sor_step;
use fusion_types::state::Grid2D;
use ndarray::Array2;
use std::hint::black_box;

fn bench_multigrid_33(c: &mut Criterion) {
    let grid = Grid2D::new(33, 33, 2.0, 10.0, -6.0, 6.0);
    let source = Array2::from_elem((33, 33), -1.0);
    let config = MultigridConfig::default();

    c.bench_function("multigrid_33x33", |b| {
        b.iter(|| {
            let mut psi = Array2::zeros((33, 33));
            let res = multigrid_solve(&mut psi, &source, &grid, &config, 20, 1e-8);
            black_box(res.cycles);
        })
    });
}

fn bench_multigrid_65(c: &mut Criterion) {
    let grid = Grid2D::new(65, 65, 2.0, 10.0, -6.0, 6.0);
    let source = Array2::from_elem((65, 65), -1.0);
    let config = MultigridConfig::default();

    c.bench_function("multigrid_65x65", |b| {
        b.iter(|| {
            let mut psi = Array2::zeros((65, 65));
            let res = multigrid_solve(&mut psi, &source, &grid, &config, 20, 1e-8);
            black_box(res.cycles);
        })
    });
}

fn bench_multigrid_129(c: &mut Criterion) {
    let grid = Grid2D::new(129, 129, 2.0, 10.0, -6.0, 6.0);
    let source = Array2::from_elem((129, 129), -1.0);
    let config = MultigridConfig::default();

    c.bench_function("multigrid_129x129", |b| {
        b.iter(|| {
            let mut psi = Array2::zeros((129, 129));
            let res = multigrid_solve(&mut psi, &source, &grid, &config, 20, 1e-8);
            black_box(res.cycles);
        })
    });
}

fn bench_all_solvers_65(c: &mut Criterion) {
    let grid = Grid2D::new(65, 65, 2.0, 10.0, -6.0, 6.0);
    let source = Array2::from_elem((65, 65), -1.0);

    let mut group = c.benchmark_group("all_solvers_65x65");
    group.sample_size(10);

    group.bench_function("sor_200iters", |b| {
        b.iter(|| {
            let mut psi = Array2::zeros((65, 65));
            for _ in 0..200 {
                sor_step(&mut psi, &source, &grid, 1.8);
            }
            black_box(psi[[32, 32]]);
        })
    });

    group.bench_function("gmres_30", |b| {
        b.iter(|| {
            let mut psi = Array2::zeros((65, 65));
            let config = GmresConfig::default();
            let res = gmres_solve(&mut psi, &source, &grid, &config);
            black_box(res.iterations);
        })
    });

    group.bench_function("multigrid_v33", |b| {
        b.iter(|| {
            let mut psi = Array2::zeros((65, 65));
            let config = MultigridConfig::default();
            let res = multigrid_solve(&mut psi, &source, &grid, &config, 20, 1e-8);
            black_box(res.cycles);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_multigrid_33,
    bench_multigrid_65,
    bench_multigrid_129,
    bench_all_solvers_65
);
criterion_main!(benches);
