use criterion::{criterion_group, criterion_main, Criterion};
use fusion_math::gmres::{gmres_solve, GmresConfig};
use fusion_math::sor::sor_step;
use fusion_types::state::Grid2D;
use ndarray::Array2;
use std::hint::black_box;

fn bench_gmres_33(c: &mut Criterion) {
    let grid = Grid2D::new(33, 33, 2.0, 10.0, -6.0, 6.0);
    let source = Array2::from_elem((33, 33), -1.0);
    let config = GmresConfig::default();

    c.bench_function("gmres_33x33", |b| {
        b.iter(|| {
            let mut psi = Array2::zeros((33, 33));
            let res = gmres_solve(&mut psi, &source, &grid, &config);
            black_box(res.iterations);
        })
    });
}

fn bench_gmres_65(c: &mut Criterion) {
    let grid = Grid2D::new(65, 65, 2.0, 10.0, -6.0, 6.0);
    let source = Array2::from_elem((65, 65), -1.0);
    let config = GmresConfig::default();

    c.bench_function("gmres_65x65", |b| {
        b.iter(|| {
            let mut psi = Array2::zeros((65, 65));
            let res = gmres_solve(&mut psi, &source, &grid, &config);
            black_box(res.iterations);
        })
    });
}

fn bench_gmres_vs_sor_65(c: &mut Criterion) {
    let grid = Grid2D::new(65, 65, 2.0, 10.0, -6.0, 6.0);
    let source = Array2::from_elem((65, 65), -1.0);

    let mut group = c.benchmark_group("gmres_vs_sor_65x65");
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

    group.bench_function("gmres_30_precond", |b| {
        b.iter(|| {
            let mut psi = Array2::zeros((65, 65));
            let config = GmresConfig::default();
            let res = gmres_solve(&mut psi, &source, &grid, &config);
            black_box(res.iterations);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_gmres_33,
    bench_gmres_65,
    bench_gmres_vs_sor_65
);
criterion_main!(benches);
