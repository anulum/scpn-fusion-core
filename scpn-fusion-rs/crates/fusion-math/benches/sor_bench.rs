use criterion::{criterion_group, criterion_main, Criterion};
use fusion_math::chebyshev::{chebyshev_sor_solve, ChebyshevConfig};
use fusion_math::sor::sor_step;
use fusion_types::state::Grid2D;
use ndarray::Array2;
use std::hint::black_box;

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

fn bench_chebyshev_vs_fixed_sor(c: &mut Criterion) {
    let grid = Grid2D::new(129, 129, 1.0, 9.0, -5.0, 5.0);
    let source = Array2::from_elem((129, 129), -1.0);

    let mut group = c.benchmark_group("sor_vs_chebyshev_129x129");
    group.sample_size(10);

    group.bench_function("fixed_sor_step_200iters", |b| {
        b.iter(|| {
            let mut psi = Array2::zeros((129, 129));
            for _ in 0..200 {
                sor_step(&mut psi, &source, &grid, 1.8);
            }
            black_box(psi[[64, 64]]);
        })
    });

    group.bench_function("chebyshev_sor_200iters", |b| {
        b.iter(|| {
            let mut psi = Array2::zeros((129, 129));
            let _ = chebyshev_sor_solve(
                &mut psi,
                &source,
                &grid,
                ChebyshevConfig {
                    warmup_iters: 3,
                    max_iters: 200,
                    tol: 0.0,
                },
            );
            black_box(psi[[64, 64]]);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_sor_128,
    bench_sor_65,
    bench_chebyshev_vs_fixed_sor
);
criterion_main!(benches);
