// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core - GPU SOR Benchmarks

use criterion::{criterion_group, criterion_main, Criterion};
use fusion_gpu::GpuGsSolver;
use fusion_math::sor::sor_solve;
use fusion_types::state::Grid2D;
use ndarray::Array2;
use std::hint::black_box;

const ITERATIONS: usize = 20;
const OMEGA: f64 = 1.3;

fn source_value(ir: usize, iz: usize, nr: usize, nz: usize) -> f64 {
    if ir == 0 || iz == 0 || ir + 1 == nr || iz + 1 == nz {
        return 0.0;
    }
    let r = ir as f64 / (nr - 1) as f64;
    let z = iz as f64 / (nz - 1) as f64;
    -((std::f64::consts::PI * r).sin() * (std::f64::consts::PI * z).sin()).abs()
}

fn make_cpu_case(nr: usize, nz: usize) -> (Grid2D, Array2<f64>, Array2<f64>) {
    let grid = Grid2D::new(nr, nz, 1.0, 3.0, -1.2, 1.2);
    let psi = Array2::zeros((nz, nr));
    let source = Array2::from_shape_fn((nz, nr), |(iz, ir)| source_value(ir, iz, nr, nz));
    (grid, psi, source)
}

fn make_gpu_case(nr: usize, nz: usize) -> (Vec<f32>, Vec<f32>) {
    let psi = vec![0.0_f32; nr * nz];
    let source = (0..nz)
        .flat_map(|iz| (0..nr).map(move |ir| source_value(ir, iz, nr, nz) as f32))
        .collect();
    (psi, source)
}

fn bench_cpu_sor(c: &mut Criterion, nr: usize, nz: usize) {
    let (grid, psi_init, source) = make_cpu_case(nr, nz);
    let name = format!("cpu_sor_solve_{nr}x{nz}_{ITERATIONS}iter");
    c.bench_function(&name, |b| {
        b.iter(|| {
            let mut psi = psi_init.clone();
            sor_solve(
                &mut psi,
                black_box(&source),
                black_box(&grid),
                black_box(OMEGA),
                black_box(ITERATIONS),
            );
            black_box(psi);
        })
    });
}

fn bench_gpu_sor_full(c: &mut Criterion, nr: usize, nz: usize) {
    let solver = GpuGsSolver::new(nr, nz, 1.0, 3.0, -1.2, 1.2)
        .unwrap_or_else(|err| panic!("GPU solver unavailable for benchmark: {err:?}"));
    let (psi, source) = make_gpu_case(nr, nz);

    let name = format!("gpu_sor_solve_full_{nr}x{nz}_{ITERATIONS}iter");
    c.bench_function(&name, |b| {
        b.iter(|| {
            let result = solver
                .solve_full(
                    black_box(&psi),
                    black_box(&source),
                    black_box(ITERATIONS),
                    black_box(OMEGA as f32),
                )
                .expect("GPU SOR solve_full benchmark failed");
            black_box(result);
        })
    });
}

fn bench_apples_to_apples(c: &mut Criterion) {
    for (nr, nz) in [(33, 33), (65, 65), (129, 129)] {
        bench_cpu_sor(c, nr, nz);
        bench_gpu_sor_full(c, nr, nz);
    }
}

criterion_group!(gpu_sor_benches, bench_apples_to_apples);
criterion_main!(gpu_sor_benches);
