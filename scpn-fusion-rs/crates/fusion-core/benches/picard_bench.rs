// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Picard Iteration Benchmark
// © 1998–2026 Miroslav Šotek. All rights reserved.
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────

use criterion::{criterion_group, criterion_main, Criterion};
use fusion_core::kernel::{FusionKernel, SolverMethod};
use fusion_types::config::{
    CoilConfig, GridDimensions, PhysicsParams, ReactorConfig, SolverConfig,
};
use std::hint::black_box;

fn iter_like_config(nz: usize, nr: usize) -> ReactorConfig {
    ReactorConfig {
        reactor_name: "bench_iter".to_string(),
        grid_resolution: [nz, nr],
        dimensions: GridDimensions {
            r_min: 4.0,
            r_max: 8.4,
            z_min: -4.6,
            z_max: 4.6,
        },
        physics: PhysicsParams {
            plasma_current_target: 15e6,
            vacuum_permeability: 1.256_637_062e-6,
            profiles: None,
        },
        coils: vec![
            CoilConfig { name: "CS1".into(), r: 2.0, z: 2.0, current: 40e6 },
            CoilConfig { name: "CS2".into(), r: 2.0, z: -2.0, current: 40e6 },
            CoilConfig { name: "PF1".into(), r: 8.5, z: 5.0, current: -10e6 },
            CoilConfig { name: "PF2".into(), r: 8.5, z: -5.0, current: -10e6 },
        ],
        solver: SolverConfig {
            max_iterations: 10,
            convergence_threshold: 1e-6,
        },
    }
}

fn bench_picard_sor(c: &mut Criterion) {
    let mut group = c.benchmark_group("picard_gs_solve");
    group.sample_size(10);

    for &(nz, nr) in &[(33, 33), (65, 65)] {
        let label = format!("sor_{}x{}", nz, nr);
        group.bench_function(&label, |b| {
            b.iter(|| {
                let config = iter_like_config(nz, nr);
                let mut kernel = FusionKernel::new(config);
                let result = kernel.solve_equilibrium().expect("solve should succeed");
                black_box(result.convergence_history.len());
            })
        });
    }

    group.finish();
}

fn bench_picard_multigrid(c: &mut Criterion) {
    let mut group = c.benchmark_group("picard_multigrid_solve");
    group.sample_size(10);

    for &(nz, nr) in &[(33, 33), (65, 65)] {
        let label = format!("mg_{}x{}", nz, nr);
        group.bench_function(&label, |b| {
            b.iter(|| {
                let mut config = iter_like_config(nz, nr);
                config.solver.max_iterations = 10;
                let mut kernel = FusionKernel::new(config);
                kernel.set_solver_method(SolverMethod::PicardMultigrid);
                let result = kernel.solve_equilibrium().expect("solve should succeed");
                black_box(result.convergence_history.len());
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_picard_sor, bench_picard_multigrid);
criterion_main!(benches);
