// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Picard Iteration Benchmarks
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────

use criterion::{criterion_group, criterion_main, Criterion};
use fusion_core::kernel::{FusionKernel, SolverMethod};
use fusion_types::config::{
    CoilConfig, GridDimensions, PhysicsParams, ReactorConfig, SolverConfig,
};
use std::hint::black_box;

/// Build an ITER-like ReactorConfig with the given grid dimensions.
///
/// The coil set (4 coils) is sufficient to produce a non-trivial vacuum field
/// that seeds the Picard iteration. `max_iterations` is kept small (10) so the
/// benchmark measures per-iteration cost rather than full convergence time.
fn iter_like_config(nz: usize, nr: usize) -> ReactorConfig {
    ReactorConfig {
        reactor_name: "bench-iter-like".to_string(),
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
            CoilConfig {
                name: "CS1".into(),
                r: 2.0,
                z: 2.0,
                current: 40e6,
            },
            CoilConfig {
                name: "CS2".into(),
                r: 2.0,
                z: -2.0,
                current: 40e6,
            },
            CoilConfig {
                name: "PF1".into(),
                r: 8.5,
                z: 5.0,
                current: -10e6,
            },
            CoilConfig {
                name: "PF2".into(),
                r: 8.5,
                z: -5.0,
                current: -10e6,
            },
        ],
        solver: SolverConfig {
            max_iterations: 10,
            convergence_threshold: 1e-6,
            relaxation_factor: 1.0,
        },
    }
}

/// Benchmark FusionKernel::solve_equilibrium with SOR on a 33x33 grid.
fn bench_picard_sor_33x33(c: &mut Criterion) {
    let mut group = c.benchmark_group("picard_gs_solve");
    group.sample_size(10);

    group.bench_function("sor_33x33", |b| {
        b.iter(|| {
            let config = iter_like_config(33, 33);
            let mut kernel = FusionKernel::new(config);
            let result = kernel
                .solve_equilibrium()
                .expect("SOR solve should succeed on 33x33");
            black_box(result.iterations);
        })
    });

    group.finish();
}

/// Benchmark FusionKernel::solve_equilibrium with multigrid on a 33x33 grid.
fn bench_picard_multigrid_33x33(c: &mut Criterion) {
    let mut group = c.benchmark_group("picard_multigrid_solve");
    group.sample_size(10);

    group.bench_function("multigrid_33x33", |b| {
        b.iter(|| {
            let config = iter_like_config(33, 33);
            let mut kernel = FusionKernel::new(config);
            kernel.set_solver_method(SolverMethod::PicardMultigrid);
            let result = kernel
                .solve_equilibrium()
                .expect("multigrid solve should succeed on 33x33");
            black_box(result.iterations);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_picard_sor_33x33,
    bench_picard_multigrid_33x33
);
criterion_main!(benches);
