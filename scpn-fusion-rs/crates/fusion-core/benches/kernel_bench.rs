// -------------------------------------------------------------------------
// SCPN Fusion Core -- Kernel Solver Benchmark
// Compares Picard+Jacobi (PicardSor) vs Picard+Multigrid (PicardMultigrid)
// on identical initial conditions at 33x33 and 65x65 grid resolutions.
// -------------------------------------------------------------------------

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fusion_core::kernel::{FusionKernel, SolverMethod};
use fusion_math::multigrid::MultigridConfig;
use fusion_types::config::{
    CoilConfig, GridDimensions, PhysicsParams, ReactorConfig, SolverConfig,
};
use std::hint::black_box;

/// Build a self-contained ReactorConfig at the given grid resolution.
/// Uses ITER-like geometry and coil set so benchmarks do not depend on
/// external JSON files.
fn make_config(n: usize) -> ReactorConfig {
    ReactorConfig {
        reactor_name: format!("bench-{}x{}", n, n),
        grid_resolution: [n, n],
        dimensions: GridDimensions {
            r_min: 2.0,
            r_max: 10.0,
            z_min: -6.0,
            z_max: 6.0,
        },
        physics: PhysicsParams {
            plasma_current_target: 15.0,
            vacuum_permeability: 1.0,
            profiles: None,
        },
        coils: vec![
            CoilConfig {
                name: "PF1".into(),
                r: 3.9,
                z: 7.6,
                current: 5.0,
            },
            CoilConfig {
                name: "PF2".into(),
                r: 8.2,
                z: 6.7,
                current: -1.0,
            },
            CoilConfig {
                name: "PF3".into(),
                r: 12.0,
                z: 2.7,
                current: 0.0,
            },
            CoilConfig {
                name: "PF4".into(),
                r: 12.6,
                z: -2.3,
                current: 0.0,
            },
            CoilConfig {
                name: "PF5".into(),
                r: 8.4,
                z: -6.7,
                current: -1.0,
            },
            CoilConfig {
                name: "PF6".into(),
                r: 4.3,
                z: -7.6,
                current: 8.0,
            },
            CoilConfig {
                name: "CS".into(),
                r: 1.7,
                z: 0.0,
                current: -5.0,
            },
        ],
        solver: SolverConfig {
            max_iterations: 200,
            convergence_threshold: 1e-4,
            relaxation_factor: 0.1,
        },
    }
}

fn run_solve(config: &ReactorConfig, method: SolverMethod) {
    let mut kernel = FusionKernel::new(config.clone());
    kernel.set_solver_method(method);
    if method == SolverMethod::PicardMultigrid {
        kernel.set_multigrid_config(MultigridConfig::default());
    }
    let result = kernel.solve_equilibrium().expect("solve should not error");
    black_box(result.residual);
}

fn bench_kernel_solvers(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_picard_sor_vs_multigrid");
    // These are full Picard solves; reduce sample size to keep wall time
    // reasonable.
    group.sample_size(10);

    for &n in &[33usize, 65usize] {
        let config = make_config(n);

        group.bench_with_input(
            BenchmarkId::new("PicardSor", format!("{}x{}", n, n)),
            &config,
            |b, cfg| b.iter(|| run_solve(cfg, SolverMethod::PicardSor)),
        );

        group.bench_with_input(
            BenchmarkId::new("PicardMultigrid", format!("{}x{}", n, n)),
            &config,
            |b, cfg| b.iter(|| run_solve(cfg, SolverMethod::PicardMultigrid)),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_kernel_solvers);
criterion_main!(benches);
