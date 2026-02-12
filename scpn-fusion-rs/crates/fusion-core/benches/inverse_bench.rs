// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Inverse Jacobian Benchmark
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────

use criterion::{criterion_group, criterion_main, Criterion};
use fusion_core::inverse::{reconstruct_equilibrium, InverseConfig, JacobianMode};
use fusion_core::source::ProfileParams;
use std::hint::black_box;

fn synthetic_problem(n_points: usize) -> (Vec<f64>, Vec<f64>, ProfileParams, ProfileParams) {
    let probes: Vec<f64> = (0..n_points)
        .map(|i| i as f64 / (n_points.saturating_sub(1)) as f64)
        .collect();

    let true_p = ProfileParams {
        ped_top: 0.91,
        ped_width: 0.08,
        ped_height: 1.20,
        core_alpha: 0.30,
    };
    let true_ff = ProfileParams {
        ped_top: 0.84,
        ped_width: 0.06,
        ped_height: 0.90,
        core_alpha: 0.15,
    };

    let measurements: Vec<f64> = probes
        .iter()
        .map(|&psi| {
            fusion_core::source::mtanh_profile(psi, &true_p)
                + fusion_core::source::mtanh_profile(psi, &true_ff)
        })
        .collect();

    let init_p = ProfileParams {
        ped_top: 0.75,
        ped_width: 0.12,
        ped_height: 0.70,
        core_alpha: 0.05,
    };
    let init_ff = ProfileParams {
        ped_top: 0.70,
        ped_width: 0.10,
        ped_height: 0.60,
        core_alpha: 0.02,
    };

    (probes, measurements, init_p, init_ff)
}

fn run_inverse(mode: JacobianMode, n_points: usize) {
    let (probes, measurements, init_p, init_ff) = synthetic_problem(n_points);
    let config = InverseConfig {
        jacobian_mode: mode,
        max_iterations: 20,
        tolerance: 1e-8,
        damping: 0.7,
        fd_step: 1e-6,
        tikhonov: 1e-4,
    };

    let result = reconstruct_equilibrium(&probes, &measurements, init_p, init_ff, &config)
        .expect("inverse benchmark reconstruction should succeed");
    black_box(result.residual);
}

fn bench_inverse_fd_vs_analytical(c: &mut Criterion) {
    let mut group = c.benchmark_group("inverse_fd_vs_analytical");
    group.sample_size(10);

    for n_points in [33usize, 65usize] {
        group.bench_function(format!("fd_{}x{}", n_points, n_points), |b| {
            b.iter(|| run_inverse(JacobianMode::FiniteDifference, n_points))
        });
        group.bench_function(format!("analytical_{}x{}", n_points, n_points), |b| {
            b.iter(|| run_inverse(JacobianMode::Analytical, n_points))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_inverse_fd_vs_analytical);
criterion_main!(benches);
