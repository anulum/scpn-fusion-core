// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Inverse Reconstruction Benchmarks
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────

use criterion::{criterion_group, criterion_main, Criterion};
use fusion_core::inverse::{reconstruct_equilibrium, InverseConfig, JacobianMode};
use fusion_core::jacobian::forward_model_response;
use fusion_core::source::ProfileParams;
use std::hint::black_box;

/// Build 60 equally-spaced probe positions in [0, 1].
fn make_probes_60() -> Vec<f64> {
    (0..60).map(|i| i as f64 / 59.0).collect()
}

/// "True" profile parameters used to generate synthetic measurements.
fn true_params_p() -> ProfileParams {
    ProfileParams {
        ped_top: 0.91,
        ped_width: 0.08,
        ped_height: 1.20,
        core_alpha: 0.30,
    }
}

fn true_params_ff() -> ProfileParams {
    ProfileParams {
        ped_top: 0.84,
        ped_width: 0.06,
        ped_height: 0.90,
        core_alpha: 0.15,
    }
}

/// Initial guess intentionally offset from the truth.
fn init_params_p() -> ProfileParams {
    ProfileParams {
        ped_top: 0.75,
        ped_width: 0.12,
        ped_height: 0.70,
        core_alpha: 0.05,
    }
}

fn init_params_ff() -> ProfileParams {
    ProfileParams {
        ped_top: 0.70,
        ped_width: 0.10,
        ped_height: 0.60,
        core_alpha: 0.02,
    }
}

/// Benchmark `reconstruct_equilibrium` using the Analytical Jacobian on 60 probes.
///
/// Measurements are generated via `forward_model_response` (the same forward model
/// used internally by the inverse solver), ensuring the problem is self-consistent.
fn bench_inverse_reconstruct_analytic(c: &mut Criterion) {
    let probes = make_probes_60();
    let measurements =
        forward_model_response(&probes, &true_params_p(), &true_params_ff()).unwrap();

    let config = InverseConfig {
        jacobian_mode: JacobianMode::Analytical,
        max_iterations: 40,
        tolerance: 1e-6,
        damping: 0.6,
        fd_step: 1e-6,
        tikhonov: 1e-4,
    };

    c.bench_function("inverse_reconstruct_analytic_60probes", |b| {
        b.iter(|| {
            let result = reconstruct_equilibrium(
                &probes,
                &measurements,
                init_params_p(),
                init_params_ff(),
                &config,
            )
            .expect("inverse benchmark reconstruction should succeed");
            black_box(result.residual);
        })
    });
}

/// Benchmark comparing FiniteDifference vs Analytical Jacobian on 60 probes.
fn bench_inverse_fd_vs_analytical(c: &mut Criterion) {
    let probes = make_probes_60();
    let measurements =
        forward_model_response(&probes, &true_params_p(), &true_params_ff()).unwrap();

    let config_fd = InverseConfig {
        jacobian_mode: JacobianMode::FiniteDifference,
        max_iterations: 20,
        tolerance: 1e-8,
        damping: 0.7,
        fd_step: 1e-6,
        tikhonov: 1e-4,
    };

    let config_an = InverseConfig {
        jacobian_mode: JacobianMode::Analytical,
        ..config_fd.clone()
    };

    let mut group = c.benchmark_group("inverse_fd_vs_analytical");
    group.sample_size(10);

    group.bench_function("finite_difference_60probes", |b| {
        b.iter(|| {
            let result = reconstruct_equilibrium(
                &probes,
                &measurements,
                init_params_p(),
                init_params_ff(),
                &config_fd,
            )
            .expect("fd inverse benchmark reconstruction should succeed");
            black_box(result.residual);
        })
    });

    group.bench_function("analytical_60probes", |b| {
        b.iter(|| {
            let result = reconstruct_equilibrium(
                &probes,
                &measurements,
                init_params_p(),
                init_params_ff(),
                &config_an,
            )
            .expect("analytical inverse benchmark reconstruction should succeed");
            black_box(result.residual);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_inverse_reconstruct_analytic,
    bench_inverse_fd_vs_analytical
);
criterion_main!(benches);
