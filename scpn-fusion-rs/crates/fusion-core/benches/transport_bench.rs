// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Transport Solver Benchmarks
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────

use criterion::{criterion_group, criterion_main, Criterion};
use fusion_core::transport::{chang_hinton_chi, NeoclassicalParams, TransportSolver};
use ndarray::Array1;
use std::hint::black_box;

/// Build ITER-like neoclassical parameters with a monotonic q-profile over 50 radial points.
fn iter_neoclassical_params() -> NeoclassicalParams {
    // q(ρ) = 1 + 2 ρ² — monotonic from 1 at axis to 3 at the edge.
    let q_profile = Array1::from_shape_fn(50, |i| {
        let rho = i as f64 / 49.0;
        1.0 + 2.0 * rho * rho
    });
    NeoclassicalParams {
        r_major: 6.2,
        a_minor: 2.0,
        b_toroidal: 5.3,
        a_ion: 2.0, // deuterium
        z_eff: 1.7,
        q_profile,
    }
}

/// Benchmark a full transport step (update model + evolve profiles + ELM check + impurity injection).
///
/// Uses ITER-like H-mode parameters: p_aux_mw = 50 MW (above H-mode threshold of 30 MW).
/// The solver is created fresh each iteration via `iter_batched` so mutable state does not
/// accumulate across benchmark samples.
fn bench_transport_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("transport_step");

    // L-mode: single step without neoclassical model.
    group.bench_function("lmode_single_step", |b| {
        b.iter_batched(
            TransportSolver::new,
            |mut solver| {
                solver
                    .step(30.0, 0.0)
                    .expect("L-mode transport step should succeed");
                black_box(solver.profiles.te[0]);
            },
            criterion::BatchSize::SmallInput,
        )
    });

    // H-mode: single step without neoclassical model (p_aux > 30 MW triggers pedestal barrier).
    group.bench_function("hmode_single_step", |b| {
        b.iter_batched(
            TransportSolver::new,
            |mut solver| {
                solver
                    .step(50.0, 0.0)
                    .expect("H-mode transport step should succeed");
                black_box(solver.profiles.te[0]);
            },
            criterion::BatchSize::SmallInput,
        )
    });

    // H-mode with neoclassical (Chang-Hinton) χ_i replacing the constant base diffusivity.
    group.bench_function("hmode_neoclassical_single_step", |b| {
        b.iter_batched(
            || {
                let mut solver = TransportSolver::new();
                solver
                    .set_neoclassical(iter_neoclassical_params())
                    .expect("neoclassical params should be valid");
                solver
            },
            |mut solver| {
                solver
                    .step(50.0, 0.0)
                    .expect("neoclassical transport step should succeed");
                black_box(solver.profiles.te[0]);
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

/// Benchmark the standalone `chang_hinton_chi` computation across the full radial grid.
///
/// This isolates the cost of a single Chang-Hinton evaluation — 50 evaluations covering
/// ρ ∈ [0, 1] with ITER-like T_i = 10 keV, n_e = 10 × 10¹⁹ m⁻³, and the same monotonic
/// q-profile used in the full transport benchmark above.
fn bench_chang_hinton_chi(c: &mut Criterion) {
    let params = iter_neoclassical_params();

    c.bench_function("chang_hinton_chi_50pts", |b| {
        b.iter(|| {
            let mut chi_sum = 0.0_f64;
            for i in 0..50usize {
                let rho = (i as f64 + 1.0) / 50.0; // avoid rho=0 (returns floor)
                let t_i_kev = 10.0;
                let n_e_19 = 10.0;
                let q = params.q_profile[i];
                chi_sum += chang_hinton_chi(rho, t_i_kev, n_e_19, q, &params);
            }
            black_box(chi_sum);
        })
    });
}

criterion_group!(benches, bench_transport_step, bench_chang_hinton_chi);
criterion_main!(benches);
