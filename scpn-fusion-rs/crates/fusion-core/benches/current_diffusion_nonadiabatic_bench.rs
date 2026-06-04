// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Non-Adiabatic Current Diffusion Benchmark

use criterion::{criterion_group, criterion_main, Criterion};
use fusion_core::current_diffusion::{solve_flux_evolution_nonadiabatic, NonadiabaticFluxInput};
use std::hint::black_box;

fn benchmark_input(n_rho: usize, n_steps: usize) -> NonadiabaticFluxInput {
    let rho = (0..n_rho)
        .map(|index| index as f64 / (n_rho as f64 - 1.0))
        .collect::<Vec<_>>();
    let psi0 = rho
        .iter()
        .map(|value| 0.08 - 0.02 * value * value)
        .collect::<Vec<_>>();
    let n_times = n_steps + 1;
    let mut e_theta_v_m = vec![vec![0.0; n_rho]; n_times];
    let mut j_theta_a_m2 = vec![vec![0.0; n_rho]; n_times];
    for time_index in 0..n_times {
        let phase = time_index as f64 / n_times as f64;
        for rho_index in 0..n_rho {
            e_theta_v_m[time_index][rho_index] = 25.0 + phase + rho[rho_index];
            j_theta_a_m2[time_index][rho_index] = 1.0e5 * (1.0 - 0.3 * rho[rho_index]);
        }
    }
    NonadiabaticFluxInput {
        rho,
        psi0,
        tau_psi_s: vec![vec![2.0e-6; n_rho]; n_times],
        r_null_m: vec![0.18; n_times],
        e_theta_v_m,
        eta_ohm_m: vec![vec![2.5e-6; n_rho]; n_times],
        j_theta_a_m2,
        dt_s: 1.0e-8,
    }
}

fn bench_nonadiabatic_flux(c: &mut Criterion) {
    let mut group = c.benchmark_group("current_diffusion_nonadiabatic");
    for (n_rho, n_steps) in [
        (64usize, 128usize),
        (256usize, 128usize),
        (1024usize, 128usize),
    ] {
        group.bench_function(format!("rust_{n_rho}_rho_{n_steps}_steps"), |b| {
            b.iter_batched(
                || benchmark_input(n_rho, n_steps),
                |input| {
                    let trajectory = solve_flux_evolution_nonadiabatic(&input)
                        .expect("benchmark input should be valid");
                    black_box(trajectory.psi[n_steps][0]);
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, bench_nonadiabatic_flux);
criterion_main!(benches);
