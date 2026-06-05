// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Pulsed Hall-MHD Criterion Benchmark
use criterion::{criterion_group, criterion_main, Criterion};
use fusion_physics::hall_mhd_pulsed::{
    initial_hall_mhd_pulsed_state, step_hall_mhd_pulsed, HallMhdPulsedConfig,
};
use std::hint::black_box;

fn config(n_grid: usize) -> HallMhdPulsedConfig {
    HallMhdPulsedConfig {
        rho_m: (0..n_grid)
            .map(|i| i as f64 * 0.4 / (n_grid - 1) as f64)
            .collect(),
        tau_psi_s: 5.0e-6,
        r_null_m: 0.2,
        electron_temperature_ev: 5_000.0,
        z_eff: 1.0,
        ln_lambda: 17.0,
        hall_scale: 1.0,
    }
}

fn bench_hall_mhd_pulsed(c: &mut Criterion) {
    let mut group = c.benchmark_group("hall_mhd_pulsed");
    for n_grid in [64_usize, 256, 1024] {
        group.bench_function(format!("rust_{n_grid}_grid_256_steps"), |b| {
            b.iter(|| {
                let cfg = config(n_grid);
                let psi = cfg.rho_m.iter().map(|r| 1.0 + r).collect::<Vec<_>>();
                let e_theta = vec![2.5; n_grid];
                let j_theta = vec![2.0e5; n_grid];
                let mut state =
                    initial_hall_mhd_pulsed_state(&cfg, psi, e_theta.clone(), j_theta.clone())
                        .expect("valid state");
                for _ in 0..256 {
                    state = step_hall_mhd_pulsed(
                        &state,
                        &cfg,
                        e_theta.clone(),
                        j_theta.clone(),
                        1.0e-8,
                    )
                    .expect("valid step");
                }
                black_box(state.energy_proxy_j_m)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_hall_mhd_pulsed);
criterion_main!(benches);
