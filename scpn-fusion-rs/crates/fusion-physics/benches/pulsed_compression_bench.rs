// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Pulsed Compression Benchmark

use criterion::{criterion_group, criterion_main, Criterion};
use fusion_physics::compression::{
    initial_pulsed_flux_state, plasma_volume_m3, pulsed_compression_trajectory_diagnostics,
    run_pulsed_compression, run_voltage_driven_pulsed_compression, CoilGeometry,
    PulsedCompressionConfig, PulsedCompressionState,
};

const ELEMENTARY_CHARGE_C: f64 = 1.602_176_634e-19;

fn config() -> PulsedCompressionConfig {
    PulsedCompressionConfig {
        coil: CoilGeometry {
            n_turns: 80,
            l_coil_m: 1.0,
            r_coil_m: 0.35,
            l_inductance_h: 2.0e-6,
            r_resistance_ohm: 0.02,
            bank_voltage_max_v: 20_000.0,
        },
        plasma_mass_kg: 2.0e-5,
        plasma_length_m: 1.0,
        gamma: 5.0 / 3.0,
        radial_loss_time_s: None,
        tau_psi_s: f64::INFINITY,
        e_theta_v_m: None,
        j_theta_a_m2: None,
        z_eff: 1.0,
        ln_lambda: 17.0,
        min_radius_m: 1.0e-4,
    }
}

fn state() -> PulsedCompressionState {
    let density = 2.0e21;
    let radius = 0.2;
    let volume = plasma_volume_m3(radius, 1.0).expect("valid volume");
    let t_i = 10_000.0;
    let t_e = 5_000.0;
    let thermal_energy = 1.5 * density * volume * (t_i + t_e) * ELEMENTARY_CHARGE_C;
    PulsedCompressionState {
        t_s: 0.0,
        r_s_m: radius,
        d_r_s_dt_m_s: 0.0,
        radial_acceleration_m_s2: 0.0,
        t_i_ev: t_i,
        t_e_ev: t_e,
        density_m3: density,
        beta: 1.0,
        b_ext_t: 0.0,
        internal_pressure_pa: density * (t_i + t_e) * ELEMENTARY_CHARGE_C,
        external_magnetic_pressure_pa: 0.0,
        thermal_energy_j: thermal_energy,
        compression_work_j: 0.0,
        radiated_loss_j: 0.0,
        energy_balance_residual: 0.0,
        flux_state: initial_pulsed_flux_state(
            (0..65).map(|index| index as f64 / 160.0).collect(),
            vec![0.0; 65],
        )
        .expect("valid flux state"),
    }
}

fn bench_pulsed_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("pulsed_compression");
    for steps in [64_usize, 256, 1024] {
        group.bench_function(format!("rust_{steps}_steps"), |b| {
            let cfg = config();
            b.iter(|| {
                std::hint::black_box({
                    let states = run_pulsed_compression(state(), &cfg, 5.0e5, 1.0e-9, steps)
                        .expect("valid run");
                    let final_state = states.last().expect("state exists");
                    assert_eq!(final_state.flux_state.budget_claim_status, "passed");
                    assert!(final_state.flux_state.update_residual_abs_max <= 1.0e-12);
                    let diagnostics =
                        pulsed_compression_trajectory_diagnostics(&states, Some(cfg.min_radius_m))
                            .expect("valid trajectory diagnostics");
                    assert!(diagnostics.monotonic_time);
                    assert!(diagnostics.all_flux_budgets_passed);
                    assert!(diagnostics.compression_ratio >= 1.0);
                    states
                })
            });
        });
    }
    for steps in [64_usize, 256] {
        group.bench_function(format!("rust_voltage_driven_{steps}_steps"), |b| {
            let cfg = config();
            b.iter(|| {
                std::hint::black_box({
                    let result = run_voltage_driven_pulsed_compression(
                        state(),
                        &cfg,
                        20_000.0,
                        1.0e-9,
                        steps,
                        5.0e5,
                    )
                    .expect("valid voltage-driven run");
                    let final_state = result.compression.last().expect("state exists");
                    assert_eq!(final_state.flux_state.budget_claim_status, "passed");
                    assert!(final_state.flux_state.update_residual_abs_max <= 1.0e-12);
                    let diagnostics = pulsed_compression_trajectory_diagnostics(
                        &result.compression,
                        Some(cfg.min_radius_m),
                    )
                    .expect("valid trajectory diagnostics");
                    assert!(diagnostics.monotonic_time);
                    assert!(diagnostics.all_flux_budgets_passed);
                    assert!(diagnostics.compression_ratio >= 1.0);
                    result
                })
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_pulsed_compression);
criterion_main!(benches);
