// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — MRTI Criterion Benchmark
use criterion::{criterion_group, criterion_main, Criterion};
use fusion_physics::compression::{
    coil_field_t, plasma_volume_m3, run_pulsed_compression, CoilGeometry, PulsedCompressionConfig,
    PulsedCompressionState,
};
use fusion_physics::mrti::MrtiSpectrumTracker;
use std::hint::black_box;

fn compression_coil() -> CoilGeometry {
    CoilGeometry {
        n_turns: 80,
        l_coil_m: 1.0,
        r_coil_m: 0.35,
        l_inductance_h: 2.0e-6,
        r_resistance_ohm: 0.02,
        bank_voltage_max_v: 20_000.0,
    }
}

fn compression_config() -> PulsedCompressionConfig {
    PulsedCompressionConfig {
        coil: compression_coil(),
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

fn compression_initial() -> PulsedCompressionState {
    let density = 2.0e21;
    let t_i = 10_000.0;
    let t_e = 5_000.0;
    let radius = 0.2;
    let field = coil_field_t(&compression_coil(), 2.0e5).expect("valid field");
    let volume = plasma_volume_m3(radius, 1.0).expect("valid volume");
    let thermal_energy_j = 1.5 * density * volume * (t_i + t_e) * 1.602_176_634e-19;
    PulsedCompressionState {
        t_s: 0.0,
        r_s_m: radius,
        d_r_s_dt_m_s: 0.0,
        radial_acceleration_m_s2: 0.0,
        t_i_ev: t_i,
        t_e_ev: t_e,
        density_m3: density,
        beta: 0.0,
        b_ext_t: field,
        internal_pressure_pa: density * (t_i + t_e) * 1.602_176_634e-19,
        external_magnetic_pressure_pa: field * field / (2.0 * 4.0e-7 * std::f64::consts::PI),
        thermal_energy_j,
        compression_work_j: 0.0,
        radiated_loss_j: 0.0,
        energy_balance_residual: 0.0,
        flux_state: Default::default(),
    }
}

fn bench_mrti_tracker(c: &mut Criterion) {
    let mut group = c.benchmark_group("mrti");
    for n_modes in [64_usize, 256, 1024] {
        group.bench_function(format!("rust_{n_modes}_modes_512_steps"), |b| {
            b.iter(|| {
                let mut tracker = MrtiSpectrumTracker::new(1.0e4, n_modes, 1.0e-9, 1.0e-3, 1.0e-3)
                    .expect("valid tracker");
                for _ in 0..512 {
                    black_box(tracker.step(2.0e-7, 6.5e6, 8.0e-4).expect("valid step"));
                }
                let state = tracker.state();
                assert!(state.max_amplitude_m.is_finite());
                assert!(state.max_log_amplitude.is_finite());
                assert!(!state.amplitude_overflow_limited);
                black_box(state.max_amplitude_m)
            });
        });
    }
    for n_modes in [64_usize, 256] {
        group.bench_function(
            format!("rust_fus_c6_coupled_{n_modes}_modes_64_steps"),
            |b| {
                b.iter(|| {
                    let states = run_pulsed_compression(
                        compression_initial(),
                        &compression_config(),
                        5.0e5,
                        1.0e-9,
                        64,
                    )
                    .expect("valid compression states");
                    let mut tracker =
                        MrtiSpectrumTracker::new(1.0e4, n_modes, 1.0e-9, 1.0e-3, 1.0e-3)
                            .expect("valid tracker");
                    let snapshots = fusion_physics::mrti::track_mrti_from_pulsed_compression(
                        &states,
                        &mut tracker,
                        1,
                        -1.0,
                        1.0,
                    )
                    .expect("valid MRTI compression coupling");
                    let final_state = snapshots.last().expect("snapshot exists");
                    assert!(final_state.max_amplitude_m.is_finite());
                    assert!(final_state.max_log_amplitude.is_finite());
                    assert!(!final_state.amplitude_overflow_limited);
                    black_box(snapshots)
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_mrti_tracker);
criterion_main!(benches);
