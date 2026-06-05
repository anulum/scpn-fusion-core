// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Faraday Recovery Benchmark

use criterion::{criterion_group, criterion_main, Criterion};
use fusion_physics::compression::{
    plasma_volume_m3, run_pulsed_compression, run_voltage_driven_pulsed_compression, CoilGeometry,
    PulsedCompressionConfig, PulsedCompressionState,
};
use fusion_physics::faraday_recovery::{
    coil_source_work_from_voltage_driven_compression,
    compression_flux_budget_from_pulsed_compression,
    compression_flux_budget_from_voltage_driven_compression,
    compression_work_from_pulsed_compression, compression_work_from_voltage_driven_compression,
    faraday_trajectory_from_pulsed_compression, faraday_trajectory_from_voltage_driven_compression,
    integrated_recovery_energy, FaradayRecoveryTrajectoryPoint,
};

fn trajectory(samples: usize) -> Vec<FaradayRecoveryTrajectoryPoint> {
    let duration = 1.0e-6;
    (0..samples)
        .map(|index| {
            let t_s = duration * index as f64 / (samples as f64 - 1.0);
            FaradayRecoveryTrajectoryPoint {
                t_s,
                separatrix_radius_m: 0.15 + 4.0e3 * t_s,
                b_ext_t: 20.0,
                d_radius_dt_m_s: None,
                d_b_ext_dt_t_s: None,
            }
        })
        .collect()
}

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
    let field = fusion_physics::compression::coil_field_t(&compression_coil(), 2.0e5).unwrap();
    let volume = plasma_volume_m3(radius, 1.0).unwrap();
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

fn bench_faraday_recovery(c: &mut Criterion) {
    let mut group = c.benchmark_group("faraday_recovery");
    for samples in [64_usize, 256, 1024] {
        let trace = trajectory(samples);
        group.bench_function(format!("rust_{samples}_samples"), |b| {
            b.iter(|| {
                let report = integrated_recovery_energy(&trace, 6, 0.08, None, None, None, 0.01)
                    .expect("valid report");
                assert!(report.flux_derivative_closure_passed);
                std::hint::black_box(report)
            });
        });
    }
    for steps in [64_usize, 256] {
        group.bench_function(format!("rust_fus_c6_coupled_{steps}_steps"), |b| {
            b.iter(|| {
                let states = run_pulsed_compression(
                    compression_initial(),
                    &compression_config(),
                    5.0e5,
                    1.0e-9,
                    steps,
                )
                .expect("valid compression states");
                let trajectory =
                    faraday_trajectory_from_pulsed_compression(&states).expect("valid trajectory");
                let work = compression_work_from_pulsed_compression(&states)
                    .expect("positive compression work");
                let flux_budget = compression_flux_budget_from_pulsed_compression(&states)
                    .expect("valid flux budget");
                let report = integrated_recovery_energy(
                    &trajectory,
                    80,
                    0.02,
                    Some(work),
                    None,
                    Some(flux_budget),
                    0.01,
                )
                .expect("valid report");
                assert!(report.flux_derivative_residual_linf.is_finite());
                std::hint::black_box(report)
            });
        });
    }
    for steps in [64_usize, 256] {
        group.bench_function(format!("rust_fus_c6_voltage_driven_{steps}_steps"), |b| {
            b.iter(|| {
                let result = run_voltage_driven_pulsed_compression(
                    compression_initial(),
                    &compression_config(),
                    20_000.0,
                    1.0e-9,
                    steps,
                    5.0e5,
                )
                .expect("valid voltage-driven compression result");
                let trajectory = faraday_trajectory_from_voltage_driven_compression(&result)
                    .expect("valid trajectory");
                let compression_work = compression_work_from_voltage_driven_compression(&result)
                    .expect("positive compression work");
                let source_work = coil_source_work_from_voltage_driven_compression(&result)
                    .expect("positive source work");
                let flux_budget = compression_flux_budget_from_voltage_driven_compression(&result)
                    .expect("valid flux budget");
                let report = integrated_recovery_energy(
                    &trajectory,
                    80,
                    0.02,
                    Some(compression_work),
                    Some(source_work),
                    Some(flux_budget),
                    0.01,
                )
                .expect("valid report");
                assert!(report.flux_derivative_residual_linf.is_finite());
                std::hint::black_box(report)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_faraday_recovery);
criterion_main!(benches);
