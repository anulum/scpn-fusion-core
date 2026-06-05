// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — FRC Tilt-Mode Criterion Benchmark
use criterion::{criterion_group, criterion_main, Criterion};
use fusion_physics::compression::PulsedCompressionState;
use fusion_physics::tilt_mode_frc::{
    tilt_mode_report, tilt_mode_trajectory_from_pulsed_compression, FrcTiltCompressionReference,
    FrcTiltModeInputs, DEUTERIUM_MASS_AMU,
};
use std::hint::black_box;

fn bench_tilt_mode_report(c: &mut Criterion) {
    let mut group = c.benchmark_group("tilt_mode_frc");
    for iterations in [1_000_usize, 10_000, 100_000] {
        group.bench_function(format!("rust_{iterations}_reports"), |b| {
            b.iter(|| {
                let mut checksum = 0.0_f64;
                for index in 0..iterations {
                    let elongation = 2.5 + (index % 16) as f64 * 0.125;
                    let report = tilt_mode_report(FrcTiltModeInputs {
                        s_parameter: 8.0,
                        b_reference_t: 5.0,
                        density_peak_m3: 3.0e21,
                        r_s_m: 0.2,
                        elongation,
                        ion_mass_amu: DEUTERIUM_MASS_AMU,
                    })
                    .expect("valid report");
                    checksum += report.growth_rate_s_inv + report.s_over_elongation;
                }
                black_box(checksum)
            });
        });
    }
    group.finish();
}

fn compression_states(n_steps: usize) -> Vec<PulsedCompressionState> {
    (0..=n_steps)
        .map(|index| {
            let fraction = index as f64 / n_steps as f64;
            PulsedCompressionState {
                t_s: index as f64 * 1.0e-8,
                r_s_m: 0.2 - 0.02 * fraction,
                d_r_s_dt_m_s: -2.0e5,
                radial_acceleration_m_s2: -1.0e10,
                t_i_ev: 10_000.0 + 1_000.0 * fraction,
                t_e_ev: 5_000.0 + 500.0 * fraction,
                density_m3: 3.0e21 * (1.0 + 0.1 * fraction),
                beta: 0.5,
                b_ext_t: 5.0 + fraction,
                internal_pressure_pa: 1.0e6,
                external_magnetic_pressure_pa: 1.0e6,
                thermal_energy_j: 1.0,
                compression_work_j: 0.0,
                radiated_loss_j: 0.0,
                energy_balance_residual: 0.0,
                flux_state: Default::default(),
            }
        })
        .collect()
}

fn bench_tilt_mode_from_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("tilt_mode_frc");
    let reference = FrcTiltCompressionReference {
        s_parameter: 8.0,
        r_s_m: 0.2,
        b_reference_t: 5.0,
        t_i_ev: 10_000.0,
        elongation: 4.0,
        ion_mass_amu: DEUTERIUM_MASS_AMU,
    };
    for n_steps in [64_usize, 256] {
        let states = compression_states(n_steps);
        group.bench_function(format!("rust_fus_c6_coupled_{n_steps}_intervals"), |b| {
            b.iter(|| {
                let trajectory = tilt_mode_trajectory_from_pulsed_compression(&states, reference)
                    .expect("valid trajectory");
                let final_point = trajectory.last().expect("trajectory point exists");
                assert!(final_point.cumulative_growth_integral.is_finite());
                assert!(final_point.perturbation_amplification.is_finite());
                assert!(!final_point.amplification_overflow_limited);
                let checksum: f64 = trajectory
                    .iter()
                    .map(|point| point.report.growth_rate_s_inv + point.report.s_parameter)
                    .sum();
                black_box(checksum)
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_tilt_mode_report,
    bench_tilt_mode_from_compression
);
criterion_main!(benches);
