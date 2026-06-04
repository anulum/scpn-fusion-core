// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — FRC Tilt-Mode Criterion Benchmark
use criterion::{criterion_group, criterion_main, Criterion};
use fusion_physics::tilt_mode_frc::{tilt_mode_report, FrcTiltModeInputs, DEUTERIUM_MASS_AMU};
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

criterion_group!(benches, bench_tilt_mode_report);
criterion_main!(benches);
