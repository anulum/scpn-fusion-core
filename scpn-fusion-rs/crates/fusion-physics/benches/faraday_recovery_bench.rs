// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Faraday Recovery Benchmark

use criterion::{criterion_group, criterion_main, Criterion};
use fusion_physics::faraday_recovery::{
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

fn bench_faraday_recovery(c: &mut Criterion) {
    let mut group = c.benchmark_group("faraday_recovery");
    for samples in [64_usize, 256, 1024] {
        let trace = trajectory(samples);
        group.bench_function(format!("rust_{samples}_samples"), |b| {
            b.iter(|| {
                std::hint::black_box(
                    integrated_recovery_energy(&trace, 6, 0.08, None, 0.01).expect("valid report"),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_faraday_recovery);
criterion_main!(benches);
