// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — FRC Rigid-Rotor Benchmark

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fusion_physics::frc::{solve_frc_equilibrium, RigidRotorFrcInputs};
use ndarray::Array1;

pub fn bench_rigid_rotor(c: &mut Criterion) {
    let mut group = c.benchmark_group("frc_rigid_rotor");

    for size in [64, 256, 1024].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &n| {
            let inputs = RigidRotorFrcInputs {
                n0: 1e20,
                t_i_ev: 10_000.0,
                t_e_ev: 5_000.0,
                theta_dot: 0.0,
                r_s: 0.20,
                b_ext: 5.0,
                delta: Some(0.02),
            };

            let step = 0.4 / (n as f64 - 1.0);
            let rho = Array1::from_iter((0..n).map(|idx| idx as f64 * step));

            b.iter(|| solve_frc_equilibrium(black_box(&inputs), black_box(&rho), black_box(1.0e-10)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_rigid_rotor);
criterion_main!(benches);
