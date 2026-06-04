// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — FRC Rigid-Rotor Benchmark

use criterion::{criterion_group, criterion_main, Criterion};
use fusion_physics::frc::{solve_frc_equilibrium, RigidRotorFrcInputs};
use ndarray::Array1;

fn linspace(start: f64, end: f64, n: usize) -> Array1<f64> {
    let step = (end - start) / (n as f64 - 1.0);
    Array1::from_iter((0..n).map(|idx| start + idx as f64 * step))
}

fn inputs() -> RigidRotorFrcInputs {
    RigidRotorFrcInputs {
        n0: 2.0e20,
        t_i_ev: 10_000.0,
        t_e_ev: 5_000.0,
        theta_dot: 0.0,
        r_s: 0.20,
        b_ext: 5.0,
        delta: Some(0.02),
    }
}

fn bench_frc_64(c: &mut Criterion) {
    let rho = linspace(0.0, 0.4, 64);
    let cfg = inputs();
    c.bench_function("frc_rigid_rotor_64", |b| {
        b.iter(|| std::hint::black_box(solve_frc_equilibrium(&cfg, &rho, 1.0e-10).unwrap()))
    });
}

fn bench_frc_256(c: &mut Criterion) {
    let rho = linspace(0.0, 0.4, 256);
    let cfg = inputs();
    c.bench_function("frc_rigid_rotor_256", |b| {
        b.iter(|| std::hint::black_box(solve_frc_equilibrium(&cfg, &rho, 1.0e-10).unwrap()))
    });
}

fn bench_frc_1024(c: &mut Criterion) {
    let rho = linspace(0.0, 0.4, 1024);
    let cfg = inputs();
    c.bench_function("frc_rigid_rotor_1024", |b| {
        b.iter(|| std::hint::black_box(solve_frc_equilibrium(&cfg, &rho, 1.0e-10).unwrap()))
    });
}

criterion_group!(benches, bench_frc_64, bench_frc_256, bench_frc_1024);
criterion_main!(benches);
