// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Full Runaway Kinetic Benchmark

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use fusion_physics::runaway_kinetic::{
    RunawayKineticCoefficients, RunawayKineticGrid, RunawayKineticOperator,
};

fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    (0..count)
        .map(|index| start + (stop - start) * index as f64 / (count - 1) as f64)
        .collect()
}

fn fixture() -> (RunawayKineticOperator, Vec<f64>, Vec<f64>) {
    let grid = RunawayKineticGrid::new(
        linspace(0.0, 1.0, 9),
        linspace(-1.0, 1.0, 17),
        linspace(0.0, 20.0, 33),
    )
    .unwrap();
    let radial_faces = (grid.nr() + 1) * grid.nxi() * grid.np();
    let momentum_faces = grid.nr() * grid.nxi() * (grid.np() + 1);
    let pitch_faces = grid.nr() * (grid.nxi() + 1) * grid.np();
    let cells = grid.cell_count();
    let coefficients = RunawayKineticCoefficients::new(
        &grid,
        vec![1.0e-3; radial_faces],
        vec![0.3; momentum_faces],
        vec![-0.2; momentum_faces],
        vec![-0.02; momentum_faces],
        vec![-0.01; momentum_faces],
        vec![0.03; pitch_faces],
        vec![-0.01; pitch_faces],
        vec![1.0e-4; radial_faces],
        vec![2.0e-3; momentum_faces],
        vec![1.0e-3; pitch_faces],
        vec![2.0e-4; momentum_faces],
        vec![2.0e-4; pitch_faces],
        vec![1.0e-25; cells],
        vec![1.0e19; grid.nr()],
        vec![2.0; grid.nr()],
        vec![1.0e8; grid.nr()],
        vec![1.0e7; cells],
    )
    .unwrap();
    let state: Vec<f64> = (0..cells)
        .map(|index| 1.0e10 * (-(index as f64) / cells as f64).exp())
        .collect();
    let density = vec![1.0e12; grid.nr()];
    (
        RunawayKineticOperator::new(grid, coefficients),
        state,
        density,
    )
}

fn benchmark_operator(c: &mut Criterion) {
    let (operator, state, density) = fixture();
    c.bench_function("runaway_kinetic_operator_8x16x32", |bencher| {
        bencher.iter(|| {
            operator
                .evaluate(black_box(&state), Some(black_box(&density)))
                .unwrap()
        });
    });
}

criterion_group!(benches, benchmark_operator);
criterion_main!(benches);
