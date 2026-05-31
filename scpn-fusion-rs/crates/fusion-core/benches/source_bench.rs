// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — GEQDSK Profile Source Benchmarks

use criterion::{criterion_group, criterion_main, Criterion};
use fusion_core::kernel::{
    total_toroidal_current_from_flux, total_toroidal_current_from_flux_masked,
};
use fusion_core::source::{
    compute_geqdsk_profile_source_components, interpolate_flux_profile_current_conserving,
    interpolate_flux_profile_second_order, select_geqdsk_source_convention_adapter,
};
use fusion_types::state::Grid2D;
use ndarray::Array2;
use std::hint::black_box;

/// Physical vacuum permeability mu0 [H/m].
const MU0: f64 = 1.256_637_062e-6;

fn synthetic_psi_norm(nz: usize, nr: usize) -> Array2<f64> {
    Array2::from_shape_fn((nz, nr), |(iz, ir)| {
        let z = iz as f64 / (nz - 1) as f64;
        let r = ir as f64 / (nr - 1) as f64;
        let radial = (r - 0.5).powi(2);
        let vertical = (z - 0.5).powi(2);
        (0.08 + 1.84 * (radial + vertical)).clamp(0.0, 1.0)
    })
}

fn synthetic_rr(nz: usize, nr: usize) -> Array2<f64> {
    Array2::from_shape_fn((nz, nr), |(_, ir)| 1.0 + 2.0 * ir as f64 / (nr - 1) as f64)
}

fn pprime_profile(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = i as f64 / (n - 1) as f64;
            1.0e4 * (1.0 - x).powi(2)
        })
        .collect()
}

fn ffprime_profile(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = i as f64 / (n - 1) as f64;
            -0.3 * (1.0 - 0.5 * x + 0.25 * x * x)
        })
        .collect()
}

fn synthetic_profile_source(nz: usize, nr: usize) -> Array2<f64> {
    Array2::from_shape_fn((nz, nr), |(iz, ir)| {
        let z = iz as f64 / (nz - 1) as f64;
        let r = ir as f64 / (nr - 1) as f64;
        let envelope =
            (1.0 - (2.0 * r - 1.0).powi(2)).max(0.0) * (1.0 - (2.0 * z - 1.0).powi(2)).max(0.0);
        1.0e-3 + envelope * (1.0 + 0.25 * r + 0.1 * z)
    })
}

fn scaled_operator_source(profile_source: &Array2<f64>) -> Array2<f64> {
    profile_source.mapv(|value| value * 2.0 * std::f64::consts::PI)
}

fn source_mask(nz: usize, nr: usize) -> Array2<bool> {
    Array2::from_shape_fn((nz, nr), |(iz, ir)| {
        iz > 0 && ir > 0 && iz + 1 < nz && ir + 1 < nr
    })
}

fn synthetic_flux(grid: &Grid2D) -> Array2<f64> {
    Array2::from_shape_fn((grid.nz, grid.nr), |(iz, ir)| {
        let r = grid.r[ir];
        let z = grid.z[iz];
        0.03125 * r.powi(4) - 0.125 * z.powi(2) + 0.05 * r.powi(2) * z.powi(2)
    })
}

fn synthetic_plasma_domain_mask(grid: &Grid2D) -> Array2<bool> {
    Array2::from_shape_fn((grid.nz, grid.nr), |(iz, ir)| {
        let r = (grid.r[ir] - 2.0) / 0.75;
        let z = grid.z[iz] / 0.65;
        r * r + z * z <= 1.0
    })
}

/// Benchmark native GEQDSK pressure/FFprime source assembly on representative grids.
fn bench_geqdsk_profile_source_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("geqdsk_profile_source_components");
    group.sample_size(10);

    for &(nz, nr) in &[(33usize, 33usize), (65usize, 65usize)] {
        let psi_norm = synthetic_psi_norm(nz, nr);
        let rr = synthetic_rr(nz, nr);
        let pprime = pprime_profile(nr);
        let ffprime = ffprime_profile(nr);

        group.bench_function(format!("source_components_{nz}x{nr}"), |b| {
            b.iter(|| {
                let components = compute_geqdsk_profile_source_components(
                    black_box(&psi_norm),
                    black_box(&rr),
                    black_box(&pprime),
                    black_box(&ffprime),
                    black_box(MU0),
                )
                .expect("GEQDSK source components should assemble");
                black_box((
                    components.pressure_source_norm,
                    components.ffprime_source_norm,
                    components.total_source_norm,
                ));
            })
        });
    }

    group.finish();
}

/// Benchmark full-domain and plasma-domain operator-current integration.
fn bench_geqdsk_operator_current_domains(c: &mut Criterion) {
    let mut group = c.benchmark_group("geqdsk_operator_current_domains");
    group.sample_size(10);

    for &(nz, nr) in &[(33usize, 33usize), (65usize, 65usize)] {
        let grid = Grid2D::new(nr, nz, 1.0, 3.0, -1.0, 1.0);
        let psi = synthetic_flux(&grid);
        let plasma_mask = synthetic_plasma_domain_mask(&grid);

        group.bench_function(format!("full_domain_current_{nz}x{nr}"), |b| {
            b.iter(|| {
                let total_current = total_toroidal_current_from_flux(
                    black_box(&psi),
                    black_box(&grid),
                    black_box(MU0),
                )
                .expect("full-domain operator current should integrate");
                black_box(total_current);
            })
        });

        group.bench_function(format!("plasma_domain_current_{nz}x{nr}"), |b| {
            b.iter(|| {
                let total_current = total_toroidal_current_from_flux_masked(
                    black_box(&psi),
                    black_box(&grid),
                    black_box(MU0),
                    black_box(&plasma_mask),
                )
                .expect("plasma-domain operator current should integrate");
                black_box(total_current);
            })
        });
    }

    group.finish();
}

/// Benchmark flux-profile interpolation primitives used by GEQDSK source assembly.
fn bench_flux_profile_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("geqdsk_flux_profile_interpolation");
    group.sample_size(10);

    for &(nz, nr) in &[(33usize, 33usize), (65usize, 65usize)] {
        let psi_norm = synthetic_psi_norm(nz, nr);
        let profile = pprime_profile(nr);
        let weights = synthetic_rr(nz, nr);
        let mask = source_mask(nz, nr);

        group.bench_function(format!("second_order_{nz}x{nr}"), |b| {
            b.iter(|| {
                let interpolated = interpolate_flux_profile_second_order(
                    black_box(&psi_norm),
                    black_box(&profile),
                )
                .expect("second-order interpolation should succeed");
                black_box(interpolated[[nz / 2, nr / 2]]);
            })
        });

        group.bench_function(format!("current_conserving_{nz}x{nr}"), |b| {
            b.iter(|| {
                let interpolated = interpolate_flux_profile_current_conserving(
                    black_box(&psi_norm),
                    black_box(&profile),
                    black_box(&weights),
                    black_box(&mask),
                )
                .expect("current-conserving interpolation should succeed");
                black_box(interpolated[[nz / 2, nr / 2]]);
            })
        });
    }

    group.finish();
}

/// Benchmark explicit named source-convention adapter selection without fitted scales.
fn bench_geqdsk_source_convention_adapter(c: &mut Criterion) {
    let mut group = c.benchmark_group("geqdsk_source_convention_adapter");
    group.sample_size(10);

    for &(nz, nr) in &[(33usize, 33usize), (65usize, 65usize)] {
        let profile_source = synthetic_profile_source(nz, nr);
        let operator_source = scaled_operator_source(&profile_source);

        group.bench_function(format!("select_adapter_{nz}x{nr}"), |b| {
            b.iter(|| {
                let adapter = select_geqdsk_source_convention_adapter(
                    black_box(&operator_source),
                    black_box(&profile_source),
                    black_box(0.4),
                    black_box(0.15),
                )
                .expect("GEQDSK source convention adapter should select");
                black_box((
                    adapter.convention.as_str(),
                    adapter.residual_l2,
                    adapter.pass,
                ));
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_flux_profile_interpolation,
    bench_geqdsk_operator_current_domains,
    bench_geqdsk_profile_source_components,
    bench_geqdsk_source_convention_adapter
);
criterion_main!(benches);
