// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — FFT Benchmarks

use criterion::{criterion_group, criterion_main, Criterion};
use fusion_math::fft::{cfft2, cifft2, fft2, ifft2};
use ndarray::Array2;
use num_complex::Complex64;
use std::hint::black_box;

fn real_grid(size: usize) -> Array2<f64> {
    Array2::from_shape_fn((size, size), |(i, j)| {
        let phase = (i as f64) * 0.017 + (j as f64) * 0.031;
        phase.sin() + 0.5 * phase.cos()
    })
}

fn complex_grid(size: usize) -> Array2<Complex64> {
    Array2::from_shape_fn((size, size), |(i, j)| {
        let phase = (i as f64) * 0.013 - (j as f64) * 0.029;
        Complex64::new(phase.sin(), phase.cos())
    })
}

fn bench_real_fft2_64(c: &mut Criterion) {
    let input = real_grid(64);
    c.bench_function("fft2_real_64x64", |b| {
        b.iter(|| {
            let spectrum = fft2(black_box(&input));
            black_box(spectrum);
        })
    });
}

fn bench_real_ifft2_64(c: &mut Criterion) {
    let input = real_grid(64);
    let spectrum = fft2(&input);
    c.bench_function("ifft2_real_64x64", |b| {
        b.iter(|| {
            let recovered = ifft2(black_box(&spectrum));
            black_box(recovered);
        })
    });
}

fn bench_complex_fft_roundtrip_64(c: &mut Criterion) {
    let input = complex_grid(64);
    c.bench_function("cfft2_cifft2_complex_64x64", |b| {
        b.iter(|| {
            let spectrum = cfft2(black_box(&input));
            let recovered = cifft2(black_box(&spectrum));
            black_box(recovered);
        })
    });
}

criterion_group!(
    fft_benches,
    bench_real_fft2_64,
    bench_real_ifft2_64,
    bench_complex_fft_roundtrip_64,
);
criterion_main!(fft_benches);
