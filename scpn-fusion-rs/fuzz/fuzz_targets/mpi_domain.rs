// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — MPI domain fuzz target
#![no_main]

use fusion_core::mpi_domain::{decompose_2d, extract_tile, serial_halo_exchange_2d};
use libfuzzer_sys::fuzz_target;
use ndarray::Array2;

fn bounded_dim(byte: u8) -> usize {
    4 + usize::from(byte % 13)
}

fn bounded_partition(byte: u8, dim: usize) -> usize {
    1 + usize::from(byte % u8::try_from(dim.min(4)).expect("bounded partition fits u8"))
}

fn finite_value(byte: u8, index: usize) -> f64 {
    let centred = f64::from(byte) - 127.5;
    centred / 64.0 + (index as f64) * 1.0e-6
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 6 {
        return;
    }

    let nz = bounded_dim(data[0]);
    let nr = bounded_dim(data[1]);
    let pz = bounded_partition(data[2], nz);
    let pr = bounded_partition(data[3], nr);
    let halo = 1usize;

    let global = Array2::from_shape_fn((nz, nr), |(iz, ir)| {
        let idx = (iz * nr + ir) % data.len();
        finite_value(data[idx], iz * nr + ir)
    });

    let tiles = decompose_2d(nz, nr, pz, pr, halo).expect("valid bounded decomposition");
    let mut locals = tiles
        .iter()
        .map(|tile| extract_tile(&global, tile).expect("valid tile extraction"))
        .collect::<Vec<_>>();

    serial_halo_exchange_2d(&mut locals, &tiles).expect("valid serial halo exchange");
});
