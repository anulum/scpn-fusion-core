// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Rust Multigrid Scaling Probe

use std::time::Instant;

use fusion_math::multigrid::{multigrid_solve, MultigridConfig};
use fusion_types::state::Grid2D;
use ndarray::Array2;

fn escape_json(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn main() {
    let config = MultigridConfig::default();
    let grids = [33_usize, 65, 129];

    println!(
        "{{\"schema\":\"rust-multigrid-scaling.v1\",\"config\":{{\"pre_smooth\":{},\"post_smooth\":{},\"omega\":{},\"coarse_iters\":{},\"min_grid_size\":{}}}}}",
        config.pre_smooth,
        config.post_smooth,
        config.omega,
        config.coarse_iters,
        config.min_grid_size
    );

    for n in grids {
        let grid = Grid2D::new(n, n, 1.0, 9.0, -5.0, 5.0);
        let mut psi = Array2::zeros((n, n));
        let source = Array2::from_elem((n, n), -1.0);

        let started = Instant::now();
        let result = multigrid_solve(&mut psi, &source, &grid, &config, 30, 1e-8);
        let elapsed = started.elapsed();
        let initial_residual = result.residual_history.first().copied().unwrap_or(f64::NAN);
        let contraction = if initial_residual.abs() > f64::EPSILON {
            result.residual / initial_residual
        } else {
            f64::NAN
        };

        println!(
            "{{\"grid\":{},\"points\":{},\"cycles\":{},\"converged\":{},\"initial_residual\":{},\"final_residual\":{},\"contraction_factor\":{},\"wall_time_ms\":{}}}",
            n,
            n * n,
            result.cycles,
            result.converged,
            initial_residual,
            result.residual,
            contraction,
            elapsed.as_secs_f64() * 1000.0
        );
    }

    eprintln!(
        "{{\"note\":\"{}\"}}",
        escape_json("timings are single-run local measurements; use Criterion or isolated runners for release claims")
    );
}
