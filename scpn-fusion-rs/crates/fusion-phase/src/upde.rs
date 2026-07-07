// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Multi-Layer UPDE Tick Kernel
//! One Euler tick of the multi-layer Unified Phase Dynamics Equation.
//!
//! Mirrors `scpn_fusion.phase.upde.UPDESystem.step` (the NumPy reference
//! tier) operation-for-operation. Layers are passed as a flat phase vector
//! plus offsets so non-uniform per-layer populations are supported:
//!
//! dθ_{m,i}/dt = ω_{m,i}
//!             + g K_{mm} R_m sin(ψ_m − θ_{m,i} − α_{mm})
//!             + Σ_{n≠m} g (1 + γ_pac (1 − R_n)) K_{nm} R_n sin(ψ_n − θ_{m,i} − α_{nm})
//!             + ζ_m sin(Ψ − θ_{m,i})

use crate::kuramoto::{lyapunov_v, order_parameter, wrap_phase};
use fusion_types::error::{FusionError, FusionResult};
use rayon::prelude::*;

/// Result of one multi-layer UPDE tick.
pub struct UpdeTickResult {
    /// Advanced flat phase vector (layer blocks in offset order).
    pub theta1: Vec<f64>,
    /// Flat phase-velocity vector.
    pub dtheta: Vec<f64>,
    /// Per-layer order-parameter magnitudes R_m (pre-step).
    pub r_layer: Vec<f64>,
    /// Per-layer order-parameter phases ψ_m (pre-step).
    pub psi_layer: Vec<f64>,
    /// Global order-parameter magnitude over all pre-step phases.
    pub r_global: f64,
    /// Global order-parameter phase over all pre-step phases.
    pub psi_r_global: f64,
    /// Per-layer Lyapunov V_m of the advanced phases against Ψ.
    pub v_layer: Vec<f64>,
    /// Global Lyapunov V of the advanced phases against Ψ.
    pub v_global: f64,
}

/// Advance all layers by one Euler tick.
///
/// `offsets` has length L+1 with `offsets[m]..offsets[m+1]` delimiting layer
/// m inside `theta`/`omega`. `k` and `alpha` are row-major L×L
/// (`k[n*l + m]` = coupling from source layer n into target layer m), and
/// `psi_global` is the resolved global driver phase Ψ.
#[allow(clippy::too_many_arguments)]
pub fn upde_tick(
    theta: &[f64],
    omega: &[f64],
    offsets: &[usize],
    k: &[f64],
    alpha: &[f64],
    zeta: &[f64],
    dt: f64,
    psi_global: f64,
    actuation_gain: f64,
    pac_gamma: f64,
    wrap: bool,
) -> FusionResult<UpdeTickResult> {
    if offsets.len() < 2 {
        return Err(FusionError::ConfigError(
            "upde_tick: offsets must delimit at least one layer".to_string(),
        ));
    }
    let l = offsets.len() - 1;
    let total = *offsets.last().expect("offsets checked non-empty");
    if theta.len() != total || omega.len() != total {
        return Err(FusionError::ConfigError(format!(
            "upde_tick: theta ({}) / omega ({}) must match offsets total ({total})",
            theta.len(),
            omega.len()
        )));
    }
    if offsets.windows(2).any(|w| w[1] < w[0]) || offsets[0] != 0 {
        return Err(FusionError::ConfigError(
            "upde_tick: offsets must start at 0 and be non-decreasing".to_string(),
        ));
    }
    if k.len() != l * l || alpha.len() != l * l || zeta.len() != l {
        return Err(FusionError::ConfigError(format!(
            "upde_tick: expected k/alpha of length {} and zeta of length {l}",
            l * l
        )));
    }
    if !dt.is_finite() {
        return Err(FusionError::ConfigError(
            "upde_tick: dt must be finite".to_string(),
        ));
    }

    // Per-layer order parameters (pre-step), then the global one.
    let mut r_layer = Vec::with_capacity(l);
    let mut psi_layer = Vec::with_capacity(l);
    for m in 0..l {
        let (r, psi) = order_parameter(&theta[offsets[m]..offsets[m + 1]]);
        r_layer.push(r);
        psi_layer.push(psi);
    }
    let (r_global, psi_r_global) = order_parameter(theta);

    let g = actuation_gain;

    let mut theta1 = vec![0.0_f64; total];
    let mut dtheta = vec![0.0_f64; total];
    advance_layers(
        theta,
        omega,
        offsets,
        k,
        alpha,
        zeta,
        dt,
        psi_global,
        g,
        pac_gamma,
        wrap,
        &r_layer,
        &psi_layer,
        &mut theta1,
        &mut dtheta,
    );

    let v_layer: Vec<f64> = (0..l)
        .map(|m| lyapunov_v(&theta1[offsets[m]..offsets[m + 1]], psi_global))
        .collect();
    let v_global = lyapunov_v(&theta1, psi_global);

    Ok(UpdeTickResult {
        theta1,
        dtheta,
        r_layer,
        psi_layer,
        r_global,
        psi_r_global,
        v_layer,
        v_global,
    })
}

/// Advance every oscillator by one Euler step into pre-allocated buffers.
///
/// Element-wise and reduction-free, so a rayon split cannot change the
/// floating-point result: cross-run and cross-thread deterministic. The
/// parallel path only engages for populations large enough to amortise the
/// fork-join overhead (per-call PyO3 granularity measured too small for it).
#[allow(clippy::too_many_arguments)]
fn advance_layers(
    theta: &[f64],
    omega: &[f64],
    offsets: &[usize],
    k: &[f64],
    alpha: &[f64],
    zeta: &[f64],
    dt: f64,
    psi_global: f64,
    g: f64,
    pac_gamma: f64,
    wrap: bool,
    r_layer: &[f64],
    psi_layer: &[f64],
    theta1: &mut [f64],
    dtheta: &mut [f64],
) {
    let l = offsets.len() - 1;
    let element = |m: usize, i: usize| -> (f64, f64) {
        let th = theta[i];
        // Intra-layer coupling.
        let mut dth =
            omega[i] + g * k[m * l + m] * r_layer[m] * (psi_layer[m] - th - alpha[m * l + m]).sin();
        // Inter-layer coupling with PAC-like gating on the source layer.
        for n in 0..l {
            if n == m {
                continue;
            }
            let pac_gate = 1.0 + pac_gamma * (1.0 - r_layer[n]);
            dth += g
                * pac_gate
                * k[n * l + m]
                * r_layer[n]
                * (psi_layer[n] - th - alpha[n * l + m]).sin();
        }
        // Global driver.
        if zeta[m] != 0.0 {
            dth += zeta[m] * (psi_global - th).sin();
        }
        let mut th1 = th + dt * dth;
        if wrap {
            th1 = wrap_phase(th1);
        }
        (th1, dth)
    };

    const PARALLEL_THRESHOLD: usize = 4096;
    let total = theta.len();
    if total >= PARALLEL_THRESHOLD {
        let mut layer_of = vec![0usize; total];
        for m in 0..l {
            for slot in layer_of.iter_mut().take(offsets[m + 1]).skip(offsets[m]) {
                *slot = m;
            }
        }
        theta1
            .par_iter_mut()
            .zip(dtheta.par_iter_mut())
            .enumerate()
            .for_each(|(i, (t1, dt_out))| {
                let (th1, dth) = element(layer_of[i], i);
                *t1 = th1;
                *dt_out = dth;
            });
    } else {
        for m in 0..l {
            for i in offsets[m]..offsets[m + 1] {
                let (th1, dth) = element(m, i);
                theta1[i] = th1;
                dtheta[i] = dth;
            }
        }
    }
}

/// Result of a batched multi-tick UPDE run.
pub struct UpdeRunResult {
    /// Final flat phase vector after `n_steps` ticks.
    pub theta_final: Vec<f64>,
    /// Per-tick per-layer order parameters, row-major (n_steps × L), pre-step.
    pub r_layer_hist: Vec<f64>,
    /// Per-tick global order parameter (pre-step).
    pub r_global_hist: Vec<f64>,
    /// Per-tick per-layer Lyapunov V (post-step), row-major (n_steps × L).
    pub v_layer_hist: Vec<f64>,
    /// Per-tick global Lyapunov V (post-step).
    pub v_global_hist: Vec<f64>,
}

/// Run `n_steps` UPDE ticks entirely in Rust (constant driver phase Ψ).
///
/// Mirrors `UPDESystem.run`/`run_lyapunov` for the external-driver mode: the
/// whole loop stays on this side of the Python boundary, which is where the
/// batched tier earns its speedup; per-tick observables are recorded exactly
/// like the per-step path records them.
#[allow(clippy::too_many_arguments)]
pub fn upde_run(
    theta0: &[f64],
    omega: &[f64],
    offsets: &[usize],
    k: &[f64],
    alpha: &[f64],
    zeta: &[f64],
    n_steps: usize,
    dt: f64,
    psi_global: f64,
    actuation_gain: f64,
    pac_gamma: f64,
    wrap: bool,
) -> FusionResult<UpdeRunResult> {
    let l = offsets.len().saturating_sub(1);
    let mut theta = theta0.to_vec();
    let mut r_layer_hist = Vec::with_capacity(n_steps * l);
    let mut r_global_hist = Vec::with_capacity(n_steps);
    let mut v_layer_hist = Vec::with_capacity(n_steps * l);
    let mut v_global_hist = Vec::with_capacity(n_steps);

    for _ in 0..n_steps {
        let out = upde_tick(
            &theta,
            omega,
            offsets,
            k,
            alpha,
            zeta,
            dt,
            psi_global,
            actuation_gain,
            pac_gamma,
            wrap,
        )?;
        r_layer_hist.extend_from_slice(&out.r_layer);
        r_global_hist.push(out.r_global);
        v_layer_hist.extend_from_slice(&out.v_layer);
        v_global_hist.push(out.v_global);
        theta = out.theta1;
    }

    Ok(UpdeRunResult {
        theta_final: theta,
        r_layer_hist,
        r_global_hist,
        v_layer_hist,
        v_global_hist,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_layer_fixture() -> (Vec<f64>, Vec<f64>, Vec<usize>) {
        let theta = vec![0.1, -0.2, 0.3, 1.0, -1.1, 0.9, 0.5];
        let omega = vec![0.0, 0.1, -0.1, 0.05, 0.0, -0.05, 0.02];
        let offsets = vec![0, 3, 7];
        (theta, omega, offsets)
    }

    #[test]
    fn upde_tick_advances_all_layers() {
        let (theta, omega, offsets) = two_layer_fixture();
        let k = vec![1.5, 0.2, 0.3, 1.0];
        let alpha = vec![0.0; 4];
        let zeta = vec![0.4, 0.0];
        let out = upde_tick(
            &theta, &omega, &offsets, &k, &alpha, &zeta, 1e-2, 0.0, 1.0, 0.0, true,
        )
        .unwrap();
        assert_eq!(out.theta1.len(), theta.len());
        assert_eq!(out.r_layer.len(), 2);
        assert!(out.r_layer.iter().all(|r| (0.0..=1.0 + 1e-12).contains(r)));
        assert!(out.v_global >= 0.0);
    }

    #[test]
    fn upde_tick_supports_non_uniform_layers() {
        let (theta, omega, offsets) = two_layer_fixture();
        assert_ne!(offsets[1] - offsets[0], offsets[2] - offsets[1]);
        let out = upde_tick(
            &theta,
            &omega,
            &offsets,
            &[1.0, 0.0, 0.0, 1.0],
            &[0.0; 4],
            &[0.0, 0.0],
            1e-2,
            0.0,
            1.0,
            0.0,
            true,
        )
        .unwrap();
        assert_eq!(out.v_layer.len(), 2);
    }

    #[test]
    fn upde_tick_converges_under_global_driver() {
        let n = 32;
        let mut theta: Vec<f64> = (0..2 * n)
            .map(|i| ((i * 37) % 100) as f64 / 50.0 - 1.0)
            .collect();
        let omega = vec![0.0; 2 * n];
        let offsets = vec![0, n, 2 * n];
        let k = vec![1.0, 0.1, 0.1, 1.0];
        let alpha = vec![0.0; 4];
        let zeta = vec![1.0, 1.0];
        for _ in 0..4000 {
            let out = upde_tick(
                &theta, &omega, &offsets, &k, &alpha, &zeta, 5e-3, 0.3, 1.0, 0.0, true,
            )
            .unwrap();
            theta = out.theta1;
        }
        let v = lyapunov_v(&theta, 0.3);
        assert!(v < 1e-3, "expected sync toward the driver, V = {v}");
    }

    #[test]
    fn upde_run_matches_iterated_ticks() {
        let (theta, omega, offsets) = two_layer_fixture();
        let k = vec![1.2, 0.3, 0.2, 0.9];
        let alpha = vec![0.0; 4];
        let zeta = vec![0.5, 0.1];
        let n_steps = 25;

        let run = upde_run(
            &theta, &omega, &offsets, &k, &alpha, &zeta, n_steps, 5e-3, 0.25, 1.1, 0.4, true,
        )
        .unwrap();

        let mut manual = theta.clone();
        let mut last_v_global = f64::NAN;
        for _ in 0..n_steps {
            let out = upde_tick(
                &manual, &omega, &offsets, &k, &alpha, &zeta, 5e-3, 0.25, 1.1, 0.4, true,
            )
            .unwrap();
            manual = out.theta1;
            last_v_global = out.v_global;
        }

        assert_eq!(run.r_layer_hist.len(), n_steps * 2);
        assert_eq!(run.v_global_hist.len(), n_steps);
        for (a, b) in run.theta_final.iter().zip(manual.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
        assert!((run.v_global_hist[n_steps - 1] - last_v_global).abs() < 1e-15);
    }

    #[test]
    fn upde_tick_rejects_bad_shapes() {
        let (theta, omega, offsets) = two_layer_fixture();
        assert!(upde_tick(
            &theta,
            &omega,
            &offsets,
            &[1.0],
            &[0.0; 4],
            &[0.0, 0.0],
            1e-2,
            0.0,
            1.0,
            0.0,
            true
        )
        .is_err());
        assert!(upde_tick(
            &theta,
            &omega,
            &[0],
            &[1.0],
            &[0.0],
            &[0.0],
            1e-2,
            0.0,
            1.0,
            0.0,
            true
        )
        .is_err());
        assert!(upde_tick(
            &theta[..3],
            &omega,
            &offsets,
            &[1.0; 4],
            &[0.0; 4],
            &[0.0, 0.0],
            1e-2,
            0.0,
            1.0,
            0.0,
            true
        )
        .is_err());
        assert!(upde_tick(
            &theta,
            &omega,
            &offsets,
            &[1.0; 4],
            &[0.0; 4],
            &[0.0, 0.0],
            f64::INFINITY,
            0.0,
            1.0,
            0.0,
            true
        )
        .is_err());
    }
}
