// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Kuramoto-Sakaguchi Phase Kernel
//! Mean-field Kuramoto-Sakaguchi step with exogenous global driver.
//!
//! Mirrors `scpn_fusion.phase.kuramoto.kuramoto_sakaguchi_step` (the NumPy
//! reference tier) operation-for-operation:
//!
//! dθ_i/dt = ω_i + K·R·sin(ψ_r − θ_i − α) + ζ·sin(Ψ − θ_i)
//!
//! The kernels are deterministic (no RNG), so cross-tier agreement is
//! bounded only by floating-point summation order (~1e-14 relative).

use fusion_types::error::{FusionError, FusionResult};
use rayon::prelude::*;

/// Population size at or above which the batched run parallelises the per-step
/// order parameter and phase update; below it a serial loop is faster than
/// paying rayon's fork-join cost.
const KURAMOTO_PARALLEL_THRESHOLD: usize = 2048;

/// Map a phase to [-π, π) (mirror of NumPy's `(x + π) % 2π − π`).
#[inline]
pub fn wrap_phase(x: f64) -> f64 {
    (x + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
}

/// Kuramoto order parameter `R·exp(i·ψ_r) = <exp(i·θ)>`; returns `(R, ψ_r)`.
pub fn order_parameter(theta: &[f64]) -> (f64, f64) {
    if theta.is_empty() {
        return (0.0, 0.0);
    }
    let n = theta.len() as f64;
    let (mut re, mut im) = (0.0_f64, 0.0_f64);
    for &th in theta {
        re += th.cos();
        im += th.sin();
    }
    re /= n;
    im /= n;
    ((re * re + im * im).sqrt(), im.atan2(re))
}

/// Lyapunov candidate `V = (1/N) Σ (1 − cos(θ_i − Ψ))`; 0 at perfect sync.
pub fn lyapunov_v(theta: &[f64], psi: f64) -> f64 {
    if theta.is_empty() {
        return 0.0;
    }
    let sum: f64 = theta.iter().map(|&th| 1.0 - (th - psi).cos()).sum();
    sum / theta.len() as f64
}

/// Result of one Kuramoto-Sakaguchi Euler step.
pub struct KuramotoStepResult {
    /// Advanced phases (wrapped to [-π, π) when requested).
    pub theta1: Vec<f64>,
    /// Phase velocities dθ/dt used for the step.
    pub dtheta: Vec<f64>,
    /// Order-parameter magnitude R before the step.
    pub r: f64,
    /// Order-parameter phase ψ_r before the step.
    pub psi_r: f64,
}

/// Single Euler step of the mean-field Kuramoto-Sakaguchi system.
///
/// `psi` is the resolved global driver phase Ψ (the caller owns the
/// external/mean-field resolution policy, exactly like the Python tier).
#[allow(clippy::too_many_arguments)]
pub fn kuramoto_step(
    theta: &[f64],
    omega: &[f64],
    dt: f64,
    k: f64,
    alpha: f64,
    zeta: f64,
    psi: f64,
    wrap: bool,
) -> FusionResult<KuramotoStepResult> {
    if theta.len() != omega.len() {
        return Err(FusionError::ConfigError(format!(
            "kuramoto_step: theta ({}) and omega ({}) length mismatch",
            theta.len(),
            omega.len()
        )));
    }
    if !dt.is_finite() {
        return Err(FusionError::ConfigError(
            "kuramoto_step: dt must be finite".to_string(),
        ));
    }

    let (r, psi_r) = order_parameter(theta);
    let kr = k * r;

    // Serial element loop: per-call PyO3 granularity is too small for a
    // thread pool to pay off (measured), and serial keeps the result
    // trivially deterministic.
    let mut theta1 = Vec::with_capacity(theta.len());
    let mut dtheta = Vec::with_capacity(theta.len());
    for (&th, &om) in theta.iter().zip(omega.iter()) {
        let mut dth = om + kr * (psi_r - th - alpha).sin();
        if zeta != 0.0 {
            dth += zeta * (psi - th).sin();
        }
        let mut th1 = th + dt * dth;
        if wrap {
            th1 = wrap_phase(th1);
        }
        theta1.push(th1);
        dtheta.push(dth);
    }

    Ok(KuramotoStepResult {
        theta1,
        dtheta,
        r,
        psi_r,
    })
}

/// Order parameter with a parallel reduction for large populations.
///
/// The parallel fold/reduce changes float summation order versus the serial
/// path, but both agree with the NumPy tier to well within the 1e-12 cross-tier
/// parity gate, so the batched run stays tier-compatible.
fn order_parameter_fast(theta: &[f64]) -> (f64, f64) {
    if theta.is_empty() {
        return (0.0, 0.0);
    }
    let n = theta.len() as f64;
    let (re, im) = if theta.len() >= KURAMOTO_PARALLEL_THRESHOLD {
        theta
            .par_iter()
            .fold(
                || (0.0_f64, 0.0_f64),
                |(re, im), &th| (re + th.cos(), im + th.sin()),
            )
            .reduce(|| (0.0_f64, 0.0_f64), |a, b| (a.0 + b.0, a.1 + b.1))
    } else {
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for &th in theta {
            re += th.cos();
            im += th.sin();
        }
        (re, im)
    };
    let re = re / n;
    let im = im / n;
    ((re * re + im * im).sqrt(), im.atan2(re))
}

/// Result of a batched multi-step Kuramoto-Sakaguchi run (constant driver Ψ).
pub struct KuramotoRunResult {
    /// Final phase vector after `n_steps` Euler steps.
    pub theta_final: Vec<f64>,
    /// Per-step order-parameter magnitude R (pre-step).
    pub r_hist: Vec<f64>,
    /// Per-step order-parameter phase ψ_r (pre-step).
    pub psi_r_hist: Vec<f64>,
}

/// Run `n_steps` mean-field Kuramoto-Sakaguchi Euler steps entirely in Rust.
///
/// Mirrors iterating [`kuramoto_step`] with a constant driver phase Ψ, but keeps
/// the whole trajectory on the Rust side of the Python boundary — where the
/// compiled tier earns its speedup — and parallelises the per-step order
/// parameter and phase update across the population. The element updates are
/// reduction-free, so the rayon split cannot change their result; only the order
/// parameter's summation order differs from the serial path, within the parity
/// gate.
#[allow(clippy::too_many_arguments)]
pub fn kuramoto_run(
    theta0: &[f64],
    omega: &[f64],
    n_steps: usize,
    dt: f64,
    k: f64,
    alpha: f64,
    zeta: f64,
    psi: f64,
    wrap: bool,
) -> FusionResult<KuramotoRunResult> {
    if theta0.len() != omega.len() {
        return Err(FusionError::ConfigError(format!(
            "kuramoto_run: theta ({}) and omega ({}) length mismatch",
            theta0.len(),
            omega.len()
        )));
    }
    if !dt.is_finite() {
        return Err(FusionError::ConfigError(
            "kuramoto_run: dt must be finite".to_string(),
        ));
    }
    let n = theta0.len();
    let mut theta = theta0.to_vec();
    let mut theta_next = vec![0.0_f64; n];
    let mut r_hist = Vec::with_capacity(n_steps);
    let mut psi_r_hist = Vec::with_capacity(n_steps);

    for _ in 0..n_steps {
        let (r, psi_r) = order_parameter_fast(&theta);
        r_hist.push(r);
        psi_r_hist.push(psi_r);
        let kr = k * r;

        let update = |th: f64, om: f64| -> f64 {
            let mut dth = om + kr * (psi_r - th - alpha).sin();
            if zeta != 0.0 {
                dth += zeta * (psi - th).sin();
            }
            let th1 = th + dt * dth;
            if wrap {
                wrap_phase(th1)
            } else {
                th1
            }
        };

        if n >= KURAMOTO_PARALLEL_THRESHOLD {
            theta_next
                .par_iter_mut()
                .zip(theta.par_iter().zip(omega.par_iter()))
                .for_each(|(out, (&th, &om))| {
                    *out = update(th, om);
                });
        } else {
            for (out, (&th, &om)) in theta_next.iter_mut().zip(theta.iter().zip(omega.iter())) {
                *out = update(th, om);
            }
        }
        std::mem::swap(&mut theta, &mut theta_next);
    }

    Ok(KuramotoRunResult {
        theta_final: theta,
        r_hist,
        psi_r_hist,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn wrap_phase_maps_into_half_open_interval() {
        assert_abs_diff_eq!(wrap_phase(0.0), 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(
            wrap_phase(3.0 * std::f64::consts::PI),
            -std::f64::consts::PI,
            epsilon = 1e-12
        );
        let w = wrap_phase(-7.5);
        assert!((-std::f64::consts::PI..std::f64::consts::PI).contains(&w));
    }

    #[test]
    fn order_parameter_is_one_for_identical_phases() {
        let theta = vec![0.7; 64];
        let (r, psi) = order_parameter(&theta);
        assert_abs_diff_eq!(r, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(psi, 0.7, epsilon = 1e-12);
    }

    #[test]
    fn order_parameter_vanishes_for_balanced_phases() {
        let theta = vec![0.0, std::f64::consts::PI];
        let (r, _) = order_parameter(&theta);
        assert!(r < 1e-12);
    }

    #[test]
    fn order_parameter_empty_is_zero() {
        assert_eq!(order_parameter(&[]), (0.0, 0.0));
    }

    #[test]
    fn lyapunov_v_zero_at_sync_two_at_antiphase() {
        assert_abs_diff_eq!(lyapunov_v(&[0.4, 0.4], 0.4), 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(
            lyapunov_v(&[std::f64::consts::PI], 0.0),
            2.0,
            epsilon = 1e-12
        );
        assert_eq!(lyapunov_v(&[], 0.0), 0.0);
    }

    #[test]
    fn kuramoto_step_synchronises_toward_driver() {
        let n = 128;
        let theta: Vec<f64> = (0..n)
            .map(|i| 0.5 * ((i as f64) / (n as f64) - 0.5))
            .collect();
        let omega = vec![0.0; n];
        let mut th = theta;
        for _ in 0..2000 {
            let out = kuramoto_step(&th, &omega, 1e-2, 2.0, 0.0, 0.5, 0.0, true).unwrap();
            th = out.theta1;
        }
        let v = lyapunov_v(&th, 0.0);
        assert!(v < 1e-3, "expected near-sync, V = {v}");
    }

    #[test]
    fn kuramoto_step_rejects_mismatched_lengths() {
        assert!(kuramoto_step(&[0.0], &[0.0, 1.0], 0.1, 1.0, 0.0, 0.0, 0.0, true).is_err());
    }

    #[test]
    fn kuramoto_step_rejects_non_finite_dt() {
        assert!(kuramoto_step(&[0.0], &[0.0], f64::NAN, 1.0, 0.0, 0.0, 0.0, true).is_err());
    }

    #[test]
    fn kuramoto_run_matches_iterated_step_serial_path() {
        // Below the parallel threshold order_parameter_fast is the serial sum,
        // so the batched run is bit-identical to iterating the single step.
        let n = 64;
        assert!(n < KURAMOTO_PARALLEL_THRESHOLD);
        let theta: Vec<f64> = (0..n)
            .map(|i| ((i * 41 % 100) as f64) / 50.0 - 1.0)
            .collect();
        let omega: Vec<f64> = (0..n)
            .map(|i| ((i * 7 % 20) as f64) / 100.0 - 0.1)
            .collect();
        let n_steps = 60;

        let run = kuramoto_run(&theta, &omega, n_steps, 5e-3, 1.5, 0.05, 0.4, 0.2, true).unwrap();

        let mut th = theta.clone();
        let mut first_r = f64::NAN;
        for step in 0..n_steps {
            let out = kuramoto_step(&th, &omega, 5e-3, 1.5, 0.05, 0.4, 0.2, true).unwrap();
            if step == 0 {
                first_r = out.r;
            }
            th = out.theta1;
        }
        assert_eq!(run.theta_final.len(), n);
        assert_eq!(run.r_hist.len(), n_steps);
        assert_eq!(run.psi_r_hist.len(), n_steps);
        assert!((run.r_hist[0] - first_r).abs() < 1e-15);
        for (a, b) in run.theta_final.iter().zip(th.iter()) {
            assert!(
                (a - b).abs() < 1e-15,
                "serial batched run must match step-by-step"
            );
        }
    }

    #[test]
    fn kuramoto_run_parallel_path_matches_within_gate() {
        // Above the threshold the parallel reduction reorders the order-parameter
        // sum; the trajectory must still agree with the serial step within the
        // 1e-12 cross-tier parity gate over a short, stable run.
        let n = 4096;
        assert!(n >= KURAMOTO_PARALLEL_THRESHOLD);
        let theta: Vec<f64> = (0..n)
            .map(|i| ((i * 53 % 211) as f64) / 105.0 - 1.0)
            .collect();
        let omega = vec![0.0_f64; n];
        let n_steps = 50;

        let run = kuramoto_run(&theta, &omega, n_steps, 5e-3, 1.0, 0.0, 0.5, 0.0, true).unwrap();

        let mut th = theta.clone();
        for _ in 0..n_steps {
            let out = kuramoto_step(&th, &omega, 5e-3, 1.0, 0.0, 0.5, 0.0, true).unwrap();
            th = out.theta1;
        }
        let max_abs = run
            .theta_final
            .iter()
            .zip(th.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_abs < 1e-12,
            "parallel run diverged from serial by {max_abs}"
        );
    }

    #[test]
    fn kuramoto_run_rejects_bad_shapes() {
        assert!(kuramoto_run(&[0.0], &[0.0, 1.0], 5, 0.1, 1.0, 0.0, 0.0, 0.0, true).is_err());
        assert!(kuramoto_run(&[0.0], &[0.0], 5, f64::INFINITY, 1.0, 0.0, 0.0, 0.0, true).is_err());
    }
}
