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

use crate::kuramoto::{fill_cos_sin, lyapunov_v, order_from_cos_sin, parallel_chunk, wrap_phase};
use fusion_types::error::{FusionError, FusionResult};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Barrier;

/// Total population at or above which the tick fills `cos θ`/`sin θ` and applies
/// the coupling update in parallel; below it a serial pass is faster than paying
/// rayon's fork-join cost.
const UPDE_PARALLEL_THRESHOLD: usize = 4096;

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

/// Validate the shared UPDE tick/run arguments, returning `(L, total)`.
///
/// `offsets` delimits `L` layers inside the flat `theta`/`omega`; `k`/`alpha`
/// are row-major `L×L` and `zeta` has length `L`.
fn upde_validate(
    theta: &[f64],
    omega: &[f64],
    offsets: &[usize],
    k: &[f64],
    alpha: &[f64],
    zeta: &[f64],
    dt: f64,
) -> FusionResult<(usize, usize)> {
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
    Ok((l, total))
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
    let (l, total) = upde_validate(theta, omega, offsets, k, alpha, zeta, dt)?;

    // Single transcendental pass for the whole tick: cache cos θ_i / sin θ_i
    // for every oscillator once, then derive both the order parameters (their
    // means) and every coupling term from them.
    let parallel = total >= UPDE_PARALLEL_THRESHOLD;
    let mut cos_th = vec![0.0_f64; total];
    let mut sin_th = vec![0.0_f64; total];
    fill_cos_sin(theta, &mut cos_th, &mut sin_th, parallel);

    // Per-layer order parameters (pre-step), then the global one — summed in
    // slice order from the cached cos/sin, so they are bit-identical to summing
    // the transcendentals inline.
    let mut r_layer = Vec::with_capacity(l);
    let mut psi_layer = Vec::with_capacity(l);
    for m in 0..l {
        let (r, psi) = order_from_cos_sin(
            &cos_th[offsets[m]..offsets[m + 1]],
            &sin_th[offsets[m]..offsets[m + 1]],
        );
        r_layer.push(r);
        psi_layer.push(psi);
    }
    let (r_global, psi_r_global) = order_from_cos_sin(&cos_th, &sin_th);

    let g = actuation_gain;

    // Collapse every per-element coupling term into two per-target-layer
    // coefficients using sin(A − θ) = sin A · cos θ − cos A · sin θ:
    //   dθ_{m,i} = ω_{m,i} + cos θ_i · Sc_m − sin θ_i · Ss_m,
    // where Sc_m / Ss_m sum sin A_{nm} / cos A_{nm} (A_{nm} = ψ_n − α_{nm})
    // weighted by the intra/inter/PAC/global gains — an L² precompute per tick
    // that leaves the per-oscillator update at two fused multiply-adds.
    let (sc, ss) = coupling_coefficients(
        l, k, alpha, zeta, g, pac_gamma, psi_global, &r_layer, &psi_layer,
    );

    let mut theta1 = vec![0.0_f64; total];
    let mut dtheta = vec![0.0_f64; total];
    advance_layers(
        theta,
        omega,
        offsets,
        dt,
        wrap,
        &cos_th,
        &sin_th,
        &sc,
        &ss,
        parallel,
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

/// Per-target-layer coupling coefficients `(Sc_m, Ss_m)` for the factored tick.
///
/// For each target layer `m`, every source term
/// `coeff_{nm} · sin(ψ_n − θ − α_{nm})` expands via
/// `sin(A − θ) = sin A · cos θ − cos A · sin θ` (with `A_{nm} = ψ_n − α_{nm}`)
/// into `coeff_{nm} · sin A_{nm} · cos θ − coeff_{nm} · cos A_{nm} · sin θ`.
/// Summing the θ-independent factors gives
/// `Sc_m = Σ_n coeff_{nm} · sin A_{nm} + ζ_m · sin Ψ` and the matching
/// `Ss_m` with cosines, so the per-oscillator update reduces to
/// `ω_i + cos θ_i · Sc_m − sin θ_i · Ss_m`. `coeff_{nm}` carries the actuation
/// gain, the `K_{nm} R_n` weight, and the inter-layer PAC gate
/// `1 + γ_pac (1 − R_n)` (unity on the intra-layer diagonal).
#[allow(clippy::too_many_arguments)]
fn coupling_coefficients(
    l: usize,
    k: &[f64],
    alpha: &[f64],
    zeta: &[f64],
    g: f64,
    pac_gamma: f64,
    psi_global: f64,
    r_layer: &[f64],
    psi_layer: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let (sin_psi_g, cos_psi_g) = (psi_global.sin(), psi_global.cos());
    let mut sc = vec![0.0_f64; l];
    let mut ss = vec![0.0_f64; l];
    for m in 0..l {
        let mut sc_m = 0.0_f64;
        let mut ss_m = 0.0_f64;
        for n in 0..l {
            let coeff = if n == m {
                g * k[m * l + m] * r_layer[m]
            } else {
                let pac_gate = 1.0 + pac_gamma * (1.0 - r_layer[n]);
                g * pac_gate * k[n * l + m] * r_layer[n]
            };
            let a = psi_layer[n] - alpha[n * l + m];
            sc_m += coeff * a.sin();
            ss_m += coeff * a.cos();
        }
        if zeta[m] != 0.0 {
            sc_m += zeta[m] * sin_psi_g;
            ss_m += zeta[m] * cos_psi_g;
        }
        sc[m] = sc_m;
        ss[m] = ss_m;
    }
    (sc, ss)
}

/// Advance every oscillator by one Euler step into pre-allocated buffers.
///
/// Each element evaluates `dθ_i/dt = ω_i + cos θ_i · Sc_m − sin θ_i · Ss_m` on
/// the cached transcendentals — reduction-free, so a rayon split cannot change
/// the floating-point result: cross-run and cross-thread deterministic. The
/// parallel path only engages for populations large enough to amortise the
/// fork-join overhead (per-call PyO3 granularity measured too small for it).
#[allow(clippy::too_many_arguments)]
fn advance_layers(
    theta: &[f64],
    omega: &[f64],
    offsets: &[usize],
    dt: f64,
    wrap: bool,
    cos_th: &[f64],
    sin_th: &[f64],
    sc: &[f64],
    ss: &[f64],
    parallel: bool,
    theta1: &mut [f64],
    dtheta: &mut [f64],
) {
    let l = offsets.len() - 1;
    let total = theta.len();
    let element = |i: usize, m: usize| -> (f64, f64) {
        let dth = omega[i] + cos_th[i] * sc[m] - sin_th[i] * ss[m];
        let mut th1 = theta[i] + dt * dth;
        if wrap {
            th1 = wrap_phase(th1);
        }
        (th1, dth)
    };

    if parallel {
        let mut layer_of = vec![0usize; total];
        for m in 0..l {
            for slot in layer_of.iter_mut().take(offsets[m + 1]).skip(offsets[m]) {
                *slot = m;
            }
        }
        let chunk = parallel_chunk(total);
        theta1
            .par_chunks_mut(chunk)
            .zip(dtheta.par_chunks_mut(chunk))
            .enumerate()
            .for_each(|(chunk_idx, (t1_chunk, dt_chunk))| {
                let base = chunk_idx * chunk;
                for (j, (t1, dt_out)) in t1_chunk.iter_mut().zip(dt_chunk.iter_mut()).enumerate() {
                    let i = base + j;
                    let (th1, dth) = element(i, layer_of[i]);
                    *t1 = th1;
                    *dt_out = dth;
                }
            });
    } else {
        for m in 0..l {
            for i in offsets[m]..offsets[m + 1] {
                let (th1, dth) = element(i, m);
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
    let (l, total) = upde_validate(theta0, omega, offsets, k, alpha, zeta, dt)?;
    let workers = rayon::current_num_threads();
    if total >= UPDE_PARALLEL_THRESHOLD && workers > 1 {
        Ok(upde_run_parallel(
            theta0,
            omega,
            offsets,
            k,
            alpha,
            zeta,
            n_steps,
            dt,
            psi_global,
            actuation_gain,
            pac_gamma,
            wrap,
            l,
            total,
            workers,
        ))
    } else {
        upde_run_serial(
            theta0,
            omega,
            offsets,
            k,
            alpha,
            zeta,
            n_steps,
            dt,
            psi_global,
            actuation_gain,
            pac_gamma,
            wrap,
        )
    }
}

/// Serial batched run: iterate [`upde_tick`], collecting per-tick observables.
#[allow(clippy::too_many_arguments)]
fn upde_run_serial(
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

/// Batched run on a persistent barrier-synchronised worker pool (mirrors
/// [`crate::kuramoto::kuramoto_run`]'s pool for the multi-layer tick).
///
/// Each worker owns a contiguous oscillator slice for the whole trajectory, so
/// the pool is entered once instead of a rayon fork-join per tick. Per tick:
/// each worker sums its slice's per-layer `(Σcos, Σsin)` into lock-free atomics
/// and meets at the order barrier; every worker then reduces the identical
/// per-layer partials (same order → identical `R_m`, `ψ_m`), builds the coupling
/// coefficients, records the order history on worker 0, and updates its own
/// slice in place while accumulating each layer's Lyapunov partial; a second
/// barrier gathers those and worker 0 records the Lyapunov history. The two
/// barriers also fence each tick's partial reads against the next tick's writes
/// (the order barrier for the Lyapunov partials, the Lyapunov barrier for the
/// order partials), so no third barrier is needed and the `Relaxed` atomics are
/// ordered throughout. Disjoint `chunks_mut` slices mean no update aliases.
#[allow(clippy::too_many_arguments)]
fn upde_run_parallel(
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
    l: usize,
    total: usize,
    workers: usize,
) -> UpdeRunResult {
    let mut theta = theta0.to_vec();
    let g = actuation_gain;
    let total_f = total as f64;
    let chunk = total.div_ceil(workers);
    let n_chunks = total.div_ceil(chunk);

    // Flat layer index per oscillator, and per-layer sizes for the means.
    let mut layer_of = vec![0usize; total];
    let mut layer_size = vec![0.0_f64; l];
    for m in 0..l {
        for slot in layer_of.iter_mut().take(offsets[m + 1]).skip(offsets[m]) {
            *slot = m;
        }
        layer_size[m] = (offsets[m + 1] - offsets[m]) as f64;
    }

    let atomics = |len: usize| -> Vec<AtomicU64> { (0..len).map(|_| AtomicU64::new(0)).collect() };
    let partial_re = atomics(n_chunks * l);
    let partial_im = atomics(n_chunks * l);
    let v_partial = atomics(n_chunks * l);
    let r_layer_bits = atomics(n_steps * l);
    let r_global_bits = atomics(n_steps);
    let v_layer_bits = atomics(n_steps * l);
    let v_global_bits = atomics(n_steps);
    let barrier = Barrier::new(n_chunks);

    std::thread::scope(|scope| {
        for (tid, (theta_slice, omega_slice)) in
            theta.chunks_mut(chunk).zip(omega.chunks(chunk)).enumerate()
        {
            let (partial_re, partial_im, v_partial) = (&partial_re, &partial_im, &v_partial);
            let (r_layer_bits, r_global_bits) = (&r_layer_bits, &r_global_bits);
            let (v_layer_bits, v_global_bits) = (&v_layer_bits, &v_global_bits);
            let (layer_of, layer_size, barrier) = (&layer_of, &layer_size, &barrier);
            scope.spawn(move || {
                let base = tid * chunk;
                let len = theta_slice.len();
                let mut cos_buf = vec![0.0_f64; len];
                let mut sin_buf = vec![0.0_f64; len];
                let mut re_loc = vec![0.0_f64; l];
                let mut im_loc = vec![0.0_f64; l];
                let mut v_loc = vec![0.0_f64; l];
                for tick in 0..n_steps {
                    // Per-layer order-parameter partials for this slice.
                    crate::sincos::fill_pairs(theta_slice, &mut cos_buf, &mut sin_buf);
                    re_loc.iter_mut().for_each(|x| *x = 0.0);
                    im_loc.iter_mut().for_each(|x| *x = 0.0);
                    for idx in 0..len {
                        let m = layer_of[base + idx];
                        re_loc[m] += cos_buf[idx];
                        im_loc[m] += sin_buf[idx];
                    }
                    for m in 0..l {
                        partial_re[tid * l + m].store(re_loc[m].to_bits(), Ordering::Relaxed);
                        partial_im[tid * l + m].store(im_loc[m].to_bits(), Ordering::Relaxed);
                    }
                    barrier.wait();

                    // Reduce per-layer + global order parameter (same order on
                    // every worker → identical R_m/ψ_m, hence coefficients).
                    let mut r_layer = vec![0.0_f64; l];
                    let mut psi_layer = vec![0.0_f64; l];
                    let (mut re_g, mut im_g) = (0.0_f64, 0.0_f64);
                    for m in 0..l {
                        let mut re_m = 0.0_f64;
                        let mut im_m = 0.0_f64;
                        for t in 0..n_chunks {
                            re_m += f64::from_bits(partial_re[t * l + m].load(Ordering::Relaxed));
                            im_m += f64::from_bits(partial_im[t * l + m].load(Ordering::Relaxed));
                        }
                        re_g += re_m;
                        im_g += im_m;
                        let sz = layer_size[m];
                        if sz > 0.0 {
                            let (re, im) = (re_m / sz, im_m / sz);
                            r_layer[m] = (re * re + im * im).sqrt();
                            psi_layer[m] = im.atan2(re);
                        }
                    }
                    let r_global = {
                        let (re, im) = (re_g / total_f, im_g / total_f);
                        (re * re + im * im).sqrt()
                    };
                    let (sc, ss) = coupling_coefficients(
                        l, k, alpha, zeta, g, pac_gamma, psi_global, &r_layer, &psi_layer,
                    );
                    if tid == 0 {
                        for m in 0..l {
                            r_layer_bits[tick * l + m]
                                .store(r_layer[m].to_bits(), Ordering::Relaxed);
                        }
                        r_global_bits[tick].store(r_global.to_bits(), Ordering::Relaxed);
                    }

                    // Update slice in place; accumulate each layer's Lyapunov V.
                    v_loc.iter_mut().for_each(|x| *x = 0.0);
                    for idx in 0..len {
                        let m = layer_of[base + idx];
                        let dth = omega_slice[idx] + cos_buf[idx] * sc[m] - sin_buf[idx] * ss[m];
                        let raw = theta_slice[idx] + dt * dth;
                        let th1 = if wrap { wrap_phase(raw) } else { raw };
                        theta_slice[idx] = th1;
                        v_loc[m] += 1.0 - (th1 - psi_global).cos();
                    }
                    for m in 0..l {
                        v_partial[tid * l + m].store(v_loc[m].to_bits(), Ordering::Relaxed);
                    }
                    barrier.wait();

                    if tid == 0 {
                        let mut v_g = 0.0_f64;
                        for m in 0..l {
                            let mut v_m = 0.0_f64;
                            for t in 0..n_chunks {
                                v_m += f64::from_bits(v_partial[t * l + m].load(Ordering::Relaxed));
                            }
                            v_g += v_m;
                            let sz = layer_size[m];
                            let v_layer = if sz > 0.0 { v_m / sz } else { 0.0 };
                            v_layer_bits[tick * l + m].store(v_layer.to_bits(), Ordering::Relaxed);
                        }
                        v_global_bits[tick].store((v_g / total_f).to_bits(), Ordering::Relaxed);
                    }
                }
            });
        }
    });

    let decode = |bits: &[AtomicU64]| -> Vec<f64> {
        bits.iter()
            .map(|b| f64::from_bits(b.load(Ordering::Relaxed)))
            .collect()
    };
    UpdeRunResult {
        theta_final: theta,
        r_layer_hist: decode(&r_layer_bits),
        r_global_hist: decode(&r_global_bits),
        v_layer_hist: decode(&v_layer_bits),
        v_global_hist: decode(&v_global_bits),
    }
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

    #[test]
    fn advance_layers_parallel_matches_serial_bit_for_bit() {
        // The element update is reduction-free, so the parallel branch must
        // reproduce the serial branch exactly, independent of the rayon split.
        let (theta, omega, offsets) = two_layer_fixture();
        let total = theta.len();
        let l = offsets.len() - 1;
        let mut cos_th = vec![0.0_f64; total];
        let mut sin_th = vec![0.0_f64; total];
        fill_cos_sin(&theta, &mut cos_th, &mut sin_th, false);
        let mut r_layer = Vec::new();
        let mut psi_layer = Vec::new();
        for m in 0..l {
            let (r, psi) = order_from_cos_sin(
                &cos_th[offsets[m]..offsets[m + 1]],
                &sin_th[offsets[m]..offsets[m + 1]],
            );
            r_layer.push(r);
            psi_layer.push(psi);
        }
        let k = vec![1.2, 0.3, 0.2, 0.9];
        let alpha = vec![0.0, 0.05, 0.02, 0.0];
        let zeta = vec![0.5, 0.1];
        let (sc, ss) =
            coupling_coefficients(l, &k, &alpha, &zeta, 1.1, 0.4, 0.25, &r_layer, &psi_layer);

        let mut t1_ser = vec![0.0_f64; total];
        let mut d_ser = vec![0.0_f64; total];
        let mut t1_par = vec![0.0_f64; total];
        let mut d_par = vec![0.0_f64; total];
        advance_layers(
            &theta,
            &omega,
            &offsets,
            5e-3,
            true,
            &cos_th,
            &sin_th,
            &sc,
            &ss,
            false,
            &mut t1_ser,
            &mut d_ser,
        );
        advance_layers(
            &theta,
            &omega,
            &offsets,
            5e-3,
            true,
            &cos_th,
            &sin_th,
            &sc,
            &ss,
            true,
            &mut t1_par,
            &mut d_par,
        );
        assert_eq!(t1_ser, t1_par);
        assert_eq!(d_ser, d_par);
    }

    #[test]
    fn upde_tick_parallel_branch_advances_large_population() {
        // Total at or above the parallel threshold exercises the parallel fill
        // and update; the tick must stay finite with well-formed observables.
        let n = 2100; // two layers -> total 4200 >= UPDE_PARALLEL_THRESHOLD
        let total = 2 * n;
        assert!(total >= UPDE_PARALLEL_THRESHOLD);
        let theta: Vec<f64> = (0..total)
            .map(|i| ((i * 37 % 211) as f64) / 105.0 - 1.0)
            .collect();
        let omega: Vec<f64> = (0..total)
            .map(|i| ((i * 13 % 50) as f64) / 100.0 - 0.25)
            .collect();
        let offsets = vec![0, n, total];
        let k = vec![1.2, 0.3, 0.2, 0.9];
        let alpha = vec![0.0, 0.05, 0.02, 0.0];
        let zeta = vec![0.5, 0.1];
        let out = upde_tick(
            &theta, &omega, &offsets, &k, &alpha, &zeta, 5e-3, 0.25, 1.1, 0.4, true,
        )
        .unwrap();
        assert_eq!(out.theta1.len(), total);
        assert_eq!(out.dtheta.len(), total);
        assert!(out.theta1.iter().all(|v| v.is_finite()));
        assert!(out.r_layer.iter().all(|r| (0.0..=1.0 + 1e-12).contains(r)));
        assert!((0.0..=1.0 + 1e-12).contains(&out.r_global));
        assert!(out.v_global >= 0.0);
    }

    #[test]
    fn upde_run_parallel_matches_serial_across_worker_counts() {
        // A two-layer population large enough for the pool; every worker count
        // (including ones whose slice boundary crosses the layer boundary, and
        // the single-worker edge) must reproduce the serial trajectory and all
        // four history streams within the parity gate.
        let n0 = 2600usize;
        let n1 = 2900usize;
        let total = n0 + n1;
        let theta: Vec<f64> = (0..total)
            .map(|i| ((i * 53 % 211) as f64) / 105.0 - 1.0)
            .collect();
        let omega: Vec<f64> = (0..total)
            .map(|i| ((i * 7 % 40) as f64) / 100.0 - 0.2)
            .collect();
        let offsets = vec![0, n0, total];
        let k = vec![1.2, 0.3, 0.2, 0.9];
        let alpha = vec![0.0, 0.05, 0.02, 0.0];
        let zeta = vec![0.5, 0.1];
        let (n_steps, dt, psi, g, pac) = (30usize, 5e-3, 0.25, 1.1, 0.4);
        let serial = upde_run_serial(
            &theta, &omega, &offsets, &k, &alpha, &zeta, n_steps, dt, psi, g, pac, true,
        )
        .unwrap();
        let (l, total_check) =
            upde_validate(&theta, &omega, &offsets, &k, &alpha, &zeta, dt).unwrap();
        let max_diff = |a: &[f64], b: &[f64]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0_f64, f64::max)
        };
        for workers in [1usize, 2, 3, 5] {
            let par = upde_run_parallel(
                &theta,
                &omega,
                &offsets,
                &k,
                &alpha,
                &zeta,
                n_steps,
                dt,
                psi,
                g,
                pac,
                true,
                l,
                total_check,
                workers,
            );
            assert!(
                max_diff(&serial.theta_final, &par.theta_final) < 1e-12,
                "workers={workers} θ"
            );
            assert!(
                max_diff(&serial.r_layer_hist, &par.r_layer_hist) < 1e-12,
                "workers={workers} R_layer"
            );
            assert!(
                max_diff(&serial.r_global_hist, &par.r_global_hist) < 1e-12,
                "workers={workers} R_global"
            );
            assert!(
                max_diff(&serial.v_layer_hist, &par.v_layer_hist) < 1e-12,
                "workers={workers} V_layer"
            );
            assert!(
                max_diff(&serial.v_global_hist, &par.v_global_hist) < 1e-12,
                "workers={workers} V_global"
            );
        }
    }
}
