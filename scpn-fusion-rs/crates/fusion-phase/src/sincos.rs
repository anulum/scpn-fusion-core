// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Vectorisable sine/cosine primitive
//! Branch-free `sin`/`cos` pair for the phase-kernel transcendental fill.
//!
//! The phase kernels spend their time turning phases into `cos θ`/`sin θ`
//! ([`crate::kuramoto::fill_cos_sin`]). `std`'s `f64::sin`/`f64::cos` are two
//! independent scalar libm calls, each with its own argument reduction, and the
//! compiler cannot vectorise across them. [`sincos`] instead computes both from
//! one Cody-Waite reduction and two minimax polynomials in straight-line,
//! branch-free arithmetic, so it is (a) cheaper than two libm calls even scalar
//! and (b) auto-vectorisable: [`fill_pairs`] runs it under a runtime-detected
//! `avx2`+`fma` clone (four f64 lanes) with a scalar fallback, so the wheel
//! stays portable — no AVX baked into the baseline build, no `SIGILL` on an
//! older CPU.
//!
//! Accuracy is within a few units in the last place of the libm reference over
//! the phase range (verified in the tests), comfortably inside the 1e-12
//! cross-tier parity gate and the 1e-14 order-parameter gate.

use std::f64::consts::{FRAC_2_PI, FRAC_PI_2};

// Two-part Cody-Waite split of π/2: `FRAC_PI_2` is the f64 nearest to π/2 and
// `PIO2_LO` its round-off, so the FMA reduction below keeps `r = x − q·(π/2)`
// accurate to ~1 ulp across the phase range.
const PIO2_LO: f64 = 6.123233995736766e-17; // π/2 − FRAC_PI_2

// cephes minimax coefficients for sin(r) on [−π/4, π/4] (highest order first);
// trailing comment = full-width cephes constant this f64 rounds to. Shortest
// round-trip f64 literals, so every digit shown affects the value.
const S5: f64 = 1.5896230157654656e-10; // 1.58962301576546568060e-10
const S4: f64 = -2.5050747762857807e-8; // -2.50507477628578072866e-8
const S3: f64 = 2.7557313621385722e-6; // 2.75573136213857245213e-6
const S2: f64 = -0.0001984126982958954; // -1.98412698295895385996e-4
const S1: f64 = 0.008333333333322118; // 8.33333333332211858878e-3
const S0: f64 = -0.1666666666666663; // -1.66666666666666307295e-1

// cephes minimax coefficients for cos(r) on [−π/4, π/4] (highest order first).
const C5: f64 = -1.1358536521387682e-11; // -1.13585365213876817300e-11
const C4: f64 = 2.087570084197473e-9; // 2.08757008419747316778e-9
const C3: f64 = -2.755731417929674e-7; // -2.75573141792967388112e-7
const C2: f64 = 2.4801587288851704e-5; // 2.48015872888517045348e-5
const C1: f64 = -0.0013888888888873056; // -1.38888888888730564116e-3
const C0: f64 = 0.041666666666666595; // 4.16666666666665929218e-2

/// Simultaneous `(sin x, cos x)` via one Cody-Waite reduction to `[−π/4, π/4]`
/// and the two cephes minimax polynomials, with a branch-free quadrant select.
///
/// Written entirely in straight-line FMA arithmetic (ternaries lower to selects,
/// sign flips to `±1.0` multiplies) so the enclosing loop auto-vectorises.
#[inline(always)]
pub(crate) fn sincos(x: f64) -> (f64, f64) {
    // Reduce to r ∈ [−π/4, π/4] and the quadrant count q.
    let q_f = (x * FRAC_2_PI).round();
    let r = q_f.mul_add(-FRAC_PI_2, x);
    let r = q_f.mul_add(-PIO2_LO, r);
    let z = r * r;

    // sin(r) = r + r·z·P_sin(z); cos(r) = (1 − ½z) + z²·P_cos(z).
    let p_sin = S5
        .mul_add(z, S4)
        .mul_add(z, S3)
        .mul_add(z, S2)
        .mul_add(z, S1)
        .mul_add(z, S0);
    let sin_r = (r * z).mul_add(p_sin, r);
    let p_cos = C5
        .mul_add(z, C4)
        .mul_add(z, C3)
        .mul_add(z, C2)
        .mul_add(z, C1)
        .mul_add(z, C0);
    let cos_r = (z * z).mul_add(p_cos, z.mul_add(-0.5, 1.0));

    // Quadrant select: q&1 swaps sin/cos; q&2 and (q+1)&2 set the signs.
    //   q mod 4 → (sin, cos): 0→(s, c) 1→(c, −s) 2→(−s, −c) 3→(−c, s).
    let q = q_f as i64;
    let swap = (q & 1) != 0;
    let sin_base = if swap { cos_r } else { sin_r };
    let cos_base = if swap { sin_r } else { cos_r };
    let sin_sign = if (q & 2) != 0 { -1.0 } else { 1.0 };
    let cos_sign = if ((q + 1) & 2) != 0 { -1.0 } else { 1.0 };
    (sin_base * sin_sign, cos_base * cos_sign)
}

/// Fill `cos_out`/`sin_out` from `theta` with [`sincos`], scalar reference path.
#[inline]
fn fill_pairs_scalar(theta: &[f64], cos_out: &mut [f64], sin_out: &mut [f64]) {
    for ((c, s), &th) in cos_out.iter_mut().zip(sin_out.iter_mut()).zip(theta.iter()) {
        let (sin, cos) = sincos(th);
        *c = cos;
        *s = sin;
    }
}

/// AVX2+FMA clone of [`fill_pairs_scalar`]: identical source, but compiled with
/// the wide target features so the straight-line [`sincos`] auto-vectorises to
/// four f64 lanes. Only reached after a runtime feature probe, never from the
/// baseline build, so the wheel never issues an unsupported instruction.
///
/// # Safety
/// The caller must ensure the CPU supports `avx2` and `fma` (guaranteed by the
/// `is_x86_feature_detected!` gate in [`fill_pairs`]).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn fill_pairs_avx2(theta: &[f64], cos_out: &mut [f64], sin_out: &mut [f64]) {
    fill_pairs_scalar(theta, cos_out, sin_out);
}

/// Fill `cos_out`/`sin_out` from `theta`, dispatching to the AVX2 clone when the
/// running CPU supports it and to the scalar path otherwise.
///
/// `theta`, `cos_out` and `sin_out` must share the same length; extra elements
/// in `cos_out`/`sin_out` past `theta.len()` are left untouched (the zip stops
/// at the shortest), matching the caller's equal-length buffers.
#[inline]
pub(crate) fn fill_pairs(theta: &[f64], cos_out: &mut [f64], sin_out: &mut [f64]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: guarded by the runtime feature probe above.
            unsafe { fill_pairs_avx2(theta, cos_out, sin_out) };
            return;
        }
    }
    fill_pairs_scalar(theta, cos_out, sin_out);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Maximum absolute error of [`sincos`] against the libm reference over a
    /// dense grid, plus the parity-critical order-parameter regime.
    fn max_abs_error(lo: f64, hi: f64, samples: usize) -> f64 {
        let mut worst = 0.0_f64;
        for i in 0..=samples {
            let x = lo + (hi - lo) * (i as f64) / (samples as f64);
            let (s, c) = sincos(x);
            worst = worst.max((s - x.sin()).abs()).max((c - x.cos()).abs());
        }
        worst
    }

    #[test]
    fn matches_libm_within_a_few_ulp_over_the_phase_range() {
        // Wrapped phases live in [−π, π]; keep a healthy margin either side.
        assert!(max_abs_error(-4.0, 4.0, 200_000) < 1e-15);
    }

    #[test]
    fn matches_libm_over_several_periods() {
        // Unwrapped phases can drift a few periods; the reduction must hold.
        assert!(max_abs_error(-40.0, 40.0, 400_000) < 1e-14);
    }

    #[test]
    fn exact_at_the_cardinal_angles() {
        for (x, want_s, want_c) in [
            (0.0, 0.0, 1.0),
            (std::f64::consts::FRAC_PI_2, 1.0, 0.0),
            (std::f64::consts::PI, 0.0, -1.0),
            (-std::f64::consts::FRAC_PI_2, -1.0, 0.0),
        ] {
            let (s, c) = sincos(x);
            assert!((s - want_s).abs() < 1e-15, "sin({x})={s}");
            assert!((c - want_c).abs() < 1e-15, "cos({x})={c}");
        }
    }

    #[test]
    fn pythagorean_identity_holds() {
        for i in -500..=500 {
            let x = i as f64 * 0.037;
            let (s, c) = sincos(x);
            assert!((s * s + c * c - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn fill_pairs_matches_scalar_reference() {
        let theta: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.011 - 5.0).collect();
        let mut c_dispatch = vec![0.0; theta.len()];
        let mut s_dispatch = vec![0.0; theta.len()];
        let mut c_scalar = vec![0.0; theta.len()];
        let mut s_scalar = vec![0.0; theta.len()];
        fill_pairs(&theta, &mut c_dispatch, &mut s_dispatch);
        fill_pairs_scalar(&theta, &mut c_scalar, &mut s_scalar);
        assert_eq!(c_dispatch, c_scalar);
        assert_eq!(s_dispatch, s_scalar);
        // And both track libm.
        for (i, &th) in theta.iter().enumerate() {
            assert!((c_dispatch[i] - th.cos()).abs() < 1e-15);
            assert!((s_dispatch[i] - th.sin()).abs() < 1e-15);
        }
    }
}
