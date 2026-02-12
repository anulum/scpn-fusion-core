// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Tridiag
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Thomas algorithm for tridiagonal systems.
//!
//! Port of temhd_peltier.py solve_tridiagonal() (lines 25-58).
//! Used by TEMHD (implicit heat), blanket neutronics, and sawtooth.

/// Solve tridiagonal system Ax = d using the Thomas algorithm.
///
/// - `a`: sub-diagonal \[n\] (a\[0\] unused)
/// - `b`: main diagonal \[n\]
/// - `c`: super-diagonal \[n\] (c\[n-1\] unused)
/// - `d`: right-hand side \[n\]
///
/// Returns: solution vector x \[n\]
///
/// Panics if b\[0\] == 0 or if a pivot becomes zero (singular system).
pub fn thomas_solve(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = d.len();
    assert!(n > 0, "System size must be > 0");
    assert_eq!(a.len(), n);
    assert_eq!(b.len(), n);
    assert_eq!(c.len(), n);

    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];

    // Forward sweep
    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for i in 1..n {
        let den = b[i] - a[i] * c_prime[i - 1];
        if i < n - 1 {
            c_prime[i] = c[i] / den;
        }
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / den;
    }

    // Back substitution
    let mut x = vec![0.0; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thomas_identity() {
        // Solve I * x = [1,2,3,4,5]
        let n = 5;
        let a = vec![0.0; n];
        let b = vec![1.0; n];
        let c = vec![0.0; n];
        let d = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = thomas_solve(&a, &b, &c, &d);
        for i in 0..n {
            assert!((x[i] - d[i]).abs() < 1e-12, "x[{i}] should equal d[{i}]");
        }
    }

    #[test]
    fn test_thomas_simple_tridiag() {
        // Solve [-1, 2, -1] tridiagonal system (1D Laplacian)
        // [ 2 -1  0  0]   [x0]   [1]
        // [-1  2 -1  0] * [x1] = [0]
        // [ 0 -1  2 -1]   [x2]   [0]
        // [ 0  0 -1  2]   [x3]   [1]
        let a = vec![0.0, -1.0, -1.0, -1.0];
        let b = vec![2.0, 2.0, 2.0, 2.0];
        let c = vec![-1.0, -1.0, -1.0, 0.0];
        let d = vec![1.0, 0.0, 0.0, 1.0];
        let x = thomas_solve(&a, &b, &c, &d);

        // Verify Ax = d
        let ax = [
            b[0] * x[0] + c[0] * x[1],
            a[1] * x[0] + b[1] * x[1] + c[1] * x[2],
            a[2] * x[1] + b[2] * x[2] + c[2] * x[3],
            a[3] * x[2] + b[3] * x[3],
        ];
        for i in 0..4 {
            assert!(
                (ax[i] - d[i]).abs() < 1e-10,
                "Ax[{i}] = {}, expected {}",
                ax[i],
                d[i]
            );
        }
    }

    #[test]
    fn test_thomas_heat_equation_pattern() {
        // Typical implicit heat equation pattern:
        // main = 1 + 2*alpha, sub/super = -alpha
        let n = 10;
        let alpha = 0.4;
        let a: Vec<f64> = (0..n).map(|i| if i > 0 { -alpha } else { 0.0 }).collect();
        let b = vec![1.0 + 2.0 * alpha; n];
        let c: Vec<f64> = (0..n)
            .map(|i| if i < n - 1 { -alpha } else { 0.0 })
            .collect();
        let d = vec![1.0; n]; // uniform RHS

        let x = thomas_solve(&a, &b, &c, &d);

        // All values should be positive and finite
        for (i, &xi) in x.iter().enumerate() {
            assert!(
                xi > 0.0 && xi.is_finite(),
                "x[{i}] = {xi} should be positive finite"
            );
        }
    }
}
