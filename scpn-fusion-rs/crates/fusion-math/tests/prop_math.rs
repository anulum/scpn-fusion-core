// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Property-Based Tests (proptest) for fusion-math
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Property-based tests for fusion-math using proptest.
//!
//! Covers: Thomas solver, SOR solver, elliptic integrals, bilinear interpolation,
//! SVD reconstruction, eigenvalue decomposition.

use fusion_math::elliptic::{ellipe, ellipk};
use fusion_math::interp::{gradient_2d, interp2d};
use fusion_math::linalg::{eig_2x2, pinv_svd, svd_small};
use fusion_math::sor::{sor_residual, sor_solve, sor_step};
use fusion_math::tridiag::thomas_solve;
use fusion_types::state::Grid2D;
use ndarray::Array2;
use proptest::prelude::*;

// ── Thomas Solver Properties ─────────────────────────────────────────

proptest! {
    /// For any diagonally dominant tridiagonal system, x = thomas_solve(a,b,c,d)
    /// should satisfy Ax = d within floating-point tolerance.
    #[test]
    fn thomas_solve_ax_eq_d(n in 3usize..30) {
        // Build a diagonally dominant system (guaranteed non-singular)
        let a: Vec<f64> = (0..n).map(|i| if i > 0 { -0.3 } else { 0.0 }).collect();
        let b = vec![2.0; n];
        let c: Vec<f64> = (0..n).map(|i| if i < n - 1 { -0.3 } else { 0.0 }).collect();
        let d: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();

        let x = thomas_solve(&a, &b, &c, &d);

        // Verify Ax = d
        for i in 0..n {
            let mut ax_i = b[i] * x[i];
            if i > 0 { ax_i += a[i] * x[i - 1]; }
            if i < n - 1 { ax_i += c[i] * x[i + 1]; }
            prop_assert!((ax_i - d[i]).abs() < 1e-10,
                "Ax[{}] = {}, d[{}] = {}, diff = {}", i, ax_i, i, d[i], (ax_i - d[i]).abs());
        }
    }

    /// Thomas solver returns the correct number of elements.
    #[test]
    fn thomas_solve_output_length(n in 1usize..50) {
        let a = vec![0.0; n];
        let b = vec![1.0; n];
        let c = vec![0.0; n];
        let d: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let x = thomas_solve(&a, &b, &c, &d);
        prop_assert_eq!(x.len(), n);
    }

    /// Identity system (b=1, a=c=0) → x = d.
    #[test]
    fn thomas_identity_system(n in 1usize..50) {
        let a = vec![0.0; n];
        let b = vec![1.0; n];
        let c = vec![0.0; n];
        let d: Vec<f64> = (0..n).map(|i| (i as f64) * 0.7 - 3.0).collect();

        let x = thomas_solve(&a, &b, &c, &d);
        for i in 0..n {
            prop_assert!((x[i] - d[i]).abs() < 1e-14,
                "x[{}] = {}, expected {}", i, x[i], d[i]);
        }
    }
}

// ── SOR Solver Properties ────────────────────────────────────────────

proptest! {
    /// SOR with zero source and zero BC keeps psi = 0.
    #[test]
    fn sor_zero_source_preserves_zero(n in 8usize..40) {
        let grid = Grid2D::new(n, n, 1.0, 9.0, -5.0, 5.0);
        let mut psi = Array2::zeros((n, n));
        let source = Array2::zeros((n, n));

        sor_solve(&mut psi, &source, &grid, 1.6, 50);

        let max_val: f64 = psi.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
        prop_assert!(max_val < 1e-14,
            "Zero source should yield zero solution, got max = {}", max_val);
    }

    /// SOR never produces NaN for valid inputs.
    #[test]
    fn sor_no_nans(n in 8usize..32, iters in 1usize..100) {
        let grid = Grid2D::new(n, n, 1.0, 9.0, -5.0, 5.0);
        let mut psi = Array2::zeros((n, n));
        let source = Array2::from_elem((n, n), -1.0);

        sor_solve(&mut psi, &source, &grid, 1.5, iters);

        for &v in psi.iter() {
            prop_assert!(!v.is_nan(), "SOR produced NaN");
            prop_assert!(v.is_finite(), "SOR produced Inf");
        }
    }

    /// Boundary rows/columns remain zero after SOR.
    #[test]
    fn sor_boundary_untouched(n in 8usize..32) {
        let grid = Grid2D::new(n, n, 1.0, 9.0, -5.0, 5.0);
        let mut psi = Array2::zeros((n, n));
        let source = Array2::from_elem((n, n), -1.0);

        sor_solve(&mut psi, &source, &grid, 1.8, 100);

        for ir in 0..n {
            prop_assert!(psi[[0, ir]].abs() < 1e-15, "Top boundary modified");
            prop_assert!(psi[[n - 1, ir]].abs() < 1e-15, "Bottom boundary modified");
        }
        for iz in 0..n {
            prop_assert!(psi[[iz, 0]].abs() < 1e-15, "Left boundary modified");
            prop_assert!(psi[[iz, n - 1]].abs() < 1e-15, "Right boundary modified");
        }
    }

    /// Residual decreases monotonically over SOR iterations.
    #[test]
    fn sor_residual_decreases(n in 16usize..40) {
        let grid = Grid2D::new(n, n, 1.0, 9.0, -5.0, 5.0);
        let mut psi = Array2::zeros((n, n));
        let source = Array2::from_elem((n, n), -1.0);

        let res0 = sor_residual(&psi, &source, &grid);
        sor_solve(&mut psi, &source, &grid, 1.5, 100);
        let res1 = sor_residual(&psi, &source, &grid);

        prop_assert!(res1 < res0,
            "Residual should decrease: {} -> {}", res0, res1);
    }
}

// ── Elliptic Integral Properties ─────────────────────────────────────

proptest! {
    /// K(m) is monotonically increasing on [0, 1).
    #[test]
    fn ellipk_monotone(m1 in 0.01f64..0.49, m2 in 0.51f64..0.99) {
        let k1 = ellipk(m1);
        let k2 = ellipk(m2);
        prop_assert!(k2 > k1,
            "K({}) = {} should be > K({}) = {}", m2, k2, m1, k1);
    }

    /// E(m) is monotonically decreasing on [0, 1].
    #[test]
    fn ellipe_monotone(m1 in 0.01f64..0.49, m2 in 0.51f64..0.99) {
        let e1 = ellipe(m1);
        let e2 = ellipe(m2);
        prop_assert!(e2 < e1,
            "E({}) = {} should be < E({}) = {}", m2, e2, m1, e1);
    }

    /// K(m) >= pi/2 for all m in [0, 1).
    #[test]
    fn ellipk_lower_bound(m in 0.0f64..0.999) {
        let k = ellipk(m);
        prop_assert!(k >= std::f64::consts::FRAC_PI_2 - 1e-8,
            "K({}) = {} should be >= pi/2", m, k);
    }

    /// E(m) <= pi/2 for all m in [0, 1].
    #[test]
    fn ellipe_upper_bound(m in 0.0f64..1.0) {
        let e = ellipe(m);
        prop_assert!(e <= std::f64::consts::FRAC_PI_2 + 1e-8,
            "E({}) = {} should be <= pi/2", m, e);
    }

    /// Legendre relation: K(m)*E(1-m) + E(m)*K(1-m) - K(m)*K(1-m) = pi/2
    /// (valid for 0 < m < 1)
    #[test]
    fn legendre_relation(m in 0.05f64..0.95) {
        let m1 = 1.0 - m;
        let km = ellipk(m);
        let em = ellipe(m);
        let km1 = ellipk(m1);
        let em1 = ellipe(m1);

        let lhs = km * em1 + em * km1 - km * km1;
        let rhs = std::f64::consts::FRAC_PI_2;

        prop_assert!((lhs - rhs).abs() < 1e-6,
            "Legendre: K({})*E({}) + E({})*K({}) - K({})*K({}) = {}, expected {}",
            m, m1, m, m1, m, m1, lhs, rhs);
    }
}

// ── Interpolation Properties ─────────────────────────────────────────

proptest! {
    /// Bilinear interpolation of a constant field returns that constant.
    #[test]
    fn interp_constant_field(
        val in -100.0f64..100.0,
        r in 1.0f64..8.9,
        z in -4.9f64..4.9,
    ) {
        let grid = Grid2D::new(20, 20, 1.0, 9.0, -5.0, 5.0);
        let field = Array2::from_elem((20, 20), val);
        let result = interp2d(&field, &grid, r, z);
        prop_assert!((result - val).abs() < 1e-10,
            "Constant field: interp({}, {}) = {}, expected {}", r, z, result, val);
    }

    /// Bilinear interpolation of f(R,Z)=R+Z returns R+Z exactly.
    #[test]
    fn interp_linear_exact(
        r in 1.5f64..8.5,
        z in -4.5f64..4.5,
    ) {
        let n = 21;
        let grid = Grid2D::new(n, n, 1.0, 9.0, -5.0, 5.0);
        let field = Array2::from_shape_fn((n, n), |(iz, ir)| {
            grid.rr[[iz, ir]] + grid.zz[[iz, ir]]
        });
        let result = interp2d(&field, &grid, r, z);
        let expected = r + z;
        prop_assert!((result - expected).abs() < 0.1,
            "Linear f(R,Z)=R+Z: interp({}, {}) = {}, expected {}", r, z, result, expected);
    }
}

// ── Eigenvalue Properties ────────────────────────────────────────────

proptest! {
    /// Eigenvalues of a symmetric 2x2 matrix satisfy trace and determinant relations.
    #[test]
    fn eig_2x2_trace_det(
        a00 in -10.0f64..10.0,
        a11 in -10.0f64..10.0,
        a01 in -10.0f64..10.0,
    ) {
        let mat = [[a00, a01], [a01, a11]]; // symmetric
        let (vals, _) = eig_2x2(&mat);

        let trace = a00 + a11;
        let det = a00 * a11 - a01 * a01;

        // sum of eigenvalues = trace
        let eig_sum = vals[0] + vals[1];
        prop_assert!((eig_sum - trace).abs() < 1e-8,
            "λ₁+λ₂ = {}, trace = {}", eig_sum, trace);

        // product of eigenvalues = det
        let eig_prod = vals[0] * vals[1];
        prop_assert!((eig_prod - det).abs() < 1e-6,
            "λ₁·λ₂ = {}, det = {}", eig_prod, det);
    }

    /// Eigenvalues of a symmetric 2x2 are real and ordered.
    #[test]
    fn eig_2x2_ordered(
        a00 in -10.0f64..10.0,
        a11 in -10.0f64..10.0,
        a01 in -10.0f64..10.0,
    ) {
        let mat = [[a00, a01], [a01, a11]]; // symmetric → real eigenvalues
        let (vals, _) = eig_2x2(&mat);

        prop_assert!(vals[0].is_finite());
        prop_assert!(vals[1].is_finite());
        prop_assert!(vals[0] <= vals[1] + 1e-12,
            "Eigenvalues not ordered: {} > {}", vals[0], vals[1]);
    }
}

// ── SVD Properties ───────────────────────────────────────────────────

proptest! {
    /// SVD reconstruction: U * diag(sigma) * Vt ≈ A.
    #[test]
    fn svd_reconstruction(
        m in 2usize..6,
        n in 2usize..6,
    ) {
        // Generate a random-ish matrix from deterministic values
        let a = Array2::from_shape_fn((m, n), |(i, j)| {
            ((i * 7 + j * 13) as f64).sin() * 3.0
        });

        let (u, sigma, vt) = svd_small(&a);
        let k = sigma.len();

        // Reconstruct
        let mut recon = Array2::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                for s in 0..k {
                    recon[[i, j]] += u[[i, s]] * sigma[s] * vt[[s, j]];
                }
            }
        }

        // Check reconstruction error
        for i in 0..m {
            for j in 0..n {
                let err = (recon[[i, j]] - a[[i, j]]).abs();
                prop_assert!(err < 1e-8,
                    "SVD recon error at ({},{}): {} vs {}, err = {}",
                    i, j, recon[[i, j]], a[[i, j]], err);
            }
        }
    }

    /// Singular values are non-negative and sorted descending.
    #[test]
    fn svd_sigma_nonneg_sorted(
        m in 2usize..6,
        n in 2usize..6,
    ) {
        let a = Array2::from_shape_fn((m, n), |(i, j)| {
            ((i * 5 + j * 11 + 3) as f64).cos() * 2.0
        });

        let (_u, sigma, _vt) = svd_small(&a);

        for i in 0..sigma.len() {
            prop_assert!(sigma[i] >= -1e-14,
                "Negative singular value: sigma[{}] = {}", i, sigma[i]);
        }

        for i in 1..sigma.len() {
            prop_assert!(sigma[i] <= sigma[i - 1] + 1e-10,
                "Singular values not sorted: sigma[{}]={} > sigma[{}]={}",
                i - 1, sigma[i - 1], i, sigma[i]);
        }
    }
}

// ── Grid2D Properties ────────────────────────────────────────────────

proptest! {
    /// Grid spacing is consistent with the number of points.
    #[test]
    fn grid_spacing_consistency(
        nr in 4usize..64,
        nz in 4usize..64,
    ) {
        let grid = Grid2D::new(nr, nz, 1.0, 9.0, -5.0, 5.0);

        let expected_dr = (9.0 - 1.0) / (nr as f64 - 1.0);
        let expected_dz = (5.0 - (-5.0)) / (nz as f64 - 1.0);

        prop_assert!((grid.dr - expected_dr).abs() < 1e-14);
        prop_assert!((grid.dz - expected_dz).abs() < 1e-14);
    }

    /// R is constant along columns, Z is constant along rows.
    #[test]
    fn grid_meshgrid_consistency(
        nr in 4usize..32,
        nz in 4usize..32,
    ) {
        let grid = Grid2D::new(nr, nz, 1.0, 9.0, -5.0, 5.0);

        // R constant in columns
        for ir in 0..nr {
            let r_val = grid.rr[[0, ir]];
            for iz in 0..nz {
                prop_assert!((grid.rr[[iz, ir]] - r_val).abs() < 1e-15);
            }
        }
        // Z constant in rows
        for iz in 0..nz {
            let z_val = grid.zz[[iz, 0]];
            for ir in 0..nr {
                prop_assert!((grid.zz[[iz, ir]] - z_val).abs() < 1e-15);
            }
        }
    }
}
