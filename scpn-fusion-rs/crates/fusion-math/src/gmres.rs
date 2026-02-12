// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — GMRES
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Restarted GMRES(m) Krylov subspace solver for the Grad-Shafranov
//! elliptic operator.
//!
//! GMRES (Generalised Minimal RESidual) builds an orthonormal Krylov
//! basis via Arnoldi iteration with modified Gram-Schmidt, then solves
//! the projected least-squares problem using Givens rotations on the
//! upper Hessenberg matrix.  When the basis reaches size `m` without
//! convergence the solver restarts from the current approximate
//! solution.
//!
//! A left-preconditioner is applied: instead of solving `A x = b`, we
//! solve `M⁻¹ A x = M⁻¹ b`, where `M⁻¹` is approximated by a few
//! SOR sweeps.
//!
//! The matrix-vector product uses the same 5-point Grad-Shafranov
//! stencil with toroidal 1/R corrections as [`crate::sor`].

use fusion_types::state::Grid2D;
use ndarray::Array2;

use crate::sor::sor_step;

// ───────────────────────────── configuration ─────────────────────────

/// Configuration for the GMRES(m) solver.
#[derive(Debug, Clone)]
pub struct GmresConfig {
    /// Krylov subspace dimension before restart (default: 30).
    pub restart: usize,
    /// Maximum number of outer (restart) iterations (default: 100).
    pub max_iter: usize,
    /// Convergence tolerance on the relative residual norm (default: 1e-8).
    pub tol: f64,
    /// Number of SOR sweeps used by the left preconditioner (default: 3).
    pub precond_sweeps: usize,
    /// SOR relaxation factor for the preconditioner (default: 1.5).
    pub precond_omega: f64,
}

impl Default for GmresConfig {
    fn default() -> Self {
        GmresConfig {
            restart: 30,
            max_iter: 100,
            tol: 1e-8,
            precond_sweeps: 3,
            precond_omega: 1.5,
        }
    }
}

/// Result of a GMRES solve.
#[derive(Debug, Clone)]
pub struct GmresResult {
    /// Total number of matrix-vector products (inner iterations summed
    /// over all restarts).
    pub iterations: usize,
    /// Final L2 residual norm.
    pub residual: f64,
    /// Whether convergence was achieved.
    pub converged: bool,
}

// ───────────────────────── helper: flat indexing ─────────────────────

/// Number of interior unknowns for a [nz, nr] grid.
#[inline]
fn interior_len(nz: usize, nr: usize) -> usize {
    if nz < 3 || nr < 3 {
        return 0;
    }
    (nz - 2) * (nr - 2)
}

/// Map interior (iz, ir) → flat index.
#[inline]
fn flat_index(iz: usize, ir: usize, nr: usize) -> usize {
    (iz - 1) * (nr - 2) + (ir - 1)
}

/// Map flat index → interior (iz, ir).
#[inline]
#[allow(dead_code)]
fn grid_index(k: usize, nr: usize) -> (usize, usize) {
    let cols = nr - 2;
    let iz = k / cols + 1;
    let ir = k % cols + 1;
    (iz, ir)
}

// ─────────────────────── helper: grid ↔ flat vector ─────────────────

/// Extract the interior of an [nz, nr] array into a flat Vec.
fn grid_to_vec(arr: &Array2<f64>, nz: usize, nr: usize) -> Vec<f64> {
    let n = interior_len(nz, nr);
    let mut v = Vec::with_capacity(n);
    for iz in 1..nz - 1 {
        for ir in 1..nr - 1 {
            v.push(arr[[iz, ir]]);
        }
    }
    v
}

/// Scatter a flat Vec back into the interior of an [nz, nr] array.
/// Boundary values are left untouched.
fn vec_to_grid(v: &[f64], arr: &mut Array2<f64>, nz: usize, nr: usize) {
    for iz in 1..nz - 1 {
        for ir in 1..nr - 1 {
            arr[[iz, ir]] = v[flat_index(iz, ir, nr)];
        }
    }
}

// ─────────────────────── Grad-Shafranov matvec ──────────────────────

/// Apply the Grad-Shafranov operator to the interior of `x_grid` and
/// write the result into the flat vector `out`.
///
/// The stencil is identical to [`crate::sor`]:
///   - `c_r_plus  = 1/dr² - 1/(2R·dr)`
///   - `c_r_minus = 1/dr² + 1/(2R·dr)`
///   - `c_z       = 1/dz²`
///   - `center    = 2/dr² + 2/dz²`
///   - `(A·x)_ij  = center·x_ij - c_z·(x_{i+1,j}+x_{i-1,j})
///                   - c_r_plus·x_{i,j+1} - c_r_minus·x_{i,j-1}`
fn gs_matvec(x_grid: &Array2<f64>, grid: &Grid2D, out: &mut [f64]) {
    let nz = grid.nz;
    let nr = grid.nr;
    let dr = grid.dr;
    let dz = grid.dz;
    let dr_sq = dr * dr;
    let dz_sq = dz * dz;

    for iz in 1..nz - 1 {
        for ir in 1..nr - 1 {
            let r = grid.rr[[iz, ir]];

            let c_r_plus = 1.0 / dr_sq - 1.0 / (2.0 * r * dr);
            let c_r_minus = 1.0 / dr_sq + 1.0 / (2.0 * r * dr);
            let c_z = 1.0 / dz_sq;
            let center = 2.0 / dr_sq + 2.0 / dz_sq;

            let val = center * x_grid[[iz, ir]]
                - c_z * (x_grid[[iz + 1, ir]] + x_grid[[iz - 1, ir]])
                - c_r_plus * x_grid[[iz, ir + 1]]
                - c_r_minus * x_grid[[iz, ir - 1]];

            out[flat_index(iz, ir, nr)] = val;
        }
    }
}

/// Convenience wrapper: flat vector in → flat vector out, using a
/// scratch grid to hold the 2D form (boundaries zeroed).
fn gs_matvec_flat(x: &[f64], grid: &Grid2D, scratch: &mut Array2<f64>, out: &mut [f64]) {
    let nz = grid.nz;
    let nr = grid.nr;

    // Zero the scratch grid (boundaries must stay zero)
    scratch.fill(0.0);
    vec_to_grid(x, scratch, nz, nr);
    gs_matvec(scratch, grid, out);
}

// ────────────────────────── SOR preconditioner ──────────────────────

/// Apply the left preconditioner: approximately solve `A z = r` by
/// performing a few SOR sweeps starting from zero.  Returns the flat
/// interior of the approximate solution.
fn precondition(
    r_vec: &[f64],
    grid: &Grid2D,
    sweeps: usize,
    omega: f64,
    scratch_psi: &mut Array2<f64>,
    scratch_src: &mut Array2<f64>,
) -> Vec<f64> {
    let nz = grid.nz;
    let nr = grid.nr;

    // Set up the source grid from the flat residual vector
    scratch_src.fill(0.0);
    vec_to_grid(r_vec, scratch_src, nz, nr);

    // Start from zero
    scratch_psi.fill(0.0);

    for _ in 0..sweeps {
        sor_step(scratch_psi, scratch_src, grid, omega);
    }

    grid_to_vec(scratch_psi, nz, nr)
}

// ───────────────────────── BLAS-like helpers ─────────────────────────

/// Euclidean (L2) norm of a slice.
#[inline]
fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Dot product.
#[inline]
fn vec_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// `y = y + alpha * x` (axpy).
#[inline]
fn vec_axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

/// `y = alpha * x` (scale-copy).
#[inline]
fn vec_scale(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi = alpha * xi;
    }
}

/// `out = a - b`.
#[inline]
fn vec_sub(a: &[f64], b: &[f64], out: &mut [f64]) {
    for ((oi, &ai), &bi) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *oi = ai - bi;
    }
}

// ───────────────────── Givens rotation helpers ──────────────────────

/// A single Givens rotation storing (c, s) such that
/// ```text
/// | c  s | | a |   | r |
/// |-s  c | | b | = | 0 |
/// ```
#[derive(Clone, Copy)]
struct GivensRotation {
    c: f64,
    s: f64,
}

impl GivensRotation {
    /// Compute the rotation that zeroes `b` in (a, b).
    fn compute(a: f64, b: f64) -> Self {
        if b.abs() < 1e-300 {
            GivensRotation { c: 1.0, s: 0.0 }
        } else if b.abs() > a.abs() {
            let tau = -a / b;
            let s = 1.0 / (1.0 + tau * tau).sqrt();
            let c = s * tau;
            GivensRotation { c, s }
        } else {
            let tau = -b / a;
            let c = 1.0 / (1.0 + tau * tau).sqrt();
            let s = c * tau;
            GivensRotation { c, s }
        }
    }

    /// Apply this rotation to (a, b) in place.
    #[inline]
    fn apply(&self, a: &mut f64, b: &mut f64) {
        let ta = *a;
        let tb = *b;
        *a = self.c * ta - self.s * tb;
        *b = self.s * ta + self.c * tb;
    }
}

// ─────────────────────────── main solver ─────────────────────────────

/// Solve the Grad-Shafranov system `A psi = source` using restarted
/// GMRES(m) with a left SOR preconditioner.
///
/// `psi` is the initial guess on entry and the solution on exit.
/// Boundary rows/columns are **never** modified.
///
/// # Algorithm
///
/// ```text
/// for each restart cycle:
///   r = source - A·psi            (residual)
///   z = M⁻¹ r                    (precondition)
///   beta = ||z||₂
///   V[0] = z / beta
///   for j = 0 .. m-1:             (Arnoldi)
///     w = M⁻¹ A V[j]
///     for i = 0 .. j:             (modified Gram-Schmidt)
///       H[i,j] = <w, V[i]>
///       w -= H[i,j] V[i]
///     H[j+1,j] = ||w||₂
///     V[j+1]   = w / H[j+1,j]
///     apply previous Givens to H[:,j]
///     compute new Givens to zero H[j+1,j]
///     update residual norm estimate
///     if converged: break
///   solve upper triangular system for y
///   psi += V · y
/// ```
pub fn gmres_solve(
    psi: &mut Array2<f64>,
    source: &Array2<f64>,
    grid: &Grid2D,
    config: &GmresConfig,
) -> GmresResult {
    let nz = grid.nz;
    let nr = grid.nr;
    let n = interior_len(nz, nr);

    if n == 0 {
        return GmresResult {
            iterations: 0,
            residual: 0.0,
            converged: true,
        };
    }

    let m = config.restart.min(n); // Krylov dimension cannot exceed n

    // Flat representations
    let b_flat = grid_to_vec(source, nz, nr);

    // Scratch grids for matvec and preconditioner
    let mut scratch_mv = Array2::zeros((nz, nr));
    let mut scratch_psi = Array2::zeros((nz, nr));
    let mut scratch_src = Array2::zeros((nz, nr));
    let mut av = vec![0.0; n]; // result of A*v

    let mut total_iters: usize = 0;
    let final_residual: f64;

    // Compute initial residual norm for relative tolerance
    let mut x_flat = grid_to_vec(psi, nz, nr);
    let mut r_flat = vec![0.0; n];
    gs_matvec_flat(&x_flat, grid, &mut scratch_mv, &mut av);
    vec_sub(&b_flat, &av, &mut r_flat);
    let initial_res_norm = vec_norm(&r_flat);

    if initial_res_norm < 1e-300 {
        return GmresResult {
            iterations: 0,
            residual: initial_res_norm,
            converged: true,
        };
    }

    let abs_tol = config.tol * initial_res_norm;

    // ───── outer restart loop ─────
    for _restart in 0..config.max_iter {
        // Recompute residual r = b - A*x
        x_flat = grid_to_vec(psi, nz, nr);
        gs_matvec_flat(&x_flat, grid, &mut scratch_mv, &mut av);
        vec_sub(&b_flat, &av, &mut r_flat);

        // Precondition: z = M⁻¹ r
        let z = precondition(
            &r_flat,
            grid,
            config.precond_sweeps,
            config.precond_omega,
            &mut scratch_psi,
            &mut scratch_src,
        );

        let beta = vec_norm(&z);
        if beta < 1e-300 {
            final_residual = vec_norm(&r_flat);
            return GmresResult {
                iterations: total_iters,
                residual: final_residual,
                converged: final_residual < abs_tol,
            };
        }

        // Krylov basis V[0..m+1], each of length n
        let mut v_basis: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        {
            let mut v0 = vec![0.0; n];
            vec_scale(1.0 / beta, &z, &mut v0);
            v_basis.push(v0);
        }

        // Upper Hessenberg matrix H[(m+1) x m] stored column-major
        // H[i][j] => h_store[j * (m+1) + i]
        let h_rows = m + 1;
        let mut h_store = vec![0.0; h_rows * m];

        // Givens rotations accumulated so far
        let mut givens: Vec<GivensRotation> = Vec::with_capacity(m);

        // Right-hand side of the Hessenberg least-squares: g = beta * e_1
        let mut g = vec![0.0; m + 1];
        g[0] = beta;

        let mut converged_inner = false;
        let mut inner_iters: usize = 0;

        // ───── Arnoldi iteration ─────
        for j in 0..m {
            inner_iters = j + 1;
            total_iters += 1;

            // w = A * V[j]
            gs_matvec_flat(&v_basis[j], grid, &mut scratch_mv, &mut av);

            // Precondition: w = M⁻¹ (A V[j])
            let mut w = precondition(
                &av,
                grid,
                config.precond_sweeps,
                config.precond_omega,
                &mut scratch_psi,
                &mut scratch_src,
            );

            // Modified Gram-Schmidt orthogonalisation
            for i in 0..=j {
                let h_ij = vec_dot(&w, &v_basis[i]);
                h_store[j * h_rows + i] = h_ij;
                vec_axpy(-h_ij, &v_basis[i], &mut w);
            }

            let h_jp1_j = vec_norm(&w);
            h_store[j * h_rows + (j + 1)] = h_jp1_j;

            // Check for breakdown (lucky or degenerate)
            if h_jp1_j > 1e-300 {
                let mut vj1 = vec![0.0; n];
                vec_scale(1.0 / h_jp1_j, &w, &mut vj1);
                v_basis.push(vj1);
            } else {
                // Happy breakdown: residual is zero in the Krylov subspace
                v_basis.push(vec![0.0; n]);
            }

            // Apply all previous Givens rotations to column j of H
            for (i, rot) in givens.iter().enumerate() {
                let a_ptr = j * h_rows + i;
                let b_ptr = j * h_rows + i + 1;
                let mut ha = h_store[a_ptr];
                let mut hb = h_store[b_ptr];
                rot.apply(&mut ha, &mut hb);
                h_store[a_ptr] = ha;
                h_store[b_ptr] = hb;
            }

            // Compute new Givens rotation to zero H[j+1, j]
            let rot =
                GivensRotation::compute(h_store[j * h_rows + j], h_store[j * h_rows + (j + 1)]);
            {
                let a_ptr = j * h_rows + j;
                let b_ptr = j * h_rows + (j + 1);
                let mut ha = h_store[a_ptr];
                let mut hb = h_store[b_ptr];
                rot.apply(&mut ha, &mut hb);
                h_store[a_ptr] = ha;
                h_store[b_ptr] = hb;
            }

            // Apply rotation to the rhs vector g
            {
                let mut ga = g[j];
                let mut gb = g[j + 1];
                rot.apply(&mut ga, &mut gb);
                g[j] = ga;
                g[j + 1] = gb;
            }

            givens.push(rot);

            // The residual norm estimate is |g[j+1]|
            let res_est = g[j + 1].abs();

            if res_est < abs_tol {
                converged_inner = true;
                break;
            }

            // Happy breakdown: can't extend Krylov space
            if h_jp1_j < 1e-300 {
                converged_inner = true;
                break;
            }
        }

        // ───── solve the upper triangular system H y = g ─────
        let k = inner_iters; // number of Arnoldi steps taken
        let mut y = vec![0.0; k];

        // Back-substitution
        for i in (0..k).rev() {
            let mut sum = g[i];
            for jj in (i + 1)..k {
                sum -= h_store[jj * h_rows + i] * y[jj];
            }
            let diag = h_store[i * h_rows + i];
            if diag.abs() > 1e-300 {
                y[i] = sum / diag;
            } else {
                y[i] = 0.0;
            }
        }

        // ───── update solution: x = x + V * y ─────
        x_flat = grid_to_vec(psi, nz, nr);
        for i in 0..k {
            vec_axpy(y[i], &v_basis[i], &mut x_flat);
        }
        vec_to_grid(&x_flat, psi, nz, nr);

        if converged_inner {
            // Compute true residual for the result
            let mut true_r = vec![0.0; n];
            gs_matvec_flat(&x_flat, grid, &mut scratch_mv, &mut av);
            vec_sub(&b_flat, &av, &mut true_r);
            final_residual = vec_norm(&true_r);

            return GmresResult {
                iterations: total_iters,
                residual: final_residual,
                converged: true,
            };
        }
    }

    // Exhausted restarts — compute true residual
    x_flat = grid_to_vec(psi, nz, nr);
    let mut true_r = vec![0.0; n];
    gs_matvec_flat(&x_flat, grid, &mut scratch_mv, &mut av);
    vec_sub(&b_flat, &av, &mut true_r);
    final_residual = vec_norm(&true_r);

    GmresResult {
        iterations: total_iters,
        residual: final_residual,
        converged: final_residual < abs_tol,
    }
}

// ═══════════════════════════════ tests ═══════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multigrid::{multigrid_solve, MultigridConfig};
    use ndarray::Array2;

    #[test]
    fn test_gmres_convergence_33() {
        let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
        let mut psi = Array2::zeros((33, 33));
        let source = Array2::from_elem((33, 33), -1.0);

        let result = gmres_solve(&mut psi, &source, &grid, &GmresConfig::default());

        assert!(
            result.converged,
            "GMRES should converge on 33x33: residual = {}, iters = {}",
            result.residual, result.iterations
        );
        // Centre value must be non-trivial
        assert!(
            psi[[16, 16]].abs() > 1e-10,
            "Centre of solution should be non-zero: {}",
            psi[[16, 16]]
        );
        // No NaN allowed
        assert!(
            !psi.iter().any(|v| v.is_nan()),
            "Solution must not contain NaN"
        );
    }

    #[test]
    fn test_gmres_convergence_65() {
        let grid = Grid2D::new(65, 65, 1.0, 9.0, -5.0, 5.0);
        let mut psi = Array2::zeros((65, 65));
        let source = Array2::from_elem((65, 65), -1.0);

        let config = GmresConfig {
            restart: 40,
            max_iter: 200,
            tol: 1e-6,
            precond_sweeps: 5,
            precond_omega: 1.5,
        };

        let result = gmres_solve(&mut psi, &source, &grid, &config);

        assert!(
            result.converged,
            "GMRES should converge on 65x65: residual = {}, iters = {}",
            result.residual, result.iterations
        );
        assert!(
            psi[[32, 32]].abs() > 1e-10,
            "Centre of 65x65 solution should be non-zero"
        );
    }

    #[test]
    fn test_gmres_zero_source() {
        let grid = Grid2D::new(17, 17, 1.0, 9.0, -5.0, 5.0);
        let mut psi = Array2::zeros((17, 17));
        let source = Array2::zeros((17, 17));

        let result = gmres_solve(&mut psi, &source, &grid, &GmresConfig::default());

        let max_val: f64 = psi.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
        assert!(
            max_val < 1e-14,
            "Zero source should give zero solution: max = {}",
            max_val
        );
        assert!(result.converged, "Zero source should converge immediately");
    }

    #[test]
    fn test_gmres_boundary_preserved() {
        let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
        let mut psi = Array2::zeros((33, 33));
        let source = Array2::from_elem((33, 33), -1.0);

        gmres_solve(&mut psi, &source, &grid, &GmresConfig::default());

        // Top and bottom boundary rows
        for ir in 0..33 {
            assert!(
                psi[[0, ir]].abs() < 1e-15,
                "Top boundary modified at col {}: {}",
                ir,
                psi[[0, ir]]
            );
            assert!(
                psi[[32, ir]].abs() < 1e-15,
                "Bottom boundary modified at col {}: {}",
                ir,
                psi[[32, ir]]
            );
        }
        // Left and right boundary columns
        for iz in 0..33 {
            assert!(
                psi[[iz, 0]].abs() < 1e-15,
                "Left boundary modified at row {}: {}",
                iz,
                psi[[iz, 0]]
            );
            assert!(
                psi[[iz, 32]].abs() < 1e-15,
                "Right boundary modified at row {}: {}",
                iz,
                psi[[iz, 32]]
            );
        }
    }

    #[test]
    fn test_gmres_reduces_residual() {
        use crate::sor::sor_residual;

        let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
        let mut psi = Array2::zeros((33, 33));
        let source = Array2::from_elem((33, 33), -1.0);

        let res_before = sor_residual(&psi, &source, &grid);

        let result = gmres_solve(&mut psi, &source, &grid, &GmresConfig::default());

        let res_after = sor_residual(&psi, &source, &grid);

        assert!(
            res_after < res_before,
            "Residual should decrease: {} -> {} (GMRES L2 residual = {})",
            res_before,
            res_after,
            result.residual
        );
    }

    #[test]
    fn test_gmres_matches_multigrid() {
        let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
        let source = Array2::from_elem((33, 33), -1.0);

        // Solve with multigrid
        let mut psi_mg = Array2::zeros((33, 33));
        multigrid_solve(
            &mut psi_mg,
            &source,
            &grid,
            &MultigridConfig::default(),
            30,
            1e-10,
        );

        // Solve with GMRES
        let mut psi_gmres = Array2::zeros((33, 33));
        let config = GmresConfig {
            restart: 30,
            max_iter: 200,
            tol: 1e-10,
            precond_sweeps: 5,
            precond_omega: 1.5,
        };
        gmres_solve(&mut psi_gmres, &source, &grid, &config);

        // Compare interior values (both solvers should find the same
        // solution to the linear system with zero Dirichlet BCs)
        let mut max_diff: f64 = 0.0;
        for iz in 1..32 {
            for ir in 1..32 {
                let diff = (psi_gmres[[iz, ir]] - psi_mg[[iz, ir]]).abs();
                max_diff = max_diff.max(diff);
            }
        }

        assert!(
            max_diff < 1e-4,
            "GMRES and multigrid solutions should agree within 1e-4: max diff = {}",
            max_diff
        );
    }
}
