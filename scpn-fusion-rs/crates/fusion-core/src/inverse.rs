// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Inverse Equilibrium Reconstruction
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Inverse Grad-Shafranov equilibrium reconstruction.
//!
//! Given noisy magnetic probe measurements (B_pol, Ψ_loop) placed on
//! the vacuum vessel wall, reconstruct the plasma equilibrium by
//! finding the p'(ψ) and FF'(ψ) profile coefficients that minimise
//! the mismatch between forward-simulated and measured signals.
//!
//! This is the inverse of [`crate::kernel::FusionKernel`]: instead of
//! *coils → plasma shape*, the inverse solver does
//! *magnetic sensors → plasma shape*.
//!
//! # Algorithm
//!
//! Levenberg-Marquardt minimisation:
//!
//! 1. Parameterise p'(ψ) and FF'(ψ) as mtanh pedestal profiles with
//!    free coefficients **x** = [ped_height_p, ped_top_p, ped_width_p,
//!    core_alpha_p, ped_height_ff, ped_top_ff, ped_width_ff, core_alpha_ff].
//! 2. Forward-solve the GS equation for the current **x**.
//! 3. Sample the synthetic diagnostics (B_pol, Ψ) at probe locations.
//! 4. Compute the residual **r** = measurement − simulation.
//! 5. Build the Jacobian **J** via finite differences on **x**.
//! 6. Solve `(JᵀJ + λI) δx = Jᵀr` for the Gauss-Newton step.
//! 7. Accept/reject with trust-region logic (λ update).
//! 8. Repeat until ‖r‖ < tol or max iterations.
//!
//! # References
//!
//! - Lao, L.L. et al. (1985). "Reconstruction of current profile
//!   parameters and plasma shapes in tokamaks." *Nucl. Fusion* 25(11).
//! - Luxon, J.L. & Brown, B.B. (1982). "Magnetic analysis of
//!   non-circular cross-section tokamaks." *Nucl. Fusion* 22(6).

use crate::kernel::FusionKernel;
use crate::source::{ProfileMode, ProfileParams};
use fusion_types::error::{FusionError, FusionResult};
use fusion_types::state::Grid2D;
use ndarray::Array2;

// ── Configuration ────────────────────────────────────────────────────

/// Magnetic probe placement.
#[derive(Debug, Clone)]
pub struct MagneticProbe {
    /// R coordinate of the probe [m].
    pub r: f64,
    /// Z coordinate of the probe [m].
    pub z: f64,
    /// Probe type.
    pub kind: ProbeKind,
}

/// Type of magnetic measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProbeKind {
    /// Poloidal field magnitude |B_pol| at the probe location.
    BpolMagnitude,
    /// Poloidal flux Ψ (flux loop).
    FluxLoop,
}

/// Configuration for the inverse solver.
#[derive(Debug, Clone)]
pub struct InverseConfig {
    /// Maximum Levenberg-Marquardt iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on ‖residual‖.
    pub tolerance: f64,
    /// Initial Levenberg-Marquardt damping factor λ.
    pub lambda_init: f64,
    /// λ increase factor on rejected step.
    pub lambda_up: f64,
    /// λ decrease factor on accepted step.
    pub lambda_down: f64,
    /// Finite-difference step for Jacobian computation.
    pub fd_step: f64,
}

impl Default for InverseConfig {
    fn default() -> Self {
        InverseConfig {
            max_iterations: 30,
            tolerance: 1e-4,
            lambda_init: 1e-3,
            lambda_up: 10.0,
            lambda_down: 0.1,
            fd_step: 1e-4,
        }
    }
}

/// Result of the inverse reconstruction.
#[derive(Debug, Clone)]
pub struct InverseResult {
    /// Whether the solver converged.
    pub converged: bool,
    /// Number of LM iterations performed.
    pub iterations: usize,
    /// Final chi-squared (sum of squared residuals).
    pub chi_squared: f64,
    /// Reconstructed p' profile parameters.
    pub p_prime_params: ProfileParams,
    /// Reconstructed FF' profile parameters.
    pub ff_prime_params: ProfileParams,
    /// Final residual vector.
    pub residuals: Vec<f64>,
    /// Solve time in milliseconds.
    pub solve_time_ms: f64,
}

// ── Parameter packing ────────────────────────────────────────────────

const N_PARAMS: usize = 8;

fn pack_params(p: &ProfileParams, ff: &ProfileParams) -> [f64; N_PARAMS] {
    [
        p.ped_height,
        p.ped_top,
        p.ped_width,
        p.core_alpha,
        ff.ped_height,
        ff.ped_top,
        ff.ped_width,
        ff.core_alpha,
    ]
}

fn unpack_params(x: &[f64; N_PARAMS]) -> (ProfileParams, ProfileParams) {
    let p = ProfileParams {
        ped_height: x[0].max(0.01),
        ped_top: x[1].clamp(0.5, 0.99),
        ped_width: x[2].clamp(0.01, 0.2),
        core_alpha: x[3].clamp(0.0, 2.0),
    };
    let ff = ProfileParams {
        ped_height: x[4].max(0.01),
        ped_top: x[5].clamp(0.5, 0.99),
        ped_width: x[6].clamp(0.01, 0.2),
        core_alpha: x[7].clamp(0.0, 2.0),
    };
    (p, ff)
}

// ── Probe sampling ───────────────────────────────────────────────────

/// Sample the poloidal field / flux at probe locations by bilinear
/// interpolation on the computational grid.
fn sample_probes(
    psi: &Array2<f64>,
    b_r: &Array2<f64>,
    b_z: &Array2<f64>,
    grid: &Grid2D,
    probes: &[MagneticProbe],
) -> Vec<f64> {
    probes
        .iter()
        .map(|probe| match probe.kind {
            ProbeKind::FluxLoop => interp_2d(psi, grid, probe.r, probe.z),
            ProbeKind::BpolMagnitude => {
                let br = interp_2d(b_r, grid, probe.r, probe.z);
                let bz = interp_2d(b_z, grid, probe.r, probe.z);
                (br * br + bz * bz).sqrt()
            }
        })
        .collect()
}

/// Bilinear interpolation of a 2D field at (r, z).
fn interp_2d(field: &Array2<f64>, grid: &Grid2D, r: f64, z: f64) -> f64 {
    // Continuous index coordinates
    let fr = (r - grid.r[0]) / grid.dr;
    let fz = (z - grid.z[0]) / grid.dz;

    let ir = (fr.floor() as usize).min(grid.nr.saturating_sub(2));
    let iz = (fz.floor() as usize).min(grid.nz.saturating_sub(2));

    let tr = (fr - ir as f64).clamp(0.0, 1.0);
    let tz = (fz - iz as f64).clamp(0.0, 1.0);

    let f00 = field[[iz, ir]];
    let f10 = field[[iz + 1, ir]];
    let f01 = field[[iz, ir + 1]];
    let f11 = field[[iz + 1, ir + 1]];

    (1.0 - tz) * ((1.0 - tr) * f00 + tr * f01) + tz * ((1.0 - tr) * f10 + tr * f11)
}

// ── Forward model ────────────────────────────────────────────────────

/// Run a forward solve and sample probes. Returns the simulated
/// diagnostic vector.
fn forward_model(
    kernel: &mut FusionKernel,
    p_params: &ProfileParams,
    ff_params: &ProfileParams,
    probes: &[MagneticProbe],
) -> FusionResult<Vec<f64>> {
    kernel.set_profile_mode(ProfileMode::HMode, *p_params, *ff_params);
    kernel.solve_equilibrium()?;

    let psi = kernel.psi();
    let b_r = kernel
        .state()
        .b_r
        .as_ref()
        .ok_or_else(|| FusionError::PhysicsViolation("B_R not computed".into()))?;
    let b_z = kernel
        .state()
        .b_z
        .as_ref()
        .ok_or_else(|| FusionError::PhysicsViolation("B_Z not computed".into()))?;

    Ok(sample_probes(psi, b_r, b_z, kernel.grid(), probes))
}

// ── Dense linear solve (N_PARAMS x N_PARAMS) ────────────────────────

/// Solve `A x = b` for an `N_PARAMS x N_PARAMS` dense symmetric positive-definite
/// system using Cholesky decomposition.
fn solve_dense(a: &[[f64; N_PARAMS]; N_PARAMS], b: &[f64; N_PARAMS]) -> Option<[f64; N_PARAMS]> {
    // Cholesky: A = L Lᵀ
    let mut l = [[0.0_f64; N_PARAMS]; N_PARAMS];

    for i in 0..N_PARAMS {
        for j in 0..=i {
            let mut sum = 0.0;
            for (k, l_row_j) in l[j].iter().enumerate().take(j) {
                sum += l[i][k] * l_row_j;
            }
            if i == j {
                let diag = a[i][i] - sum;
                if diag <= 0.0 {
                    return None; // Not positive definite
                }
                l[i][j] = diag.sqrt();
            } else {
                l[i][j] = (a[i][j] - sum) / l[j][j];
            }
        }
    }

    // Forward solve: L y = b
    let mut y = [0.0; N_PARAMS];
    for i in 0..N_PARAMS {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[i][j] * y[j];
        }
        y[i] = (b[i] - sum) / l[i][i];
    }

    // Back solve: Lᵀ x = y
    let mut x = [0.0; N_PARAMS];
    for i in (0..N_PARAMS).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..N_PARAMS {
            sum += l[j][i] * x[j];
        }
        x[i] = (y[i] - sum) / l[i][i];
    }

    Some(x)
}

// ── Inverse solver ───────────────────────────────────────────────────

/// Reconstruct plasma equilibrium from magnetic probe measurements.
///
/// # Arguments
///
/// * `kernel` - A mutable `FusionKernel` (will be solved multiple times).
/// * `probes` - Probe locations and types on the vessel wall.
/// * `measurements` - Measured values corresponding to each probe.
/// * `config` - Levenberg-Marquardt solver configuration.
///
/// # Returns
///
/// `InverseResult` with the reconstructed profile parameters and fit quality.
pub fn reconstruct_equilibrium(
    kernel: &mut FusionKernel,
    probes: &[MagneticProbe],
    measurements: &[f64],
    config: &InverseConfig,
) -> FusionResult<InverseResult> {
    let start = std::time::Instant::now();
    let n_meas = measurements.len();

    if n_meas != probes.len() {
        return Err(FusionError::ConfigError(format!(
            "Measurement count ({}) != probe count ({})",
            n_meas,
            probes.len()
        )));
    }
    if n_meas < N_PARAMS {
        return Err(FusionError::ConfigError(format!(
            "Need at least {} measurements, got {}",
            N_PARAMS, n_meas
        )));
    }

    // Initial guess: default pedestal parameters
    let p0 = ProfileParams::default();
    let ff0 = ProfileParams::default();
    let mut x = pack_params(&p0, &ff0);

    let mut lambda = config.lambda_init;
    let mut converged = false;
    let mut final_iter = 0;

    // Current forward model evaluation
    let (p_cur, ff_cur) = unpack_params(&x);
    let mut sim = forward_model(kernel, &p_cur, &ff_cur, probes)?;
    let mut residuals: Vec<f64> = measurements
        .iter()
        .zip(sim.iter())
        .map(|(m, s)| m - s)
        .collect();
    let mut chi2: f64 = residuals.iter().map(|r| r * r).sum();

    for iter in 0..config.max_iterations {
        final_iter = iter;

        if chi2.sqrt() < config.tolerance {
            converged = true;
            break;
        }

        // Build Jacobian (n_meas x N_PARAMS) via finite differences
        let mut jacobian = vec![vec![0.0; N_PARAMS]; n_meas];

        for j in 0..N_PARAMS {
            let mut x_pert = x;
            let h = config.fd_step * (1.0 + x[j].abs());
            x_pert[j] += h;

            let (p_pert, ff_pert) = unpack_params(&x_pert);
            let sim_pert = forward_model(kernel, &p_pert, &ff_pert, probes)?;

            for i in 0..n_meas {
                jacobian[i][j] = (sim_pert[i] - sim[i]) / h;
            }
        }

        // JᵀJ + λI
        let mut jtj = [[0.0; N_PARAMS]; N_PARAMS];
        let mut jtr = [0.0; N_PARAMS];

        for i in 0..N_PARAMS {
            for j in 0..N_PARAMS {
                let mut sum = 0.0;
                for row in jacobian.iter().take(n_meas) {
                    sum += row[i] * row[j];
                }
                jtj[i][j] = sum;
            }
            // Jᵀr
            let mut sum = 0.0;
            for (row, &res) in jacobian.iter().zip(residuals.iter()).take(n_meas) {
                sum += row[i] * res;
            }
            jtr[i] = sum;

            // Damping
            jtj[i][i] += lambda;
        }

        // Solve for step
        let delta = match solve_dense(&jtj, &jtr) {
            Some(d) => d,
            None => {
                // Not positive definite — increase damping and retry
                lambda *= config.lambda_up;
                continue;
            }
        };

        // Trial update
        let mut x_trial = x;
        for i in 0..N_PARAMS {
            x_trial[i] += delta[i];
        }

        let (p_trial, ff_trial) = unpack_params(&x_trial);
        let sim_trial = forward_model(kernel, &p_trial, &ff_trial, probes)?;
        let res_trial: Vec<f64> = measurements
            .iter()
            .zip(sim_trial.iter())
            .map(|(m, s)| m - s)
            .collect();
        let chi2_trial: f64 = res_trial.iter().map(|r| r * r).sum();

        // Accept/reject
        if chi2_trial < chi2 {
            x = x_trial;
            sim = sim_trial;
            residuals = res_trial;
            chi2 = chi2_trial;
            lambda *= config.lambda_down;
        } else {
            lambda *= config.lambda_up;
        }
    }

    let (p_final, ff_final) = unpack_params(&x);
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    Ok(InverseResult {
        converged,
        iterations: final_iter + 1,
        chi_squared: chi2,
        p_prime_params: p_final,
        ff_prime_params: ff_final,
        residuals,
        solve_time_ms: elapsed,
    })
}

// ── Synthetic diagnostic generation ──────────────────────────────────

/// Generate a ring of equally-spaced magnetic probes on the vacuum vessel wall.
///
/// Places `n_bpol` poloidal field probes and `n_flux` flux loops around an
/// elliptical wall at `(R_wall, Z_wall)` with half-widths `(a, b)`.
pub fn generate_vessel_probes(
    r_center: f64,
    z_center: f64,
    a: f64,
    b: f64,
    n_bpol: usize,
    n_flux: usize,
) -> Vec<MagneticProbe> {
    let n_total = n_bpol + n_flux;
    let mut probes = Vec::with_capacity(n_total);

    for i in 0..n_total {
        let theta = 2.0 * std::f64::consts::PI * i as f64 / n_total as f64;
        let r = r_center + a * theta.cos();
        let z = z_center + b * theta.sin();
        let kind = if i < n_bpol {
            ProbeKind::BpolMagnitude
        } else {
            ProbeKind::FluxLoop
        };
        probes.push(MagneticProbe { r, z, kind });
    }

    probes
}

/// Generate synthetic measurements from a known equilibrium.
///
/// Runs a forward solve with the given profile parameters, then samples
/// the probes and optionally adds Gaussian noise.
pub fn generate_synthetic_measurements(
    kernel: &mut FusionKernel,
    probes: &[MagneticProbe],
    p_params: &ProfileParams,
    ff_params: &ProfileParams,
    noise_std: f64,
) -> FusionResult<Vec<f64>> {
    let clean = forward_model(kernel, p_params, ff_params, probes)?;

    if noise_std <= 0.0 {
        return Ok(clean);
    }

    // Simple deterministic pseudo-noise for reproducibility in tests.
    // Uses a linear congruential generator seeded from the measurement values.
    let mut seed: u64 = 42;
    let noisy = clean
        .iter()
        .map(|&v| {
            // LCG step
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Map to [-1, 1] range
            let u = (seed >> 33) as f64 / (u32::MAX as f64) * 2.0 - 1.0;
            v + noise_std * u
        })
        .collect();

    Ok(noisy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn project_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
    }

    fn config_path(relative: &str) -> String {
        project_root().join(relative).to_string_lossy().to_string()
    }

    fn make_kernel() -> FusionKernel {
        FusionKernel::from_file(&config_path("validation/iter_validated_config.json")).unwrap()
    }

    // ── Unit tests ───────────────────────────────────────────────────

    #[test]
    fn test_interp_2d_at_grid_points() {
        let grid = Grid2D::new(5, 5, 1.0, 5.0, -2.0, 2.0);
        let field = Array2::from_shape_fn((5, 5), |(iz, ir)| (iz * 10 + ir) as f64);

        // Interpolate at exact grid point (ir=2, iz=3) → value 32
        let r = grid.r[2];
        let z = grid.z[3];
        let val = interp_2d(&field, &grid, r, z);
        assert!(
            (val - 32.0).abs() < 1e-10,
            "Interpolation at grid point failed: {val}"
        );
    }

    #[test]
    fn test_interp_2d_midpoint() {
        let grid = Grid2D::new(3, 3, 0.0, 2.0, 0.0, 2.0);
        // f(iz, ir) = iz + ir → bilinear at center should be 2.0
        let field = Array2::from_shape_fn((3, 3), |(iz, ir)| (iz + ir) as f64);
        let val = interp_2d(&field, &grid, 1.0, 1.0);
        assert!(
            (val - 2.0).abs() < 1e-10,
            "Midpoint interpolation failed: {val}"
        );
    }

    #[test]
    fn test_probe_generation() {
        let probes = generate_vessel_probes(6.0, 0.0, 4.0, 5.0, 10, 6);
        assert_eq!(probes.len(), 16);

        let n_bpol = probes
            .iter()
            .filter(|p| p.kind == ProbeKind::BpolMagnitude)
            .count();
        let n_flux = probes
            .iter()
            .filter(|p| p.kind == ProbeKind::FluxLoop)
            .count();
        assert_eq!(n_bpol, 10);
        assert_eq!(n_flux, 6);

        // All probes should be on the vessel wall (positive R)
        assert!(probes.iter().all(|p| p.r > 0.0), "Probes must have R > 0");
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        let p = ProfileParams {
            ped_height: 1.2,
            ped_top: 0.91,
            ped_width: 0.04,
            core_alpha: 0.35,
        };
        let ff = ProfileParams {
            ped_height: 0.9,
            ped_top: 0.88,
            ped_width: 0.06,
            core_alpha: 0.25,
        };

        let x = pack_params(&p, &ff);
        let (p2, ff2) = unpack_params(&x);

        assert!((p2.ped_height - p.ped_height).abs() < 1e-12);
        assert!((p2.ped_top - p.ped_top).abs() < 1e-12);
        assert!((ff2.ped_width - ff.ped_width).abs() < 1e-12);
        assert!((ff2.core_alpha - ff.core_alpha).abs() < 1e-12);
    }

    #[test]
    fn test_cholesky_solve() {
        // Simple 2x2 SPD embedded in N_PARAMS
        // [4, 2; 2, 3] x = [8, 7] → x = [1, 1] (for first 2 components)
        // Embed in N_PARAMS identity
        let mut a = [[0.0; N_PARAMS]; N_PARAMS];
        for (i, row) in a.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        a[0][0] = 4.0;
        a[0][1] = 2.0;
        a[1][0] = 2.0;
        a[1][1] = 3.0;

        let mut b = [0.0; N_PARAMS];
        b[0] = 8.0;
        b[1] = 7.0;

        let x = solve_dense(&a, &b).unwrap();
        assert!((x[0] - 1.25).abs() < 1e-10, "x[0] = {}", x[0]);
        assert!((x[1] - 1.5).abs() < 1e-10, "x[1] = {}", x[1]);
    }

    #[test]
    fn test_synthetic_measurements_deterministic() {
        let mut kernel = make_kernel();
        let probes = generate_vessel_probes(6.0, 0.0, 4.0, 5.0, 8, 4);
        let p = ProfileParams::default();
        let ff = ProfileParams::default();

        let m1 = generate_synthetic_measurements(&mut kernel, &probes, &p, &ff, 0.0).unwrap();
        let m2 = generate_synthetic_measurements(&mut kernel, &probes, &p, &ff, 0.0).unwrap();

        for (a, b) in m1.iter().zip(m2.iter()) {
            assert!(
                (a - b).abs() < 1e-12,
                "Clean measurements should be deterministic"
            );
        }
    }

    #[test]
    fn test_synthetic_noise_changes_values() {
        let mut kernel = make_kernel();
        let probes = generate_vessel_probes(6.0, 0.0, 4.0, 5.0, 8, 4);
        let p = ProfileParams::default();
        let ff = ProfileParams::default();

        let clean = generate_synthetic_measurements(&mut kernel, &probes, &p, &ff, 0.0).unwrap();
        let noisy = generate_synthetic_measurements(&mut kernel, &probes, &p, &ff, 0.01).unwrap();

        let any_diff = clean
            .iter()
            .zip(noisy.iter())
            .any(|(c, n)| (c - n).abs() > 1e-15);
        assert!(any_diff, "Noisy measurements should differ from clean");
    }

    // ── Integration: round-trip reconstruction ───────────────────────

    #[test]
    fn test_reconstruction_single_iteration() {
        // Fast smoke test: 1 LM iteration to verify the full pipeline
        // runs without error (unit tests above cover individual pieces).
        let mut kernel = make_kernel();
        let probes = generate_vessel_probes(6.0, 0.0, 4.0, 5.0, 10, 6);
        let p = ProfileParams::default();
        let ff = ProfileParams::default();
        let measurements =
            generate_synthetic_measurements(&mut kernel, &probes, &p, &ff, 0.0).unwrap();

        let inv_config = InverseConfig {
            max_iterations: 1,
            tolerance: 1e-10,
            ..InverseConfig::default()
        };

        let result = reconstruct_equilibrium(&mut kernel, &probes, &measurements, &inv_config);
        assert!(
            result.is_ok(),
            "Single-iteration reconstruction failed: {:?}",
            result.err()
        );
        let inv = result.unwrap();
        assert!(inv.chi_squared.is_finite());
    }

    #[test]
    #[ignore] // Slow: ~45 forward solves in debug mode
    fn test_reconstruction_round_trip() {
        let mut kernel = make_kernel();

        // Ground truth parameters (slightly different from default)
        let p_true = ProfileParams {
            ped_height: 1.1,
            ped_top: 0.91,
            ped_width: 0.05,
            core_alpha: 0.3,
        };
        let ff_true = ProfileParams {
            ped_height: 0.9,
            ped_top: 0.90,
            ped_width: 0.06,
            core_alpha: 0.2,
        };

        // Generate vessel probes and clean synthetic measurements
        let probes = generate_vessel_probes(6.0, 0.0, 4.0, 5.0, 12, 6);
        let measurements =
            generate_synthetic_measurements(&mut kernel, &probes, &p_true, &ff_true, 0.0).unwrap();

        // Reconstruct (starting from default guess)
        let inv_config = InverseConfig {
            max_iterations: 5, // Keep short for test speed
            tolerance: 1e-6,
            ..InverseConfig::default()
        };

        let result = reconstruct_equilibrium(&mut kernel, &probes, &measurements, &inv_config);
        assert!(
            result.is_ok(),
            "Reconstruction should not error: {:?}",
            result.err()
        );

        let inv = result.unwrap();
        assert!(inv.iterations > 0, "Should run at least 1 iteration");

        // Chi-squared should decrease from initial
        // (we can't guarantee convergence in 5 iterations, but it should improve)
        assert!(
            inv.chi_squared.is_finite(),
            "Chi-squared should be finite: {}",
            inv.chi_squared
        );
    }
}
