// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Stability
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Stability analysis: decay index, force balance, eigenvalue analysis.
//!
//! Port of `stability_analyzer.py` (192 lines) and `force_balance.py` (103 lines).

use crate::vacuum::calculate_vacuum_field;
use fusion_math::interp::{gradient_2d, interp2d};
use fusion_math::linalg::eig_2x2;
use fusion_types::config::ReactorConfig;
use fusion_types::constants::MU0_SI;
use fusion_types::error::{FusionError, FusionResult};
use fusion_types::state::{Grid2D, StabilityResult};
use ndarray::Array2;

/// Minor radius approximation: a = R/3. Python line 66.
const MINOR_RADIUS_RATIO: f64 = 3.0;

/// Internal inductance. Python line 67.
const INTERNAL_INDUCTANCE: f64 = 0.8;

/// Poloidal beta. Python line 68.
const BETA_POLOIDAL: f64 = 0.5;

/// Finite difference perturbation for Jacobian [m]. Python line 106.
const FORCE_PERTURBATION: f64 = 0.01;

/// Maximum decay index for stability. Python line 139.
const MAX_STABLE_DECAY_INDEX: f64 = 1.5;

/// Force balance convergence tolerance [N]. Python line 39.
const FORCE_TOLERANCE: f64 = 1e4;

/// Maximum Newton iterations. Python line 34.
const MAX_NEWTON_ITER: usize = 10;

/// Coil current perturbation for Jacobian [MA]. Python line 45.
const COIL_PERTURBATION_MA: f64 = 0.1;

/// Maximum current correction per step [MA]. Python line 72.
const MAX_CURRENT_CORRECTION_MA: f64 = 5.0;

const MIN_RADIUS_M: f64 = 1e-6;

fn validate_stability_inputs(psi: &Array2<f64>, grid: &Grid2D, r: f64, z: f64) -> FusionResult<()> {
    if psi.nrows() != grid.nz || psi.ncols() != grid.nr {
        return Err(FusionError::ConfigError(format!(
            "stability psi shape mismatch: expected ({}, {}), got ({}, {})",
            grid.nz,
            grid.nr,
            psi.nrows(),
            psi.ncols()
        )));
    }
    if psi.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::ConfigError(
            "stability psi must contain only finite values".to_string(),
        ));
    }
    if !r.is_finite() || r <= MIN_RADIUS_M {
        return Err(FusionError::ConfigError(format!(
            "stability radius r must be finite and > {MIN_RADIUS_M}, got {r}"
        )));
    }
    if !z.is_finite() {
        return Err(FusionError::ConfigError(format!(
            "stability vertical coordinate z must be finite, got {z}"
        )));
    }
    Ok(())
}

/// Calculate the decay index at a given position.
///
/// n = -(R/B_Z) · dB_Z/dR
///
/// Stable if 0 < n < 1.5.
pub fn decay_index(psi: &Array2<f64>, grid: &Grid2D, r: f64, z: f64) -> FusionResult<f64> {
    validate_stability_inputs(psi, grid, r, z)?;

    let (_dpsi_dz, dpsi_dr) = gradient_2d(psi, grid);

    // B_Z = (1/R) dΨ/dR at the target point
    let bz = interp2d(&dpsi_dr, grid, r, z) / r;
    if !bz.is_finite() {
        return Err(FusionError::ConfigError(
            "decay_index computed non-finite B_Z".to_string(),
        ));
    }

    if bz.abs() < 1e-12 {
        return Err(FusionError::ConfigError(
            "decay_index undefined when |B_Z| is near zero".to_string(),
        ));
    }

    // dB_Z/dR via finite difference
    let eps = FORCE_PERTURBATION;
    if r <= eps {
        return Err(FusionError::ConfigError(format!(
            "decay_index requires r > perturbation epsilon ({eps})"
        )));
    }
    let bz_plus = interp2d(&dpsi_dr, grid, r + eps, z) / (r + eps);
    let bz_minus = interp2d(&dpsi_dr, grid, r - eps, z) / (r - eps);
    if !bz_plus.is_finite() || !bz_minus.is_finite() {
        return Err(FusionError::ConfigError(
            "decay_index finite-difference B_Z samples became non-finite".to_string(),
        ));
    }
    let dbz_dr = (bz_plus - bz_minus) / (2.0 * eps);
    if !dbz_dr.is_finite() {
        return Err(FusionError::ConfigError(
            "decay_index dB_Z/dR became non-finite".to_string(),
        ));
    }

    let n = -(r / bz) * dbz_dr;
    if !n.is_finite() {
        return Err(FusionError::ConfigError(
            "decay_index result became non-finite".to_string(),
        ));
    }
    Ok(n)
}

/// Calculate forces acting on a current-carrying plasma ring.
///
/// Returns (F_radial [N], F_vertical [N]).
///
/// F_hoop = (μ₀ I²/2) · [ln(8R/a) + β_p + l_i/2 - 1.5] / R
/// F_Lorentz_R = I · B_Z · 2πR
/// F_Lorentz_Z = -I · B_R · 2πR
pub fn calculate_forces(
    psi: &Array2<f64>,
    grid: &Grid2D,
    r: f64,
    z: f64,
    i_plasma_ma: f64,
) -> FusionResult<(f64, f64)> {
    validate_stability_inputs(psi, grid, r, z)?;
    if !i_plasma_ma.is_finite() {
        return Err(FusionError::ConfigError(format!(
            "i_plasma_ma must be finite, got {i_plasma_ma}"
        )));
    }

    let i_amp = i_plasma_ma * 1e6;
    if !i_amp.is_finite() {
        return Err(FusionError::ConfigError(
            "plasma current conversion to amperes became non-finite".to_string(),
        ));
    }
    let (dpsi_dz, dpsi_dr) = gradient_2d(psi, grid);

    // B-field at target point
    let bz = interp2d(&dpsi_dr, grid, r, z) / r;
    let br = -interp2d(&dpsi_dz, grid, r, z) / r;
    if !bz.is_finite() || !br.is_finite() {
        return Err(FusionError::ConfigError(
            "calculate_forces computed non-finite B-field components".to_string(),
        ));
    }

    // Shafranov hoop force (outward)
    let a = r / MINOR_RADIUS_RATIO;
    let shafranov_term = (8.0 * r / a).ln() + BETA_POLOIDAL + INTERNAL_INDUCTANCE / 2.0 - 1.5;
    let f_hoop = (MU0_SI * i_amp * i_amp / 2.0) * shafranov_term / r;

    // Lorentz forces
    let f_lorentz_r = i_amp * bz * 2.0 * std::f64::consts::PI * r;
    let f_lorentz_z = -i_amp * br * 2.0 * std::f64::consts::PI * r;

    let f_radial = f_hoop + f_lorentz_r;
    let f_vertical = f_lorentz_z;
    if !f_radial.is_finite() || !f_vertical.is_finite() {
        return Err(FusionError::ConfigError(
            "calculate_forces produced non-finite force values".to_string(),
        ));
    }

    Ok((f_radial, f_vertical))
}

/// Full stability analysis: eigenvalue decomposition of the stiffness matrix.
///
/// Builds Jacobian K = [[dFr/dR, dFr/dZ], [dFz/dR, dFz/dZ]] via finite differences.
/// Eigenvalues > 0 → stable, < 0 → unstable.
pub fn analyze_stability(
    psi: &Array2<f64>,
    grid: &Grid2D,
    r_eq: f64,
    z_eq: f64,
    i_plasma_ma: f64,
) -> FusionResult<StabilityResult> {
    validate_stability_inputs(psi, grid, r_eq, z_eq)?;
    if !i_plasma_ma.is_finite() {
        return Err(FusionError::ConfigError(format!(
            "i_plasma_ma must be finite, got {i_plasma_ma}"
        )));
    }
    let eps = FORCE_PERTURBATION;

    // Force at equilibrium
    let (fr0, fz0) = calculate_forces(psi, grid, r_eq, z_eq, i_plasma_ma)?;

    // Perturb R
    let (fr_rp, fz_rp) = calculate_forces(psi, grid, r_eq + eps, z_eq, i_plasma_ma)?;
    let (fr_rm, fz_rm) = calculate_forces(psi, grid, r_eq - eps, z_eq, i_plasma_ma)?;

    // Perturb Z
    let (fr_zp, fz_zp) = calculate_forces(psi, grid, r_eq, z_eq + eps, i_plasma_ma)?;
    let (fr_zm, fz_zm) = calculate_forces(psi, grid, r_eq, z_eq - eps, i_plasma_ma)?;

    // Stiffness matrix K = -dF/dx (negative sign: restoring force convention)
    let k_rr = -(fr_rp - fr_rm) / (2.0 * eps);
    let k_rz = -(fr_zp - fr_zm) / (2.0 * eps);
    let k_zr = -(fz_rp - fz_rm) / (2.0 * eps);
    let k_zz = -(fz_zp - fz_zm) / (2.0 * eps);

    let k_matrix = [[k_rr, k_rz], [k_zr, k_zz]];
    let (eigenvalues, eigenvectors) = eig_2x2(&k_matrix);

    let n = decay_index(psi, grid, r_eq, z_eq)?;
    let is_stable = n > 0.0 && n < MAX_STABLE_DECAY_INDEX;

    let result = StabilityResult {
        eigenvalues,
        eigenvectors,
        decay_index: n,
        radial_force_mn: fr0 / 1e6,
        vertical_force_mn: fz0 / 1e6,
        is_stable,
    };
    if !result.eigenvalues[0].is_finite()
        || !result.eigenvalues[1].is_finite()
        || !result.decay_index.is_finite()
        || !result.radial_force_mn.is_finite()
        || !result.vertical_force_mn.is_finite()
    {
        return Err(FusionError::ConfigError(
            "analyze_stability produced non-finite outputs".to_string(),
        ));
    }

    Ok(result)
}

/// Newton-Raphson force balance solver.
///
/// Adjusts outer coil currents (PF3, PF4) to achieve zero radial force
/// at the target position.
///
/// Returns the number of iterations and final force residual.
pub fn solve_force_balance(
    config: &mut ReactorConfig,
    grid: &Grid2D,
    r_target: f64,
    z_target: f64,
    control_coil_indices: &[usize],
) -> FusionResult<(usize, f64)> {
    if !r_target.is_finite() || !z_target.is_finite() {
        return Err(FusionError::ConfigError(
            "force-balance target coordinates must be finite".to_string(),
        ));
    }
    if control_coil_indices.is_empty() {
        return Err(FusionError::ConfigError(
            "force-balance requires at least one control coil index".to_string(),
        ));
    }

    let mu0 = config.physics.vacuum_permeability;
    let i_plasma_ma = config.physics.plasma_current_target / 1e6;
    if !mu0.is_finite() || !i_plasma_ma.is_finite() {
        return Err(FusionError::ConfigError(
            "force-balance physics constants must be finite".to_string(),
        ));
    }

    for iter in 0..MAX_NEWTON_ITER {
        // Compute vacuum field and force
        let psi = calculate_vacuum_field(grid, &config.coils, mu0);
        let (f_r, _f_z) = calculate_forces(&psi, grid, r_target, z_target, i_plasma_ma)?;

        if f_r.abs() < FORCE_TOLERANCE {
            return Ok((iter, f_r.abs()));
        }

        // Compute Jacobian numerically for each control coil
        let n_controls = control_coil_indices.len();
        let mut jacobian = vec![0.0; n_controls];

        for (j, &coil_idx) in control_coil_indices.iter().enumerate() {
            if coil_idx < config.coils.len() {
                let original = config.coils[coil_idx].current;
                config.coils[coil_idx].current = original + COIL_PERTURBATION_MA;

                let psi_pert = calculate_vacuum_field(grid, &config.coils, mu0);
                let (f_r_pert, _) =
                    calculate_forces(&psi_pert, grid, r_target, z_target, i_plasma_ma)?;

                jacobian[j] = (f_r_pert - f_r) / COIL_PERTURBATION_MA;
                config.coils[coil_idx].current = original;
            }
        }

        // Newton correction: ΔI = -F_r / J (distribute equally among control coils)
        let j_total: f64 = jacobian.iter().sum();
        if j_total.abs() < 1e-10 {
            break;
        }

        let delta_i = (-f_r / j_total).clamp(-MAX_CURRENT_CORRECTION_MA, MAX_CURRENT_CORRECTION_MA);

        for &coil_idx in control_coil_indices {
            if coil_idx < config.coils.len() {
                config.coils[coil_idx].current += delta_i;
            }
        }
    }

    // Final force evaluation
    let psi = calculate_vacuum_field(grid, &config.coils, mu0);
    let (f_r, _) = calculate_forces(&psi, grid, r_target, z_target, i_plasma_ma)?;
    Ok((MAX_NEWTON_ITER, f_r.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusion_types::config::ReactorConfig;
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

    #[test]
    fn test_decay_index_finite() {
        let cfg = ReactorConfig::from_file(&config_path("iter_config.json")).unwrap();
        let grid = cfg.create_grid();
        let psi = calculate_vacuum_field(&grid, &cfg.coils, cfg.physics.vacuum_permeability);

        let n = decay_index(&psi, &grid, 6.2, 0.0).expect("valid finite decay-index inputs");
        assert!(n.is_finite(), "Decay index should be finite: {n}");
    }

    #[test]
    fn test_forces_finite() {
        let cfg = ReactorConfig::from_file(&config_path("iter_config.json")).unwrap();
        let grid = cfg.create_grid();
        let psi = calculate_vacuum_field(&grid, &cfg.coils, cfg.physics.vacuum_permeability);

        let (fr, fz) =
            calculate_forces(&psi, &grid, 6.2, 0.0, 15.0).expect("valid finite force inputs");
        assert!(fr.is_finite(), "Radial force should be finite: {fr}");
        assert!(fz.is_finite(), "Vertical force should be finite: {fz}");
    }

    #[test]
    fn test_stability_analysis() {
        let cfg = ReactorConfig::from_file(&config_path("iter_config.json")).unwrap();
        let grid = cfg.create_grid();
        let psi = calculate_vacuum_field(&grid, &cfg.coils, cfg.physics.vacuum_permeability);

        let result = analyze_stability(&psi, &grid, 6.2, 0.0, 15.0)
            .expect("valid finite stability-analysis inputs");
        assert!(
            result.eigenvalues[0].is_finite() && result.eigenvalues[1].is_finite(),
            "Eigenvalues should be finite"
        );
        assert!(
            result.decay_index.is_finite(),
            "Decay index should be finite"
        );
    }

    #[test]
    fn test_force_balance_runs() {
        let mut cfg =
            ReactorConfig::from_file(&config_path("validation/iter_validated_config.json"))
                .unwrap();
        let grid = cfg.create_grid();

        // Use PF3 (index 2) and PF4 (index 3) as control coils
        let (iters, final_force) = solve_force_balance(&mut cfg, &grid, 6.2, 0.0, &[2, 3])
            .expect("valid finite force-balance inputs");

        // Should run without panic (may converge on first iteration or exhaust max_iter)
        assert!(iters <= 10, "Should not exceed max iterations");
        assert!(final_force.is_finite(), "Final force should be finite");
    }

    #[test]
    fn test_stability_rejects_invalid_runtime_inputs() {
        let cfg = ReactorConfig::from_file(&config_path("iter_config.json")).unwrap();
        let grid = cfg.create_grid();
        let psi = calculate_vacuum_field(&grid, &cfg.coils, cfg.physics.vacuum_permeability);

        assert!(decay_index(&psi, &grid, f64::NAN, 0.0).is_err());
        assert!(calculate_forces(&psi, &grid, 0.0, 0.0, 15.0).is_err());
        assert!(analyze_stability(&psi, &grid, 0.0, 0.0, 15.0).is_err());

        let mut cfg2 = ReactorConfig::from_file(&config_path("iter_config.json")).unwrap();
        assert!(solve_force_balance(&mut cfg2, &grid, 6.2, 0.0, &[]).is_err());
    }
}
