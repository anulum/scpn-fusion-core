// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Analytical Jacobian Utilities
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Jacobian builders for inverse profile reconstruction.
//!
//! The forward model used in this crate-level inverse module maps probe-space
//! normalized flux samples to synthetic measurements:
//!   m_i = p(psi_i) + ff(psi_i)
//! where p and ff are mTanh profiles with independent parameter sets.

use crate::source::{
    mtanh_profile, mtanh_profile_derivatives as source_mtanh_profile_derivatives, ProfileParams,
};
use fusion_types::error::{FusionError, FusionResult};

/// Re-export profile derivatives for callers expecting this symbol in `jacobian`.
pub fn mtanh_profile_derivatives(psi_norm: f64, params: &ProfileParams) -> [f64; 4] {
    source_mtanh_profile_derivatives(psi_norm, params)
}

/// Forward model used by inverse reconstruction.
pub fn forward_model_response(
    probe_psi_norm: &[f64],
    params_p: &ProfileParams,
    params_ff: &ProfileParams,
) -> Vec<f64> {
    probe_psi_norm
        .iter()
        .map(|&psi| mtanh_profile(psi, params_p) + mtanh_profile(psi, params_ff))
        .collect()
}

/// Build analytical Jacobian with 8 columns:
/// [p_ped_height, p_ped_top, p_ped_width, p_core_alpha,
///  ff_ped_height, ff_ped_top, ff_ped_width, ff_core_alpha].
pub fn compute_analytical_jacobian(
    params_p: &ProfileParams,
    params_ff: &ProfileParams,
    probe_psi_norm: &[f64],
) -> Vec<Vec<f64>> {
    let mut jac = Vec::with_capacity(probe_psi_norm.len());
    for &psi in probe_psi_norm {
        let dp = mtanh_profile_derivatives(psi, params_p);
        let df = mtanh_profile_derivatives(psi, params_ff);
        jac.push(vec![dp[0], dp[1], dp[2], dp[3], df[0], df[1], df[2], df[3]]);
    }
    jac
}

fn perturb_param(
    params_p: &ProfileParams,
    params_ff: &ProfileParams,
    idx: usize,
    delta: f64,
) -> (ProfileParams, ProfileParams) {
    let mut p = *params_p;
    let mut ff = *params_ff;
    match idx {
        0 => p.ped_height += delta,
        1 => p.ped_top += delta,
        2 => p.ped_width += delta,
        3 => p.core_alpha += delta,
        4 => ff.ped_height += delta,
        5 => ff.ped_top += delta,
        6 => ff.ped_width += delta,
        7 => ff.core_alpha += delta,
        _ => {}
    }
    (p, ff)
}

/// Finite-difference Jacobian using forward difference.
pub fn compute_fd_jacobian(
    params_p: &ProfileParams,
    params_ff: &ProfileParams,
    probe_psi_norm: &[f64],
    fd_step: f64,
) -> FusionResult<Vec<Vec<f64>>> {
    if !fd_step.is_finite() || fd_step <= 0.0 {
        return Err(FusionError::ConfigError(
            "jacobian fd_step must be finite and > 0".to_string(),
        ));
    }

    let base = forward_model_response(probe_psi_norm, params_p, params_ff);
    let h = fd_step;
    const PARAM_INDICES: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

    let mut jac = vec![vec![0.0; 8]; probe_psi_norm.len()];
    for col in PARAM_INDICES {
        let (p_pert, ff_pert) = perturb_param(params_p, params_ff, col, h);
        let f_pert = forward_model_response(probe_psi_norm, &p_pert, &ff_pert);
        for (row, (&fp, &b)) in jac.iter_mut().zip(f_pert.iter().zip(base.iter())) {
            row[col] = (fp - b) / h;
        }
    }
    Ok(jac)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analytical_vs_fd_jacobian() {
        let p = ProfileParams {
            ped_top: 0.9,
            ped_width: 0.08,
            ped_height: 1.1,
            core_alpha: 0.25,
        };
        let ff = ProfileParams {
            ped_top: 0.85,
            ped_width: 0.06,
            ped_height: 0.95,
            core_alpha: 0.15,
        };
        let probes: Vec<f64> = (0..32).map(|i| i as f64 / 31.0).collect();

        let ja = compute_analytical_jacobian(&p, &ff, &probes);
        let jf = compute_fd_jacobian(&p, &ff, &probes, 1e-6).expect("valid fd_step");
        for i in 0..probes.len() {
            for j in 0..8 {
                let abs = (ja[i][j] - jf[i][j]).abs();
                let denom = jf[i][j].abs().max(1e-8);
                let rel = abs / denom;
                assert!(
                    abs < 1e-8 || rel < 5e-2,
                    "Mismatch at row {i}, col {j}: analytical={}, fd={}, abs={}, rel={}",
                    ja[i][j],
                    jf[i][j],
                    abs,
                    rel
                );
            }
        }
    }

    #[test]
    fn test_analytical_jacobian_symmetry() {
        let params = ProfileParams {
            ped_top: 0.92,
            ped_width: 0.07,
            ped_height: 1.0,
            core_alpha: 0.2,
        };
        let probes: Vec<f64> = (0..16).map(|i| i as f64 / 15.0).collect();
        let jac = compute_analytical_jacobian(&params, &params, &probes);

        // If p and ff parameters are identical, corresponding Jacobian columns should match.
        for row in &jac {
            for k in 0..4 {
                assert!(
                    (row[k] - row[k + 4]).abs() < 1e-12,
                    "Expected symmetry in Jacobian columns {k} and {}",
                    k + 4
                );
            }
        }
    }

    #[test]
    fn test_fd_jacobian_rejects_invalid_fd_step() {
        let p = ProfileParams {
            ped_top: 0.9,
            ped_width: 0.08,
            ped_height: 1.1,
            core_alpha: 0.25,
        };
        let ff = p;
        let probes: Vec<f64> = (0..8).map(|i| i as f64 / 7.0).collect();

        assert!(compute_fd_jacobian(&p, &ff, &probes, 0.0).is_err());
        assert!(compute_fd_jacobian(&p, &ff, &probes, -1e-6).is_err());
        assert!(compute_fd_jacobian(&p, &ff, &probes, f64::NAN).is_err());
    }
}
