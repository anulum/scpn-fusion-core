// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Inverse Profile Reconstruction
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Inverse reconstruction module with selectable Jacobian mode.

use crate::jacobian::{compute_analytical_jacobian, compute_fd_jacobian, forward_model_response};
use crate::kernel::FusionKernel;
use crate::source::ProfileParams;
use fusion_math::linalg::pinv_svd;
use fusion_types::config::ReactorConfig;
use fusion_types::error::{FusionError, FusionResult};
use ndarray::{Array1, Array2};

const N_PARAMS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum JacobianMode {
    #[default]
    FiniteDifference,
    Analytical,
}

#[derive(Debug, Clone)]
pub struct InverseConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub damping: f64,
    pub fd_step: f64,
    pub tikhonov: f64,
    pub jacobian_mode: JacobianMode,
}

impl Default for InverseConfig {
    fn default() -> Self {
        Self {
            max_iterations: 40,
            tolerance: 1e-6,
            damping: 0.6,
            fd_step: 1e-6,
            tikhonov: 1e-4,
            jacobian_mode: JacobianMode::FiniteDifference,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InverseResult {
    pub params_p: ProfileParams,
    pub params_ff: ProfileParams,
    pub converged: bool,
    pub iterations: usize,
    pub residual: f64,
    pub residual_history: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct KernelInverseConfig {
    pub inverse: InverseConfig,
    pub kernel_max_iterations: usize,
    pub require_kernel_converged: bool,
}

impl Default for KernelInverseConfig {
    fn default() -> Self {
        Self {
            inverse: InverseConfig::default(),
            kernel_max_iterations: 80,
            require_kernel_converged: false,
        }
    }
}

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

fn sanitize_params(mut p: ProfileParams) -> ProfileParams {
    p.ped_top = p.ped_top.abs().max(1e-3);
    p.ped_width = p.ped_width.abs().max(1e-4);
    p
}

fn unpack_params(x: &[f64; N_PARAMS]) -> (ProfileParams, ProfileParams) {
    let p = sanitize_params(ProfileParams {
        ped_height: x[0],
        ped_top: x[1],
        ped_width: x[2],
        core_alpha: x[3],
    });
    let ff = sanitize_params(ProfileParams {
        ped_height: x[4],
        ped_top: x[5],
        ped_width: x[6],
        core_alpha: x[7],
    });
    (p, ff)
}

fn to_array2(jac: Vec<Vec<f64>>) -> Array2<f64> {
    let rows = jac.len();
    let cols = jac.first().map(|r| r.len()).unwrap_or(0);
    let mut out = Array2::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            out[[i, j]] = jac[i][j];
        }
    }
    out
}

fn kernel_forward_observables(
    reactor_config: &ReactorConfig,
    probes_rz: &[(f64, f64)],
    params_p: ProfileParams,
    params_ff: ProfileParams,
    kernel_cfg: &KernelInverseConfig,
) -> FusionResult<Vec<f64>> {
    let mut cfg = reactor_config.clone();
    cfg.solver.max_iterations = kernel_cfg.kernel_max_iterations.max(1);

    let mut kernel = FusionKernel::new(cfg);
    let result = kernel.solve_equilibrium_with_profiles(params_p, params_ff)?;
    if kernel_cfg.require_kernel_converged && !result.converged {
        return Err(FusionError::SolverDiverged {
            iteration: result.iterations,
            message: "Kernel forward solve did not converge under inverse constraints".to_string(),
        });
    }

    Ok(kernel.sample_psi_at_probes(probes_rz))
}

fn kernel_fd_jacobian(
    reactor_config: &ReactorConfig,
    probes_rz: &[(f64, f64)],
    params_p: ProfileParams,
    params_ff: ProfileParams,
    kernel_cfg: &KernelInverseConfig,
) -> FusionResult<Vec<Vec<f64>>> {
    let base =
        kernel_forward_observables(reactor_config, probes_rz, params_p, params_ff, kernel_cfg)?;
    let mut jac = vec![vec![0.0; N_PARAMS]; probes_rz.len()];
    let h = kernel_cfg.inverse.fd_step.abs().max(1e-6);

    for col in 0..N_PARAMS {
        let mut x = pack_params(&params_p, &params_ff);
        x[col] += h;
        let (p_pert, ff_pert) = unpack_params(&x);
        let pert =
            kernel_forward_observables(reactor_config, probes_rz, p_pert, ff_pert, kernel_cfg)?;
        for i in 0..probes_rz.len() {
            jac[i][col] = (pert[i] - base[i]) / h;
        }
    }
    Ok(jac)
}

/// Reconstruct profile parameters from probe-space measurements.
pub fn reconstruct_equilibrium(
    probe_psi_norm: &[f64],
    measurements: &[f64],
    initial_params_p: ProfileParams,
    initial_params_ff: ProfileParams,
    config: &InverseConfig,
) -> FusionResult<InverseResult> {
    if probe_psi_norm.is_empty() || measurements.is_empty() {
        return Err(FusionError::ConfigError(
            "Probe and measurement vectors must be non-empty".to_string(),
        ));
    }
    if probe_psi_norm.len() != measurements.len() {
        return Err(FusionError::ConfigError(format!(
            "Length mismatch: probes={}, measurements={}",
            probe_psi_norm.len(),
            measurements.len()
        )));
    }

    let mut x = pack_params(
        &sanitize_params(initial_params_p),
        &sanitize_params(initial_params_ff),
    );
    let mut residual_history = Vec::with_capacity(config.max_iterations + 1);
    let mut converged = false;
    let mut iter_done = 0;
    let mut damping = config.damping.clamp(1e-3, 1.0);

    for iter in 0..config.max_iterations {
        let (params_p, params_ff) = unpack_params(&x);
        let prediction = forward_model_response(probe_psi_norm, &params_p, &params_ff);
        let residual_vec: Vec<f64> = prediction
            .iter()
            .zip(measurements.iter())
            .map(|(p, m)| p - m)
            .collect();
        let residual =
            (residual_vec.iter().map(|v| v * v).sum::<f64>() / residual_vec.len() as f64).sqrt();
        residual_history.push(residual);
        iter_done = iter + 1;

        if residual < config.tolerance {
            converged = true;
            break;
        }

        let jac = match config.jacobian_mode {
            JacobianMode::Analytical => {
                compute_analytical_jacobian(&params_p, &params_ff, probe_psi_norm)
            }
            JacobianMode::FiniteDifference => {
                compute_fd_jacobian(&params_p, &params_ff, probe_psi_norm, config.fd_step)
            }
        };

        let j = to_array2(jac);
        let lambda = config.tikhonov.max(1e-10);
        let sqrt_lambda = lambda.sqrt();

        let mut j_aug = Array2::zeros((j.nrows() + N_PARAMS, N_PARAMS));
        for i in 0..j.nrows() {
            for k in 0..N_PARAMS {
                j_aug[[i, k]] = j[[i, k]];
            }
        }
        for k in 0..N_PARAMS {
            j_aug[[j.nrows() + k, k]] = sqrt_lambda;
        }

        let mut r_aug = Array1::zeros(j.nrows() + N_PARAMS);
        for i in 0..j.nrows() {
            r_aug[i] = residual_vec[i];
        }

        let pinv = pinv_svd(&j_aug, 1e-12);
        let delta_raw = pinv.dot(&r_aug).mapv(|v| -v);
        let delta = delta_raw.mapv(|v| v.clamp(-0.5, 0.5));

        if delta.len() != N_PARAMS {
            return Err(FusionError::LinAlg(format!(
                "Unexpected update dimension: got {}, expected {N_PARAMS}",
                delta.len()
            )));
        }

        let mut accepted = false;
        let mut local_damping = damping;
        for _ in 0..8 {
            let mut x_trial = x;
            for i in 0..N_PARAMS {
                x_trial[i] += local_damping * delta[i];
            }
            let (p_trial, ff_trial) = unpack_params(&x_trial);
            let x_trial_sanitized = pack_params(&p_trial, &ff_trial);
            let pred_trial = forward_model_response(probe_psi_norm, &p_trial, &ff_trial);
            let residual_trial = (pred_trial
                .iter()
                .zip(measurements.iter())
                .map(|(p, m)| (p - m).powi(2))
                .sum::<f64>()
                / measurements.len() as f64)
                .sqrt();

            if residual_trial <= residual {
                x = x_trial_sanitized;
                damping = (local_damping * 1.2).min(1.0);
                accepted = true;
                break;
            }
            local_damping *= 0.5;
        }

        if !accepted {
            break;
        }
    }

    let (params_p, params_ff) = unpack_params(&x);
    let prediction = forward_model_response(probe_psi_norm, &params_p, &params_ff);
    let final_residual = (prediction
        .iter()
        .zip(measurements.iter())
        .map(|(p, m)| (p - m).powi(2))
        .sum::<f64>()
        / measurements.len() as f64)
        .sqrt();

    Ok(InverseResult {
        params_p,
        params_ff,
        converged,
        iterations: iter_done,
        residual: final_residual,
        residual_history,
    })
}

/// Reconstruct profile parameters against actual `FusionKernel` observables.
///
/// This API evaluates residuals at physical probe coordinates `(R, Z)` by running
/// the Grad-Shafranov kernel with candidate profile parameters.
pub fn reconstruct_equilibrium_with_kernel(
    reactor_config: &ReactorConfig,
    probes_rz: &[(f64, f64)],
    measurements: &[f64],
    initial_params_p: ProfileParams,
    initial_params_ff: ProfileParams,
    kernel_cfg: &KernelInverseConfig,
) -> FusionResult<InverseResult> {
    if probes_rz.is_empty() || measurements.is_empty() {
        return Err(FusionError::ConfigError(
            "Probe and measurement vectors must be non-empty".to_string(),
        ));
    }
    if probes_rz.len() != measurements.len() {
        return Err(FusionError::ConfigError(format!(
            "Length mismatch: probes={}, measurements={}",
            probes_rz.len(),
            measurements.len()
        )));
    }

    let mut x = pack_params(
        &sanitize_params(initial_params_p),
        &sanitize_params(initial_params_ff),
    );
    let mut residual_history = Vec::with_capacity(kernel_cfg.inverse.max_iterations + 1);
    let mut converged = false;
    let mut iter_done = 0usize;
    let mut damping = kernel_cfg.inverse.damping.clamp(1e-3, 1.0);

    for iter in 0..kernel_cfg.inverse.max_iterations {
        let (params_p, params_ff) = unpack_params(&x);
        let prediction =
            kernel_forward_observables(reactor_config, probes_rz, params_p, params_ff, kernel_cfg)?;
        let residual_vec: Vec<f64> = prediction
            .iter()
            .zip(measurements.iter())
            .map(|(p, m)| p - m)
            .collect();
        let residual =
            (residual_vec.iter().map(|v| v * v).sum::<f64>() / residual_vec.len() as f64).sqrt();
        residual_history.push(residual);
        iter_done = iter + 1;

        if residual < kernel_cfg.inverse.tolerance {
            converged = true;
            break;
        }

        // Full analytical Jacobian of the kernel is not available yet.
        // Use kernel finite-difference Jacobian in both modes for robustness.
        let jac = kernel_fd_jacobian(reactor_config, probes_rz, params_p, params_ff, kernel_cfg)?;
        let j = to_array2(jac);

        let lambda = kernel_cfg.inverse.tikhonov.max(1e-10);
        let sqrt_lambda = lambda.sqrt();
        let mut j_aug = Array2::zeros((j.nrows() + N_PARAMS, N_PARAMS));
        for i in 0..j.nrows() {
            for k in 0..N_PARAMS {
                j_aug[[i, k]] = j[[i, k]];
            }
        }
        for k in 0..N_PARAMS {
            j_aug[[j.nrows() + k, k]] = sqrt_lambda;
        }
        let mut r_aug = Array1::zeros(j.nrows() + N_PARAMS);
        for i in 0..j.nrows() {
            r_aug[i] = residual_vec[i];
        }

        let pinv = pinv_svd(&j_aug, 1e-12);
        let delta_raw = pinv.dot(&r_aug).mapv(|v| -v);
        let delta = delta_raw.mapv(|v| v.clamp(-0.5, 0.5));

        let mut accepted = false;
        let mut local_damping = damping;
        for _ in 0..6 {
            let mut x_trial = x;
            for i in 0..N_PARAMS {
                x_trial[i] += local_damping * delta[i];
            }
            let (p_trial, ff_trial) = unpack_params(&x_trial);
            let pred_trial = kernel_forward_observables(
                reactor_config,
                probes_rz,
                p_trial,
                ff_trial,
                kernel_cfg,
            )?;
            let residual_trial = (pred_trial
                .iter()
                .zip(measurements.iter())
                .map(|(p, m)| (p - m).powi(2))
                .sum::<f64>()
                / measurements.len() as f64)
                .sqrt();
            if residual_trial <= residual {
                let (p_s, ff_s) = unpack_params(&x_trial);
                x = pack_params(&p_s, &ff_s);
                damping = (local_damping * 1.1).min(1.0);
                accepted = true;
                break;
            }
            local_damping *= 0.5;
        }

        if !accepted {
            break;
        }
    }

    let (params_p, params_ff) = unpack_params(&x);
    let prediction =
        kernel_forward_observables(reactor_config, probes_rz, params_p, params_ff, kernel_cfg)?;
    let final_residual = (prediction
        .iter()
        .zip(measurements.iter())
        .map(|(p, m)| (p - m).powi(2))
        .sum::<f64>()
        / measurements.len() as f64)
        .sqrt();

    Ok(InverseResult {
        params_p,
        params_ff,
        converged,
        iterations: iter_done,
        residual: final_residual,
        residual_history,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jacobian::forward_model_response;
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
    fn test_reconstruct_with_analytical_jacobian() {
        let true_p = ProfileParams {
            ped_top: 0.91,
            ped_width: 0.08,
            ped_height: 1.2,
            core_alpha: 0.3,
        };
        let true_ff = ProfileParams {
            ped_top: 0.84,
            ped_width: 0.06,
            ped_height: 0.9,
            core_alpha: 0.15,
        };
        let probes: Vec<f64> = (0..60).map(|i| i as f64 / 59.0).collect();
        let measurements = forward_model_response(&probes, &true_p, &true_ff);

        let init_p = ProfileParams {
            ped_top: 0.75,
            ped_width: 0.12,
            ped_height: 0.7,
            core_alpha: 0.05,
        };
        let init_ff = ProfileParams {
            ped_top: 0.70,
            ped_width: 0.10,
            ped_height: 0.6,
            core_alpha: 0.02,
        };
        let init_prediction = forward_model_response(&probes, &init_p, &init_ff);
        let init_residual = (init_prediction
            .iter()
            .zip(measurements.iter())
            .map(|(p, m)| (p - m).powi(2))
            .sum::<f64>()
            / measurements.len() as f64)
            .sqrt();

        let cfg = InverseConfig {
            jacobian_mode: JacobianMode::Analytical,
            max_iterations: 50,
            damping: 0.7,
            tolerance: 1e-9,
            ..Default::default()
        };
        let result =
            reconstruct_equilibrium(&probes, &measurements, init_p, init_ff, &cfg).unwrap();

        assert!(
            result.residual < init_residual * 0.75,
            "Expected >=25% residual reduction: initial={}, final={}",
            init_residual,
            result.residual,
        );
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_fd_and_analytical_modes_agree() {
        let true_p = ProfileParams {
            ped_top: 0.9,
            ped_width: 0.07,
            ped_height: 1.1,
            core_alpha: 0.2,
        };
        let true_ff = ProfileParams {
            ped_top: 0.83,
            ped_width: 0.05,
            ped_height: 0.95,
            core_alpha: 0.1,
        };
        let probes: Vec<f64> = (0..48).map(|i| i as f64 / 47.0).collect();
        let measurements = forward_model_response(&probes, &true_p, &true_ff);

        let init = ProfileParams {
            ped_top: 0.8,
            ped_width: 0.11,
            ped_height: 0.8,
            core_alpha: 0.05,
        };

        let cfg_a = InverseConfig {
            jacobian_mode: JacobianMode::Analytical,
            max_iterations: 35,
            ..Default::default()
        };
        let cfg_fd = InverseConfig {
            jacobian_mode: JacobianMode::FiniteDifference,
            max_iterations: 35,
            ..Default::default()
        };

        let ra = reconstruct_equilibrium(&probes, &measurements, init, init, &cfg_a).unwrap();
        let rf = reconstruct_equilibrium(&probes, &measurements, init, init, &cfg_fd).unwrap();

        assert!(
            (ra.residual - rf.residual).abs() < 1e-3,
            "Modes diverged too much: analytical={}, fd={}",
            ra.residual,
            rf.residual
        );
    }

    #[test]
    fn test_kernel_inverse_api_input_validation() {
        let cfg = ReactorConfig::from_file(&config_path("validation/iter_validated_config.json"))
            .unwrap();
        let kcfg = KernelInverseConfig::default();
        let init = ProfileParams::default();

        let err = reconstruct_equilibrium_with_kernel(
            &cfg,
            &[(6.2, 0.0), (6.3, 0.1)],
            &[1.0],
            init,
            init,
            &kcfg,
        )
        .unwrap_err();

        match err {
            FusionError::ConfigError(msg) => {
                assert!(msg.contains("Length mismatch"), "Unexpected message: {msg}")
            }
            _ => panic!("Expected ConfigError for mismatched inputs"),
        }
    }
}
