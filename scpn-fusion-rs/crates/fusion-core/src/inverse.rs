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
use crate::source::{mtanh_profile, mtanh_profile_derivatives, ProfileParams};
use fusion_math::linalg::pinv_svd;
use fusion_types::config::ReactorConfig;
use fusion_types::error::{FusionError, FusionResult};
use fusion_types::state::Grid2D;
use ndarray::{Array1, Array2};

const N_PARAMS: usize = 8;
const SOURCE_BETA_MIX: f64 = 0.5;
const MIN_FLUX_DENOMINATOR: f64 = 1e-9;
const MIN_CURRENT_INTEGRAL: f64 = 1e-9;
const MIN_RADIUS: f64 = 1e-9;
const SENSITIVITY_RELAXATION: f64 = 0.8;

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

fn nearest_index(axis: &Array1<f64>, value: f64) -> usize {
    let mut best_idx = 0usize;
    let mut best_dist = f64::INFINITY;
    for (idx, &x) in axis.iter().enumerate() {
        let d = (x - value).abs();
        if d < best_dist {
            best_dist = d;
            best_idx = idx;
        }
    }
    best_idx
}

fn probe_indices(grid: &Grid2D, probes_rz: &[(f64, f64)]) -> Vec<(usize, usize)> {
    probes_rz
        .iter()
        .map(|&(r, z)| (nearest_index(&grid.z, z), nearest_index(&grid.r, r)))
        .collect()
}

fn mtanh_profile_dpsi_norm(psi_norm: f64, params: &ProfileParams) -> f64 {
    let w = params.ped_width.abs().max(1e-8);
    let ped_top = params.ped_top.abs().max(1e-8);
    let y = (params.ped_top - psi_norm) / w;
    let tanh_y = y.tanh();
    let sech2 = 1.0 - tanh_y * tanh_y;

    let d_edge = -0.5 * params.ped_height * sech2 / w;
    let d_core = if psi_norm.abs() < ped_top {
        -2.0 * params.core_alpha * psi_norm / ped_top.powi(2)
    } else {
        0.0
    };
    d_edge + d_core
}

fn validate_inverse_config(config: &InverseConfig) -> FusionResult<()> {
    if config.max_iterations == 0 {
        return Err(FusionError::ConfigError(
            "inverse.max_iterations must be >= 1".to_string(),
        ));
    }
    if !config.tolerance.is_finite() || config.tolerance <= 0.0 {
        return Err(FusionError::ConfigError(
            "inverse.tolerance must be finite and > 0".to_string(),
        ));
    }
    if !config.damping.is_finite()
        || !(0.0..=1.0).contains(&config.damping)
        || config.damping == 0.0
    {
        return Err(FusionError::ConfigError(
            "inverse.damping must be finite and in (0, 1]".to_string(),
        ));
    }
    if !config.fd_step.is_finite() || config.fd_step <= 0.0 {
        return Err(FusionError::ConfigError(
            "inverse.fd_step must be finite and > 0".to_string(),
        ));
    }
    if !config.tikhonov.is_finite() || config.tikhonov <= 0.0 {
        return Err(FusionError::ConfigError(
            "inverse.tikhonov must be finite and > 0".to_string(),
        ));
    }
    Ok(())
}

fn validate_kernel_inverse_config(kernel_cfg: &KernelInverseConfig) -> FusionResult<()> {
    validate_inverse_config(&kernel_cfg.inverse)?;
    if kernel_cfg.kernel_max_iterations == 0 {
        return Err(FusionError::ConfigError(
            "kernel_max_iterations must be >= 1".to_string(),
        ));
    }
    Ok(())
}

fn kernel_forward_observables(
    reactor_config: &ReactorConfig,
    probes_rz: &[(f64, f64)],
    params_p: ProfileParams,
    params_ff: ProfileParams,
    kernel_cfg: &KernelInverseConfig,
) -> FusionResult<Vec<f64>> {
    validate_kernel_inverse_config(kernel_cfg)?;
    let mut cfg = reactor_config.clone();
    cfg.solver.max_iterations = kernel_cfg.kernel_max_iterations;

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

fn solve_linearized_sensitivity(
    grid: &Grid2D,
    ds_dx: &Array2<f64>,
    ds_dpsi: &Array2<f64>,
    iterations: usize,
) -> Array2<f64> {
    let mut delta = Array2::zeros((grid.nz, grid.nr));
    if grid.nz < 3 || grid.nr < 3 || iterations == 0 {
        return delta;
    }

    let mut next = delta.clone();
    let dr_sq = grid.dr * grid.dr;

    for _ in 0..iterations {
        for iz in 1..grid.nz - 1 {
            for ir in 1..grid.nr - 1 {
                let coupled_rhs = ds_dpsi[[iz, ir]] * delta[[iz, ir]] + ds_dx[[iz, ir]];
                let jacobi = 0.25
                    * (delta[[iz - 1, ir]]
                        + delta[[iz + 1, ir]]
                        + delta[[iz, ir - 1]]
                        + delta[[iz, ir + 1]]
                        - dr_sq * coupled_rhs);
                next[[iz, ir]] = (1.0 - SENSITIVITY_RELAXATION) * delta[[iz, ir]]
                    + SENSITIVITY_RELAXATION * jacobi;
            }
        }

        for ir in 0..grid.nr {
            next[[0, ir]] = 0.0;
            next[[grid.nz - 1, ir]] = 0.0;
        }
        for iz in 0..grid.nz {
            next[[iz, 0]] = 0.0;
            next[[iz, grid.nr - 1]] = 0.0;
        }

        std::mem::swap(&mut delta, &mut next);
    }

    delta
}

fn kernel_analytical_forward_and_jacobian(
    reactor_config: &ReactorConfig,
    probes_rz: &[(f64, f64)],
    params_p: ProfileParams,
    params_ff: ProfileParams,
    kernel_cfg: &KernelInverseConfig,
) -> FusionResult<(Vec<f64>, Vec<Vec<f64>>)> {
    validate_kernel_inverse_config(kernel_cfg)?;
    let mut cfg = reactor_config.clone();
    cfg.solver.max_iterations = kernel_cfg.kernel_max_iterations;

    let mut kernel = FusionKernel::new(cfg);
    let solve_result = kernel.solve_equilibrium_with_profiles(params_p, params_ff)?;
    if kernel_cfg.require_kernel_converged && !solve_result.converged {
        return Err(FusionError::SolverDiverged {
            iteration: solve_result.iterations,
            message: "Kernel forward solve did not converge under inverse constraints".to_string(),
        });
    }

    let base_observables = kernel.sample_psi_at_probes(probes_rz);
    let grid = kernel.grid();
    let psi = kernel.psi();
    let mu0 = kernel.config().physics.vacuum_permeability;
    let i_target = kernel.config().physics.plasma_current_target;

    let mut flux_denom = solve_result.psi_boundary - solve_result.psi_axis;
    if flux_denom.abs() < MIN_FLUX_DENOMINATOR {
        flux_denom = MIN_FLUX_DENOMINATOR;
    }

    let mut psi_norm = Array2::zeros((grid.nz, grid.nr));
    let mut inside = Array2::from_elem((grid.nz, grid.nr), false);
    let mut raw = Array2::zeros((grid.nz, grid.nr));
    let mut raw_dpsi_norm = Array2::zeros((grid.nz, grid.nr));

    for iz in 0..grid.nz {
        for ir in 0..grid.nr {
            let psi_n = (psi[[iz, ir]] - solve_result.psi_axis) / flux_denom;
            psi_norm[[iz, ir]] = psi_n;
            if !(0.0..1.0).contains(&psi_n) {
                continue;
            }

            let r = grid.rr[[iz, ir]].abs().max(MIN_RADIUS);
            let p = mtanh_profile(psi_n, &params_p);
            let ff = mtanh_profile(psi_n, &params_ff);
            let dp_dpsi = mtanh_profile_dpsi_norm(psi_n, &params_p);
            let dff_dpsi = mtanh_profile_dpsi_norm(psi_n, &params_ff);

            raw[[iz, ir]] = SOURCE_BETA_MIX * r * p + (1.0 - SOURCE_BETA_MIX) * (ff / (mu0 * r));
            raw_dpsi_norm[[iz, ir]] =
                SOURCE_BETA_MIX * r * dp_dpsi + (1.0 - SOURCE_BETA_MIX) * (dff_dpsi / (mu0 * r));
            inside[[iz, ir]] = true;
        }
    }

    let i_raw = raw.iter().sum::<f64>() * grid.dr * grid.dz;
    if i_raw.abs() <= MIN_CURRENT_INTEGRAL {
        return Ok((base_observables, vec![vec![0.0; N_PARAMS]; probes_rz.len()]));
    }
    let scale = i_target / i_raw;

    // Approximate local dS/dPsi for linearized kernel solve.
    let mut ds_dpsi = Array2::zeros((grid.nz, grid.nr));
    for iz in 0..grid.nz {
        for ir in 0..grid.nr {
            if !inside[[iz, ir]] {
                continue;
            }
            let r = grid.rr[[iz, ir]];
            let dj_dpsi = scale * raw_dpsi_norm[[iz, ir]] / flux_denom;
            ds_dpsi[[iz, ir]] = -mu0 * r * dj_dpsi;
        }
    }

    let probe_idx = probe_indices(grid, probes_rz);
    let mut jac = vec![vec![0.0; N_PARAMS]; probes_rz.len()];
    let sens_iters = (kernel_cfg.kernel_max_iterations.max(20) * 2).min(600);

    for (col, _) in [(); N_PARAMS].iter().enumerate() {
        let mut d_raw = Array2::zeros((grid.nz, grid.nr));
        for iz in 0..grid.nz {
            for ir in 0..grid.nr {
                if !inside[[iz, ir]] {
                    continue;
                }

                let psi_n = psi_norm[[iz, ir]];
                let r = grid.rr[[iz, ir]].abs().max(MIN_RADIUS);
                if col < 4 {
                    let dp = mtanh_profile_derivatives(psi_n, &params_p)[col];
                    d_raw[[iz, ir]] = SOURCE_BETA_MIX * r * dp;
                } else {
                    let dff = mtanh_profile_derivatives(psi_n, &params_ff)[col - 4];
                    d_raw[[iz, ir]] = (1.0 - SOURCE_BETA_MIX) * (dff / (mu0 * r));
                }
            }
        }

        let d_i = d_raw.iter().sum::<f64>() * grid.dr * grid.dz;
        let mut ds_dx = Array2::zeros((grid.nz, grid.nr));
        for iz in 0..grid.nz {
            for ir in 0..grid.nr {
                if !inside[[iz, ir]] {
                    continue;
                }
                let r = grid.rr[[iz, ir]];
                let dj = scale * (d_raw[[iz, ir]] - raw[[iz, ir]] * d_i / i_raw);
                ds_dx[[iz, ir]] = -mu0 * r * dj;
            }
        }

        let delta_psi = solve_linearized_sensitivity(grid, &ds_dx, &ds_dpsi, sens_iters);
        for (row, &(iz, ir)) in probe_idx.iter().enumerate() {
            jac[row][col] = delta_psi[[iz, ir]];
        }
    }

    Ok((base_observables, jac))
}

fn kernel_fd_jacobian_from_base(
    reactor_config: &ReactorConfig,
    probes_rz: &[(f64, f64)],
    params_p: ProfileParams,
    params_ff: ProfileParams,
    kernel_cfg: &KernelInverseConfig,
    base: &[f64],
) -> FusionResult<Vec<Vec<f64>>> {
    let mut jac = vec![vec![0.0; N_PARAMS]; probes_rz.len()];
    validate_kernel_inverse_config(kernel_cfg)?;
    let h = kernel_cfg.inverse.fd_step;

    for (col, _) in [(); N_PARAMS].iter().enumerate() {
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
    validate_inverse_config(config)?;

    let mut x = pack_params(
        &sanitize_params(initial_params_p),
        &sanitize_params(initial_params_ff),
    );
    let mut residual_history = Vec::with_capacity(config.max_iterations + 1);
    let mut converged = false;
    let mut iter_done = 0;
    let mut damping = config.damping;

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
        let lambda = config.tikhonov;
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
    validate_kernel_inverse_config(kernel_cfg)?;

    let mut x = pack_params(
        &sanitize_params(initial_params_p),
        &sanitize_params(initial_params_ff),
    );
    let mut residual_history = Vec::with_capacity(kernel_cfg.inverse.max_iterations + 1);
    let mut converged = false;
    let mut iter_done = 0usize;
    let mut damping = kernel_cfg.inverse.damping;

    for iter in 0..kernel_cfg.inverse.max_iterations {
        let (params_p, params_ff) = unpack_params(&x);
        let (prediction, jac) = match kernel_cfg.inverse.jacobian_mode {
            JacobianMode::Analytical => kernel_analytical_forward_and_jacobian(
                reactor_config,
                probes_rz,
                params_p,
                params_ff,
                kernel_cfg,
            )?,
            JacobianMode::FiniteDifference => {
                let pred = kernel_forward_observables(
                    reactor_config,
                    probes_rz,
                    params_p,
                    params_ff,
                    kernel_cfg,
                )?;
                let jac = kernel_fd_jacobian_from_base(
                    reactor_config,
                    probes_rz,
                    params_p,
                    params_ff,
                    kernel_cfg,
                    &pred,
                )?;
                (pred, jac)
            }
        };
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

        let j = to_array2(jac);

        let lambda = kernel_cfg.inverse.tikhonov;
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
    fn test_inverse_rejects_invalid_solver_config() {
        let probes: Vec<f64> = (0..8).map(|i| i as f64 / 7.0).collect();
        let measurements = vec![0.1; probes.len()];
        let init = ProfileParams::default();
        let cfg = InverseConfig {
            damping: 0.0,
            ..Default::default()
        };
        let err = reconstruct_equilibrium(&probes, &measurements, init, init, &cfg)
            .expect_err("invalid config must error");
        match err {
            FusionError::ConfigError(msg) => assert!(msg.contains("inverse.damping")),
            other => panic!("Unexpected error: {other:?}"),
        }
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

    #[test]
    fn test_kernel_inverse_rejects_invalid_kernel_iteration_config() {
        let cfg = ReactorConfig::from_file(&config_path("validation/iter_validated_config.json"))
            .unwrap();
        let mut kcfg = KernelInverseConfig::default();
        kcfg.kernel_max_iterations = 0;
        let init = ProfileParams::default();
        let err = reconstruct_equilibrium_with_kernel(
            &cfg,
            &[(6.2, 0.0), (6.3, 0.1)],
            &[1.0, 1.0],
            init,
            init,
            &kcfg,
        )
        .expect_err("invalid kernel config must error");
        match err {
            FusionError::ConfigError(msg) => assert!(msg.contains("kernel_max_iterations")),
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_kernel_analytical_jacobian_computes() {
        let cfg = ReactorConfig::from_file(&config_path("validation/iter_validated_config.json"))
            .unwrap();
        let probes = vec![(6.2, 0.0), (6.35, 0.1), (6.45, -0.15)];
        let params_p = ProfileParams {
            ped_top: 0.9,
            ped_width: 0.08,
            ped_height: 1.1,
            core_alpha: 0.25,
        };
        let params_ff = ProfileParams {
            ped_top: 0.86,
            ped_width: 0.07,
            ped_height: 0.95,
            core_alpha: 0.12,
        };
        let kcfg = KernelInverseConfig {
            inverse: InverseConfig {
                jacobian_mode: JacobianMode::Analytical,
                fd_step: 1e-5,
                ..Default::default()
            },
            kernel_max_iterations: 50,
            require_kernel_converged: false,
        };

        let (pred, jac) =
            kernel_analytical_forward_and_jacobian(&cfg, &probes, params_p, params_ff, &kcfg)
                .unwrap();
        assert_eq!(pred.len(), probes.len());
        assert_eq!(jac.len(), probes.len());
        assert!(jac.iter().all(|row| row.len() == N_PARAMS));
        assert!(jac.iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn test_kernel_analytical_jacobian_tracks_fd() {
        let cfg = ReactorConfig::from_file(&config_path("validation/iter_validated_config.json"))
            .unwrap();
        let probes = vec![(6.2, 0.0), (6.3, 0.08), (6.4, -0.12), (6.5, 0.0)];
        let params_p = ProfileParams {
            ped_top: 0.9,
            ped_width: 0.08,
            ped_height: 1.1,
            core_alpha: 0.25,
        };
        let params_ff = ProfileParams {
            ped_top: 0.86,
            ped_width: 0.07,
            ped_height: 0.95,
            core_alpha: 0.12,
        };
        let kcfg = KernelInverseConfig {
            inverse: InverseConfig {
                jacobian_mode: JacobianMode::Analytical,
                fd_step: 1e-5,
                ..Default::default()
            },
            kernel_max_iterations: 50,
            require_kernel_converged: false,
        };

        let (pred, jac_analytical) =
            kernel_analytical_forward_and_jacobian(&cfg, &probes, params_p, params_ff, &kcfg)
                .unwrap();
        let jac_fd =
            kernel_fd_jacobian_from_base(&cfg, &probes, params_p, params_ff, &kcfg, &pred).unwrap();

        assert_eq!(jac_analytical.len(), jac_fd.len());
        assert!(jac_analytical.iter().all(|row| row.len() == N_PARAMS));
        assert!(jac_fd.iter().all(|row| row.len() == N_PARAMS));

        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        let mut same_sign = 0usize;
        let mut comparable = 0usize;

        for (row_a, row_f) in jac_analytical.iter().zip(jac_fd.iter()) {
            for (&a, &f) in row_a.iter().zip(row_f.iter()) {
                assert!(a.is_finite() && f.is_finite());
                let d = a - f;
                num += d * d;
                den += f * f;

                if a.abs() > 1e-6 || f.abs() > 1e-6 {
                    comparable += 1;
                    if a.signum() == f.signum() {
                        same_sign += 1;
                    }
                }
            }
        }

        let nrmse = (num / den.max(1e-14)).sqrt();
        let sign_match = same_sign as f64 / comparable.max(1) as f64;

        // The analytical kernel Jacobian uses a linearized local sensitivity model.
        // It should track FD directionality and scale well enough for LM updates.
        assert!(
            nrmse < 1.5,
            "Kernel analytical Jacobian deviates too much from FD (NRMSE={nrmse})"
        );
        assert!(
            sign_match >= 0.65,
            "Kernel analytical Jacobian sign agreement too low ({sign_match})"
        );
    }

    #[test]
    fn test_kernel_inverse_analytical_mode_reduces_residual() {
        let cfg = ReactorConfig::from_file(&config_path("validation/iter_validated_config.json"))
            .unwrap();
        let probes = vec![(6.2, 0.0), (6.3, 0.1), (6.4, -0.1), (6.55, 0.0)];

        let true_p = ProfileParams {
            ped_top: 0.91,
            ped_width: 0.08,
            ped_height: 1.15,
            core_alpha: 0.28,
        };
        let true_ff = ProfileParams {
            ped_top: 0.85,
            ped_width: 0.06,
            ped_height: 0.92,
            core_alpha: 0.14,
        };
        let init_p = ProfileParams {
            ped_top: 0.78,
            ped_width: 0.12,
            ped_height: 0.7,
            core_alpha: 0.03,
        };
        let init_ff = ProfileParams {
            ped_top: 0.74,
            ped_width: 0.11,
            ped_height: 0.65,
            core_alpha: 0.02,
        };

        let base_cfg = KernelInverseConfig {
            inverse: InverseConfig {
                max_iterations: 4,
                damping: 0.6,
                tolerance: 1e-8,
                tikhonov: 1e-4,
                ..Default::default()
            },
            kernel_max_iterations: 50,
            require_kernel_converged: false,
        };

        let measurements =
            kernel_forward_observables(&cfg, &probes, true_p, true_ff, &base_cfg).unwrap();
        let init_prediction =
            kernel_forward_observables(&cfg, &probes, init_p, init_ff, &base_cfg).unwrap();
        let init_residual = (init_prediction
            .iter()
            .zip(measurements.iter())
            .map(|(p, m)| (p - m).powi(2))
            .sum::<f64>()
            / measurements.len() as f64)
            .sqrt();

        let mut analytical_cfg = base_cfg.clone();
        analytical_cfg.inverse.jacobian_mode = JacobianMode::Analytical;
        let result = reconstruct_equilibrium_with_kernel(
            &cfg,
            &probes,
            &measurements,
            init_p,
            init_ff,
            &analytical_cfg,
        )
        .unwrap();

        assert!(result.residual.is_finite());
        assert!(
            result.residual < init_residual,
            "Expected analytical kernel mode to reduce residual: init={}, final={}",
            init_residual,
            result.residual
        );
    }
}
