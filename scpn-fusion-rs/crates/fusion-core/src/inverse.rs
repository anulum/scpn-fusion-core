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

fn validate_profile_params(params: &ProfileParams, label: &str) -> FusionResult<()> {
    if !params.ped_top.is_finite() || params.ped_top <= 0.0 {
        return Err(FusionError::ConfigError(format!(
            "{label}.ped_top must be finite and > 0, got {}",
            params.ped_top
        )));
    }
    if !params.ped_width.is_finite() || params.ped_width <= 0.0 {
        return Err(FusionError::ConfigError(format!(
            "{label}.ped_width must be finite and > 0, got {}",
            params.ped_width
        )));
    }
    if !params.ped_height.is_finite() || !params.core_alpha.is_finite() {
        return Err(FusionError::ConfigError(format!(
            "{label}.ped_height/core_alpha must be finite"
        )));
    }
    Ok(())
}

fn unpack_params(x: &[f64; N_PARAMS]) -> (ProfileParams, ProfileParams) {
    let p = ProfileParams {
        ped_height: x[0],
        ped_top: x[1],
        ped_width: x[2],
        core_alpha: x[3],
    };
    let ff = ProfileParams {
        ped_height: x[4],
        ped_top: x[5],
        ped_width: x[6],
        core_alpha: x[7],
    };
    (p, ff)
}

fn to_array2(jac: &[Vec<f64>], expected_cols: usize, label: &str) -> FusionResult<Array2<f64>> {
    if jac.is_empty() {
        return Err(FusionError::ConfigError(format!(
            "{label} must contain at least one row"
        )));
    }
    if jac.iter().any(|row| row.len() != expected_cols) {
        return Err(FusionError::ConfigError(format!(
            "{label} column count mismatch: expected {expected_cols}"
        )));
    }
    if jac.iter().flatten().any(|v| !v.is_finite()) {
        return Err(FusionError::ConfigError(format!(
            "{label} contains non-finite values"
        )));
    }
    let rows = jac.len();
    let mut out: Array2<f64> = Array2::zeros((rows, expected_cols));
    for i in 0..rows {
        for j in 0..expected_cols {
            out[[i, j]] = jac[i][j];
        }
    }
    Ok(out)
}

fn validate_flux_denom(flux_denom: f64) -> FusionResult<f64> {
    if !flux_denom.is_finite() || flux_denom.abs() <= MIN_FLUX_DENOMINATOR {
        return Err(FusionError::ConfigError(format!(
            "kernel inverse flux denominator must be finite and |psi_boundary - psi_axis| > {MIN_FLUX_DENOMINATOR}, got {flux_denom}"
        )));
    }
    Ok(flux_denom)
}

fn validate_radius(radius_m: f64, label: &str) -> FusionResult<f64> {
    if !radius_m.is_finite() || radius_m <= MIN_RADIUS {
        return Err(FusionError::ConfigError(format!(
            "{label} must be finite and > {MIN_RADIUS}, got {radius_m}"
        )));
    }
    Ok(radius_m)
}

fn validate_probe_psi_norm(probe_psi_norm: &[f64]) -> FusionResult<()> {
    if probe_psi_norm
        .iter()
        .any(|psi| !psi.is_finite() || *psi < 0.0 || *psi > 1.0)
    {
        return Err(FusionError::ConfigError(
            "probe_psi_norm values must be finite and within [0, 1]".to_string(),
        ));
    }
    Ok(())
}

fn validate_probe_coordinates(probes_rz: &[(f64, f64)]) -> FusionResult<()> {
    if probes_rz
        .iter()
        .any(|(r, z)| !r.is_finite() || !z.is_finite())
    {
        return Err(FusionError::ConfigError(
            "probes_rz coordinates must be finite".to_string(),
        ));
    }
    Ok(())
}

fn validate_measurements(measurements: &[f64]) -> FusionResult<()> {
    if measurements.iter().any(|m| !m.is_finite()) {
        return Err(FusionError::ConfigError(
            "measurements must be finite".to_string(),
        ));
    }
    Ok(())
}

fn validate_observables(observables: &[f64], expected_len: usize, label: &str) -> FusionResult<()> {
    if observables.len() != expected_len {
        return Err(FusionError::ConfigError(format!(
            "{label} length mismatch: got {}, expected {expected_len}",
            observables.len()
        )));
    }
    if observables.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::ConfigError(format!("{label} must be finite")));
    }
    Ok(())
}

fn validate_jacobian_matrix(
    jac: &[Vec<f64>],
    expected_rows: usize,
    expected_cols: usize,
    label: &str,
) -> FusionResult<()> {
    if jac.len() != expected_rows {
        return Err(FusionError::ConfigError(format!(
            "{label} row count mismatch: got {}, expected {expected_rows}",
            jac.len()
        )));
    }
    if jac.iter().any(|row| row.len() != expected_cols) {
        return Err(FusionError::ConfigError(format!(
            "{label} column count mismatch: expected {expected_cols}"
        )));
    }
    if jac.iter().flatten().any(|v| !v.is_finite()) {
        return Err(FusionError::ConfigError(format!(
            "{label} contains non-finite values"
        )));
    }
    Ok(())
}

fn validate_update_vector(delta: &Array1<f64>, label: &str) -> FusionResult<()> {
    if delta.len() != N_PARAMS {
        return Err(FusionError::LinAlg(format!(
            "{label} has unexpected dimension: got {}, expected {N_PARAMS}",
            delta.len()
        )));
    }
    if delta.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::LinAlg(format!(
            "{label} contains non-finite values"
        )));
    }
    Ok(())
}

fn compute_rmse(prediction: &[f64], measurements: &[f64], label: &str) -> FusionResult<f64> {
    if prediction.is_empty() || measurements.is_empty() {
        return Err(FusionError::ConfigError(format!(
            "{label} vectors must be non-empty"
        )));
    }
    if prediction.len() != measurements.len() {
        return Err(FusionError::ConfigError(format!(
            "{label} length mismatch: prediction={}, measurements={}",
            prediction.len(),
            measurements.len()
        )));
    }
    if prediction.iter().any(|v| !v.is_finite()) || measurements.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::ConfigError(format!(
            "{label} inputs must be finite"
        )));
    }
    let rmse = (prediction
        .iter()
        .zip(measurements.iter())
        .map(|(p, m)| (p - m).powi(2))
        .sum::<f64>()
        / measurements.len() as f64)
        .sqrt();
    if !rmse.is_finite() {
        return Err(FusionError::LinAlg(format!(
            "{label} rmse became non-finite"
        )));
    }
    Ok(rmse)
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

fn mtanh_profile_dpsi_norm(
    psi_norm: f64,
    params: &ProfileParams,
    label: &str,
) -> FusionResult<f64> {
    if !psi_norm.is_finite() {
        return Err(FusionError::ConfigError(format!(
            "{label}.psi_norm must be finite, got {psi_norm}"
        )));
    }
    validate_profile_params(params, label)?;
    let w = params.ped_width;
    let ped_top = params.ped_top;
    let y = (params.ped_top - psi_norm) / w;
    let tanh_y = y.tanh();
    let sech2 = 1.0 - tanh_y * tanh_y;

    let d_edge = -0.5 * params.ped_height * sech2 / w;
    let d_core = if psi_norm.abs() < ped_top {
        -2.0 * params.core_alpha * psi_norm / ped_top.powi(2)
    } else {
        0.0
    };
    let d = d_edge + d_core;
    if !d.is_finite() {
        return Err(FusionError::ConfigError(format!(
            "{label}.d_profile_dpsi_norm became non-finite"
        )));
    }
    Ok(d)
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
    validate_profile_params(&params_p, "params_p")?;
    validate_profile_params(&params_ff, "params_ff")?;
    validate_probe_coordinates(probes_rz)?;
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

    let observables = kernel.sample_psi_at_probes(probes_rz)?;
    validate_observables(&observables, probes_rz.len(), "kernel forward observables")?;
    Ok(observables)
}

fn solve_linearized_sensitivity(
    grid: &Grid2D,
    ds_dx: &Array2<f64>,
    ds_dpsi: &Array2<f64>,
    iterations: usize,
) -> FusionResult<Array2<f64>> {
    if grid.nz < 3 || grid.nr < 3 {
        return Err(FusionError::ConfigError(format!(
            "sensitivity grid must be at least 3x3, got nz={}, nr={}",
            grid.nz, grid.nr
        )));
    }
    if iterations == 0 {
        return Err(FusionError::ConfigError(
            "sensitivity iterations must be >= 1".to_string(),
        ));
    }
    if !grid.dr.is_finite() || !grid.dz.is_finite() || grid.dr <= 0.0 || grid.dz <= 0.0 {
        return Err(FusionError::ConfigError(format!(
            "sensitivity grid spacing must be finite and > 0, got dr={}, dz={}",
            grid.dr, grid.dz
        )));
    }
    if ds_dx.dim() != (grid.nz, grid.nr) || ds_dpsi.dim() != (grid.nz, grid.nr) {
        return Err(FusionError::ConfigError(format!(
            "sensitivity source shape mismatch: ds_dx={:?}, ds_dpsi={:?}, expected=({}, {})",
            ds_dx.dim(),
            ds_dpsi.dim(),
            grid.nz,
            grid.nr
        )));
    }
    if ds_dx.iter().any(|v| !v.is_finite()) || ds_dpsi.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::ConfigError(
            "sensitivity sources must be finite".to_string(),
        ));
    }

    let mut delta: Array2<f64> = Array2::zeros((grid.nz, grid.nr));
    let mut next = delta.clone();
    let dr_sq: f64 = grid.dr * grid.dr;

    for _ in 0..iterations {
        for iz in 1..grid.nz - 1 {
            for ir in 1..grid.nr - 1 {
                let coupled_rhs: f64 = ds_dpsi[[iz, ir]] * delta[[iz, ir]] + ds_dx[[iz, ir]];
                if !coupled_rhs.is_finite() {
                    return Err(FusionError::LinAlg(
                        "sensitivity coupled RHS became non-finite".to_string(),
                    ));
                }
                let jacobi: f64 = 0.25_f64
                    * (delta[[iz - 1, ir]]
                        + delta[[iz + 1, ir]]
                        + delta[[iz, ir - 1]]
                        + delta[[iz, ir + 1]]
                        - dr_sq * coupled_rhs);
                if !jacobi.is_finite() {
                    return Err(FusionError::LinAlg(
                        "sensitivity jacobi update became non-finite".to_string(),
                    ));
                }
                next[[iz, ir]] = (1.0 - SENSITIVITY_RELAXATION) * delta[[iz, ir]]
                    + SENSITIVITY_RELAXATION * jacobi;
                if !next[[iz, ir]].is_finite() {
                    return Err(FusionError::LinAlg(
                        "sensitivity state update became non-finite".to_string(),
                    ));
                }
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

    if delta.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::LinAlg(
            "sensitivity solution contains non-finite values".to_string(),
        ));
    }
    Ok(delta)
}

fn kernel_analytical_forward_and_jacobian(
    reactor_config: &ReactorConfig,
    probes_rz: &[(f64, f64)],
    params_p: ProfileParams,
    params_ff: ProfileParams,
    kernel_cfg: &KernelInverseConfig,
) -> FusionResult<(Vec<f64>, Vec<Vec<f64>>)> {
    validate_kernel_inverse_config(kernel_cfg)?;
    validate_profile_params(&params_p, "params_p")?;
    validate_profile_params(&params_ff, "params_ff")?;
    validate_probe_coordinates(probes_rz)?;
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

    let base_observables = kernel.sample_psi_at_probes(probes_rz)?;
    validate_observables(
        &base_observables,
        probes_rz.len(),
        "kernel analytical observables",
    )?;
    let grid = kernel.grid();
    let psi = kernel.psi();
    let mu0 = kernel.config().physics.vacuum_permeability;
    let i_target = kernel.config().physics.plasma_current_target;
    if !mu0.is_finite() || mu0 <= 0.0 {
        return Err(FusionError::ConfigError(format!(
            "kernel inverse mu0 must be finite and > 0, got {mu0}"
        )));
    }
    if !i_target.is_finite() {
        return Err(FusionError::ConfigError(format!(
            "kernel inverse i_target must be finite, got {i_target}"
        )));
    }

    let flux_denom = validate_flux_denom(solve_result.psi_boundary - solve_result.psi_axis)?;

    let mut psi_norm = Array2::zeros((grid.nz, grid.nr));
    let mut inside = Array2::from_elem((grid.nz, grid.nr), false);
    let mut raw = Array2::zeros((grid.nz, grid.nr));
    let mut raw_dpsi_norm = Array2::zeros((grid.nz, grid.nr));

    for iz in 0..grid.nz {
        for ir in 0..grid.nr {
            let psi_n = (psi[[iz, ir]] - solve_result.psi_axis) / flux_denom;
            if !psi_n.is_finite() {
                return Err(FusionError::ConfigError(format!(
                    "kernel inverse psi_norm must be finite, got {psi_n}"
                )));
            }
            psi_norm[[iz, ir]] = psi_n;
            if !(0.0..1.0).contains(&psi_n) {
                continue;
            }

            let r = validate_radius(grid.rr[[iz, ir]], "kernel inverse in-plasma radius")?;
            let p = mtanh_profile(psi_n, &params_p);
            let ff = mtanh_profile(psi_n, &params_ff);
            let dp_dpsi = mtanh_profile_dpsi_norm(psi_n, &params_p, "params_p")?;
            let dff_dpsi = mtanh_profile_dpsi_norm(psi_n, &params_ff, "params_ff")?;

            raw[[iz, ir]] = SOURCE_BETA_MIX * r * p + (1.0 - SOURCE_BETA_MIX) * (ff / (mu0 * r));
            raw_dpsi_norm[[iz, ir]] =
                SOURCE_BETA_MIX * r * dp_dpsi + (1.0 - SOURCE_BETA_MIX) * (dff_dpsi / (mu0 * r));
            inside[[iz, ir]] = true;
        }
    }

    let i_raw = raw.iter().sum::<f64>() * grid.dr * grid.dz;
    if !i_raw.is_finite() || i_raw.abs() <= MIN_CURRENT_INTEGRAL {
        return Err(FusionError::ConfigError(format!(
            "kernel inverse raw current integral must be finite and |i_raw| > {MIN_CURRENT_INTEGRAL}, got {i_raw}"
        )));
    }
    let scale = i_target / i_raw;
    if !scale.is_finite() {
        return Err(FusionError::ConfigError(
            "kernel inverse source scaling became non-finite".to_string(),
        ));
    }

    // Approximate local dS/dPsi for linearized kernel solve.
    let mut ds_dpsi = Array2::zeros((grid.nz, grid.nr));
    for iz in 0..grid.nz {
        for ir in 0..grid.nr {
            if !inside[[iz, ir]] {
                continue;
            }
            let r = validate_radius(grid.rr[[iz, ir]], "kernel inverse in-plasma radius")?;
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
                let r = validate_radius(grid.rr[[iz, ir]], "kernel inverse in-plasma radius")?;
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
                let r = validate_radius(grid.rr[[iz, ir]], "kernel inverse in-plasma radius")?;
                let dj = scale * (d_raw[[iz, ir]] - raw[[iz, ir]] * d_i / i_raw);
                ds_dx[[iz, ir]] = -mu0 * r * dj;
            }
        }

        let delta_psi = solve_linearized_sensitivity(grid, &ds_dx, &ds_dpsi, sens_iters)?;
        for (row, &(iz, ir)) in probe_idx.iter().enumerate() {
            jac[row][col] = delta_psi[[iz, ir]];
        }
    }

    validate_jacobian_matrix(
        &jac,
        probes_rz.len(),
        N_PARAMS,
        "kernel analytical jacobian",
    )?;
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
    validate_probe_coordinates(probes_rz)?;
    validate_observables(base, probes_rz.len(), "kernel fd base observables")?;
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
    validate_jacobian_matrix(&jac, probes_rz.len(), N_PARAMS, "kernel fd jacobian")?;
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
    validate_probe_psi_norm(probe_psi_norm)?;
    validate_measurements(measurements)?;
    validate_profile_params(&initial_params_p, "initial_params_p")?;
    validate_profile_params(&initial_params_ff, "initial_params_ff")?;

    let mut x = pack_params(&initial_params_p, &initial_params_ff);
    let mut residual_history = Vec::with_capacity(config.max_iterations + 1);
    let mut converged = false;
    let mut iter_done = 0;
    let mut damping = config.damping;

    for iter in 0..config.max_iterations {
        let (params_p, params_ff) = unpack_params(&x);
        let prediction = forward_model_response(probe_psi_norm, &params_p, &params_ff)?;
        let residual_vec: Vec<f64> = prediction
            .iter()
            .zip(measurements.iter())
            .map(|(p, m)| p - m)
            .collect();
        let residual = compute_rmse(&prediction, measurements, "inverse residual")?;
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
        }?;

        let j = to_array2(&jac, N_PARAMS, "inverse jacobian")?;
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
        validate_update_vector(&delta, "inverse.delta")?;

        let mut accepted = false;
        let mut local_damping = damping;
        for _ in 0..8 {
            let mut x_trial = x;
            for i in 0..N_PARAMS {
                x_trial[i] += local_damping * delta[i];
            }
            let (p_trial, ff_trial) = unpack_params(&x_trial);
            if validate_profile_params(&p_trial, "params_p").is_err()
                || validate_profile_params(&ff_trial, "params_ff").is_err()
            {
                local_damping *= 0.5;
                continue;
            }
            let pred_trial = forward_model_response(probe_psi_norm, &p_trial, &ff_trial)?;
            let residual_trial = compute_rmse(&pred_trial, measurements, "inverse trial residual")?;

            if residual_trial <= residual {
                x = x_trial;
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
    validate_profile_params(&params_p, "params_p")?;
    validate_profile_params(&params_ff, "params_ff")?;
    let prediction = forward_model_response(probe_psi_norm, &params_p, &params_ff)?;
    let final_residual = compute_rmse(&prediction, measurements, "inverse final residual")?;

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
    validate_probe_coordinates(probes_rz)?;
    validate_measurements(measurements)?;
    validate_profile_params(&initial_params_p, "initial_params_p")?;
    validate_profile_params(&initial_params_ff, "initial_params_ff")?;

    let mut x = pack_params(&initial_params_p, &initial_params_ff);
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
        let residual = compute_rmse(&prediction, measurements, "kernel inverse residual")?;
        residual_history.push(residual);
        iter_done = iter + 1;

        if residual < kernel_cfg.inverse.tolerance {
            converged = true;
            break;
        }

        let j = to_array2(&jac, N_PARAMS, "kernel inverse jacobian")?;

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
        validate_update_vector(&delta, "kernel_inverse.delta")?;

        let mut accepted = false;
        let mut local_damping = damping;
        for _ in 0..6 {
            let mut x_trial = x;
            for i in 0..N_PARAMS {
                x_trial[i] += local_damping * delta[i];
            }
            let (p_trial, ff_trial) = unpack_params(&x_trial);
            if validate_profile_params(&p_trial, "params_p").is_err()
                || validate_profile_params(&ff_trial, "params_ff").is_err()
            {
                local_damping *= 0.5;
                continue;
            }
            let pred_trial = kernel_forward_observables(
                reactor_config,
                probes_rz,
                p_trial,
                ff_trial,
                kernel_cfg,
            )?;
            let residual_trial =
                compute_rmse(&pred_trial, measurements, "kernel inverse trial residual")?;
            if residual_trial <= residual {
                x = x_trial;
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
    validate_profile_params(&params_p, "params_p")?;
    validate_profile_params(&params_ff, "params_ff")?;
    let prediction =
        kernel_forward_observables(reactor_config, probes_rz, params_p, params_ff, kernel_cfg)?;
    let final_residual = compute_rmse(&prediction, measurements, "kernel inverse final residual")?;

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
    use ndarray::{Array1, Array2};
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
        let measurements = forward_model_response(&probes, &true_p, &true_ff).unwrap();

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
        let init_prediction = forward_model_response(&probes, &init_p, &init_ff).unwrap();
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
        let measurements = forward_model_response(&probes, &true_p, &true_ff).unwrap();

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
    fn test_inverse_rejects_invalid_initial_profile_params() {
        let probes: Vec<f64> = (0..8).map(|i| i as f64 / 7.0).collect();
        let measurements = vec![0.1; probes.len()];
        let bad_init = ProfileParams {
            ped_top: 0.0,
            ..ProfileParams::default()
        };
        let cfg = InverseConfig::default();
        let err = reconstruct_equilibrium(
            &probes,
            &measurements,
            bad_init,
            ProfileParams::default(),
            &cfg,
        )
        .expect_err("invalid initial profile params must error");
        match err {
            FusionError::ConfigError(msg) => assert!(msg.contains("initial_params_p.ped_top")),
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_inverse_runtime_scalar_guards_reject_invalid_flux_and_radius() {
        assert!(validate_flux_denom(0.0).is_err());
        assert!(validate_flux_denom(f64::NAN).is_err());
        assert!(validate_flux_denom(1e-3).is_ok());

        assert!(validate_radius(0.0, "r").is_err());
        assert!(validate_radius(-0.5, "r").is_err());
        assert!(validate_radius(f64::INFINITY, "r").is_err());
        assert!(validate_radius(0.25, "r").is_ok());
    }

    #[test]
    fn test_inverse_update_vector_validation_guards() {
        let bad_len = Array1::zeros(N_PARAMS - 1);
        assert!(validate_update_vector(&bad_len, "delta").is_err());

        let mut bad_values = Array1::zeros(N_PARAMS);
        bad_values[2] = f64::NAN;
        assert!(validate_update_vector(&bad_values, "delta").is_err());

        let ok = Array1::zeros(N_PARAMS);
        assert!(validate_update_vector(&ok, "delta").is_ok());
    }

    #[test]
    fn test_inverse_observable_and_jacobian_validation_guards() {
        assert!(validate_observables(&[1.0, 2.0], 2, "obs").is_ok());
        assert!(validate_observables(&[1.0, f64::NAN], 2, "obs").is_err());
        assert!(validate_observables(&[1.0], 2, "obs").is_err());

        let jac_ok = vec![vec![0.0; N_PARAMS]; 2];
        assert!(validate_jacobian_matrix(&jac_ok, 2, N_PARAMS, "jac").is_ok());

        let jac_bad_rows = vec![vec![0.0; N_PARAMS]; 1];
        assert!(validate_jacobian_matrix(&jac_bad_rows, 2, N_PARAMS, "jac").is_err());

        let jac_bad_cols = vec![vec![0.0; N_PARAMS - 1]; 2];
        assert!(validate_jacobian_matrix(&jac_bad_cols, 2, N_PARAMS, "jac").is_err());

        let mut jac_bad_vals = vec![vec![0.0; N_PARAMS]; 2];
        jac_bad_vals[0][3] = f64::INFINITY;
        assert!(validate_jacobian_matrix(&jac_bad_vals, 2, N_PARAMS, "jac").is_err());
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
        let kcfg = KernelInverseConfig {
            kernel_max_iterations: 0,
            ..Default::default()
        };
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
    fn test_kernel_inverse_rejects_invalid_initial_profile_params() {
        let cfg = ReactorConfig::from_file(&config_path("validation/iter_validated_config.json"))
            .unwrap();
        let kcfg = KernelInverseConfig::default();
        let bad_init = ProfileParams {
            ped_width: 0.0,
            ..ProfileParams::default()
        };
        let err = reconstruct_equilibrium_with_kernel(
            &cfg,
            &[(6.2, 0.0), (6.3, 0.1)],
            &[1.0, 1.0],
            bad_init,
            ProfileParams::default(),
            &kcfg,
        )
        .expect_err("invalid kernel initial profile params must error");
        match err {
            FusionError::ConfigError(msg) => assert!(msg.contains("initial_params_p.ped_width")),
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_inverse_rejects_non_finite_measurements() {
        let probes: Vec<f64> = (0..8).map(|i| i as f64 / 7.0).collect();
        let mut measurements = vec![0.1; probes.len()];
        measurements[3] = f64::NAN;
        let err = reconstruct_equilibrium(
            &probes,
            &measurements,
            ProfileParams::default(),
            ProfileParams::default(),
            &InverseConfig::default(),
        )
        .expect_err("non-finite measurements must error");
        match err {
            FusionError::ConfigError(msg) => assert!(msg.contains("measurements must be finite")),
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_kernel_inverse_rejects_non_finite_measurements_or_probes() {
        let cfg = ReactorConfig::from_file(&config_path("validation/iter_validated_config.json"))
            .unwrap();
        let kcfg = KernelInverseConfig::default();
        let bad_probe_err = reconstruct_equilibrium_with_kernel(
            &cfg,
            &[(6.2, 0.0), (f64::INFINITY, 0.1)],
            &[1.0, 1.0],
            ProfileParams::default(),
            ProfileParams::default(),
            &kcfg,
        )
        .expect_err("non-finite probe coordinates must error");
        match bad_probe_err {
            FusionError::ConfigError(msg) => {
                assert!(msg.contains("probes_rz coordinates must be finite"))
            }
            other => panic!("Unexpected error: {other:?}"),
        }

        let bad_measurement_err = reconstruct_equilibrium_with_kernel(
            &cfg,
            &[(6.2, 0.0), (6.3, 0.1)],
            &[1.0, f64::NAN],
            ProfileParams::default(),
            ProfileParams::default(),
            &kcfg,
        )
        .expect_err("non-finite measurements must error");
        match bad_measurement_err {
            FusionError::ConfigError(msg) => assert!(msg.contains("measurements must be finite")),
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_kernel_fd_jacobian_rejects_base_length_mismatch() {
        let cfg = ReactorConfig::from_file(&config_path("validation/iter_validated_config.json"))
            .unwrap();
        let probes = vec![(6.2, 0.0), (6.3, 0.1)];
        let kcfg = KernelInverseConfig::default();
        let params = ProfileParams::default();
        let err = kernel_fd_jacobian_from_base(&cfg, &probes, params, params, &kcfg, &[0.1])
            .expect_err("base observables length mismatch must error");
        match err {
            FusionError::ConfigError(msg) => {
                assert!(msg.contains("kernel fd base observables length mismatch"))
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_sensitivity_solver_rejects_invalid_inputs() {
        let grid = Grid2D::new(4, 4, 1.0, 2.0, -1.0, 1.0);
        let ds = Array2::zeros((4, 4));
        assert!(solve_linearized_sensitivity(&grid, &ds, &ds, 2).is_ok());

        let bad_shape = Array2::zeros((3, 4));
        assert!(solve_linearized_sensitivity(&grid, &bad_shape, &ds, 2).is_err());
        assert!(solve_linearized_sensitivity(&grid, &ds, &bad_shape, 2).is_err());

        let mut bad_values = Array2::zeros((4, 4));
        bad_values[[1, 1]] = f64::NAN;
        assert!(solve_linearized_sensitivity(&grid, &bad_values, &ds, 2).is_err());

        assert!(solve_linearized_sensitivity(&grid, &ds, &ds, 0).is_err());

        let small_grid = Grid2D::new(2, 2, 1.0, 2.0, -1.0, 1.0);
        let ds_small = Array2::zeros((2, 2));
        assert!(solve_linearized_sensitivity(&small_grid, &ds_small, &ds_small, 1).is_err());
    }

    #[test]
    fn test_to_array2_rejects_invalid_shapes_or_values() {
        let jac_ok = vec![vec![0.0; N_PARAMS]; 2];
        assert!(to_array2(&jac_ok, N_PARAMS, "jac").is_ok());

        let jac_jagged = vec![vec![0.0; N_PARAMS], vec![0.0; N_PARAMS - 1]];
        assert!(to_array2(&jac_jagged, N_PARAMS, "jac").is_err());

        let mut jac_bad = vec![vec![0.0; N_PARAMS]; 2];
        jac_bad[1][2] = f64::NAN;
        assert!(to_array2(&jac_bad, N_PARAMS, "jac").is_err());

        let empty: Vec<Vec<f64>> = Vec::new();
        assert!(to_array2(&empty, N_PARAMS, "jac").is_err());
    }

    #[test]
    fn test_rmse_helper_rejects_invalid_inputs() {
        assert!(compute_rmse(&[1.0, 2.0], &[1.0, 2.0], "rmse").is_ok());
        assert!(compute_rmse(&[], &[1.0], "rmse").is_err());
        assert!(compute_rmse(&[1.0], &[], "rmse").is_err());
        assert!(compute_rmse(&[1.0], &[1.0, 2.0], "rmse").is_err());
        assert!(compute_rmse(&[1.0, f64::NAN], &[1.0, 2.0], "rmse").is_err());
        assert!(compute_rmse(&[1.0, 2.0], &[1.0, f64::INFINITY], "rmse").is_err());
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
