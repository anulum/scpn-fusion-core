// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — VMEC Interface
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Lightweight VMEC-compatible boundary-state wrapper.
//!
//! This module intentionally does not implement a full 3D force-balance solve.
//! Instead it provides a deterministic interoperability lane for exchanging
//! reduced Fourier boundary states with external VMEC-class workflows.

use fusion_types::error::{FusionError, FusionResult};
use ndarray::{Array1, Array2};
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VmecFourierMode {
    pub m: i32,
    pub n: i32,
    pub r_cos: f64,
    pub r_sin: f64,
    pub z_cos: f64,
    pub z_sin: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VmecBoundaryState {
    pub r_axis: f64,
    pub z_axis: f64,
    pub a_minor: f64,
    pub kappa: f64,
    pub triangularity: f64,
    pub nfp: usize,
    pub modes: Vec<VmecFourierMode>,
}

impl VmecBoundaryState {
    pub fn validate(&self) -> FusionResult<()> {
        if !self.r_axis.is_finite()
            || !self.z_axis.is_finite()
            || !self.a_minor.is_finite()
            || !self.kappa.is_finite()
            || !self.triangularity.is_finite()
        {
            return Err(FusionError::PhysicsViolation(
                "VMEC boundary contains non-finite scalar".to_string(),
            ));
        }
        if self.a_minor <= 0.0 {
            return Err(FusionError::PhysicsViolation(
                "VMEC boundary requires a_minor > 0".to_string(),
            ));
        }
        if self.kappa <= 0.0 {
            return Err(FusionError::PhysicsViolation(
                "VMEC boundary requires kappa > 0".to_string(),
            ));
        }
        if self.nfp < 1 {
            return Err(FusionError::PhysicsViolation(
                "VMEC boundary requires nfp >= 1".to_string(),
            ));
        }
        let mut seen_modes: HashSet<(i32, i32)> = HashSet::with_capacity(self.modes.len());
        for (idx, mode) in self.modes.iter().enumerate() {
            if mode.m < 0 {
                return Err(FusionError::PhysicsViolation(format!(
                    "VMEC boundary mode[{idx}] requires m >= 0, got {}",
                    mode.m
                )));
            }
            if !mode.r_cos.is_finite()
                || !mode.r_sin.is_finite()
                || !mode.z_cos.is_finite()
                || !mode.z_sin.is_finite()
            {
                return Err(FusionError::PhysicsViolation(format!(
                    "VMEC boundary mode[{idx}] contains non-finite coefficients"
                )));
            }
            let key = (mode.m, mode.n);
            if !seen_modes.insert(key) {
                return Err(FusionError::PhysicsViolation(format!(
                    "VMEC boundary contains duplicate mode (m={}, n={})",
                    mode.m, mode.n
                )));
            }
        }
        Ok(())
    }
}

pub fn export_vmec_like_text(state: &VmecBoundaryState) -> FusionResult<String> {
    state.validate()?;
    let mut out = String::new();
    out.push_str("format=vmec_like_v1\n");
    out.push_str(&format!("r_axis={:.16e}\n", state.r_axis));
    out.push_str(&format!("z_axis={:.16e}\n", state.z_axis));
    out.push_str(&format!("a_minor={:.16e}\n", state.a_minor));
    out.push_str(&format!("kappa={:.16e}\n", state.kappa));
    out.push_str(&format!("triangularity={:.16e}\n", state.triangularity));
    out.push_str(&format!("nfp={}\n", state.nfp));
    for mode in &state.modes {
        out.push_str(&format!(
            "mode,{},{},{:.16e},{:.16e},{:.16e},{:.16e}\n",
            mode.m, mode.n, mode.r_cos, mode.r_sin, mode.z_cos, mode.z_sin
        ));
    }
    Ok(out)
}

fn parse_float(key: &str, text: &str) -> FusionResult<f64> {
    let val = text.parse::<f64>().map_err(|e| {
        FusionError::PhysicsViolation(format!("Failed to parse VMEC key '{key}' as float: {e}"))
    })?;
    if !val.is_finite() {
        return Err(FusionError::PhysicsViolation(format!(
            "VMEC key '{key}' must be finite, got {val}"
        )));
    }
    Ok(val)
}

fn parse_int<T>(key: &str, text: &str) -> FusionResult<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    text.parse::<T>().map_err(|e| {
        FusionError::PhysicsViolation(format!("Failed to parse VMEC key '{key}' as integer: {e}"))
    })
}

pub fn import_vmec_like_text(text: &str) -> FusionResult<VmecBoundaryState> {
    let mut format_seen = false;
    let mut r_axis: Option<f64> = None;
    let mut z_axis: Option<f64> = None;
    let mut a_minor: Option<f64> = None;
    let mut kappa: Option<f64> = None;
    let mut triangularity: Option<f64> = None;
    let mut nfp: Option<usize> = None;
    let mut modes: Vec<VmecFourierMode> = Vec::new();

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(format_name) = line.strip_prefix("format=") {
            if format_seen {
                return Err(FusionError::PhysicsViolation(
                    "Duplicate VMEC key: format".to_string(),
                ));
            }
            if format_name.trim() != "vmec_like_v1" {
                return Err(FusionError::PhysicsViolation(format!(
                    "Unsupported VMEC format: {}",
                    format_name.trim()
                )));
            }
            format_seen = true;
            continue;
        }
        if let Some(rest) = line.strip_prefix("mode,") {
            let cols: Vec<&str> = rest.split(',').map(|v| v.trim()).collect();
            if cols.len() != 6 {
                return Err(FusionError::PhysicsViolation(
                    "VMEC mode line must contain exactly 6 columns".to_string(),
                ));
            }
            modes.push(VmecFourierMode {
                m: parse_int("mode.m", cols[0])?,
                n: parse_int("mode.n", cols[1])?,
                r_cos: parse_float("mode.r_cos", cols[2])?,
                r_sin: parse_float("mode.r_sin", cols[3])?,
                z_cos: parse_float("mode.z_cos", cols[4])?,
                z_sin: parse_float("mode.z_sin", cols[5])?,
            });
            continue;
        }
        let (key, value) = line.split_once('=').ok_or_else(|| {
            FusionError::PhysicsViolation(format!("Invalid VMEC line (missing '='): {line}"))
        })?;
        let key = key.trim();
        let value = value.trim();
        match key {
            "r_axis" => {
                if r_axis.is_some() {
                    return Err(FusionError::PhysicsViolation(
                        "Duplicate VMEC key: r_axis".to_string(),
                    ));
                }
                r_axis = Some(parse_float(key, value)?);
            }
            "z_axis" => {
                if z_axis.is_some() {
                    return Err(FusionError::PhysicsViolation(
                        "Duplicate VMEC key: z_axis".to_string(),
                    ));
                }
                z_axis = Some(parse_float(key, value)?);
            }
            "a_minor" => {
                if a_minor.is_some() {
                    return Err(FusionError::PhysicsViolation(
                        "Duplicate VMEC key: a_minor".to_string(),
                    ));
                }
                a_minor = Some(parse_float(key, value)?);
            }
            "kappa" => {
                if kappa.is_some() {
                    return Err(FusionError::PhysicsViolation(
                        "Duplicate VMEC key: kappa".to_string(),
                    ));
                }
                kappa = Some(parse_float(key, value)?);
            }
            "triangularity" => {
                if triangularity.is_some() {
                    return Err(FusionError::PhysicsViolation(
                        "Duplicate VMEC key: triangularity".to_string(),
                    ));
                }
                triangularity = Some(parse_float(key, value)?);
            }
            "nfp" => {
                if nfp.is_some() {
                    return Err(FusionError::PhysicsViolation(
                        "Duplicate VMEC key: nfp".to_string(),
                    ));
                }
                nfp = Some(parse_int(key, value)?);
            }
            other => {
                return Err(FusionError::PhysicsViolation(format!(
                    "Unknown VMEC key: {other}"
                )));
            }
        }
    }

    let state = VmecBoundaryState {
        r_axis: r_axis
            .ok_or_else(|| FusionError::PhysicsViolation("Missing VMEC key: r_axis".to_string()))?,
        z_axis: z_axis
            .ok_or_else(|| FusionError::PhysicsViolation("Missing VMEC key: z_axis".to_string()))?,
        a_minor: a_minor.ok_or_else(|| {
            FusionError::PhysicsViolation("Missing VMEC key: a_minor".to_string())
        })?,
        kappa: kappa
            .ok_or_else(|| FusionError::PhysicsViolation("Missing VMEC key: kappa".to_string()))?,
        triangularity: triangularity.ok_or_else(|| {
            FusionError::PhysicsViolation("Missing VMEC key: triangularity".to_string())
        })?,
        nfp: nfp
            .ok_or_else(|| FusionError::PhysicsViolation("Missing VMEC key: nfp".to_string()))?,
        modes,
    };
    state.validate()?;
    Ok(state)
}

// ═══════════════════════════════════════════════════════════════════════
// VMEC-like Fixed-Boundary 3D Equilibrium Solver
// ═══════════════════════════════════════════════════════════════════════
//
// Implements a variational equilibrium solver following Hirshman & Whitson
// (1983). Given boundary Fourier modes, pressure p(s), and rotational
// transform iota(s), finds force-balanced interior flux surface shapes via
// steepest descent on the MHD energy functional.
//
// Stellarator symmetry: R uses cos(mθ − nNζ), Z uses sin(mθ − nNζ).

/// Solver configuration for the VMEC fixed-boundary equilibrium.
#[derive(Debug, Clone)]
pub struct VmecSolverConfig {
    /// Maximum poloidal mode number.
    pub m_pol: usize,
    /// Maximum toroidal mode number (0 = axisymmetric).
    pub n_tor: usize,
    /// Number of flux surfaces (radial, including axis + boundary).
    pub ns: usize,
    /// Poloidal angle grid points.
    pub ntheta: usize,
    /// Toroidal angle grid points per field period.
    pub nzeta: usize,
    /// Maximum steepest-descent iterations.
    pub max_iter: usize,
    /// Force residual convergence tolerance.
    pub tol: f64,
    /// Steepest descent step size.
    pub step_size: f64,
}

impl Default for VmecSolverConfig {
    fn default() -> Self {
        Self {
            m_pol: 6,
            n_tor: 0,
            ns: 25,
            ntheta: 32,
            nzeta: 1,
            max_iter: 500,
            tol: 1e-8,
            step_size: 5e-3,
        }
    }
}

impl VmecSolverConfig {
    pub fn validate(&self) -> FusionResult<()> {
        if self.ns < 3 {
            return Err(FusionError::PhysicsViolation(
                "VMEC solver requires ns >= 3".into(),
            ));
        }
        if self.ntheta < 8 {
            return Err(FusionError::PhysicsViolation(
                "VMEC solver requires ntheta >= 8".into(),
            ));
        }
        if self.max_iter == 0 {
            return Err(FusionError::PhysicsViolation(
                "VMEC solver requires max_iter >= 1".into(),
            ));
        }
        if !self.tol.is_finite() || self.tol <= 0.0 {
            return Err(FusionError::PhysicsViolation(
                "VMEC solver tol must be finite and > 0".into(),
            ));
        }
        if !self.step_size.is_finite() || self.step_size <= 0.0 {
            return Err(FusionError::PhysicsViolation(
                "VMEC solver step_size must be finite and > 0".into(),
            ));
        }
        Ok(())
    }
}

/// Solution from the VMEC fixed-boundary solver.
#[derive(Debug, Clone)]
pub struct VmecEquilibrium {
    /// R cosine Fourier coefficients per surface [ns × n_modes].
    pub rmnc: Array2<f64>,
    /// Z sine Fourier coefficients per surface [ns × n_modes].
    pub zmns: Array2<f64>,
    /// Rotational transform profile iota(s) [ns].
    pub iota: Array1<f64>,
    /// Pressure profile [Pa] [ns].
    pub pressure: Array1<f64>,
    /// Total toroidal flux [Wb].
    pub phi_edge: f64,
    /// Plasma volume [m³].
    pub volume: f64,
    /// Volume-averaged beta.
    pub beta_avg: f64,
    /// Final force residual norm.
    pub force_residual: f64,
    /// Number of iterations.
    pub iterations: usize,
    /// Whether the solver converged.
    pub converged: bool,
    /// Grid parameters.
    pub ns_grid: usize,
    pub m_pol: usize,
    pub n_tor: usize,
    pub nfp: usize,
}

/// Number of Fourier modes for given (m_pol, n_tor).
pub fn vmec_n_modes(m_pol: usize, n_tor: usize) -> usize {
    if n_tor == 0 {
        m_pol + 1
    } else {
        (m_pol + 1) * (2 * n_tor + 1)
    }
}

/// Flat index for mode (m, n). Returns None if out of range.
pub fn vmec_mode_idx(m: usize, n: i32, m_pol: usize, n_tor: usize) -> Option<usize> {
    if m > m_pol {
        return None;
    }
    if n_tor == 0 {
        if n != 0 {
            return None;
        }
        Some(m)
    } else {
        let n_abs = n.unsigned_abs() as usize;
        if n_abs > n_tor {
            return None;
        }
        Some(m * (2 * n_tor + 1) + (n + n_tor as i32) as usize)
    }
}

/// Evaluate R(θ,ζ) and Z(θ,ζ) from one surface's Fourier coefficients.
fn eval_surface_point(
    rmnc: &[f64],
    zmns: &[f64],
    theta: f64,
    zeta: f64,
    m_pol: usize,
    n_tor: usize,
    nfp: usize,
) -> (f64, f64) {
    let mut r = 0.0;
    let mut z = 0.0;
    if n_tor == 0 {
        for m in 0..=m_pol {
            let angle = m as f64 * theta;
            let (sin_a, cos_a) = angle.sin_cos();
            r += rmnc[m] * cos_a;
            z += zmns[m] * sin_a;
        }
    } else {
        for m in 0..=m_pol {
            for nn in -(n_tor as i32)..=(n_tor as i32) {
                let idx = m * (2 * n_tor + 1) + (nn + n_tor as i32) as usize;
                let angle = m as f64 * theta - nn as f64 * nfp as f64 * zeta;
                let (sin_a, cos_a) = angle.sin_cos();
                r += rmnc[idx] * cos_a;
                z += zmns[idx] * sin_a;
            }
        }
    }
    (r, z)
}

/// Analytic ∂R/∂θ and ∂Z/∂θ from Fourier coefficients.
fn eval_surface_deriv_theta(
    rmnc: &[f64],
    zmns: &[f64],
    theta: f64,
    zeta: f64,
    m_pol: usize,
    n_tor: usize,
    nfp: usize,
) -> (f64, f64) {
    let mut dr = 0.0;
    let mut dz = 0.0;
    if n_tor == 0 {
        for m in 0..=m_pol {
            let mf = m as f64;
            let angle = mf * theta;
            let (sin_a, cos_a) = angle.sin_cos();
            dr -= mf * rmnc[m] * sin_a;
            dz += mf * zmns[m] * cos_a;
        }
    } else {
        for m in 0..=m_pol {
            let mf = m as f64;
            for nn in -(n_tor as i32)..=(n_tor as i32) {
                let idx = m * (2 * n_tor + 1) + (nn + n_tor as i32) as usize;
                let angle = mf * theta - nn as f64 * nfp as f64 * zeta;
                let (sin_a, cos_a) = angle.sin_cos();
                dr -= mf * rmnc[idx] * sin_a;
                dz += mf * zmns[idx] * cos_a;
            }
        }
    }
    (dr, dz)
}

/// Solve VMEC fixed-boundary 3D equilibrium.
///
/// Given boundary shape (from `VmecBoundaryState`), pressure and rotational
/// transform profiles, finds force-balanced interior flux surface shapes.
///
/// The algorithm minimises the MHD energy functional W = ∫(B²/2μ₀ + p)dV
/// via steepest descent on the interior Fourier coefficients R_mn^c(s),
/// Z_mn^s(s) while holding the axis and boundary fixed.
pub fn vmec_fixed_boundary_solve(
    boundary: &VmecBoundaryState,
    config: &VmecSolverConfig,
    pressure: &[f64],
    iota: &[f64],
    phi_edge: f64,
) -> FusionResult<VmecEquilibrium> {
    boundary.validate()?;
    config.validate()?;

    let ns = config.ns;
    let m_pol = config.m_pol;
    let n_tor = config.n_tor;
    let nfp = boundary.nfp;
    let nmodes = vmec_n_modes(m_pol, n_tor);
    let ntheta = config.ntheta;
    let nzeta = if n_tor == 0 { 1 } else { config.nzeta.max(4) };

    if pressure.len() != ns || iota.len() != ns {
        return Err(FusionError::PhysicsViolation(format!(
            "pressure/iota length must match ns={ns}, got p={} iota={}",
            pressure.len(),
            iota.len()
        )));
    }
    if pressure.iter().any(|v| !v.is_finite()) || iota.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::PhysicsViolation(
            "pressure/iota profiles must be finite".into(),
        ));
    }
    if !phi_edge.is_finite() || phi_edge <= 0.0 {
        return Err(FusionError::PhysicsViolation(
            "phi_edge must be finite and > 0".into(),
        ));
    }

    // Build boundary Fourier coefficients from VmecBoundaryState
    let mut bnd_rmnc = vec![0.0; nmodes];
    let mut bnd_zmns = vec![0.0; nmodes];

    if let Some(idx) = vmec_mode_idx(0, 0, m_pol, n_tor) {
        bnd_rmnc[idx] = boundary.r_axis;
    }
    if let Some(idx) = vmec_mode_idx(1, 0, m_pol, n_tor) {
        bnd_rmnc[idx] = boundary.a_minor;
        bnd_zmns[idx] = boundary.a_minor * boundary.kappa;
    }
    if m_pol >= 2 {
        if let Some(idx) = vmec_mode_idx(2, 0, m_pol, n_tor) {
            bnd_rmnc[idx] = -boundary.triangularity * boundary.a_minor * 0.5;
        }
    }
    for mode in &boundary.modes {
        let m = mode.m as usize;
        if let Some(idx) = vmec_mode_idx(m, mode.n, m_pol, n_tor) {
            bnd_rmnc[idx] += mode.r_cos;
            bnd_zmns[idx] += mode.z_sin;
        }
    }

    // Initialize: linear interpolation from axis to boundary
    let mut rmnc = Array2::zeros((ns, nmodes));
    let mut zmns = Array2::zeros((ns, nmodes));
    let s_grid: Vec<f64> = (0..ns).map(|i| i as f64 / (ns - 1) as f64).collect();

    for js in 0..ns {
        let s = s_grid[js];
        for k in 0..nmodes {
            let axis_r = if k == vmec_mode_idx(0, 0, m_pol, n_tor).unwrap_or(usize::MAX) {
                boundary.r_axis
            } else {
                0.0
            };
            rmnc[[js, k]] = axis_r * (1.0 - s) + bnd_rmnc[k] * s;
            zmns[[js, k]] = bnd_zmns[k] * s;
        }
    }

    let pi = std::f64::consts::PI;
    let mu0 = 4.0e-7 * pi;
    let ds = 1.0 / (ns - 1) as f64;
    let dtheta = 2.0 * pi / ntheta as f64;
    let dzeta = 2.0 * pi / (nzeta as f64 * nfp.max(1) as f64);
    let n_angle_pts = (ntheta * nzeta) as f64;

    // Precompute angles
    let theta_arr: Vec<f64> = (0..ntheta).map(|i| 2.0 * pi * i as f64 / ntheta as f64).collect();
    let zeta_arr: Vec<f64> = (0..nzeta)
        .map(|i| 2.0 * pi * i as f64 / (nzeta as f64 * nfp.max(1) as f64))
        .collect();

    // Steepest descent iteration
    let mut global_force = f64::MAX;
    let mut converged = false;
    let mut iter_count = 0;
    let mut total_volume = 0.0;
    let mut total_beta_num = 0.0;
    let mut total_b2_vol = 0.0;

    for iteration in 0..config.max_iter {
        let mut force_sq_sum = 0.0;
        total_volume = 0.0;
        total_beta_num = 0.0;
        total_b2_vol = 0.0;

        let mut grad_r: Array2<f64> = Array2::zeros((ns, nmodes));
        let mut grad_z: Array2<f64> = Array2::zeros((ns, nmodes));

        // Evaluate force on each interior surface
        for js in 1..(ns - 1) {
            let p = pressure[js];
            let iota_s = iota[js];
            let dp_ds = (pressure[(js + 1).min(ns - 1)] - pressure[js.saturating_sub(1)])
                / (2.0 * ds);

            let rmnc_s: Vec<f64> = (0..nmodes).map(|k| rmnc[[js, k]]).collect();
            let zmns_s: Vec<f64> = (0..nmodes).map(|k| zmns[[js, k]]).collect();
            let rmnc_p: Vec<f64> = (0..nmodes).map(|k| rmnc[[js + 1, k]]).collect();
            let zmns_p: Vec<f64> = (0..nmodes).map(|k| zmns[[js + 1, k]]).collect();
            let rmnc_m: Vec<f64> = (0..nmodes).map(|k| rmnc[[js - 1, k]]).collect();
            let zmns_m: Vec<f64> = (0..nmodes).map(|k| zmns[[js - 1, k]]).collect();

            for it in 0..ntheta {
                let theta = theta_arr[it];
                for iz in 0..nzeta {
                    let zeta = zeta_arr[iz];

                    let (r_val, _z_val) =
                        eval_surface_point(&rmnc_s, &zmns_s, theta, zeta, m_pol, n_tor, nfp);
                    let (r_p, _) =
                        eval_surface_point(&rmnc_p, &zmns_p, theta, zeta, m_pol, n_tor, nfp);
                    let (r_m, _) =
                        eval_surface_point(&rmnc_m, &zmns_m, theta, zeta, m_pol, n_tor, nfp);
                    let (_, z_p) =
                        eval_surface_point(&rmnc_p, &zmns_p, theta, zeta, m_pol, n_tor, nfp);
                    let (_, z_m) =
                        eval_surface_point(&rmnc_m, &zmns_m, theta, zeta, m_pol, n_tor, nfp);

                    let r_s = (r_p - r_m) / (2.0 * ds);
                    let z_s = (z_p - z_m) / (2.0 * ds);

                    let (r_theta, z_theta) =
                        eval_surface_deriv_theta(&rmnc_s, &zmns_s, theta, zeta, m_pol, n_tor, nfp);

                    // Jacobian: √g = R · (R_s Z_θ − R_θ Z_s)
                    let jac = r_val * (r_s * z_theta - r_theta * z_s);
                    let jac_abs = jac.abs().max(1e-20);

                    // B-field: B^ζ = Φ'/(2π√g), B^θ = ι·B^ζ
                    let b_zeta = phi_edge / (2.0 * pi * jac_abs);
                    let b_theta = iota_s * b_zeta;

                    // |B|² via covariant metric
                    let g_tt = r_theta * r_theta + z_theta * z_theta;
                    let g_zz = r_val * r_val;
                    let b_sq = b_theta * b_theta * g_tt + b_zeta * b_zeta * g_zz;

                    // Force residual: F_s = dp/ds + d(B²/2μ₀)/ds
                    let force_s = dp_ds + b_sq / (2.0 * mu0 * jac_abs.max(1e-10));
                    force_sq_sum += force_s * force_s;

                    let dvol = jac_abs * dtheta * dzeta * ds;
                    total_volume += dvol;
                    total_beta_num += p * dvol;
                    total_b2_vol += b_sq * dvol;

                    // Accumulate gradient for steepest descent
                    if n_tor == 0 {
                        for m in 0..=m_pol {
                            let angle = m as f64 * theta;
                            let (sin_a, cos_a) = angle.sin_cos();
                            grad_r[[js, m]] += force_s * cos_a * dtheta * dzeta;
                            grad_z[[js, m]] += force_s * sin_a * dtheta * dzeta;
                        }
                    } else {
                        for m in 0..=m_pol {
                            for nn in -(n_tor as i32)..=(n_tor as i32) {
                                let idx = m * (2 * n_tor + 1) + (nn + n_tor as i32) as usize;
                                let angle = m as f64 * theta - nn as f64 * nfp as f64 * zeta;
                                let (sin_a, cos_a) = angle.sin_cos();
                                grad_r[[js, idx]] += force_s * cos_a * dtheta * dzeta;
                                grad_z[[js, idx]] += force_s * sin_a * dtheta * dzeta;
                            }
                        }
                    }
                }
            }
        }

        let n_interior = ((ns - 2) as f64 * n_angle_pts).max(1.0);
        global_force = (force_sq_sum / n_interior).sqrt();

        // Update interior surfaces
        for js in 1..(ns - 1) {
            for k in 0..nmodes {
                rmnc[[js, k]] -= config.step_size * grad_r[[js, k]] / n_angle_pts.max(1.0);
                zmns[[js, k]] -= config.step_size * grad_z[[js, k]] / n_angle_pts.max(1.0);
            }
        }

        iter_count = iteration + 1;
        if global_force < config.tol {
            converged = true;
            break;
        }
        if !global_force.is_finite() {
            return Err(FusionError::PhysicsViolation(format!(
                "VMEC solver diverged at iteration {}: force={}",
                iteration, global_force
            )));
        }
    }

    let beta_avg = if total_b2_vol > 0.0 {
        2.0 * mu0 * total_beta_num / total_b2_vol
    } else {
        0.0
    };

    Ok(VmecEquilibrium {
        rmnc,
        zmns,
        iota: Array1::from_vec(iota.to_vec()),
        pressure: Array1::from_vec(pressure.to_vec()),
        phi_edge,
        volume: total_volume.abs(),
        beta_avg,
        force_residual: global_force,
        iterations: iter_count,
        converged,
        ns_grid: ns,
        m_pol,
        n_tor,
        nfp,
    })
}

/// Evaluate the equilibrium geometry at given (s, θ, ζ) coordinates.
pub fn vmec_eval_geometry(
    eq: &VmecEquilibrium,
    s_idx: usize,
    theta: f64,
    zeta: f64,
) -> FusionResult<(f64, f64)> {
    if s_idx >= eq.ns_grid {
        return Err(FusionError::PhysicsViolation(format!(
            "Surface index {} >= ns_grid {}",
            s_idx, eq.ns_grid
        )));
    }
    let nmodes = vmec_n_modes(eq.m_pol, eq.n_tor);
    let rmnc_row: Vec<f64> = (0..nmodes).map(|k| eq.rmnc[[s_idx, k]]).collect();
    let zmns_row: Vec<f64> = (0..nmodes).map(|k| eq.zmns[[s_idx, k]]).collect();
    Ok(eval_surface_point(
        &rmnc_row, &zmns_row, theta, zeta, eq.m_pol, eq.n_tor, eq.nfp,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_state() -> VmecBoundaryState {
        VmecBoundaryState {
            r_axis: 6.2,
            z_axis: 0.0,
            a_minor: 2.0,
            kappa: 1.7,
            triangularity: 0.33,
            nfp: 1,
            modes: vec![
                VmecFourierMode {
                    m: 1,
                    n: 1,
                    r_cos: 0.03,
                    r_sin: -0.01,
                    z_cos: 0.0,
                    z_sin: 0.02,
                },
                VmecFourierMode {
                    m: 2,
                    n: 1,
                    r_cos: 0.015,
                    r_sin: 0.0,
                    z_cos: 0.0,
                    z_sin: 0.008,
                },
            ],
        }
    }

    #[test]
    fn test_vmec_text_roundtrip() {
        let state = sample_state();
        let text = export_vmec_like_text(&state).expect("export must succeed");
        let parsed = import_vmec_like_text(&text).expect("import must succeed");
        assert_eq!(parsed.nfp, state.nfp);
        assert_eq!(parsed.modes.len(), state.modes.len());
        assert!((parsed.r_axis - state.r_axis).abs() < 1e-12);
        assert!((parsed.kappa - state.kappa).abs() < 1e-12);
    }

    #[test]
    fn test_vmec_missing_key_errors() {
        let text = "format=vmec_like_v1\nr_axis=6.2\n";
        let err = import_vmec_like_text(text).expect_err("missing keys should error");
        match err {
            FusionError::PhysicsViolation(msg) => assert!(msg.contains("Missing VMEC key")),
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_vmec_invalid_minor_radius_errors() {
        let text = "\
format=vmec_like_v1
r_axis=6.2
z_axis=0.0
a_minor=0.0
kappa=1.7
triangularity=0.2
nfp=1
";
        let err = import_vmec_like_text(text).expect_err("a_minor=0 must fail");
        match err {
            FusionError::PhysicsViolation(msg) => assert!(msg.contains("a_minor")),
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_vmec_rejects_invalid_modes_and_duplicates() {
        let mut state = sample_state();
        state.modes[0].r_cos = f64::NAN;
        let err = export_vmec_like_text(&state).expect_err("non-finite mode coeff must fail");
        match err {
            FusionError::PhysicsViolation(msg) => assert!(msg.contains("mode")),
            other => panic!("Unexpected error: {other:?}"),
        }

        let dup_text = "\
format=vmec_like_v1
r_axis=6.2
r_axis=6.3
z_axis=0.0
a_minor=2.0
kappa=1.7
triangularity=0.2
nfp=1
";
        let err = import_vmec_like_text(dup_text).expect_err("duplicate keys must fail");
        match err {
            FusionError::PhysicsViolation(msg) => assert!(msg.contains("Duplicate VMEC key")),
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_vmec_rejects_duplicate_modes_and_bad_format() {
        let mut state = sample_state();
        state.modes.push(VmecFourierMode {
            m: state.modes[0].m,
            n: state.modes[0].n,
            r_cos: 0.001,
            r_sin: 0.0,
            z_cos: 0.0,
            z_sin: 0.001,
        });
        let err = export_vmec_like_text(&state).expect_err("duplicate mode index must fail");
        match err {
            FusionError::PhysicsViolation(msg) => assert!(msg.contains("duplicate mode")),
            other => panic!("Unexpected error: {other:?}"),
        }

        let bad_format = "\
format=vmec_like_v2
r_axis=6.2
z_axis=0.0
a_minor=2.0
kappa=1.7
triangularity=0.2
nfp=1
";
        let err = import_vmec_like_text(bad_format).expect_err("unsupported format must fail");
        match err {
            FusionError::PhysicsViolation(msg) => assert!(msg.contains("Unsupported VMEC format")),
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_vmec_rejects_malformed_mode_and_duplicate_format() {
        let malformed_mode = "\
format=vmec_like_v1
r_axis=6.2
z_axis=0.0
a_minor=2.0
kappa=1.7
triangularity=0.2
nfp=1
mode,1,1,0.1,0.0,0.2
";
        let err =
            import_vmec_like_text(malformed_mode).expect_err("mode with wrong arity must fail");
        match err {
            FusionError::PhysicsViolation(msg) => assert!(msg.contains("exactly 6 columns")),
            other => panic!("Unexpected error: {other:?}"),
        }

        let duplicate_format = "\
format=vmec_like_v1
format=vmec_like_v1
r_axis=6.2
z_axis=0.0
a_minor=2.0
kappa=1.7
triangularity=0.2
nfp=1
";
        let err =
            import_vmec_like_text(duplicate_format).expect_err("duplicate format key must fail");
        match err {
            FusionError::PhysicsViolation(msg) => {
                assert!(msg.contains("Duplicate VMEC key: format"))
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    // === VMEC Solver Tests ===

    fn iter_like_boundary() -> VmecBoundaryState {
        VmecBoundaryState {
            r_axis: 6.2,
            z_axis: 0.0,
            a_minor: 2.0,
            kappa: 1.7,
            triangularity: 0.33,
            nfp: 1,
            modes: vec![],
        }
    }

    #[test]
    fn test_vmec_mode_indexing() {
        assert_eq!(vmec_n_modes(6, 0), 7);
        assert_eq!(vmec_n_modes(3, 2), 4 * 5);
        assert_eq!(vmec_mode_idx(0, 0, 6, 0), Some(0));
        assert_eq!(vmec_mode_idx(3, 0, 6, 0), Some(3));
        assert_eq!(vmec_mode_idx(0, 1, 6, 0), None);
        assert_eq!(vmec_mode_idx(7, 0, 6, 0), None);
        assert_eq!(vmec_mode_idx(1, -1, 3, 2), Some(1 * 5 + 1));
    }

    #[test]
    fn test_vmec_eval_surface_axis_is_circle() {
        let m_pol = 4;
        let n_tor = 0;
        let nmodes = vmec_n_modes(m_pol, n_tor);
        let mut rmnc = vec![0.0; nmodes];
        let mut zmns = vec![0.0; nmodes];
        rmnc[0] = 6.2; // R_00 = major radius
        rmnc[1] = 2.0; // R_10 = minor radius
        zmns[1] = 2.0; // Z_10 = minor radius (circular cross-section)

        let (r, z) = eval_surface_point(&rmnc, &zmns, 0.0, 0.0, m_pol, n_tor, 1);
        assert!((r - 8.2).abs() < 1e-12, "outboard midplane R");
        assert!(z.abs() < 1e-12, "midplane Z");

        let (r2, z2) = eval_surface_point(
            &rmnc,
            &zmns,
            std::f64::consts::FRAC_PI_2,
            0.0,
            m_pol,
            n_tor,
            1,
        );
        assert!((r2 - 6.2).abs() < 1e-12, "top R = R_axis");
        assert!((z2 - 2.0).abs() < 1e-12, "top Z = a_minor");
    }

    #[test]
    fn test_vmec_solver_runs_axisymmetric() {
        let boundary = iter_like_boundary();
        let config = VmecSolverConfig {
            m_pol: 4,
            n_tor: 0,
            ns: 11,
            ntheta: 16,
            nzeta: 1,
            max_iter: 50,
            tol: 1e-6,
            step_size: 1e-3,
        };
        let ns = config.ns;
        let pressure: Vec<f64> = (0..ns)
            .map(|i| {
                let s = i as f64 / (ns - 1) as f64;
                1e5 * (1.0 - s * s)
            })
            .collect();
        let iota: Vec<f64> = (0..ns)
            .map(|i| {
                let s = i as f64 / (ns - 1) as f64;
                0.3 + 0.7 * s * s
            })
            .collect();

        let eq = vmec_fixed_boundary_solve(&boundary, &config, &pressure, &iota, 1.0)
            .expect("VMEC solve should succeed");

        assert!(eq.iterations > 0);
        assert!(eq.force_residual.is_finite());
        assert!(eq.volume > 0.0, "Volume must be positive");
        assert_eq!(eq.ns_grid, ns);
        assert_eq!(eq.m_pol, 4);
        assert_eq!(eq.nfp, 1);
    }

    #[test]
    fn test_vmec_solver_boundary_preserved() {
        let boundary = iter_like_boundary();
        let config = VmecSolverConfig {
            m_pol: 4,
            n_tor: 0,
            ns: 11,
            ntheta: 16,
            nzeta: 1,
            max_iter: 10,
            tol: 1e-12,
            step_size: 1e-3,
        };
        let ns = config.ns;
        let pressure: Vec<f64> = (0..ns).map(|_| 1e4).collect();
        let iota: Vec<f64> = (0..ns).map(|_| 0.5).collect();

        let eq = vmec_fixed_boundary_solve(&boundary, &config, &pressure, &iota, 1.0).unwrap();

        // Boundary (last surface) R_10 should be a_minor = 2.0
        let r10_bnd = eq.rmnc[[ns - 1, 1]];
        assert!(
            (r10_bnd - 2.0).abs() < 1e-10,
            "Boundary R_10 must be preserved: got {r10_bnd}"
        );
    }

    #[test]
    fn test_vmec_solver_rejects_invalid_inputs() {
        let boundary = iter_like_boundary();
        let config = VmecSolverConfig::default();
        let ns = config.ns;
        let p: Vec<f64> = vec![0.0; ns];
        let iota: Vec<f64> = vec![0.5; ns];

        // Invalid phi_edge
        assert!(vmec_fixed_boundary_solve(&boundary, &config, &p, &iota, -1.0).is_err());
        assert!(vmec_fixed_boundary_solve(&boundary, &config, &p, &iota, f64::NAN).is_err());

        // Wrong profile length
        let short_p: Vec<f64> = vec![0.0; 3];
        assert!(vmec_fixed_boundary_solve(&boundary, &config, &short_p, &iota, 1.0).is_err());

        // Invalid config
        let bad_cfg = VmecSolverConfig {
            ns: 2,
            ..Default::default()
        };
        assert!(bad_cfg.validate().is_err());
    }

    #[test]
    fn test_vmec_eval_geometry_valid() {
        let boundary = iter_like_boundary();
        let config = VmecSolverConfig {
            m_pol: 4,
            n_tor: 0,
            ns: 7,
            ntheta: 16,
            nzeta: 1,
            max_iter: 5,
            tol: 1e-12,
            step_size: 1e-3,
        };
        let ns = config.ns;
        let p: Vec<f64> = (0..ns)
            .map(|i| 1e4 * (1.0 - (i as f64 / (ns - 1) as f64)))
            .collect();
        let iota: Vec<f64> = vec![0.5; ns];

        let eq = vmec_fixed_boundary_solve(&boundary, &config, &p, &iota, 1.0).unwrap();
        let (r, z) = vmec_eval_geometry(&eq, ns - 1, 0.0, 0.0).unwrap();
        assert!(r > 0.0 && r.is_finite());
        assert!(z.is_finite());

        // Out of bounds
        assert!(vmec_eval_geometry(&eq, ns + 5, 0.0, 0.0).is_err());
    }

    #[test]
    fn test_vmec_deriv_theta_consistency() {
        let m_pol = 3;
        let n_tor = 0;
        let _nmodes = vmec_n_modes(m_pol, n_tor);
        let rmnc: Vec<f64> = vec![6.2, 2.0, 0.1, 0.05];
        let zmns: Vec<f64> = vec![0.0, 1.8, 0.08, 0.03];
        let theta = 1.0;
        let eps = 1e-7;

        let (r_p, z_p) = eval_surface_point(&rmnc, &zmns, theta + eps, 0.0, m_pol, n_tor, 1);
        let (r_m, z_m) = eval_surface_point(&rmnc, &zmns, theta - eps, 0.0, m_pol, n_tor, 1);
        let dr_num = (r_p - r_m) / (2.0 * eps);
        let dz_num = (z_p - z_m) / (2.0 * eps);

        let (dr_ana, dz_ana) =
            eval_surface_deriv_theta(&rmnc, &zmns, theta, 0.0, m_pol, n_tor, 1);

        assert!(
            (dr_ana - dr_num).abs() < 1e-5,
            "dR/dθ mismatch: analytic={dr_ana}, numerical={dr_num}"
        );
        assert!(
            (dz_ana - dz_num).abs() < 1e-5,
            "dZ/dθ mismatch: analytic={dz_ana}, numerical={dz_num}"
        );
    }
}
