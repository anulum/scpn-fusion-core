// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — BOUT++ 3D MHD Coupling Interface
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! BOUT++ coupling interface for 3D MHD stability analysis.
//!
//! Generates field-aligned coordinate grids from 2D GS equilibria,
//! computes metric tensors (g^ij, Jacobian, |B|), and exports in
//! a text format compatible with BOUT++ input requirements.
//!
//! Also provides import routines for BOUT++ stability results
//! (growth rates, mode structure).

use fusion_types::error::{FusionError, FusionResult};
use ndarray::Array2;

/// Configuration for BOUT++ grid generation.
#[derive(Debug, Clone)]
pub struct BoutGridConfig {
    /// Number of radial flux surfaces (x-direction in BOUT++).
    pub nx: usize,
    /// Number of poloidal points per surface (y-direction in BOUT++).
    pub ny: usize,
    /// Number of toroidal points (z-direction in BOUT++).
    pub nz: usize,
    /// Inner normalised flux boundary (0 = axis).
    pub psi_inner: f64,
    /// Outer normalised flux boundary (1 = separatrix).
    pub psi_outer: f64,
}

impl Default for BoutGridConfig {
    fn default() -> Self {
        Self {
            nx: 36,
            ny: 64,
            nz: 32,
            psi_inner: 0.1,
            psi_outer: 0.95,
        }
    }
}

impl BoutGridConfig {
    pub fn validate(&self) -> FusionResult<()> {
        if self.nx < 4 {
            return Err(FusionError::PhysicsViolation(
                "BOUT++ grid requires nx >= 4".into(),
            ));
        }
        if self.ny < 8 {
            return Err(FusionError::PhysicsViolation(
                "BOUT++ grid requires ny >= 8".into(),
            ));
        }
        if self.nz < 4 {
            return Err(FusionError::PhysicsViolation(
                "BOUT++ grid requires nz >= 4".into(),
            ));
        }
        if !self.psi_inner.is_finite()
            || !self.psi_outer.is_finite()
            || self.psi_inner < 0.0
            || self.psi_outer > 1.0
            || self.psi_inner >= self.psi_outer
        {
            return Err(FusionError::PhysicsViolation(format!(
                "BOUT++ grid psi bounds invalid: inner={}, outer={}",
                self.psi_inner, self.psi_outer
            )));
        }
        Ok(())
    }
}

/// BOUT++ field-aligned grid with metric tensors.
#[derive(Debug, Clone)]
pub struct BoutGrid {
    /// Number of radial points.
    pub nx: usize,
    /// Number of poloidal points.
    pub ny: usize,
    /// R coordinates on the grid [nx × ny].
    pub r_grid: Array2<f64>,
    /// Z coordinates on the grid [nx × ny].
    pub z_grid: Array2<f64>,
    /// Normalised poloidal flux ψ_N on the grid [nx × ny].
    pub psi_n: Array2<f64>,
    /// Magnetic field magnitude |B| [T] on the grid [nx × ny].
    pub b_mag: Array2<f64>,
    /// Contravariant metric g^{xx} (radial-radial) [nx × ny].
    pub g_xx: Array2<f64>,
    /// Contravariant metric g^{yy} (poloidal-poloidal) [nx × ny].
    pub g_yy: Array2<f64>,
    /// Contravariant metric g^{zz} (toroidal-toroidal) [nx × ny].
    pub g_zz: Array2<f64>,
    /// Contravariant metric g^{xy} (radial-poloidal) [nx × ny].
    pub g_xy: Array2<f64>,
    /// Jacobian J [nx × ny].
    pub jacobian: Array2<f64>,
    /// Safety factor q(ψ) [nx].
    pub q_profile: Vec<f64>,
    /// Toroidal field B_toroidal [T].
    pub b_toroidal: f64,
}

/// Generate a BOUT++ field-aligned grid from a 2D equilibrium.
///
/// Takes the poloidal flux ψ(R,Z) on a rectangular (R,Z) grid and
/// traces flux surfaces to build field-aligned coordinates.
///
/// # Arguments
/// * `psi` — Poloidal flux on (nz_eq, nr_eq) rectangular grid
/// * `r_axis` — R coordinates of the equilibrium grid [nr_eq]
/// * `z_axis` — Z coordinates of the equilibrium grid [nz_eq]
/// * `psi_axis` — Flux at the magnetic axis
/// * `psi_boundary` — Flux at the separatrix/boundary
/// * `b_toroidal` — Toroidal magnetic field at geometric center [T]
/// * `config` — BOUT++ grid configuration
pub fn generate_bout_grid(
    psi: &Array2<f64>,
    r_axis: &[f64],
    z_axis: &[f64],
    psi_axis: f64,
    psi_boundary: f64,
    b_toroidal: f64,
    config: &BoutGridConfig,
) -> FusionResult<BoutGrid> {
    config.validate()?;

    let nz_eq = psi.nrows();
    let nr_eq = psi.ncols();
    if nz_eq < 4 || nr_eq < 4 {
        return Err(FusionError::PhysicsViolation(format!(
            "Equilibrium grid too small: {}×{}",
            nz_eq, nr_eq
        )));
    }
    if r_axis.len() != nr_eq || z_axis.len() != nz_eq {
        return Err(FusionError::PhysicsViolation(
            "r_axis/z_axis length must match psi dimensions".into(),
        ));
    }
    if !psi_axis.is_finite() || !psi_boundary.is_finite() {
        return Err(FusionError::PhysicsViolation(
            "psi_axis/psi_boundary must be finite".into(),
        ));
    }
    let psi_range = (psi_boundary - psi_axis).abs();
    if psi_range < 1e-12 {
        return Err(FusionError::PhysicsViolation(
            "psi_axis and psi_boundary too close".into(),
        ));
    }
    if !b_toroidal.is_finite() || b_toroidal.abs() < 1e-6 {
        return Err(FusionError::PhysicsViolation(
            "b_toroidal must be finite and non-negligible".into(),
        ));
    }

    let nx = config.nx;
    let ny = config.ny;
    let dr = (r_axis[nr_eq - 1] - r_axis[0]) / (nr_eq - 1) as f64;
    let dz_eq = (z_axis[nz_eq - 1] - z_axis[0]) / (nz_eq - 1) as f64;

    // Generate normalised flux surfaces
    let psi_n_surfaces: Vec<f64> = (0..nx)
        .map(|i| config.psi_inner + (config.psi_outer - config.psi_inner) * i as f64 / (nx - 1) as f64)
        .collect();

    let mut r_grid = Array2::zeros((nx, ny));
    let mut z_grid = Array2::zeros((nx, ny));
    let mut psi_n_grid = Array2::zeros((nx, ny));
    let mut b_mag = Array2::zeros((nx, ny));
    let mut g_xx = Array2::zeros((nx, ny));
    let mut g_yy = Array2::zeros((nx, ny));
    let mut g_zz = Array2::zeros((nx, ny));
    let mut g_xy = Array2::zeros((nx, ny));
    let mut jacobian_grid = Array2::zeros((nx, ny));
    let mut q_profile = vec![0.0; nx];

    // Find magnetic axis position (maximum of psi)
    let mut r_ax = r_axis[nr_eq / 2];
    let mut z_ax = z_axis[nz_eq / 2];
    let mut max_psi = f64::NEG_INFINITY;
    for iz in 0..nz_eq {
        for ir in 0..nr_eq {
            if psi[[iz, ir]] > max_psi {
                max_psi = psi[[iz, ir]];
                r_ax = r_axis[ir];
                z_ax = z_axis[iz];
            }
        }
    }

    // For each flux surface, trace poloidal contour
    let pi2 = 2.0 * std::f64::consts::PI;
    for ix in 0..nx {
        let psi_target = psi_axis + psi_n_surfaces[ix] * (psi_boundary - psi_axis);
        let rho_est = psi_n_surfaces[ix].sqrt()
            * 0.5
            * (r_axis[nr_eq - 1] - r_axis[0]);

        for iy in 0..ny {
            let theta = pi2 * iy as f64 / ny as f64;
            let (cos_t, sin_t) = theta.sin_cos();

            // Initial guess: approximate elliptical contour
            let r_guess = r_ax + rho_est * sin_t;
            let z_guess = z_ax + rho_est * 1.5 * cos_t;

            // Newton iteration to find (R,Z) on the ψ contour
            let mut r_pt = r_guess;
            let mut z_pt = z_guess;

            for _newton in 0..20 {
                // Bilinear interpolation of ψ at (r_pt, z_pt)
                let ir_f = (r_pt - r_axis[0]) / dr;
                let iz_f = (z_pt - z_axis[0]) / dz_eq;
                let ir0 = (ir_f as usize).min(nr_eq - 2);
                let iz0 = (iz_f as usize).min(nz_eq - 2);
                let fr = (ir_f - ir0 as f64).clamp(0.0, 1.0);
                let fz = (iz_f - iz0 as f64).clamp(0.0, 1.0);

                let psi_interp = psi[[iz0, ir0]] * (1.0 - fr) * (1.0 - fz)
                    + psi[[iz0, ir0 + 1]] * fr * (1.0 - fz)
                    + psi[[iz0 + 1, ir0]] * (1.0 - fr) * fz
                    + psi[[iz0 + 1, ir0 + 1]] * fr * fz;

                let residual = psi_interp - psi_target;
                if residual.abs() < 1e-10 * psi_range {
                    break;
                }

                // Gradient of ψ (finite difference)
                let dpsi_dr = (psi[[iz0, (ir0 + 1).min(nr_eq - 1)]]
                    - psi[[iz0, ir0.saturating_sub(1)]])
                    / (2.0 * dr);
                let dpsi_dz = (psi[[(iz0 + 1).min(nz_eq - 1), ir0]]
                    - psi[[iz0.saturating_sub(1), ir0]])
                    / (2.0 * dz_eq);

                let grad_sq = dpsi_dr * dpsi_dr + dpsi_dz * dpsi_dz;
                if grad_sq < 1e-30 {
                    break;
                }

                // Move along ∇ψ direction
                let step = residual / grad_sq;
                r_pt -= step * dpsi_dr;
                z_pt -= step * dpsi_dz;

                // Clamp to domain
                r_pt = r_pt.clamp(r_axis[0], r_axis[nr_eq - 1]);
                z_pt = z_pt.clamp(z_axis[0], z_axis[nz_eq - 1]);
            }

            r_grid[[ix, iy]] = r_pt;
            z_grid[[ix, iy]] = z_pt;
            psi_n_grid[[ix, iy]] = psi_n_surfaces[ix];

            // |B| ≈ B_toroidal * R_0 / R (leading order)
            let b_t = b_toroidal * r_ax / r_pt.max(0.1);
            // Poloidal field from ∇ψ
            let ir_f = ((r_pt - r_axis[0]) / dr).clamp(0.0, (nr_eq - 2) as f64);
            let iz_f = ((z_pt - z_axis[0]) / dz_eq).clamp(0.0, (nz_eq - 2) as f64);
            let ir0 = ir_f as usize;
            let iz0 = iz_f as usize;
            let dpsi_dr = (psi[[iz0.min(nz_eq - 1), (ir0 + 1).min(nr_eq - 1)]]
                - psi[[iz0.min(nz_eq - 1), ir0.saturating_sub(1)]])
                / (2.0 * dr);
            let dpsi_dz = (psi[[(iz0 + 1).min(nz_eq - 1), ir0.min(nr_eq - 1)]]
                - psi[[iz0.saturating_sub(1), ir0.min(nr_eq - 1)]])
                / (2.0 * dz_eq);

            let b_r = -dpsi_dz / r_pt.max(0.1);
            let b_z = dpsi_dr / r_pt.max(0.1);
            let b_p = (b_r * b_r + b_z * b_z).sqrt();
            let b_total = (b_t * b_t + b_p * b_p).sqrt();
            b_mag[[ix, iy]] = b_total;

            // Metric tensors (flux coordinates)
            // g^{xx} = |∇ψ|² / (R²B_p²) (radial)
            let grad_psi_sq = dpsi_dr * dpsi_dr + dpsi_dz * dpsi_dz;
            g_xx[[ix, iy]] = grad_psi_sq / (r_pt * r_pt * b_p * b_p + 1e-30);
            // g^{yy} = B_p² (poloidal arc length)
            g_yy[[ix, iy]] = b_p * b_p;
            // g^{zz} = 1/R² (toroidal)
            g_zz[[ix, iy]] = 1.0 / (r_pt * r_pt);
            // g^{xy} ≈ 0 for orthogonal flux coordinates
            g_xy[[ix, iy]] = 0.0;
            // Jacobian J = R / B_p
            jacobian_grid[[ix, iy]] = r_pt / b_p.max(1e-20);
        }

        // Safety factor: q ≈ (r B_tor) / (R B_pol) averaged over poloidal angle
        let mut q_sum = 0.0;
        for iy in 0..ny {
            let r_pt = r_grid[[ix, iy]];
            let b_pol = (b_mag[[ix, iy]] * b_mag[[ix, iy]]
                - (b_toroidal * r_ax / r_pt.max(0.1)).powi(2))
            .max(0.0)
            .sqrt()
            .max(1e-10);
            q_sum += b_toroidal * r_ax / (r_pt.max(0.1) * b_pol);
        }
        q_profile[ix] = q_sum / ny as f64;
    }

    Ok(BoutGrid {
        nx,
        ny,
        r_grid,
        z_grid,
        psi_n: psi_n_grid,
        b_mag,
        g_xx,
        g_yy,
        g_zz,
        g_xy,
        jacobian: jacobian_grid,
        q_profile,
        b_toroidal,
    })
}

/// Export BOUT++ grid to text format.
///
/// Generates a human-readable text file with all metric data
/// that can be converted to NetCDF for BOUT++ input.
pub fn export_bout_grid_text(grid: &BoutGrid) -> FusionResult<String> {
    let mut out = String::new();
    out.push_str("# BOUT++ grid file generated by SCPN-Fusion-Core\n");
    out.push_str(&format!("nx={}\n", grid.nx));
    out.push_str(&format!("ny={}\n", grid.ny));
    out.push_str(&format!("b_toroidal={:.16e}\n", grid.b_toroidal));

    out.push_str("\n# q profile\n");
    for (i, q) in grid.q_profile.iter().enumerate() {
        out.push_str(&format!("q[{}]={:.16e}\n", i, q));
    }

    out.push_str("\n# Grid data: ix iy R Z psi_n |B| g_xx g_yy g_zz g_xy J\n");
    for ix in 0..grid.nx {
        for iy in 0..grid.ny {
            out.push_str(&format!(
                "{} {} {:.10e} {:.10e} {:.10e} {:.10e} {:.10e} {:.10e} {:.10e} {:.10e} {:.10e}\n",
                ix,
                iy,
                grid.r_grid[[ix, iy]],
                grid.z_grid[[ix, iy]],
                grid.psi_n[[ix, iy]],
                grid.b_mag[[ix, iy]],
                grid.g_xx[[ix, iy]],
                grid.g_yy[[ix, iy]],
                grid.g_zz[[ix, iy]],
                grid.g_xy[[ix, iy]],
                grid.jacobian[[ix, iy]],
            ));
        }
    }
    Ok(out)
}

/// Parsed BOUT++ stability result (growth rate + mode structure).
#[derive(Debug, Clone)]
pub struct BoutStabilityResult {
    /// Toroidal mode number n.
    pub n_toroidal: i32,
    /// Growth rate γ [1/s]. Positive = unstable.
    pub growth_rate: f64,
    /// Real frequency ω [rad/s].
    pub real_frequency: f64,
    /// Radial mode structure amplitude [nx].
    pub mode_amplitude: Vec<f64>,
}

/// Parse BOUT++ stability output (text format).
///
/// Expected format:
/// ```text
/// n=<toroidal_mode_number>
/// gamma=<growth_rate>
/// omega=<real_frequency>
/// amplitude=<val0>,<val1>,...
/// ```
pub fn parse_bout_stability(text: &str) -> FusionResult<BoutStabilityResult> {
    let mut n_tor: Option<i32> = None;
    let mut gamma: Option<f64> = None;
    let mut omega: Option<f64> = None;
    let mut amplitude: Option<Vec<f64>> = None;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("n=") {
            n_tor = Some(rest.trim().parse::<i32>().map_err(|e| {
                FusionError::PhysicsViolation(format!("BOUT++ parse n: {e}"))
            })?);
        } else if let Some(rest) = line.strip_prefix("gamma=") {
            gamma = Some(rest.trim().parse::<f64>().map_err(|e| {
                FusionError::PhysicsViolation(format!("BOUT++ parse gamma: {e}"))
            })?);
        } else if let Some(rest) = line.strip_prefix("omega=") {
            omega = Some(rest.trim().parse::<f64>().map_err(|e| {
                FusionError::PhysicsViolation(format!("BOUT++ parse omega: {e}"))
            })?);
        } else if let Some(rest) = line.strip_prefix("amplitude=") {
            let vals: Result<Vec<f64>, _> = rest.split(',').map(|s| s.trim().parse::<f64>()).collect();
            amplitude = Some(vals.map_err(|e| {
                FusionError::PhysicsViolation(format!("BOUT++ parse amplitude: {e}"))
            })?);
        }
    }

    Ok(BoutStabilityResult {
        n_toroidal: n_tor.ok_or_else(|| {
            FusionError::PhysicsViolation("Missing BOUT++ field: n".into())
        })?,
        growth_rate: gamma.ok_or_else(|| {
            FusionError::PhysicsViolation("Missing BOUT++ field: gamma".into())
        })?,
        real_frequency: omega.ok_or_else(|| {
            FusionError::PhysicsViolation("Missing BOUT++ field: omega".into())
        })?,
        mode_amplitude: amplitude.ok_or_else(|| {
            FusionError::PhysicsViolation("Missing BOUT++ field: amplitude".into())
        })?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_equilibrium() -> (Array2<f64>, Vec<f64>, Vec<f64>, f64, f64) {
        // Create a simple Solov'ev-like ψ on a 33×33 grid
        let nr = 33;
        let nz = 33;
        let r_min = 4.0;
        let r_max = 8.4;
        let z_min = -4.0;
        let z_max = 4.0;
        let r_axis: Vec<f64> = (0..nr)
            .map(|i| r_min + (r_max - r_min) * i as f64 / (nr - 1) as f64)
            .collect();
        let z_axis: Vec<f64> = (0..nz)
            .map(|i| z_min + (z_max - z_min) * i as f64 / (nz - 1) as f64)
            .collect();

        let r0 = 6.2;
        let a = 2.0;
        let kappa = 1.7;

        let mut psi = Array2::zeros((nz, nr));
        let mut psi_max = f64::NEG_INFINITY;
        for iz in 0..nz {
            for ir in 0..nr {
                let r = r_axis[ir];
                let z = z_axis[iz];
                let x = (r - r0) / a;
                let y = z / (kappa * a);
                let rho_sq = x * x + y * y;
                psi[[iz, ir]] = (1.0 - rho_sq).max(0.0);
                if psi[[iz, ir]] > psi_max {
                    psi_max = psi[[iz, ir]];
                }
            }
        }

        (psi, r_axis, z_axis, psi_max, 0.0)
    }

    #[test]
    fn test_bout_grid_generation() {
        let (psi, r_axis, z_axis, psi_ax, psi_bnd) = mock_equilibrium();
        let config = BoutGridConfig {
            nx: 8,
            ny: 16,
            nz: 8,
            psi_inner: 0.15,
            psi_outer: 0.85,
        };

        let grid = generate_bout_grid(&psi, &r_axis, &z_axis, psi_ax, psi_bnd, 5.3, &config)
            .expect("BOUT++ grid generation should succeed");

        assert_eq!(grid.nx, 8);
        assert_eq!(grid.ny, 16);
        assert_eq!(grid.q_profile.len(), 8);
        assert!(grid.b_mag.iter().all(|v| v.is_finite() && *v > 0.0));
        assert!(grid.jacobian.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_bout_grid_export_text() {
        let (psi, r_axis, z_axis, psi_ax, psi_bnd) = mock_equilibrium();
        let config = BoutGridConfig {
            nx: 4,
            ny: 8,
            nz: 4,
            psi_inner: 0.2,
            psi_outer: 0.8,
        };
        let grid = generate_bout_grid(&psi, &r_axis, &z_axis, psi_ax, psi_bnd, 5.3, &config)
            .unwrap();
        let text = export_bout_grid_text(&grid).expect("export should succeed");
        assert!(text.contains("nx=4"));
        assert!(text.contains("ny=8"));
        assert!(text.contains("q[0]="));
    }

    #[test]
    fn test_bout_stability_parse() {
        let text = "\
# BOUT++ stability output
n=1
gamma=1.5e4
omega=-2.3e3
amplitude=0.1,0.3,0.8,0.5,0.2
";
        let result = parse_bout_stability(text).expect("parse should succeed");
        assert_eq!(result.n_toroidal, 1);
        assert!((result.growth_rate - 1.5e4).abs() < 1.0);
        assert!((result.real_frequency - (-2.3e3)).abs() < 1.0);
        assert_eq!(result.mode_amplitude.len(), 5);
    }

    #[test]
    fn test_bout_stability_parse_missing_field() {
        let text = "n=1\ngamma=1.5e4\n";
        assert!(parse_bout_stability(text).is_err());
    }

    #[test]
    fn test_bout_grid_rejects_invalid_config() {
        let cfg = BoutGridConfig {
            nx: 2,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());

        let cfg2 = BoutGridConfig {
            psi_inner: 0.9,
            psi_outer: 0.1,
            ..Default::default()
        };
        assert!(cfg2.validate().is_err());
    }

    #[test]
    fn test_bout_grid_rejects_bad_equilibrium() {
        let psi = Array2::zeros((4, 4));
        let r = vec![4.0, 5.0, 6.0, 7.0];
        let z = vec![-2.0, -1.0, 1.0, 2.0];
        let config = BoutGridConfig::default();

        // psi_axis ≈ psi_boundary → error
        assert!(generate_bout_grid(&psi, &r, &z, 0.0, 0.0, 5.0, &config).is_err());

        // Bad b_toroidal
        assert!(generate_bout_grid(&psi, &r, &z, 1.0, 0.0, 0.0, &config).is_err());
    }
}
