// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — SPI Fragment Ablation Solver
// © 1998–2026 Miroslav Šotek. All rights reserved.
// ─────────────────────────────────────────────────────────────────────
//! Multi-fragment Shattered Pellet Injection (SPI) ablation solver.
//!
//! Port of `spi_ablation.py`.
//! Lagrangian tracking of N fragments with Parks ablation model.
//! Struct-of-arrays layout for cache efficiency.
//!
//! Reference: Parks, NF 57 (2017) Eq. 8.

use fusion_types::error::{FusionError, FusionResult};
use rand::prelude::*;
use rand_distr::Normal;
use std::f64::consts::PI;

/// Neon atom mass [kg], 20.18 AMU.
const M_NEON: f64 = 20.18 * 1.66e-27;
/// Solid neon density [kg/m^3], CRC Handbook.
const RHO_NEON_SOLID: f64 = 1444.0;
/// Parks ablation prefactor, Parks NF 57 (2017) Eq. 8 in mixed code units.
const PARKS_COEFFICIENT: f64 = 2.0;
/// Major radius [m], ITER-like.
const R_MAJOR: f64 = 6.2;
/// Minor radius [m].
const A_MINOR: f64 = 2.0;
/// Plasma elongation.
const ELONGATION: f64 = 2.0;

/// SPI fragment ablation solver — struct-of-arrays layout.
pub struct SpiAblationSolver {
    pub n_fragments: usize,
    pub radius: Vec<f64>,
    pub mass: Vec<f64>,
    pub pos_x: Vec<f64>,
    pub pos_y: Vec<f64>,
    pub pos_z: Vec<f64>,
    pub vel_x: Vec<f64>,
    pub vel_y: Vec<f64>,
    pub vel_z: Vec<f64>,
    pub active: Vec<bool>,
}

impl SpiAblationSolver {
    pub fn new(
        n_fragments: usize,
        total_mass_kg: f64,
        velocity_mps: f64,
        dispersion: f64,
    ) -> FusionResult<Self> {
        if n_fragments == 0 {
            return Err(FusionError::ConfigError(
                "n_fragments must be > 0".to_string(),
            ));
        }
        if !total_mass_kg.is_finite() || total_mass_kg <= 0.0 {
            return Err(FusionError::ConfigError(
                "total_mass must be finite and > 0".to_string(),
            ));
        }

        let m_frag = total_mass_kg / n_fragments as f64;
        let vol = m_frag / RHO_NEON_SOLID;
        let r_frag = (3.0 * vol / (4.0 * PI)).cbrt();

        let mut rng = StdRng::seed_from_u64(42);
        let dir_noise = Normal::new(0.0, 1.0)
            .map_err(|e| FusionError::ConfigError(e.to_string()))?;
        let vel_noise = Normal::new(1.0, 0.1)
            .map_err(|e| FusionError::ConfigError(e.to_string()))?;
        let pos_noise = Normal::new(0.0, 0.05)
            .map_err(|e| FusionError::ConfigError(e.to_string()))?;

        let mut solver = SpiAblationSolver {
            n_fragments,
            radius: vec![r_frag; n_fragments],
            mass: vec![m_frag; n_fragments],
            pos_x: vec![0.0; n_fragments],
            pos_y: vec![0.0; n_fragments],
            pos_z: vec![0.0; n_fragments],
            vel_x: vec![0.0; n_fragments],
            vel_y: vec![0.0; n_fragments],
            vel_z: vec![0.0; n_fragments],
            active: vec![true; n_fragments],
        };

        // Outboard midplane injector, radial inward
        for i in 0..n_fragments {
            let mut vd = [
                -1.0 + dir_noise.sample(&mut rng) * dispersion,
                dir_noise.sample(&mut rng) * dispersion,
                dir_noise.sample(&mut rng) * dispersion,
            ];
            let norm = (vd[0] * vd[0] + vd[1] * vd[1] + vd[2] * vd[2])
                .sqrt()
                .max(1e-30);
            vd[0] /= norm;
            vd[1] /= norm;
            vd[2] /= norm;

            let v_mag = velocity_mps * vel_noise.sample(&mut rng);
            solver.vel_x[i] = vd[0] * v_mag;
            solver.vel_y[i] = vd[1] * v_mag;
            solver.vel_z[i] = vd[2] * v_mag;

            solver.pos_x[i] = 10.0 + pos_noise.sample(&mut rng);
            solver.pos_y[i] = pos_noise.sample(&mut rng);
            solver.pos_z[i] = pos_noise.sample(&mut rng);
        }

        Ok(solver)
    }

    /// Advance fragments by dt. Returns deposition profile [particles/m^3/s].
    pub fn step(
        &mut self,
        dt: f64,
        plasma_ne: &[f64],
        plasma_te: &[f64],
        r_grid: &[f64],
    ) -> FusionResult<Vec<f64>> {
        let n_grid = r_grid.len();
        let mut deposition = vec![0.0; n_grid];
        let dr = if n_grid > 1 {
            r_grid[1] - r_grid[0]
        } else {
            1.0
        };

        let rho_ne = linspace(0.0, 1.0, plasma_ne.len());
        let rho_te = linspace(0.0, 1.0, plasma_te.len());

        for i in 0..self.n_fragments {
            if !self.active[i] {
                continue;
            }

            self.pos_x[i] += self.vel_x[i] * dt;
            self.pos_y[i] += self.vel_y[i] * dt;
            self.pos_z[i] += self.vel_z[i] * dt;

            let r_loc = (self.pos_x[i].powi(2) + self.pos_y[i].powi(2)).sqrt();
            let z_loc = self.pos_z[i];
            let rho_loc = (((r_loc - R_MAJOR) / A_MINOR).powi(2)
                + (z_loc / ELONGATION).powi(2))
            .sqrt();

            if !(0.0..=1.2).contains(&rho_loc) {
                continue;
            }

            let n_e = interp(rho_loc, &rho_ne, plasma_ne);
            let t_e = interp(rho_loc, &rho_te, plasma_te);

            if t_e < 0.01 {
                continue;
            }

            // Parks ablation, Parks NF 57 (2017) Eq. 8
            let ne_20 = (n_e / 10.0).max(0.0);
            let rp_cm = self.radius[i] * 100.0;
            let dm_dt_g =
                PARKS_COEFFICIENT * ne_20.powf(0.33) * t_e.powf(1.64) * rp_cm.powf(1.33);
            let dm_dt_kg = dm_dt_g / 1000.0;

            let mut delta_m = dm_dt_kg * dt;
            if delta_m > self.mass[i] {
                delta_m = self.mass[i];
                self.active[i] = false;
                self.mass[i] = 0.0;
                self.radius[i] = 0.0;
            } else {
                self.mass[i] -= delta_m;
                self.radius[i] = (3.0 * (self.mass[i] / RHO_NEON_SOLID) / (4.0 * PI)).cbrt();
            }

            let n_particles = delta_m / M_NEON;
            let idx = (rho_loc * (n_grid - 1) as f64).round() as usize;
            if idx < n_grid {
                let rate = n_particles / dt;
                let r_minor = rho_loc * A_MINOR;
                let dv = 4.0 * PI * PI * R_MAJOR * r_minor * (A_MINOR * dr);
                deposition[idx] += rate / dv.max(1.0);
            }
        }

        Ok(deposition)
    }

    pub fn n_active(&self) -> usize {
        self.active.iter().filter(|&&a| a).count()
    }

    pub fn total_mass(&self) -> f64 {
        self.mass.iter().sum()
    }
}

fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![start];
    }
    (0..n)
        .map(|i| start + (end - start) * i as f64 / (n - 1) as f64)
        .collect()
}

fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    if xp.is_empty() {
        return 0.0;
    }
    if x <= xp[0] {
        return fp[0];
    }
    if x >= *xp.last().unwrap() {
        return *fp.last().unwrap();
    }
    let mut lo = 0;
    let mut hi = xp.len() - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xp[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = (x - xp[lo]) / (xp[hi] - xp[lo]).max(1e-30);
    fp[lo] + t * (fp[hi] - fp[lo])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let s = SpiAblationSolver::new(100, 0.01, 200.0, 0.1).unwrap();
        assert_eq!(s.n_fragments, 100);
        assert_eq!(s.n_active(), 100);
        assert!((s.total_mass() - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_step_returns_deposition() {
        let mut s = SpiAblationSolver::new(100, 0.01, 200.0, 0.1).unwrap();
        let ne = vec![5.0; 50];
        let te = vec![10.0; 50];
        let r_grid: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
        let dep = s.step(1e-4, &ne, &te, &r_grid).unwrap();
        assert_eq!(dep.len(), 50);
    }

    #[test]
    fn test_mass_decreases() {
        let mut s = SpiAblationSolver::new(10, 0.01, 200.0, 0.1).unwrap();
        let ne = vec![5.0; 50];
        let te = vec![10.0; 50];
        let r_grid: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
        let m0 = s.total_mass();
        for _ in 0..100 {
            s.step(1e-4, &ne, &te, &r_grid).unwrap();
        }
        assert!(s.total_mass() <= m0);
    }

    #[test]
    fn test_rejects_invalid() {
        assert!(SpiAblationSolver::new(0, 0.01, 200.0, 0.1).is_err());
        assert!(SpiAblationSolver::new(100, -1.0, 200.0, 0.1).is_err());
    }
}
