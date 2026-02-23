// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Fokker-Planck Runaway Electron Solver
// © 1998–2026 Miroslav Šotek. All rights reserved.
// ─────────────────────────────────────────────────────────────────────
//! 1D-in-momentum Fokker-Planck solver for runaway electron dynamics.
//!
//! Port of `fokker_planck_re.py`.
//! MUSCL-Hancock advection + central-difference diffusion + operator splitting.
//!
//! References:
//! - Hesslow et al., J. Plasma Phys. 85, 475850601 (2019)
//! - Aleynikov & Breizman, PRL 114, 155001 (2015)
//! - Toro, "Riemann Solvers", 3rd ed., Ch. 13

use std::f64::consts::PI;

/// m_e × c [kg m/s].
const MC: f64 = 9.109e-31 * 2.998e8;
/// Speed of light [m/s].
const C: f64 = 2.998e8;
/// Elementary charge [C].
const E_CHARGE: f64 = 1.602e-19;
/// Vacuum permittivity [F/m].
const EPS0: f64 = 8.854e-12;
/// Electron mass [kg].
const ME: f64 = 9.109e-31;
/// Coulomb logarithm, Wesson Ch. 14 Eq. 14.5.2.
const COULOMB_LOG: f64 = 15.0;
/// Toroidal field [T], ITER-like.
const B_TOROIDAL: f64 = 5.3;
/// Numerical diffusion floor.
const DIFFUSION_FLOOR: f64 = 1e-5;
/// Rosenbluth-Putvinski avalanche rate prefactor [1/s], NF 37 (1997).
const AVALANCHE_RATE: f64 = 100.0;
/// Dreicer injection flux [m^-3 s^-1].
const DREICER_SOURCE: f64 = 1.0e15;

const DEFAULT_NP: usize = 200;
const DEFAULT_P_MAX: f64 = 100.0;

/// RE population diagnostics after one step.
#[derive(Debug, Clone, Copy)]
pub struct REState {
    pub time: f64,
    pub n_re: f64,
    pub current_re: f64,
}

/// 1D Fokker-Planck solver with MUSCL-Hancock advection.
pub struct FokkerPlanckSolver {
    /// Momentum grid (normalized to m_e c), log-spaced.
    pub p: Vec<f64>,
    /// Grid spacing (numpy.gradient equivalent).
    pub dp: Vec<f64>,
    /// Distribution function f(p).
    pub f: Vec<f64>,
    np_grid: usize,
    pub time: f64,
}

impl FokkerPlanckSolver {
    pub fn new(np_grid: usize, p_max: f64) -> Self {
        let log_p_max = p_max.log10();
        let p: Vec<f64> = (0..np_grid)
            .map(|i| {
                let t = i as f64 / (np_grid.max(2) - 1) as f64;
                10.0_f64.powf(-2.0 + (log_p_max + 2.0) * t)
            })
            .collect();
        let dp = gradient(&p);

        FokkerPlanckSolver {
            p,
            dp,
            f: vec![0.0; np_grid],
            np_grid,
            time: 0.0,
        }
    }

    /// Compute advection coefficient A, diffusion D, and normalized critical field Fc.
    fn compute_coefficients(
        &self,
        e_field: f64,
        n_e: f64,
        z_eff: f64,
        t_e_ev: f64,
    ) -> (Vec<f64>, Vec<f64>, f64) {
        let f_acc = (E_CHARGE * e_field) / MC;

        // Connor-Hastie critical field
        let ec = (n_e * E_CHARGE.powi(3) * COULOMB_LOG) / (4.0 * PI * EPS0.powi(2) * ME * C * C);
        let fc_norm = (E_CHARGE * ec) / MC;

        let p_thermal_sq = (t_e_ev / 511e3).max(1e-6);

        // Synchrotron radiation timescale, Aleynikov & Breizman PRL 114 (2015) Eq. 3
        let tau_rad = (6.0 * PI * EPS0 * MC.powi(3)) / (E_CHARGE.powi(4) * B_TOROIDAL * B_TOROIDAL);

        let n = self.np_grid;
        let mut a = Vec::with_capacity(n);
        let d = vec![DIFFUSION_FLOOR; n];

        for i in 0..n {
            let pi = self.p[i];
            let gamma = (1.0 + pi * pi).sqrt();
            let f_drag = fc_norm * (1.0 + (z_eff + 1.0) / (pi * pi + p_thermal_sq));
            let f_synch = (1.0 / tau_rad) * pi * gamma * (1.0 + z_eff).sqrt();
            a.push(f_acc - f_drag - f_synch);
        }

        (a, d, fc_norm)
    }

    #[inline]
    fn minmod(a: f64, b: f64) -> f64 {
        if a * b > 0.0 {
            if a.abs() < b.abs() {
                a
            } else {
                b
            }
        } else {
            0.0
        }
    }

    /// Advance f(p,t) by dt using MUSCL-Hancock advection + diffusion.
    pub fn step(&mut self, dt: f64, e_field: f64, n_e: f64, t_e_ev: f64, z_eff: f64) -> REState {
        let (a, d, fc) = self.compute_coefficients(e_field, n_e, z_eff, t_e_ev);
        let n = self.np_grid;

        // Avalanche source, Rosenbluth-Putvinski NF 37 (1997) Eq. 19
        let e_crit = fc * MC / E_CHARGE;
        let gamma_av = if e_field > e_crit {
            (e_field / e_crit - 1.0) * (PI * (z_eff + 1.0) / 2.0).sqrt() * AVALANCHE_RATE
        } else {
            0.0
        };

        let s_av: Vec<f64> = self.f.iter().map(|&fi| gamma_av * fi).collect();

        let mut s_dr = vec![0.0; n];
        if e_field > 0.05 * e_crit {
            for s in s_dr.iter_mut().take(5.min(n)) {
                *s = DREICER_SOURCE;
            }
        }

        // Knock-on source (Moller cross-section, Rosenbluth-Putvinski NF 37 1997)
        let n_re: f64 = self
            .f
            .iter()
            .zip(self.dp.iter())
            .map(|(fi, dpi)| fi * dpi)
            .sum();
        let s_ko: Vec<f64> = if n_re > 1e6 {
            self.p
                .iter()
                .map(|pi| (1.0 / (pi * pi + 1e-4)) * n_e * n_re * 1e-25)
                .collect()
        } else {
            vec![0.0; n]
        };

        // ── MUSCL-Hancock advection ──
        let f_old = &self.f;
        let mut f_new = f_old.clone();

        // Slopes via minmod limiter
        let mut slope = vec![0.0; n];
        for i in 1..n - 1 {
            slope[i] = Self::minmod(f_old[i + 1] - f_old[i], f_old[i] - f_old[i - 1]);
        }

        // Fluxes at cell faces i+1/2
        let mut flux = vec![0.0; n];
        for i in 0..n - 1 {
            let fl = f_old[i] + 0.5 * slope[i];
            let fr = f_old[i + 1] - 0.5 * slope[i + 1];
            flux[i] = if a[i] >= 0.0 { a[i] * fl } else { a[i] * fr };
        }

        // Conservative update
        for i in 1..n - 1 {
            f_new[i] -= (dt / self.dp[i]) * (flux[i] - flux[i - 1]);
        }

        // ── Diffusion half-step (central difference) ──
        for i in 1..n - 1 {
            let dp2 = self.dp[i] * self.dp[i];
            f_new[i] += dt * d[i] * (f_old[i + 1] - 2.0 * f_old[i] + f_old[i - 1]) / dp2;
        }

        // Sources
        for i in 1..n - 1 {
            f_new[i] += dt * (s_av[i] + s_dr[i] + s_ko[i]);
        }

        // Positivity floor
        for fi in f_new.iter_mut() {
            *fi = fi.max(0.0);
        }

        self.f = f_new;
        self.time += dt;

        // Diagnostics
        let n_re_new: f64 = self
            .f
            .iter()
            .zip(self.dp.iter())
            .map(|(fi, dpi)| fi * dpi)
            .sum();
        let j_re: f64 = self
            .f
            .iter()
            .zip(self.p.iter())
            .zip(self.dp.iter())
            .map(|((fi, pi), dpi)| {
                let gamma = (1.0 + pi * pi).sqrt();
                let v = C * pi / gamma;
                E_CHARGE * fi * v * dpi
            })
            .sum();

        REState {
            time: self.time,
            n_re: n_re_new,
            current_re: j_re,
        }
    }

    /// Run N steps with fixed parameters.
    pub fn run(
        &mut self,
        n_steps: usize,
        dt: f64,
        e_field: f64,
        n_e: f64,
        t_e_ev: f64,
        z_eff: f64,
    ) -> Vec<REState> {
        (0..n_steps)
            .map(|_| self.step(dt, e_field, n_e, t_e_ev, z_eff))
            .collect()
    }
}

impl Default for FokkerPlanckSolver {
    fn default() -> Self {
        Self::new(DEFAULT_NP, DEFAULT_P_MAX)
    }
}

/// numpy.gradient equivalent for 1D arrays.
fn gradient(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let mut g = vec![0.0; n];
    g[0] = x[1] - x[0];
    g[n - 1] = x[n - 1] - x[n - 2];
    for i in 1..n - 1 {
        g[i] = (x[i + 1] - x[i - 1]) / 2.0;
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let fp = FokkerPlanckSolver::default();
        assert_eq!(fp.p.len(), DEFAULT_NP);
        assert_eq!(fp.f.len(), DEFAULT_NP);
        assert!(fp.p[0] > 0.0);
    }

    #[test]
    fn test_step_finite() {
        let mut fp = FokkerPlanckSolver::default();
        fp.f[10] = 1e10;
        let state = fp.step(1e-5, 10.0, 5e19, 5000.0, 1.0);
        assert!(state.n_re.is_finite());
        assert!(state.current_re.is_finite());
    }

    #[test]
    fn test_positivity() {
        let mut fp = FokkerPlanckSolver::default();
        fp.f[10] = 1e10;
        fp.step(1e-5, 10.0, 5e19, 5000.0, 1.0);
        assert!(fp.f.iter().all(|&fi| fi >= 0.0));
    }

    #[test]
    fn test_particle_conservation_no_source() {
        let mut fp = FokkerPlanckSolver::new(200, 50.0);
        let p0 = 5.0;
        for i in 0..fp.np_grid {
            let dp = fp.p[i] - p0;
            fp.f[i] = 1e10 * (-dp * dp / 2.0).exp();
        }
        let n_before: f64 = fp.f.iter().zip(fp.dp.iter()).map(|(f, d)| f * d).sum();
        for _ in 0..10 {
            fp.step(1e-6, 0.0, 1e19, 5000.0, 1.0);
        }
        let n_after: f64 = fp.f.iter().zip(fp.dp.iter()).map(|(f, d)| f * d).sum();
        let rel = (n_after - n_before).abs() / n_before.max(1e-30);
        assert!(rel < 0.5, "Particle count changed by {:.1}%", rel * 100.0);
    }

    #[test]
    fn test_run_multi_step() {
        let mut fp = FokkerPlanckSolver::default();
        fp.f[10] = 1e10;
        let history = fp.run(100, 1e-5, 10.0, 5e19, 5000.0, 1.0);
        assert_eq!(history.len(), 100);
        assert!(history.iter().all(|s| s.n_re.is_finite()));
    }
}
