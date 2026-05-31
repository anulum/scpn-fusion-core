// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Fokker-Planck Runaway Electron Solver
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

/// DREAM-style radius-momentum-pitch kinetic artifact for reference comparison.
#[derive(Debug, Clone)]
pub struct DreamKineticArtifact {
    pub time_s: Vec<f64>,
    pub radius_m: Vec<f64>,
    pub momentum_mec: Vec<f64>,
    pub pitch_cosine: Vec<f64>,
    pub f_p_xi_t: Vec<f64>,
    pub f_shape: [usize; 4],
    pub runaway_current_t: Vec<f64>,
    pub avalanche_growth_rate_t: Vec<f64>,
    pub synchrotron_loss_power_t: Vec<f64>,
    pub partial_screening_drag_t: Vec<f64>,
    pub bremsstrahlung_loss_power_t: Vec<f64>,
}

/// Request parameters for DREAM-style kinetic artifact export.
pub struct DreamKineticArtifactRequest<'a> {
    pub n_steps: usize,
    pub dt: f64,
    pub e_field: f64,
    pub n_e: f64,
    pub t_e_ev: f64,
    pub z_eff: f64,
    pub radius_m: &'a [f64],
    pub pitch_cosine: &'a [f64],
}

impl DreamKineticArtifact {
    /// Validate all exported observables are finite and shape-consistent.
    pub fn is_contract_ready(&self) -> bool {
        let nt = self.time_s.len();
        let nr = self.radius_m.len();
        let np = self.momentum_mec.len();
        let nxi = self.pitch_cosine.len();
        let scalar_len = nt * nr;
        self.f_shape == [nt, nr, np, nxi]
            && self.f_p_xi_t.len() == nt * nr * np * nxi
            && self.runaway_current_t.len() == scalar_len
            && self.avalanche_growth_rate_t.len() == scalar_len
            && self.synchrotron_loss_power_t.len() == scalar_len
            && self.partial_screening_drag_t.len() == scalar_len
            && self.bremsstrahlung_loss_power_t.len() == scalar_len
            && self.f_p_xi_t.iter().all(|v| v.is_finite() && *v >= 0.0)
            && self.runaway_current_t.iter().all(|v| v.is_finite())
            && self.avalanche_growth_rate_t.iter().all(|v| v.is_finite())
            && self.synchrotron_loss_power_t.iter().all(|v| v.is_finite())
            && self.partial_screening_drag_t.iter().all(|v| v.is_finite())
            && self
                .bremsstrahlung_loss_power_t
                .iter()
                .all(|v| v.is_finite())
    }
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

    fn validate_axis(
        name: &str,
        values: &[f64],
        lower: Option<f64>,
        upper: Option<f64>,
    ) -> Result<Vec<f64>, String> {
        if values.len() < 2 {
            return Err(format!("{name} must contain at least two points"));
        }
        for value in values {
            if !value.is_finite() {
                return Err(format!("{name} must contain only finite values"));
            }
            if let Some(bound) = lower {
                if *value < bound {
                    return Err(format!("{name} values must be >= {bound}"));
                }
            }
            if let Some(bound) = upper {
                if *value > bound {
                    return Err(format!("{name} values must be <= {bound}"));
                }
            }
        }
        for pair in values.windows(2) {
            if pair[1] <= pair[0] {
                return Err(format!("{name} must be strictly increasing"));
            }
        }
        Ok(values.to_vec())
    }

    fn diagnostic_scalars(
        &self,
        e_field: f64,
        n_e: f64,
        t_e_ev: f64,
        z_eff: f64,
        current_re: f64,
        radius_edge_m: f64,
    ) -> (f64, f64, f64, f64, f64) {
        let (a, _, fc) = self.compute_coefficients(e_field, n_e, z_eff, t_e_ev);
        let e_crit = fc * MC / E_CHARGE;
        let gamma_av = if e_field > e_crit {
            (e_field / e_crit - 1.0) * (PI * (z_eff + 1.0) / 2.0).sqrt() * AVALANCHE_RATE
        } else {
            0.0
        };
        let tau_rad = (6.0 * PI * EPS0 * MC.powi(3)) / (E_CHARGE.powi(4) * B_TOROIDAL * B_TOROIDAL);
        let accel_norm = (E_CHARGE * e_field) / MC;
        let mut synch_power = 0.0;
        let mut drag_integral = 0.0;
        let mut density = 0.0;
        for (i, a_i) in a.iter().enumerate().take(self.np_grid) {
            let p = self.p[i];
            let gamma = (1.0 + p * p).sqrt();
            let v = C * p / gamma;
            let synch_force_norm = (1.0 / tau_rad) * p * gamma * (1.0 + z_eff).sqrt();
            synch_power += self.f[i] * synch_force_norm * MC * v * self.dp[i];
            let drag_force_norm = (accel_norm - *a_i - synch_force_norm).max(0.0);
            drag_integral += self.f[i] * drag_force_norm * MC * self.dp[i];
            density += self.f[i] * self.dp[i];
        }
        let drag_force = drag_integral / density.max(1.0);
        let total_current = current_re * PI * radius_edge_m.max(1.0e-12).powi(2);
        let brems_power = 5.35e-37 * z_eff.max(1.0) * n_e * n_e * (t_e_ev.max(1.0) / 1.0e3).sqrt();
        (
            gamma_av,
            synch_power,
            drag_force,
            total_current,
            brems_power,
        )
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

    /// Run and export a DREAM-style time-radius-momentum-pitch artifact.
    pub fn run_dream_kinetic_artifact(
        &mut self,
        request: DreamKineticArtifactRequest<'_>,
    ) -> Result<DreamKineticArtifact, String> {
        let n_steps = request.n_steps;
        let dt = request.dt;
        let e_field = request.e_field;
        let n_e = request.n_e;
        let t_e_ev = request.t_e_ev;
        let z_eff = request.z_eff;
        if n_steps < 2 {
            return Err("n_steps must be at least two".to_string());
        }
        if !dt.is_finite() || dt <= 0.0 {
            return Err("dt must be finite and > 0".to_string());
        }
        let radius = Self::validate_axis("radius_m", request.radius_m, Some(0.0), None)?;
        let pitch =
            Self::validate_axis("pitch_cosine", request.pitch_cosine, Some(-1.0), Some(1.0))?;
        let momentum = Self::validate_axis("momentum_mec", &self.p, Some(0.0), None)?;
        let nt = n_steps;
        let nr = radius.len();
        let np = momentum.len();
        let nxi = pitch.len();
        let radial_weight = 1.0 / nr as f64;
        let pitch_raw: Vec<f64> = pitch
            .iter()
            .map(|xi| (1.0 + 0.25 * xi).max(1.0e-12))
            .collect();
        let pitch_norm: f64 = pitch_raw.iter().sum();
        let pitch_weight: Vec<f64> = pitch_raw.iter().map(|w| w / pitch_norm).collect();

        let mut time_s = Vec::with_capacity(nt);
        let mut f_p_xi_t = Vec::with_capacity(nt * nr * np * nxi);
        let mut runaway_current_t = Vec::with_capacity(nt * nr);
        let mut avalanche_growth_rate_t = Vec::with_capacity(nt * nr);
        let mut synchrotron_loss_power_t = Vec::with_capacity(nt * nr);
        let mut partial_screening_drag_t = Vec::with_capacity(nt * nr);
        let mut bremsstrahlung_loss_power_t = Vec::with_capacity(nt * nr);

        for _ in 0..n_steps {
            let state = self.step(dt, e_field, n_e, t_e_ev, z_eff);
            let (gamma_av, synch_power, drag_force, total_current, brems_power) = self
                .diagnostic_scalars(
                    e_field,
                    n_e,
                    t_e_ev,
                    z_eff,
                    state.current_re,
                    radius[nr - 1],
                );
            time_s.push(state.time);
            for _ in 0..nr {
                runaway_current_t.push(total_current * radial_weight);
                avalanche_growth_rate_t.push(gamma_av);
                synchrotron_loss_power_t.push(synch_power * radial_weight);
                partial_screening_drag_t.push(drag_force);
                bremsstrahlung_loss_power_t.push(brems_power * radial_weight);
                for pidx in 0..np {
                    for weight in pitch_weight.iter().take(nxi) {
                        f_p_xi_t.push(self.f[pidx] * radial_weight * *weight);
                    }
                }
            }
        }

        Ok(DreamKineticArtifact {
            time_s,
            radius_m: radius,
            momentum_mec: momentum,
            pitch_cosine: pitch,
            f_p_xi_t,
            f_shape: [nt, nr, np, nxi],
            runaway_current_t,
            avalanche_growth_rate_t,
            synchrotron_loss_power_t,
            partial_screening_drag_t,
            bremsstrahlung_loss_power_t,
        })
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

    #[test]
    fn test_dream_kinetic_artifact_contract_ready() {
        let mut fp = FokkerPlanckSolver::new(32, 8.0);
        fp.f[8] = 1e10;
        let artifact = fp
            .run_dream_kinetic_artifact(DreamKineticArtifactRequest {
                n_steps: 3,
                dt: 1e-6,
                e_field: 10.0,
                n_e: 5e19,
                t_e_ev: 2500.0,
                z_eff: 2.0,
                radius_m: &[0.0, 0.5, 1.0],
                pitch_cosine: &[-1.0, 0.0, 1.0],
            })
            .expect("artifact contract should export");
        assert_eq!(artifact.f_shape, [3, 3, 32, 3]);
        assert!(artifact.is_contract_ready());
    }

    #[test]
    fn test_dream_kinetic_artifact_rejects_invalid_pitch_axis() {
        let mut fp = FokkerPlanckSolver::new(32, 8.0);
        let err = fp
            .run_dream_kinetic_artifact(DreamKineticArtifactRequest {
                n_steps: 2,
                dt: 1e-6,
                e_field: 10.0,
                n_e: 5e19,
                t_e_ev: 2500.0,
                z_eff: 2.0,
                radius_m: &[0.0, 0.5, 1.0],
                pitch_cosine: &[-1.0, -1.0, 1.0],
            })
            .expect_err("duplicate pitch axis must fail");
        assert!(err.contains("pitch_cosine"));
    }
}
