// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Nonlinear δf Gyrokinetic Solver
//! Nonlinear δf gyrokinetic solver in flux-tube geometry.

use fusion_math::fft::{cfft2, cfft_axis0, cifft2, cifft_axis0};
use ndarray::{s, Array1, Array2, Array3, Array4, Array5, Array6, Axis};
use num_complex::Complex64;
use rand::Rng;
use rand_distr::StandardNormal;
use std::f64::consts::PI;

pub const E_CHARGE: f64 = 1.602176634e-19;
pub const M_PROTON: f64 = 1.67262192369e-27;
pub const M_ELECTRON: f64 = 9.1093837015e-31;

#[derive(Clone, Debug)]
pub struct NonlinearGKConfig {
    pub n_kx: usize,
    pub n_ky: usize,
    pub n_theta: usize,
    pub n_vpar: usize,
    pub n_mu: usize,
    pub n_species: usize,

    pub dt: f64,
    pub n_steps: usize,
    pub save_interval: usize,

    pub lx: f64,
    pub ly: f64,
    pub vpar_max: f64,
    pub mu_max: f64,

    pub dealiasing: String,
    pub hyper_order: usize,
    pub hyper_coeff: f64,
    pub cfl_factor: f64,
    pub cfl_adapt: bool,

    pub collisions: bool,
    pub nu_collision: f64,
    pub collision_model: String,
    pub nonlinear: bool,
    pub kinetic_electrons: bool,
    pub mass_ratio_me_mi: f64,
    pub implicit_electrons: bool,
    pub electromagnetic: bool,
    pub beta_e: f64,

    pub r0: f64,
    pub a: f64,
    pub b0: f64,
    pub q: f64,
    pub s_hat: f64,

    pub r_l_ti: f64,
    pub r_l_te: f64,
    pub r_l_ne: f64,
}

impl Default for NonlinearGKConfig {
    fn default() -> Self {
        Self {
            n_kx: 16,
            n_ky: 16,
            n_theta: 64,
            n_vpar: 16,
            n_mu: 8,
            n_species: 2,
            dt: 0.05,
            n_steps: 5000,
            save_interval: 100,
            lx: 80.0,
            ly: 62.83,
            vpar_max: 3.0,
            mu_max: 9.0,
            dealiasing: "2/3".to_string(),
            hyper_order: 4,
            hyper_coeff: 0.1,
            cfl_factor: 0.5,
            cfl_adapt: true,
            collisions: true,
            nu_collision: 0.01,
            collision_model: "krook".to_string(),
            nonlinear: true,
            kinetic_electrons: false,
            mass_ratio_me_mi: 1.0 / 400.0,
            implicit_electrons: false,
            electromagnetic: false,
            beta_e: 0.01,
            r0: 2.78,
            a: 1.0,
            b0: 2.0,
            q: 1.4,
            s_hat: 0.78,
            r_l_ti: 6.9,
            r_l_te: 6.9,
            r_l_ne: 2.2,
        }
    }
}

#[derive(Clone)]
pub struct NonlinearGKState {
    pub f: Array6<Complex64>,   // (n_species, n_kx, n_ky, n_theta, n_vpar, n_mu)
    pub phi: Array3<Complex64>, // (n_kx, n_ky, n_theta)
    pub time: f64,
    pub a_par: Option<Array3<Complex64>>,
}

#[derive(Clone, Debug)]
pub struct MillerGeometry {
    pub theta: Array1<f64>,
    pub r: Array1<f64>,
    pub z: Array1<f64>,
    pub b_mag: Array1<f64>,
    pub jacobian: Array1<f64>,
    pub g_rr: Array1<f64>,
    pub g_rt: Array1<f64>,
    pub g_tt: Array1<f64>,
    pub kappa_n: Array1<f64>,
    pub kappa_g: Array1<f64>,
    pub b_dot_grad_theta: Array1<f64>,
}

pub fn circular_geometry(
    r0: f64,
    a: f64,
    rho: f64,
    q: f64,
    s_hat: f64,
    b0: f64,
    n_theta: usize,
    n_period: usize,
) -> MillerGeometry {
    let r = rho * a;
    let n_total = n_theta * n_period;
    let theta = Array1::linspace(
        -(n_period as f64) * PI,
        (n_period as f64) * PI * (1.0 - 1.0 / (n_total as f64)),
        n_total,
    );

    let r_s = theta.mapv(|t| r0 + r * t.cos());
    let z_s = theta.mapv(|t| r * t.sin());

    let dr_dt = theta.mapv(|t| -r * t.sin());
    let dz_dt = theta.mapv(|t| r * t.cos());

    let dr_dr_tot = theta.mapv(|t| t.cos());
    let dz_dr_r = theta.mapv(|t| t.sin());

    let jac = Array1::from_shape_fn(n_total, |i| {
        let val = dr_dr_tot[i] * dz_dt[i] - dr_dt[i] * dz_dr_r[i];
        if val.abs() < 1e-30 {
            1e-30
        } else {
            val
        }
    });

    let g_rr = Array1::from_shape_fn(n_total, |i| {
        (dr_dt[i].powi(2) + dz_dt[i].powi(2)) / jac[i].powi(2)
    });
    let g_rt = Array1::from_shape_fn(n_total, |i| {
        -(dr_dr_tot[i] * dr_dt[i] + dz_dr_r[i] * dz_dt[i]) / jac[i].powi(2)
    });
    let g_tt = Array1::from_shape_fn(n_total, |i| {
        (dr_dr_tot[i].powi(2) + dz_dr_r[i].powi(2)) / jac[i].powi(2)
    });

    let b_phi = r_s.mapv(|rv| b0 * r0 / rv);
    let r_max = r.max(1e-6);
    let b_p = Array1::from_shape_fn(n_total, |i| 1.0 / (q * (jac[i].abs() / r_max) + 1e-30));

    let b_mag = Array1::from_shape_fn(n_total, |i| (b_phi[i].powi(2) + b_p[i].powi(2)).sqrt());
    let b_dot_grad_theta = r_s.mapv(|rv| 1.0 / (q * rv));

    let kappa_n = Array1::from_shape_fn(n_total, |i| {
        let t = theta[i];
        -(1.0 / r_s[i]) * (t.cos() + s_hat * t * t.sin())
    });
    let kappa_g = Array1::from_shape_fn(n_total, |i| {
        let t = theta[i];
        -(1.0 / r_s[i]) * (t.sin() - s_hat * t * t.cos())
    });

    MillerGeometry {
        theta,
        r: r_s,
        z: z_s,
        b_mag,
        jacobian: jac,
        g_rr,
        g_rt,
        g_tt,
        kappa_n,
        kappa_g,
        b_dot_grad_theta,
    }
}

#[derive(Clone, Debug)]
pub struct GKSpecies {
    pub mass_amu: f64,
    pub charge_e: f64,
    pub temperature_kev: f64,
    pub density_19: f64,
    pub r_l_t: f64,
    pub r_l_n: f64,
    pub is_adiabatic: bool,
}

impl GKSpecies {
    pub fn mass_kg(&self) -> f64 {
        self.mass_amu * M_PROTON
    }
    pub fn thermal_speed(&self) -> f64 {
        let t_j = self.temperature_kev * 1e3 * E_CHARGE;
        (2.0 * t_j / self.mass_kg()).sqrt()
    }
}

pub fn deuterium_ion(t_kev: f64, n_19: f64, r_l_t: f64, r_l_n: f64) -> GKSpecies {
    GKSpecies {
        mass_amu: 2.0,
        charge_e: 1.0,
        temperature_kev: t_kev,
        density_19: n_19,
        r_l_t,
        r_l_n,
        is_adiabatic: false,
    }
}

pub fn electron(t_kev: f64, n_19: f64, r_l_t: f64, r_l_n: f64, adiabatic: bool) -> GKSpecies {
    GKSpecies {
        mass_amu: M_ELECTRON / M_PROTON,
        charge_e: -1.0,
        temperature_kev: t_kev,
        density_19: n_19,
        r_l_t,
        r_l_n,
        is_adiabatic: adiabatic,
    }
}

pub struct NonlinearGKResult {
    pub chi_i: f64,
    pub chi_e: f64,
    pub chi_i_gb: f64,
    pub q_i_t: Vec<f64>,
    pub q_e_t: Vec<f64>,
    pub phi_rms_t: Vec<f64>,
    pub zonal_rms_t: Vec<f64>,
    pub time: Vec<f64>,
    pub converged: bool,
    pub final_state: Option<NonlinearGKState>,
}

pub struct NonlinearGKSolver {
    pub cfg: NonlinearGKConfig,
    pub kx: Array1<f64>,
    pub ky: Array1<f64>,
    pub kperp2: Array2<f64>,
    pub theta: Array1<f64>,
    pub dtheta: f64,
    pub vpar: Array1<f64>,
    pub dvpar: f64,
    pub mu: Array1<f64>,
    pub dmu: f64,
    pub dealias_mask: Array2<bool>,

    pub ball_phase_fwd: Array2<Complex64>,
    pub ball_phase_bwd: Array2<Complex64>,

    pub geom: MillerGeometry,
    pub b_dot_grad: Array1<f64>,
    pub kappa_n: Array1<f64>,
    pub kappa_g: Array1<f64>,
    pub b_ratio: Array1<f64>,

    pub ion: GKSpecies,
    pub elec: GKSpecies,
    pub c_s: f64,
    pub rho_s: f64,
    pub chi_gb: f64,
    pub rho_ratio: f64,
    pub rho_ratio_e: f64,
    pub vth_ratio_e: f64,

    pub rh_neo_pol: f64,
    pub rh_residual: f64,
    pub rh_tau: f64,
    pub rh_rate: f64,
    pub ky_zero_5d: Array5<bool>,
}

impl NonlinearGKSolver {
    pub fn new(cfg: NonlinearGKConfig) -> Self {
        let mut solver = Self::allocate_empty(cfg);
        solver.setup_grids();
        solver.setup_ballooning();
        solver.setup_geometry();
        solver.setup_species();
        solver
    }

    fn allocate_empty(cfg: NonlinearGKConfig) -> Self {
        Self {
            cfg,
            kx: Array1::zeros(0),
            ky: Array1::zeros(0),
            kperp2: Array2::zeros((0, 0)),
            theta: Array1::zeros(0),
            dtheta: 0.0,
            vpar: Array1::zeros(0),
            dvpar: 0.0,
            mu: Array1::zeros(0),
            dmu: 0.0,
            dealias_mask: Array2::default((0, 0)),
            ball_phase_fwd: Array2::zeros((0, 0)),
            ball_phase_bwd: Array2::zeros((0, 0)),
            geom: circular_geometry(2.78, 1.0, 0.5, 1.4, 0.78, 2.0, 64, 1),
            b_dot_grad: Array1::zeros(0),
            kappa_n: Array1::zeros(0),
            kappa_g: Array1::zeros(0),
            b_ratio: Array1::zeros(0),
            ion: deuterium_ion(2.0, 5.0, 6.9, 2.2),
            elec: electron(2.0, 5.0, 6.9, 2.2, true),
            c_s: 0.0,
            rho_s: 0.0,
            chi_gb: 0.0,
            rho_ratio: 0.0,
            rho_ratio_e: 0.0,
            vth_ratio_e: 0.0,
            rh_neo_pol: 0.0,
            rh_residual: 0.0,
            rh_tau: 0.0,
            rh_rate: 0.0,
            ky_zero_5d: Array5::default((0, 0, 0, 0, 0)),
        }
    }

    fn setup_grids(&mut self) {
        let c = &self.cfg;
        self.kx = Array1::from_shape_fn(c.n_kx, |i| {
            let freq = if i <= c.n_kx / 2 {
                i as f64
            } else {
                i as f64 - c.n_kx as f64
            };
            2.0 * PI * freq / c.lx
        });
        self.ky = Array1::from_shape_fn(c.n_ky, |i| {
            let freq = if i <= c.n_ky / 2 {
                i as f64
            } else {
                i as f64 - c.n_ky as f64
            };
            2.0 * PI * freq / c.ly
        });

        self.kperp2 = Array2::from_shape_fn((c.n_kx, c.n_ky), |(i, j)| {
            self.kx[i].powi(2) + self.ky[j].powi(2)
        });

        self.theta = Array1::linspace(-PI, PI * (1.0 - 1.0 / (c.n_theta as f64)), c.n_theta);
        self.dtheta = if c.n_theta > 1 {
            self.theta[1] - self.theta[0]
        } else {
            1.0
        };

        self.vpar = Array1::linspace(-c.vpar_max, c.vpar_max, c.n_vpar);
        self.dvpar = if c.n_vpar > 1 {
            self.vpar[1] - self.vpar[0]
        } else {
            1.0
        };
        self.mu = Array1::linspace(0.0, c.mu_max, c.n_mu);
        self.dmu = if c.n_mu > 1 {
            self.mu[1] - self.mu[0]
        } else {
            1.0
        };

        self.dealias_mask = Array2::from_elem((c.n_kx, c.n_ky), true);
        if c.dealiasing == "2/3" {
            let kx_max = self
                .kx
                .mapv(|v| v.abs())
                .iter()
                .cloned()
                .fold(0. / 0., f64::max)
                * 2.0
                / 3.0;
            let ky_max = self
                .ky
                .mapv(|v| v.abs())
                .iter()
                .cloned()
                .fold(0. / 0., f64::max)
                * 2.0
                / 3.0;
            for i in 0..c.n_kx {
                for j in 0..c.n_ky {
                    if self.kx[i].abs() > kx_max || self.ky[j].abs() > ky_max {
                        self.dealias_mask[[i, j]] = false;
                    }
                }
            }
        }
    }

    fn setup_ballooning(&mut self) {
        let c = &self.cfg;
        let mut ball_fwd = Array2::zeros((c.n_kx, c.n_ky));
        let mut ball_bwd = Array2::zeros((c.n_kx, c.n_ky));

        for i in 0..c.n_kx {
            let x = (i as f64) * c.lx / (c.n_kx as f64);
            for j in 0..c.n_ky {
                let delta_kx = c.s_hat * self.ky[j];
                let phase = delta_kx * x;
                ball_fwd[[i, j]] = Complex64::new(0.0, phase).exp();
                ball_bwd[[i, j]] = ball_fwd[[i, j]].conj();
            }
        }
        self.ball_phase_fwd = ball_fwd;
        self.ball_phase_bwd = ball_bwd;
    }

    fn apply_kx_shift(
        &self,
        f_slice: &Array4<Complex64>,
        phase: &Array2<Complex64>,
    ) -> Array4<Complex64> {
        let shape = f_slice.raw_dim();
        let (nkx, nky, nvpar, nmu) = (shape[0], shape[1], shape[2], shape[3]);

        let mut f_flat = Array2::zeros((nkx, nky * nvpar * nmu));
        for i in 0..nkx {
            for j in 0..nky {
                for v in 0..nvpar {
                    for m in 0..nmu {
                        let flat_idx = j * (nvpar * nmu) + v * nmu + m;
                        f_flat[[i, flat_idx]] = f_slice[[i, j, v, m]];
                    }
                }
            }
        }

        let mut f_x = cifft_axis0(&f_flat);
        for i in 0..nkx {
            for j in 0..nky {
                let p = phase[[i, j]];
                for v in 0..nvpar {
                    for m in 0..nmu {
                        let flat_idx = j * (nvpar * nmu) + v * nmu + m;
                        f_x[[i, flat_idx]] *= p;
                    }
                }
            }
        }

        let f_shifted = cfft_axis0(&f_x);
        let mut out = Array4::zeros(shape);
        for i in 0..nkx {
            for j in 0..nky {
                for v in 0..nvpar {
                    for m in 0..nmu {
                        let flat_idx = j * (nvpar * nmu) + v * nmu + m;
                        out[[i, j, v, m]] = f_shifted[[i, flat_idx]];
                    }
                }
            }
        }
        out
    }

    fn roll_ballooning(&self, f_s: &Array5<Complex64>, shift: isize) -> Array5<Complex64> {
        let (nkx, nky, ntheta, nvpar, nmu) = (
            self.cfg.n_kx,
            self.cfg.n_ky,
            self.cfg.n_theta,
            self.cfg.n_vpar,
            self.cfg.n_mu,
        );
        let mut rolled = Array5::zeros((nkx, nky, ntheta, nvpar, nmu));

        for t in 0..ntheta {
            let mut src_t = t as isize - shift;
            let mut apply_bwd = false;
            let mut apply_fwd = false;

            if src_t < 0 {
                src_t += ntheta as isize;
                apply_bwd = true;
            } else if src_t >= ntheta as isize {
                src_t -= ntheta as isize;
                apply_fwd = true;
            }
            let src_t = src_t as usize;

            let slice = f_s.slice(s![.., .., src_t, .., ..]).to_owned();
            let shifted_slice = if apply_bwd {
                self.apply_kx_shift(&slice, &self.ball_phase_bwd)
            } else if apply_fwd {
                self.apply_kx_shift(&slice, &self.ball_phase_fwd)
            } else {
                slice
            };

            rolled
                .slice_mut(s![.., .., t, .., ..])
                .assign(&shifted_slice);
        }
        rolled
    }

    fn setup_geometry(&mut self) {
        let c = &self.cfg;
        self.geom = circular_geometry(c.r0, c.a, 0.5, c.q, c.s_hat, c.b0, c.n_theta, 1);
        self.b_dot_grad = self.geom.b_dot_grad_theta.clone();
        self.kappa_n = self.geom.kappa_n.clone();
        self.kappa_g = self.geom.kappa_g.clone();
        let b_mean = self.geom.b_mag.mean().unwrap();
        self.b_ratio = self.geom.b_mag.mapv(|v| v / b_mean);
    }

    fn setup_species(&mut self) {
        let c = &self.cfg;
        self.ion = deuterium_ion(2.0, 5.0, c.r_l_ti, c.r_l_ne);
        self.elec = electron(2.0, 5.0, c.r_l_te, c.r_l_ne, !c.kinetic_electrons);

        let m_i = self.ion.mass_amu * M_PROTON;
        let t_i_j = self.ion.temperature_kev * 1e3 * E_CHARGE;
        self.c_s = (t_i_j / m_i).sqrt();
        self.rho_s = m_i * self.c_s / (E_CHARGE * c.b0);
        self.chi_gb = self.rho_s.powi(2) * self.c_s / c.a;

        self.rho_ratio =
            (2.0 * self.ion.temperature_kev / self.elec.temperature_kev.max(0.01)).sqrt();
        self.rho_ratio_e = (2.0 * c.mass_ratio_me_mi).sqrt();
        self.vth_ratio_e = (1.0 / c.mass_ratio_me_mi.max(1e-6)).sqrt();

        let eps = 0.5 * c.a / c.r0.max(0.01);
        self.rh_neo_pol = 1.6 * c.q.powi(2) / eps.sqrt().max(0.01);
        self.rh_residual = 1.0 / (1.0 + self.rh_neo_pol);
        self.rh_tau = c.q / eps.sqrt().max(0.01);
        self.rh_rate = (1.0 - self.rh_residual) / self.rh_tau;

        self.ky_zero_5d = Array5::from_shape_fn(
            (c.n_kx, c.n_ky, c.n_theta, c.n_vpar, c.n_mu),
            |(_, j, _, _, _)| self.ky[j].abs() < 1e-10,
        );
    }

    pub fn field_solve(&self, f: &Array6<Complex64>) -> Array3<Complex64> {
        let c = &self.cfg;
        let mut n_ion = Array3::zeros((c.n_kx, c.n_ky, c.n_theta));
        let mut n_elec = Array3::zeros((c.n_kx, c.n_ky, c.n_theta));

        let f_ion = f.slice(s![0, .., .., .., .., ..]);
        for v in 0..c.n_vpar {
            for m in 0..c.n_mu {
                n_ion += &f_ion
                    .slice(s![.., .., .., v, m])
                    .mapv(|x| x * self.dvpar * self.dmu);
            }
        }

        let mut phi = Array3::zeros((c.n_kx, c.n_ky, c.n_theta));
        let mut gamma0_i = Array2::zeros((c.n_kx, c.n_ky));
        for i in 0..c.n_kx {
            for j in 0..c.n_ky {
                let b_i = 0.5 * self.kperp2[[i, j]] * self.rho_ratio.powi(2);
                gamma0_i[[i, j]] = 1.0 / (1.0 + b_i);
            }
        }

        if c.kinetic_electrons {
            let f_e = f.slice(s![1, .., .., .., .., ..]);
            for v in 0..c.n_vpar {
                for m in 0..c.n_mu {
                    n_elec += &f_e
                        .slice(s![.., .., .., v, m])
                        .mapv(|x| x * self.dvpar * self.dmu);
                }
            }

            for i in 0..c.n_kx {
                for j in 0..c.n_ky {
                    let b_e = 0.5 * self.kperp2[[i, j]] * self.rho_ratio_e.powi(2);
                    let gamma0_e = 1.0 / (1.0 + b_e);
                    let denom = (1.0 - gamma0_i[[i, j]]) + (1.0 - gamma0_e);
                    let denom = denom.max(1e-10);

                    for t in 0..c.n_theta {
                        let rhs_qn =
                            gamma0_i[[i, j]] * n_ion[[i, j, t]] - gamma0_e * n_elec[[i, j, t]];
                        phi[[i, j, t]] = rhs_qn / denom;
                    }
                }
            }
        } else {
            for i in 0..c.n_kx {
                for j in 0..c.n_ky {
                    let ky_nonzero = if self.ky[j].abs() > 1e-10 { 1.0 } else { 0.0 };
                    let denom = (1.0 - gamma0_i[[i, j]]) + ky_nonzero;
                    let denom = denom.max(1e-10);

                    for t in 0..c.n_theta {
                        phi[[i, j, t]] = gamma0_i[[i, j]] * n_ion[[i, j, t]] / denom;
                    }
                }
            }
        }

        for t in 0..c.n_theta {
            phi[[0, 0, t]] = Complex64::new(0.0, 0.0);
        }
        phi
    }

    pub fn ampere_solve(&self, f: &Array6<Complex64>) -> Array3<Complex64> {
        let c = &self.cfg;
        if !c.electromagnetic {
            return Array3::zeros((c.n_kx, c.n_ky, c.n_theta));
        }

        let mut j_par = Array3::zeros((c.n_kx, c.n_ky, c.n_theta));
        let f_ion = f.slice(s![0, .., .., .., .., ..]);

        for v in 0..c.n_vpar {
            let vp = self.vpar[v];
            for m in 0..c.n_mu {
                j_par += &f_ion
                    .slice(s![.., .., .., v, m])
                    .mapv(|x| x * vp * self.dvpar * self.dmu);
            }
        }

        if c.kinetic_electrons {
            let f_e = f.slice(s![1, .., .., .., .., ..]);
            for v in 0..c.n_vpar {
                let vp = self.vpar[v];
                for m in 0..c.n_mu {
                    let j_e = f_e
                        .slice(s![.., .., .., v, m])
                        .mapv(|x| x * vp * self.dvpar * self.dmu);
                    j_par = j_par - j_e * self.vth_ratio_e;
                }
            }
        }

        let mut a_par = Array3::zeros((c.n_kx, c.n_ky, c.n_theta));
        for i in 0..c.n_kx {
            for j in 0..c.n_ky {
                let kp2 = self.kperp2[[i, j]].max(1e-10);
                for t in 0..c.n_theta {
                    a_par[[i, j, t]] = c.beta_e * j_par[[i, j, t]] / kp2;
                }
            }
        }

        for t in 0..c.n_theta {
            a_par[[0, 0, t]] = Complex64::new(0.0, 0.0);
        }
        a_par
    }

    pub fn exb_bracket(
        &self,
        phi: &Array3<Complex64>,
        f_s: &Array5<Complex64>,
    ) -> Array5<Complex64> {
        let c = &self.cfg;
        let mut dphi_dx = Array3::zeros((c.n_kx, c.n_ky, c.n_theta));
        let mut dphi_dy = Array3::zeros((c.n_kx, c.n_ky, c.n_theta));

        for i in 0..c.n_kx {
            let kx = self.kx[i];
            for j in 0..c.n_ky {
                let ky = self.ky[j];
                for t in 0..c.n_theta {
                    dphi_dx[[i, j, t]] = Complex64::new(0.0, 1.0) * kx * phi[[i, j, t]];
                    dphi_dy[[i, j, t]] = Complex64::new(0.0, 1.0) * ky * phi[[i, j, t]];
                }
            }
        }

        let mut bracket_k = Array5::zeros((c.n_kx, c.n_ky, c.n_theta, c.n_vpar, c.n_mu));

        for t in 0..c.n_theta {
            let dphi_dx_r = cifft2(&dphi_dx.slice(s![.., .., t]).to_owned());
            let dphi_dy_r = cifft2(&dphi_dy.slice(s![.., .., t]).to_owned());

            for v in 0..c.n_vpar {
                for m in 0..c.n_mu {
                    let f_slice = f_s.slice(s![.., .., t, v, m]).to_owned();
                    let mut df_dx = Array2::zeros((c.n_kx, c.n_ky));
                    let mut df_dy = Array2::zeros((c.n_kx, c.n_ky));

                    for i in 0..c.n_kx {
                        let kx = self.kx[i];
                        for j in 0..c.n_ky {
                            let ky = self.ky[j];
                            df_dx[[i, j]] = Complex64::new(0.0, 1.0) * kx * f_slice[[i, j]];
                            df_dy[[i, j]] = Complex64::new(0.0, 1.0) * ky * f_slice[[i, j]];
                        }
                    }

                    let df_dx_r = cifft2(&df_dx);
                    let df_dy_r = cifft2(&df_dy);

                    let mut bracket_r = Array2::zeros((c.n_kx, c.n_ky));
                    for i in 0..c.n_kx {
                        for j in 0..c.n_ky {
                            bracket_r[[i, j]] = dphi_dx_r[[i, j]] * df_dy_r[[i, j]]
                                - dphi_dy_r[[i, j]] * df_dx_r[[i, j]];
                        }
                    }

                    let bracket_slice = cfft2(&bracket_r);
                    for i in 0..c.n_kx {
                        for j in 0..c.n_ky {
                            if self.dealias_mask[[i, j]] {
                                bracket_k[[i, j, t, v, m]] = bracket_slice[[i, j]];
                            }
                        }
                    }
                }
            }
        }
        bracket_k
    }

    pub fn parallel_streaming(&self, f_s: &Array5<Complex64>) -> Array5<Complex64> {
        let c = &self.cfg;
        let h = self.dtheta;

        let f_m2 = self.roll_ballooning(f_s, -2);
        let f_m1 = self.roll_ballooning(f_s, -1);
        let f_p1 = self.roll_ballooning(f_s, 1);
        let f_p2 = self.roll_ballooning(f_s, 2);

        let mut dfdt = Array5::zeros((c.n_kx, c.n_ky, c.n_theta, c.n_vpar, c.n_mu));
        for i in 0..c.n_kx {
            for j in 0..c.n_ky {
                for t in 0..c.n_theta {
                    let bdg = self.b_dot_grad[t];
                    for v in 0..c.n_vpar {
                        let vp = self.vpar[v];
                        for m in 0..c.n_mu {
                            let diff = -f_m2[[i, j, t, v, m]] + 8.0 * f_m1[[i, j, t, v, m]]
                                - 8.0 * f_p1[[i, j, t, v, m]]
                                + f_p2[[i, j, t, v, m]];
                            dfdt[[i, j, t, v, m]] = vp * bdg * diff / (12.0 * h);
                        }
                    }
                }
            }
        }
        dfdt
    }

    pub fn magnetic_drift(&self, f_s: &Array5<Complex64>) -> Array5<Complex64> {
        let c = &self.cfg;
        let mut omega_d = Array5::zeros((c.n_kx, c.n_ky, c.n_theta, c.n_vpar, c.n_mu));

        for t in 0..c.n_theta {
            let kn = self.kappa_n[t];
            let kg = self.kappa_g[t];
            let br = self.b_ratio[t];

            for v in 0..c.n_vpar {
                let vp2 = self.vpar[v].powi(2);
                for m in 0..c.n_mu {
                    let mu_b = self.mu[m] * br;
                    let energy = 0.5 * vp2 + mu_b;

                    let xi_sq = if vp2 + 2.0 * mu_b > 1e-30 {
                        (vp2 / (vp2 + 2.0 * mu_b)).max(0.0)
                    } else {
                        0.0
                    };

                    let drift_fac = 2.0 * energy * (kn * xi_sq + kg * xi_sq.max(0.0).sqrt());

                    for i in 0..c.n_kx {
                        for j in 0..c.n_ky {
                            omega_d[[i, j, t, v, m]] = Complex64::new(0.0, 1.0)
                                * self.ky[j]
                                * drift_fac
                                * f_s[[i, j, t, v, m]];
                        }
                    }
                }
            }
        }
        omega_d
    }

    pub fn collide(&self, f_s: &Array5<Complex64>) -> Array5<Complex64> {
        // Simple Krook for now
        let c = &self.cfg;
        let nu = c.nu_collision;
        let mut coll = Array5::zeros((c.n_kx, c.n_ky, c.n_theta, c.n_vpar, c.n_mu));
        for i in 0..c.n_kx {
            for j in 0..c.n_ky {
                let kp2 = self.kperp2[[i, j]];
                for t in 0..c.n_theta {
                    for v in 0..c.n_vpar {
                        for m in 0..c.n_mu {
                            coll[[i, j, t, v, m]] = -nu * kp2 * f_s[[i, j, t, v, m]];
                        }
                    }
                }
            }
        }
        coll
    }

    pub fn gradient_drive(
        &self,
        phi: &Array3<Complex64>,
        a_par: Option<&Array3<Complex64>>,
    ) -> Array6<Complex64> {
        let c = &self.cfg;
        let mut drive = Array6::zeros((c.n_species, c.n_kx, c.n_ky, c.n_theta, c.n_vpar, c.n_mu));

        for t in 0..c.n_theta {
            for v in 0..c.n_vpar {
                let vp = self.vpar[v];
                let vp2 = vp.powi(2);
                for m in 0..c.n_mu {
                    let mu_val = self.mu[m];
                    let energy = 0.5 * vp2 + mu_val;
                    let fm = (-energy).exp() / PI.powf(1.5);

                    for i in 0..c.n_kx {
                        for j in 0..c.n_ky {
                            let ky = self.ky[j];
                            let phi_val = phi[[i, j, t]];
                            let a_val = if let Some(a) = a_par {
                                a[[i, j, t]]
                            } else {
                                Complex64::new(0.0, 0.0)
                            };

                            let phi_eff = if c.electromagnetic {
                                phi_val - vp * a_val
                            } else {
                                phi_val
                            };

                            // Ion
                            let eta_i = if c.r_l_ne > 0.0 {
                                c.r_l_ti / c.r_l_ne.max(0.1)
                            } else {
                                0.0
                            };
                            let omega_star_i = ky * c.r_l_ne * (1.0 + eta_i * (energy - 1.5));
                            drive[[0, i, j, t, v, m]] =
                                Complex64::new(0.0, -1.0) * omega_star_i * phi_eff * fm;

                            // Electron
                            if c.kinetic_electrons {
                                let eta_e = if c.r_l_ne > 0.0 {
                                    c.r_l_te / c.r_l_ne.max(0.1)
                                } else {
                                    0.0
                                };
                                let omega_star_e = -ky * c.r_l_ne * (1.0 + eta_e * (energy - 1.5));
                                drive[[1, i, j, t, v, m]] =
                                    Complex64::new(0.0, -1.0) * omega_star_e * phi_val * fm;
                            }
                        }
                    }
                }
            }
        }
        drive
    }

    pub fn hyperdiffusion(&self, f_s: &Array5<Complex64>) -> Array5<Complex64> {
        let c = &self.cfg;
        let mut hd = Array5::zeros((c.n_kx, c.n_ky, c.n_theta, c.n_vpar, c.n_mu));
        let p = (c.hyper_order / 2) as i32;

        for i in 0..c.n_kx {
            for j in 0..c.n_ky {
                let kp2 = self.kperp2[[i, j]];
                let factor = -c.hyper_coeff * kp2.powi(p);
                for t in 0..c.n_theta {
                    for v in 0..c.n_vpar {
                        for m in 0..c.n_mu {
                            hd[[i, j, t, v, m]] = factor * f_s[[i, j, t, v, m]];
                        }
                    }
                }
            }
        }
        hd
    }

    pub fn rhs(&self, state: &NonlinearGKState) -> Array6<Complex64> {
        let c = &self.cfg;
        let mut dfdt = Array6::zeros((c.n_species, c.n_kx, c.n_ky, c.n_theta, c.n_vpar, c.n_mu));

        let a_par_ref = state.a_par.as_ref();

        for s in 0..c.n_species {
            if s == 1 && !c.kinetic_electrons {
                continue;
            }
            let is_elec = s == 1 && c.kinetic_electrons;
            let v_scale = if is_elec && !c.implicit_electrons {
                self.vth_ratio_e
            } else {
                1.0
            };
            let charge_sign = if s == 1 { -1.0 } else { 1.0 };

            let f_s = state.f.slice(s![s, .., .., .., .., ..]).to_owned();

            let mut terms = Array5::zeros((c.n_kx, c.n_ky, c.n_theta, c.n_vpar, c.n_mu));

            if c.nonlinear {
                terms = terms - self.exb_bracket(&state.phi, &f_s);
            }

            terms = terms - self.parallel_streaming(&f_s).mapv(|x| x * v_scale);
            terms = terms - self.magnetic_drift(&f_s).mapv(|x| x * charge_sign);

            if c.collisions {
                terms = terms + self.collide(&f_s);
            }

            terms = terms + self.hyperdiffusion(&f_s);

            dfdt.slice_mut(s![s, .., .., .., .., ..]).assign(&terms);
        }

        let drive = self.gradient_drive(&state.phi, a_par_ref);
        dfdt = dfdt + drive;

        for s in 0..c.n_species {
            for i in 0..c.n_kx {
                for j in 0..c.n_ky {
                    for t in 0..c.n_theta {
                        for v in 0..c.n_vpar {
                            for m in 0..c.n_mu {
                                if self.ky_zero_5d[[i, j, t, v, m]] {
                                    dfdt[[s, i, j, t, v, m]] -=
                                        self.rh_rate * state.f[[s, i, j, t, v, m]];
                                }
                            }
                        }
                    }
                }
            }
        }
        dfdt
    }

    pub fn rk4_step(&self, state: &NonlinearGKState, dt: f64) -> NonlinearGKState {
        let f0 = &state.f;
        let t0 = state.time;

        let k1 = self.rhs(state);

        let f2 = f0 + &k1.mapv(|x| x * 0.5 * dt);
        let phi2 = self.field_solve(&f2);
        let state2 = NonlinearGKState {
            f: f2.clone(),
            phi: phi2,
            time: t0 + 0.5 * dt,
            a_par: if self.cfg.electromagnetic {
                Some(self.ampere_solve(&f2))
            } else {
                None
            },
        };
        let k2 = self.rhs(&state2);

        let f3 = f0 + &k2.mapv(|x| x * 0.5 * dt);
        let phi3 = self.field_solve(&f3);
        let state3 = NonlinearGKState {
            f: f3.clone(),
            phi: phi3,
            time: t0 + 0.5 * dt,
            a_par: if self.cfg.electromagnetic {
                Some(self.ampere_solve(&f3))
            } else {
                None
            },
        };
        let k3 = self.rhs(&state3);

        let f4 = f0 + &k3.mapv(|x| x * dt);
        let phi4 = self.field_solve(&f4);
        let state4 = NonlinearGKState {
            f: f4.clone(),
            phi: phi4,
            time: t0 + dt,
            a_par: if self.cfg.electromagnetic {
                Some(self.ampere_solve(&f4))
            } else {
                None
            },
        };
        let k4 = self.rhs(&state4);

        let f_new = f0
            + &(k1 + &k2.mapv(|x| x * 2.0) + &k3.mapv(|x| x * 2.0) + &k4).mapv(|x| x * (dt / 6.0));

        let phi_new = self.field_solve(&f_new);
        let a_par_new = if self.cfg.electromagnetic {
            Some(self.ampere_solve(&f_new))
        } else {
            None
        };

        NonlinearGKState {
            f: f_new,
            phi: phi_new,
            time: t0 + dt,
            a_par: a_par_new,
        }
    }

    pub fn cfl_dt(&self, state: &NonlinearGKState) -> f64 {
        let c = &self.cfg;
        if !c.cfl_adapt {
            return c.dt;
        }

        let phi_max = state
            .phi
            .mapv(|x| x.norm())
            .iter()
            .cloned()
            .fold(0. / 0., f64::max)
            + 1e-30;
        let kmax = self
            .kx
            .mapv(|x| x.abs())
            .iter()
            .cloned()
            .fold(0. / 0., f64::max)
            .max(
                self.ky
                    .mapv(|x| x.abs())
                    .iter()
                    .cloned()
                    .fold(0. / 0., f64::max),
            );
        let vmax = self
            .vpar
            .mapv(|x| x.abs())
            .iter()
            .cloned()
            .fold(0. / 0., f64::max)
            .max(1.0);

        let v_exb = kmax * phi_max;
        let v_scale = if c.kinetic_electrons && !c.implicit_electrons {
            self.vth_ratio_e
        } else {
            1.0
        };
        let b_dot_max = self
            .b_dot_grad
            .mapv(|x| x.abs())
            .iter()
            .cloned()
            .fold(0. / 0., f64::max);
        let v_par_eff = vmax * v_scale * b_dot_max;

        let kperp2_max = self.kperp2.iter().cloned().fold(0. / 0., f64::max);
        let v_hyper = c.hyper_coeff * kperp2_max.powi((c.hyper_order / 2) as i32);

        let dt_cfl = c.cfl_factor / (v_exb + v_par_eff + v_hyper).max(1e-30);
        dt_cfl.min(c.dt)
    }
}
