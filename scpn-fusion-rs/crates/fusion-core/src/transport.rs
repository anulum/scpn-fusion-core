//! 1.5D radial transport solver with Bohm/Gyro-Bohm turbulence model.
//!
//! Port of `integrated_transport_solver.py` lines 12-162.
//! Solves heat and particle diffusion on radial grid ρ ∈ [0, 1].

use crate::pedestal::{PedestalConfig, PedestalModel};
use fusion_math::tridiag::thomas_solve;
use fusion_types::error::{FusionError, FusionResult};
use fusion_types::state::RadialProfiles;
use ndarray::Array1;

/// Number of radial grid points. Python line 21.
const TRANSPORT_NR: usize = 50;

/// Default base thermal diffusivity [m²/s] when neoclassical is not configured.
const CHI_BASE_DEFAULT: f64 = 0.5;

/// Floor for neoclassical chi [m²/s] to prevent unphysical values.
const CHI_NC_FLOOR: f64 = 0.01;

/// Boltzmann constant in J/keV for neoclassical calculations.
const NC_BOLTZMANN_J_PER_KEV: f64 = 1.602_176_634e-16;
/// Elementary charge [C].
const NC_ELEM_CHARGE: f64 = 1.602_176_634e-19;
/// Proton mass [kg].
const NC_PROTON_MASS: f64 = 1.672_621_923_69e-27;

/// Turbulent transport multiplier. Python line 80.
const CHI_TURB: f64 = 5.0;

/// Impurity diffusion coefficient [m²/s]. Python line 54.
const D_IMPURITY: f64 = 1.0;

/// Critical temperature gradient for turbulence onset [keV/m]. Python line 74.
const CRIT_GRADIENT: f64 = 2.0;

/// Minimum transport multiplier inside the pedestal transport barrier.
const HMODE_CHI_MIN_FACTOR: f64 = 0.08;

/// H-mode power threshold [MW]. Python line 83.
const HMODE_POWER_THRESHOLD: f64 = 30.0;

/// Edge boundary temperature [keV]. Python line 125.
const EDGE_TEMPERATURE: f64 = 0.1;
/// Stability cap for explicit-euler temperature updates [keV].
const MAX_TEMPERATURE: f64 = 100.0;

/// Radiation cooling coefficient. Python line 105.
const COOLING_FACTOR: f64 = 5.0;

/// Gaussian heating profile width. Python line 100.
const HEATING_WIDTH: f64 = 0.1;

/// Cap for low-order toroidal-mode coupling multiplier.
const TOROIDAL_COUPLING_MAX_FACTOR: f64 = 3.0;

/// Parameters for the Chang-Hinton (1982) neoclassical transport model.
#[derive(Debug, Clone, PartialEq)]
pub struct NeoclassicalParams {
    /// Tokamak major radius R₀ [m].
    pub r_major: f64,
    /// Minor radius a [m].
    pub a_minor: f64,
    /// Toroidal magnetic field B₀ [T].
    pub b_toroidal: f64,
    /// Ion mass number (e.g. 2 for deuterium).
    pub a_ion: f64,
    /// Effective charge Z_eff.
    pub z_eff: f64,
    /// Safety factor profile q(ρ) at each radial grid point.
    pub q_profile: Array1<f64>,
}

impl Default for NeoclassicalParams {
    fn default() -> Self {
        Self {
            r_major: 6.2,
            a_minor: 2.0,
            b_toroidal: 5.3,
            a_ion: 2.0,
            z_eff: 1.5,
            q_profile: Array1::linspace(1.0, 3.0, TRANSPORT_NR),
        }
    }
}

/// Chang-Hinton (1982) neoclassical ion thermal diffusivity [m²/s].
///
/// χ_i^NC = 0.66 (1 + 1.54 α) q² ε^{-3/2} ρ_i² ν_ii / (1 + 0.74 ν*^{2/3})
///
/// where ε = r/R is the local inverse aspect ratio, ρ_i is the ion gyroradius,
/// and ν* = ν_ii qR / (ε^{3/2} v_ti) is the collisionality parameter.
pub fn chang_hinton_chi(
    rho: f64,
    t_i_kev: f64,
    n_e_19: f64,
    q: f64,
    params: &NeoclassicalParams,
) -> f64 {
    if rho <= 0.0 || rho > 1.0 || t_i_kev <= 0.0 || n_e_19 <= 0.0 || q <= 0.0 {
        return CHI_NC_FLOOR;
    }
    let epsilon = (rho * params.a_minor) / params.r_major;
    if epsilon <= 1e-6 {
        return CHI_NC_FLOOR;
    }

    let t_i_j = t_i_kev * NC_BOLTZMANN_J_PER_KEV;
    let m_i = params.a_ion * NC_PROTON_MASS;
    let n_e = n_e_19 * 1e19;

    // Ion thermal velocity: v_ti = sqrt(2 T_i / m_i)
    let v_ti = (2.0 * t_i_j / m_i).sqrt();
    if !v_ti.is_finite() || v_ti <= 0.0 {
        return CHI_NC_FLOOR;
    }

    // Ion gyroradius: ρ_i = m_i v_ti / (Z_i e B)
    let rho_i = m_i * v_ti / (NC_ELEM_CHARGE * params.b_toroidal);

    // Ion-ion collision frequency (simplified):
    // ν_ii ≈ n_e Z_eff² e⁴ ln Λ / (12 π^{3/2} ε₀² m_i^{1/2} T_i^{3/2})
    let ln_lambda = 17.0; // typical for fusion plasma
    let eps0 = 8.854_187_812_8e-12;
    let e4 = NC_ELEM_CHARGE.powi(4);
    let nu_ii = n_e * params.z_eff * params.z_eff * e4 * ln_lambda
        / (12.0 * std::f64::consts::PI.powf(1.5) * eps0 * eps0 * m_i.sqrt() * t_i_j.powf(1.5));

    // Collisionality: ν* = ν_ii q R₀ / (ε^{3/2} v_ti)
    let eps32 = epsilon.powf(1.5);
    let nu_star = nu_ii * q * params.r_major / (eps32 * v_ti);

    // α = (R/r) correction for Shafranov shift effects
    let alpha = epsilon; // simplified

    // Chang-Hinton formula
    let chi = 0.66 * (1.0 + 1.54 * alpha) * q * q * rho_i * rho_i * nu_ii
        / (eps32 * (1.0 + 0.74 * nu_star.powf(2.0 / 3.0)));

    if chi.is_finite() && chi > 0.0 {
        chi.max(CHI_NC_FLOOR)
    } else {
        CHI_NC_FLOOR
    }
}

/// Vectorized Chang-Hinton ion thermal diffusivity over radial grid.
pub fn chang_hinton_chi_profile(
    rho: &Array1<f64>,
    t_i_kev: &Array1<f64>,
    n_e_19: &Array1<f64>,
    q: &Array1<f64>,
    params: &NeoclassicalParams,
) -> Array1<f64> {
    if rho.len() != t_i_kev.len() || rho.len() != n_e_19.len() || rho.len() != q.len() {
        return Array1::from_elem(rho.len(), CHI_NC_FLOOR);
    }
    Array1::from_iter(
        (0..rho.len()).map(|i| chang_hinton_chi(rho[i], t_i_kev[i], n_e_19[i], q[i], params)),
    )
}

/// Vectorized gyro-Bohm diffusivity over radial grid.
pub fn gyro_bohm_chi_profile(
    rho: &Array1<f64>,
    t_i_kev: &Array1<f64>,
    b_field: f64,
    mass_amu: f64,
    chi_gb_coeff: f64,
) -> Array1<f64> {
    let n = rho.len();
    if n == 0 || t_i_kev.len() != n {
        return Array1::zeros(n);
    }
    if !b_field.is_finite() || b_field.abs() < 1e-12 || !mass_amu.is_finite() || mass_amu <= 0.0 {
        return Array1::from_elem(n, CHI_NC_FLOOR);
    }
    let b2 = b_field * b_field;
    let mass_term = mass_amu.sqrt();
    let coeff = chi_gb_coeff.abs().max(1e-9);
    Array1::from_iter((0..n).map(|i| {
        let t = t_i_kev[i].max(0.01);
        let r = rho[i].clamp(0.0, 1.0);
        let chi = coeff * (t.powf(1.5) / b2) * (1.0 + r) / mass_term;
        if chi.is_finite() && chi > 0.0 {
            chi.max(CHI_NC_FLOOR)
        } else {
            CHI_NC_FLOOR
        }
    }))
}

/// Sauter bootstrap current density over radial grid.
pub fn sauter_bootstrap_current_profile(
    rho: &Array1<f64>,
    t_e_kev: &Array1<f64>,
    t_i_kev: &Array1<f64>,
    n_e_19: &Array1<f64>,
    q: &Array1<f64>,
    epsilon: &Array1<f64>,
    b_field: f64,
) -> Array1<f64> {
    let n = rho.len();
    if n < 3
        || t_e_kev.len() != n
        || t_i_kev.len() != n
        || n_e_19.len() != n
        || q.len() != n
        || epsilon.len() != n
        || !b_field.is_finite()
        || b_field <= 0.0
    {
        return Array1::zeros(n);
    }

    let e_charge = NC_ELEM_CHARGE;
    let m_e = 9.109_383_70e-31;
    let eps0: f64 = 8.854_187_812e-12;
    let z_eff = 1.5;
    let mut j_bs = Array1::zeros(n);

    for i in 1..(n - 1) {
        let eps = epsilon[i];
        let te = t_e_kev[i];
        let ti = t_i_kev[i];
        let ne19 = n_e_19[i];
        let qi = q[i];
        if eps < 1e-6 || te <= 0.0 || ti <= 0.0 || ne19 <= 0.0 || qi <= 0.0 {
            continue;
        }

        let eps_sq = (eps * eps).clamp(0.0, 0.999_999);
        let trapped_denom = (1.0 - eps_sq).sqrt() * (1.0 + 1.46 * eps.sqrt());
        if trapped_denom <= 1e-12 {
            continue;
        }
        let mut f_t = 1.0 - (1.0 - eps).powi(2) / trapped_denom;
        f_t = f_t.clamp(0.0, 1.0);

        let t_e_j = te * 1e3 * e_charge;
        if !t_e_j.is_finite() || t_e_j <= 0.0 {
            continue;
        }
        let v_te = (2.0 * t_e_j / m_e).sqrt();
        if !v_te.is_finite() || v_te <= 0.0 {
            continue;
        }

        let n_e = ne19 * 1e19;
        let ln_lambda = 17.0;
        let nu_ei = n_e * z_eff * e_charge.powi(4) * ln_lambda
            / (12.0 * std::f64::consts::PI.powf(1.5) * eps0.powi(2) * m_e.sqrt() * t_e_j.powf(1.5));
        if !nu_ei.is_finite() || nu_ei <= 0.0 {
            continue;
        }
        let nu_star_e = nu_ei * qi / (eps.powf(1.5).max(1e-9) * v_te);
        let alpha_31 = 1.0 / (1.0 + 0.36 / z_eff);
        let l31 = f_t * alpha_31
            / (1.0 + alpha_31 * nu_star_e.sqrt() + 0.25 * nu_star_e * (1.0 - f_t).powi(2));
        let mut l32 = f_t * (0.05 + 0.62 * z_eff) / (z_eff * (1.0 + 0.44 * z_eff));
        l32 /= 1.0 + 0.22 * nu_star_e.sqrt() + 0.19 * nu_star_e * (1.0 - f_t);
        let l34 = l31 * ti / te.max(0.01);

        let dr = (rho[i + 1] - rho[i - 1]).abs().max(1e-12);
        let dn_dr = (n_e_19[i + 1] - n_e_19[i - 1]) * 1e19 / dr;
        let dte_dr = (t_e_kev[i + 1] - t_e_kev[i - 1]) * 1e3 * e_charge / dr;
        let dti_dr = (t_i_kev[i + 1] - t_i_kev[i - 1]) * 1e3 * e_charge / dr;

        let b_pol = (b_field * eps / qi.max(0.1)).max(1e-10);
        let p_e = n_e * t_e_j;
        let denom_ti = (ti * 1e3 * e_charge).max(1e-30);
        let j_val = -(p_e / b_pol)
            * (l31 * dn_dr / n_e.max(1e10)
                + l32 * dte_dr / t_e_j.max(1e-30)
                + l34 * dti_dr / denom_ti);
        if j_val.is_finite() {
            j_bs[i] = j_val;
        }
    }
    j_bs
}

/// Build Crank-Nicolson tridiagonal coefficients for 1D diffusion.
///
/// Returns `(a, b, c, d)` where `a/b/c` are sub/main/super diagonals for
/// `fusion_math::tridiag::thomas_solve` and `d` is a zero-initialized RHS
/// template that callers may fill with source terms.
pub fn build_cn_tridiag(
    chi: &Array1<f64>,
    rho: &Array1<f64>,
    dt: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = rho.len();
    let mut a = vec![0.0; n];
    let mut b = vec![1.0; n];
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];
    if n < 3 || chi.len() != n || !dt.is_finite() || dt <= 0.0 {
        return (a, b, c, d);
    }

    let dr = (rho[1] - rho[0]).abs().max(1e-12);
    for i in 1..(n - 1) {
        let r = rho[i].abs().max(1e-6);
        let chi_i = chi[i].max(0.0);
        let chi_ip = 0.5 * (chi_i + chi[i + 1].max(0.0));
        let chi_im = 0.5 * (chi_i + chi[i - 1].max(0.0));
        let r_ip = r + 0.5 * dr;
        let r_im = (r - 0.5 * dr).max(1e-6);
        let coeff_ip = chi_ip * r_ip / (r * dr * dr);
        let coeff_im = chi_im * r_im / (r * dr * dr);

        b[i] = 1.0 + 0.5 * dt * (coeff_ip + coeff_im);
        c[i] = -0.5 * dt * coeff_ip;
        a[i] = -0.5 * dt * coeff_im;
    }
    d[0] = EDGE_TEMPERATURE;
    d[n - 1] = EDGE_TEMPERATURE;
    (a, b, c, d)
}

/// Single transport timestep helper with explicit dt control.
pub fn transport_step(solver: &mut TransportSolver, p_aux_mw: f64, dt: f64) -> FusionResult<()> {
    if !p_aux_mw.is_finite() || p_aux_mw < 0.0 {
        return Err(FusionError::ConfigError(format!(
            "transport step requires finite p_aux_mw >= 0, got {p_aux_mw}"
        )));
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err(FusionError::ConfigError(format!(
            "transport step requires finite dt > 0, got {dt}"
        )));
    }

    let dt_prev = solver.dt;
    solver.dt = dt;
    let step_result = (|| -> FusionResult<()> {
        solver.update_transport_model(p_aux_mw)?;
        let te_before = solver.profiles.te.clone();
        solver.evolve_profiles(p_aux_mw)?;

        // CN smoothing pass using current diffusivity for added robustness.
        let (a, b, c, mut d) = build_cn_tridiag(&solver.chi, &solver.profiles.rho, dt);
        let n = solver.profiles.te.len();
        if n >= 3 {
            for (i, d_i) in d.iter_mut().enumerate().skip(1).take(n - 2) {
                *d_i = solver.profiles.te[i];
            }
            let solved = thomas_solve(&a, &b, &c, &d);
            for i in 1..(n - 1) {
                let val = solved[i];
                solver.profiles.te[i] = if val.is_finite() {
                    val.clamp(EDGE_TEMPERATURE, MAX_TEMPERATURE)
                } else {
                    te_before[i]
                };
                solver.profiles.ti[i] = solver.profiles.te[i];
            }
            solver.profiles.te[n - 1] = EDGE_TEMPERATURE;
            solver.profiles.ti[n - 1] = EDGE_TEMPERATURE;
        }

        if solver.profiles.te.iter().any(|v| !v.is_finite())
            || solver.profiles.ti.iter().any(|v| !v.is_finite())
            || solver.profiles.ne.iter().any(|v| !v.is_finite())
        {
            return Err(FusionError::ConfigError(
                "transport step produced non-finite profile outputs".to_string(),
            ));
        }
        Ok(())
    })();
    solver.dt = dt_prev;
    step_result
}

/// 1.5D radial transport solver.
pub struct TransportSolver {
    pub profiles: RadialProfiles,
    pub chi: Array1<f64>,
    pub dt: f64,
    pub pedestal: PedestalModel,
    pub toroidal_mode_amplitudes: Vec<f64>,
    pub toroidal_coupling_gain: f64,
    /// Optional neoclassical transport model (replaces constant CHI_BASE).
    pub neoclassical: Option<NeoclassicalParams>,
}

impl TransportSolver {
    /// Create a new transport solver with default ITER-like profiles.
    pub fn new() -> Self {
        let rho = Array1::linspace(0.0, 1.0, TRANSPORT_NR);

        // Initial parabolic profiles
        let te = Array1::from_shape_fn(TRANSPORT_NR, |i| {
            let r: f64 = rho[i];
            10.0 * (1.0 - r * r).max(0.0) + EDGE_TEMPERATURE
        });
        let ti = te.clone();
        let ne = Array1::from_shape_fn(TRANSPORT_NR, |i| {
            let r: f64 = rho[i];
            10.0 * (1.0 - r * r).max(0.0) + 0.5
        });
        let n_impurity = Array1::zeros(TRANSPORT_NR);
        let chi = Array1::from_elem(TRANSPORT_NR, CHI_BASE_DEFAULT);
        let pedestal = PedestalModel::new(PedestalConfig {
            beta_p_ped: 0.35,
            rho_s: 2.0e-3,
            r_major: 6.2,
            alpha_crit: 2.5,
            tau_elm: 1.0e-3,
        })
        .expect("default pedestal config must be valid");

        TransportSolver {
            profiles: RadialProfiles {
                rho,
                te,
                ti,
                ne,
                n_impurity,
            },
            chi,
            dt: 0.01,
            pedestal,
            toroidal_mode_amplitudes: vec![0.0; 3],
            toroidal_coupling_gain: 0.0,
            neoclassical: None,
        }
    }

    /// Configure the neoclassical transport model (Chang-Hinton 1982).
    ///
    /// When set, replaces the constant CHI_BASE with q-profile-dependent neoclassical χ_i.
    pub fn set_neoclassical(&mut self, params: NeoclassicalParams) -> FusionResult<()> {
        if !params.r_major.is_finite() || params.r_major <= 0.0 {
            return Err(FusionError::PhysicsViolation(
                "neoclassical r_major must be finite and > 0".into(),
            ));
        }
        if !params.a_minor.is_finite() || params.a_minor <= 0.0 {
            return Err(FusionError::PhysicsViolation(
                "neoclassical a_minor must be finite and > 0".into(),
            ));
        }
        if !params.b_toroidal.is_finite() || params.b_toroidal <= 0.0 {
            return Err(FusionError::PhysicsViolation(
                "neoclassical b_toroidal must be finite and > 0".into(),
            ));
        }
        if !params.a_ion.is_finite() || params.a_ion <= 0.0 {
            return Err(FusionError::PhysicsViolation(
                "neoclassical a_ion must be finite and > 0".into(),
            ));
        }
        if !params.z_eff.is_finite() || params.z_eff < 1.0 {
            return Err(FusionError::PhysicsViolation(
                "neoclassical z_eff must be finite and >= 1".into(),
            ));
        }
        if params.q_profile.len() != self.profiles.rho.len() {
            return Err(FusionError::ConfigError(format!(
                "neoclassical q_profile length {} must match radial grid {}",
                params.q_profile.len(),
                self.profiles.rho.len()
            )));
        }
        if params.q_profile.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(FusionError::PhysicsViolation(
                "neoclassical q_profile values must be finite and > 0".into(),
            ));
        }
        self.neoclassical = Some(params);
        Ok(())
    }

    /// Configure low-order toroidal mode spectrum `n=1..N` for reduced transport closure.
    ///
    /// The solver remains 1.5D; this adds a radial transport multiplier using spectral
    /// energy from low-order `n != 0` modes to mimic toroidal asymmetry effects.
    pub fn set_toroidal_mode_spectrum(
        &mut self,
        amplitudes: &[f64],
        gain: f64,
    ) -> FusionResult<()> {
        if !gain.is_finite() || gain < 0.0 {
            return Err(FusionError::PhysicsViolation(
                "toroidal coupling gain must be finite and >= 0".to_string(),
            ));
        }
        if amplitudes.iter().any(|a| !a.is_finite() || *a < 0.0) {
            return Err(FusionError::PhysicsViolation(
                "toroidal mode amplitudes must be finite and >= 0".to_string(),
            ));
        }
        self.toroidal_mode_amplitudes.clear();
        self.toroidal_mode_amplitudes
            .extend(amplitudes.iter().copied());
        self.toroidal_coupling_gain = gain;
        Ok(())
    }

    fn toroidal_mode_coupling_factor(&self, rho: f64) -> f64 {
        if self.toroidal_coupling_gain <= 0.0 || self.toroidal_mode_amplitudes.is_empty() {
            return 1.0;
        }

        // Weight higher-n modes by n to reflect stronger short-scale asymmetry drive.
        let spectral_rms = self
            .toroidal_mode_amplitudes
            .iter()
            .enumerate()
            .map(|(idx, amp)| {
                let n = (idx + 1) as f64;
                (n * amp).powi(2)
            })
            .sum::<f64>()
            .sqrt();
        if spectral_rms <= 1e-12 {
            return 1.0;
        }

        // Edge-weighted envelope keeps core near baseline while allowing edge asymmetry.
        let envelope = rho.clamp(0.0, 1.0).powi(2);
        (1.0 + self.toroidal_coupling_gain * envelope * spectral_rms)
            .clamp(1.0, TOROIDAL_COUPLING_MAX_FACTOR)
    }

    /// Update thermal diffusivity using Bohm/Gyro-Bohm + EPED-like pedestal.
    ///
    /// χ = χ_base + χ_turb · max(0, |∇T| - threshold)
    /// Pedestal barrier starts at ρ = 1 - Δ_ped and suppresses edge transport in H-mode.
    pub fn update_transport_model(&mut self, p_aux_mw: f64) -> FusionResult<()> {
        if !p_aux_mw.is_finite() || p_aux_mw < 0.0 {
            return Err(FusionError::ConfigError(format!(
                "transport update requires finite p_aux_mw >= 0, got {p_aux_mw}"
            )));
        }
        let n = self.profiles.rho.len();
        if n < 2 {
            return Err(FusionError::ConfigError(
                "transport update requires at least 2 radial points".to_string(),
            ));
        }
        let dr = if n > 1 { 1.0 / (n as f64 - 1.0) } else { 1.0 };
        if !dr.is_finite() || dr <= 0.0 {
            return Err(FusionError::ConfigError(format!(
                "transport update produced invalid dr={dr}"
            )));
        }
        let ped_width = self.pedestal.pedestal_width();
        if !ped_width.is_finite() || ped_width <= 0.0 {
            return Err(FusionError::ConfigError(format!(
                "transport pedestal width must be finite and > 0, got {ped_width}"
            )));
        }
        let barrier_rho = (1.0 - ped_width).clamp(0.75, 0.995);

        for i in 0..n {
            if !self.profiles.te[i].is_finite() || !self.profiles.rho[i].is_finite() {
                return Err(FusionError::ConfigError(format!(
                    "transport profile contains non-finite values at index {}",
                    i
                )));
            }
            // Compute local temperature gradient
            let grad_t = if i == 0 {
                (self.profiles.te[1] - self.profiles.te[0]) / dr
            } else if i == n - 1 {
                (self.profiles.te[n - 1] - self.profiles.te[n - 2]) / dr
            } else {
                (self.profiles.te[i + 1] - self.profiles.te[i - 1]) / (2.0 * dr)
            };
            if !grad_t.is_finite() {
                return Err(FusionError::ConfigError(format!(
                    "transport gradient became non-finite at index {}",
                    i
                )));
            }

            // Bohm/Gyro-Bohm transport
            let excess = (-grad_t - CRIT_GRADIENT).max(0.0);
            let chi_base = if let Some(ref nc) = self.neoclassical {
                let rho_val = self.profiles.rho[i];
                let t_i = self.profiles.ti[i].max(0.01);
                let n_e_19 = self.profiles.ne[i].max(0.01);
                let q = nc.q_profile[i];
                chang_hinton_chi(rho_val, t_i, n_e_19, q, nc)
            } else {
                CHI_BASE_DEFAULT
            };
            self.chi[i] = chi_base + CHI_TURB * excess;

            // Reduced toroidal coupling closure from low-order n!=0 mode spectrum.
            self.chi[i] *= self.toroidal_mode_coupling_factor(self.profiles.rho[i]);

            // EPED-like H-mode pedestal suppression
            if p_aux_mw > HMODE_POWER_THRESHOLD && self.profiles.rho[i] >= barrier_rho {
                let edge_weight =
                    ((self.profiles.rho[i] - barrier_rho) / ped_width.max(1e-5)).clamp(0.0, 1.0);
                let factor = (1.0 - 0.92 * edge_weight).clamp(HMODE_CHI_MIN_FACTOR, 1.0);
                self.chi[i] *= factor;
            }
            if !self.chi[i].is_finite() {
                return Err(FusionError::ConfigError(format!(
                    "transport chi became non-finite at index {}",
                    i
                )));
            }
        }
        Ok(())
    }

    fn compute_pedestal_pressure_gradient(&self) -> FusionResult<f64> {
        let n = self.profiles.rho.len();
        if n < 3 {
            return Err(FusionError::ConfigError(
                "transport pressure-gradient calculation requires at least 3 radial points"
                    .to_string(),
            ));
        }

        let dr = 1.0 / (n as f64 - 1.0);
        let ped_width = self.pedestal.pedestal_width();
        let barrier_rho = (1.0 - ped_width).clamp(0.75, 0.995);
        let pressure =
            &self.profiles.ne * (&self.profiles.te + &self.profiles.ti).mapv(|v| v.max(0.0));

        let mut max_grad: f64 = 0.0;
        for i in 1..n - 1 {
            if self.profiles.rho[i] < barrier_rho {
                continue;
            }
            let grad = (pressure[i + 1] - pressure[i - 1]) / (2.0 * dr);
            max_grad = max_grad.max(grad.abs());
        }
        if !max_grad.is_finite() {
            return Err(FusionError::ConfigError(
                "transport pressure-gradient calculation produced non-finite result".to_string(),
            ));
        }
        Ok(max_grad)
    }

    /// Evolve temperature profiles by one time step (explicit Euler).
    ///
    /// ∂T/∂t = (1/r)∂(r χ ∂T/∂r)/∂r + S_heat - S_rad
    pub fn evolve_profiles(&mut self, p_aux_mw: f64) -> FusionResult<()> {
        if !p_aux_mw.is_finite() || p_aux_mw < 0.0 {
            return Err(FusionError::ConfigError(format!(
                "transport evolve requires finite p_aux_mw >= 0, got {p_aux_mw}"
            )));
        }
        let n = self.profiles.rho.len();
        if n < 2 {
            return Err(FusionError::ConfigError(
                "transport evolve requires at least 2 radial points".to_string(),
            ));
        }
        let dr = if n > 1 { 1.0 / (n as f64 - 1.0) } else { 1.0 };
        let dt = self.dt;
        if !dt.is_finite() || dt <= 0.0 {
            return Err(FusionError::ConfigError(format!(
                "transport evolve requires finite dt > 0, got {dt}"
            )));
        }

        let te_old = self.profiles.te.clone();

        for i in 1..n - 1 {
            let rho_i = self.profiles.rho[i];
            if !rho_i.is_finite() || rho_i <= 0.0 {
                return Err(FusionError::ConfigError(format!(
                    "transport evolve requires finite rho > 0 at interior index {}, got {}",
                    i, rho_i
                )));
            }

            // Diffusion: (1/r)∂(r χ ∂T/∂r)/∂r via central differences
            let flux_plus =
                0.5 * (self.chi[i] + self.chi[i + 1]) * (te_old[i + 1] - te_old[i]) / dr;
            let flux_minus =
                0.5 * (self.chi[i - 1] + self.chi[i]) * (te_old[i] - te_old[i - 1]) / dr;
            let rho_plus = rho_i + 0.5 * dr;
            let rho_minus = rho_i - 0.5 * dr;
            if rho_plus <= 0.0 || rho_minus <= 0.0 {
                return Err(FusionError::ConfigError(format!(
                    "transport evolve requires positive rho +/- dr/2 at index {}",
                    i
                )));
            }
            let div_flux = (rho_plus * flux_plus - rho_minus * flux_minus) / (rho_i * dr);
            if !div_flux.is_finite() {
                return Err(FusionError::ConfigError(format!(
                    "transport evolve produced non-finite div_flux at index {}",
                    i
                )));
            }

            // Heating source (Gaussian centered at axis)
            let s_heat = p_aux_mw * (-self.profiles.rho[i].powi(2) / HEATING_WIDTH).exp();
            if !s_heat.is_finite() {
                return Err(FusionError::ConfigError(format!(
                    "transport evolve produced non-finite heating source at index {}",
                    i
                )));
            }

            // Radiation sink
            let s_rad = COOLING_FACTOR
                * self.profiles.ne[i]
                * self.profiles.n_impurity[i]
                * te_old[i].abs().sqrt();
            if !s_rad.is_finite() {
                return Err(FusionError::ConfigError(format!(
                    "transport evolve produced non-finite radiation sink at index {}",
                    i
                )));
            }

            // Euler step
            let te_new = te_old[i] + dt * (div_flux + s_heat - s_rad);
            if !te_new.is_finite() {
                return Err(FusionError::ConfigError(format!(
                    "transport evolve produced non-finite temperature at index {}",
                    i
                )));
            }
            self.profiles.te[i] = te_new.clamp(EDGE_TEMPERATURE, MAX_TEMPERATURE);
        }

        // Ti tracks Te (simplified)
        for i in 1..n - 1 {
            self.profiles.ti[i] = self.profiles.te[i];
        }

        // Boundary conditions
        self.profiles.te[n - 1] = EDGE_TEMPERATURE;
        self.profiles.ti[n - 1] = EDGE_TEMPERATURE;
        if self.profiles.te.iter().any(|v| !v.is_finite())
            || self.profiles.ti.iter().any(|v| !v.is_finite())
        {
            return Err(FusionError::ConfigError(
                "transport evolve produced non-finite profile outputs".to_string(),
            ));
        }
        Ok(())
    }

    /// Inject impurities (erosion model). Python lines 39-66.
    pub fn inject_impurities(&mut self, erosion_rate: f64) -> FusionResult<()> {
        if !erosion_rate.is_finite() || erosion_rate < 0.0 {
            return Err(FusionError::ConfigError(format!(
                "transport impurity injection requires finite erosion_rate >= 0, got {erosion_rate}"
            )));
        }
        if !self.dt.is_finite() || self.dt <= 0.0 {
            return Err(FusionError::ConfigError(format!(
                "transport impurity injection requires finite dt > 0, got {}",
                self.dt
            )));
        }
        let n = self.profiles.rho.len();
        let dr = if n > 1 { 1.0 / (n as f64 - 1.0) } else { 1.0 };

        // Add impurity source at edge
        if n > 1 {
            self.profiles.n_impurity[n - 1] += erosion_rate * 1e-18 * self.dt;
        }

        // Diffuse inward (explicit Euler on impurity diffusion equation)
        let n_imp_old = self.profiles.n_impurity.clone();
        for i in 1..n - 1 {
            let laplacian = (n_imp_old[i + 1] - 2.0 * n_imp_old[i] + n_imp_old[i - 1]) / (dr * dr);
            self.profiles.n_impurity[i] =
                (n_imp_old[i] + D_IMPURITY * self.dt * laplacian).max(0.0);
        }
        if self.profiles.n_impurity.iter().any(|v| !v.is_finite()) {
            return Err(FusionError::ConfigError(
                "transport impurity injection produced non-finite values".to_string(),
            ));
        }
        Ok(())
    }

    /// Full transport step: update model, evolve, pedestal crash, inject.
    pub fn step(&mut self, p_aux_mw: f64, erosion_rate: f64) -> FusionResult<()> {
        if !p_aux_mw.is_finite() || p_aux_mw < 0.0 {
            return Err(FusionError::ConfigError(format!(
                "transport step requires finite p_aux_mw >= 0, got {p_aux_mw}"
            )));
        }
        if !erosion_rate.is_finite() || erosion_rate < 0.0 {
            return Err(FusionError::ConfigError(format!(
                "transport step requires finite erosion_rate >= 0, got {erosion_rate}"
            )));
        }
        self.pedestal.advance(self.dt);
        self.update_transport_model(p_aux_mw)?;
        self.evolve_profiles(p_aux_mw)?;

        if p_aux_mw > HMODE_POWER_THRESHOLD {
            let grad_p = self.compute_pedestal_pressure_gradient()?;
            self.pedestal.record_gradient(grad_p);
            if self.pedestal.is_elm_triggered(grad_p) {
                self.pedestal.apply_elm_crash(&mut self.profiles);
            }
        }

        // Keep strict edge boundary after any pedestal crash adjustment.
        let n = self.profiles.te.len();
        if n > 0 {
            self.profiles.te[n - 1] = EDGE_TEMPERATURE;
            self.profiles.ti[n - 1] = EDGE_TEMPERATURE;
        }

        self.inject_impurities(erosion_rate)?;
        if self.profiles.te.iter().any(|v| !v.is_finite())
            || self.profiles.ti.iter().any(|v| !v.is_finite())
            || self.profiles.n_impurity.iter().any(|v| !v.is_finite())
            || self.chi.iter().any(|v| !v.is_finite())
        {
            return Err(FusionError::ConfigError(
                "transport step produced non-finite runtime state".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for TransportSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transport_creation() {
        let ts = TransportSolver::new();
        assert_eq!(ts.profiles.rho.len(), 50);
        assert!(ts.profiles.te[0] > ts.profiles.te[49]);
    }

    #[test]
    fn test_transport_step_no_panic() {
        let mut ts = TransportSolver::new();
        ts.step(50.0, 1e14).expect("valid transport step");
        // Should not panic and profiles should remain finite
        assert!(ts.profiles.te.iter().all(|v| v.is_finite()));
        assert!(ts.profiles.ti.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_hmode_barrier() {
        let mut ts = TransportSolver::new();

        // Without H-mode
        ts.update_transport_model(10.0)
            .expect("valid transport-model update");
        let chi_edge_no_hmode = ts.chi[48]; // near ρ=0.96

        // With H-mode
        ts.update_transport_model(50.0)
            .expect("valid transport-model update");
        let chi_edge_hmode = ts.chi[48];

        assert!(
            chi_edge_hmode < chi_edge_no_hmode,
            "H-mode should reduce edge chi: {} vs {}",
            chi_edge_hmode,
            chi_edge_no_hmode
        );
    }

    #[test]
    fn test_toroidal_mode_coupling_disabled_by_default() {
        let mut baseline = TransportSolver::new();
        baseline
            .update_transport_model(20.0)
            .expect("valid transport-model update");
        let baseline_chi = baseline.chi.clone();

        let mut coupled = TransportSolver::new();
        coupled
            .set_toroidal_mode_spectrum(&[0.2, 0.1, 0.05], 0.0)
            .expect("valid spectrum");
        coupled
            .update_transport_model(20.0)
            .expect("valid transport-model update");

        let max_err = baseline_chi
            .iter()
            .zip(coupled.chi.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            max_err < 1e-12,
            "Expected baseline parity when toroidal coupling gain is zero, max_err={max_err}"
        );
    }

    #[test]
    fn test_toroidal_mode_coupling_boosts_edge_more_than_core() {
        let mut baseline = TransportSolver::new();
        baseline
            .update_transport_model(20.0)
            .expect("valid transport-model update");
        let base_edge = baseline.chi[48];
        let base_core = baseline.chi[4];

        let mut coupled = TransportSolver::new();
        coupled
            .set_toroidal_mode_spectrum(&[0.2, 0.1, 0.04], 0.7)
            .expect("valid spectrum");
        coupled
            .update_transport_model(20.0)
            .expect("valid transport-model update");
        let edge_gain = coupled.chi[48] / base_edge.max(1e-12);
        let core_gain = coupled.chi[4] / base_core.max(1e-12);

        assert!(
            edge_gain > 1.0,
            "Expected edge transport gain > 1 from toroidal coupling, got {edge_gain}"
        );
        assert!(
            edge_gain > core_gain,
            "Expected stronger edge gain than core gain: edge={edge_gain}, core={core_gain}"
        );
    }

    #[test]
    fn test_toroidal_mode_coupling_factor_is_clamped() {
        let mut ts = TransportSolver::new();
        ts.set_toroidal_mode_spectrum(&[10.0, 10.0, 10.0], 10.0)
            .expect("valid spectrum");
        let factor = ts.toroidal_mode_coupling_factor(1.0);
        assert!(
            (factor - TOROIDAL_COUPLING_MAX_FACTOR).abs() < 1e-12,
            "Expected clamped factor={}, got {}",
            TOROIDAL_COUPLING_MAX_FACTOR,
            factor
        );
    }

    #[test]
    fn test_toroidal_mode_spectrum_rejects_invalid_inputs() {
        let mut ts = TransportSolver::new();
        for bad_gain in [f64::NAN, -0.1] {
            let err = ts
                .set_toroidal_mode_spectrum(&[0.1, 0.2], bad_gain)
                .expect_err("invalid gain must error");
            match err {
                FusionError::PhysicsViolation(msg) => {
                    assert!(msg.contains("gain"));
                }
                other => panic!("Unexpected error: {other:?}"),
            }
        }
        for bad_amp in [f64::NAN, -0.2] {
            let err = ts
                .set_toroidal_mode_spectrum(&[0.1, bad_amp], 0.3)
                .expect_err("invalid amplitude must error");
            match err {
                FusionError::PhysicsViolation(msg) => {
                    assert!(msg.contains("amplitudes"));
                }
                other => panic!("Unexpected error: {other:?}"),
            }
        }
    }

    #[test]
    fn test_transport_edge_boundary() {
        let mut ts = TransportSolver::new();
        for _ in 0..10 {
            ts.step(50.0, 0.0).expect("valid transport step");
        }
        // Edge should stay at boundary temperature
        let last = ts.profiles.te.len() - 1;
        assert!(
            (ts.profiles.te[last] - EDGE_TEMPERATURE).abs() < 1e-10,
            "Edge temperature should be fixed at {EDGE_TEMPERATURE}"
        );
    }

    #[test]
    fn test_elm_crash_reduces_pedestal_temperature() {
        let mut ts = TransportSolver::new();

        // Build a steep pedestal pressure gradient near the edge.
        for i in 0..ts.profiles.rho.len() {
            if ts.profiles.rho[i] < 0.9 {
                ts.profiles.te[i] = 2.0;
                ts.profiles.ti[i] = 2.0;
                ts.profiles.ne[i] = 6.0;
            } else {
                ts.profiles.te[i] = 9.0;
                ts.profiles.ti[i] = 9.0;
                ts.profiles.ne[i] = 9.5;
            }
        }

        let edge_idx = ts.profiles.rho.len() - 2;
        let te_before = ts.profiles.te[edge_idx];
        ts.step(60.0, 0.0).expect("valid transport step");
        let te_after = ts.profiles.te[edge_idx];

        assert!(
            te_after < te_before,
            "ELM crash should reduce pedestal Te: before={te_before}, after={te_after}"
        );
        assert!(ts.pedestal.last_gradient() > 0.0);
    }

    #[test]
    fn test_transport_rejects_invalid_runtime_inputs() {
        let mut ts = TransportSolver::new();
        let err = ts
            .step(f64::NAN, 0.0)
            .expect_err("non-finite p_aux must fail");
        match err {
            FusionError::ConfigError(msg) => assert!(msg.contains("p_aux")),
            other => panic!("Unexpected error variant: {other:?}"),
        }

        let err = ts
            .step(50.0, -1.0)
            .expect_err("negative erosion_rate must fail");
        match err {
            FusionError::ConfigError(msg) => assert!(msg.contains("erosion_rate")),
            other => panic!("Unexpected error variant: {other:?}"),
        }

        ts.dt = 0.0;
        let err = ts
            .evolve_profiles(20.0)
            .expect_err("non-positive dt must fail");
        match err {
            FusionError::ConfigError(msg) => assert!(msg.contains("dt")),
            other => panic!("Unexpected error variant: {other:?}"),
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Neoclassical transport tests
    // ═══════════════════════════════════════════════════════════════════

    fn iter_neoclassical_params(n: usize) -> NeoclassicalParams {
        let rho = Array1::linspace(0.0, 1.0, n);
        let q_profile = Array1::from_shape_fn(n, |i| {
            let r = rho[i];
            1.0 + 2.0 * r * r // monotonic q-profile q(0)=1, q(1)=3
        });
        NeoclassicalParams {
            r_major: 6.2,
            a_minor: 2.0,
            b_toroidal: 5.3,
            a_ion: 2.0,
            z_eff: 1.5,
            q_profile,
        }
    }

    #[test]
    fn test_neoclassical_chi_positive() {
        let nc = iter_neoclassical_params(50);
        for i in 1..50 {
            let rho = i as f64 / 49.0;
            let chi = chang_hinton_chi(rho, 10.0, 10.0, 1.0 + 2.0 * rho * rho, &nc);
            assert!(chi > 0.0, "chi must be > 0 at rho={rho}, got {chi}");
            assert!(chi.is_finite(), "chi must be finite at rho={rho}");
        }
    }

    #[test]
    fn test_neoclassical_chi_increases_with_q() {
        let nc = iter_neoclassical_params(50);
        let chi_low_q = chang_hinton_chi(0.5, 10.0, 10.0, 1.0, &nc);
        let chi_high_q = chang_hinton_chi(0.5, 10.0, 10.0, 3.0, &nc);
        assert!(
            chi_high_q > chi_low_q,
            "Higher q should give larger chi: q=1→{chi_low_q}, q=3→{chi_high_q}"
        );
    }

    #[test]
    fn test_neoclassical_chi_floor_for_invalid() {
        let nc = iter_neoclassical_params(50);
        // rho=0 should return floor
        let chi = chang_hinton_chi(0.0, 10.0, 10.0, 1.0, &nc);
        assert!((chi - CHI_NC_FLOOR).abs() < 1e-12);
        // negative temperature
        let chi = chang_hinton_chi(0.5, -1.0, 10.0, 1.5, &nc);
        assert!((chi - CHI_NC_FLOOR).abs() < 1e-12);
    }

    #[test]
    fn test_neoclassical_step_no_panic() {
        let mut ts = TransportSolver::new();
        let nc = iter_neoclassical_params(50);
        ts.set_neoclassical(nc).expect("valid neoclassical config");
        ts.step(50.0, 1e14)
            .expect("valid transport step with neoclassical");
        assert!(ts.profiles.te.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_neoclassical_differs_from_constant() {
        let mut ts_const = TransportSolver::new();
        ts_const.update_transport_model(20.0).unwrap();
        let chi_const = ts_const.chi.clone();

        let mut ts_nc = TransportSolver::new();
        let nc = iter_neoclassical_params(50);
        ts_nc.set_neoclassical(nc).unwrap();
        ts_nc.update_transport_model(20.0).unwrap();

        let max_diff = chi_const
            .iter()
            .zip(ts_nc.chi.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff > 0.0,
            "Neoclassical chi should differ from constant base"
        );
    }

    #[test]
    fn test_neoclassical_rejects_invalid_params() {
        let mut ts = TransportSolver::new();
        let mut nc = iter_neoclassical_params(50);
        nc.r_major = -1.0;
        assert!(ts.set_neoclassical(nc).is_err());

        let mut nc2 = iter_neoclassical_params(50);
        nc2.q_profile = Array1::zeros(10); // wrong length
        assert!(ts.set_neoclassical(nc2).is_err());
    }

    #[test]
    fn test_chang_hinton_profile_vectorized_matches_scalar() {
        let n = 50;
        let rho = Array1::linspace(0.02, 1.0, n);
        let t_i = Array1::from_shape_fn(n, |i| {
            let r = rho[i];
            12.0_f64 * (1.0_f64 - r * r).max(0.05_f64)
        });
        let n_e = Array1::from_shape_fn(n, |i| 8.0 * (1.0 - 0.6 * rho[i] * rho[i]) + 0.5);
        let q = Array1::from_shape_fn(n, |i| 1.0 + 2.5 * rho[i] * rho[i]);
        let params = NeoclassicalParams {
            q_profile: q.clone(),
            ..NeoclassicalParams::default()
        };

        let vectorized = chang_hinton_chi_profile(&rho, &t_i, &n_e, &q, &params);
        for i in 0..n {
            let scalar = chang_hinton_chi(rho[i], t_i[i], n_e[i], q[i], &params);
            assert!(
                (vectorized[i] - scalar).abs() < 1e-12,
                "vectorized and scalar mismatch at i={i}: vec={}, scalar={scalar}",
                vectorized[i]
            );
        }
    }

    #[test]
    fn test_gyro_bohm_profile_positive_and_scaling() {
        let rho = Array1::linspace(0.02, 1.0, 32);
        let t_base = Array1::from_elem(32, 4.0);
        let t_hot = Array1::from_elem(32, 8.0);

        let chi_base = gyro_bohm_chi_profile(&rho, &t_base, 5.0, 2.0, 0.1);
        let chi_hot = gyro_bohm_chi_profile(&rho, &t_hot, 5.0, 2.0, 0.1);
        let chi_high_b = gyro_bohm_chi_profile(&rho, &t_hot, 10.0, 2.0, 0.1);

        assert!(chi_base.iter().all(|v| v.is_finite() && *v > 0.0));
        let ratio_t = chi_hot[10] / chi_base[10].max(1e-12);
        let ratio_b = chi_hot[10] / chi_high_b[10].max(1e-12);

        assert!(
            ratio_t > 2.6,
            "Expected T^(3/2) scaling (>2.6 for 4->8 keV), got {ratio_t}"
        );
        assert!(
            ratio_b > 3.5,
            "Expected ~B^-2 scaling (>3.5 for 5T->10T), got {ratio_b}"
        );
    }

    #[test]
    fn test_sauter_bootstrap_profile_is_mostly_positive() {
        let n = 60;
        let rho = Array1::linspace(0.01, 1.0, n);
        let t_i = Array1::from_shape_fn(n, |i| {
            let r = rho[i];
            12.0_f64 * (1.0_f64 - r * r).max(0.05_f64) + 0.5_f64
        });
        let t_e = Array1::from_shape_fn(n, |i| {
            let r = rho[i];
            10.0_f64 * (1.0_f64 - r * r).max(0.05_f64) + 0.3_f64
        });
        let n_e = Array1::from_shape_fn(n, |i| {
            let r = rho[i];
            9.0_f64 * (1.0_f64 - 0.7_f64 * r * r) + 1.0_f64
        });
        let q = Array1::from_shape_fn(n, |i| {
            let r = rho[i];
            1.0_f64 + 2.5_f64 * r * r
        });
        let epsilon = rho.mapv(|r| r * 0.32);

        let j_bs = sauter_bootstrap_current_profile(&rho, &t_e, &t_i, &n_e, &q, &epsilon, 5.3);
        let positive_fraction =
            j_bs.iter().filter(|v| **v > 0.0).count() as f64 / j_bs.len() as f64;
        assert!(
            positive_fraction > 0.75,
            "Expected mostly positive bootstrap current, fraction={positive_fraction:.3}"
        );
    }

    #[test]
    fn test_cn_tridiag_diagonal_dominance() {
        let n = 50;
        let rho = Array1::linspace(0.0, 1.0, n);
        let chi = Array1::from_elem(n, 0.5);
        let (a, b, c, _) = build_cn_tridiag(&chi, &rho, 0.01);

        for i in 1..(n - 1) {
            let dominance = b[i].abs() - (a[i].abs() + c[i].abs());
            assert!(
                dominance > 0.0,
                "Expected strict diagonal dominance at i={i}: b={}, a={}, c={}",
                b[i],
                a[i],
                c[i]
            );
        }
    }

    fn thermal_energy_proxy(ts: &TransportSolver) -> f64 {
        ts.profiles
            .rho
            .iter()
            .zip(ts.profiles.ne.iter())
            .zip(ts.profiles.ti.iter())
            .map(|((r, n_e), t_i)| r.max(0.0) * n_e.max(0.0) * t_i.max(0.0))
            .sum::<f64>()
    }

    #[test]
    fn test_transport_step_conserves_energy_proxy() {
        let mut ts = TransportSolver::new();
        let p_aux = 20.0;
        let dt = 0.01;
        let before = thermal_energy_proxy(&ts);
        transport_step(&mut ts, p_aux, dt).expect("valid transport step");
        let after = thermal_energy_proxy(&ts);

        let delta = (after - before).abs();
        // Transport update is numerically diffusive; allow generous source-scaled bound.
        let bound = p_aux * dt * 5e4;
        assert!(
            delta <= bound,
            "Energy proxy change too large: delta={delta}, bound={bound}"
        );
    }
}
