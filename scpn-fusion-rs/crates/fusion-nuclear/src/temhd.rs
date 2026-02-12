// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — TEMHD
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! TEMHD (Thermo-Electric MHD) Peltier effect stabiliser.
//!
//! Port of `temhd_peltier.py`.
//! Models implicit 1D heat diffusion with Lorentz-driven convection
//! enhancement in a liquid metal layer (lithium).

use fusion_math::tridiag::thomas_solve;

/// Grid points. Python: N=50.
const N: usize = 50;

/// Liquid metal density [kg/m³]. Python: rho=500.
const RHO: f64 = 500.0;

/// Specific heat [J/(kg·K)]. Python: cp=4200.
const CP: f64 = 4200.0;

/// Thermal conductivity [W/(m·K)]. Python: k_thermal=50.
const K_THERMAL: f64 = 50.0;

/// Seebeck coefficient [V/K]. Python: S=20e-6.
const SEEBECK: f64 = 20e-6;

/// Electrical conductivity [S/m]. Python: sigma=3e6.
const SIGMA: f64 = 3e6;

/// Dynamic viscosity [Pa·s]. Python: viscosity=1e-3.
const VISCOSITY: f64 = 1e-3;

/// Initial / wall temperature [K]. Python: T_wall=300.
const T_WALL: f64 = 300.0;

/// Max Peclet number clip. Python: 200.
const PE_CLIP_MAX: f64 = 200.0;

/// Effective-conductivity Peclet factor. Python: 0.2.
const PE_FACTOR: f64 = 0.2;

/// TEMHD stabiliser state.
pub struct TemhdStabilizer {
    /// Layer thickness [m].
    pub layer_m: f64,
    /// Magnetic field [T].
    pub b0: f64,
    /// Grid spacing.
    dz: f64,
    /// Temperature field [K].
    pub t_field: Vec<f64>,
}

impl TemhdStabilizer {
    /// Create with layer thickness in mm and B-field in T.
    pub fn new(layer_thickness_mm: f64, b_field: f64) -> Self {
        let l = layer_thickness_mm * 1e-3;
        let dz = l / (N - 1) as f64;
        let t_field = vec![T_WALL; N];
        TemhdStabilizer {
            layer_m: l,
            b0: b_field,
            dz,
            t_field,
        }
    }

    /// One implicit time step. Returns (T_surface, T_avg).
    pub fn step(&mut self, heat_flux_mw_m2: f64, dt: f64) -> (f64, f64) {
        let n = N;
        let dz = self.dz;

        // Compute effective conductivity via TEMHD enhancement
        let mut k_eff = vec![K_THERMAL; n];
        for i in 1..n - 1 {
            let grad_t = (self.t_field[i + 1] - self.t_field[i - 1]) / (2.0 * dz);
            let j_te = -SIGMA * SEEBECK * grad_t;
            let f_lorentz = (j_te * self.b0).abs();
            let v_conv = f_lorentz * dz * dz / (VISCOSITY + 1e-20);
            let alpha = K_THERMAL / (RHO * CP);
            let pe = (v_conv * dz / alpha).clamp(0.0, PE_CLIP_MAX);
            k_eff[i] = K_THERMAL * (1.0 + PE_FACTOR * pe);
        }

        // Build tridiagonal system (Crank-Nicolson implicit)
        // Interior points only (i = 1..n-2)
        let interior = n - 2;
        let mut a_sub = vec![0.0; interior];
        let mut b_main = vec![0.0; interior];
        let mut c_sup = vec![0.0; interior];
        let mut d_rhs = vec![0.0; interior];

        for j in 0..interior {
            let i = j + 1;
            let r = k_eff[i] * dt / (RHO * CP * dz * dz);
            b_main[j] = 1.0 + 2.0 * r;
            if j > 0 {
                a_sub[j] = -r;
            }
            if j < interior - 1 {
                c_sup[j] = -r;
            }
            d_rhs[j] = self.t_field[i];
        }

        // Boundary: T[0] = T_wall (Dirichlet) — add to first interior equation RHS
        {
            let r = k_eff[1] * dt / (RHO * CP * dz * dz);
            d_rhs[0] += r * T_WALL;
        }

        // Boundary: Neumann at z=L (heat flux in)
        // dT/dz|_{z=L} = q_in / k  →  T[n-1] = T[n-2] + q_in*dz/k
        let q_in = heat_flux_mw_m2 * 1e6; // W/m²
        {
            let i = n - 2;
            let r = k_eff[i] * dt / (RHO * CP * dz * dz);
            // Ghost node: T[n-1] = T[n-2] + q_in*dz/k_eff[n-1]
            let t_ghost = self.t_field[n - 2] + q_in * dz / k_eff[n - 1];
            d_rhs[interior - 1] += r * t_ghost;
        }

        let x = thomas_solve(&a_sub, &b_main, &c_sup, &d_rhs);

        // Update temperatures
        self.t_field[0] = T_WALL;
        self.t_field[1..(interior + 1)].copy_from_slice(&x[..interior]);
        // Surface (z=L): extrapolate from Neumann BC
        self.t_field[n - 1] = self.t_field[n - 2] + q_in * dz / k_eff[n - 1];

        let t_surf = self.t_field[n - 1];
        let t_avg = self.t_field.iter().sum::<f64>() / n as f64;
        (t_surf, t_avg)
    }

    /// Run multiple steps at given flux. Returns final (T_surface, T_avg).
    pub fn run(&mut self, heat_flux_mw_m2: f64, n_steps: usize, dt: f64) -> (f64, f64) {
        let mut result = (T_WALL, T_WALL);
        for _ in 0..n_steps {
            result = self.step(heat_flux_mw_m2, dt);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temhd_creation() {
        let sim = TemhdStabilizer::new(5.0, 10.0);
        assert_eq!(sim.t_field.len(), N);
        assert!((sim.layer_m - 0.005).abs() < 1e-10);
    }

    #[test]
    fn test_temhd_step_warms() {
        let mut sim = TemhdStabilizer::new(5.0, 10.0);
        let (t_surf, _) = sim.run(10.0, 50, 0.5);
        assert!(
            t_surf > T_WALL,
            "Surface should be warmer than wall: {t_surf}"
        );
    }

    #[test]
    fn test_temhd_zero_flux_stays_cold() {
        let mut sim = TemhdStabilizer::new(5.0, 10.0);
        let (t_surf, t_avg) = sim.run(0.0, 100, 0.5);
        assert!(
            (t_surf - T_WALL).abs() < 1.0,
            "No flux → T≈T_wall: {t_surf}"
        );
        assert!(
            (t_avg - T_WALL).abs() < 1.0,
            "No flux → T_avg≈T_wall: {t_avg}"
        );
    }

    #[test]
    fn test_temhd_higher_flux_hotter() {
        let mut sim_lo = TemhdStabilizer::new(5.0, 10.0);
        let (t_lo, _) = sim_lo.run(10.0, 50, 0.5);
        let mut sim_hi = TemhdStabilizer::new(5.0, 10.0);
        let (t_hi, _) = sim_hi.run(50.0, 50, 0.5);
        assert!(
            t_hi > t_lo,
            "Higher flux should be hotter: {t_hi} vs {t_lo}"
        );
    }
}
