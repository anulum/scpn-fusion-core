//! 1.5D radial transport solver with Bohm/Gyro-Bohm turbulence model.
//!
//! Port of `integrated_transport_solver.py` lines 12-162.
//! Solves heat and particle diffusion on radial grid ρ ∈ [0, 1].

use crate::pedestal::{PedestalConfig, PedestalModel};
use fusion_types::state::RadialProfiles;
use ndarray::Array1;

/// Number of radial grid points. Python line 21.
const TRANSPORT_NR: usize = 50;

/// Base neoclassical thermal diffusivity [m²/s]. Python line 77.
const CHI_BASE: f64 = 0.5;

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

/// Radiation cooling coefficient. Python line 105.
const COOLING_FACTOR: f64 = 5.0;

/// Gaussian heating profile width. Python line 100.
const HEATING_WIDTH: f64 = 0.1;

/// 1.5D radial transport solver.
pub struct TransportSolver {
    pub profiles: RadialProfiles,
    pub chi: Array1<f64>,
    pub dt: f64,
    pub pedestal: PedestalModel,
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
        let chi = Array1::from_elem(TRANSPORT_NR, CHI_BASE);
        let pedestal = PedestalModel::new(PedestalConfig {
            beta_p_ped: 0.35,
            rho_s: 2.0e-3,
            r_major: 6.2,
            alpha_crit: 2.5,
            tau_elm: 1.0e-3,
        });

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
        }
    }

    /// Update thermal diffusivity using Bohm/Gyro-Bohm + EPED-like pedestal.
    ///
    /// χ = χ_base + χ_turb · max(0, |∇T| - threshold)
    /// Pedestal barrier starts at ρ = 1 - Δ_ped and suppresses edge transport in H-mode.
    pub fn update_transport_model(&mut self, p_aux_mw: f64) {
        let n = self.profiles.rho.len();
        let dr = if n > 1 { 1.0 / (n as f64 - 1.0) } else { 1.0 };
        let ped_width = self.pedestal.pedestal_width();
        let barrier_rho = (1.0 - ped_width).clamp(0.75, 0.995);

        for i in 0..n {
            // Compute local temperature gradient
            let grad_t = if i == 0 {
                (self.profiles.te[1] - self.profiles.te[0]) / dr
            } else if i == n - 1 {
                (self.profiles.te[n - 1] - self.profiles.te[n - 2]) / dr
            } else {
                (self.profiles.te[i + 1] - self.profiles.te[i - 1]) / (2.0 * dr)
            };

            // Bohm/Gyro-Bohm transport
            let excess = (-grad_t - CRIT_GRADIENT).max(0.0);
            self.chi[i] = CHI_BASE + CHI_TURB * excess;

            // EPED-like H-mode pedestal suppression
            if p_aux_mw > HMODE_POWER_THRESHOLD && self.profiles.rho[i] >= barrier_rho {
                let edge_weight =
                    ((self.profiles.rho[i] - barrier_rho) / ped_width.max(1e-5)).clamp(0.0, 1.0);
                let factor = (1.0 - 0.92 * edge_weight).clamp(HMODE_CHI_MIN_FACTOR, 1.0);
                self.chi[i] *= factor;
            }
        }
    }

    fn compute_pedestal_pressure_gradient(&self) -> f64 {
        let n = self.profiles.rho.len();
        if n < 3 {
            return 0.0;
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
        max_grad
    }

    /// Evolve temperature profiles by one time step (explicit Euler).
    ///
    /// ∂T/∂t = (1/r)∂(r χ ∂T/∂r)/∂r + S_heat - S_rad
    pub fn evolve_profiles(&mut self, p_aux_mw: f64) {
        let n = self.profiles.rho.len();
        let dr = if n > 1 { 1.0 / (n as f64 - 1.0) } else { 1.0 };
        let dt = self.dt;

        let te_old = self.profiles.te.clone();

        for i in 1..n - 1 {
            let rho_i = self.profiles.rho[i].max(1e-6);

            // Diffusion: (1/r)∂(r χ ∂T/∂r)/∂r via central differences
            let flux_plus =
                0.5 * (self.chi[i] + self.chi[i + 1]) * (te_old[i + 1] - te_old[i]) / dr;
            let flux_minus =
                0.5 * (self.chi[i - 1] + self.chi[i]) * (te_old[i] - te_old[i - 1]) / dr;
            let rho_plus = (rho_i + 0.5 * dr).max(1e-6);
            let rho_minus = (rho_i - 0.5 * dr).max(1e-6);
            let div_flux = (rho_plus * flux_plus - rho_minus * flux_minus) / (rho_i * dr);

            // Heating source (Gaussian centered at axis)
            let s_heat = p_aux_mw * (-self.profiles.rho[i].powi(2) / HEATING_WIDTH).exp();

            // Radiation sink
            let s_rad = COOLING_FACTOR
                * self.profiles.ne[i]
                * self.profiles.n_impurity[i]
                * te_old[i].abs().sqrt();

            // Euler step
            self.profiles.te[i] =
                (te_old[i] + dt * (div_flux + s_heat - s_rad)).max(EDGE_TEMPERATURE);
        }

        // Ti tracks Te (simplified)
        for i in 1..n - 1 {
            self.profiles.ti[i] = self.profiles.te[i];
        }

        // Boundary conditions
        self.profiles.te[n - 1] = EDGE_TEMPERATURE;
        self.profiles.ti[n - 1] = EDGE_TEMPERATURE;
    }

    /// Inject impurities (erosion model). Python lines 39-66.
    pub fn inject_impurities(&mut self, erosion_rate: f64) {
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
    }

    /// Full transport step: update model, evolve, pedestal crash, inject.
    pub fn step(&mut self, p_aux_mw: f64, erosion_rate: f64) {
        self.pedestal.advance(self.dt);
        self.update_transport_model(p_aux_mw);
        self.evolve_profiles(p_aux_mw);

        if p_aux_mw > HMODE_POWER_THRESHOLD {
            let grad_p = self.compute_pedestal_pressure_gradient();
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

        self.inject_impurities(erosion_rate);
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
        ts.step(50.0, 1e14);
        // Should not panic and profiles should remain finite
        assert!(ts.profiles.te.iter().all(|v| v.is_finite()));
        assert!(ts.profiles.ti.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_hmode_barrier() {
        let mut ts = TransportSolver::new();

        // Without H-mode
        ts.update_transport_model(10.0);
        let chi_edge_no_hmode = ts.chi[48]; // near ρ=0.96

        // With H-mode
        ts.update_transport_model(50.0);
        let chi_edge_hmode = ts.chi[48];

        assert!(
            chi_edge_hmode < chi_edge_no_hmode,
            "H-mode should reduce edge chi: {} vs {}",
            chi_edge_hmode,
            chi_edge_no_hmode
        );
    }

    #[test]
    fn test_transport_edge_boundary() {
        let mut ts = TransportSolver::new();
        for _ in 0..10 {
            ts.step(50.0, 0.0);
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
        ts.step(60.0, 0.0);
        let te_after = ts.profiles.te[edge_idx];

        assert!(
            te_after < te_before,
            "ELM crash should reduce pedestal Te: before={te_before}, after={te_after}"
        );
        assert!(ts.pedestal.last_gradient() > 0.0);
    }
}
