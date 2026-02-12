// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Transport
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! 1.5D radial transport solver with Bohm/Gyro-Bohm turbulence model.
//!
//! Port of `integrated_transport_solver.py` lines 12-162.
//! Solves heat and particle diffusion on radial grid ρ ∈ [0, 1].

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

/// H-mode pedestal barrier location. Python line 86.
const HMODE_BARRIER_RHO: f64 = 0.9;

/// Transport reduction in H-mode barrier. Python line 88.
const HMODE_CHI_REDUCTION: f64 = 0.1;

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
        }
    }

    /// Update thermal diffusivity using Bohm/Gyro-Bohm model.
    ///
    /// χ = χ_base + χ_turb · max(0, |∇T| - threshold)
    /// With H-mode barrier suppression at ρ > 0.9 when P_aux > 30 MW.
    pub fn update_transport_model(&mut self, p_aux_mw: f64) {
        let n = self.profiles.rho.len();
        let dr = if n > 1 { 1.0 / (n as f64 - 1.0) } else { 1.0 };

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

            // H-mode barrier
            if p_aux_mw > HMODE_POWER_THRESHOLD && self.profiles.rho[i] > HMODE_BARRIER_RHO {
                self.chi[i] *= HMODE_CHI_REDUCTION;
            }
        }
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

    /// Full transport step: update model, evolve, inject.
    pub fn step(&mut self, p_aux_mw: f64, erosion_rate: f64) {
        self.update_transport_model(p_aux_mw);
        self.evolve_profiles(p_aux_mw);
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
}
