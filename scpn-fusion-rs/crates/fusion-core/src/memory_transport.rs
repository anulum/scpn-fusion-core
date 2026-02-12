// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Memory-Kernel Transport
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Phase-space memory-kernel transport model.
//!
//! Replaces instantaneous heat flux with a short-memory kernel:
//!   q(rho, t) = - ∫ K(t - t') * chi(rho, t') * dT/drho(rho, t') dt'
//! with exponential kernel K(t) = (1/tau_d) exp(-t/tau_d).

use fusion_types::state::RadialProfiles;
use ndarray::Array1;

const TRANSPORT_NR: usize = 50;
const CHI_BASE: f64 = 0.5;
const CHI_TURB: f64 = 5.0;
const CRIT_GRADIENT: f64 = 2.0;
const HMODE_BARRIER_RHO: f64 = 0.9;
const HMODE_CHI_REDUCTION: f64 = 0.1;
const HMODE_POWER_THRESHOLD: f64 = 30.0;
const EDGE_TEMPERATURE: f64 = 0.1;
const HEATING_WIDTH: f64 = 0.1;
const COOLING_FACTOR: f64 = 1.0;
const MAX_MEMORY_DRIVE: f64 = 1.0e4;
const MAX_DIV_FLUX: f64 = 1.0e5;
const MAX_HEATING: f64 = 1.0e4;
const MAX_TEMPERATURE: f64 = 100.0;

#[derive(Debug, Clone, Copy)]
pub struct MemoryKernelConfig {
    /// Memory decay time [s].
    pub tau_d: f64,
}

impl Default for MemoryKernelConfig {
    fn default() -> Self {
        Self { tau_d: 1e-3 }
    }
}

pub struct MemoryTransportSolver {
    pub profiles: RadialProfiles,
    pub chi: Array1<f64>,
    /// Memory variable Q_mem = ∫ K * (chi * dT/drho) dt'
    pub q_memory: Array1<f64>,
    pub config: MemoryKernelConfig,
}

impl MemoryTransportSolver {
    pub fn new(config: MemoryKernelConfig) -> Self {
        let rho = Array1::linspace(0.0, 1.0, TRANSPORT_NR);
        let te = Array1::from_shape_fn(TRANSPORT_NR, |i| {
            let r: f64 = rho[i];
            6.0 * (1.0_f64 - r * r).max(0.0) + EDGE_TEMPERATURE
        });
        let ti = te.clone();
        let ne = Array1::from_shape_fn(TRANSPORT_NR, |i| {
            let r: f64 = rho[i];
            10.0 * (1.0_f64 - r * r).max(0.0) + 0.5
        });
        let n_impurity = Array1::zeros(TRANSPORT_NR);
        let chi = Array1::from_elem(TRANSPORT_NR, CHI_BASE);
        let q_memory = Array1::zeros(TRANSPORT_NR);

        Self {
            profiles: RadialProfiles {
                rho,
                te,
                ti,
                ne,
                n_impurity,
            },
            chi,
            q_memory,
            config,
        }
    }

    fn gradient(values: &Array1<f64>, dr: f64) -> Array1<f64> {
        let n = values.len();
        let mut grad = Array1::zeros(n);
        for i in 0..n {
            grad[i] = if i == 0 {
                (values[1] - values[0]) / dr
            } else if i == n - 1 {
                (values[n - 1] - values[n - 2]) / dr
            } else {
                (values[i + 1] - values[i - 1]) / (2.0 * dr)
            };
        }
        grad
    }

    fn divergence_flux_cylindrical(flux: &Array1<f64>, rho: &Array1<f64>, dr: f64) -> Array1<f64> {
        let n = flux.len();
        let mut div = Array1::zeros(n);
        for i in 1..n - 1 {
            let rho_i = rho[i].max(1e-6);
            let rho_plus = (rho_i + 0.5 * dr).max(1e-6);
            let rho_minus = (rho_i - 0.5 * dr).max(1e-6);
            let f_plus = 0.5 * (flux[i] + flux[i + 1]);
            let f_minus = 0.5 * (flux[i - 1] + flux[i]);
            div[i] = (rho_plus * f_plus - rho_minus * f_minus) / (rho_i * dr);
        }
        div
    }

    fn update_transport_model(&mut self, p_aux_mw: f64) {
        let n = self.profiles.rho.len();
        let dr = if n > 1 { 1.0 / (n as f64 - 1.0) } else { 1.0 };
        let grad_t = Self::gradient(&self.profiles.te, dr);

        for i in 0..n {
            let excess = (-grad_t[i] - CRIT_GRADIENT).max(0.0);
            self.chi[i] = CHI_BASE + CHI_TURB * excess;

            if p_aux_mw > HMODE_POWER_THRESHOLD && self.profiles.rho[i] > HMODE_BARRIER_RHO {
                self.chi[i] *= HMODE_CHI_REDUCTION;
            }
        }
    }

    fn update_memory_from_drive(&mut self, drive: &Array1<f64>, dt: f64) {
        let drive = drive.mapv(|v| v.clamp(-MAX_MEMORY_DRIVE, MAX_MEMORY_DRIVE));
        if self.config.tau_d <= 1e-9 {
            self.q_memory.assign(&drive);
            return;
        }

        let decay = (-dt / self.config.tau_d).exp();
        let scale = dt / self.config.tau_d;
        self.q_memory = (self.q_memory.mapv(|q| q * decay) + drive * scale)
            .mapv(|v| v.clamp(-MAX_MEMORY_DRIVE, MAX_MEMORY_DRIVE));
    }

    /// Advance the transport system by one step.
    pub fn step(&mut self, p_aux_mw: f64, dt: f64) {
        let n = self.profiles.rho.len();
        let dr = if n > 1 { 1.0 / (n as f64 - 1.0) } else { 1.0 };

        self.update_transport_model(p_aux_mw);

        let grad_t = Self::gradient(&self.profiles.te, dr);
        let drive = &self.chi * &grad_t;
        self.update_memory_from_drive(&drive, dt);

        // q = -Q_mem
        let flux = self.q_memory.mapv(|v| -v);
        let div_flux = Self::divergence_flux_cylindrical(&flux, &self.profiles.rho, dr).mapv(|v| {
            if v.is_finite() {
                v.clamp(-MAX_DIV_FLUX, MAX_DIV_FLUX)
            } else {
                0.0
            }
        });

        let te_old = self.profiles.te.clone();
        for i in 1..n - 1 {
            let s_heat = (p_aux_mw * (-self.profiles.rho[i].powi(2) / HEATING_WIDTH).exp())
                .clamp(0.0, MAX_HEATING);
            let s_cool = COOLING_FACTOR * te_old[i].abs().sqrt() * self.profiles.n_impurity[i];
            let te_new = te_old[i] + dt * (div_flux[i] + s_heat - s_cool);
            self.profiles.te[i] = if te_new.is_finite() {
                te_new.clamp(EDGE_TEMPERATURE, MAX_TEMPERATURE)
            } else {
                EDGE_TEMPERATURE
            };
            self.profiles.ti[i] = self.profiles.te[i];
        }

        self.profiles.te[n - 1] = EDGE_TEMPERATURE;
        self.profiles.ti[n - 1] = EDGE_TEMPERATURE;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Zip;

    #[test]
    fn test_zero_tau_d_matches_local() {
        let mut solver = MemoryTransportSolver::new(MemoryKernelConfig { tau_d: 0.0 });
        let n = solver.profiles.rho.len();
        let dr = 1.0 / (n as f64 - 1.0);
        let grad_t = MemoryTransportSolver::gradient(&solver.profiles.te, dr);
        let drive = &solver.chi * &grad_t;

        solver.update_memory_from_drive(&drive, 1e-3);

        let max_err = solver
            .q_memory
            .iter()
            .zip(drive.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            max_err < 1e-12,
            "Expected local limit match, max_err={max_err}"
        );
    }

    #[test]
    fn test_memory_kernel_step_response() {
        let tau_d = 2e-3;
        let dt = 2e-4;
        let expected_gain = dt / tau_d;
        let mut solver = MemoryTransportSolver::new(MemoryKernelConfig { tau_d });

        let drive = Array1::from_elem(solver.q_memory.len(), 1.0);
        solver.update_memory_from_drive(&drive, dt);
        let first = solver.q_memory[10];

        for _ in 0..20 {
            solver.update_memory_from_drive(&drive, dt);
        }
        let later = solver.q_memory[10];

        assert!(
            first > 0.0 && first <= expected_gain + 1e-6,
            "First response should be small and positive: first={first}, expected_gain={expected_gain}"
        );
        assert!(
            later > first && later < 1.0 + 1e-6,
            "Memory response should relax monotonically toward 1: first={first}, later={later}"
        );
    }

    #[test]
    fn test_memory_finite_after_1000_steps() {
        let mut solver = MemoryTransportSolver::new(MemoryKernelConfig { tau_d: 1e-3 });
        for _ in 0..1000 {
            solver.step(35.0, 1e-4);
        }

        assert!(solver.profiles.te.iter().all(|v| v.is_finite()));
        assert!(solver.profiles.ti.iter().all(|v| v.is_finite()));
        assert!(solver.q_memory.iter().all(|v| v.is_finite()));
        assert!(solver.chi.iter().all(|v| v.is_finite()));
        assert!(solver.profiles.te.iter().all(|v| *v >= EDGE_TEMPERATURE));

        let mut changed = false;
        Zip::from(&solver.q_memory).for_each(|q| {
            if q.abs() > 1e-12 {
                changed = true;
            }
        });
        assert!(changed, "Expected non-trivial memory state after long run");
    }
}
