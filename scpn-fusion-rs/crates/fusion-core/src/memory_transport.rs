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

use fusion_types::error::{FusionError, FusionResult};
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
    pub fn new(config: MemoryKernelConfig) -> FusionResult<Self> {
        if !config.tau_d.is_finite() || config.tau_d < 0.0 {
            return Err(FusionError::ConfigError(format!(
                "memory transport tau_d must be finite and >= 0, got {}",
                config.tau_d
            )));
        }

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

        let solver = Self {
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
        };
        if solver.profiles.te.iter().any(|v| !v.is_finite())
            || solver.profiles.ti.iter().any(|v| !v.is_finite())
            || solver.profiles.ne.iter().any(|v| !v.is_finite())
            || solver.chi.iter().any(|v| !v.is_finite())
            || solver.q_memory.iter().any(|v| !v.is_finite())
        {
            return Err(FusionError::ConfigError(
                "memory transport initialization produced non-finite values".to_string(),
            ));
        }
        Ok(solver)
    }

    fn gradient(values: &Array1<f64>, dr: f64) -> FusionResult<Array1<f64>> {
        if values.len() < 2 {
            return Err(FusionError::ConfigError(
                "memory transport gradient requires at least 2 samples".to_string(),
            ));
        }
        if !dr.is_finite() || dr <= 0.0 {
            return Err(FusionError::ConfigError(format!(
                "memory transport gradient requires finite dr > 0, got {dr}"
            )));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(FusionError::ConfigError(
                "memory transport gradient input contains non-finite values".to_string(),
            ));
        }
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
        if grad.iter().any(|v| !v.is_finite()) {
            return Err(FusionError::ConfigError(
                "memory transport gradient output contains non-finite values".to_string(),
            ));
        }
        Ok(grad)
    }

    fn divergence_flux_cylindrical(
        flux: &Array1<f64>,
        rho: &Array1<f64>,
        dr: f64,
    ) -> FusionResult<Array1<f64>> {
        if flux.len() != rho.len() || flux.len() < 2 {
            return Err(FusionError::ConfigError(format!(
                "memory transport divergence requires matching flux/rho lengths >= 2, got {} and {}",
                flux.len(),
                rho.len()
            )));
        }
        if !dr.is_finite() || dr <= 0.0 {
            return Err(FusionError::ConfigError(format!(
                "memory transport divergence requires finite dr > 0, got {dr}"
            )));
        }
        if flux.iter().any(|v| !v.is_finite()) || rho.iter().any(|v| !v.is_finite()) {
            return Err(FusionError::ConfigError(
                "memory transport divergence input contains non-finite values".to_string(),
            ));
        }
        let n = flux.len();
        let mut div = Array1::zeros(n);
        for i in 1..n - 1 {
            let rho_i = rho[i];
            if rho_i <= 0.0 {
                return Err(FusionError::ConfigError(format!(
                    "memory transport rho must be > 0 for interior points, got {} at index {}",
                    rho_i, i
                )));
            }
            let rho_plus = rho_i + 0.5 * dr;
            let rho_minus = rho_i - 0.5 * dr;
            if rho_plus <= 0.0 || rho_minus <= 0.0 {
                return Err(FusionError::ConfigError(format!(
                    "memory transport cylindrical stencil requires positive rho +/- dr/2 at index {}",
                    i
                )));
            }
            let f_plus = 0.5 * (flux[i] + flux[i + 1]);
            let f_minus = 0.5 * (flux[i - 1] + flux[i]);
            div[i] = (rho_plus * f_plus - rho_minus * f_minus) / (rho_i * dr);
        }
        if div.iter().any(|v| !v.is_finite()) {
            return Err(FusionError::ConfigError(
                "memory transport divergence output contains non-finite values".to_string(),
            ));
        }
        Ok(div)
    }

    fn update_transport_model(&mut self, p_aux_mw: f64) -> FusionResult<()> {
        if !p_aux_mw.is_finite() || p_aux_mw < 0.0 {
            return Err(FusionError::ConfigError(format!(
                "memory transport auxiliary heating power must be finite and >= 0, got {p_aux_mw}"
            )));
        }
        let n = self.profiles.rho.len();
        let dr = if n > 1 { 1.0 / (n as f64 - 1.0) } else { 1.0 };
        let grad_t = Self::gradient(&self.profiles.te, dr)?;

        for i in 0..n {
            let excess = (-grad_t[i] - CRIT_GRADIENT).max(0.0);
            self.chi[i] = CHI_BASE + CHI_TURB * excess;

            if p_aux_mw > HMODE_POWER_THRESHOLD && self.profiles.rho[i] > HMODE_BARRIER_RHO {
                self.chi[i] *= HMODE_CHI_REDUCTION;
            }
        }
        if self.chi.iter().any(|v| !v.is_finite()) {
            return Err(FusionError::ConfigError(
                "memory transport chi update produced non-finite values".to_string(),
            ));
        }
        Ok(())
    }

    fn update_memory_from_drive(&mut self, drive: &Array1<f64>, dt: f64) -> FusionResult<()> {
        if drive.len() != self.q_memory.len() {
            return Err(FusionError::ConfigError(format!(
                "memory transport drive length mismatch: expected {}, got {}",
                self.q_memory.len(),
                drive.len()
            )));
        }
        if drive.iter().any(|v| !v.is_finite()) {
            return Err(FusionError::ConfigError(
                "memory transport drive contains non-finite values".to_string(),
            ));
        }
        if !dt.is_finite() || dt <= 0.0 {
            return Err(FusionError::ConfigError(format!(
                "memory transport dt must be finite and > 0, got {dt}"
            )));
        }
        if !self.config.tau_d.is_finite() || self.config.tau_d < 0.0 {
            return Err(FusionError::ConfigError(format!(
                "memory transport tau_d must remain finite and >= 0, got {}",
                self.config.tau_d
            )));
        }
        let drive = drive.mapv(|v| v.clamp(-MAX_MEMORY_DRIVE, MAX_MEMORY_DRIVE));
        if self.config.tau_d <= 1e-9 {
            self.q_memory.assign(&drive);
            return Ok(());
        }

        let decay = (-dt / self.config.tau_d).exp();
        let scale = dt / self.config.tau_d;
        if !decay.is_finite() || !scale.is_finite() {
            return Err(FusionError::ConfigError(
                "memory transport decay coefficients became non-finite".to_string(),
            ));
        }
        self.q_memory = (self.q_memory.mapv(|q| q * decay) + drive * scale)
            .mapv(|v| v.clamp(-MAX_MEMORY_DRIVE, MAX_MEMORY_DRIVE));
        if self.q_memory.iter().any(|v| !v.is_finite()) {
            return Err(FusionError::ConfigError(
                "memory transport memory state became non-finite".to_string(),
            ));
        }
        Ok(())
    }

    /// Advance the transport system by one step.
    pub fn step(&mut self, p_aux_mw: f64, dt: f64) -> FusionResult<()> {
        if !p_aux_mw.is_finite() || p_aux_mw < 0.0 {
            return Err(FusionError::ConfigError(format!(
                "memory transport step requires finite p_aux_mw >= 0, got {p_aux_mw}"
            )));
        }
        if !dt.is_finite() || dt <= 0.0 {
            return Err(FusionError::ConfigError(format!(
                "memory transport step requires finite dt > 0, got {dt}"
            )));
        }
        let n = self.profiles.rho.len();
        let dr = if n > 1 { 1.0 / (n as f64 - 1.0) } else { 1.0 };

        self.update_transport_model(p_aux_mw)?;

        let grad_t = Self::gradient(&self.profiles.te, dr)?;
        let drive = &self.chi * &grad_t;
        if drive.iter().any(|v| !v.is_finite()) {
            return Err(FusionError::ConfigError(
                "memory transport drive update became non-finite".to_string(),
            ));
        }
        self.update_memory_from_drive(&drive, dt)?;

        // q = -Q_mem
        let flux = self.q_memory.mapv(|v| -v);
        let div_flux = Self::divergence_flux_cylindrical(&flux, &self.profiles.rho, dr)?
            .mapv(|v| v.clamp(-MAX_DIV_FLUX, MAX_DIV_FLUX));

        let te_old = self.profiles.te.clone();
        for i in 1..n - 1 {
            let s_heat = (p_aux_mw * (-self.profiles.rho[i].powi(2) / HEATING_WIDTH).exp())
                .clamp(0.0, MAX_HEATING);
            let s_cool = COOLING_FACTOR * te_old[i].abs().sqrt() * self.profiles.n_impurity[i];
            if !s_heat.is_finite() || !s_cool.is_finite() {
                return Err(FusionError::ConfigError(format!(
                    "memory transport source terms became non-finite at index {}",
                    i
                )));
            }
            let te_new = te_old[i] + dt * (div_flux[i] + s_heat - s_cool);
            if !te_new.is_finite() {
                return Err(FusionError::ConfigError(format!(
                    "memory transport temperature update became non-finite at index {}",
                    i
                )));
            }
            self.profiles.te[i] = te_new.clamp(EDGE_TEMPERATURE, MAX_TEMPERATURE);
            self.profiles.ti[i] = self.profiles.te[i];
        }

        self.profiles.te[n - 1] = EDGE_TEMPERATURE;
        self.profiles.ti[n - 1] = EDGE_TEMPERATURE;
        if self.profiles.te.iter().any(|v| !v.is_finite())
            || self.profiles.ti.iter().any(|v| !v.is_finite())
            || self.q_memory.iter().any(|v| !v.is_finite())
            || self.chi.iter().any(|v| !v.is_finite())
        {
            return Err(FusionError::ConfigError(
                "memory transport state contains non-finite values after step".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Zip;

    #[test]
    fn test_zero_tau_d_matches_local() {
        let mut solver =
            MemoryTransportSolver::new(MemoryKernelConfig { tau_d: 0.0 }).expect("valid config");
        let n = solver.profiles.rho.len();
        let dr = 1.0 / (n as f64 - 1.0);
        let grad_t =
            MemoryTransportSolver::gradient(&solver.profiles.te, dr).expect("valid gradient input");
        let drive = &solver.chi * &grad_t;

        solver
            .update_memory_from_drive(&drive, 1e-3)
            .expect("valid memory update");

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
        let mut solver =
            MemoryTransportSolver::new(MemoryKernelConfig { tau_d }).expect("valid config");

        let drive = Array1::from_elem(solver.q_memory.len(), 1.0);
        solver
            .update_memory_from_drive(&drive, dt)
            .expect("valid memory update");
        let first = solver.q_memory[10];

        for _ in 0..20 {
            solver
                .update_memory_from_drive(&drive, dt)
                .expect("valid memory update");
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
        let mut solver =
            MemoryTransportSolver::new(MemoryKernelConfig { tau_d: 1e-3 }).expect("valid config");
        for _ in 0..1000 {
            solver.step(35.0, 1e-4).expect("valid transport step");
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

    #[test]
    fn test_memory_transport_rejects_invalid_runtime_inputs() {
        match MemoryTransportSolver::new(MemoryKernelConfig { tau_d: f64::NAN }) {
            Err(FusionError::ConfigError(_)) => {}
            Err(other) => panic!("unexpected error variant: {other:?}"),
            Ok(_) => panic!("non-finite tau_d must fail"),
        }

        let mut solver =
            MemoryTransportSolver::new(MemoryKernelConfig { tau_d: 1e-3 }).expect("valid config");
        let err = solver
            .step(35.0, 0.0)
            .expect_err("non-positive dt must fail");
        assert!(matches!(err, FusionError::ConfigError(_)));

        let bad_drive = Array1::from_elem(solver.q_memory.len(), f64::INFINITY);
        let err = solver
            .update_memory_from_drive(&bad_drive, 1e-4)
            .expect_err("non-finite drive must fail");
        assert!(matches!(err, FusionError::ConfigError(_)));
    }
}
