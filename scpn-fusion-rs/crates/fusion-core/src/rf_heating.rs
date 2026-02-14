// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — RF Heating
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! ICRH ray tracing for RF heating deposition.
//!
//! Port of `rf_heating.py` (200 lines).
//! Uses Hamiltonian ray equations in cold plasma approximation.

use fusion_types::constants::MU0_SI;
use fusion_types::error::{FusionError, FusionResult};

/// Deuterium charge [C]. Python line 23.
const Q_D: f64 = 1.602e-19;

/// Deuterium mass [kg]. Python line 24.
const M_D: f64 = 3.34e-27;

/// ICRH frequency [Hz]. Python line 25.
const FREQ_HZ: f64 = 50e6;

/// Angular frequency [rad/s].
const OMEGA_WAVE: f64 = 2.0 * std::f64::consts::PI * FREQ_HZ;

/// Magnetic field on axis [T]. Python line 34.
const B0: f64 = 5.3;

/// Major radius [m]. Python line 35.
const R0: f64 = 6.2;

/// Peak electron density [m⁻³]. Python line 56.
const N_E_PEAK: f64 = 1.0e20;

/// Density Gaussian width [m²]. Python line 56.
const DENSITY_WIDTH: f64 = 2.0;

/// Finite difference step for gradient [m]. Python line 96.
const GRAD_EPS: f64 = 1e-3;

/// Initial wavenumber [m⁻¹]. Python line 139.
const K0_INITIAL: f64 = 10.0;

/// Number of rays to trace. Python line 131.
const N_RAYS: usize = 10;

/// Antenna radial position [m]. Python line 131.
const R_ANTENNA: f64 = 9.0;

/// Resonance layer intersection tolerance [m]. Python line 179.
const RESONANCE_TOL: f64 = 0.1;

/// Ray state: (R, Z, k_R, k_Z).
pub type RayState = [f64; 4];

/// Result of a single ray trace.
#[derive(Debug, Clone)]
pub struct RayTraceResult {
    /// Ray trajectory points (R, Z).
    pub trajectory: Vec<(f64, f64)>,
    /// Resonance crossing position, if found.
    pub resonance_point: Option<(f64, f64)>,
    /// Resonance layer R coordinate.
    pub r_resonance: f64,
}

/// ICRH ray tracing system.
pub struct RFHeatingSystem {
    pub omega: f64,
    pub b0: f64,
    pub r0: f64,
}

impl RFHeatingSystem {
    /// Create with ITER-like parameters.
    pub fn new() -> Self {
        RFHeatingSystem {
            omega: OMEGA_WAVE,
            b0: B0,
            r0: R0,
        }
    }

    fn validate_runtime_parameters(&self) -> FusionResult<()> {
        if !self.omega.is_finite() || self.omega <= 0.0 {
            return Err(FusionError::ConfigError(format!(
                "RF omega must be finite and > 0, got {}",
                self.omega
            )));
        }
        if !self.b0.is_finite() || self.b0 <= 0.0 {
            return Err(FusionError::ConfigError(format!(
                "RF b0 must be finite and > 0, got {}",
                self.b0
            )));
        }
        if !self.r0.is_finite() || self.r0 <= 0.0 {
            return Err(FusionError::ConfigError(format!(
                "RF r0 must be finite and > 0, got {}",
                self.r0
            )));
        }
        Ok(())
    }

    /// Toroidal magnetic field at radius R: B(R) = B0 · R0 / R.
    fn b_field(&self, r: f64) -> f64 {
        self.b0 * self.r0 / r
    }

    /// Electron density profile (Gaussian). Python line 56.
    fn density(&self, r: f64, z: f64) -> f64 {
        let dist_sq = (r - self.r0).powi(2) + z.powi(2);
        N_E_PEAK * (-dist_sq / DENSITY_WIDTH).exp()
    }

    /// Alfvén speed: v_A = B / sqrt(μ₀ n_e m_D).
    fn alfven_speed(&self, r: f64, z: f64) -> f64 {
        let b = self.b_field(r);
        let n = self.density(r, z).max(1e10); // floor to avoid division by zero
        b / (MU0_SI * n * M_D).sqrt()
    }

    /// Dispersion relation: D = k²v_A² - ω².
    fn dispersion(&self, r: f64, z: f64, kr: f64, kz: f64) -> f64 {
        let k_sq = kr * kr + kz * kz;
        let v_a = self.alfven_speed(r, z);
        k_sq * v_a * v_a - self.omega * self.omega
    }

    /// Ray equations (Hamiltonian): ds/dt = (dR/dt, dZ/dt, dk_R/dt, dk_Z/dt).
    ///
    /// dR/dt = -∂D/∂k_R,  dZ/dt = -∂D/∂k_Z
    /// dk_R/dt = ∂D/∂R,   dk_Z/dt = ∂D/∂Z
    fn ray_rhs(&self, state: &RayState) -> RayState {
        let [r, z, kr, kz] = *state;
        let eps = GRAD_EPS;

        // ∂D/∂k_R, ∂D/∂k_Z via finite differences
        let dd_dkr = (self.dispersion(r, z, kr + eps, kz) - self.dispersion(r, z, kr - eps, kz))
            / (2.0 * eps);
        let dd_dkz = (self.dispersion(r, z, kr, kz + eps) - self.dispersion(r, z, kr, kz - eps))
            / (2.0 * eps);

        // ∂D/∂R, ∂D/∂Z via finite differences
        let dd_dr = (self.dispersion(r + eps, z, kr, kz) - self.dispersion(r - eps, z, kr, kz))
            / (2.0 * eps);
        let dd_dz = (self.dispersion(r, z + eps, kr, kz) - self.dispersion(r, z - eps, kr, kz))
            / (2.0 * eps);

        [
            -dd_dkr, // dR/dt
            -dd_dkz, // dZ/dt
            dd_dr,   // dk_R/dt
            dd_dz,   // dk_Z/dt
        ]
    }

    /// Adaptive RK4 integration step with sub-stepping.
    /// The Alfvén speed can be ~1e7 m/s, so group velocity dR/dt ~ k·v_A²
    /// can be extremely large. We use adaptive sub-stepping to keep
    /// displacement per step < 0.1 m.
    fn rk4_step(&self, state: &RayState, dt: f64) -> RayState {
        let k1 = self.ray_rhs(state);

        let s2 = [
            state[0] + 0.5 * dt * k1[0],
            state[1] + 0.5 * dt * k1[1],
            state[2] + 0.5 * dt * k1[2],
            state[3] + 0.5 * dt * k1[3],
        ];
        let k2 = self.ray_rhs(&s2);

        let s3 = [
            state[0] + 0.5 * dt * k2[0],
            state[1] + 0.5 * dt * k2[1],
            state[2] + 0.5 * dt * k2[2],
            state[3] + 0.5 * dt * k2[3],
        ];
        let k3 = self.ray_rhs(&s3);

        let s4 = [
            state[0] + dt * k3[0],
            state[1] + dt * k3[1],
            state[2] + dt * k3[2],
            state[3] + dt * k3[3],
        ];
        let k4 = self.ray_rhs(&s4);

        [
            state[0] + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
            state[1] + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
            state[2] + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
            state[3] + (dt / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]),
        ]
    }

    /// Ion cyclotron resonance radius.
    ///
    /// ω_ci = qB/m → B_res = ωm/q → R_res = B0·R0/B_res
    pub fn resonance_radius(&self) -> FusionResult<f64> {
        self.validate_runtime_parameters()?;
        let b_res = self.omega * M_D / Q_D;
        if !b_res.is_finite() || b_res <= 0.0 {
            return Err(FusionError::ConfigError(
                "RF resonance field must be finite and > 0".to_string(),
            ));
        }
        let r_res = self.b0 * self.r0 / b_res;
        if !r_res.is_finite() || r_res <= 0.0 {
            return Err(FusionError::ConfigError(
                "RF resonance radius became non-finite or non-positive".to_string(),
            ));
        }
        Ok(r_res)
    }

    /// Trace a single ray from given initial conditions.
    ///
    /// Uses adaptive time stepping: estimates RHS magnitude and scales dt
    /// so the spatial displacement per step is bounded.
    pub fn trace_single_ray(
        &self,
        r0: f64,
        z0: f64,
        kr0: f64,
        kz0: f64,
        n_steps: usize,
        _dt_hint: f64,
    ) -> FusionResult<RayTraceResult> {
        self.validate_runtime_parameters()?;
        if !r0.is_finite() || r0 <= 0.0 || !z0.is_finite() || !kr0.is_finite() || !kz0.is_finite() {
            return Err(FusionError::ConfigError(
                "RF ray initial conditions must be finite with r0 > 0".to_string(),
            ));
        }
        if n_steps == 0 {
            return Err(FusionError::ConfigError(
                "RF ray tracing requires n_steps >= 1".to_string(),
            ));
        }

        let r_res = self.resonance_radius()?;
        let mut state: RayState = [r0, z0, kr0, kz0];
        let mut trajectory = Vec::with_capacity(n_steps);
        let mut resonance_point = None;

        // Maximum spatial displacement per step [m]
        const MAX_DISP: f64 = 0.05;

        for _ in 0..n_steps {
            if state.iter().any(|v| !v.is_finite()) {
                return Err(FusionError::ConfigError(
                    "RF ray state became non-finite".to_string(),
                ));
            }
            trajectory.push((state[0], state[1]));

            // Check resonance crossing
            if resonance_point.is_none() && (state[0] - r_res).abs() < RESONANCE_TOL {
                resonance_point = Some((state[0], state[1]));
            }

            // Adaptive dt: estimate RHS magnitude
            let rhs = self.ray_rhs(&state);
            if rhs.iter().any(|v| !v.is_finite()) {
                return Err(FusionError::ConfigError(
                    "RF ray RHS became non-finite".to_string(),
                ));
            }
            let v_mag = (rhs[0] * rhs[0] + rhs[1] * rhs[1]).sqrt();
            if !v_mag.is_finite() {
                return Err(FusionError::ConfigError(
                    "RF ray velocity magnitude became non-finite".to_string(),
                ));
            }
            let dt = if v_mag > 1e-10 {
                MAX_DISP / v_mag
            } else {
                1e-3
            };
            if !dt.is_finite() || dt <= 0.0 {
                return Err(FusionError::ConfigError(
                    "RF adaptive timestep became invalid".to_string(),
                ));
            }

            state = self.rk4_step(&state, dt);

            // Boundary check (stay within reasonable domain)
            if state[0] < 0.5 || state[0] > 15.0 || state[1].abs() > 10.0 {
                break;
            }
        }

        Ok(RayTraceResult {
            trajectory,
            resonance_point,
            r_resonance: r_res,
        })
    }

    /// Trace all rays from the antenna array. Python lines 126-157.
    pub fn trace_rays(&self) -> FusionResult<Vec<RayTraceResult>> {
        self.validate_runtime_parameters()?;
        let n_steps = 500;
        let dt = 0.0; // adaptive

        let mut results = Vec::with_capacity(N_RAYS);
        for i in 0..N_RAYS {
            let z_antenna = -1.0 + 2.0 * (i as f64) / ((N_RAYS - 1) as f64);
            let result =
                self.trace_single_ray(R_ANTENNA, z_antenna, -K0_INITIAL, 0.0, n_steps, dt)?;
            results.push(result);
        }
        Ok(results)
    }
}

impl Default for RFHeatingSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resonance_radius() {
        let rf = RFHeatingSystem::new();
        let r_res = rf
            .resonance_radius()
            .expect("valid RF resonance-radius parameters");
        // For ITER: B0=5.3T, R0=6.2m, f=50MHz
        // B_res = 2π·50e6·3.34e-27/1.602e-19 ≈ 6.54 T
        // R_res = 5.3·6.2/6.54 ≈ 5.02 m
        assert!(r_res > 3.0 && r_res < 8.0, "Resonance radius: {r_res}");
    }

    #[test]
    fn test_trace_single_ray() {
        let rf = RFHeatingSystem::new();
        let result = rf
            .trace_single_ray(R_ANTENNA, 0.0, -K0_INITIAL, 0.0, 500, 0.0)
            .expect("valid RF single-ray trace parameters");

        // Ray should propagate inward
        assert!(result.trajectory.len() > 1, "Should have trajectory points");

        // First point at antenna
        let (r0, _z0) = result.trajectory[0];
        assert!((r0 - R_ANTENNA).abs() < 1e-10, "Should start at antenna");
    }

    #[test]
    fn test_trace_rays_array() {
        let rf = RFHeatingSystem::new();
        let results = rf
            .trace_rays()
            .expect("valid RF multi-ray trace parameters");
        assert_eq!(results.len(), N_RAYS);
        for result in &results {
            assert!(!result.trajectory.is_empty());
        }
    }

    #[test]
    fn test_alfven_speed_positive() {
        let rf = RFHeatingSystem::new();
        let v_a = rf.alfven_speed(6.2, 0.0);
        assert!(
            v_a > 0.0 && v_a.is_finite(),
            "v_A should be positive: {v_a}"
        );
    }

    #[test]
    fn test_rf_heating_rejects_invalid_runtime_inputs() {
        let mut rf = RFHeatingSystem::new();
        rf.b0 = f64::NAN;
        assert!(rf.resonance_radius().is_err());
        assert!(rf.trace_rays().is_err());

        let rf = RFHeatingSystem::new();
        assert!(rf
            .trace_single_ray(0.0, 0.0, -K0_INITIAL, 0.0, 500, 0.0)
            .is_err());
        assert!(rf
            .trace_single_ray(R_ANTENNA, 0.0, -K0_INITIAL, 0.0, 0, 0.0)
            .is_err());
    }
}
