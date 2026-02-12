// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Divertor
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Divertor thermal simulation with Eich scaling and vapor shielding.
//!
//! Port of `divertor_thermal_sim.py`.
//! Models heat flux width, solid tungsten conduction, and
//! self-regulating lithium vapor shield.

/// Eich scaling coefficient [mm]. Python: 0.63.
const EICH_COEFF: f64 = 0.63;

/// Eich B-field exponent. Python: -1.19.
const EICH_EXP: f64 = -1.19;

/// Tungsten thermal conductivity [W/(m·K)]. Python: 100.
const K_W: f64 = 100.0;

/// Monoblock thickness [m]. Python: 0.01.
const D_BLOCK: f64 = 0.01;

/// Water coolant temperature [°C]. Python: 100.
const T_COOLANT: f64 = 100.0;

/// Tungsten melting point [°C]. Python: 3422.
const T_MELT_W: f64 = 3422.0;

/// Lithium boiling point [°C]. Python: 1342.
const T_BOIL_LI: f64 = 1342.0;

/// Lithium effective conductivity [W/(m·K)]. Python: 200.
const K_EFF_LI: f64 = 200.0;

/// Lithium layer depth [m]. Python: 0.005.
const D_LI_LAYER: f64 = 0.005;

/// Radiation onset temperature [°C]. Python: 400.
const T_MIN_RAD: f64 = 400.0;

/// Sigmoid midpoint [°C]. Python: 700.
const T_MID_RAD: f64 = 700.0;

/// Sigmoid width [°C]. Python: 100.
const T_WIDTH_RAD: f64 = 100.0;

/// Asymptotic shielding fraction. Python: 0.95.
const F_RAD_SAT: f64 = 0.95;

/// Relaxation factor. Python: 0.5.
const RELAXATION: f64 = 0.5;

/// Ambient temperature [°C].
const T_AMBIENT: f64 = 300.0;

/// Status of a divertor surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SurfaceStatus {
    Ok,
    Melted,
    BoilingLithium,
}

/// Divertor thermal analysis results.
#[derive(Debug, Clone)]
pub struct DivertorResult {
    /// Target heat flux [MW/m²].
    pub q_target_mw_m2: f64,
    /// Tungsten surface temperature [°C].
    pub t_surface_w: f64,
    /// Tungsten status.
    pub status_w: SurfaceStatus,
    /// Lithium surface temperature [°C].
    pub t_surface_li: f64,
    /// Radiative shielding fraction.
    pub f_rad: f64,
    /// Lithium status.
    pub status_li: SurfaceStatus,
}

/// Divertor thermal simulator.
pub struct DivertorLab {
    /// SOL power [MW].
    pub p_sol_mw: f64,
    /// Major radius [m].
    pub r_major: f64,
    /// Poloidal field [T].
    pub b_pol: f64,
}

impl DivertorLab {
    pub fn new(p_sol_mw: f64, r_major: f64, b_pol: f64) -> Self {
        DivertorLab {
            p_sol_mw,
            r_major,
            b_pol,
        }
    }

    /// Eich heat flux width [mm].
    pub fn lambda_q_mm(&self) -> f64 {
        EICH_COEFF * self.b_pol.powf(EICH_EXP)
    }

    /// Calculate target heat flux [MW/m²] given expansion factor.
    pub fn calculate_heat_load(&self, expansion_factor: f64) -> f64 {
        let lambda_q_m = self.lambda_q_mm() * 1e-3;
        let q_parallel = self.p_sol_mw * 1e6
            / (2.0 * std::f64::consts::PI * self.r_major * lambda_q_m);
        let q_target = q_parallel / expansion_factor;
        q_target / 1e6 // MW/m²
    }

    /// Simulate tungsten monoblock. Returns (T_surface [°C], status).
    pub fn simulate_tungsten(&self, q_target_mw_m2: f64) -> (f64, SurfaceStatus) {
        let q = q_target_mw_m2 * 1e6; // W/m²
        let t_surface = T_COOLANT + q * D_BLOCK / K_W;
        let status = if t_surface > T_MELT_W {
            SurfaceStatus::Melted
        } else {
            SurfaceStatus::Ok
        };
        (t_surface, status)
    }

    /// Simulate lithium vapor shield (self-regulating iterative).
    /// Returns (T_surface [°C], f_rad, status).
    pub fn simulate_lithium_vapor(&self, q_target_mw_m2: f64) -> (f64, f64, SurfaceStatus) {
        let q = q_target_mw_m2 * 1e6; // W/m²
        let mut t_li = T_AMBIENT;
        let mut f_rad = 0.0;

        for _ in 0..50 {
            // Radiative fraction (sigmoid)
            f_rad = if t_li < T_MIN_RAD {
                0.0
            } else {
                F_RAD_SAT / (1.0 + (-(t_li - T_MID_RAD) / T_WIDTH_RAD).exp())
            };

            let q_surface = q * (1.0 - f_rad);
            let t_new = T_AMBIENT + q_surface * D_LI_LAYER / K_EFF_LI;
            t_li = RELAXATION * t_li + (1.0 - RELAXATION) * t_new;
        }

        let status = if t_li > T_BOIL_LI {
            SurfaceStatus::BoilingLithium
        } else {
            SurfaceStatus::Ok
        };

        (t_li, f_rad, status)
    }

    /// Full divertor analysis at given expansion factor.
    pub fn analyze(&self, expansion_factor: f64) -> DivertorResult {
        let q_target = self.calculate_heat_load(expansion_factor);
        let (t_w, status_w) = self.simulate_tungsten(q_target);
        let (t_li, f_rad, status_li) = self.simulate_lithium_vapor(q_target);

        DivertorResult {
            q_target_mw_m2: q_target,
            t_surface_w: t_w,
            status_w,
            t_surface_li: t_li,
            f_rad,
            status_li,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eich_lambda_q() {
        let lab = DivertorLab::new(50.0, 2.1, 2.0);
        let lq = lab.lambda_q_mm();
        assert!(lq > 0.0, "λ_q should be positive: {lq}");
        assert!(lq < 5.0, "λ_q should be < 5 mm: {lq}");
    }

    #[test]
    fn test_heat_load_positive() {
        let lab = DivertorLab::new(50.0, 2.1, 2.0);
        let q = lab.calculate_heat_load(15.0);
        assert!(q > 0.0, "Heat load should be positive: {q}");
        assert!(q.is_finite(), "Heat load must be finite");
    }

    #[test]
    fn test_tungsten_melts_at_high_flux() {
        let lab = DivertorLab::new(80.0, 2.1, 2.5);
        let q = lab.calculate_heat_load(5.0); // Low expansion → high load
        let (t_w, status) = lab.simulate_tungsten(q);
        assert!(
            t_w > T_COOLANT,
            "Surface should be hotter than coolant: {t_w}"
        );
        // At low expansion, tungsten should melt
        if q > 30.0 {
            assert_eq!(status, SurfaceStatus::Melted);
        }
    }

    #[test]
    fn test_lithium_shields() {
        let lab = DivertorLab::new(80.0, 2.1, 2.5);
        let q = lab.calculate_heat_load(15.0);
        let (t_li, f_rad, _) = lab.simulate_lithium_vapor(q);
        assert!(t_li.is_finite(), "Li temp should be finite: {t_li}");
        assert!(
            f_rad >= 0.0 && f_rad <= 1.0,
            "Radiative fraction in [0,1]: {f_rad}"
        );
    }

    #[test]
    fn test_lithium_cooler_than_tungsten() {
        let lab = DivertorLab::new(50.0, 2.1, 2.0);
        let q = lab.calculate_heat_load(10.0);
        let (t_w, _) = lab.simulate_tungsten(q);
        let (t_li, _, _) = lab.simulate_lithium_vapor(q);
        assert!(
            t_li < t_w,
            "Lithium with vapor shield should be cooler: Li={t_li} vs W={t_w}"
        );
    }
}
