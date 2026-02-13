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

use ndarray::Array2;
use std::f64::consts::PI;

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

/// Lower-divertor strike-point centroid angle [rad].
const STRIKE_THETA_LOWER: f64 = 1.35 * PI;
/// Primary strike footprint angular width [rad].
const STRIKE_WIDTH_PRIMARY: f64 = 0.20;
/// Secondary/private-flux footprint angular width [rad].
const STRIKE_WIDTH_SECONDARY: f64 = 0.32;
/// Minimum positive toroidal modulation floor.
const TOROIDAL_MOD_MIN: f64 = 0.10;
/// Baseline floor for poloidal footprint weighting.
const STRIKE_FOOTPRINT_FLOOR: f64 = 0.05;

fn wrapped_angle_distance(a: f64, b: f64) -> f64 {
    let d = (a - b + PI).rem_euclid(2.0 * PI) - PI;
    d.abs()
}

/// Status of a divertor surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SurfaceStatus {
    Ok,
    Melted,
    BoilingLithium,
}

/// Low-order toroidal mode descriptor used for reduced 3D heat-flux closure.
#[derive(Debug, Clone, Copy)]
pub struct ToroidalMode {
    /// Toroidal mode number n (n >= 1).
    pub n: usize,
    /// Mode amplitude.
    pub amplitude: f64,
    /// Phase offset [rad].
    pub phase_rad: f64,
}

impl ToroidalMode {
    pub fn new(n: usize, amplitude: f64, phase_rad: f64) -> Self {
        Self {
            n: n.max(1),
            amplitude,
            phase_rad,
        }
    }
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
        let q_parallel =
            self.p_sol_mw * 1e6 / (2.0 * std::f64::consts::PI * self.r_major * lambda_q_m);
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

    /// Reduced 3D strike-point heat-flux map projection.
    ///
    /// Returns `q_target(theta, phi)` in MW/m² with average equal to the
    /// axisymmetric Eich-derived target heat load for the same expansion factor.
    pub fn project_heat_flux_3d(
        &self,
        expansion_factor: f64,
        n_poloidal: usize,
        n_toroidal: usize,
        modes: &[ToroidalMode],
    ) -> Array2<f64> {
        let n_poloidal = n_poloidal.max(1);
        let n_toroidal = n_toroidal.max(1);
        let q_base = self.calculate_heat_load(expansion_factor.max(1e-9));

        let mut heat_map = Array2::zeros((n_poloidal, n_toroidal));
        for i_theta in 0..n_poloidal {
            let theta = 2.0 * PI * i_theta as f64 / n_poloidal as f64;
            let d_primary = wrapped_angle_distance(theta, STRIKE_THETA_LOWER);
            let d_secondary = wrapped_angle_distance(theta, STRIKE_THETA_LOWER + PI);
            let primary = (-(0.5) * (d_primary / STRIKE_WIDTH_PRIMARY).powi(2)).exp();
            let secondary = 0.35 * (-(0.5) * (d_secondary / STRIKE_WIDTH_SECONDARY).powi(2)).exp();
            let poloidal_weight = (primary + secondary + STRIKE_FOOTPRINT_FLOOR).max(1e-9);

            for j_phi in 0..n_toroidal {
                let phi = 2.0 * PI * j_phi as f64 / n_toroidal as f64;
                let mut toroidal_mod = 1.0;
                for mode in modes {
                    let mode_n = mode.n.max(1) as f64;
                    let amp = mode.amplitude.clamp(-0.95, 0.95);
                    toroidal_mod += amp * (mode_n * phi + mode.phase_rad).cos();
                }
                toroidal_mod = toroidal_mod.max(TOROIDAL_MOD_MIN);
                heat_map[[i_theta, j_phi]] = poloidal_weight * toroidal_mod;
            }
        }

        // Normalize map so mean heat flux matches q_base.
        let mean_weight = heat_map.sum() / (n_poloidal * n_toroidal) as f64;
        if mean_weight > 1e-12 {
            heat_map.mapv_inplace(|v| q_base * v / mean_weight);
        } else {
            heat_map.fill(q_base);
        }

        heat_map
    }

    /// Max/min asymmetry ratio for a projected heat map.
    pub fn heat_flux_asymmetry_ratio(heat_map: &Array2<f64>) -> f64 {
        let mut q_max = f64::NEG_INFINITY;
        let mut q_min = f64::INFINITY;
        for q in heat_map.iter().copied() {
            q_max = q_max.max(q);
            q_min = q_min.min(q);
        }
        if !q_max.is_finite() || !q_min.is_finite() {
            return 1.0;
        }
        q_max / q_min.max(1e-12)
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

    fn toroidal_row_std(map: &Array2<f64>, row: usize) -> f64 {
        let row = row.min(map.nrows().saturating_sub(1));
        let values = map.row(row);
        let mean = values.iter().copied().sum::<f64>() / values.len().max(1) as f64;
        let var =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len().max(1) as f64;
        var.sqrt()
    }

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
            (0.0..=1.0).contains(&f_rad),
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

    #[test]
    fn test_heat_flux_3d_projection_preserves_mean_heat_load() {
        let lab = DivertorLab::new(60.0, 2.1, 2.0);
        let expansion = 14.0;
        let q_base = lab.calculate_heat_load(expansion);
        let map = lab.project_heat_flux_3d(expansion, 64, 48, &[]);
        let mean_q = map.sum() / map.len() as f64;
        let rel_err = ((mean_q - q_base) / q_base.max(1e-12)).abs();
        assert!(
            rel_err < 1e-12,
            "Mean projected heat load must match axisymmetric target: base={q_base}, mean={mean_q}, rel_err={rel_err}"
        );
    }

    #[test]
    fn test_heat_flux_3d_projection_has_strike_point_localization() {
        let lab = DivertorLab::new(80.0, 2.1, 2.4);
        let n_poloidal = 72;
        let n_toroidal = 64;
        let map = lab.project_heat_flux_3d(12.0, n_poloidal, n_toroidal, &[]);

        let mut best_idx = 0usize;
        let mut best_val = f64::NEG_INFINITY;
        for i in 0..n_poloidal {
            let row_mean = map.row(i).iter().copied().sum::<f64>() / n_toroidal as f64;
            if row_mean > best_val {
                best_val = row_mean;
                best_idx = i;
            }
        }

        let expected_idx = ((STRIKE_THETA_LOWER / (2.0 * PI)) * n_poloidal as f64).round() as isize;
        let mut d = (best_idx as isize - expected_idx).abs();
        d = d.min(n_poloidal as isize - d);
        assert!(
            d <= 4,
            "Expected strike localization near lower divertor index {} but got {}",
            expected_idx,
            best_idx
        );
    }

    #[test]
    fn test_heat_flux_3d_toroidal_modes_increase_toroidal_asymmetry() {
        let lab = DivertorLab::new(70.0, 2.1, 2.1);
        let expansion = 15.0;
        let n_poloidal = 64;
        let n_toroidal = 60;
        let map_base = lab.project_heat_flux_3d(expansion, n_poloidal, n_toroidal, &[]);
        let map_asym = lab.project_heat_flux_3d(
            expansion,
            n_poloidal,
            n_toroidal,
            &[
                ToroidalMode::new(1, 0.18, 0.0),
                ToroidalMode::new(2, 0.10, 0.4),
            ],
        );

        // Compare toroidal variation on the dominant strike row.
        let mut strike_row = 0usize;
        let mut best = f64::NEG_INFINITY;
        for i in 0..n_poloidal {
            let row_mean = map_base.row(i).iter().copied().sum::<f64>() / n_toroidal as f64;
            if row_mean > best {
                best = row_mean;
                strike_row = i;
            }
        }

        let std_base = toroidal_row_std(&map_base, strike_row);
        let std_asym = toroidal_row_std(&map_asym, strike_row);
        assert!(
            std_asym > std_base,
            "Expected stronger toroidal variation with modes: base_std={std_base}, asym_std={std_asym}"
        );

        let ratio = DivertorLab::heat_flux_asymmetry_ratio(&map_asym);
        assert!(
            ratio > 1.05,
            "Expected non-trivial asymmetry ratio, got {ratio}"
        );
    }
}
