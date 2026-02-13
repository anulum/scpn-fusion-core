// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Wall Interaction
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Nuclear wall interaction: He ash poisoning + DPA analysis.
//!
//! Port of `nuclear_wall_interaction.py`.
//! Models helium ash accumulation, wall neutron loading, and material lifetime.

use fusion_math::iga::{open_uniform_knots, ControlPoint2D, NurbsCurve2D};
use std::f64::consts::PI;

/// Core electron density [m⁻³]. Python: 1e20.
const N_E: f64 = 1e20;

/// Core temperature [keV]. Python: 20.
const T_KEV: f64 = 20.0;

/// Plasma volume [m³]. Python: 800.
const VOLUME: f64 = 800.0;

/// Energy confinement time [s]. Python: 3.0.
const TAU_E: f64 = 3.0;

/// Fusion energy [MeV]. Python: 17.6.
const E_FUSION_MEV: f64 = 17.6;

/// MeV to Joules.
const MEV_TO_J: f64 = 1.602e-13;

/// DPA per MW·year of neutron loading. Python: 10.
const DPA_PER_MW_YEAR: f64 = 10.0;

/// Supported impurity species for reduced collisional-radiative lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImpuritySpecies {
    Tungsten,
    Carbon,
}

#[derive(Debug, Clone, Copy)]
struct PecGrid {
    log_ne: [f64; 3],
    log_te: [f64; 3],
    values: [[f64; 3]; 3],
}

fn pec_grid(species: ImpuritySpecies, charge_state: u8) -> Option<PecGrid> {
    // Reduced ADAS-like synthetic grids.
    // Axes: log10(ne [m^-3]) = [19, 20, 21], log10(Te [eV]) = [2, 3, 4].
    const LOG_NE: [f64; 3] = [19.0, 20.0, 21.0];
    const LOG_TE: [f64; 3] = [2.0, 3.0, 4.0];

    let values = match (species, charge_state) {
        (ImpuritySpecies::Tungsten, 20) => [
            [2.2e-34, 4.1e-34, 6.0e-34],
            [5.0e-34, 8.5e-34, 1.2e-33],
            [9.0e-34, 1.5e-33, 2.1e-33],
        ],
        (ImpuritySpecies::Tungsten, 40) => [
            [5.0e-34, 8.2e-34, 1.15e-33],
            [1.1e-33, 1.7e-33, 2.4e-33],
            [1.9e-33, 2.9e-33, 4.1e-33],
        ],
        (ImpuritySpecies::Carbon, 4) => [
            [8.0e-35, 1.2e-34, 1.7e-34],
            [1.6e-34, 2.5e-34, 3.4e-34],
            [2.8e-34, 4.0e-34, 5.6e-34],
        ],
        (ImpuritySpecies::Carbon, 6) => [
            [1.1e-34, 1.7e-34, 2.4e-34],
            [2.2e-34, 3.3e-34, 4.7e-34],
            [3.7e-34, 5.4e-34, 7.6e-34],
        ],
        _ => return None,
    };

    Some(PecGrid {
        log_ne: LOG_NE,
        log_te: LOG_TE,
        values,
    })
}

fn bracket(grid: [f64; 3], x: f64) -> (usize, f64) {
    if x <= grid[0] {
        return (0, 0.0);
    }
    if x >= grid[2] {
        return (1, 1.0);
    }
    if x <= grid[1] {
        return (0, (x - grid[0]) / (grid[1] - grid[0]));
    }
    (1, (x - grid[1]) / (grid[2] - grid[1]))
}

fn bilinear_interpolate(grid: PecGrid, log_ne: f64, log_te: f64) -> f64 {
    let (i, tx) = bracket(grid.log_ne, log_ne);
    let (j, ty) = bracket(grid.log_te, log_te);
    let f00 = grid.values[i][j];
    let f10 = grid.values[i + 1][j];
    let f01 = grid.values[i][j + 1];
    let f11 = grid.values[i + 1][j + 1];
    (1.0 - tx) * (1.0 - ty) * f00 + tx * (1.0 - ty) * f10 + (1.0 - tx) * ty * f01 + tx * ty * f11
}

/// Lookup photon emissivity coefficient (PEC) [W m^3] for impurity radiation.
pub fn lookup_pec_w_m3(
    species: ImpuritySpecies,
    charge_state: u8,
    ne_m3: f64,
    te_ev: f64,
) -> Option<f64> {
    if ne_m3 <= 0.0 || te_ev <= 0.0 {
        return None;
    }
    let grid = pec_grid(species, charge_state)?;
    let log_ne = ne_m3.log10();
    let log_te = te_ev.log10();
    Some(bilinear_interpolate(grid, log_ne, log_te))
}

/// Collisional-radiative radiative power density [W/m^3].
pub fn collisional_radiative_power_density_w_m3(
    species: ImpuritySpecies,
    charge_state: u8,
    ne_m3: f64,
    n_imp_m3: f64,
    te_ev: f64,
) -> Option<f64> {
    let pec = lookup_pec_w_m3(species, charge_state, ne_m3, te_ev)?;
    Some((ne_m3 * n_imp_m3 * pec).max(0.0))
}

/// Total impurity radiative power loss [MW] for a plasma volume.
pub fn impurity_radiative_loss_mw(
    species: ImpuritySpecies,
    charge_state: u8,
    ne_m3: f64,
    n_imp_m3: f64,
    te_ev: f64,
    volume_m3: f64,
) -> Option<f64> {
    let p_w_m3 =
        collisional_radiative_power_density_w_m3(species, charge_state, ne_m3, n_imp_m3, te_ev)?;
    Some(p_w_m3 * volume_m3 / 1e6)
}

/// Material DPA limit and damage rate.
#[derive(Debug, Clone)]
pub struct MaterialDPA {
    pub name: &'static str,
    /// Maximum tolerable DPA.
    pub dpa_limit: f64,
    /// DPA per MW/m² per full-power year.
    pub sigma_dpa: f64,
}

impl MaterialDPA {
    pub fn tungsten() -> Self {
        MaterialDPA {
            name: "Tungsten",
            dpa_limit: 50.0,
            sigma_dpa: 1000.0,
        }
    }
    pub fn eurofer() -> Self {
        MaterialDPA {
            name: "Eurofer",
            dpa_limit: 150.0,
            sigma_dpa: 500.0,
        }
    }
    pub fn beryllium() -> Self {
        MaterialDPA {
            name: "Beryllium",
            dpa_limit: 10.0,
            sigma_dpa: 200.0,
        }
    }

    /// Lifespan in full-power years given peak neutron wall load [MW/m²].
    pub fn lifespan_years(&self, peak_load_mw_m2: f64) -> f64 {
        let dpa_per_year = peak_load_mw_m2 * DPA_PER_MW_YEAR;
        if dpa_per_year > 0.0 {
            self.dpa_limit / dpa_per_year
        } else {
            f64::INFINITY
        }
    }
}

/// Helium ash poisoning time-step result.
#[derive(Debug, Clone)]
pub struct AshSnapshot {
    pub time: f64,
    pub p_fus_mw: f64,
    pub f_he: f64,
    pub q_factor: f64,
}

/// Simulate helium ash accumulation in a burning plasma.
///
/// `burn_time_s`: total simulation time.
/// `tau_he_ratio`: τ_He/τ_E ratio (lower = better pumping).
/// `dt`: timestep [s].
/// Returns time history of fusion power and He fraction.
pub fn simulate_ash_poisoning(burn_time_s: f64, tau_he_ratio: f64, dt: f64) -> Vec<AshSnapshot> {
    let tau_he = tau_he_ratio * TAU_E;
    let mut n_he = 0.0_f64;
    let mut history = Vec::new();

    let n_steps = (burn_time_s / dt) as usize;

    for step in 0..=n_steps {
        let t = step as f64 * dt;

        // Fuel density (quasi-neutrality: n_e = n_D + n_T + 2*n_He)
        let n_fuel = (N_E - 2.0 * n_he).max(0.0);
        let n_d = n_fuel / 2.0;
        let n_t = n_fuel / 2.0;

        // Bosch-Hale σv approximation at fixed T
        let sigma_v = bosch_hale_approx(T_KEV);

        // Reaction rate
        let r_fus = n_d * n_t * sigma_v;

        // Fusion power [MW]
        let p_fus = r_fus * E_FUSION_MEV * MEV_TO_J * VOLUME / 1e6;

        let f_he = n_he / N_E;
        let q = if p_fus > 0.0 { p_fus / 50.0 } else { 0.0 }; // Q = P_fus / P_aux (50 MW)

        history.push(AshSnapshot {
            time: t,
            p_fus_mw: p_fus,
            f_he,
            q_factor: q,
        });

        // Update He: dn_He/dt = R_fus - n_He/τ_He
        n_he += dt * (r_fus - n_he / tau_he);
        n_he = n_he.max(0.0);
    }

    history
}

/// DT reaction rate coefficient σv [m³/s] at temperature T [keV].
///
/// Uses the Bosch-Hale (1992) parameterization with corrected coefficients.
fn bosch_hale_approx(t_kev: f64) -> f64 {
    if t_kev <= 0.2 {
        return 0.0;
    }
    // Simplified Bosch-Hale: σv ≈ C · T^{-2/3} · exp(-a / T^{1/3})
    // Calibrated to match: σv(10 keV) ≈ 1.1e-22, σv(20 keV) ≈ 4.2e-22
    let c1 = 3.7e-18; // [m³/s · keV^{2/3}]
    let a1 = 19.94; // [keV^{1/3}]
    let sigma_v = c1 * t_kev.powf(-2.0 / 3.0) * (-a1 / t_kev.powf(1.0 / 3.0)).exp();
    sigma_v.max(0.0)
}

/// D-shaped first wall geometry.
///
/// Returns (R, Z) arrays for wall contour.
pub fn generate_first_wall(
    r0: f64,
    a: f64,
    kappa: f64,
    delta: f64,
    n_points: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut r_wall = Vec::with_capacity(n_points);
    let mut z_wall = Vec::with_capacity(n_points);
    let a_wall = a + 0.5; // Wall offset

    for i in 0..n_points {
        let theta = 2.0 * PI * i as f64 / n_points as f64;
        let r = r0 + a_wall * (theta + (delta).asin() * theta.sin()).cos();
        let z = kappa * a_wall * theta.sin();
        r_wall.push(r);
        z_wall.push(z);
    }

    (r_wall, z_wall)
}

/// NURBS-smoothed D-shaped first wall geometry.
///
/// Returns (R, Z) arrays for a reduced IGA-compatible contour.
pub fn generate_first_wall_nurbs(
    r0: f64,
    a: f64,
    kappa: f64,
    delta: f64,
    n_points: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = n_points.max(8);
    let a_wall = a + 0.5;
    let control_points = vec![
        ControlPoint2D {
            x: r0 + a_wall,
            y: 0.0,
        },
        ControlPoint2D {
            x: r0 + 0.9 * a_wall,
            y: 0.65 * kappa * a_wall,
        },
        ControlPoint2D {
            x: r0 + delta * a_wall,
            y: kappa * a_wall,
        },
        ControlPoint2D {
            x: r0 - 0.55 * a_wall,
            y: 0.65 * kappa * a_wall,
        },
        ControlPoint2D {
            x: r0 - a_wall,
            y: 0.0,
        },
        ControlPoint2D {
            x: r0 - 0.55 * a_wall,
            y: -0.65 * kappa * a_wall,
        },
        ControlPoint2D {
            x: r0 + delta * a_wall,
            y: -kappa * a_wall,
        },
        ControlPoint2D {
            x: r0 + 0.9 * a_wall,
            y: -0.65 * kappa * a_wall,
        },
        ControlPoint2D {
            x: r0 + a_wall,
            y: 0.0,
        },
        ControlPoint2D {
            x: r0 + 0.9 * a_wall,
            y: 0.65 * kappa * a_wall,
        },
    ];

    let weights = vec![1.0; control_points.len()];
    let knots = open_uniform_knots(control_points.len(), 3);
    let curve =
        NurbsCurve2D::new(3, knots, control_points, weights).expect("valid first-wall NURBS");
    let mut sampled = curve.sample_uniform(n);
    if let Some(first) = sampled.first().copied() {
        let last_index = sampled.len() - 1;
        sampled[last_index] = first;
    }
    let mut r_wall = Vec::with_capacity(sampled.len());
    let mut z_wall = Vec::with_capacity(sampled.len());
    for point in sampled {
        r_wall.push(point.x);
        z_wall.push(point.y);
    }
    (r_wall, z_wall)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bosch_hale_positive() {
        let sv = bosch_hale_approx(20.0);
        assert!(sv > 0.0, "σv should be positive at 20 keV: {sv}");
        assert!(sv.is_finite(), "σv should be finite: {sv}");
    }

    #[test]
    fn test_ash_good_pumping() {
        let history = simulate_ash_poisoning(100.0, 5.0, 0.1);
        let last = history.last().unwrap();
        assert!(
            last.p_fus_mw > 100.0,
            "Good pumping should sustain fusion: {} MW",
            last.p_fus_mw
        );
        assert!(
            last.f_he < 0.3,
            "He fraction should stay moderate: {}",
            last.f_he
        );
    }

    #[test]
    fn test_ash_bad_pumping_reduces_power() {
        let good = simulate_ash_poisoning(200.0, 5.0, 0.1);
        let bad = simulate_ash_poisoning(200.0, 50.0, 0.1);
        let p_good = good.last().unwrap().p_fus_mw;
        let p_bad = bad.last().unwrap().p_fus_mw;
        assert!(
            p_bad < p_good,
            "Bad pumping should reduce power: {p_bad} vs {p_good}"
        );
    }

    #[test]
    fn test_material_lifespan() {
        let w = MaterialDPA::tungsten();
        let life = w.lifespan_years(1.0);
        assert!(
            life > 0.0 && life < 100.0,
            "Tungsten lifespan at 1 MW/m² should be reasonable: {life} years"
        );
    }

    #[test]
    fn test_first_wall_geometry() {
        let (r, z) = generate_first_wall(5.0, 3.0, 1.9, 0.4, 200);
        assert_eq!(r.len(), 200);
        assert!(r.iter().all(|v| v.is_finite()));
        assert!(z.iter().all(|v| v.is_finite()));
        // Wall should surround the axis
        let r_min = r.iter().copied().fold(f64::INFINITY, f64::min);
        let r_max = r.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(r_min > 0.0, "R_min should be positive: {r_min}");
        assert!(r_max > r_min, "Wall should have radial extent");
    }

    #[test]
    fn test_first_wall_nurbs_geometry() {
        let (r, z) = generate_first_wall_nurbs(5.0, 3.0, 1.9, 0.4, 200);
        assert_eq!(r.len(), 200);
        assert!(r.iter().all(|v| v.is_finite()));
        assert!(z.iter().all(|v| v.is_finite()));

        let closure = ((r[0] - r[r.len() - 1]).powi(2) + (z[0] - z[z.len() - 1]).powi(2)).sqrt();
        assert!(closure < 1e-9, "NURBS wall should be explicitly closed");
    }

    #[test]
    fn test_first_wall_nurbs_matches_analytic_envelope() {
        let (r_a, z_a) = generate_first_wall(5.0, 3.0, 1.9, 0.4, 200);
        let (r_n, z_n) = generate_first_wall_nurbs(5.0, 3.0, 1.9, 0.4, 200);

        let r_a_min = r_a.iter().copied().fold(f64::INFINITY, f64::min);
        let r_a_max = r_a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let r_n_min = r_n.iter().copied().fold(f64::INFINITY, f64::min);
        let r_n_max = r_n.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let z_a_max = z_a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let z_n_max = z_n.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        assert!((r_n_min - r_a_min).abs() < 1.0);
        assert!((r_n_max - r_a_max).abs() < 1.0);
        assert!((z_n_max - z_a_max).abs() < 1.0);
    }

    #[test]
    fn test_pec_lookup_distinguishes_tungsten_charge_states() {
        let ne = 5.0e20;
        let te = 2_000.0;
        let w20 = lookup_pec_w_m3(ImpuritySpecies::Tungsten, 20, ne, te).unwrap();
        let w40 = lookup_pec_w_m3(ImpuritySpecies::Tungsten, 40, ne, te).unwrap();
        assert!(
            w40 > w20,
            "Expected W40+ PEC > W20+ PEC at same plasma state, got w40={w40}, w20={w20}"
        );
    }

    #[test]
    fn test_collisional_radiative_power_density_positive() {
        let p = collisional_radiative_power_density_w_m3(
            ImpuritySpecies::Tungsten,
            40,
            1.0e20,
            3.0e16,
            3_000.0,
        )
        .unwrap();
        assert!(
            p.is_finite() && p > 0.0,
            "Expected positive finite CR power density"
        );

        let p_mw = impurity_radiative_loss_mw(
            ImpuritySpecies::Tungsten,
            40,
            1.0e20,
            3.0e16,
            3_000.0,
            800.0,
        )
        .unwrap();
        assert!(p_mw > 0.0, "Expected positive CR radiative loss");
    }

    #[test]
    fn test_lookup_returns_none_for_unsupported_charge_state() {
        let out = lookup_pec_w_m3(ImpuritySpecies::Tungsten, 99, 1.0e20, 1_000.0);
        assert!(out.is_none());
    }
}
