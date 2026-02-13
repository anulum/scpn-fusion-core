// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Neutronics
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Tritium breeding blanket neutronics.
//!
//! Port of `blanket_neutronics.py`.
//! 1D neutron diffusion-reaction PDE with Li-6 breeding.

use fusion_math::tridiag::thomas_solve;
use std::f64::consts::PI;

/// Default blanket thickness [cm]. Python: 100.
const DEFAULT_THICKNESS_CM: f64 = 100.0;

/// Grid points. Python: 100.
const POINTS: usize = 100;

/// Li-6 capture cross-section [1/cm]. Python: Sigma_capture_Li6=0.15.
const SIGMA_LI6: f64 = 0.15;

/// Scattering cross-section [1/cm]. Python: 0.2.
const SIGMA_SCATTER: f64 = 0.2;

/// Parasitic absorption [1/cm]. Python: 0.02.
const SIGMA_PARASITIC: f64 = 0.02;

/// (n,2n) multiplier cross-section [1/cm]. Python: 0.08.
const SIGMA_MULTIPLY: f64 = 0.08;

/// Multiplier gain. Python: 1.8.
const MULTIPLIER_GAIN: f64 = 1.8;

/// Default incident flux [n/(cm²·s)]. Python: 1e14.
#[cfg(test)]
const DEFAULT_FLUX: f64 = 1e14;
/// Reference slab thickness used by reduced volumetric surrogate [cm].
const VOLUMETRIC_REFERENCE_THICKNESS_CM: f64 = 80.0;

/// Result of a breeding blanket calculation.
#[derive(Debug, Clone)]
pub struct BlanketResult {
    /// Neutron flux profile.
    pub flux: Vec<f64>,
    /// Tritium production rate profile.
    pub production: Vec<f64>,
    /// Tritium Breeding Ratio.
    pub tbr: f64,
}

/// Configuration for reduced 3D volumetric blanket surrogate.
#[derive(Debug, Clone, Copy)]
pub struct VolumetricBlanketConfig {
    /// Tokamak major radius [m].
    pub major_radius_m: f64,
    /// Plasma minor radius [m].
    pub minor_radius_m: f64,
    /// Elongation factor.
    pub elongation: f64,
    /// Radial discretization through blanket thickness.
    pub radial_cells: usize,
    /// Poloidal discretization.
    pub poloidal_cells: usize,
    /// Toroidal discretization.
    pub toroidal_cells: usize,
    /// Incident neutron flux [n/(cm²·s)] at first wall.
    pub incident_flux: f64,
}

impl Default for VolumetricBlanketConfig {
    fn default() -> Self {
        Self {
            major_radius_m: 6.2,
            minor_radius_m: 2.0,
            elongation: 1.7,
            radial_cells: 24,
            poloidal_cells: 72,
            toroidal_cells: 48,
            incident_flux: 1e14,
        }
    }
}

/// Reduced 3D blanket surrogate summary.
#[derive(Debug, Clone)]
pub struct VolumetricBlanketResult {
    /// Volumetric Tritium Breeding Ratio.
    pub tbr: f64,
    /// Total tritium production [1/s].
    pub total_production_per_s: f64,
    /// Total incident neutrons [1/s].
    pub incident_neutrons_per_s: f64,
    /// Integrated blanket volume [m^3].
    pub blanket_volume_m3: f64,
}

/// 1D breeding blanket model.
pub struct BreedingBlanket {
    /// Blanket thickness [cm].
    pub thickness: f64,
    /// Grid points.
    pub points: usize,
    /// Li-6 enrichment factor (0-1, applied to SIGMA_LI6).
    pub enrichment: f64,
    /// Grid spacing [cm].
    dx: f64,
}

impl BreedingBlanket {
    /// Create with thickness in cm and enrichment fraction (0-1).
    pub fn new(thickness_cm: f64, enrichment: f64) -> Self {
        let dx = thickness_cm / (POINTS - 1) as f64;
        BreedingBlanket {
            thickness: thickness_cm,
            points: POINTS,
            enrichment: enrichment.clamp(0.0, 1.0),
            dx,
        }
    }

    /// Solve steady-state neutron transport. Returns flux profile and TBR.
    ///
    /// Two-group approach: fast neutrons diffuse and multiply in Be/Pb,
    /// then breed tritium in Li-6.
    pub fn solve_transport(&self, incident_flux: f64) -> BlanketResult {
        let n = self.points;
        let dx = self.dx;
        let sigma_li6 = SIGMA_LI6 * self.enrichment;

        // Total absorption (removes neutrons from the system)
        let sigma_abs = sigma_li6 + SIGMA_PARASITIC;

        // Neutron multiplication source: each (n,2n) adds (gain-1) extra neutrons
        let nu_sigma_f = SIGMA_MULTIPLY * (MULTIPLIER_GAIN - 1.0);

        // Net removal = absorption - multiplication source
        // This can be very small or negative for a well-designed blanket
        let sigma_net = sigma_abs - nu_sigma_f;

        // Transport cross-section for diffusion coefficient
        let sigma_tr = sigma_abs + SIGMA_SCATTER + SIGMA_MULTIPLY;
        let d = 1.0 / (3.0 * sigma_tr);

        // Build tridiagonal system: D·∇²Φ - σ_net·Φ = 0
        let interior = n - 2;
        let coeff = d / (dx * dx);

        let mut a_sub = vec![0.0; interior];
        let mut b_main = vec![0.0; interior];
        let mut c_sup = vec![0.0; interior];
        let mut d_rhs = vec![0.0; interior];

        for j in 0..interior {
            b_main[j] = 2.0 * coeff + sigma_net;
            if j > 0 {
                a_sub[j] = -coeff;
            }
            if j < interior - 1 {
                c_sup[j] = -coeff;
            }
        }

        // BC: Φ[0] = incident_flux (Dirichlet at first wall)
        d_rhs[0] += coeff * incident_flux;

        // BC: Φ[n-1] = 0 (vacuum at shield edge)

        let x = thomas_solve(&a_sub, &b_main, &c_sup, &d_rhs);

        let mut flux = vec![0.0; n];
        flux[0] = incident_flux;
        flux[1..(interior + 1)].copy_from_slice(&x[..interior]);
        flux[n - 1] = 0.0;

        // Tritium production: rate = σ_Li6 · Φ(x)
        let production: Vec<f64> = flux.iter().map(|&phi| sigma_li6 * phi).collect();

        // Trapezoidal integration of production
        let mut total_production = 0.0;
        for i in 0..n - 1 {
            total_production += 0.5 * (production[i] + production[i + 1]) * dx;
        }

        // TBR = total tritium production per incident 14 MeV neutron
        // Each incident neutron produces (gain) fast neutrons via (n,2n)
        // The 14 MeV neutron also directly breeds
        // TBR = ∫ σ_Li6·Φ dx / (incident_current)
        // With incident current = Φ[0] * v/4 ≈ Φ[0] (normalized)
        // Use D·|dΦ/dx|_{x=0} as the actual current entering the blanket
        let j_in_diffusion = d * (flux[0] - flux[1]).abs() / dx;
        let j_in = j_in_diffusion.max(incident_flux * 0.01); // Avoid division by zero
        let tbr = total_production / j_in;

        BlanketResult {
            flux,
            production,
            tbr,
        }
    }

    /// Reduced 3D volumetric blanket surrogate built on the 1D depth profile.
    ///
    /// The method uses:
    /// - 1D attenuation from `solve_transport` mapped to radial blanket depth,
    /// - shaped toroidal shell geometry,
    /// - deterministic poloidal/toroidal weighting for first-order asymmetry.
    pub fn solve_volumetric_surrogate(
        &self,
        config: VolumetricBlanketConfig,
    ) -> VolumetricBlanketResult {
        let major_radius_m = config.major_radius_m.max(0.1);
        let minor_radius_m = config.minor_radius_m.max(0.05);
        let elongation = config.elongation.max(0.1);
        let radial_cells = config.radial_cells.max(2);
        let poloidal_cells = config.poloidal_cells.max(8);
        let toroidal_cells = config.toroidal_cells.max(8);
        let incident_flux = config.incident_flux.max(1.0);

        // Use a nominal enriched reference profile for a stable reduced 3D surrogate.
        let profile =
            BreedingBlanket::new(VOLUMETRIC_REFERENCE_THICKNESS_CM, 0.9).solve_transport(
                incident_flux,
            );
        let sigma_li6 = SIGMA_LI6 * self.enrichment;

        let thickness_m = (self.thickness * 0.01).max(1e-9);
        let dr = thickness_m / radial_cells as f64;
        let dtheta = 2.0 * PI / poloidal_cells as f64;
        let dphi = 2.0 * PI / toroidal_cells as f64;

        let mut total_production = 0.0;
        let mut blanket_volume_m3 = 0.0;

        for i in 0..radial_cells {
            let depth_m = (i as f64 + 0.5) * dr;
            let depth_norm = depth_m / thickness_m;
            let base_flux = self.interpolate_flux_normalized(depth_norm, &profile.flux);
            let shell_r = minor_radius_m + depth_m;

            for j in 0..poloidal_cells {
                let theta = (j as f64 + 0.5) * dtheta;
                let incidence_factor = (0.6 + 0.4 * theta.cos().powi(2)).max(0.2);
                let major_local = (major_radius_m + shell_r * theta.cos()).max(0.1);

                for k in 0..toroidal_cells {
                    let phi = (k as f64 + 0.5) * dphi;
                    let toroidal_factor = 1.0 + 0.05 * phi.cos();
                    let local_flux = base_flux * incidence_factor * toroidal_factor;
                    let production_density = sigma_li6 * local_flux; // [1/cm^3/s]

                    let dvol_m3 = elongation * shell_r * dr * dtheta * dphi * major_local;
                    blanket_volume_m3 += dvol_m3;
                    total_production += production_density * dvol_m3 * 1e6; // m^3 -> cm^3
                }
            }
        }

        let first_wall_area_m2 = 4.0 * PI * PI * major_radius_m * minor_radius_m * elongation;
        let incident_neutrons_per_s = incident_flux * first_wall_area_m2 * 1e4; // m^2 -> cm^2
        let tbr = total_production / incident_neutrons_per_s.max(1e-9);

        VolumetricBlanketResult {
            tbr,
            total_production_per_s: total_production,
            incident_neutrons_per_s,
            blanket_volume_m3,
        }
    }

    fn interpolate_flux_normalized(&self, depth_norm: f64, flux: &[f64]) -> f64 {
        if flux.is_empty() {
            return 0.0;
        }
        if flux.len() == 1 {
            return flux[0];
        }

        let idx_f = depth_norm.clamp(0.0, 1.0) * (flux.len() - 1) as f64;
        let idx0 = idx_f.floor() as usize;
        let idx1 = (idx0 + 1).min(flux.len() - 1);
        let w = idx_f - idx0 as f64;
        flux[idx0] * (1.0 - w) + flux[idx1] * w
    }
}

impl Default for BreedingBlanket {
    fn default() -> Self {
        Self::new(DEFAULT_THICKNESS_CM, 0.9)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blanket_creation() {
        let b = BreedingBlanket::new(80.0, 0.9);
        assert_eq!(b.points, POINTS);
        assert!((b.thickness - 80.0).abs() < 1e-10);
    }

    #[test]
    fn test_tbr_above_unity() {
        let blanket = BreedingBlanket::new(80.0, 0.9);
        let result = blanket.solve_transport(DEFAULT_FLUX);
        assert!(
            result.tbr > 1.05,
            "TBR must exceed 1.05 for tritium self-sufficiency: {}",
            result.tbr
        );
    }

    #[test]
    fn test_flux_monotone() {
        let blanket = BreedingBlanket::default();
        let result = blanket.solve_transport(DEFAULT_FLUX);
        // Flux should be non-negative everywhere
        for (i, &phi) in result.flux.iter().enumerate() {
            assert!(phi >= 0.0, "Flux should be non-negative: φ[{i}]={phi}");
        }
    }

    #[test]
    fn test_thicker_blanket_higher_tbr() {
        let thin = BreedingBlanket::new(40.0, 0.9);
        let thick = BreedingBlanket::new(100.0, 0.9);
        let r_thin = thin.solve_transport(DEFAULT_FLUX);
        let r_thick = thick.solve_transport(DEFAULT_FLUX);
        assert!(
            r_thick.tbr > r_thin.tbr,
            "Thicker blanket should breed more: {} vs {}",
            r_thick.tbr,
            r_thin.tbr
        );
    }

    #[test]
    fn test_tbr_finite() {
        let blanket = BreedingBlanket::default();
        let result = blanket.solve_transport(DEFAULT_FLUX);
        assert!(result.tbr.is_finite(), "TBR must be finite: {}", result.tbr);
        assert!(result.tbr > 0.0, "TBR must be positive: {}", result.tbr);
    }

    #[test]
    fn test_volumetric_surrogate_finite() {
        let blanket = BreedingBlanket::new(80.0, 0.9);
        let result = blanket.solve_volumetric_surrogate(VolumetricBlanketConfig {
            radial_cells: 10,
            poloidal_cells: 20,
            toroidal_cells: 16,
            ..VolumetricBlanketConfig::default()
        });
        assert!(result.tbr.is_finite(), "Volumetric TBR must be finite");
        assert!(result.tbr > 0.0, "Volumetric TBR must be positive");
        assert!(
            result.total_production_per_s > 0.0,
            "Production must be positive"
        );
        assert!(
            result.incident_neutrons_per_s > 0.0,
            "Incident neutrons must be positive"
        );
        assert!(
            result.blanket_volume_m3 > 0.0,
            "Blanket volume must be positive"
        );
    }

    #[test]
    fn test_volumetric_tbr_increases_with_thickness() {
        let thin = BreedingBlanket::new(40.0, 0.9);
        let thick = BreedingBlanket::new(100.0, 0.9);
        let cfg = VolumetricBlanketConfig {
            radial_cells: 8,
            poloidal_cells: 16,
            toroidal_cells: 12,
            ..VolumetricBlanketConfig::default()
        };
        let thin_result = thin.solve_volumetric_surrogate(cfg);
        let thick_result = thick.solve_volumetric_surrogate(cfg);
        assert!(
            thick_result.tbr > thin_result.tbr,
            "Thicker blanket should improve volumetric TBR: {} vs {}",
            thick_result.tbr,
            thin_result.tbr
        );
    }

    #[test]
    fn test_volumetric_tbr_increases_with_enrichment() {
        let low = BreedingBlanket::new(80.0, 0.5);
        let high = BreedingBlanket::new(80.0, 0.95);
        let cfg = VolumetricBlanketConfig {
            radial_cells: 8,
            poloidal_cells: 16,
            toroidal_cells: 12,
            ..VolumetricBlanketConfig::default()
        };
        let low_result = low.solve_volumetric_surrogate(cfg);
        let high_result = high.solve_volumetric_surrogate(cfg);
        assert!(
            high_result.tbr > low_result.tbr,
            "Higher Li-6 enrichment should improve volumetric TBR: {} vs {}",
            high_result.tbr,
            low_result.tbr
        );
    }
}
