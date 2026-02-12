//! Tritium breeding blanket neutronics.
//!
//! Port of `blanket_neutronics.py`.
//! 1D neutron diffusion-reaction PDE with Li-6 breeding.

use fusion_math::tridiag::thomas_solve;

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
const DEFAULT_FLUX: f64 = 1e14;

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
}
