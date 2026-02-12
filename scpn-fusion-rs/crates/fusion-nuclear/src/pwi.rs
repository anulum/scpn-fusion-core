// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — PWI
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Plasma-Wall Interaction: Eckstein-Bohdansky sputtering model.
//!
//! Port of `pwi_erosion.py`.
//! Calculates sputtering yield, erosion rate, and material lifetime.

/// Avogadro's number [1/mol].
const _AVOGADRO: f64 = 6.022e23;

/// Atomic mass unit [kg].
const AMU_KG: f64 = 1.66e-27;

/// Seconds per year.
const SECONDS_PER_YEAR: f64 = 31_536_000.0;

/// Redeposition fraction (95% returns to wall).
const REDEPOSITION: f64 = 0.95;

/// Material-specific sputtering parameters.
#[derive(Debug, Clone)]
pub struct MaterialParams {
    /// Sputtering energy threshold [eV].
    pub e_th: f64,
    /// Yield factor Q.
    pub q: f64,
    /// Atomic mass [amu].
    pub atomic_mass: f64,
    /// Density [g/cm³].
    pub density: f64,
    /// Material name.
    pub name: &'static str,
}

impl MaterialParams {
    /// Tungsten (W) parameters.
    pub fn tungsten() -> Self {
        MaterialParams {
            e_th: 200.0,
            q: 0.03,
            atomic_mass: 183.84,
            density: 19.25,
            name: "Tungsten",
        }
    }

    /// Carbon (C) parameters.
    pub fn carbon() -> Self {
        MaterialParams {
            e_th: 30.0,
            q: 0.1,
            atomic_mass: 12.0,
            density: 2.2,
            name: "Carbon",
        }
    }
}

/// Result of an erosion calculation.
#[derive(Debug, Clone)]
pub struct ErosionResult {
    /// Sputtering yield [atoms/ion].
    pub yield_val: f64,
    /// Impact energy [eV].
    pub e_impact: f64,
    /// Net erosion flux [atoms/(m²·s)].
    pub net_flux: f64,
    /// Erosion rate [mm/year].
    pub erosion_mm_year: f64,
}

/// Sputtering physics calculator.
pub struct SputteringPhysics {
    pub params: MaterialParams,
}

impl SputteringPhysics {
    pub fn new(params: MaterialParams) -> Self {
        SputteringPhysics { params }
    }

    pub fn tungsten() -> Self {
        Self::new(MaterialParams::tungsten())
    }

    pub fn carbon() -> Self {
        Self::new(MaterialParams::carbon())
    }

    /// Eckstein-Bohdansky sputtering yield.
    ///
    /// `e_ion_ev`: ion energy [eV], `angle_deg`: incidence angle [degrees].
    pub fn calculate_yield(&self, e_ion_ev: f64, angle_deg: f64) -> f64 {
        if e_ion_ev < self.params.e_th {
            return 0.0;
        }
        let reduced_e = e_ion_ev / self.params.e_th;
        let f_e = reduced_e.ln() * (reduced_e - 1.0) / (reduced_e * reduced_e + 1.0);
        let f_a = 1.0 + (angle_deg / 90.0).powi(2);
        (self.params.q * f_e * f_a).max(0.0)
    }

    /// Calculate erosion rate for given particle flux and ion temperature.
    ///
    /// `flux`: [particles/(m²·s)], `t_ion_ev`: ion temperature [eV].
    pub fn calculate_erosion_rate(&self, flux: f64, t_ion_ev: f64) -> ErosionResult {
        let e_impact = 5.0 * t_ion_ev;
        let y = self.calculate_yield(e_impact, 45.0);
        let gross_flux = flux * y;
        let net_flux = gross_flux * (1.0 - REDEPOSITION);

        // Recession speed [m/s]
        let v = net_flux * (self.params.atomic_mass * AMU_KG) / (self.params.density * 1000.0);
        let erosion_mm_year = v * 1000.0 * SECONDS_PER_YEAR;

        ErosionResult {
            yield_val: y,
            e_impact,
            net_flux,
            erosion_mm_year,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sputtering_threshold() {
        let physics = SputteringPhysics::tungsten();
        let y = physics.calculate_yield(100.0, 0.0);
        assert!(y < 1e-10, "No sputtering below threshold energy: {y}");
    }

    #[test]
    fn test_sputtering_above_threshold() {
        let physics = SputteringPhysics::tungsten();
        let y = physics.calculate_yield(500.0, 45.0);
        assert!(y > 0.0, "Should sputter above threshold: {y}");
    }

    #[test]
    fn test_yield_increases_with_energy() {
        let physics = SputteringPhysics::tungsten();
        let y_low = physics.calculate_yield(300.0, 45.0);
        let y_high = physics.calculate_yield(1000.0, 45.0);
        assert!(
            y_high > y_low,
            "Yield should increase with energy: {y_high} vs {y_low}"
        );
    }

    #[test]
    fn test_erosion_rate_positive() {
        let physics = SputteringPhysics::tungsten();
        let result = physics.calculate_erosion_rate(1e24, 50.0);
        assert!(
            result.erosion_mm_year > 0.0,
            "Erosion rate should be positive: {}",
            result.erosion_mm_year
        );
        assert!(
            result.erosion_mm_year.is_finite(),
            "Erosion rate must be finite"
        );
    }

    #[test]
    fn test_carbon_erodes_faster() {
        let w = SputteringPhysics::tungsten();
        let c = SputteringPhysics::carbon();
        let r_w = w.calculate_erosion_rate(1e24, 50.0);
        let r_c = c.calculate_erosion_rate(1e24, 50.0);
        assert!(
            r_c.yield_val > r_w.yield_val,
            "Carbon should have higher yield: {} vs {}",
            r_c.yield_val,
            r_w.yield_val
        );
    }
}
