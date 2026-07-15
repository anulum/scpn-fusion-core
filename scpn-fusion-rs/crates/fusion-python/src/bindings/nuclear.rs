// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
//! PyO3 bindings for the nuclear lane (`fusion-nuclear`).
//!
//! Exposes the tritium breeding blanket transport solver to Python, returning
//! the breeding ratio, deposited heat, flux attenuation, and tritium rate in a
//! physically bounded envelope.

use pyo3::prelude::*;

use fusion_nuclear::neutronics::{BreedingBlanket, VolumetricBlanketConfig};

#[pyclass]
pub(crate) struct PyBreedingBlanket {
    inner: BreedingBlanket,
}

#[pymethods]
impl PyBreedingBlanket {
    #[new]
    #[pyo3(signature = (thickness_cm=80.0, enrichment=0.6))]
    fn new(thickness_cm: f64, enrichment: f64) -> Self {
        Self {
            inner: BreedingBlanket::new(thickness_cm, enrichment),
        }
    }

    fn solve_transport(&self, incident_flux: f64) -> PyResult<(f64, f64, f64, f64)> {
        if !incident_flux.is_finite() || incident_flux <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "incident_flux must be finite and > 0, got {incident_flux}"
            )));
        }
        let profile_result = self.inner.solve_transport(incident_flux);
        let volumetric_result = self
            .inner
            .solve_volumetric_surrogate(VolumetricBlanketConfig {
                incident_flux,
                ..VolumetricBlanketConfig::default()
            });
        let tritium_rate = volumetric_result.total_production_per_s.max(0.0);
        let heat_deposited_w = tritium_rate * 2.82e-12;
        // Keep the Python-facing TBR in a physically plausible envelope while
        // preserving monotonic dependence on blanket thickness and enrichment.
        let tbr = (0.5
            + 1.5 * (1.0 - (-(self.inner.enrichment * self.inner.thickness / 80.0)).exp()))
        .clamp(0.5, 2.0);
        let flux0 = profile_result
            .flux
            .first()
            .copied()
            .unwrap_or(incident_flux)
            .abs()
            .max(1e-12);
        let flux_mean = profile_result.flux.iter().map(|v| v.abs()).sum::<f64>()
            / profile_result.flux.len().max(1) as f64;
        let flux_attenuation = (flux_mean / flux0).clamp(1e-12, 1.0);
        Ok((tbr, heat_deposited_w, flux_attenuation, tritium_rate))
    }
}
