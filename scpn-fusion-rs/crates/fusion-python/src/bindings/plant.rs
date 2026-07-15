// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
//! PyO3 bindings for the reactor plant-model lane (`fusion-engineering`).
//!
//! Exposes the whole-plant engineering economics model (`PyPlantModel`):
//! tritium breeding ratio, neutron wall loading, ARIES cost scaling, cost of
//! electricity, and the major-radius design scan, mirroring the NumPy tier's API.

use pyo3::prelude::*;

use fusion_engineering::blanket::neutron_wall_loading;
use fusion_engineering::layout::{
    aries_cost_scaling, cost_of_electricity as engineering_coe, scan_major_radius,
};
use fusion_engineering::tritium::tritium_breeding_ratio;

#[pyclass]
pub(crate) struct PyPlantModel;

#[pymethods]
impl PyPlantModel {
    #[new]
    fn new() -> Self {
        Self
    }

    fn tritium_breeding_ratio(
        &self,
        n_li6: f64,
        sigma_li6: f64,
        neutron_flux: f64,
        blanket_vol: f64,
    ) -> f64 {
        tritium_breeding_ratio(n_li6, sigma_li6, neutron_flux, blanket_vol)
    }

    fn wall_loading(&self, p_neutron: f64, r: f64, a: f64, kappa: f64) -> f64 {
        neutron_wall_loading(p_neutron, r, a, kappa)
    }

    fn aries_cost_scaling(&self, c0: f64, r: f64, b: f64) -> f64 {
        aries_cost_scaling(c0, r, b)
    }

    fn cost_of_electricity(&self, capital_annuity: f64, o_and_m: f64, p_net: f64, cf: f64) -> f64 {
        engineering_coe(capital_annuity, o_and_m, p_net, cf)
    }

    fn scan_radius(
        &self,
        r_min: f64,
        r_max: f64,
        steps: usize,
    ) -> Vec<(f64, f64, f64, f64, f64, f64)> {
        scan_major_radius(r_min, r_max, steps)
            .into_iter()
            .map(|d| {
                (
                    d.r_major,
                    d.b_field,
                    d.p_net,
                    d.capacity_factor,
                    d.capital_cost,
                    d.coe,
                )
            })
            .collect()
    }
}
