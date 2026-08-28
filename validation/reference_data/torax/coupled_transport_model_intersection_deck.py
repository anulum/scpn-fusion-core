# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Coupled TORAX Model-Intersection Deck
"""Frozen circular-geometry deck shared by TORAX and native transport.

The deck uses only the declared intersection available in both codes: constant
radial transport, Gaussian heat/particle/current sources, collisional
ion-electron exchange, prescribed edge values, and circular ITER-scale geometry.
It is not a full-fidelity ITER scenario and carries no performance or superiority
claim.
"""

from __future__ import annotations

from typing import Any, cast

MODEL_INTERSECTION: dict[str, Any] = {
    "schema": "scpn-fusion-core.coupled-transport-model-intersection.v1",
    "geometry": {
        "name": "elongation-one circular ITER-scale comparison geometry",
        "major_radius_m": 6.2,
        "minor_radius_m": 2.0,
        "magnetic_field_t": 5.3,
        "elongation_lcfs": 1.0,
        "torax_cells": 16,
        "comparison_points": 17,
    },
    "time": {
        "initial_s": 0.0,
        "final_s": 0.02,
        "primary_dt_s": 0.01,
        "refined_dt_s": 0.005,
    },
    "profiles": {
        "ion_temperature_core_kev": 5.0,
        "ion_temperature_edge_initial_kev": 0.5,
        "ion_temperature_edge_final_kev": 0.6,
        "electron_temperature_core_kev": 4.0,
        "electron_temperature_edge_initial_kev": 0.4,
        "electron_temperature_edge_final_kev": 0.5,
        "electron_density_core_m3": 5.0e19,
        "electron_density_edge_initial_m3": 2.0e19,
        "electron_density_edge_final_m3": 2.1e19,
        "plasma_current_a": 5.0e6,
        "effective_charge": 1.5,
    },
    "transport": {
        "ion_heat_diffusivity_m2_s": 1.0,
        "electron_heat_diffusivity_m2_s": 1.0,
        "electron_particle_diffusivity_m2_s": 0.3,
        "electron_particle_convection_m_s": 0.0,
        "resistivity_multiplier": 20.0,
        "native_exchange_rate_s": 1.0,
        "torax_exchange_multiplier": 1.0,
    },
    "sources": {
        "heat_power_w": 1.0e7,
        "electron_heat_fraction": 0.6,
        "heat_center_rho": 0.3,
        "heat_width_rho": 0.25,
        "particle_rate_s": 1.0e20,
        "particle_center_rho": 0.45,
        "particle_width_rho": 0.25,
        "driven_current_a": 1.0e5,
        "current_center_rho": 0.35,
        "current_width_rho": 0.2,
    },
    "thresholds": {
        "initial_profile_relative_l2": 0.04,
        "ion_temperature_final_relative_l2": 0.12,
        "electron_temperature_final_relative_l2": 0.12,
        "electron_density_final_relative_l2": 0.08,
        "poloidal_flux_final_relative_l2": 0.20,
        "source_total_relative_error": 5.0e-3,
        "native_linear_residual_linf": 1.0e-8,
        "native_exchange_relative_closure": 1.0e-12,
        "native_refinement_relative_l2": 0.02,
        "torax_refinement_relative_l2": 0.02,
        "maximum_warm_cost_ratio": 100.0,
    },
}

_GEOMETRY = cast(dict[str, Any], MODEL_INTERSECTION["geometry"])
_TIME = cast(dict[str, Any], MODEL_INTERSECTION["time"])
_PROFILES = cast(dict[str, Any], MODEL_INTERSECTION["profiles"])
_TRANSPORT = cast(dict[str, Any], MODEL_INTERSECTION["transport"])
_SOURCES = cast(dict[str, Any], MODEL_INTERSECTION["sources"])

CONFIG: dict[str, Any] = {
    "profile_conditions": {
        "Ip": _PROFILES["plasma_current_a"],
        "initial_psi_mode": "j",
        "current_profile_nu": 1.0,
        "T_i": {
            0.0: {
                0.0: _PROFILES["ion_temperature_core_kev"],
                1.0: _PROFILES["ion_temperature_edge_initial_kev"],
            }
        },
        "T_i_right_bc": {
            _TIME["initial_s"]: _PROFILES["ion_temperature_edge_initial_kev"],
            _TIME["final_s"]: _PROFILES["ion_temperature_edge_final_kev"],
        },
        "T_e": {
            0.0: {
                0.0: _PROFILES["electron_temperature_core_kev"],
                1.0: _PROFILES["electron_temperature_edge_initial_kev"],
            }
        },
        "T_e_right_bc": {
            _TIME["initial_s"]: _PROFILES["electron_temperature_edge_initial_kev"],
            _TIME["final_s"]: _PROFILES["electron_temperature_edge_final_kev"],
        },
        "n_e": {
            0.0: {
                0.0: _PROFILES["electron_density_core_m3"],
                1.0: _PROFILES["electron_density_edge_initial_m3"],
            }
        },
        "n_e_right_bc": {
            _TIME["initial_s"]: _PROFILES["electron_density_edge_initial_m3"],
            _TIME["final_s"]: _PROFILES["electron_density_edge_final_m3"],
        },
        "n_e_nbar_is_fGW": False,
        "normalize_n_e_to_nbar": False,
    },
    "plasma_composition": {
        "main_ion": "D",
        "impurity": "Ne",
        "Z_eff": _PROFILES["effective_charge"],
    },
    "numerics": {
        "t_initial": _TIME["initial_s"],
        "t_final": _TIME["final_s"],
        "fixed_dt": _TIME["primary_dt_s"],
        "evolve_ion_heat": True,
        "evolve_electron_heat": True,
        "evolve_current": True,
        "evolve_density": True,
        "resistivity_multiplier": _TRANSPORT["resistivity_multiplier"],
        "adaptive_dt": False,
    },
    "geometry": {
        "geometry_type": "circular",
        "n_rho": _GEOMETRY["torax_cells"],
        "R_major": _GEOMETRY["major_radius_m"],
        "a_minor": _GEOMETRY["minor_radius_m"],
        "B_0": _GEOMETRY["magnetic_field_t"],
        "elongation_LCFS": _GEOMETRY["elongation_lcfs"],
    },
    "neoclassical": {"bootstrap_current": {"bootstrap_multiplier": 0.0}},
    "sources": {
        "generic_current": {
            "I_generic": _SOURCES["driven_current_a"],
            "use_absolute_current": True,
            "gaussian_width": _SOURCES["current_width_rho"],
            "gaussian_location": _SOURCES["current_center_rho"],
        },
        "generic_particle": {
            "S_total": _SOURCES["particle_rate_s"],
            "particle_width": _SOURCES["particle_width_rho"],
            "deposition_location": _SOURCES["particle_center_rho"],
        },
        "generic_heat": {
            "P_total": _SOURCES["heat_power_w"],
            "electron_heat_fraction": _SOURCES["electron_heat_fraction"],
            "gaussian_width": _SOURCES["heat_width_rho"],
            "gaussian_location": _SOURCES["heat_center_rho"],
        },
        "ei_exchange": {"Qei_multiplier": _TRANSPORT["torax_exchange_multiplier"]},
    },
    "pedestal": {"set_pedestal": False},
    "transport": {
        "model_name": "constant",
        "chi_i": _TRANSPORT["ion_heat_diffusivity_m2_s"],
        "chi_e": _TRANSPORT["electron_heat_diffusivity_m2_s"],
        "D_e": _TRANSPORT["electron_particle_diffusivity_m2_s"],
        "V_e": _TRANSPORT["electron_particle_convection_m_s"],
    },
    "solver": {"solver_type": "linear"},
    "time_step_calculator": {"calculator_type": "fixed"},
}

__all__ = ["CONFIG", "MODEL_INTERSECTION"]
