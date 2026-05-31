# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Contracts and constants for the nonlinear gyrokinetic solver."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


_E_CHARGE = 1.602176634e-19
_M_PROTON = 1.67262192369e-27


@dataclass
class NonlinearGKConfig:
    """Grid and physics parameters for the nonlinear solver."""

    n_kx: int = 16
    n_ky: int = 16
    n_theta: int = 64
    n_vpar: int = 16
    n_mu: int = 8
    n_species: int = 2

    dt: float = 0.05
    n_steps: int = 5000
    save_interval: int = 100

    Lx: float = 80.0
    Ly: float = 62.83

    vpar_max: float = 3.0
    mu_max: float = 9.0

    dealiasing: str = "2/3"
    hyper_order: int = 4
    hyper_coeff: float = 0.1
    cfl_factor: float = 0.5
    cfl_adapt: bool = True

    collisions: bool = True
    nu_collision: float = 0.01
    collision_model: str = "krook"
    nonlinear: bool = True
    kinetic_electrons: bool = False
    mass_ratio_me_mi: float = 1.0 / 400.0
    implicit_electrons: bool = False
    electromagnetic: bool = False
    beta_e: float = 0.01

    R0: float = 2.78
    a: float = 1.0
    B0: float = 2.0
    q: float = 1.4
    s_hat: float = 0.78

    R_L_Ti: float = 6.9
    R_L_Te: float = 6.9
    R_L_ne: float = 2.2


@dataclass
class NonlinearGKState:
    """Full 5D+1 state of the nonlinear solver."""

    f: NDArray[np.complex128]
    phi: NDArray[np.complex128]
    time: float
    A_par: NDArray[np.complex128] | None = None
    B_par: NDArray[np.complex128] | None = None


@dataclass(frozen=True)
class NonlinearGKPhaseSpaceContract:
    """Explicit five-dimensional nonlinear GK state/grid contract."""

    distribution_shape: tuple[int, int, int, int, int, int]
    field_shape: tuple[int, int, int]
    distribution_axes: tuple[str, str, str, str, str, str]
    field_axes: tuple[str, str, str]
    field_components: tuple[str, str, str]
    axis_units: dict[str, str]
    boundary_semantics: dict[str, str]
    dealiasing: str


@dataclass
class NonlinearGKResult:
    """Time-averaged transport and diagnostics."""

    chi_i: float
    chi_e: float
    chi_i_gB: float = 0.0
    Q_i_t: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    Q_e_t: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    Q_i_kxky_t: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    Q_e_kxky_t: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    saturated_Q_i_kxky: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    saturated_Q_e_kxky: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    saturated_phi_rms: float = 0.0
    saturated_zonal_flow_energy: float = 0.0
    saturated_phi_energy: float = 0.0
    saturated_A_parallel_energy: float = 0.0
    saturated_B_parallel_energy: float = 0.0
    saturated_total_energy: float = 0.0
    saturated_particle_free_energy_species_kxky: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0)
    )
    saturated_phi_energy_kxky: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    saturated_A_parallel_energy_kxky: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0)
    )
    saturated_B_parallel_energy_kxky: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0)
    )
    phi_rms_t: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    zonal_rms_t: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    zonal_flow_energy_t: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    particle_free_energy_t: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    particle_free_energy_species_kxky_t: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0)
    )
    phi_energy_t: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    A_parallel_energy_t: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    B_parallel_energy_t: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    total_energy_t: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    phi_energy_kxky_t: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    A_parallel_energy_kxky_t: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    B_parallel_energy_kxky_t: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    exb_free_energy_production_t: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    exb_relative_free_energy_production_t: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0)
    )
    dealiased_high_k_max_abs_t: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    nonlinear_invariant_pass_t: NDArray[np.bool_] = field(default_factory=lambda: np.empty(0))
    kx_rhos: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    ky_rhos: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    theta_rad: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    vpar_vth: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    mu_normalized: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    time: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    converged: bool = False
    final_state: NonlinearGKState | None = None

    def to_reference_artifact(self) -> dict[str, object]:
        """Return a JSON-compatible nonlinear GK parity artifact."""
        coordinates = {
            "time_s": self.time.tolist(),
            "kx_rhos": self.kx_rhos.tolist(),
            "ky_rhos": self.ky_rhos.tolist(),
            "theta_rad": self.theta_rad.tolist(),
            "vpar_vth": self.vpar_vth.tolist(),
            "mu_normalized": self.mu_normalized.tolist(),
        }
        coordinate_units = {
            "time_s": "s",
            "kx_rhos": "rho_s^-1",
            "ky_rhos": "rho_s^-1",
            "theta_rad": "rad",
            "vpar_vth": "v_th",
            "mu_normalized": "T_ref/B_ref",
        }
        observables = {
            "ion_heat_flux_spectrum": self.Q_i_kxky_t.tolist(),
            "electron_heat_flux_spectrum": self.Q_e_kxky_t.tolist(),
            "particle_free_energy_spectrum": self.particle_free_energy_species_kxky_t.tolist(),
            "phi_energy_spectrum": self.phi_energy_kxky_t.tolist(),
            "electromagnetic_apar_energy_spectrum": self.A_parallel_energy_kxky_t.tolist(),
            "electromagnetic_bpar_energy_spectrum": self.B_parallel_energy_kxky_t.tolist(),
            "zonal_flow_energy": self.zonal_flow_energy_t.tolist(),
            "saturated_phi_rms": float(self.saturated_phi_rms),
            "saturated_zonal_flow_energy": float(self.saturated_zonal_flow_energy),
            "electromagnetic_apar_energy": self.A_parallel_energy_t.tolist(),
            "electromagnetic_bpar_energy": self.B_parallel_energy_t.tolist(),
            "particle_free_energy": self.particle_free_energy_t.tolist(),
            "total_energy": self.total_energy_t.tolist(),
        }
        observable_units = {
            "ion_heat_flux_spectrum": "gyroBohm",
            "electron_heat_flux_spectrum": "gyroBohm",
            "particle_free_energy_spectrum": "solver_normalized",
            "phi_energy_spectrum": "solver_normalized",
            "electromagnetic_apar_energy_spectrum": "solver_normalized",
            "electromagnetic_bpar_energy_spectrum": "solver_normalized",
            "zonal_flow_energy": "solver_normalized",
            "saturated_phi_rms": "solver_normalized",
            "saturated_zonal_flow_energy": "solver_normalized",
            "electromagnetic_apar_energy": "solver_normalized",
            "electromagnetic_bpar_energy": "solver_normalized",
            "particle_free_energy": "solver_normalized",
            "total_energy": "solver_normalized",
        }
        observable_axes = {
            "ion_heat_flux_spectrum": ["time_s", "kx_rhos", "ky_rhos"],
            "electron_heat_flux_spectrum": ["time_s", "kx_rhos", "ky_rhos"],
            "particle_free_energy_spectrum": ["time_s", "species", "kx_rhos", "ky_rhos"],
            "phi_energy_spectrum": ["time_s", "kx_rhos", "ky_rhos"],
            "electromagnetic_apar_energy_spectrum": ["time_s", "kx_rhos", "ky_rhos"],
            "electromagnetic_bpar_energy_spectrum": ["time_s", "kx_rhos", "ky_rhos"],
            "zonal_flow_energy": ["time_s"],
            "saturated_phi_rms": [],
            "saturated_zonal_flow_energy": [],
            "electromagnetic_apar_energy": ["time_s"],
            "electromagnetic_bpar_energy": ["time_s"],
            "particle_free_energy": ["time_s"],
            "total_energy": ["time_s"],
        }
        return {
            "schema": "nonlinear-gk-reference-artifact.v1",
            "surface": "native_nonlinear_gyrokinetics",
            "reference_family": "GENE/CGYRO/GS2",
            "status": "diagnostic_native_not_reference_parity",
            "coordinates": coordinates,
            "coordinate_units": coordinate_units,
            "observables": observables,
            "observable_units": observable_units,
            "observable_axes": observable_axes,
        }


@dataclass(frozen=True)
class NonlinearGKInvariantDiagnostics:
    """Discrete nonlinear-operator invariant diagnostics."""

    exb_free_energy_production: float
    exb_relative_free_energy_production: float
    dealiased_high_k_max_abs: float
    finite: bool
    passes: bool


@dataclass(frozen=True)
class NonlinearGKFieldEnergyDiagnostics:
    """Electromagnetic field-energy components in solver-normalised units."""

    phi: float
    A_parallel: float
    B_parallel: float
    total: float
    finite: bool


__all__ = [
    "NonlinearGKConfig",
    "NonlinearGKFieldEnergyDiagnostics",
    "NonlinearGKInvariantDiagnostics",
    "NonlinearGKPhaseSpaceContract",
    "NonlinearGKResult",
    "NonlinearGKState",
    "_E_CHARGE",
    "_M_PROTON",
]
