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
    nonlinear_invariant_pass_t: NDArray[np.bool_] = field(
        default_factory=lambda: np.empty(0, dtype=np.bool_)
    )
    kx_rhos: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    ky_rhos: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    theta_rad: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    vpar_vth: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    mu_normalized: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    time: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    converged: bool = False
    final_state: NonlinearGKState | None = None

    def to_reference_artifact(self) -> dict[str, object]:
        """Return a JSON-compatible full-dimensional nonlinear GK artifact.

        The spectral distribution is complex-valued in the native solver.  The
        public parity artifact keeps the full species/kx/ky/theta/vpar/mu grid
        and serializes the real and imaginary components separately so external
        GENE/CGYRO/GS2 comparisons cannot silently collapse phase information.
        """
        if self.final_state is None:
            raise ValueError("nonlinear GK reference artifacts require a final 5D state")

        distribution = np.asarray(self.final_state.f, dtype=np.complex128)
        species_index: NDArray[np.float64] = np.arange(
            distribution.shape[0], dtype=np.float64
        )
        zonal_ky_index = int(np.argmin(np.abs(self.ky_rhos)))
        zonal_flow_kx_t = np.asarray(
            self.phi_energy_kxky_t[:, :, zonal_ky_index], dtype=np.float64
        )

        coordinates = {
            "species_index": species_index.tolist(),
            "time_s": np.asarray(self.time, dtype=np.float64).tolist(),
            "kx_rhos": np.asarray(self.kx_rhos, dtype=np.float64).tolist(),
            "ky_rhos": np.asarray(self.ky_rhos, dtype=np.float64).tolist(),
            "theta_rad": np.asarray(self.theta_rad, dtype=np.float64).tolist(),
            "vpar_vth": np.asarray(self.vpar_vth, dtype=np.float64).tolist(),
            "mu_normalized": np.asarray(self.mu_normalized, dtype=np.float64).tolist(),
        }
        coordinate_units = {
            "species_index": "species_index",
            "time_s": "s",
            "kx_rhos": "rho_s^-1",
            "ky_rhos": "rho_s^-1",
            "theta_rad": "rad",
            "vpar_vth": "v_th",
            "mu_normalized": "dimensionless",
        }
        observables = {
            "nonlinear_distribution_function": np.real(distribution).tolist(),
            "nonlinear_distribution_function_imag": np.imag(distribution).tolist(),
            "ion_heat_flux_spectrum": np.asarray(self.Q_i_kxky_t, dtype=np.float64).tolist(),
            "electron_heat_flux_spectrum": np.asarray(self.Q_e_kxky_t, dtype=np.float64).tolist(),
            "particle_free_energy_spectrum": np.asarray(
                self.particle_free_energy_species_kxky_t, dtype=np.float64
            ).tolist(),
            "phi_energy_spectrum": np.asarray(self.phi_energy_kxky_t, dtype=np.float64).tolist(),
            "zonal_flow_energy": zonal_flow_kx_t.tolist(),
            "zonal_flow_energy_total": np.asarray(
                self.zonal_flow_energy_t, dtype=np.float64
            ).tolist(),
            "saturated_phi_rms": np.asarray(self.phi_rms_t, dtype=np.float64).tolist(),
            "saturated_phi_rms_scalar": float(self.saturated_phi_rms),
            "electromagnetic_apar_energy": np.asarray(
                self.A_parallel_energy_kxky_t, dtype=np.float64
            ).tolist(),
            "electromagnetic_bpar_energy": np.asarray(
                self.B_parallel_energy_kxky_t, dtype=np.float64
            ).tolist(),
            "total_energy": np.asarray(self.total_energy_t, dtype=np.float64).tolist(),
            "exb_free_energy_production": np.asarray(
                self.exb_free_energy_production_t, dtype=np.float64
            ).tolist(),
        }
        observable_units = {
            "nonlinear_distribution_function": "delta_f_over_f0",
            "nonlinear_distribution_function_imag": "delta_f_over_f0",
            "ion_heat_flux_spectrum": "gyroBohm_heat_flux",
            "electron_heat_flux_spectrum": "gyroBohm_heat_flux",
            "particle_free_energy_spectrum": "solver_normalized",
            "phi_energy_spectrum": "normalized_field_energy",
            "zonal_flow_energy": "normalized_field_energy",
            "zonal_flow_energy_total": "normalized_field_energy",
            "saturated_phi_rms": "e_phi_over_Te",
            "saturated_phi_rms_scalar": "e_phi_over_Te",
            "electromagnetic_apar_energy": "normalized_apar_energy",
            "electromagnetic_bpar_energy": "normalized_bpar_energy",
            "total_energy": "solver_normalized",
            "exb_free_energy_production": "solver_normalized_per_second",
        }
        observable_axes = {
            "nonlinear_distribution_function": [
                "species_index",
                "kx_rhos",
                "ky_rhos",
                "theta_rad",
                "vpar_vth",
                "mu_normalized",
            ],
            "nonlinear_distribution_function_imag": [
                "species_index",
                "kx_rhos",
                "ky_rhos",
                "theta_rad",
                "vpar_vth",
                "mu_normalized",
            ],
            "ion_heat_flux_spectrum": ["time_s", "kx_rhos", "ky_rhos"],
            "electron_heat_flux_spectrum": ["time_s", "kx_rhos", "ky_rhos"],
            "particle_free_energy_spectrum": [
                "time_s",
                "species_index",
                "kx_rhos",
                "ky_rhos",
            ],
            "phi_energy_spectrum": ["time_s", "kx_rhos", "ky_rhos"],
            "zonal_flow_energy": ["time_s", "kx_rhos"],
            "zonal_flow_energy_total": ["time_s"],
            "saturated_phi_rms": ["time_s"],
            "saturated_phi_rms_scalar": [],
            "electromagnetic_apar_energy": ["time_s", "kx_rhos", "ky_rhos"],
            "electromagnetic_bpar_energy": ["time_s", "kx_rhos", "ky_rhos"],
            "total_energy": ["time_s"],
            "exb_free_energy_production": ["time_s"],
        }
        return {
            "schema": "nonlinear-gk-reference-artifact.v1",
            "status": "diagnostic_native_not_reference_parity",
            "surface": "native_nonlinear_gyrokinetics",
            "reference_family": "GENE/CGYRO/GS2",
            "coordinates": coordinates,
            "coordinate_units": coordinate_units,
            "observables": observables,
            "observable_units": observable_units,
            "observable_axes": observable_axes,
            "notes": [
                "Native nonlinear GK artifact preserves the full 5D distribution grid.",
                "Complex spectral distribution values are split into real and imaginary observables.",
                "This artifact is not external solver parity evidence until compared against same-deck GENE/CGYRO/GS2 outputs.",
            ],
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


@dataclass(frozen=True)
class NonlinearGKMaxwellClosureDiagnostics:
    """Compact electromagnetic closure residuals and full-Maxwell readiness flags."""

    ampere_parallel_linf_residual: float
    pressure_balance_linf_residual: float
    compact_closure_finite: bool
    compact_closure_passes: bool
    full_faraday_displacement_current_supported: bool
    full_vlasov_maxwell_parity_ready: bool


__all__ = [
    "NonlinearGKConfig",
    "NonlinearGKFieldEnergyDiagnostics",
    "NonlinearGKInvariantDiagnostics",
    "NonlinearGKMaxwellClosureDiagnostics",
    "NonlinearGKPhaseSpaceContract",
    "NonlinearGKResult",
    "NonlinearGKState",
    "_E_CHARGE",
    "_M_PROTON",
]
