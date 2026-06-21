# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Initialization helper for the integrated transport solver."""

from __future__ import annotations

from typing import Any

import numpy as np

from scpn_fusion.core._integrated_transport_solver_base import TransportSolverState
from scpn_fusion.core.eped_pedestal import EpedPedestalModel


class TransportSolverInitializationMixin(TransportSolverState):
    """Initialize mutable profiles, contracts, and backend configuration."""

    def _initialize_transport_solver_state(self, *, multi_ion: bool) -> None:
        import scpn_fusion.core.integrated_transport_solver as solver_mod

        self.external_profile_mode = True
        self.nr = 50
        self.rho = np.linspace(0, 1, self.nr, dtype=np.float64)
        self.drho = 1.0 / (self.nr - 1)

        self.multi_ion: bool = multi_ion

        self.Te = 1.0 * (1 - self.rho**2)
        self.Ti = 1.0 * (1 - self.rho**2)
        self.ne = 5.0 * (1 - self.rho**2) ** 0.5

        self.chi_e = np.ones(self.nr)
        self.chi_i = np.ones(self.nr)
        self.D_n = np.ones(self.nr)

        self.n_impurity = np.zeros(self.nr)

        self.T_edge_keV = 0.08
        self.pedestal_model: EpedPedestalModel | None = None
        if self.cfg.get("physics", {}).get("pedestal_mode") == "eped":
            R0 = (self.cfg["dimensions"]["R_min"] + self.cfg["dimensions"]["R_max"]) / 2.0
            a = (self.cfg["dimensions"]["R_max"] - self.cfg["dimensions"]["R_min"]) / 2.0
            B0 = self.cfg.get("physics", {}).get("B0", self.cfg.get("B0", 5.3))
            self.pedestal_model = EpedPedestalModel(
                R0=R0,
                a=a,
                B0=B0,
                Ip_MA=self.cfg.get("physics", {}).get("plasma_current_target", 5.0),
            )

        self.neoclassical_params: dict[str, Any] | None = None
        self._last_conservation_error: float = 0.0

        if self.multi_ion:
            self.n_D = 0.5 * self.ne.copy()
            self.n_T = 0.5 * self.ne.copy()
            self.n_He = np.zeros(self.nr)
        else:
            self.n_D = None  # type: ignore[assignment]
            self.n_T = None  # type: ignore[assignment]
            self.n_He = None  # type: ignore[assignment]

        self.tau_He_factor: float = 5.0
        self.D_species: float = 0.3
        self._Z_eff: float = 1.5

        self.aux_heating_profile_width: float = 0.1
        self.aux_heating_electron_fraction: float = 0.5
        self.transport_backend: str = str(
            self.cfg.get("physics", {}).get("transport_backend", "reduced_multichannel")
        )
        self.neural_transport_weights_path: str | None = self.cfg.get("physics", {}).get(
            "neural_transport_weights_path"
        )
        self.neural_transport_tglf_ood_sigma: float = float(
            self.cfg.get("physics", {}).get("neural_transport_tglf_ood_sigma", 5.0)
        )
        self.neural_transport_tglf_max_points: int = int(
            self.cfg.get("physics", {}).get("neural_transport_tglf_max_points", 7)
        )
        self._neural_transport_model = None
        self._neural_transport_model_weights_path: str | None = None
        self.tglf_binary_path: str | None = self.cfg.get("physics", {}).get("tglf_binary_path")
        self.tglf_timeout_s: float = float(self.cfg.get("physics", {}).get("tglf_timeout_s", 120.0))
        self.tglf_max_retries: int = int(self.cfg.get("physics", {}).get("tglf_max_retries", 2))

        self._last_aux_heating_balance: dict[str, float] = {
            "target_total_MW": 0.0,
            "target_ion_MW": 0.0,
            "target_electron_MW": 0.0,
            "reconstructed_ion_MW": 0.0,
            "reconstructed_electron_MW": 0.0,
            "reconstructed_total_MW": 0.0,
        }
        self._last_gyro_bohm_contract: dict[str, Any] = {
            "used": False,
            "source": "uninitialized",
            "path": str(solver_mod._GYRO_BOHM_COEFF_PATH),
            "c_gB": float(solver_mod._GYRO_BOHM_DEFAULT),
            "fallback_used": False,
            "error": None,
        }
        self._last_transport_closure_contract: dict[str, Any] = {
            "used": False,
            "model": "uninitialized",
            "fallback_used": False,
            "base_source": "uninitialized",
            "q_profile_source": "uninitialized",
            "dominant_channel": "unknown",
            "channel_counts": {"ITG": 0, "TEM": 0, "ETG": 0, "stable": 0},
            "channel_energy": {"ITG": 0.0, "TEM": 0.0, "ETG": 0.0},
            "gradient_clip_counts": {"grad_te": 0, "grad_ti": 0, "grad_ne": 0},
            "profile_contract": {"n_points": 0},
            "chi_base_mean": 0.0,
            "chi_e_turb_mean": 0.0,
            "chi_i_turb_mean": 0.0,
            "d_turb_mean": 0.0,
            "chi_gb_reference_mean": None,
            "weights_loaded": False,
            "weights_path": None,
            "weights_checksum": None,
            "classification_mode": "unknown",
            "ood_fraction_3sigma": 0.0,
            "ood_fraction_5sigma": 0.0,
            "ood_point_count": 0,
            "max_abs_z": 0.0,
            "tglf_sample_count": 0,
            "tglf_sample_indices": [],
            "error": None,
        }
        self._last_pedestal_contract: dict[str, Any] = {
            "used": False,
            "in_domain": True,
            "extrapolation_score": 0.0,
            "extrapolation_penalty": 1.0,
            "domain_violations": [],
            "fallback_used": False,
        }
        self._last_pedestal_bc_contract: dict[str, Any] = {
            "used": False,
            "updated": False,
            "fallback_used": False,
            "in_domain": None,
            "extrapolation_penalty": None,
            "n_ped_1e19": None,
            "t_edge_keV_before": float(self.T_edge_keV),
            "t_edge_keV_after": float(self.T_edge_keV),
            "error": None,
        }

        self._last_numerical_recovery_count: int = 0
        self._last_numerical_recovery_breakdown: dict[str, int] = {}
        self._last_numerical_recovery_limit: int | None = None

        raw_recovery_cap = self.cfg.get("solver", {}).get("max_numerical_recoveries_per_step")
        if raw_recovery_cap is None:
            self.max_numerical_recoveries_per_step: int | None = None
        else:
            if isinstance(raw_recovery_cap, bool) or int(raw_recovery_cap) < 0:
                raise ValueError(
                    "solver.max_numerical_recoveries_per_step must be a non-negative integer."
                )
            self.max_numerical_recoveries_per_step = int(raw_recovery_cap)


__all__ = ["TransportSolverInitializationMixin"]
