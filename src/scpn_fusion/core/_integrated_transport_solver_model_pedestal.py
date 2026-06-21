# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Pedestal correction mixin for integrated transport solver profiles."""

from __future__ import annotations

import numpy as np

from scpn_fusion.core._integrated_transport_solver_base import FloatArray, TransportSolverState
from scpn_fusion.core._integrated_transport_solver_model_common import _solver_module
from scpn_fusion.core.eped_pedestal import EpedPedestalModel
from scpn_fusion.fallback_telemetry import record_fallback_event


class TransportSolverPedestalMixin(TransportSolverState):
    """Apply EPED-derived or fallback pedestal suppression to edge transport."""

    pedestal_model: EpedPedestalModel | None

    def _apply_transport_pedestal_modifier(
        self,
        *,
        P_aux: float,
        chi_turb_e: FloatArray,
        chi_turb_i: FloatArray,
        d_turb: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Modify turbulent diffusivity profiles for H-mode pedestal conditions."""
        solver_mod = _solver_module()
        is_h_mode = P_aux > 30.0
        self._last_pedestal_contract = {
            "used": False,
            "in_domain": True,
            "extrapolation_score": 0.0,
            "extrapolation_penalty": 1.0,
            "domain_violations": [],
            "fallback_used": False,
        }

        if is_h_mode and self.neoclassical_params is not None:
            try:
                p = self.neoclassical_params
                if self.pedestal_model is None or getattr(
                    self.pedestal_model, "_neo_params_hash", None
                ) != id(self.neoclassical_params):
                    eped = EpedPedestalModel(
                        R0=p["R0"],
                        a=p["a"],
                        B0=p["B0"],
                        Ip_MA=p.get("Ip_MA", 15.0),
                        kappa=p.get("kappa", 1.7),
                        A_ion=p.get("A_ion", 2.0),
                        Z_eff=p.get("Z_eff", 1.5),
                    )
                    eped._neo_params_hash = id(self.neoclassical_params)  # type: ignore[attr-defined]
                    self.pedestal_model = eped
                else:
                    eped = self.pedestal_model
                n_ped = max(float(self.ne[-5]), 1.0)
                ped = eped.predict(n_ped)
                self._last_pedestal_contract = {
                    "used": True,
                    "in_domain": bool(ped.in_domain),
                    "extrapolation_score": float(ped.extrapolation_score),
                    "extrapolation_penalty": float(ped.extrapolation_penalty),
                    "domain_violations": list(ped.domain_violations),
                    "fallback_used": False,
                    "n_ped_1e19": float(n_ped),
                }

                ped_start = 1.0 - ped.Delta_ped
                edge_mask = self.rho > ped_start
                chi_turb_e[edge_mask] *= 0.05
                chi_turb_i[edge_mask] *= 0.05
                d_turb[edge_mask] *= 0.05

                ped_idx = np.searchsorted(self.rho, ped_start)
                if ped_idx < len(self.Te):
                    self.Te[ped_idx:] = np.minimum(
                        self.Te[ped_idx:],
                        ped.T_ped_keV * np.linspace(1.0, 0.1, len(self.Te[ped_idx:])),
                    )
                    self.Ti[ped_idx:] = np.minimum(
                        self.Ti[ped_idx:],
                        ped.T_ped_keV * np.linspace(1.0, 0.1, len(self.Ti[ped_idx:])),
                    )
            except solver_mod._EPED_FALLBACK_EXCEPTIONS as exc:
                edge_mask = self.rho > 0.9
                chi_turb_e[edge_mask] *= 0.1
                chi_turb_i[edge_mask] *= 0.1
                d_turb[edge_mask] *= 0.1
                self._last_pedestal_contract = {
                    "used": False,
                    "in_domain": False,
                    "extrapolation_score": 0.0,
                    "extrapolation_penalty": 1.0,
                    "domain_violations": [f"eped_failure:{exc}"],
                    "fallback_used": True,
                }
                record_fallback_event(
                    "integrated_transport_solver",
                    "eped_model_fallback",
                    context={"error": exc.__class__.__name__},
                )
        elif is_h_mode:
            edge_mask = self.rho > 0.9
            chi_turb_e[edge_mask] *= 0.1
            chi_turb_i[edge_mask] *= 0.1
            d_turb[edge_mask] *= 0.1
            self._last_pedestal_contract = {
                "used": False,
                "in_domain": False,
                "extrapolation_score": 0.0,
                "extrapolation_penalty": 1.0,
                "domain_violations": ["neoclassical_params_missing"],
                "fallback_used": True,
            }
            record_fallback_event(
                "integrated_transport_solver",
                "eped_neoclassical_missing_fallback",
                context={"mode": "h_mode"},
            )

        return chi_turb_e, chi_turb_i, d_turb
