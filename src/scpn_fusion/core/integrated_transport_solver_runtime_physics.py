# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Integrated Transport Runtime Physics Mixins
"""Physics-side runtime mixins extracted from integrated transport solver runtime."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from scpn_fusion.core.uncertainty import _dt_reactivity

_logger = logging.getLogger(__name__)


def _physics_error_type() -> type[RuntimeError]:
    """Lazily resolve PhysicsError to avoid circular imports at module load."""
    from scpn_fusion.core.integrated_transport_solver import PhysicsError

    return PhysicsError


def _eped_fallback_exceptions() -> tuple[type[BaseException], ...]:
    """Lazily resolve EPED fallback exception tuple from the host module."""
    from scpn_fusion.core.integrated_transport_solver import _EPED_FALLBACK_EXCEPTIONS

    return _EPED_FALLBACK_EXCEPTIONS


class TransportSolverRuntimePhysicsMixin:
    def _rho_volume_element(self) -> np.ndarray:
        """Toroidal volume element per radial cell [m^3]."""
        cached: np.ndarray | None = getattr(self, "_dV_cache", None)
        if cached is not None:
            return cached

        dims = self.cfg["dimensions"]
        r_min = float(dims["R_min"])
        r_max = float(dims["R_max"])
        r0 = 0.5 * (r_min + r_max)
        a_minor = 0.5 * (r_max - r_min)
        if (not np.isfinite(r0)) or r0 <= 0.0:
            raise _physics_error_type()(f"Invalid major radius from config: r0={r0!r}")
        if (not np.isfinite(a_minor)) or a_minor <= 0.0:
            raise _physics_error_type()(f"Invalid minor radius from config: a={a_minor!r}")

        rho = np.clip(
            np.nan_to_num(np.asarray(self.rho, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )
        d_v = 2.0 * np.pi * r0 * 2.0 * np.pi * rho * a_minor**2 * self.drho
        if (not np.all(np.isfinite(d_v))) or np.any(d_v < 0.0):
            raise _physics_error_type()("Invalid toroidal volume element computed from rho grid")
        self._dV_cache = d_v
        return d_v

    def _compute_aux_heating_sources(self, P_aux_MW: float) -> tuple[np.ndarray, np.ndarray]:
        """Return ion/electron auxiliary-heating sources in keV/s."""
        if (not np.isfinite(P_aux_MW)) or P_aux_MW <= 0.0:
            zeros = np.zeros(self.nr, dtype=np.float64)
            self._last_aux_heating_balance = {
                "target_total_MW": float(max(P_aux_MW, 0.0)) if np.isfinite(P_aux_MW) else 0.0,
                "target_ion_MW": 0.0,
                "target_electron_MW": 0.0,
                "reconstructed_ion_MW": 0.0,
                "reconstructed_electron_MW": 0.0,
                "reconstructed_total_MW": 0.0,
            }
            return zeros, zeros

        profile_width = max(self.aux_heating_profile_width, 1e-6)
        rho_raw = np.asarray(self.rho, dtype=np.float64)
        rho_safe = np.nan_to_num(rho_raw, nan=0.0, posinf=1.0, neginf=0.0)
        rho_safe = np.clip(rho_safe, 0.0, 1.0)
        shape = np.exp(-(rho_safe**2) / profile_width)
        shape = np.nan_to_num(shape, nan=1.0, posinf=1.0, neginf=1.0)
        dV = self._rho_volume_element()
        norm = float(np.sum(shape * dV))
        if (not np.isfinite(norm)) or norm <= 0.0:
            shape = np.ones_like(self.rho)
            norm = float(np.sum(shape * dV))
        if (not np.isfinite(norm)) or norm <= 0.0:
            zeros = np.zeros(self.nr, dtype=np.float64)
            self._last_aux_heating_balance = {
                "target_total_MW": float(P_aux_MW),
                "target_ion_MW": 0.0,
                "target_electron_MW": 0.0,
                "reconstructed_ion_MW": 0.0,
                "reconstructed_electron_MW": 0.0,
                "reconstructed_total_MW": 0.0,
            }
            return zeros, zeros

        e_keV_J = 1.602176634e-16
        ne_raw = np.asarray(self.ne, dtype=np.float64)
        ne_safe = np.nan_to_num(ne_raw, nan=0.1, posinf=1e3, neginf=0.1)
        ne_safe = np.clip(ne_safe, 0.1, 1e3) * 1e19

        electron_frac = (
            float(np.clip(self.aux_heating_electron_fraction, 0.0, 1.0)) if self.multi_ion else 0.0
        )
        ion_frac = 1.0 - electron_frac
        p_aux_w = float(P_aux_MW) * 1e6

        p_i_wm3 = ion_frac * p_aux_w * shape / norm
        p_e_wm3 = electron_frac * p_aux_w * shape / norm

        s_heat_i = (2.0 / 3.0) * p_i_wm3 / (ne_safe * e_keV_J)
        s_heat_e = (2.0 / 3.0) * p_e_wm3 / (ne_safe * e_keV_J)

        rec_i_w = 1.5 * np.sum(ne_safe * s_heat_i * e_keV_J * dV)
        rec_e_w = 1.5 * np.sum(ne_safe * s_heat_e * e_keV_J * dV)
        self._last_aux_heating_balance = {
            "target_total_MW": float(P_aux_MW),
            "target_ion_MW": ion_frac * float(P_aux_MW),
            "target_electron_MW": electron_frac * float(P_aux_MW),
            "reconstructed_ion_MW": float(rec_i_w / 1e6),
            "reconstructed_electron_MW": float(rec_e_w / 1e6),
            "reconstructed_total_MW": float((rec_i_w + rec_e_w) / 1e6),
        }

        return s_heat_i, s_heat_e

    @staticmethod
    def _bosch_hale_sigmav(T_keV: np.ndarray) -> np.ndarray:
        """D-T <sigma v> [m^3/s]. Bosch & Hale, NF 32 (1992) 611."""
        T_raw = np.asarray(T_keV, dtype=np.float64)
        T = np.nan_to_num(T_raw, nan=0.2, posinf=200.0, neginf=0.2)
        T = np.clip(T, 0.2, 200.0)
        return _dt_reactivity(T)

    @staticmethod
    def _tungsten_radiation_rate(Te_keV: np.ndarray) -> np.ndarray:
        """Coronal-equilibrium tungsten radiation rate coefficient [W*m^3]."""
        te_raw = np.asarray(Te_keV, dtype=np.float64)
        Te = np.nan_to_num(te_raw, nan=0.01, posinf=1e3, neginf=0.01)
        Te = np.clip(Te, 0.01, 1e3)
        Lz = np.where(
            Te < 1.0,
            5.0e-31 * np.sqrt(Te),
            np.where(
                Te < 5.0,
                5.0e-31 * np.ones_like(Te),
                np.where(
                    Te < 20.0,
                    2.0e-31 * Te**0.3,
                    8.0e-31 * np.ones_like(Te),
                ),
            ),
        )
        return np.nan_to_num(Lz, nan=5.0e-31, posinf=8.0e-31, neginf=5.0e-31)

    @staticmethod
    def _bremsstrahlung_power_density(
        ne_1e19: np.ndarray, Te_keV: np.ndarray, Z_eff: float
    ) -> np.ndarray:
        """Bremsstrahlung power density [W/m^3]."""
        ne_raw = np.asarray(ne_1e19, dtype=np.float64)
        te_raw = np.asarray(Te_keV, dtype=np.float64)
        ne_safe = np.nan_to_num(ne_raw, nan=0.0, posinf=1e6, neginf=0.0)
        ne_safe = np.clip(ne_safe, 0.0, 1e6)
        te_safe = np.nan_to_num(te_raw, nan=0.01, posinf=1e3, neginf=0.01)
        te_safe = np.clip(te_safe, 0.01, 1e3)
        z_eff = float(np.nan_to_num(Z_eff, nan=1.0, posinf=100.0, neginf=1.0))
        z_eff = float(np.clip(z_eff, 1.0e-6, 100.0))
        ne_m3 = ne_safe * 1e19
        p_brem = 5.35e-37 * z_eff * ne_m3**2 * np.sqrt(te_safe)
        return np.nan_to_num(p_brem, nan=0.0, posinf=np.finfo(np.float64).max, neginf=0.0)

    def _evolve_species(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Evolve D/T/He-ash species densities by one time-step."""
        if not self.multi_ion or self.n_D is None:
            return np.zeros(self.nr), np.zeros(self.nr)

        sigmav = self._bosch_hale_sigmav(self.Ti)
        S_fus = (self.n_D * 1e19) * (self.n_T * 1e19) * sigmav
        S_fuel = S_fus / 1e19
        S_He = S_fus / 1e19

        tau_E = float(self.compute_confinement_time(1.0))
        if (not np.isfinite(tau_E)) or tau_E <= 0.0:
            tau_E = 0.5 / max(float(self.tau_He_factor), 1e-6)
        tau_He = max(self.tau_He_factor * tau_E, 0.5)
        if (not np.isfinite(tau_He)) or tau_He <= 0.0:
            tau_He = 0.5
        S_He_pump_rate = 1.0 / tau_He

        L_D_exp = self._explicit_diffusion_rhs(self.n_D, self.D_species * np.ones(self.nr))
        rhs_D = self.n_D + 0.5 * dt * L_D_exp - dt * S_fuel
        a, b, c = self._build_cn_tridiag(self.D_species * np.ones(self.nr), dt)
        self.n_D = self._thomas_solve(a, b, c, rhs_D)
        self.n_D[0] = self.n_D[1]
        self.n_D[-1] = 0.01
        self.n_D = np.maximum(0.001, self.n_D)

        L_T_exp = self._explicit_diffusion_rhs(self.n_T, self.D_species * np.ones(self.nr))
        rhs_T = self.n_T + 0.5 * dt * L_T_exp - dt * S_fuel
        self.n_T = self._thomas_solve(a, b, c, rhs_T)
        self.n_T[0] = self.n_T[1]
        self.n_T[-1] = 0.01
        self.n_T = np.maximum(0.001, self.n_T)

        L_He_exp = self._explicit_diffusion_rhs(self.n_He, self.D_species * np.ones(self.nr))
        rhs_He = self.n_He + 0.5 * dt * L_He_exp + dt * (S_He - S_He_pump_rate * self.n_He)
        self.n_He = self._thomas_solve(a, b, c, rhs_He)
        self.n_He[0] = self.n_He[1]
        self.n_He[-1] = 0.0
        self.n_He = np.maximum(0.0, self.n_He)

        te_safe = np.nan_to_num(
            np.asarray(self.Te, dtype=np.float64), nan=0.1, posinf=1e3, neginf=0.1
        )
        te_safe = np.clip(te_safe, 0.1, 1e3)
        log_te = np.log10(te_safe)
        Z_W = np.clip(15.0 + 12.0 * log_te, 10.0, 50.0)

        self.ne = self.n_D + self.n_T + 2.0 * self.n_He + Z_W * np.maximum(self.n_impurity, 0.0)
        self.ne = np.maximum(self.ne, 0.1)

        ne_m3 = self.ne * 1e19
        ne_safe = np.maximum(ne_m3, 1e10)
        sum_nZ2 = (
            self.n_D * 1e19 * 1.0
            + self.n_T * 1e19 * 1.0
            + self.n_He * 1e19 * 4.0
            + np.maximum(self.n_impurity, 0.0) * 1e19 * Z_W**2
        )
        self._Z_eff = float(np.clip(np.mean(sum_nZ2 / ne_safe), 1.0, 10.0))

        Lz = self._tungsten_radiation_rate(self.Te)
        n_W_m3 = np.maximum(self.n_impurity, 0.0) * 1e19
        P_rad_line = ne_m3 * n_W_m3 * Lz

        return S_He, P_rad_line

    def update_pedestal_bc(self) -> None:
        """Update T_edge_keV using EPED model if active."""
        contract: dict[str, Any] = {
            "used": self.pedestal_model is not None,
            "updated": False,
            "fallback_used": False,
            "in_domain": None,
            "extrapolation_penalty": None,
            "n_ped_1e19": None,
            "t_edge_keV_before": float(self.T_edge_keV),
            "t_edge_keV_after": float(self.T_edge_keV),
            "error": None,
        }
        if self.pedestal_model is None:
            self._last_pedestal_bc_contract = contract
            return

        n_ped_idx = int(0.95 * self.nr)
        n_ped = float(self.ne[n_ped_idx])
        contract["n_ped_1e19"] = n_ped
        if (not np.isfinite(n_ped)) or n_ped <= 0.0:
            contract["fallback_used"] = True
            contract["error"] = f"invalid_n_ped:{n_ped!r}"
            contract["t_edge_keV_after"] = float(self.T_edge_keV)
            self._last_pedestal_bc_contract = contract
            return

        try:
            result = self.pedestal_model.predict(n_ped, T_ped_guess_keV=self.T_edge_keV)
            in_domain = bool(getattr(result, "in_domain", False))
            penalty = float(getattr(result, "extrapolation_penalty", 0.0))
            contract["in_domain"] = in_domain
            contract["extrapolation_penalty"] = penalty
            if in_domain or penalty > 0.5:
                candidate = 0.8 * self.T_edge_keV + 0.2 * float(result.T_ped_keV)
                if (not np.isfinite(candidate)) or candidate <= 0.0:
                    raise ValueError(f"Invalid pedestal boundary candidate {candidate!r}")
                self.T_edge_keV = float(candidate)
                contract["updated"] = True
        except _eped_fallback_exceptions() as exc:
            contract["fallback_used"] = True
            contract["error"] = f"{exc.__class__.__name__}:{exc}"
            _logger.debug(
                "update_pedestal_bc: retaining previous T_edge_keV=%.6f due to %s",
                float(self.T_edge_keV),
                exc,
            )
        contract["t_edge_keV_after"] = float(self.T_edge_keV)
        self._last_pedestal_bc_contract = contract


__all__ = ["TransportSolverRuntimePhysicsMixin"]
